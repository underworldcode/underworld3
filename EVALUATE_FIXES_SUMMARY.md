# uw.function.evaluate() Fixes - 2025-10-13

## Final Solution Summary

**Problem**: `uw.function.evaluate()` returned plain numpy arrays with no way to query what units the values represented, breaking the principle that "everything has units (or nothing has units)".

**Solution**: Return `UnitAwareArray` (numpy ndarray subclass) with unit metadata that:
- ✅ Works with all numpy operations (`np.linalg.norm`, `np.max`, etc.)
- ✅ Carries unit information via `uw.get_units(result)`
- ✅ Allows direct assignment to MeshVariables
- ✅ Maintains full backward compatibility

## Problem Statement (Original Issues)

`uw.function.evaluate()` had three issues that broke natural usage patterns:

### Issue 1: Couldn't Accept MeshVariable Directly
```python
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")

# This FAILED with AttributeError: 'EnhancedMeshVariable' object has no attribute 'args'
V = uw.function.evaluate(velocity, coords)

# Had to manually extract .sym
V = uw.function.evaluate(velocity.sym, coords)  # Verbose and annoying
```

### Issue 2: Returned UWQuantity Wrapper Instead of Plain Array
```python
vel = uw.function.evaluate(velocity.sym, coords)

# Result was UWQuantity wrapper, not numpy array
# This broke numpy operations:
np.linalg.norm(vel, axis=1)  # AttributeError: 'UWQuantity' object has no attribute 'conjugate'

# Also broke direct assignment:
temperature.array[...] = vel  # Failed because vel is wrapped
```

### Issue 3: No Way to Query Result Units (NEW - 2025-10-13)
```python
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")
vel = uw.function.evaluate(velocity, coords)

# Result is plain numpy array with no unit information
# No way to verify: "What units does this array represent?"
uw.get_units(vel)  # None - unit information lost!

# This breaks the scaling principle:
# evaluate() applies dimensional scaling but result has no way to prove/track units
```

## Root Cause Analysis

### Issue 1: No Auto-Extraction of .sym
The wrapper function `make_evaluate_unit_aware()` in `unit_conversion.py` didn't check if the input was a MeshVariable and extract `.sym` automatically.

**Location**: `/src/underworld3/function/unit_conversion.py:715`

### Issue 2: Result Wrapping in UWQuantity
The function `add_expression_units_to_result()` was wrapping numpy arrays in `UWQuantity` objects when it detected units on the expression.

**Location**: `/src/underworld3/function/unit_conversion.py:779`

**Problem**: While unit tracking is valuable, wrapping the result breaks:
- Numpy operations (`np.linalg.norm()`, `np.max()`, etc.)
- Direct array assignment (`meshVar.array[...] = result`)
- Any code expecting plain numpy arrays

## Solutions Applied (Chronological)

### Fix 1: Auto-Extract .sym from MeshVariable (2025-10-13 morning)

**File**: `src/underworld3/function/unit_conversion.py` lines 742-746

```python
def unit_aware_evaluate(expr, coords=None, coord_sys=None, coord_units=None, **kwargs):
    # Auto-extract .sym from MeshVariable for user convenience
    import underworld3 as uw
    if hasattr(expr, 'sym') and hasattr(expr, 'mesh'):
        # This is likely a MeshVariable - extract the symbolic representation
        expr = expr.sym

    # ... rest of function
```

**Result**: Both patterns now work:
```python
V = uw.function.evaluate(velocity, coords)       # ✓ Works!
V = uw.function.evaluate(velocity.sym, coords)   # ✓ Still works!
```

### Fix 2: Return Plain Numpy Arrays (Don't Wrap) (2025-10-13 morning)

**File**: `src/underworld3/function/unit_conversion.py` lines 784-789 (SUPERSEDED)

~~This fix returned plain numpy arrays with NO unit information.~~

**Problem discovered**: While this fixed numpy operations, it lost unit information entirely.

### Fix 3: Return UnitAwareArray with Metadata (2025-10-13 afternoon - FINAL)

**Files**:
- `src/underworld3/function/unit_conversion.py` lines 12-70: `UnitAwareArray` class
- `src/underworld3/function/unit_conversion.py` lines 845-858: Integration into `evaluate()`
- `src/underworld3/__init__.py` line 145: Export `uw.get_units()`

**Implementation**:

```python
class UnitAwareArray(np.ndarray):
    """
    Numpy array subclass that carries unit metadata without breaking numpy operations.

    - Behaves exactly like numpy arrays for all operations
    - Works with all numpy functions (linalg.norm, max, min, etc.)
    - Carries unit information accessible via .units or ._units attribute
    - Loses unit metadata through numpy operations (correct behavior)
    """

    def __new__(cls, input_array, units=None):
        obj = np.asarray(input_array).view(cls)
        obj._units = units
        return obj

    @property
    def units(self):
        return self._units
```

**Integration into evaluate()**:

```python
# In make_evaluate_unit_aware():
result = original_evaluate_func(expr, internal_coords, coord_sys, **filtered_kwargs)

# Determine what units the result should have
try:
    expr_units = determine_expression_units(expr, mesh_info)
    if expr_units is not None:
        # Return UnitAwareArray with unit metadata attached
        return UnitAwareArray(result, units=expr_units)
except Exception:
    pass

# Return plain array if no units detected
return result
```

**Result**: Best of both worlds - unit tracking AND numpy compatibility:

```python
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")
vel = uw.function.evaluate(velocity, coords)

# ✓ Unit information preserved
print(uw.get_units(vel))  # "m/s"

# ✓ Works with numpy operations
magnitudes = np.linalg.norm(vel, axis=1)  # Returns plain array
max_vel = np.max(vel)

# ✓ Works with direct assignment
temperature.array[:, 0, 0] = magnitudes

# ✓ No need for .magnitude extraction
# ✓ Can verify units when needed
```

## Design Philosophy

### Unit Tracking Strategy (UPDATED)

**Key Principle**: Units are tracked at **multiple levels** depending on context:

1. **MeshVariables/SwarmVariables have units**: Variables track units as metadata
2. **Arrays from evaluate() carry unit metadata**: Lightweight numpy subclass preserves unit information
3. **Arrays from numpy operations are plain**: Operations may change dimensions, so return plain arrays
4. **Unit queries are always possible**: `uw.get_units(obj)` works on variables, arrays, and quantities

**Why this is correct**:
1. **MeshVariables have units**: `velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")`
2. **Arrays are just data**: `velocity.array` is a numpy array of numbers
3. **Evaluate returns data**: `uw.function.evaluate()` returns the numerical result
4. **Assignment preserves units**: When assigning to a MeshVariable, units are validated at that level

**What we DON'T do**:
- ❌ Wrap arrays in heavy UWQuantity objects (breaks numpy operations)
- ❌ Force unit conversions at the array operation level
- ❌ Propagate units through numpy operations (dimensions change)

**What we DO**:
- ✓ Track units on MeshVariables and SwarmVariables
- ✓ Attach lightweight unit metadata to `evaluate()` results
- ✓ Make unit queries always possible via `uw.get_units()`
- ✓ Maintain full compatibility with numpy operations
- ✓ Return plain arrays from numpy operations (correct behavior)

### Natural Usage Pattern (UPDATED)

The final implementation enables this natural, intuitive pattern with unit transparency:

```python
# Create variables with units
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")
temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="K")

# Evaluate (returns UnitAwareArray)
vel = uw.function.evaluate(velocity, coords)  # No .sym needed!

# Query units when needed
print(f"Velocity units: {uw.get_units(vel)}")  # "m/s"

# Use numpy operations naturally (returns plain arrays)
magnitudes = np.linalg.norm(vel, axis=1)  # Plain array, no units
max_vel = np.max(vel)

# Direct assignment works
temperature.array[:] = magnitudes

# Solvers work naturally
dt_adv = min_dx / np.linalg.norm(vel, axis=1).max()  # No .magnitude needed!

# Unit transparency throughout
assert uw.get_units(velocity) == "m/s"  # MeshVariable
assert uw.get_units(vel) == "m/s"  # evaluate() result
assert uw.get_units(magnitudes) is None  # numpy operation result (dimensions changed)
```

## Testing

**Test files**:
- `test_evaluate_fixes.py` - Original tests for Issue 1 and 2
- `test_unit_tracking_final.py` - Comprehensive tests for Issue 3 (unit tracking)

All key tests passing:
- ✓ `evaluate()` accepts MeshVariable directly (no .sym needed)
- ✓ `evaluate()` returns UnitAwareArray (numpy-compatible)
- ✓ `uw.get_units()` can query units from results
- ✓ Results work with `np.linalg.norm()` and all numpy operations
- ✓ `global_evaluate()` also returns UnitAwareArray with units
- ✓ Direct assignment to MeshVariable works
- ✓ Backward compatible with `.sym` pattern
- ✓ Expression evaluation preserves unit information

## Impact on Existing Code

### Code That Now Works

```python
# BEFORE: Required .sym extraction
V = uw.function.evaluate(velocity.sym, temperature.coords)
if hasattr(V, 'magnitude'):
    V = V.magnitude  # Extract from wrapper

# AFTER: Natural, direct usage
V = uw.function.evaluate(velocity, temperature.coords)
```

### Code That Still Works (Backward Compatible)

```python
# Old pattern with .sym still works
V = uw.function.evaluate(velocity.sym, coords)  # ✓ Still fine

# Old pattern with .magnitude extraction (no longer needed but harmless)
if hasattr(V, 'magnitude'):
    V = V.magnitude  # No-op now, but doesn't break
```

### Solvers Fixed

The fix in `src/underworld3/systems/solvers.py:1453-1455` is now **unnecessary** but harmless:

```python
# Extract magnitude if vel is a UWQuantity (unit-aware)
if hasattr(vel, 'magnitude'):
    vel = vel.magnitude
```

This check will now always fail (vel is a plain array), but the code still works. This defensive check can be removed in future cleanup.

## Files Modified (Final State)

1. **`src/underworld3/function/unit_conversion.py`**:
   - Lines 12-70: NEW `UnitAwareArray` class (numpy ndarray subclass)
   - Lines 742-746: Auto-extract `.sym` from MeshVariable
   - Lines 845-858: Return UnitAwareArray with detected units (replaces plain array return)

2. **`src/underworld3/__init__.py`**:
   - Line 145: Export `uw.get_units()` for user convenience

2. **`src/underworld3/systems/solvers.py`** (previous session):
   - Lines 1453-1455: Added defensive check for `.magnitude` (now unnecessary but harmless)

3. **`src/underworld3/coordinates.py`** (previous session):
   - Line 239: Fixed deprecation warning (use `CoordinateSystem.coords` instead of `mesh.data`)

## Rebuild Required

After modifying source files, rebuild is required:

```bash
pixi run underworld-build
```

## Summary

**Original Problem** (morning): `uw.function.evaluate()` returned UWQuantity wrappers that broke numpy operations.

**Initial Solution** (morning): Return plain numpy arrays - fixed numpy operations but lost unit information.

**Real Problem Discovered** (afternoon): No way to query what units the result represents - `uw.get_units(result)` returned `None`.

**Final Solution** (afternoon): Return `UnitAwareArray` - numpy ndarray subclass with unit metadata.

**Benefits**:
1. **Unit Transparency**: Can always query `uw.get_units(result)` to see what units values represent
2. **Numpy Compatibility**: Works perfectly with all numpy operations (linalg, max, min, etc.)
3. **Natural Workflow**: No `.magnitude` extraction, no `.sym` requirement, no special handling
4. **Backward Compatible**: Old patterns with `.sym` still work
5. **Lightweight**: Metadata-only wrapper, no behavioral changes to numpy operations

**Philosophy**: Unit information should be **queryable but not operational** - arrays carry unit metadata for inspection, but numpy operations return plain arrays (since dimensions may change).

## Relationship to Existing Unit System

**Important**: `UnitAwareArray` is **complementary** to the existing `UnitAwareMixin` system, not duplicative.

- **`UnitAwareMixin`** (existing): Full-featured units for **variables** (MeshVariable, SwarmVariable)
  - Provides: Unit conversion, scaling, dimensional analysis, compatibility checking
  - Used with: Variable objects via multiple inheritance

- **`UnitAwareArray`** (new): Lightweight metadata for **evaluate() results**
  - Provides: Simple unit string metadata, numpy compatibility, query support
  - Used with: Arrays returned from `uw.function.evaluate()`

**See**: `UNIT_TRACKING_COMPARISON.md` for detailed comparison and design rationale.

**Key distinction**: Variables need operational unit support (conversion, checking), while arrays need informational unit support (query what units they represent). These are complementary systems serving different needs.
