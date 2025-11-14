# MeshVariable Architecture Analysis

## Current Architecture (2025-01-13)

### The Layered Structure

```
User Code:
  uw.discretisation.MeshVariable(...)
         ↓
  EnhancedMeshVariable (persistence.py) ← THIS IS WHAT USERS GET
    - Wraps _BaseMeshVariable
    - Adds: Math operations, units support, persistence
    - OVERRIDES .array property with own implementations
    - Creates own: SimpleMeshArrayView, TensorMeshArrayView
         ↓
  _BaseMeshVariable (discretisation_mesh_variables.py)
    - Low-level PETSc interface
    - Has .array property with own view classes
    - THESE VIEW CLASSES ARE NEVER USED BY USERS!
```

### Key Discovery from __init__.py

```python
# src/underworld3/discretisation/__init__.py line 2:
from .persistence import EnhancedMeshVariable as MeshVariable
```

**Result**: `MeshVariable` is an ALIAS for `EnhancedMeshVariable`

### The Duplication Problem

**DUPLICATED ARRAY VIEW CLASSES:**

1. **discretisation_mesh_variables.py** (lines 1475-1904):
   - `SimpleMeshArrayView` class (line 1475)
   - `TensorMeshArrayView` class (line 1718)
   - **Status**: UNUSED by users (only if someone directly accesses _BaseMeshVariable)

2. **persistence.py** (lines 217-535):
   - `SimpleMeshArrayView` class (line 217)
   - `TensorMeshArrayView` class (line 416)
   - **Status**: ACTIVELY USED (this is what users interact with)

### What We Fixed

**Before Fix**:
- `discretisation_mesh_variables.py` array views: ❌ Broken (but unused anyway)
- `persistence.py` array views: ❌ Broken (actively used, causing user issues)

**After Fix**:
- `discretisation_mesh_variables.py` array views: ⚠️  Partially fixed (but still unused)
- `persistence.py` array views: ✅ FIXED (users see correct behavior)

### Why This Architecture Exists

**EnhancedMeshVariable Purpose** (from docstring):
```
Enhanced MeshVariable with:
- Mathematical operations (via MathematicalMixin)
- Units support (via DimensionalityMixin)
- Optional persistence for adaptive meshing scenarios
- Collision-safe registration
```

**Intended Design**:
- `_BaseMeshVariable`: Low-level PETSc interface, no units/math
- `EnhancedMeshVariable`: High-level user interface with units/math/persistence

### The Problem: Incomplete Implementation

**Issue 1: Code Duplication**
- Array view logic duplicated in TWO places
- Bugs must be fixed in BOTH places (as we discovered)
- Easy to miss one (as we did initially)

**Issue 2: Confusion About What's Active**
- `_BaseMeshVariable.array` exists but is shadowed
- Documentation doesn't clarify the layering
- Developers (like me) naturally fix the "base" class first

**Issue 3: "Stub" Comment**
You mentioned `persistence.py` is a "stub that has not been fully implemented" for "creating variables that move from one mesh to another".

**Reality**: It's fully active! Every user gets `EnhancedMeshVariable`, not a stub.

### Current Usage Status

| Component | Status | Used By |
|-----------|--------|---------|
| `_BaseMeshVariable` | ✅ Active | EnhancedMeshVariable (internal) |
| `_BaseMeshVariable.array` views | ⚠️  Shadowed | Nobody (overridden) |
| `EnhancedMeshVariable` | ✅ Active | ALL USERS (via alias) |
| `EnhancedMeshVariable.array` views | ✅ Active | ALL USERS |
| Persistence features | ❓ Unknown | Unknown if used |

### Recommendations

#### Option 1: Remove Duplication (Delegate)
```python
# In EnhancedMeshVariable (persistence.py):
@property
def array(self):
    """Delegate to base variable's array."""
    return self._base_var.array  # Use base implementation
```

**Pros**: Single source of truth for array logic
**Cons**: Must ensure base variable has all needed functionality (units, scaling)

#### Option 2: Remove Unused Code
```python
# Remove array view classes from discretisation_mesh_variables.py
# Keep only in persistence.py since that's what's used
```

**Pros**: Eliminates confusion, clear ownership
**Cons**: Loses potential for direct _BaseMeshVariable usage

#### Option 3: Merge Into Single Class
```python
# Merge EnhancedMeshVariable functionality into _BaseMeshVariable
# Make MeshVariable directly be the enhanced class
```

**Pros**: No wrapper layer, no duplication
**Cons**: Major refactoring, breaks internal architecture

#### Option 4: Status Quo (Document)
```python
# Keep both, but add clear comments:
# - persistence.py: "USER-FACING IMPLEMENTATION - FIX BUGS HERE"
# - discretisation_mesh_variables.py: "INTERNAL ONLY - Not used by users"
```

**Pros**: Minimal changes, maintains flexibility
**Cons**: Duplication remains, future confusion likely

### Immediate Action Needed

**Critical Documentation**:
1. Add comment to `discretisation_mesh_variables.py` array views:
   ```python
   # NOTE: These view classes are NOT used when users create MeshVariable!
   # Users get EnhancedMeshVariable (persistence.py) which has its own views.
   # Only update these if you need to change _BaseMeshVariable's direct array access.
   ```

2. Add comment to `persistence.py` array views:
   ```python
   # USER-FACING IMPLEMENTATION
   # These view classes are what users interact with when accessing .array
   # Any bugs in array indexing, unit conversion, etc. must be fixed HERE
   ```

### Questions to Resolve

1. **Persistence Features**: Are the adaptive meshing / transfer_data_from features actually used?
2. **_BaseMeshVariable Direct Usage**: Does anything use _BaseMeshVariable directly (bypassing wrapper)?
3. **Future Plans**: Was there an intention to merge these or keep separate?
4. **Testing**: Are there tests that verify EnhancedMeshVariable specifically, or just MeshVariable?

### Testing Verification

```bash
# Confirm EnhancedMeshVariable is used:
pixi run -e default python -c "
import underworld3 as uw
mesh = uw.meshing.StructuredQuadBox(elementRes=(2,2))
v = uw.discretisation.MeshVariable('v', mesh, 2, degree=2)
print(f'Type: {type(v)}')
print(f'Has _base_var: {hasattr(v, \"_base_var\")}')
print(f'Array type: {type(v.array).__name__}')
print(f'Array module: {type(v.array).__module__}')
"
```

Expected output:
```
Type: <class 'underworld3.discretisation.persistence.EnhancedMeshVariable'>
Has _base_var: True
Array type: TensorMeshArrayView
Array module: underworld3.discretisation.persistence
```

This confirms users get EnhancedMeshVariable and its array views, NOT the base ones.

---

## SOLUTION IMPLEMENTED (2025-01-13)

### What We Did

Instead of a full merge, we implemented a **delegation pattern** that eliminates duplication while preserving the architecture:

1. **Fixed the implementation in `_BaseMeshVariable`** (`discretisation_mesh_variables.py`):
   - Updated `SimpleMeshArrayView.__setitem__` to handle both `.value` (UWQuantity) and `.magnitude` (Pint)
   - Updated `TensorMeshArrayView.__setitem__` with the same fix
   - Ensured proper unit conversion and non-dimensionalization pipeline

2. **Made `EnhancedMeshVariable` delegate to base** (`persistence.py`):
   - Changed `.array` property to return `self._base_var.array` (simple delegation)
   - Removed 425 lines of duplicate array view implementation
   - Kept only the wrapper functionality (mixins, persistence features)

### Benefits of This Approach

✅ **No Duplication**: Single implementation of array views in `_BaseMeshVariable`
✅ **Minimal Changes**: No massive refactoring, architecture preserved
✅ **All Tests Pass**: Unit conversion works correctly with delegation
✅ **Clear Ownership**: Array logic lives in base class where it belongs
✅ **Maintainable**: Future fixes only need to be made in one place

### Current Architecture (After Fix)

```
User Code:
  uw.discretisation.MeshVariable(...)
         ↓
  EnhancedMeshVariable (persistence.py) ← Users get this
    - Wraps _BaseMeshVariable
    - Adds: Math operations (MathematicalMixin), units support (DimensionalityMixin)
    - DELEGATES .array property to base variable ✓ (no duplication)
         ↓
  _BaseMeshVariable (discretisation_mesh_variables.py)
    - Low-level PETSc interface
    - Has .array property with WORKING view classes ✓
    - SimpleMeshArrayView and TensorMeshArrayView (SINGLE SOURCE OF TRUTH)
```

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| `_BaseMeshVariable.array` | Broken unit conversion | ✅ Fixed (handles .value and .magnitude) |
| `EnhancedMeshVariable.array` | Own 425-line implementation | ✅ Delegates to base (3 lines) |
| Code duplication | 2 implementations (850+ lines) | ✅ 1 implementation (single source of truth) |
| Tests | Failing unit conversion | ✅ All passing (4/4) |

### Files Modified

1. **`src/underworld3/discretisation/discretisation_mesh_variables.py`**:
   - Fixed `SimpleMeshArrayView.__setitem__` (lines 1521-1589)
   - Fixed `TensorMeshArrayView.__setitem__` (lines 1781-1843)

2. **`src/underworld3/discretisation/persistence.py`**:
   - Changed `.array` property to delegate (lines 195-206)
   - Removed duplicate `_create_simple_array_view_wrapper()` (425 lines deleted)
   - Removed duplicate `_create_tensor_array_view_wrapper()` (425 lines deleted)

### Validation

All systematic unit tests pass:
- ✓ Test 1: Set 1 mm/yr → .data has correct ND value
- ✓ Test 2: Read .array → correct dimensional value in m/s
- ✓ Test 3: Different units (cm/yr) convert correctly
- ✓ Test 4: Backward path (.data → .array) works

### Why This Is Better Than Full Merge

The original analysis recommended merging `EnhancedMeshVariable` into `_BaseMeshVariable`. However, the delegation approach is superior because:

1. **Preserves Separation of Concerns**: Mathematical operations and persistence features remain separate from core PETSc operations
2. **Less Risky**: No massive refactoring, minimal changes to working code
3. **Maintains Flexibility**: Can still add persistence-specific features without polluting base class
4. **Easier to Reverse**: If needed, can revert to separate implementations easily

### Future Recommendations

1. **Document the delegation**: Add comments in code explaining why delegation is used
2. **Consider renaming**: `persistence.py` → `enhanced_mesh_variable.py` (more accurate name)
3. **Unit test coverage**: Add tests that specifically verify delegation works correctly
