# Units System Usability Gaps

**Status**: Active tracking document
**Created**: 2025-12-03
**Last Updated**: 2025-12-03

## Overview

This document tracks usability gaps in the units system where users encounter unexpected failures or need workarounds. These represent opportunities for API improvements.

---

## Gap 1: Mesh Creation with UWexpression/UWQuantity (FIXED)

**Date Identified**: 2025-12-03
**Status**: FIXED

### The Problem

Users naturally expect to pass unit-aware quantities directly to mesh creation functions:

```python
outer_radius = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
inner_radius = uw.expression(r"r_i", uw.quantity(3000, "km"), "inner radius")
mantle_thickness = outer_radius.sym - inner_radius.sym

# BEFORE FIX - This failed:
meshball = uw.meshing.Annulus(
    radiusOuter=outer_radius,
    radiusInner=inner_radius,
    cellSize=mantle_thickness / res,
)
# TypeError: Cannot convert expression to float
```

### Root Cause

1. gmsh (underlying meshing library) calls `float()` on the parameters
2. `UWexpression` inherits from SymPy Symbol, which raises `TypeError` on `float()` conversion
3. Even if it could convert, it would give the raw numeric value (6370) without unit conversion to model units

### Fix Applied

All mesh creation functions now use `model.to_model_magnitude()` at the start to convert parameters:

```python
model = uw.get_default_model()
radiusOuter = model.to_model_magnitude(radiusOuter)
radiusInner = model.to_model_magnitude(radiusInner)
cellSize = model.to_model_magnitude(cellSize)
```

This approach:
1. Detects UWexpression, UWQuantity, or Pint Quantity objects
2. Converts them to model units via `model.to_model_units()`
3. Extracts the numerical magnitude

**Note**: Initially we added a `_to_model_float()` helper, but discovered that `model.to_model_magnitude()` already handles all these cases correctly. The code was refactored to use the existing method for consistency with `cartesian.py`.

### Verification

```python
# AFTER FIX - This now works!
meshball = uw.meshing.Annulus(
    radiusOuter=outer_radius,
    radiusInner=inner_radius,
    cellSize=mantle_thickness / res,
)
# Creates correctly scaled mesh with r_prime ranging 0 to 1 ✓
```

### Still a Danger: Using `.value` Directly

Using `.value` directly **appears to work** but gives wrong results:

```python
# WRONG - This runs but creates incorrectly scaled mesh!
meshball = uw.meshing.Annulus(
    radiusOuter=outer_radius.value,  # 6370 interpreted as model units!
    radiusInner=inner_radius.value,  # Should be 6.37 model units
    cellSize=mantle_thickness.value / res,
)
```

With `characteristic_length = 1000 km`:
- User intends: 6370 km = 6.37 model units
- Actual: 6370 model units = 6,370,000 km

**Solution**: Pass the UWexpression directly (now works!), or use `model.to_model_units()` explicitly.

### Gatekeeper Warning System

To help catch the `.value` anti-pattern, mesh functions now use the `expected_dimension` parameter:

```python
# In mesh function:
model.to_model_magnitude(radiusOuter, expected_dimension='[length]')
```

When units are active and a plain number is passed, this issues a warning:

```
UserWarning: Plain number 6370 passed for [length] parameter when units
are active. If you intended physical units, use uw.quantity(6370, 'unit').
Value is being interpreted as 6370 model units.
```

**Behavior**:
- ✅ UWQuantity/UWexpression → No warning, converted correctly
- ✅ Plain number with units NOT active → No warning (dimensionless is fine)
- ⚠️ Plain number with units active → Warning issued

### Mesh Functions Updated

The following mesh creation functions now accept UWexpression/UWQuantity directly:

**Cartesian (`cartesian.py`)** - Already had this support:
- [x] `UnstructuredSimplexBox()`
- [x] `StructuredQuadBox()`
- [x] `BoxInternalBoundary()`

**Annulus (`annulus.py`)** - Updated 2025-12-03:
- [x] `Annulus()`
- [x] `QuarterAnnulus()`
- [x] `SegmentofAnnulus()`
- [x] `AnnulusWithSpokes()`
- [x] `AnnulusInternalBoundary()`
- [x] `DiscInternalBoundaries()`

**Spherical (`spherical.py`)** - TODO:
- [ ] `SphericalShell()`
- [ ] Other spherical mesh functions

---

## Gap 2: Coordinate Units Lost in Arithmetic (FIXED)

**Date Identified**: 2025-12-03
**Status**: FIXED

### The Problem

Coordinate symbols `x, y = mesh.X` have units, but those units were lost in arithmetic:

```python
# BEFORE FIX:
uw.get_units(x)      # → kilometer ✓
uw.get_units(2 * x)  # → None ✗ (should be kilometer)
uw.get_units(x**2)   # → None ✗ (should be kilometer²)
```

### Fix Applied

Modified `compute_expression_units()` in `src/underworld3/function/unit_conversion.py` to extract `._units` from coordinate symbols (BaseScalar objects).

### Verification

```python
# AFTER FIX:
uw.get_units(x)      # → kilometer ✓
uw.get_units(2 * x)  # → kilometer ✓
uw.get_units(x**2)   # → kilometer² ✓
uw.get_units(sqrt(x**2 + y**2))  # → kilometer ✓
```

---

## Gap 3: UWexpression Division by UWQuantity (FIXED)

**Date Identified**: 2025-12-03
**Status**: FIXED

### The Problem

Dividing a symbolic expression containing coordinates by a UWQuantity failed:

```python
r = sympy.sqrt(x**2 + y**2)
inner_radius = uw.quantity(3000, "km")
outer_radius = uw.quantity(6370, "km")

# BEFORE FIX:
(r - inner_radius) / (outer_radius - inner_radius)
# DimensionalityError: Cannot convert from 'kilometer' to 'dimensionless'
```

### Root Cause

`UWexpression.__truediv__` tried to do Pint arithmetic on SymPy expressions, which failed when SymPy tried to `sympify()` the Pint Quantity.

### Fix Applied

Modified `__truediv__` in `src/underworld3/function/expressions.py` to detect symbolic expressions and return lazy SymPy quotients instead of attempting Pint arithmetic.

### Verification

```python
# AFTER FIX:
result = (r - inner_radius) / (outer_radius - inner_radius)
uw.get_units(result)  # → dimensionless ✓
```

---

## Gap 4: Division Units Not Canceling (FIXED)

**Date Identified**: 2025-12-04
**Status**: FIXED

### The Problem

When dividing two unit-aware expressions that should cancel to dimensionless, `uw.get_units()` reported incorrect units:

```python
outer_radius = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
inner_radius = uw.expression(r"r_i", uw.quantity(3000, "km"), "inner radius")
mantle_thickness = outer_radius - inner_radius  # km

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR  # r = sqrt(x**2 + y**2), has km units

# Create r_prime: should be dimensionless (0 to 1 range)
r_prime = (r - inner_radius) / mantle_thickness

# BEFORE FIX:
uw.get_units(r_prime)  # → 1/kilometer ✗ (should be dimensionless!)
```

### Root Cause

Two issues in `compute_expression_units()` in `src/underworld3/function/unit_conversion.py`:

1. **Name-based lookup instead of direct isinstance check**: The code tried to find UWexpressions by looking up `expr.name` in dictionaries (`_expr_names`, `_ephemeral_expr_names`). This failed for ephemeral expressions with auto-generated names.

2. **Missing recursion for UWexpressions wrapping SymPy trees**: When a UWexpression has `expr.units = None` but wraps a SymPy expression containing unit-aware symbols, the code didn't recurse into `._sym` to discover units.

### Fix Applied (DRY Approach)

Modified `compute_expression_units()` to use direct isinstance checks and recursion:

```python
# BEFORE: Complex name-based dictionary lookups
if isinstance(expr, sympy.Symbol):
    symbol_name = expr.name
    if symbol_name in UWexpression._expr_names:
        uw_expr = UWexpression._expr_names[symbol_name]
        ...

# AFTER: Simple and DRY
if isinstance(expr, UWexpression):
    # Case 1: Explicit units
    if expr.has_units and expr.units is not None:
        return expr.units

    # Case 2: Recurse into wrapped SymPy expression
    if hasattr(expr, '_sym') and expr._sym is not None:
        inner_sym = expr._sym
        if isinstance(inner_sym, sympy.Basic) and not isinstance(inner_sym, sympy.Number):
            inner_units = compute_expression_units(inner_sym)
            if inner_units is not None:
                return inner_units
```

This follows the **Transparent Container Principle**: UWexpression derives units from its contents rather than storing them separately.

### Verification

```python
# AFTER FIX:
r_prime = (r - inner_radius) / mantle_thickness
uw.get_units(r_prime)  # → dimensionless ✓

# The Mul args now correctly report:
# Arg 0: 1/(r_o-r_i) → 1/kilometer
# Arg 1: sqrt(N.x**2 + N.y**2) - 3000 → kilometer
# Combined: kilometer * (1/kilometer) = dimensionless ✓
```

---

## Gap 5: Symbolic Expressions Mix Coordinate Units with Raw Values (INVESTIGATING)

**Date Identified**: 2025-12-04
**Status**: INVESTIGATING

### The Problem

When evaluating `r_prime = (r - inner_radius) / mantle_thickness`, the result is numerically wrong:

```python
# Expected: r_prime should range from 0 to 1
# Actual: r_prime evaluates to -889 to -888
```

### Root Cause

The symbolic expression `sqrt(N.x**2 + N.y**2) - 3000` mixes:
- **Coordinates in meters**: `mesh.X.coords` returns values in SI base units (e.g., 3,000,000 to 6,370,000 meters)
- **Raw quantity values**: `inner_radius.value` returns 3000 (the raw numeric value in km)

So the evaluation computes: `(3,000,000 - 3000) / 3370 ≈ 889` instead of `(3.0 - 3.0) / 3.37 ≈ 0` (in model units).

### Analysis

This is a **design question** about what should happen during arithmetic:

**Option A**: Keep expressions fully symbolic
- `r - inner_radius` stays as `sqrt(x**2 + y**2) - r_i` (symbols, not values)
- Units are preserved through symbolic operations
- Requires evaluation-time non-dimensionalization

**Option B**: Use .data (ND values) when scaling is active
- `r - inner_radius` uses `inner_radius.data` (3.0 model units) not `.value` (3000 km)
- Matches the coordinate system of `mesh.X.coords`
- But loses symbolic structure

**Option C**: Coordinate normalization
- `mesh.X.coords` could return model units instead of meters
- All expressions would then be in consistent units

### Current Workaround

Use fully symbolic expressions that don't extract numeric values:

```python
# Instead of: r_prime = (r - inner_radius) / mantle_thickness
# Use symbolic form:
r_prime = (r - inner_radius.sym) / mantle_thickness.sym
```

### TODO

- Determine the intended design direction
- Either fix `mesh.X.coords` to return model units, or fix arithmetic to use `.data` when scaling is active

---

## Future Improvements

### Consistent Float Conversion

Consider adding `__float__` to `UWexpression` and `UWQuantity` that:
1. Converts to model units
2. Returns the dimensionless magnitude

This would make many APIs work automatically, but needs careful design to avoid silent unit confusion.

### Mesh Coordinate Return Type

Currently `mesh.X.coords` returns values in meters (SI base units), not model units. Consider:
- Always returning model units for consistency
- Or clearly documenting the current behavior

---

---

## Gap 6: SI Internal Storage for UWQuantity (PROPOSED)

**Date Identified**: 2025-12-05
**Status**: PROPOSED DESIGN

### The Problem

When evaluating expressions like `r_prime = (r - inner_radius) / mantle_thickness`:
- `mesh.X.coords` returns coordinates in SI meters (from PETSc dimensionless × model scale)
- `inner_radius.value` returns 3000 (km, user's original units)
- The expression `sqrt(N.x**2 + N.y**2) - 3000` mixes meters with km!

Result: `(3,000,000 - 3000) / 3370 ≈ 889` instead of `(3,000,000 - 3,000,000) / 3,370,000 ≈ 0`

### Root Cause

`UWQuantity` stores values in the user's original units:
```python
qty = uw.quantity(3000, "km")
qty._value = 3000        # User's input, NOT SI
qty._pint_qty = 3000 km  # Pint stores as-is
```

### Proposed Solution: SI Internal Storage

Store all quantities in SI base units internally, with a "display preference" for user output:

```python
# Proposed behavior
qty = uw.quantity(3000, "km")
qty._si_value = 3000000       # SI base (meters) - internal
qty._si_pint_qty = 3000000 m  # Pint quantity in SI
qty._display_unit = km        # Remember user's preferred display

# Properties
qty.value → 3000              # Converts to display units for user
qty.si_value → 3000000        # Direct SI access
```

### Benefits

1. **Consistent arithmetic**: `r - inner_radius` works because both are in meters
2. **Pint handles conversions**: `.to()` works naturally
3. **User experience preserved**: Display shows original units
4. **Simpler code**: Remove complex unit-matching in arithmetic methods

### Implementation Changes

**UWQuantity (`quantities.py`)**:
1. `__init__`: Convert input to SI immediately
2. `value` property: Convert SI back to display units
3. `si_value` property: Direct SI access (used in expressions)
4. `_display_unit`: Store user's preferred unit
5. `__repr__`: Show in display units

**UWexpression (`expressions.py`)**:
1. When building expressions, use `.si_value` (or internal Pint magnitude)
2. Simplify arithmetic overloads - all values are already in SI

**Evaluation**:
1. Coordinates are in SI (meters from PETSc)
2. Expression values are in SI (inner_radius → 3000000)
3. Everything matches!

### Design Principles

1. Robust unit storage/conversion via Pint + our `.to()` machinery
2. Users specify natural units, easy conversion back and forth
3. Reduce complexity/fragility for maintenance
4. Every object has `.to(units)` and `.to_compact()` built in
5. Preferred units as a display feature (implement after foundation is solid)

### Core Principle: SI as Common Standard (2025-12-05)

**Why SI Internal Storage?**

There are times when we must detach values from their units (e.g., passing to external libraries,
numeric array operations, PETSc integration). When this separation happens, there's a risk of
combining mismatched but dimensionally compatible values (e.g., km with meters).

**The Solution**: By storing all quantities in SI base units internally, we minimize this risk.
When values are extracted, they're guaranteed to be in a consistent unit system.

**The `.value` Red Flag**

Using `.value` directly should be treated as a **red flag** in code review. The first question
should always be: "Why can't we use the Pint quantity instead?"

- ✅ **Preferred**: Use `._pint_qty` or Pint arithmetic throughout
- ⚠️ **Caution**: Using `.value` for external APIs that require floats
- 🚫 **Dangerous**: Using `.value` in arithmetic where units could mismatch

**When `.value` is Acceptable**:
1. Passing to external C/Fortran libraries (PETSc, gmsh) that require raw floats
2. Creating numpy arrays for numerical operations (after ensuring all values are in SI)
3. Final output to files or visualization (after explicit unit conversion)

**When `.value` is NOT Acceptable**:
1. Building symbolic expressions - use `.sym` or the full Pint quantity
2. Arithmetic between quantities - let Pint handle unit conversion
3. Anywhere units could be mismatched - always go through SI first

---

## Gap 7: DRY Unit Conversion via Delegation (PROPOSED)

**Date Identified**: 2025-12-05
**Status**: PROPOSED DESIGN

### The Problem

Unit conversion functionality is scattered across the codebase with inconsistent implementations.

### Complete Audit of Unit Conversion Methods (2025-12-05)

#### Top-Level Functions (`units.py`)

| Function | Location | Status | Notes |
|----------|----------|--------|-------|
| `uw.get_units(expr)` | units.py:515 | ✅ Working | Walks expression trees |
| `uw.convert_units(qty, target)` | units.py:1068 | ⚠️ **Stub** | Warns "not fully implemented" |
| `uw.to_compact()` | - | ❌ Missing | Does not exist |
| `uw.to_base_units()` | - | ❌ Missing | Does not exist |
| `uw.to_reduced_units()` | - | ❌ Missing | Does not exist |

#### Model Methods (`model.py`)

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `model.to_model_units(qty)` | model.py:3323 | ✅ Working | Converts to model's reference units |
| `model.to_model_magnitude(qty)` | model.py:3586 | ✅ Working | Extracts float for gateways (gmsh, etc.) |

#### UWQuantity (`quantities.py`) - **Reference Implementation**

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `.to(target)` | quantities.py:235 | ✅ Complete | Returns new UWQuantity |
| `.to_base_units()` | quantities.py:255 | ✅ Complete | Converts to SI base |
| `.to_reduced_units()` | quantities.py:263 | ✅ Complete | Simplifies compound units |
| `.to_compact()` | quantities.py:271 | ✅ Complete | Human-readable form |

#### UWexpression (`expressions.py`)

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `.to()` | - | ❌ Missing | Docstring explicitly says "no .to() on expressions" |
| `.to_base_units()` | - | ❌ Missing | |
| `.to_reduced_units()` | - | ❌ Missing | |
| `.to_compact()` | - | ❌ Missing | |

#### MathematicalMixin (`mathematical_mixin.py`)

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `.to(target)` | mathematical_mixin.py:562 | ⚠️ Deprecated | Returns scaled SymPy expr, not unit-aware type |
| `.to(target)` (derivatives) | mathematical_mixin.py:795 | ⚠️ Partial | For derivative matrices |

#### UnitAwareArray (`unit_aware_array.py`)

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `.to(target)` | unit_aware_array.py:232 | ✅ Working | Returns new UnitAwareArray |
| `.to_base_units()` | - | ❌ Missing | |
| `.to_compact()` | - | ❌ Missing | |

#### MeshVariable (`discretisation_mesh_variables.py`)

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `.to(target)` | mesh_variables.py:1682 | ⚠️ Nested | Inside array property class |
| `.to(target)` | mesh_variables.py:1991 | ⚠️ Nested | Inside another array class |

#### Backend Mixins (`units_mixin.py`)

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `PintBackend.get_units()` | units_mixin.py:111 | ✅ Working | Low-level Pint wrapper |
| `PintBackend.convert_units()` | units_mixin.py:137 | ✅ Working | Low-level Pint wrapper |

### Analysis: Current Architecture Issues

1. **Fragmentation**: 6+ different implementations of `.to()` with different behaviors
2. **Missing Methods**: `to_base_units()`, `to_compact()` only on UWQuantity
3. **Inconsistent Returns**: Some return unit-aware types, some return raw SymPy
4. **Stub Code**: `uw.convert_units()` exists but warns "not fully implemented"
5. **Deprecated Code**: `MathematicalMixin.to()` marked deprecated but still used

### Proposed Solution: Delegation Pattern

**Core Principle**: Expressions don't "know" their units - they delegate to what they wrap.

**Phase 1: Complete Base Functions**
```python
# In units.py - these become the SINGLE SOURCE OF TRUTH
uw.get_units(expr)         # ✅ Already works
uw.convert_units(expr, target)  # Complete this
uw.to_base_units(expr)     # Add this
uw.to_reduced_units(expr)  # Add this
uw.to_compact(expr)        # Add this
```

**Phase 2: Delegation Methods**
```python
# Every unit-aware type delegates to base functions
class UWexpression:
    def to(self, target_units):
        return uw.convert_units(self, target_units)

    def to_compact(self):
        return uw.to_compact(self)

    def to_base_units(self):
        return uw.to_base_units(self)
```

**Phase 3: Deprecate/Remove Duplicates**
- Remove standalone implementations in MathematicalMixin
- Simplify UnitAwareArray to delegate
- Remove nested implementations in MeshVariable

### Benefits

1. **DRY**: Unit conversion logic in ONE place (`units.py`)
2. **Consistent API**: Every unit-aware type gets all 4 methods
3. **Transparent Containers**: Expressions naturally delegate to contents
4. **Maintainable**: Fix bugs in one place, all types benefit
5. **Testable**: Test base functions once, delegation is trivial

### Migration Path

| Step | Action | Risk | Status |
|------|--------|------|--------|
| 1 | Complete `uw.convert_units()` | Low | ✅ DONE |
| 2 | Add `uw.to_base_units()`, etc. | Low | ✅ DONE |
| 3 | Add delegation to UWexpression | Low | ✅ DONE |
| 4 | Add delegation to UnitAwareArray | Low | 🔄 Pending |
| 5 | Deprecate MathematicalMixin.to() | Medium | 🔄 Pending |
| 6 | Remove deprecated code | High | 🔄 Future |

### Implementation Summary (2025-12-05)

**Base Functions Added to `units.py`:**
- `uw.convert_units(quantity, target_units)` - Full implementation handling all UW3 types
- `uw.to_base_units(quantity)` - Convert to SI base units
- `uw.to_reduced_units(quantity)` - Simplify compound units
- `uw.to_compact(quantity)` - Human-readable unit prefixes

**UWexpression Delegation Methods:**
- `.to(target_units)` → delegates to `uw.convert_units()`
- `.to_base_units()` → delegates to `uw.to_base_units()`
- `.to_reduced_units()` → delegates to `uw.to_reduced_units()`
- `.to_compact()` → delegates to `uw.to_compact()`

---

## Version History

- 2025-12-03: Initial document with 3 gaps identified (2 fixed, 1 pending)
- 2025-12-03: Refactored mesh creation to use `model.to_model_magnitude()` instead of custom helper
- 2025-12-03: Applied fix to all annulus mesh functions
- 2025-12-03: Added gatekeeper warning system via `expected_dimension` parameter
- 2025-12-04: Gap 4 fixed - Division units now cancel correctly (DRY refactor of `compute_expression_units()`)
- 2025-12-04: Gap 5 documented - Coordinate/value mixing during evaluation (needs design decision)
- 2025-12-05: Gap 6 proposed - SI internal storage for UWQuantity (design document added)
- 2025-12-05: Added "Core Principle: SI as Common Standard" - `.value` is a red flag, prefer Pint quantities
- 2025-12-05: Gap 7 - Complete audit of unit conversion methods across codebase, proposed DRY delegation pattern
- 2025-12-05: Gap 7 IMPLEMENTED - Base functions and UWexpression delegation methods complete
