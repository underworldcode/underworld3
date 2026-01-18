# Units Handling for Integrals and Derivatives (2025-12-01)

## Executive Summary

Enhanced the units system with two key improvements:
1. **Derivative units computation** - `get_units(dv/dx)` now correctly returns derivative units (variable units / coordinate units)
2. **Integral unit propagation** - `Integral.evaluate()` now returns `UWQuantity` with proper units when the mesh has coordinate units

**Status**: ✅ **COMPLETE** - All tests passing, backward compatible

**Files Modified**:
- `src/underworld3/units.py` - Derivative units computation in `_extract_units_info()`
- `src/underworld3/cython/petsc_maths.pyx` - Integral unit propagation in `Integral.evaluate()`

**Tests Verified**:
- `tests/test_0501_integrals.py` - 9/9 passing, 1 expected failure

---

## Bug Fix 1: Derivative Units Computation

### Problem

Before this fix, `get_units()` on derivative expressions returned the wrong units:

```python
# Velocity variable with units m/s on mesh with km coordinates
v = uw.discretisation.MeshVariable("v", mesh, 1, units="m/s")

# Derivative dv/dx
dv_dx = v.sym.diff(mesh.X[0])

# BEFORE FIX (WRONG):
uw.get_units(dv_dx)  # → m/s (same as variable units!)

# AFTER FIX (CORRECT):
uw.get_units(dv_dx)  # → m/km/s = 0.001/s (derivative units!)
```

### Root Cause

The `_extract_units_info()` function was trying to get units from `sympy_expr.func`, which is a **class** not an **instance**. It doesn't have the `.units` attribute.

UW3 derivative functions store a reference to their parent MeshVariable via a weakref attribute `func.meshvar`, but this wasn't being accessed correctly.

### Solution

Access the parent MeshVariable via `func.meshvar` weakref and compute derivative units:

```python
# In _extract_units_info() - derivative units computation
if hasattr(sympy_expr, 'diffindex'):
    try:
        var_units = None
        # Get units from the underlying MeshVariable via meshvar weakref
        if hasattr(sympy_expr, 'func') and hasattr(sympy_expr.func, 'meshvar'):
            meshvar_ref = sympy_expr.func.meshvar
            meshvar = meshvar_ref() if callable(meshvar_ref) else meshvar_ref
            if meshvar is not None and hasattr(meshvar, 'units'):
                var_units = meshvar.units

        # Get coordinate units being differentiated with respect to
        coord_units = None
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > sympy_expr.diffindex:
            coord = sympy_expr.args[sympy_expr.diffindex]
            coord_units_info = _extract_units_info(coord)
            coord_units = coord_units_info[1] if coord_units_info[0] else None

        # Compute derivative units: var_units / coord_units
        if var_units and coord_units:
            derivative_units = var_units / coord_units
            return True, derivative_units, backend
```

### Verification

```python
import underworld3 as uw
from underworld3.utilities.unit_aware_coordinates import patch_coordinate_units

mesh = uw.meshing.UnstructuredSimplexBox()
patch_coordinate_units(mesh)  # Adds km units to coordinates

v = uw.discretisation.MeshVariable("v", mesh, 1, units="m/s")
dv_dx = v.sym.diff(mesh.X[0])

print(uw.get_units(dv_dx))  # → meter / kilometer / second = 0.001/s ✅
```

---

## Feature: Integral Unit Propagation

### Overview

`Integral.evaluate()` now returns `UWQuantity` with proper units when:
- The mesh has coordinate units (via `patch_coordinate_units()`), OR
- The integrand has meaningful units (not `None` or `dimensionless`)

Plain meshes without units continue to return `float` for backward compatibility.

### Implementation

**Location**: `src/underworld3/cython/petsc_maths.pyx`, lines 118-207

**Key Logic**:

```python
def evaluate(self, verbose=False):
    # ... PETSc computation gives raw ND value 'vald' ...

    try:
        integrand_units = underworld3.get_units(self.fn)
        coord_units = underworld3.get_units(self.mesh.X[0])

        from underworld3.scaling import units as ureg

        # Helper: check if units are "meaningful" (not None, not dimensionless)
        def has_meaningful_units(u):
            if u is None:
                return False
            try:
                if u == ureg.dimensionless:
                    return False
            except:
                pass
            return True

        integrand_has_units = has_meaningful_units(integrand_units)
        coord_has_units = has_meaningful_units(coord_units)

        if integrand_has_units or coord_has_units:
            # Compute result units
            if coord_has_units:
                volume_units = coord_units ** self.mesh.dim  # km² or km³
            else:
                volume_units = None

            if integrand_has_units and volume_units is not None:
                result_units = integrand_units * volume_units
            elif integrand_has_units:
                result_units = integrand_units
            elif volume_units is not None:
                result_units = volume_units
            else:
                return vald  # No meaningful units - return float

            # Scale by integrand reference if ND scaling active
            physical_value = vald
            if underworld3.is_nondimensional_scaling_active() and integrand_has_units:
                # Re-dimensionalize using Model scales
                ...

            return underworld3.quantity(physical_value, result_units)
    except Exception:
        pass

    return vald  # Fall back to plain float
```

### Unit Computation Rules

| Integrand Units | Coordinate Units | Result Units |
|-----------------|------------------|--------------|
| None/dimensionless | None | `float` (backward compatible) |
| None/dimensionless | km | km² (2D) or km³ (3D) |
| K (kelvin) | None | K |
| K (kelvin) | km | K·km² (2D) or K·km³ (3D) |

### Why `dimensionless` is Treated as "No Units"

When users pass `fn=1.0`, SymPy's `sympify()` converts it to a SymPy Float. The `compute_expression_units()` function returns `dimensionless` for such values (a Pint Unit object, not `None`).

For backward compatibility, we treat `dimensionless` as equivalent to "no meaningful units":

```python
# Plain float integrand
integral = uw.maths.Integral(mesh=plain_mesh, fn=1.0)
result = integral.evaluate()
# type(result) == float  (backward compatible!)

# Same with unit-aware mesh
integral = uw.maths.Integral(mesh=unit_mesh, fn=1.0)
result = integral.evaluate()
# type(result) == UWQuantity, result.units == km² (coordinate units only)
```

### Verification

#### Test 1: 2D Plain Mesh (Backward Compatible)
```python
mesh = uw.meshing.UnstructuredSimplexBox()  # No units
v = uw.discretisation.MeshVariable("v", mesh, 1)

integral = uw.maths.Integral(mesh=mesh, fn=1.0)
result = integral.evaluate()

assert isinstance(result, float)
assert abs(result - 1.0) < 1e-8
print("✅ Plain 2D mesh returns float")
```

#### Test 2: 2D Mesh with Units
```python
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    time=uw.quantity(1, "Myr"),
    mass=uw.quantity(1e20, "kg"),
    temperature=uw.quantity(1000, "K"),
)

mesh = uw.meshing.UnstructuredSimplexBox()
patch_coordinate_units(mesh)
v = uw.discretisation.MeshVariable("v", mesh, 1)

integral = uw.maths.Integral(mesh=mesh, fn=1.0)
result = integral.evaluate()

assert isinstance(result, uw.UWQuantity)
assert result.units.dimensionality == {'[length]': 2}  # km²
print(f"✅ 2D unit mesh: {result}")  # → 1.0 km²
```

#### Test 3: 3D Mesh with Units
```python
mesh3d = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0., 0., 0.),
    maxCoords=(1., 1., 1.),
    cellSize=0.2
)
patch_coordinate_units(mesh3d)
v = uw.discretisation.MeshVariable("v3d", mesh3d, 1)

integral = uw.maths.Integral(mesh=mesh3d, fn=1.0)
result = integral.evaluate()

assert isinstance(result, uw.UWQuantity)
assert result.units.dimensionality == {'[length]': 3}  # km³
print(f"✅ 3D unit mesh: {result}")  # → 1.0 km³
```

---

## Design Rationale

### Gateway Pattern Compliance

This implementation follows the "gateway pattern" documented in the units system design:
- Units are handled at system boundaries (user input/output)
- Internal PETSc computations remain nondimensional
- Only at the final `evaluate()` call do we attach units

### Backward Compatibility Priority

The primary design constraint was maintaining backward compatibility:
- **All existing tests must pass** - achieved (9/9 passing)
- **Plain meshes return float** - maintained
- **`abs(result)` must work** - works on float results

### Meaningful Units Detection

The `has_meaningful_units()` helper ensures we don't accidentally attach units when:
- Both integrand and coordinates are unitless
- The integrand is a bare numeric constant (sympified to Float → dimensionless)

---

## Related Documentation

**Core Units System:**
- `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md` - Current architecture
- `docs/reviews/2025-11/UNITS-EVALUATION-FIXES-2025-11-25.md` - Previous evaluation fixes
- `docs/reviews/2025-11/UNITS-SYSTEM-FIXES-REVIEW.md` - Non-dimensionalization fixes

**Integral System:**
- `src/underworld3/cython/petsc_maths.pyx` - Implementation
- `tests/test_0501_integrals.py` - Test suite

**CHANGELOG Entry:**
- `docs/developer/CHANGELOG.md` - Added entry for December 2025

---

## Verification Checklist

✅ Derivative units: `get_units(dv/dx)` returns `var_units / coord_units`
✅ 2D integral with units: Returns `UWQuantity` with km²
✅ 3D integral with units: Returns `UWQuantity` with km³
✅ Plain mesh integral: Returns `float` (backward compatible)
✅ `abs()` works on plain mesh results
✅ All 9 integral tests pass
✅ Test uses `sympify(1.0)` → dimensionless → treated as "no units"

**Status:** Feature complete, production-ready ✅

---

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude (AI) | 2025-12-01 | Submitted |
| Primary Reviewer | TBD | TBD | Pending |
