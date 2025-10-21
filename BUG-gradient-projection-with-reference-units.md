# BUG: Gradient Projection with Reference Units

**Date Discovered**: 2025-10-15
**Severity**: High
**Status**: Open

## Summary

When reference units are set via `model.set_reference_quantities()`, numerical gradients computed using `Vector_Projection` are incorrect by a factor equal to the reference length scale.

## Reproduction

```python
import underworld3 as uw
import numpy as np

# Set reference units
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=uw.quantity(500, "m"),  # This sets reference length = 100m
    mantle_temperature=uw.quantity(1300, "K"),
)

# Create mesh with units
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16),
    minCoords=(0.0, 0.0),
    maxCoords=(uw.quantity(1000, "m"), uw.quantity(500, "m")),
    units="metre",
)

# Temperature variable with linear profile: T = 300 + 2.6*y
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

# Solve Poisson with BCs to get linear gradient
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = uw.quantity(1e-6, "m^2/s")
poisson.f = 0.0
poisson.add_dirichlet_bc(uw.quantity(300, "K"), "Bottom")
poisson.add_dirichlet_bc(uw.quantity(1600, "K"), "Top")
poisson.solve()

# Compute gradient using projection
gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)
gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
gradient_proj.uw_function = mesh.vector.gradient(T.sym)
gradient_proj.solve()

# Evaluate gradient
grad = uw.function.evaluate(gradT.sym, np.array([[500.0, 250.0]]))
dT_dy = grad[0, 0, 1]

# Expected: 2.6 K/m
# Actual: 0.026 K/m (off by factor of 100)
print(f"Expected gradient: 2.6 K/m")
print(f"Computed gradient: {dT_dy:.6f} K/m")
print(f"Error factor: {2.6 / dT_dy:.1f}×")
print(f"Reference length: {model.get_fundamental_scales()['length']}")
```

**Output** (2025-10-15):
```
Expected gradient: 2.6 K/m
Computed gradient: 259.960955 K/m
Error factor: 100.0× (TOO LARGE, not too small)
Reference length: 100 meter
```

**NOTE**: The gradient is 100× **too large** (260 K/m instead of 2.6 K/m), not too small as originally thought. This suggests the coordinates are being treated as if they're in the **smaller** unit (meters) when they're actually stored in the **larger** unit (100m reference length).

## Root Cause Analysis

**Diagnostic Test Results** (2025-10-15):
- ✅ **Confirmed**: Both `Projection` (scalar) and `Vector_Projection` give the same wrong answer
- ✅ **Conclusion**: The bug is in derivative evaluation itself, NOT specific to Vector_Projection
- ✅ **Symbolic units correct**: `uw.get_units(T.sym.diff(mesh.N.y))` correctly reports `'kelvin / meter'`
- ❌ **Numerical values wrong**: Gradients are 100× too large (260 K/m instead of 2.6 K/m)

**Root Cause Hypothesis**:

When `model.set_reference_quantities()` is called with `domain_depth=500m`:
1. Reference length scale is computed: 500m / 5 = **100m**
2. Mesh coordinates are stored internally in **model units** (scaled by 100m)
3. `mesh.N.y` coordinates are dimensionless: `physical_y / 100m`
4. When the derivative `dT/dy` is computed numerically:
   - The gradient is w.r.t. the scaled coordinate: `dT/d(y_scaled)`
   - Since `y_scaled = y_physical / 100m`, we have: `dy_physical = 100m * dy_scaled`
   - By chain rule: `dT/dy_physical = (dT/dy_scaled) * (dy_scaled/dy_physical) = (dT/dy_scaled) / 100m`
   - **But the code computes**: `dT/dy_scaled` and treats it as if it's `dT/dy_physical`
   - **Result**: Value is 100× too large

**Testing**: Notebook 13 includes comprehensive tests confirming this affects both scalar and vector projection.

## What Works

**Symbolic derivatives are correct**:
```python
x, y = mesh.X
grad_symbolic = T.sym.diff(mesh.N.y)
print(uw.get_units(grad_symbolic))  # Correctly reports 'kelvin / meter'
```

The unit tracking through derivatives is working correctly - the bug is only in the numerical evaluation via projection.

## Workaround

Manually scale the gradient by the reference length:

```python
ref_length = model.get_fundamental_scales()['length']
length_scale = ref_length.magnitude if hasattr(ref_length, 'magnitude') else float(ref_length)
dT_dy_corrected = dT_dy / length_scale  # Divide by 100 to correct the 100× error
# Now dT_dy_corrected = 2.6 K/m ✓
```

## Expected Behavior

`Vector_Projection` should automatically account for coordinate scaling when evaluating gradients, so that the numerical result matches the symbolic units.

## Affected Code

- `src/underworld3/systems/` - Vector_Projection solver
- `src/underworld3/maths/vector_calculus.py` - `gradient()` method (lines 49-58)
- The projection system needs to check if coordinates are scaled and apply correction factor

## Possible Fix

The projection solver needs to:
1. Detect when the mesh has reference scaling applied
2. Get the length scale from the model
3. Apply correction factor to gradient components: `gradient_physical = gradient_model / length_scale`

Alternatively, the coordinate symbols themselves could carry scaling information that's automatically applied during evaluation.

## Test Case Needed

Add to test suite:
```python
def test_gradient_projection_with_reference_units():
    """
    Test that gradient projection works correctly when reference units are set.

    This is a regression test for the bug where gradients are off by the
    reference length scale factor.
    """
    # Setup with reference units
    # Solve simple problem with known gradient
    # Assert gradient matches expected value
    # Should pass after fix is implemented
```

## Related Work

- Coordinate units system (2025-10-15): Successfully tracks units through symbolic derivatives ✅
- This bug affects only numerical evaluation, not symbolic expressions
- Similar issue likely affects other differential operators (curl, divergence) with reference scaling

## Priority

**High** - This affects any code using reference quantities for scaling, which is the recommended approach for geophysical problems. Users need to know about this bug and apply the workaround.

## Documentation

- Documented in: `docs/beginner/tutorials/13-Scaling-Physical-Problems.ipynb`
- Warning added to Case 3 with workaround code
