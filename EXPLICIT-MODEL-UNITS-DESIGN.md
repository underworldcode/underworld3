# Explicit Model Units Design

**Date**: 2025-10-16
**Status**: Design Document
**Supersedes**: Hidden coordinate scaling approach

## Problem Statement

When reference units are set, the current system uses "hidden" coordinate scaling:
- Internally: coordinates stored as `x_model = x_physical / L_ref` (scaled)
- Externally: `mesh.X.coords` shows physical coordinates
- Derivatives: Computed w.r.t. scaled coordinates but reported as physical units
- **Result**: Numerical derivative values wrong by factor of `L_ref`

### Root Cause

The issue affects ALL differential operators:
- `mesh.vector.gradient(T.sym)`
- `mesh.vector.divergence(v.sym)`
- `mesh.vector.strain_tensor(v.sym)`
- `mesh.vector.curl(v.sym)`

All derivatives are w.r.t. `mesh.N` (scaled coordinates), but users expect physical units.

### Why Hidden Scaling is Fragile

To maintain the illusion of physical units, we'd need conversion at every entry/exit:
1. ✅ Boundary conditions - converted
2. ✅ Variable values - converted
3. ❌ Derivatives - NOT converted (requires chain rule correction)
4. ❌ All differential operators - NOT converted
5. ❌ Constitutive models - NOT converted

**Fragility**: Every new differential operator needs patching. Easy to miss, hard to test.

## Solution: Explicit Model Units

### Core Principle

**The model owns the unit system. Everything uses model units explicitly.**

No hidden conversions. Users work in model units throughout, with explicit conversions at boundaries.

## Architecture

### 1. Model as Unit Authority

```python
class Model:
    def get_model_units(self) -> dict:
        """
        Return the model's unit system derived from reference quantities.

        If reference quantities not set, returns SI base units.
        If set, returns engineering units rounded from fundamental scales.

        Returns:
            dict: {'length': 'Mm', 'time': 'Gs', 'temperature': 'kK', ...}
        """
```

**Engineering Unit Rounding**:
- `domain_depth = 500 km` → `L_ref = 100 km` → rounds to `'Mm'`
- `mantle_temperature = 1300 K` → `T_ref = 1000 K` → rounds to `'kK'`
- `reference_viscosity = 1e21 Pa.s` → rounds to `'ZPa.s'`

**Rounding modes**:
- `powers_of_10`: Round to nearest power of 10 (10, 100, 1000, ...)
- `engineering`: Round to SI prefixes (k, M, G, T, ...)

### 2. Mesh Uses Model Units

```python
mesh = uw.meshing.StructuredQuadBox(
    minCoords=(0.0, 0.0),
    maxCoords=(uw.quantity(1000, "km"), uw.quantity(500, "km"))
)

# Internally:
# 1. mesh.units = model.get_model_units()['length']  # 'Mm' (from model)
# 2. User coords converted: 1000 km → 1.0 Mm, 500 km → 0.5 Mm
# 3. mesh stores: maxCoords=(1.0, 0.5) in Mm
# 4. mesh.N coordinates in Mm
```

**Key**: User does NOT specify mesh units. They come from model.

### 3. Variables Use Model Units

```python
T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")

# Internally:
# - Variable asks model for temperature units → 'kK'
# - User specified units="kelvin" (optional convenience)
# - Scale factor stored: 1000 (K → kK conversion)
# - Data stored in kK internally
```

**Setting values**:
```python
T.array[...] = uw.quantity(300, "K")
# → Converts 300 K → 0.3 kK, stores 0.3
```

**Getting values**:
```python
value = T.array[0]  # Returns 0.3 (in kK, model units)

# User converts to desired units
T_physical = uw.quantity(value, model.get_model_units()['temperature']).to("K")
# → 300 K
```

### 4. Derivatives in Model Units (Automatically!)

```python
# Gradient
grad = mesh.vector.gradient(T.sym)
# Returns: kK/Mm (model units)

grad_value = uw.function.evaluate(grad, coords)
# Returns: numerical values in kK/Mm

# User converts to desired output
grad_physical = uw.quantity(grad_value, "kK/Mm").to("K/km")
# → 2.6 K/km ✓
```

**No conversion needed** - derivatives just work because:
- `T.sym` is in kK
- `mesh.N.y` is in Mm
- `dT/dy` is naturally in kK/Mm
- No hidden scaling, no chain rule issues

### 5. Constitutive Models in Model Units

```python
# Strain rate
strain_rate = mesh.vector.strain_tensor(velocity.sym)
# velocity in Mm/Gs → strain_rate in 1/Gs (model units)

# Stress
stress = 2 * viscosity * strain_rate
# viscosity in ZPa.s, strain_rate in 1/Gs → stress in Pa (consistent)
```

**All internally consistent** - no conversion layers needed.

## Implementation Plan

### Phase 1: Model Unit Registry

**File**: `src/underworld3/model.py`

Add methods:
```python
def get_model_units(self) -> dict:
    """Get engineering units derived from reference quantities"""

def _round_to_engineering_unit(self, scale: UWQuantity) -> str:
    """Round a scale to nearest engineering unit (Mm, Gs, etc.)"""

def get_coordinate_unit(self) -> str:
    """Convenience: get model length unit"""
    return self.get_model_units()['length']
```

### Phase 2: Mesh Coordinate Conversion

**File**: `src/underworld3/discretisation/discretisation_mesh.py`

Changes:
```python
class Mesh:
    def __init__(self, minCoords=None, maxCoords=None, ...):
        # Get units from model (not user)
        model = uw.get_default_model()
        self.units = model.get_coordinate_unit()  # e.g., 'Mm'

        # Convert user coordinates to model units
        if maxCoords:
            self._maxCoords = self._convert_coords_to_model(maxCoords)

    def _convert_coords_to_model(self, coords):
        """Convert user coordinates (with units) to model units"""
        converted = []
        for coord in coords:
            if isinstance(coord, uw.UWQuantity):
                converted.append(coord.to(self.units).magnitude)
            else:
                # Assume already in model units
                converted.append(coord)
        return tuple(converted)
```

**Remove**: The coordinate unit patching system in `src/underworld3/coordinates.py` (no longer needed)

### Phase 3: Variable Unit Handling

**File**: `src/underworld3/discretisation/discretisation_mesh_variables.py`

Add to `MeshVariable.__init__`:
```python
# Get model units
model = uw.get_default_model()
model_units = model.get_model_units()

# Determine variable's model units based on dimensionality
# (temperature, velocity, etc.)
self._model_units = self._infer_model_units(units, model_units)

# If user specified units, compute scale factor
if units and units != self._model_units:
    self._scale_factor = uw.quantity(1, units).to(self._model_units).magnitude
else:
    self._scale_factor = 1.0
```

### Phase 4: Boundary Condition Conversion

**File**: `src/underworld3/systems/` (various solver files)

Add gatekeeper conversion:
```python
def add_dirichlet_bc(self, value, boundary):
    # Convert to model units at boundary
    if isinstance(value, uw.UWQuantity):
        model_units = self._get_variable_model_units()
        value_model = value.to(model_units).magnitude
    else:
        value_model = value

    # Apply internally in model units
    self._apply_bc(value_model, boundary)
```

### Phase 5: Evaluation Conversion

**File**: `src/underworld3/function/function.py` (or evaluation module)

Provide helpers:
```python
def evaluate_in_physical_units(fn, coords, target_units=None):
    """
    Evaluate function and convert to physical units.

    Args:
        fn: Function to evaluate
        coords: Coordinates (in any units)
        target_units: Desired output units (optional)

    Returns:
        Result in target_units or model units
    """
    # Evaluate in model units
    result = uw.function.evaluate(fn, coords)

    # Convert if requested
    if target_units:
        model_units = uw.get_units(fn)
        result_physical = uw.quantity(result, model_units).to(target_units)
        return result_physical.magnitude

    return result
```

## Migration Path

### What Changes for Users

**Before (hidden model units)**:
```python
mesh = uw.meshing.Box(maxCoords=(1000, 500), units="metre")
T.array[...] = 300 + 2.6 * mesh.X.coords[:, 1]  # In meters
grad = mesh.vector.gradient(T.sym)
# Expected K/m, got wrong values (bug!)
```

**After (explicit model units)**:
```python
mesh = uw.meshing.Box(maxCoords=(1000*uw.units.km, 500*uw.units.km))
# mesh.units = 'Mm' (from model)

T.array[...] = 300 + 2600 * mesh.X.coords[:, 1]  # In Mm, grad is 2600 K/Mm
grad = mesh.vector.gradient(T.sym)  # Returns kK/Mm

# Convert to physical units
grad_value = uw.function.evaluate(grad, coords)
grad_physical = uw.quantity(grad_value, "kK/Mm").to("K/km")  # 2.6 K/km ✓
```

### What Stays the Same

**Solvers**: Completely unchanged
- Still work in model units internally
- BCs already converted at boundary
- No changes to solver internals needed

**Constitutive models**: Unchanged
- Already work symbolically
- Units flow through naturally

## Testing Strategy

### 1. Solver Consistency Tests

**Critical**: Ensure solvers produce identical results

```python
def test_stokes_with_explicit_units():
    """Verify Stokes solver results unchanged with explicit units"""
    # Set up problem in model units
    # Solve
    # Compare to known benchmark
    # Assert identical to reference solution
```

### 2. Unit Conversion Tests

```python
def test_coordinate_conversion():
    """Verify coordinate conversion to model units"""
    mesh = uw.meshing.Box(maxCoords=(1000*uw.units.km, 500*uw.units.km))
    assert mesh.units == 'Mm'
    assert np.allclose(mesh.maxCoords, (1.0, 0.5))

def test_gradient_units():
    """Verify gradient has correct model units"""
    grad = mesh.vector.gradient(T.sym)
    grad_units = uw.get_units(grad)
    assert grad_units == 'kK/Mm'
```

### 3. Backward Compatibility (Temporary)

```python
def test_legacy_mesh_with_units_warns():
    """Legacy units parameter should warn"""
    with pytest.warns(DeprecationWarning):
        mesh = uw.meshing.Box(maxCoords=(1000, 500), units="metre")
```

## Benefits

### 1. Robustness
- ✅ No fragile injection points
- ✅ Derivatives automatically correct
- ✅ All differential operators work
- ✅ Easy to audit and test

### 2. Transparency
- ✅ Users see actual units used
- ✅ No hidden conversions
- ✅ Explicit conversions at boundaries
- ✅ No surprises

### 3. Simplicity
- ✅ Model owns unit system
- ✅ Everything uses model units
- ✅ Conversions only at input/output
- ✅ Less code, fewer bugs

### 4. Performance
- ✅ O(1) numbers by design
- ✅ No runtime conversion overhead
- ✅ Optimal numerics

## Documentation Requirements

### User Guide Section: "Working with Model Units"

```markdown
# Working with Model Units

Underworld3 uses a model-wide unit system for optimal numerics and clarity.

## Setting Up Model Units

When you set reference quantities, the model automatically selects
engineering units:

```python
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "km"),
    mantle_temperature=uw.quantity(1300, "K"),
)

# Check model units
print(model.get_model_units())
# {'length': 'Mm', 'time': 'Gs', 'temperature': 'kK', ...}
```

## The Three-Step Workflow

### 1. Input: Convert to Model Units

```python
# Mesh coordinates
mesh = uw.meshing.Box(maxCoords=(1000*uw.units.km, 500*uw.units.km))
# Converted to: (1.0 Mm, 0.5 Mm)

# Boundary conditions
stokes.add_dirichlet_bc(5*uw.units.cm/uw.units.yr, "Top")
# Converted to: ~1.58e-3 Mm/Gs
```

### 2. Compute: Work in Model Units

```python
# Variables store values in model units
T.array[...] = 0.3  # 0.3 kK (300 K)

# Derivatives in model units
grad = mesh.vector.gradient(T.sym)  # Returns kK/Mm

# All consistent - no conversion needed
```

### 3. Output: Convert to Desired Units

```python
# Get gradient value (in model units)
grad_value = uw.function.evaluate(grad, coords)

# Convert to physical units
grad_physical = uw.quantity(grad_value, "kK/Mm").to("K/km")
print(f"Gradient: {grad_physical}")  # 2.6 K/km
```

## Why This Approach?

1. **No hidden conversions** - you always know what units you're working with
2. **Derivatives just work** - no scaling issues
3. **Optimal numerics** - O(1) numbers throughout
4. **Standard scientific workflow** - same pattern as unit conversion in data analysis
```

## Related Files

### To Modify
- `src/underworld3/model.py` - Add `get_model_units()`
- `src/underworld3/discretisation/discretisation_mesh.py` - Unit conversion
- `src/underworld3/discretisation/discretisation_mesh_variables.py` - Variable units
- `src/underworld3/systems/solvers.py` - BC conversion
- `docs/beginner/tutorials/13-Scaling-Physical-Problems.ipynb` - Update examples

### To Remove
- Coordinate unit patching system in `src/underworld3/coordinates.py`
- `patch_coordinate_units()` function
- `UnitAwareBaseScalar` class (if only used for patching)

### To Keep Unchanged
- All solver internals
- Constitutive model implementations
- PETSc integration layer

## Open Questions

1. **Variable unit inference**: How to map variable types to model unit dimensions?
   - Temperature → 'temperature'
   - Velocity → 'velocity' (derived: length/time)
   - Pressure → 'pressure' (derived: force/area)

2. **Unit validation**: Should we validate dimensional consistency at system boundaries?
   - Pros: Catch user errors early
   - Cons: Performance overhead

3. **Default behavior**: If no reference quantities set?
   - Option A: Use SI base units (m, s, kg, K)
   - Option B: Require reference quantities
   - **Recommendation**: Option A (graceful degradation)

## Next Steps

1. Implement `model.get_model_units()` with unit rounding
2. Update mesh to get units from model
3. Test with simple Poisson solver
4. Expand to Stokes
5. Update all documentation
6. Remove coordinate patching system

## References

- Original bug report: `BUG-gradient-projection-with-reference-units.md`
- Derivative units implementation: `DERIVATIVE_UNITS_SUMMARY.md`
- Coordinate units technical note: `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`
