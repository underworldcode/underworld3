# Phase 4: Reference Quantity Validation - Implementation Summary

**Completed**: 2025-10-22
**Status**: ✅ All tests passing

## Overview

Phase 4 implements comprehensive validation for non-dimensional scaling reference quantities. This ensures users are notified when they create variables with units but haven't provided the necessary reference quantities to derive proper scaling coefficients.

## What Was Implemented

### 1. Reference Quantity Requirements Mapping

**File**: `src/underworld3/utilities/nondimensional.py`

Added `get_required_reference_quantities(units_str)` function that maps physical units to required reference quantities:

- **Velocity** (`m/s`): requires `plate_velocity`
- **Pressure** (`Pa`): requires `mantle_viscosity`, `plate_velocity`, `domain_depth`
- **Viscosity** (`Pa*s`): requires `mantle_viscosity`
- **Temperature** (`K`): requires `temperature_difference`
- **Length** (`m`): requires `domain_depth`
- **Time** (`s`): requires `domain_depth`, `plate_velocity`

### 2. Variable-Level Validation

**File**: `src/underworld3/utilities/nondimensional.py`

Added `validate_variable_reference_quantities(var_name, units_str, model)` function that:
- Checks if reference quantities are set on the model
- Verifies all required quantities for the variable's units are present
- Returns (is_valid, warning_message) tuple

### 3. Automatic Validation at Variable Creation

**File**: `src/underworld3/discretisation/discretisation_mesh_variables.py`

Modified `_BaseMeshVariable.__init__()` to:
- Call validation function when variable has units
- Issue `UserWarning` if required reference quantities are missing
- Provide helpful error messages guiding users to fix the issue

**Example Warning**:
```
UserWarning: Variable 'p' with units 'Pa' is missing required reference quantities:
  Missing: mantle_viscosity
  Pressure scale P₀ = η₀·V₀/L₀ requires: mantle_viscosity, plate_velocity, domain_depth
  Call model.set_reference_quantities() with the missing quantities.
Variable will use scaling_coefficient=1.0, which may lead to poor numerical conditioning.
```

### 4. Model-Level Comprehensive Validation

**File**: `src/underworld3/model.py`

Added `Model.validate_reference_quantities(raise_on_error=False)` method that:
- Validates all registered variables with units
- Returns dict with validation results:
  - `valid`: bool - overall validation status
  - `errors`: list of error messages
  - `warnings`: list of warning messages
- Optional `raise_on_error=True` to raise ValueError on validation failure

**Example Usage**:
```python
# Check validation before solving
result = model.validate_reference_quantities()
if not result['valid']:
    print(f"Found {len(result['errors'])} errors:")
    for error in result['errors']:
        print(f"  {error}")

# Or use strict mode
model.validate_reference_quantities(raise_on_error=True)  # Raises if invalid
```

## Testing

### Test Files Created

1. **`test_reference_quantity_validation.py`**:
   - Tests 3 scenarios: no reference quantities, incomplete quantities, complete quantities
   - Validates that appropriate warnings are generated
   - Confirms scaling coefficients are computed correctly when quantities are complete

2. **`test_model_validation_method.py`**:
   - Tests `Model.validate_reference_quantities()` method
   - Verifies error detection for missing quantities
   - Tests both `raise_on_error=False` and `raise_on_error=True` modes
   - Confirms validation passes when all quantities present

### Regression Testing

All existing ND scaling tests continue to pass:
- ✅ `tests/test_0818_stokes_nd.py`: 5/5 tests passing
- ✅ `test_nd_comprehensive_suite.py`: 3/3 tests passing (shear, SolCx, buoyancy)

## Benefits

### For Users

1. **Early Detection**: Warnings appear immediately when creating variables, not during solve
2. **Clear Guidance**: Error messages explain exactly which quantities are missing and why
3. **Flexible Validation**: Can use automatic warnings or explicit validation method
4. **Production Safety**: `raise_on_error=True` mode for strict validation in production code

### For Code Quality

1. **Prevents Silent Failures**: No more `scaling_coefficient=1.0` defaults silently used
2. **Better Error Messages**: Users know exactly what to fix and how to fix it
3. **Dimensional Correctness**: Validates the physics relationships (e.g., P₀ = η₀·V₀/L₀)
4. **Comprehensive Checking**: Model-level validation ensures all variables are properly configured

## Example Workflows

### Workflow 1: Basic Usage with Warnings
```python
import underworld3 as uw

# Set incomplete reference quantities
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
)

mesh = uw.meshing.StructuredQuadBox(...)

# Get automatic warning when creating pressure variable
p = uw.discretisation.MeshVariable('p', mesh, 1, degree=1, units='Pa')
# UserWarning: Variable 'p' with units 'Pa' is missing required reference quantities:
#   Missing: mantle_viscosity
#   ...

# Fix by adding missing quantity
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s"),  # Added!
)
```

### Workflow 2: Pre-Solve Validation
```python
import underworld3 as uw

# Set up model with variables
model = uw.get_default_model()
model.set_reference_quantities(...)
mesh = uw.meshing.StructuredQuadBox(...)
v = uw.discretisation.MeshVariable('v', mesh, 2, degree=2, units='m/s')
p = uw.discretisation.MeshVariable('p', mesh, 1, degree=1, units='Pa')

# Validate before solving
result = model.validate_reference_quantities()
if not result['valid']:
    raise RuntimeError(f"Invalid reference quantities: {result['errors']}")

# Proceed with solve
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.solve()
```

### Workflow 3: Strict Production Mode
```python
import underworld3 as uw

# Production code with strict validation
model = uw.get_default_model()
model.set_reference_quantities(...)

# This will raise ValueError immediately if any variable has missing quantities
model.validate_reference_quantities(raise_on_error=True)

# Safe to proceed - all variables properly configured
```

## Files Modified

1. `src/underworld3/utilities/nondimensional.py` (+56 lines)
   - `get_required_reference_quantities()` function
   - `validate_variable_reference_quantities()` function

2. `src/underworld3/discretisation/discretisation_mesh_variables.py` (+14 lines)
   - Validation check in `__init__()` after variable registration

3. `src/underworld3/model.py` (+63 lines)
   - `Model.validate_reference_quantities()` method

4. `planning/ND_SCALING_UX_ENHANCEMENTS.md` (updated)
   - Documented Phase 4 completion
   - Updated priority list

## Next Steps (Phase 5)

1. **User Documentation**: Create comprehensive guide for ND scaling workflow
2. **Migration Guide**: Document transition from manual ND to automatic
3. **Deprecation Warnings**: Add warnings for legacy `.data` write patterns
4. **Best Practices**: Document recommended patterns for setting reference quantities

## Technical Notes

### Pint Dimensionality Comparison

The validation uses Pint's dimensionality comparison to match units:
```python
from ..scaling import units as ureg
qty = ureg(units_str)
dim = qty.dimensionality

# Compare against known patterns
if dim == ureg.pascal.dimensionality:
    # This is a pressure variable
    required = ['mantle_viscosity', 'plate_velocity', 'domain_depth']
```

This handles unit equivalence automatically (e.g., 'Pa' = 'N/m²' = 'kg/(m·s²)').

### Warning Stacklevel

Uses `stacklevel=2` in warnings to show the user's code location, not the internal __init__:
```python
warnings.warn(message, UserWarning, stacklevel=2)
```

This makes warnings more actionable by pointing to where the variable was created.

### Model Registration

Validation relies on variables being registered with the default model:
```python
if _register:
    uw.get_default_model()._register_variable(self.name, self)
```

Variables created with `_register=False` won't be validated by `model.validate_reference_quantities()`.

## Conclusion

Phase 4 successfully implements comprehensive validation for non-dimensional scaling reference quantities. Users now receive clear, actionable warnings when they create variables without proper reference quantities, preventing silent failures and improving the user experience for ND scaling workflows.

The implementation:
- ✅ Passes all existing tests
- ✅ Provides helpful error messages
- ✅ Works both automatically (warnings) and explicitly (validation method)
- ✅ Guides users to fix configuration issues
- ✅ Prevents poor numerical conditioning from missing scales
