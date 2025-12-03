# Non-Dimensional Scaling UX Enhancements

## Future Features for User Experience

### 1. Solver Inspection - `.view()` Method

**Goal**: Allow users to inspect what the solver is actually working with in ND form.

**Feature**: Enhanced `.view()` method for solvers that shows:
- F0, F1 expressions in both symbolic and ND form
- Jacobian structure with scaling information
- What PETSc actually sees (ND values)
- Mapping between dimensional and ND quantities

**Example Usage**:
```python
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# View the solver setup
stokes.view(show_scaling=True)

# Output:
# ========================================
# Stokes Solver Configuration
# ========================================
#
# Scaling Coefficients:
#   V₀ = 1.584e-09 m/s
#   P₀ = 1.584e+04 Pa
#   L₀ = 1.000e+06 m
#   T₀ = 6.312e+14 s
#
# F0 (Body Force):
#   Dimensional: (0, -ρ₀*α*g*T)
#   Non-dimensional: (0, -Ra*T̂)
#   PETSc sees: (0, -1e4*T̂)
#
# F1 (Stress Tensor):
#   Dimensional: -pI + 2η∇v
#   Non-dimensional: -p̂I + 2η̂∇̂v̂
#   PETSc sees: -p̂I + 2*v̂_ij
#
# Jacobian Structure:
#   [K_vv  K_vp]  (9×9 velocity-velocity coupling)
#   [K_pv  K_pp]  (3×3 pressure blocks)
#   Condition number (estimated): ~1e3 (well-conditioned)
```

**Implementation Location**: `src/underworld3/systems/solvers.py`

**Benefits**:
- Helps users understand what's happening under the hood
- Debugging tool for dimensional issues
- Educational for learning ND scaling
- Verification that scaling is working correctly

---

### 2. Unit Checking After Solver Setup

**Goal**: Validate dimensional consistency of all solver terms before solve.

**Feature**: Automatic unit checking when solver is created/configured:

```python
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.bodyforce = (0, -Ra * T.sym[0])  # Rayleigh number term

# Automatic unit check (happens on solver creation or first solve)
# Validates:
#   - F0 has correct force/volume units
#   - F1 has correct stress units
#   - BCs are dimensionally consistent
#   - Body forces match momentum equation dimensions

# If inconsistent:
# UnitError: Body force term has units 'm/s²' but should have 'N/m³'
#   In: stokes.bodyforce = (0, -Ra * T.sym[0])
#   Ra is dimensionless, T has units 'K', but force needs 'N/m³'
#   Suggestion: multiply by density*alpha*g to get proper units
```

**Implementation Location**:
- `src/underworld3/systems/solvers.py` - validation methods
- `src/underworld3/utilities/units.py` - unit checking logic

**Checks to Perform**:
1. **F0 (Body Force)**: Should have units of force/volume [N/m³] or equivalent
2. **F1 (Flux)**: Should have units of stress [Pa] or equivalent
3. **Boundary Conditions**: Should match variable units
4. **Source Terms**: Should match equation requirements
5. **Material Properties**: Should have consistent units (viscosity in Pa·s, etc.)

**User Control**:
```python
# Enable/disable unit checking
uw.enable_unit_checking(strict=True)  # Raises errors
uw.enable_unit_checking(strict=False) # Warnings only
uw.disable_unit_checking()             # Skip checks (for performance)

# Check specific solver
stokes.check_units()  # Manual validation
```

**Benefits**:
- Catches dimensional errors early (before solve)
- Provides helpful suggestions for fixes
- Educational - teaches proper dimensional analysis
- Prevents subtle bugs from unit inconsistencies

---

### 3. Reference Quantity Validation (Phase 4) ✅ **IMPLEMENTED**

**Status**: Complete (2025-10-22)

**Goal**: Ensure all required reference quantities are properly defined.

**Implementation**:

1. **Automatic Validation at Variable Creation**:
   ```python
   # Creating a variable with units triggers validation
   v = uw.discretisation.MeshVariable('v', mesh, 2, degree=2, units='m/s')
   # UserWarning: Variable 'v' has units 'm/s' but no reference quantities are set.
   #   Call model.set_reference_quantities() before creating variables with units.
   ```

2. **Helpful Error Messages for Missing Quantities**:
   ```python
   model.set_reference_quantities(
       domain_depth=uw.quantity(1000, "km"),
       plate_velocity=uw.quantity(5, "cm/year"),
       # Missing: mantle_viscosity!
   )

   p = uw.discretisation.MeshVariable('p', mesh, 1, degree=1, units='Pa')
   # UserWarning: Variable 'p' with units 'Pa' is missing required reference quantities:
   #   Missing: mantle_viscosity
   #   Pressure scale P₀ = η₀·V₀/L₀ requires: mantle_viscosity, plate_velocity, domain_depth
   #   Call model.set_reference_quantities() with the missing quantities.
   ```

3. **Comprehensive Validation Method**:
   ```python
   # Validate all variables before solving
   result = model.validate_reference_quantities(raise_on_error=False)

   if not result['valid']:
       print(f"Found {len(result['errors'])} validation errors:")
       for error in result['errors']:
           print(f"  - {error}")

   # Or use strict mode to raise exception
   model.validate_reference_quantities(raise_on_error=True)  # Raises ValueError if invalid
   ```

**Files Modified**:
- `src/underworld3/utilities/nondimensional.py`:
  - Added `get_required_reference_quantities()` function
  - Added `validate_variable_reference_quantities()` function
  - Maps common units (Pa, m/s, K, etc.) to required reference quantities

- `src/underworld3/discretisation/discretisation_mesh_variables.py`:
  - Added validation check in `__init__()` after variable registration
  - Warns users when required reference quantities are missing

- `src/underworld3/model.py`:
  - Added `Model.validate_reference_quantities()` method
  - Validates all registered variables with units
  - Returns dict with validation results or raises ValueError

**Testing**:
- `test_reference_quantity_validation.py` - Phase 4 requirements test
- `test_model_validation_method.py` - Comprehensive validation method test
- All existing ND tests still pass (5/5 Stokes tests passing)

---

### 4. Read-Only `.data` Property (Deprecation)

**Goal**: Make `.data` read-only to clarify ND vs dimensional interfaces.

**Timeline**:
- Phase 5.1: Add deprecation warnings on `.data` writes
- Phase 5.2: Make `.data` read-only (raises error on write)
- Phase 6: Remove write capability entirely

**See**: Main CLAUDE.md for details on data property migration

---

## Priority

1. ✅ **Phase 4** (Complete): Reference quantity validation
2. **Phase 5** (Next): User documentation and deprecation warnings
3. **Future**: Solver `.view()` method
4. **Future**: Unit checking system
5. **Future**: Read-only `.data` enforcement

## Completed Phases

### Phase 4: Reference Quantity Validation (Complete 2025-10-22)

**What was implemented**:
- Automatic validation warnings when creating MeshVariables with units but missing reference quantities
- `get_required_reference_quantities()` function to map units to required reference quantities
- `validate_variable_reference_quantities()` function to validate individual variables
- `Model.validate_reference_quantities()` method for comprehensive pre-solve validation
- Helpful error messages guiding users to fix missing quantities

**Impact**:
- Early detection of misconfigured ND scaling (warnings at variable creation)
- Comprehensive validation before solving (explicit validation method)
- Clear guidance on which reference quantities are needed for each unit type
- Prevents silent failures where scaling_coefficient defaults to 1.0

---

## Implementation Notes

### Solver `.view()` Implementation Sketch

```python
class Stokes:
    def view(self, show_scaling=False, show_jacobian=False):
        """Display solver configuration."""
        print("="*60)
        print("Stokes Solver Configuration")
        print("="*60)

        if show_scaling and uw.is_nondimensional_scaling_active():
            print("\nScaling Coefficients:")
            print(f"  V₀ = {self.u.scaling_coefficient}")
            print(f"  P₀ = {self.p.scaling_coefficient}")
            print(f"  L₀ = {self.mesh.length_scale}")

        print("\nEquations:")
        print(f"  F0 (body force): {self.F0}")
        print(f"  F1 (stress): {self.F1}")

        if show_jacobian:
            # Show Jacobian structure
            print("\nJacobian Structure:")
            # ... implementation ...
```

### Unit Checking Implementation Sketch

```python
def check_equation_units(equation, expected_units):
    """Validate dimensional consistency of equation terms."""

    # Get units of all terms
    for term in equation.atoms():
        term_units = uw.get_units(term)

        if not are_units_compatible(term_units, expected_units):
            raise UnitError(
                f"Term '{term}' has units '{term_units}' "
                f"but equation requires '{expected_units}'"
            )
```

---

## Related Documentation

- Main implementation: `CLAUDE.md` - ND scaling architecture
- Technical details: `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`
- User guide: `docs/user/nondimensional-scaling.qmd` (to be created in Phase 5)
