# Unified Units Interface - Future Enhancements

**Date**: 2025-10-16
**Status**: Discussion Topics for Future Development
**Related**: `UNIFIED-UNITS-INTERFACE-DESIGN.md`

## Current Status (Phase 1 Complete)

✅ Core unified interface implemented:
- `uw.function.evaluate()` returns `UWQuantity`/`UnitAwareArray`
- `MeshVariable.array` returns `UnitAwareArray` with `.to()` method
- `mesh.X.coords` returns `UnitAwareArray`
- `uw.get_units()` safely queries units from any object
- All have consistent `.to(target_units)` interface

✅ Test Results:
- 144/169 tests passing (85%)
- 9 "failures" are test isolation issues - all pass individually
- 16 skipped (planned features not yet implemented)

## Future Enhancement: UnitAwareArray Reduction Methods

### Issue Discovered (2025-10-16)

**Problem**: `UnitAwareArray` doesn't have `min()`, `max()`, `mean()`, `std()` methods that preserve units.

**Current Workaround**:
```python
# MeshVariable has min()/max() methods
gradT.min()  # Works - returns value with units
gradT.max()  # Works - returns value with units

# UnitAwareArray requires .magnitude
gradT_array = gradT.array.to("K/m")
gradT_array.magnitude.min()  # Need to access underlying numpy array
gradT_array.magnitude.max()  # Not as intuitive
```

**Desired Behavior**:
```python
# Natural pattern - reduction methods that preserve units
gradT_array = gradT.array.to("K/m")  # UnitAwareArray
gradT_array.min()  # Should return UWQuantity with units
gradT_array.max()  # Should return UWQuantity with units
gradT_array.mean() # Should return UWQuantity with units
gradT_array.std()  # Should return UWQuantity with units
```

**Benefits**:
1. **More intuitive**: Matches NumPy API expectations
2. **Consistent with design**: Units preserved through operations
3. **Better discoverability**: Users expect array methods to work
4. **Unified interface**: Same pattern as `.to()` - works everywhere

### Implementation Approach

**File**: `src/underworld3/utilities/unit_aware_array.py`

**Add methods**:
```python
class UnitAwareArray(np.ndarray):
    # ... existing code ...

    def min(self, axis=None, **kwargs):
        """Minimum value, preserving units."""
        from ..function.quantities import quantity
        result = np.min(self.view(np.ndarray), axis=axis, **kwargs)
        if self._units and np.isscalar(result):
            return quantity(float(result), self._units)
        elif self._units:
            return UnitAwareArray(result, units=self._units)
        return result

    def max(self, axis=None, **kwargs):
        """Maximum value, preserving units."""
        from ..function.quantities import quantity
        result = np.max(self.view(np.ndarray), axis=axis, **kwargs)
        if self._units and np.isscalar(result):
            return quantity(float(result), self._units)
        elif self._units:
            return UnitAwareArray(result, units=self._units)
        return result

    def mean(self, axis=None, **kwargs):
        """Mean value, preserving units."""
        from ..function.quantities import quantity
        result = np.mean(self.view(np.ndarray), axis=axis, **kwargs)
        if self._units and np.isscalar(result):
            return quantity(float(result), self._units)
        elif self._units:
            return UnitAwareArray(result, units=self._units)
        return result

    def std(self, axis=None, **kwargs):
        """Standard deviation, preserving units."""
        from ..function.quantities import quantity
        result = np.std(self.view(np.ndarray), axis=axis, **kwargs)
        if self._units and np.isscalar(result):
            return quantity(float(result), self._units)
        elif self._units:
            return UnitAwareArray(result, units=self._units)
        return result

    # Could also add: sum, var, ptp, argmin, argmax, etc.
```

**Testing Considerations**:
- Test scalar reduction (returns UWQuantity)
- Test axis-wise reduction (returns UnitAwareArray)
- Test with and without units
- Test kwargs pass-through (dtype, out, keepdims, etc.)
- Ensure backward compatibility

**Priority**: Medium - Nice to have, not critical for functionality

### Related Enhancement: SimpleMeshArrayView Methods

**Issue**: `MeshVariable.array` returns `SimpleMeshArrayView` which wraps `UnitAwareArray`, but the view doesn't delegate reduction methods.

**Current**:
```python
temperature.array.min()  # AttributeError: 'SimpleMeshArrayView' object has no attribute 'min'
temperature.min()        # Works - on MeshVariable
```

**Two options**:

1. **Add delegation in SimpleMeshArrayView** (simpler):
   ```python
   class SimpleMeshArrayView:
       def min(self, *args, **kwargs):
           return self._get_array_data().min(*args, **kwargs)
       # etc.
   ```

2. **Return UnitAwareArray directly** (bigger change):
   - Remove SimpleMeshArrayView wrapper entirely
   - Return UnitAwareArray with custom `__setitem__` for sync
   - Simpler architecture, more Pythonic

**Recommendation**: Start with option 1 (delegation), consider option 2 for future major refactoring.

## Other Future Enhancements

### 1. Temperature Offset Conversions (K ↔ °C)

**Status**: Known limitation - Pint offset conversions conflict with UnitAwareArray arithmetic.

**Current**:
```python
T_array = temperature.array  # In kelvin
T_celsius = T_array.to("degC")  # Fails with dimensionality error
```

**Solution**: Requires special handling in `.to()` method for offset units.

**Priority**: Low - Most conversions are multiplicative

### 2. Unit Validation Helpers

**From design doc** (`UNIFIED-UNITS-INTERFACE-DESIGN.md` lines 301-379):
```python
uw.check_units(obj, "m/s")           # Raises ValueError if incompatible
uw.assert_units(obj, "m/s", "BC")   # Defensive programming
uw.check_dimensionality(obj, "[length]/[time]")
```

**Priority**: Medium - Useful for boundary conditions and solver inputs

### 3. Evaluate with Unit-Aware Coordinates

**Status**: Tests skipped - planned feature not implemented

**Desired**:
```python
coords_with_units = uw.quantity([[0, 1], [2, 3]], "km")
result = uw.function.evaluate(T.sym, coords_with_units, coord_units="km")
```

**Priority**: Low - Current pattern works fine

### 4. Better Error Messages

**Enhancement**: When unit conversions fail or units are incompatible, provide suggestions:
```python
# Current:
# DimensionalityError: Cannot convert from 'cm/yr' to 'dimensionless'

# Improved:
# UnitConversionError: Cannot convert velocity (cm/yr) to dimensionless value.
# Hint: Did you mean to use .magnitude to get raw numerical values?
# Or: Did you forget to specify units for the target variable?
```

**Priority**: Medium - Better developer experience

## Critical Issue: Coordinate Scaling (2025-10-16) - FIXED ✅

### Problem Discovered

**User observation**: Derivatives failing with error:
```
ValueError: Can't calculate derivative wrt 2900000.0*N.y.
```

**Root cause**: The `_apply_units_scaling()` method in `coordinates.py` was multiplying mesh coordinates by scale factors, creating symbolic expressions like `2900000.0*N.y` instead of just `N.y`, which breaks differentiation.

**Example of the problem**:
```python
model.set_reference_quantities(domain_depth=uw.quantity(2900, "km"))
mesh = uw.meshing.Box(...)
# mesh.X[1] returned: 2900000.0*N.y  (scaled by km→m conversion)
# Should return: N.y
```

### Solution Applied (2025-10-16)

**File**: `src/underworld3/coordinates.py` line 723

**Change**: Disabled the `_apply_units_scaling()` call:
```python
# For all meshes: Apply scaling if the mesh has a model with units
# DISABLED: With explicit model units system, coordinates are already in model units
# No scaling needed - mesh.X should be same as mesh.N (both in model units)
# self._apply_units_scaling()
```

**Rationale**: With the explicit model units system:
- Mesh coordinates are stored in model units (e.g., kilometers)
- No conversion scaling is needed
- mesh.X should return pure symbolic coordinates (N.x, N.y, N.z)
- Units are metadata on the array, not embedded in the symbolic expression

**Impact**:
- ✅ `mesh.X[1]` now returns `N.y` (not scaled)
- ✅ Derivatives work correctly: `temperature.diff(mesh.X[1])` succeeds
- ✅ Projection solvers can compute derivatives at mesh nodes
- ✅ Consistent with explicit model units design

### Testing

**Test command**: See test script in session (2025-10-16)

**Results**:
```
Coordinate symbol: N.y
Expected: N.y (no scaling factor)

✅ Derivative expression created successfully!
Expression: Matrix([[{ \hspace{ 0.04pt } {T} }_{,1}(N.x, N.y)]])

✅ Gradient projection succeeded!
```

**Status**: RESOLVED - coordinate scaling disabled, derivatives working

### Future Consideration

The `_apply_units_scaling()` method (lines 732-791) may no longer be needed with explicit model units. Consider removing it entirely in future cleanup if no other code paths depend on it.

## Critical Issue: Mesh Units Parameter (2025-10-16) - FIXED ✅

### Problem Discovered

**User observation**: `mesh = uw.meshing.UnstructuredSimplexBox(..., units="inches")` creates a mesh with units="inches" successfully.

**Expected behavior** (per EXPLICIT-MODEL-UNITS-DESIGN.md):
- Mesh units should come from `model.get_coordinate_unit()`, not user parameter
- User-specified `units` parameter should be ignored or deprecated
- All meshes in the same model should have the same coordinate units

**Actual behavior**:
- `units` parameter is accepted and stored: `self.units = units` (discretisation_mesh.py:189)
- Arbitrary units like "inches" work fine
- Docstring says "DEPRECATED" but no enforcement

**Root cause**: Explicit model units design documented but not fully implemented.

### Impact

**Allows inconsistent configurations**:
```python
model = uw.get_default_model()
model.set_reference_quantities(domain_depth=uw.quantity(500, "km"))
# Model expects coordinates in 'Mm'

mesh = uw.meshing.Box(..., units="inches")  # Shouldn't work!
# mesh.X.coords returns values with units="inches"
# But derivatives expect 'Mm'
# → Dimensional chaos
```

**This could lead to**:
- Wrong derivative units (dT/dx in K/inches instead of K/Mm)
- Inconsistent calculations between variables
- Silent errors in models with mixed units

### Recommended Fix

**Phase 1: Deprecation Warning** (Low risk)
```python
class Mesh:
    def __init__(self, ..., units=None, ...):
        if units is not None:
            import warnings
            warnings.warn(
                "The 'units' parameter is deprecated. Mesh units are determined "
                "by model.get_coordinate_unit(). This parameter will be ignored in "
                "future versions.",
                DeprecationWarning,
                stacklevel=2
            )

        # Get units from model (override user input)
        model = uw.get_default_model()
        self.units = model.get_coordinate_unit()  # From model, not user!
```

**Phase 2: Remove Parameter** (Breaking change - major version)
```python
def __init__(self, ..., verbose=False):  # Remove units parameter entirely
    model = uw.get_default_model()
    self.units = model.get_coordinate_unit()
```

**Phase 3: Update All Mesh Constructors**
- Remove `units` from `UnstructuredSimplexBox()` signature (cartesian.py:43)
- Remove from `StructuredQuadBox()` (cartesian.py:671)
- Remove from `BoxInternalBoundary()` (cartesian.py:215)
- Remove from all meshing functions
- Update documentation

### Migration Path for Users

**Old pattern** (currently works, shouldn't):
```python
mesh = uw.meshing.Box(maxCoords=(1000, 500), units="km")
```

**New pattern** (explicit):
```python
# Set model units once
model = uw.get_default_model()
model.set_reference_quantities(domain_depth=uw.quantity(500, "km"))

# Mesh uses model units automatically
mesh = uw.meshing.Box(
    maxCoords=(1000*uw.units.km, 500*uw.units.km)
)
# mesh.units = 'Mm' (from model)
# Coordinates auto-converted: 1000 km → 1.0 Mm
```

### Status: RESOLVED ✅ (2025-10-16)

**Implementation completed**:
1. ✅ Added `Model.get_coordinate_unit()` method
2. ✅ Added `Model._lock_units()` and `_check_units_locked()` methods
3. ✅ Modified `Mesh.__init__` to get units from model (with user warning)
4. ✅ Model units locked after first mesh creation
5. ✅ Updated Notebook 12 to demonstrate correct pattern
6. ✅ All test scenarios passing

**Files modified**:
- `src/underworld3/model.py` - Added unit locking mechanism
- `src/underworld3/discretisation/discretisation_mesh.py` - Mesh inherits from model
- `docs/beginner/tutorials/12-Units_System.ipynb` - Updated documentation

### Related

- See `EXPLICIT-MODEL-UNITS-DESIGN.md` for full design
- Related to coordinate scaling fix (see above)
- Affects: All meshing functions, coordinate scaling, derivative calculations

## Discussion Points

### Design Philosophy

**Question**: How far should unit awareness extend?

**Options**:
1. **Conservative**: Keep it explicit - use `.magnitude` when you need numbers
2. **Aggressive**: Make everything unit-aware - arrays behave like quantities
3. **Hybrid** (current): Unit-aware when helpful, explicit extraction when needed

**Current approach leans toward option 3**, with room to expand toward option 2 if user feedback supports it.

### Performance Considerations

**Question**: Do unit-aware operations add significant overhead?

**Current**: Minimal overhead - units are metadata, operations are still NumPy
- `.to()` conversion: One multiplication, negligible
- `get_units()`: Attribute lookup, fast
- Wrapping in UnitAwareArray: View creation, no data copy

**To monitor**: If unit checking in hot loops becomes issue, could add `@fast` decorator to skip checks.

### Backward Compatibility

**Principle**: Never break existing code

**Strategy**:
- New features should extend, not replace
- Old patterns continue to work
- Deprecate with warnings, never remove suddenly
- Document migration paths

**Example**: `.magnitude` will always work, even if we add reduction methods.

## Next Steps

1. **Gather user feedback** on Notebook 12 and examples
2. **Identify common pain points** in real workflows
3. **Prioritize enhancements** based on user needs
4. **Implement high-value additions** (reduction methods likely first)
5. **Update documentation** as features are added

## Related Documentation

- `UNIFIED-UNITS-INTERFACE-DESIGN.md` - Main design document
- `EXPLICIT-MODEL-UNITS-DESIGN.md` - Model units system
- `DERIVATIVE_UNITS_SUMMARY.md` - Derivative units implementation
- `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md` - Technical details
- `docs/beginner/tutorials/12-Units_System.ipynb` - User-facing examples
