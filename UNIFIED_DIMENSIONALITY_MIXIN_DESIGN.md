# Unified DimensionalityMixin Design

**Date:** 2025-01-07
**Goal:** Merge UnitAwareMixin functionality into DimensionalityMixin
**Strategy:** Keep the name DimensionalityMixin, absorb all units functionality

---

## Design Principle

**"Dimensionality includes units"**

Non-dimensionalization is fundamentally a units operation - scaling coefficients have units, dimensional analysis requires units. Therefore, one mixin handles both.

---

## Unified DimensionalityMixin API

### Core Properties

```python
class DimensionalityMixin:
    """
    Unified mixin for units tracking, dimensional analysis, and non-dimensionalization.

    Replaces both the old UnitAwareMixin and DimensionalityMixin.
    Now handles everything related to physical dimensions and units.
    """

    def __init__(self, *args, units=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Units tracking (from UnitAwareMixin)
        self._units = None
        self._pint_backend = None  # Lazy initialization

        # Scaling/non-dimensionalization (from old DimensionalityMixin)
        self._scaling_coefficient = 1.0
        self._is_nondimensional = False
        self._original_units = None
        self._original_dimensionality = None

        # Initialize units if provided
        if units:
            self.set_units(units)

    # ========================================
    # PROPERTIES - Read-only interfaces
    # ========================================

    @property
    def units(self) -> Optional[str]:
        """
        Get units string (e.g., 'm', 'kg/s', 'kelvin').

        Returns None if dimensionless or non-dimensional.
        """
        if self._is_nondimensional:
            return None  # Non-dimensional has no units
        return self._units

    @property
    def has_units(self) -> bool:
        """Check if this object has dimensional units."""
        return self._units is not None and not self._is_nondimensional

    @property
    def dimensionality(self) -> Optional[dict]:
        """
        Get Pint dimensionality dict (e.g., {'[length]': 1, '[time]': -1} for velocity).

        Returns None for dimensionless or non-dimensional quantities.
        """
        if self._is_nondimensional:
            return self._original_dimensionality

        if not self._units:
            return None

        # Use Pint to get proper dimensionality
        backend = self._get_backend()
        qty = backend.create_quantity(1.0, self._units)
        return backend.get_dimensionality(qty)

    @property
    def scaling_coefficient(self) -> float:
        """Get the reference scale for non-dimensionalization."""
        return self._scaling_coefficient

    @scaling_coefficient.setter
    def scaling_coefficient(self, value):
        """
        Set the reference scale for non-dimensionalization.

        Args:
            value: Scaling coefficient (float, UWQuantity, or Pint quantity)
        """
        if value is None or value == 0:
            raise ValueError("Scaling coefficient must be non-zero")

        # Handle UWQuantity or Pint quantities - extract magnitude
        if hasattr(value, 'magnitude'):
            # If it has units, convert to our units first
            if hasattr(value, 'to') and self._units:
                try:
                    value_in_my_units = value.to(self._units)
                    self._scaling_coefficient = float(value_in_my_units.magnitude)
                except:
                    self._scaling_coefficient = float(value.magnitude)
            else:
                self._scaling_coefficient = float(value.magnitude)
        else:
            self._scaling_coefficient = float(value)

    @property
    def is_nondimensional(self) -> bool:
        """Check if currently in non-dimensional state."""
        return self._is_nondimensional

    # ========================================
    # UNITS OPERATIONS (from UnitAwareMixin)
    # ========================================

    def set_units(self, units: str):
        """
        Set units for this object.

        Args:
            units: Units string (e.g., 'm', 'kg/s', 'kelvin')
        """
        if units is None:
            self._units = None
            return

        # Validate units via Pint
        backend = self._get_backend()
        try:
            qty = backend.create_quantity(1.0, units)
            self._units = str(backend.get_units(qty))
        except Exception as e:
            raise ValueError(f"Invalid units '{units}': {e}")

    def to(self, target_units: str):
        """
        Convert to different units (Pint-compatible interface).

        Args:
            target_units: Target units string

        Returns:
            New instance with converted values and units
        """
        if not self._units:
            raise ValueError("Cannot convert units - object has no units")

        backend = self._get_backend()

        # Get current value
        if hasattr(self, 'value'):
            current_value = self.value
        elif hasattr(self, 'array'):
            current_value = np.array(self.array)
        else:
            raise ValueError("Cannot convert - no value or array attribute")

        # Convert using Pint
        source_qty = backend.create_quantity(current_value, self._units)
        target_qty = backend.convert_units(source_qty, target_units)

        # Create new instance with converted values
        # (Implementation depends on whether this is UWQuantity, Variable, etc.)
        return self._create_converted_instance(
            backend.get_magnitude(target_qty),
            str(backend.get_units(target_qty))
        )

    def check_units_compatibility(self, other) -> bool:
        """
        Check if units are compatible with another object.

        Args:
            other: Another object with units

        Returns:
            True if dimensionally compatible, False otherwise
        """
        if not self.has_units or not hasattr(other, 'has_units') or not other.has_units:
            return False

        backend = self._get_backend()
        qty1 = backend.create_quantity(1.0, self._units)
        qty2 = backend.create_quantity(1.0, other.units)

        return backend.check_dimensionality(qty1, qty2)

    # ========================================
    # NON-DIMENSIONALIZATION (from old DimensionalityMixin)
    # ========================================

    @property
    def nd_array(self):
        """
        Get non-dimensional array values.

        Convenience property for accessing array data in non-dimensional form.
        """
        if not hasattr(self, 'array'):
            raise AttributeError(f"{type(self).__name__} does not have array property")

        return np.array(self.array) / self._scaling_coefficient

    def from_nd(self, nd_value):
        """
        Convert a non-dimensional value back to dimensional form.

        Args:
            nd_value: Non-dimensional value

        Returns:
            Dimensional value (nd_value * scaling_coefficient)
        """
        return nd_value * self._scaling_coefficient

    def set_reference_scale(self, scale):
        """
        Set reference scale for non-dimensionalization.

        Alias for setting scaling_coefficient property.

        Args:
            scale: Reference scale value
        """
        self.scaling_coefficient = scale

    # ========================================
    # BACKEND MANAGEMENT (internal)
    # ========================================

    def _get_backend(self):
        """Get or create Pint backend (lazy initialization)."""
        if self._pint_backend is None:
            from .units_mixin import PintBackend
            self._pint_backend = PintBackend()
        return self._pint_backend

    def _create_converted_instance(self, magnitude, units):
        """
        Create new instance with converted values.

        Subclasses should override this to return appropriate type.
        Default implementation for variables.
        """
        # For UWQuantity
        if hasattr(self, 'value'):
            from underworld3.function.quantities import UWQuantity
            return UWQuantity(magnitude, units)

        # For Variables - would need proper implementation
        # This is a placeholder
        raise NotImplementedError(
            f"{type(self).__name__} must implement _create_converted_instance()"
        )
```

---

## Migration Strategy

### Phase 1: Add Units Functionality to DimensionalityMixin

**File:** `src/underworld3/utilities/dimensionality_mixin.py`

**Changes:**
1. Add `_units`, `_pint_backend` attributes
2. Add `.units`, `.has_units` properties
3. Add `.set_units()`, `.to()`, `.check_units_compatibility()` methods
4. Keep all existing scaling/non-dimensional methods
5. Update `.dimensionality` to use Pint backend

**Result:** DimensionalityMixin is now a complete units + dimensionality handler

### Phase 2: Migrate Classes Currently Using Both Mixins

**UWQuantity** (`function/quantities.py`):
```python
# BEFORE
class UWQuantity(DimensionalityMixin, UnitAwareMixin):
    pass

# AFTER
class UWQuantity(DimensionalityMixin):
    pass  # DimensionalityMixin now has everything!
```

**EnhancedMeshVariable** (`discretisation/persistence.py`):
```python
# BEFORE
class EnhancedMeshVariable(DimensionalityMixin, UnitAwareMixin, MathematicalMixin):
    pass

# AFTER
class EnhancedMeshVariable(DimensionalityMixin, MathematicalMixin):
    pass
```

### Phase 3: Migrate Classes Using UnitAwareMixin Only

**EnhancedSwarmVariable** (`discretisation/enhanced_variables.py`):
```python
# BEFORE
class EnhancedSwarmVariable(UnitAwareMixin, _SwarmVariable):
    pass

# AFTER
class EnhancedSwarmVariable(DimensionalityMixin, _SwarmVariable):
    pass
```

### Phase 4: Remove UnitAwareMixin

**File:** `src/underworld3/utilities/units_mixin.py`

**Keep:**
- `UnitsBackend` abstract class (needed by DimensionalityMixin)
- `PintBackend` implementation (used by DimensionalityMixin)

**Remove:**
- `UnitAwareMixin` class entirely
- All example/test code in that file

**Update deprecation warning:**
```python
# At top of file
"""
DEPRECATED: Most of this module has been removed.

The UnitAwareMixin class has been merged into DimensionalityMixin.
Only the backend classes (UnitsBackend, PintBackend) remain for use
by DimensionalityMixin.

See utilities/dimensionality_mixin.py for the unified implementation.
"""
```

---

## Key Advantages

### 1. Conceptual Clarity ✅
**One mixin for all dimension-related functionality**
- Units tracking
- Unit conversion
- Dimensional analysis
- Non-dimensionalization
- Scaling coefficients

All in one place because they're all related!

### 2. Simpler Inheritance ✅
**BEFORE:**
```python
class UWQuantity(DimensionalityMixin, UnitAwareMixin):
    # Two mixins with overlapping functionality
    # Which .dimensionality gets called?
```

**AFTER:**
```python
class UWQuantity(DimensionalityMixin):
    # One mixin, clear behavior
```

### 3. No Synchronization Issues ✅
**No more worrying about:**
- DimensionalityMixin and UnitAwareMixin getting out of sync
- Which mixin's `.dimensionality` takes precedence
- MRO (Method Resolution Order) complexity

### 4. Keep the Good Name ✅
**DimensionalityMixin** is actually the better name:
- "Dimensionality" encompasses both units and scaling
- Already used in production (SwarmVariable, _MeshVariable)
- More physics-oriented than "UnitAwareMixin"

### 5. Backward Compatible ✅
**Existing code keeps working:**
```python
# All existing DimensionalityMixin usage unchanged
var.scaling_coefficient = 1000
var.nd_array
var.from_nd(value)

# Plus new functionality
var.set_units("m")
var.to("km")
var.check_units_compatibility(other)
```

---

## Implementation Checklist

### Core Implementation
- [ ] Add `_units`, `_pint_backend` to `__init__`
- [ ] Add `.units` property (read-only)
- [ ] Add `.has_units` property
- [ ] Enhance `.dimensionality` to use Pint backend
- [ ] Add `.set_units(units)` method
- [ ] Add `.to(target_units)` method
- [ ] Add `.check_units_compatibility(other)` method
- [ ] Add `._get_backend()` helper (lazy Pint initialization)
- [ ] Keep all existing scaling methods unchanged

### Migration
- [ ] Update `UWQuantity` to use single mixin
- [ ] Update `EnhancedMeshVariable` to use single mixin
- [ ] Update `EnhancedSwarmVariable` to use DimensionalityMixin
- [ ] Remove `UnitAwareMixin` class from units_mixin.py
- [ ] Keep `PintBackend` class for use by DimensionalityMixin
- [ ] Update all imports

### Testing
- [ ] Run existing DimensionalityMixin tests (should still pass)
- [ ] Run existing UnitAwareMixin tests (after migration)
- [ ] Run closure tests (should still be 24/30)
- [ ] Test unit conversion on variables
- [ ] Test non-dimensionalization still works

### Documentation
- [ ] Update DimensionalityMixin docstring
- [ ] Update method table document
- [ ] Add migration notes to UNITS_REFACTOR_PROGRESS.md
- [ ] Update CLAUDE.md if needed

---

## What NOT to Change

### Keep UnitAwareArray Separate ✅
**UnitAwareArray is NOT a mixin** - it's a concrete numpy subclass.
- Has its own complete implementation
- No mixin complexity
- Works perfectly as-is
- **Leave it alone!**

### Keep UnitAwareExpression Separate ✅
**UnitAwareExpression** is for SymPy expressions.
- Different use case (symbolic vs data)
- Has its own complete implementation
- **Leave it alone!**

### Keep Backend Classes ✅
**UnitsBackend and PintBackend** in units_mixin.py
- Used by DimensionalityMixin for Pint integration
- Abstract interface pattern is good
- Just remove UnitAwareMixin class, keep backends

---

## Estimated Effort

### Phase 1: Core Implementation
**Time:** 2-3 hours
- Add new methods to DimensionalityMixin
- Keep all existing methods
- No breaking changes

### Phase 2: Migration
**Time:** 1-2 hours
- Update class inheritance (3 classes)
- Remove duplicate imports
- Simple mechanical changes

### Phase 3: Testing
**Time:** 1 hour
- Run test suite
- Fix any issues (expect minimal)

### Phase 4: Cleanup
**Time:** 30 minutes
- Remove UnitAwareMixin class
- Update documentation

**Total:** ~5-7 hours of focused work

---

## Risk Assessment

**Risk:** LOW ✅

**Why:**
1. Additive changes to DimensionalityMixin (backward compatible)
2. Simple inheritance changes (no logic changes)
3. Comprehensive test suite will catch issues
4. Can test incrementally (one class at a time)
5. Easy to roll back if needed

**Mitigation:**
- Do Phase 1 first, test thoroughly before migration
- Migrate one class at a time, test after each
- Keep git commits granular for easy rollback

---

## Success Criteria

### After Completion:
- ✅ One mixin handles units + dimensionality
- ✅ No duplicate functionality
- ✅ All tests still passing (24/30 closure, 100% units)
- ✅ Clearer inheritance hierarchy
- ✅ Better conceptual model

---

## Next Steps

1. **Get approval** for this design
2. **Implement Phase 1** (add units methods to DimensionalityMixin)
3. **Test Phase 1** (ensure no regressions)
4. **Migrate classes** (Phase 2)
5. **Clean up** (Phase 3-4)
6. **Then proceed to Phase 3 bug investigation** (power units)

---

**Ready to implement?** This is a clean, simple consolidation that makes the codebase more maintainable.

