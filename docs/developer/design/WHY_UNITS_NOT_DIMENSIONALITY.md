# Why "Units" Not "Dimensionality" - User-Facing Terminology

**Date:** 2025-01-07
**Decision:** Merge DimensionalityMixin INTO UnitAwareMixin, keep "units" terminology
**Reason:** User communication and API consistency

---

## The User Perspective

### What Users Say:
- ✅ "This variable has **units** of meters"
- ✅ "I need to convert **units** to kilometers"
- ✅ "The **units** don't match"
- ❌ "This variable has **dimensionality** [length]"
- ❌ "I need to convert **dimensionality**"

### What Users Understand:
- **Units**: ✅ Concrete, everyday concept (meters, seconds, kilograms)
- **Dimensionality**: ⚠️ Abstract physics concept ([length], [time], [mass])

### What the API Shows:
```python
# User-friendly - clear what's happening
var.units = "m"
var.to("km")
mesh.units = "km"

# Less clear - what does this mean?
var.dimensionality = ???
var.to_dimensionality(???)
```

---

## Current Investment in "Units" Terminology

### Documentation:
- "Units System" (not "Dimensionality System")
- "Unit-Aware Arrays"
- "Unit Conversion"
- "Units Tests" (test_07*_units*.py, test_08*_*.py)

### Classes:
- `UnitAwareArray` ✅
- `UnitAwareExpression` ✅
- `UWQuantity` (units, not dimensionality) ✅

### Properties:
- `.units` (everywhere)
- `.has_units` (everywhere)
- `.to()` - unit conversion

### Functions:
- `uw.get_units()`
- `uw.quantity()` - creates unit-aware quantity

### Files:
- `units.py`
- `unit_conversion.py`
- `unit_aware_array.py`
- `test_0700_units_system.py`
- `UNITS_REFACTOR_PLAN.md`
- `UNITS_TEST_RESULTS_BASELINE.md`

---

## The Problem with "Dimensionality"

### 1. Not User-Facing
**Dimensionality is an implementation detail:**
- [length]¹ [time]⁻¹ is the **dimensionality** of velocity
- "m/s" is the **units** of velocity
- Users care about "m/s", not "[length]/[time]"

### 2. Technical Jargon
**"Dimensionality" is physics jargon:**
- Most users won't know what it means
- Even physicists say "units" in casual conversation
- It's a barrier to adoption

### 3. API Confusion
```python
# Which is clearer?
var.units = "m"              # ✅ Clear
var.dimensionality = "[L]"   # ❌ What?

# Which is more intuitive?
var.to("km")                 # ✅ Obvious
var.to_dimensionality(...)   # ❌ How?
```

---

## What About Non-Dimensionalization?

### The Key Insight:
**Non-dimensionalization is a UNITS operation!**

```python
# Non-dimensionalization example
var.units = "m"
var.scaling_coefficient = 1000  # meters

# Get non-dimensional value
nd_value = var.nd_array  # Array divided by 1000

# The scaling coefficient HAS UNITS (meters)!
# So non-dimensionalization IS part of the units system
```

### Better Terminology:
- **"Reference units"** instead of "dimensionality"
- **"Scaled units"** instead of "non-dimensional"
- **"Unit scaling"** instead of "dimensionalization"

---

## Correct Architecture: Units First

### UnitAwareMixin (keep this name!)

```python
class UnitAwareMixin:
    """
    Mixin for units tracking, conversion, and scaling.

    Provides:
    - Units tracking (meters, kelvin, etc.)
    - Unit conversion (m → km)
    - Dimensional analysis via Pint
    - Non-dimensionalization (scaling)
    - Reference scales
    """

    def __init__(self, *args, units=None, **kwargs):
        super().__init__(*args, **kwargs)

        # ========================================
        # UNITS TRACKING
        # ========================================
        self._units = None
        self._pint_backend = None  # Lazy init

        # ========================================
        # SCALING / NON-DIMENSIONALIZATION
        # (absorbed from DimensionalityMixin)
        # ========================================
        self._scaling_coefficient = 1.0
        self._is_nondimensional = False
        self._original_units = None

        if units:
            self.set_units(units)

    # ========================================
    # UNITS INTERFACE (user-facing)
    # ========================================

    @property
    def units(self) -> Optional[str]:
        """Get units string (e.g., 'm', 'kg/s')."""
        if self._is_nondimensional:
            return None
        return self._units

    @property
    def has_units(self) -> bool:
        """Check if object has units."""
        return self._units is not None and not self._is_nondimensional

    def set_units(self, units: str):
        """Set units (e.g., 'm', 'kelvin')."""
        ...

    def to(self, target_units: str):
        """Convert to different units."""
        ...

    def check_units_compatibility(self, other) -> bool:
        """Check if units are compatible."""
        ...

    # ========================================
    # DIMENSIONALITY (implementation detail)
    # ========================================

    @property
    def dimensionality(self) -> Optional[dict]:
        """
        Get Pint dimensionality dict.

        This is an advanced property for dimensional analysis.
        Most users should use .units instead.
        """
        if not self._units:
            return None

        backend = self._get_backend()
        qty = backend.create_quantity(1.0, self._units)
        return backend.get_dimensionality(qty)

    # ========================================
    # SCALING / NON-DIMENSIONALIZATION
    # (absorbed from DimensionalityMixin)
    # ========================================

    @property
    def scaling_coefficient(self) -> float:
        """Get reference scale for non-dimensionalization."""
        return self._scaling_coefficient

    @scaling_coefficient.setter
    def scaling_coefficient(self, value):
        """Set reference scale (can have units)."""
        ...

    @property
    def nd_array(self):
        """Get non-dimensional array (array / scaling_coefficient)."""
        return np.array(self.array) / self._scaling_coefficient

    def from_nd(self, nd_value):
        """Convert from non-dimensional (nd_value * scaling_coefficient)."""
        return nd_value * self._scaling_coefficient

    def set_reference_scale(self, scale):
        """Set reference scale for non-dimensionalization."""
        self.scaling_coefficient = scale
```

---

## Migration Strategy: Absorb DimensionalityMixin

### Step 1: Enhance UnitAwareMixin
**File:** `src/underworld3/utilities/units_mixin.py`

**Add from DimensionalityMixin:**
- `_scaling_coefficient` attribute
- `_is_nondimensional` attribute
- `_original_units` attribute
- `.scaling_coefficient` property
- `.nd_array` property
- `.from_nd()` method
- `.set_reference_scale()` method

**Keep existing:**
- All current units functionality
- `.units`, `.has_units`, `.dimensionality` properties
- `.set_units()`, `.to()` methods
- Pint backend integration

**Result:** UnitAwareMixin has everything

### Step 2: Update Classes Using DimensionalityMixin Only

**SwarmVariable** (`swarm.py`):
```python
# BEFORE
class SwarmVariable(DimensionalityMixin, MathematicalMixin, ...):
    def __init__(self, ...):
        DimensionalityMixin.__init__(self)
        ...

# AFTER
class SwarmVariable(UnitAwareMixin, MathematicalMixin, ...):
    def __init__(self, ...):
        UnitAwareMixin.__init__(self)
        ...
```

**_MeshVariable** (`discretisation_mesh_variables.py`):
```python
# BEFORE
class _MeshVariable(DimensionalityMixin, MathematicalMixin, ...):
    pass

# AFTER
class _MeshVariable(UnitAwareMixin, MathematicalMixin, ...):
    pass
```

### Step 3: Update Classes Using Both Mixins

**UWQuantity** (`function/quantities.py`):
```python
# BEFORE
class UWQuantity(DimensionalityMixin, UnitAwareMixin):
    pass

# AFTER
class UWQuantity(UnitAwareMixin):
    pass  # UnitAwareMixin has everything now!
```

**EnhancedMeshVariable** (`discretisation/persistence.py`):
```python
# BEFORE
class EnhancedMeshVariable(DimensionalityMixin, UnitAwareMixin, MathematicalMixin):
    pass

# AFTER
class EnhancedMeshVariable(UnitAwareMixin, MathematicalMixin):
    pass
```

### Step 4: Delete DimensionalityMixin

**File:** `src/underworld3/utilities/dimensionality_mixin.py`

**Action:** Delete entire file (functionality absorbed into UnitAwareMixin)

### Step 5: Update All Imports

```bash
# Find all imports
grep -r "from.*dimensionality_mixin import" src/

# Replace with
from underworld3.utilities.units_mixin import UnitAwareMixin
```

---

## Why This Is Better

### 1. Consistent Terminology ✅
**Everything is "units":**
- `UnitAwareArray`
- `UnitAwareExpression`
- `UnitAwareMixin`
- `.units` property
- `get_units()` function
- Units tests

### 2. User-Friendly ✅
**Users understand "units":**
- "Set the units to meters"
- "Convert units to kilometers"
- "The units don't match"

**Users don't understand "dimensionality":**
- "Set the dimensionality to [length]" ❌
- What does that even mean?

### 3. API Clarity ✅
```python
# Clear and intuitive
var.units = "m"
var.to("km")
var.has_units

# Not clear
var.dimensionality = ???
var.has_dimensionality
```

### 4. Documentation Consistency ✅
**All docs say "units":**
- UNITS_REFACTOR_PLAN.md
- UNITS_TEST_RESULTS_BASELINE.md
- test_07*_units*.py
- Unit-aware arrays, expressions, quantities

### 5. Less Cognitive Load ✅
**One concept to learn:** Units
- Units can be concrete ("m", "kg/s")
- Units can be abstract ([length], [mass]/[time])
- Units can be scaled (non-dimensionalization)
- **All under "units" umbrella**

---

## Counter-Arguments Addressed

### "But dimensionality is the correct physics term!"

**Response:** Yes, but it's an implementation detail.

- **Under the hood:** We use Pint's dimensionality system
- **User-facing:** We expose "units" API
- **Property:** `.dimensionality` still exists for advanced users

```python
# User-facing (simple)
var.units = "m/s"

# Advanced/internal (complex)
var.dimensionality  # {'[length]': 1, '[time]': -1}
```

### "Dimensionality is more general!"

**Response:** Units are already general.

- Units can be compound: "kg·m/s²"
- Units can be derived: "Pa", "N", "J"
- Units have dimensionality: "Pa" → [mass]/([length]·[time]²)
- **Units subsume dimensionality**

### "Non-dimensionalization is about dimensions!"

**Response:** Non-dimensionalization is about **scaled units**.

```python
# The scaling coefficient HAS UNITS
var.scaling_coefficient = UWQuantity(1000, "m")

# Non-dimensional means "divided by reference units"
nd_value = var.array / var.scaling_coefficient  # Still unit arithmetic!
```

---

## Estimated Effort

### Merge DimensionalityMixin INTO UnitAwareMixin

**Phase 1: Enhance UnitAwareMixin**
- Add scaling attributes (3 attributes)
- Add scaling methods (3 methods)
- Keep all existing functionality
- **Time:** 2-3 hours

**Phase 2: Migrate Classes**
- Update 4 production classes
- Simple import/inheritance changes
- **Time:** 2 hours

**Phase 3: Remove DimensionalityMixin**
- Delete dimensionality_mixin.py
- Update all imports (6-8 files)
- **Time:** 1 hour

**Phase 4: Testing**
- Run full test suite
- Verify closure tests (24/30)
- Verify units tests (85/85)
- **Time:** 1-2 hours

**Total:** 6-8 hours

**Comparison to other direction:** Same effort, but better result!

---

## Success Criteria

### After Completion:
- ✅ Single mixin: `UnitAwareMixin`
- ✅ Consistent terminology: "units" everywhere
- ✅ All scaling functionality available
- ✅ `.dimensionality` still exists (for advanced users)
- ✅ All tests passing (24/30 closure, 85/85 units)
- ✅ Clearer user-facing API

---

## Recommendation: Do This!

### Why:
1. **User-facing names matter** - "units" is what users say
2. **Investment protection** - keeps all your "units" terminology
3. **Same effort** - 6-8 hours either direction
4. **Better result** - clearer API, consistent naming

### When:
- **Not now** - finish Phase 2 (complete ✅)
- **After Phase 3** - bug investigation
- **Part of Phase 4** - refactoring phase

### How:
1. Enhance `UnitAwareMixin` with scaling functionality
2. Migrate classes from `DimensionalityMixin` to `UnitAwareMixin`
3. Delete `dimensionality_mixin.py`
4. Remove deprecation warning from `units_mixin.py`

**This is the right call!** Keep your investment in "units" terminology.

