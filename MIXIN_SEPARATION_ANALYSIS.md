# Why Are DimensionalityMixin and UnitAwareMixin Separate?

**Date:** 2025-01-07
**Question:** Should these be merged?
**Answer:** **YES** - but carefully, as part of Phase 4 refactoring

---

## Current Situation

### DimensionalityMixin (`utilities/dimensionality_mixin.py`)

**Purpose:** Non-dimensionalization and scaling coefficients

**Key Features:**
- `_scaling_coefficient`: Reference scale for non-dimensionalization
- `_is_nondimensional`: State tracking (dimensional ↔ non-dimensional)
- `_original_units`: Backup units before non-dimensionalization
- `.nd_array`: Get non-dimensional array values
- `.from_nd()`: Convert from non-dimensional
- `.set_reference_scale()`: Set scaling coefficient
- ~~`.to_nd()`~~: **REMOVED Phase 2** (broken implementation)

**Focus:** SCALING and STATE - works with float coefficients

### UnitAwareMixin (`utilities/units_mixin.py`)

**Status:** ⚠️ **DEPRECATED** (warning on import)
**Purpose:** Units tracking with Pint backend

**Key Features:**
- `_units`: Units string (e.g., "m", "m/s")
- `_units_backend`: Pint backend for conversions
- `_dimensional_quantity`: Cached Pint quantity
- `.set_units()`: Set units with backend
- `.create_quantity()`: Make Pint quantity
- `.non_dimensional_value()`: Scale to non-dimensional
- `.dimensional_value()`: Scale from non-dimensional
- `.check_units_compatibility()`: Check if units compatible
- ~~`.to_units()`~~: **REMOVED Phase 2** (alias for `.to()`)

**Focus:** UNITS and CONVERSION - works with Pint quantities

**Documentation says:**
> "DEPRECATED: This module contains experimental code that was abandoned in favor of the hierarchical units system (enhanced_variables.py)"

---

## Overlap Analysis

### Both Provide:
1. ✅ `.dimensionality` property
2. ✅ Non-dimensionalization functionality
3. ✅ Units tracking (sort of)

### Key Differences:

| Feature | DimensionalityMixin | UnitAwareMixin |
|---------|-------------------|----------------|
| **Primary Purpose** | Scaling & state management | Units & Pint integration |
| **Works With** | Float coefficients | Pint quantities |
| **State Tracking** | ✅ Yes (_is_nondimensional) | ❌ No |
| **Pint Integration** | ⚠️ Indirect (via units string) | ✅ Direct (backend) |
| **Non-Dimensional** | Stores coefficient | Computes on-the-fly |
| **Status** | ✅ Active | ⚠️ Deprecated |

---

## Current Usage

### Classes Using BOTH Mixins:
1. **`UWQuantity`** (`function/quantities.py`)
   ```python
   class UWQuantity(DimensionalityMixin, UnitAwareMixin):
   ```
   - Uses UnitAwareMixin for Pint integration
   - Uses DimensionalityMixin for scaling coefficients

2. **`EnhancedMeshVariable`** (`discretisation/persistence.py`)
   ```python
   class EnhancedMeshVariable(DimensionalityMixin, UnitAwareMixin, MathematicalMixin):
   ```
   - Legacy experimental class (not used in production)

### Classes Using DimensionalityMixin ONLY:
1. **`SwarmVariable`** (`swarm.py`)
   - Production class
   - Uses DimensionalityMixin for non-dimensional scaling
   - Gets units from `get_units()` function, not mixin

2. **`_MeshVariable`** (`discretisation_mesh_variables.py`)
   - Production class
   - Same pattern as SwarmVariable

### Classes Using UnitAwareMixin ONLY:
1. **`EnhancedSwarmVariable`** (`discretisation/enhanced_variables.py`)
   - Experimental class
   - Part of abandoned hierarchical system

---

## The Problem: Historical Fragmentation

### What Happened:

1. **Original Design** (2023?):
   - `UnitAwareMixin` created to add Pint units to any class
   - Seemed like a good idea: "mixin for orthogonal functionality"

2. **Non-Dimensional Needs** (2024?):
   - Physics simulations need non-dimensional scaling
   - Created `DimensionalityMixin` separately
   - **Why separate?** Probably thought "non-dimensionalization is different from units"

3. **Reality Check**:
   - Non-dimensionalization **IS** a units operation!
   - Scaling coefficients **HAVE** units!
   - The two are **NOT** orthogonal

4. **Current State** (2025):
   - `UnitAwareMixin` officially deprecated
   - `DimensionalityMixin` still used in production
   - Both provide overlapping functionality
   - **Confusion**: Why two mixins for related concepts?

---

## Should They Be Merged?

### Short Answer: **YES**, but...

### The Right Approach:

**Don't merge the mixins** → **Replace them both** with the hierarchical system

**Rationale:**
1. **UnitAwareMixin already deprecated**: Don't invest in it
2. **DimensionalityMixin has good ideas**: Scaling coefficients, state tracking
3. **Mixins are the wrong pattern**: Not orthogonal, cause MRO complexity
4. **Better solution exists**: `enhanced_variables.py` hierarchical system

---

## Recommended Architecture (Phase 4)

### Instead of Mixins, Use Composition:

```python
# CURRENT (messy)
class _MeshVariable(DimensionalityMixin, MathematicalMixin, Stateful, uw_object):
    # Multiple inheritance hell
    pass

# PROPOSED (clean)
class _MeshVariable(MathematicalMixin, Stateful, uw_object):
    def __init__(self, ...):
        self._units_handler = UnitsHandler(self)  # Composition!
        self._scaling_handler = ScalingHandler(self)  # Composition!
```

### Single Unified Handler:

```python
class UnitsAndScalingHandler:
    """
    Single class handling both units and non-dimensionalization.

    Replaces both UnitAwareMixin and DimensionalityMixin.
    """
    def __init__(self, owner):
        # Units (from UnitAwareMixin)
        self._units = None
        self._pint_backend = PintBackend()

        # Scaling (from DimensionalityMixin)
        self._scaling_coefficient = 1.0
        self._is_nondimensional = False
        self._original_units = None

    # Units methods
    def to(self, target_units):
        """Convert to different units"""
        ...

    # Scaling methods
    def to_nd(self):
        """Convert to non-dimensional using scaling coefficient"""
        ...

    def from_nd(self, nd_value):
        """Convert from non-dimensional"""
        ...

    # Unified interface
    @property
    def dimensionality(self):
        """Works for both dimensional and non-dimensional"""
        if self._is_nondimensional:
            return self._original_dimensionality
        return self._pint_backend.get_dimensionality(self._units)
```

---

## What's Wrong with the Current Separation?

### 1. Conceptual Confusion
**Problem:** Users ask "Why two mixins for units?"
**Root Cause:** Non-dimensionalization **IS** a units operation
**Example:** Scaling coefficient of `1000 km` has units! It's not separate from the units system.

### 2. Redundant Functionality
**Both provide:**
- `.dimensionality` property (different implementations!)
- Non-dimensionalization (different approaches!)
- Units tracking (one direct, one indirect!)

**Result:** Code doing the same thing two different ways

### 3. Multiple Inheritance Complexity
**Example:**
```python
class UWQuantity(DimensionalityMixin, UnitAwareMixin):
```

**Problems:**
- MRO (Method Resolution Order) complexity
- `super().__init__()` chain fragility
- Which `.dimensionality` gets called?
- Debugging nightmare

### 4. State Synchronization
**DimensionalityMixin** tracks state:
- `_is_nondimensional`
- `_original_units`
- `_scaling_coefficient`

**UnitAwareMixin** tracks state:
- `_units`
- `_dimensional_quantity`
- `_scale_factor`

**Problem:** What if they get out of sync?

---

## Migration Path (Phase 4)

### Step 1: Create Unified Handler
```python
# New file: utilities/units_and_scaling.py
class UnitsAndScalingHandler:
    """Unified units and scaling management"""
    pass
```

### Step 2: Migrate Variables
```python
# OLD
class _MeshVariable(DimensionalityMixin, ...):
    pass

# NEW
class _MeshVariable(...):
    def __init__(self):
        self._units_handler = UnitsAndScalingHandler(self)
```

### Step 3: Update All Usage
- Replace `self.dimensionality` → `self._units_handler.dimensionality`
- Replace `self.scaling_coefficient` → `self._units_handler.scaling_coefficient`
- Replace `self.to_nd()` → `self._units_handler.to_nd()`

### Step 4: Remove Old Mixins
- Delete `UnitAwareMixin` (already deprecated)
- Delete `DimensionalityMixin` (after migration)

---

## Benefits of Merging (via Unified Handler)

### 1. Clarity ✅
One place for all units and scaling logic

### 2. Consistency ✅
Single implementation of `.dimensionality`, no conflicts

### 3. Simplicity ✅
No multiple inheritance, clearer `__init__` chains

### 4. Correctness ✅
Scaling coefficients have proper units tracking

### 5. Maintainability ✅
One class to test, one class to document

---

## Immediate Answer to Your Question

**Q:** "Why is there a dimensionality mixin separate from units?"

**A:** **Historical accident + conceptual confusion**

1. **Initially**: Seemed like two separate concerns (units vs scaling)
2. **Reality**: They're the same concern (scaling **HAS** units!)
3. **Current**: `UnitAwareMixin` deprecated, `DimensionalityMixin` still used
4. **Future**: Both should be replaced by unified handler (Phase 4)

**Recommendation:** Don't merge the mixins directly. Instead, create a unified `UnitsAndScalingHandler` and migrate to composition pattern.

---

## What About `UnitAwareArray`?

**UnitAwareArray is different!**

It's not a mixin - it's a concrete class that:
- Extends `numpy.ndarray` directly
- Has its own complete implementation
- Doesn't suffer from mixin complexity

**UnitAwareArray should NOT be merged with anything.**

It's actually the best example of the right pattern:
- Single class
- Clear purpose
- Self-contained
- No mixin hell

---

## Timeline Recommendation

### Don't Do This Now (Phase 2)
Phase 2 was about removing deprecated methods - completed ✅

### Do This in Phase 4
Phase 4 is "Complete UnitAwareExpression Integration"

**Expand Phase 4 scope to include:**
1. Create `UnitsAndScalingHandler`
2. Migrate variables to composition pattern
3. Remove both mixins
4. Ensure 100% closure property

**Estimated:** 3-4 weeks (already estimated 2-3 weeks, add 1 week for mixin removal)

---

## Conclusion

**Your intuition is correct!** These should not be separate.

**The separation happened because:**
1. Non-dimensionalization was treated as orthogonal to units (it's not)
2. Mixin pattern seemed elegant (it's not for non-orthogonal concerns)
3. Code evolved organically without architectural review

**The fix:**
- Not a simple merge
- Replace both with unified handler using composition
- Part of larger Phase 4 refactoring
- Already on the roadmap, just needs explicit scope

**Benefits:**
- Clearer architecture
- Easier to understand
- Fewer bugs
- Better testing

