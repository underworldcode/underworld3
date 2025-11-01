# Dimensionality Preservation Implementation - Change Log

**Date**: 2025-10-31
**Purpose**: Track all changes for generalizing `non_dimensionalise()` / `dimensionalise()` functions
**Revert Instructions**: If needed, revert changes in reverse order using the "Before" sections

---

## Change 1: UWQuantity - Add Dimensionality Support

**File**: `src/underworld3/function/quantities.py`
**Status**: NOT YET APPLIED
**Lines Modified**: TBD

### Before:
```python
# No dimensionality tracking
```

### After:
```python
# Added _dimensionality attribute and property
# Constructor accepts dimensionality parameter
# Automatically extracts from Pint quantities
```

**Revert Command**: `git diff src/underworld3/function/quantities.py` to see exact changes

---

## Change 2: UnitAwareArray - Add Dimensionality Support

**File**: `src/underworld3/function/unit_conversion.py`
**Status**: NOT YET APPLIED
**Lines Modified**: TBD

### Before:
```python
# Only tracks _units attribute
```

### After:
```python
# Added _dimensionality attribute
# __new__() accepts dimensionality parameter
# __array_finalize__() preserves dimensionality
```

**Revert Command**: `git diff src/underworld3/function/unit_conversion.py`

---

## Change 3: Model - Add get_scale_for_dimensionality()

**File**: `src/underworld3/model.py`
**Status**: NOT YET APPLIED
**Lines Modified**: TBD

### Before:
```python
# No method to get scales for arbitrary dimensionality
```

### After:
```python
# Added get_scale_for_dimensionality(dimensionality) method
# Computes composite scales from fundamentals
```

**Revert Command**: `git diff src/underworld3/model.py`

---

## Change 4: Generalized non_dimensionalise()

**File**: `src/underworld3/units.py`
**Status**: NOT YET APPLIED
**Lines Modified**: TBD (around line 344-384)

### Before:
```python
def non_dimensionalise(expression, scaling_system: Optional[Dict] = None) -> Any:
    # Only works with .non_dimensional_value() or .data attributes
```

### After:
```python
def non_dimensionalise(expression, model=None) -> Any:
    # Protocol-based: works with UWQuantity, UnitAwareArray, Variables
    # Preserves dimensionality metadata
```

**Revert Command**: `git diff src/underworld3/units.py`

---

## Change 5: New dimensionalise() Function

**File**: `src/underworld3/units.py`
**Status**: NOT YET APPLIED
**Lines Modified**: TBD (new function added)

### Before:
```python
# Function did not exist
```

### After:
```python
def dimensionalise(expression, target_dimensionality=None, model=None) -> Any:
    # Companion function to restore dimensional form
```

**Revert Command**: Delete the function, or `git diff`

---

## Change 6: Export New Functions

**File**: `src/underworld3/__init__.py`
**Status**: NOT YET APPLIED
**Lines Modified**: TBD (around line 173-187)

### Before:
```python
# dimensionalise not exported
```

### After:
```python
from .units import (
    ...,
    dimensionalise,  # NEW
    ...
)
```

**Revert Command**: `git diff src/underworld3/__init__.py`

---

## Testing Log

### Test 1: Basic Dimensionality Preservation
**Status**: PENDING
**Command**: TBD
**Result**: TBD

### Test 2: UnitAwareArray Non-dimensionalization
**Status**: PENDING
**Command**: TBD
**Result**: TBD

### Test 3: get_scale_for_dimensionality()
**Status**: PENDING
**Command**: TBD
**Result**: TBD

### Test 4: Semi-Lagrangian Advection Fix
**Status**: PENDING
**Command**: TBD
**Result**: TBD

---

## Full Revert Procedure

If all changes need to be reverted:

1. **Don't commit yet** - changes are only in working directory
2. **Option A - Git revert**:
   ```bash
   cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
   git checkout src/underworld3/function/quantities.py
   git checkout src/underworld3/function/unit_conversion.py
   git checkout src/underworld3/model.py
   git checkout src/underworld3/units.py
   git checkout src/underworld3/__init__.py
   ```

3. **Option B - Manual revert**:
   - Restore each file from "Before" sections above
   - Rebuild: `pixi run underworld-build`
   - Verify: `pixi run -e default pytest tests/test_0700_units_system.py -v`

4. **Remove this log**: `rm DIMENSIONALITY_CHANGES_LOG.md`

---

## Notes

- All changes preserve backward compatibility
- Existing code using `.non_dimensional_value()` continues to work
- Protocol-based design allows future extensions
- Consistent use of `_dimensionality` across all types

---

**Last Updated**: 2025-10-31 (creation)
