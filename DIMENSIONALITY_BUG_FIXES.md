# Dimensionality Bug Fixes - Session 2025-11-08

**Date:** 2025-11-08
**Status:** ✅ FIXED
**Issues Fixed:** 2 critical bugs + code quality improvements

---

## Summary

Fixed two critical bugs related to units and dimensionality:

1. **Power units bug**: `T**2` returned `'kelvin'` instead of `'kelvin ** 2'`
2. **Dimensionality parameter bug**: `dimensionalise()` passed invalid `dimensionality=` parameter to `UnitAwareArray()`

Also improved code quality by **removing silent exception catching** for internal imports.

---

## Bug 1: Power Units (T**2 → 'kelvin' issue)

### The Problem

Power operations on unit-aware expressions returned incorrect units:
```python
T = MeshVariable("T", mesh, 1, units="kelvin")
result = T.sym ** 2

# BEFORE: uw.get_units(result) → 'kelvin' ❌
# AFTER:  uw.get_units(result) → 'kelvin ** 2' ✓
```

### Root Cause

**File:** `src/underworld3/units.py`
**Location:** `_extract_units_info()` function (lines 98-110)

Attempted to import **non-existent function**:
```python
# BROKEN - this function doesn't exist!
from underworld3.function.unit_conversion import get_units as function_get_units
```

The import failed silently (`except Exception: pass`), causing fallback to simpler logic that couldn't handle compound expressions.

### The Fix

Changed import to use the **correct function** that already existed:

```python
# BEFORE (broken):
try:
    from underworld3.function.unit_conversion import get_units as function_get_units
    units_result = function_get_units(obj)
    if units_result is not None:
        backend = _get_default_backend()
        return True, units_result, backend
except Exception:
    pass  # Fall through - SILENT FAILURE!

# AFTER (fixed):
from underworld3.function.unit_conversion import compute_expression_units

units_result = compute_expression_units(obj)
if units_result is not None:
    backend = _get_default_backend()
    # compute_expression_units returns pint.Unit objects, convert to string
    units_str = str(units_result)
    return True, units_str, backend
```

**Key changes:**
1. Import `compute_expression_units()` (which exists and works correctly)
2. Convert `pint.Unit` result to string for consistency
3. **Removed silent exception catching** - internal imports should fail loudly!

---

## Bug 2: Invalid `dimensionality=` Parameter

### The Problem

**Error encountered:**
```python
TypeError: NDArray_With_Callback.__new__() got an unexpected keyword argument 'dimensionality'
```

**Where:** When calling `uw.dimensionalise()` on arrays

### Root Cause

**File:** `src/underworld3/units.py`
**Function:** `dimensionalise()` (lines 705-720)

The function tried to pass `dimensionality=` parameter to `UnitAwareArray()`, but that parameter doesn't exist in `UnitAwareArray.__new__()`:

```python
# BEFORE (broken):
return UnitAwareArray(
    result_qty.magnitude,
    units=str(result_qty.units),
    dimensionality=dimensionality  # ❌ Parameter doesn't exist!
)
```

**UnitAwareArray signature:**
```python
def __new__(cls, input_array=None, units=None, owner=None,
            callback=None, unit_checking=True, auto_convert=True, **kwargs):
    # No dimensionality parameter!
```

### The Fix

Removed invalid `dimensionality=` parameter from two locations:

**Location 1:** Line 705-709 (UnitAwareArray from expression)
```python
# BEFORE (broken):
elif isinstance(expression, UnitAwareArray):
    result_qty = expression.view(np.ndarray) * scale
    return UnitAwareArray(
        result_qty.magnitude,
        units=str(result_qty.units),
        dimensionality=dimensionality  # ❌ INVALID!
    )

# AFTER (fixed):
elif isinstance(expression, UnitAwareArray):
    result_qty = expression.view(np.ndarray) * scale
    # NOTE: UnitAwareArray doesn't store dimensionality, only units
    return UnitAwareArray(
        result_qty.magnitude,
        units=str(result_qty.units)  # ✓ Valid parameters only
    )
```

**Location 2:** Line 716-720 (UnitAwareArray from np.ndarray)
```python
# BEFORE (broken):
if isinstance(expression, np.ndarray):
    return UnitAwareArray(
        result_qty.magnitude,
        units=str(result_qty.units),
        dimensionality=dimensionality  # ❌ INVALID!
    )

# AFTER (fixed):
if isinstance(expression, np.ndarray):
    # NOTE: UnitAwareArray doesn't store dimensionality, only units
    return UnitAwareArray(
        result_qty.magnitude,
        units=str(result_qty.units)  # ✓ Valid parameters only
    )
```

**Note:** `UWQuantity` still accepts and stores `dimensionality=` - this is correct and unchanged.

---

## Code Quality Improvement: No Silent Failures

### The Problem

**Silent exception catching** was hiding bugs:
```python
try:
    from underworld3.some_module import some_function  # Internal import!
    result = some_function(obj)
    return result
except Exception:
    pass  # DANGEROUS - hides bugs in internal code!
```

**Why this is bad:**
1. **Masks import errors** - Non-existent functions fail silently
2. **Hides API changes** - Broken internal APIs go unnoticed
3. **Leads to fallback code** - Replacement logic that shouldn't exist
4. **Makes debugging impossible** - No error message, just wrong behavior

**User feedback:**
> "I am not keen on exceptions being caught for internal important and other cases which are essentially bugs - it would be different importing an external, optional package but this is a situation where failing imports leads you to write replacement code and other problematic pathways."

### The Fix

**Removed silent exception catching** for internal imports in `_extract_units_info()`:

```python
# BEFORE (hiding bugs):
try:
    from underworld3.function.unit_conversion import compute_expression_units
    units_result = compute_expression_units(obj)
    if units_result is not None:
        backend = _get_default_backend()
        units_str = str(units_result)
        return True, units_str, backend
except Exception:
    pass  # SILENT FAILURE - hides bugs!

# AFTER (fail loud):
from underworld3.function.unit_conversion import compute_expression_units

units_result = compute_expression_units(obj)
if units_result is not None:
    backend = _get_default_backend()
    units_str = str(units_result)
    return True, units_str, backend
# If import fails → ImportError (as it should!)
# If function fails → exception propagates (as it should!)
```

### When Silent Catching IS Appropriate

**External optional packages:**
```python
# GOOD - optional external dependency
try:
    import pyvista as pv
    has_pyvista = True
except ImportError:
    has_pyvista = False  # Expected behavior for optional package
```

**Internal Underworld3 modules:**
```python
# BAD - internal module should always be there!
try:
    from underworld3.function import evaluate
except ImportError:
    pass  # ❌ This hides bugs!

# GOOD - let it fail if module is broken
from underworld3.function import evaluate  # ✓ Fails loudly if broken
```

---

## Files Changed

### Modified

**`src/underworld3/units.py`:**
1. Lines 98-110: Fixed power units bug (changed import)
2. Lines 98-110: Removed silent exception catching
3. Lines 705-709: Removed invalid `dimensionality=` parameter (UnitAwareArray case)
4. Lines 716-720: Removed invalid `dimensionality=` parameter (np.ndarray case)

**Total changes:** ~20 lines modified

---

## Testing

### Tests Passing

✅ **Power units test:**
```bash
pixi run -e default pytest tests/test_0850_units_closure_comprehensive.py::test_units_temperature_squared -v
# PASSED
```

✅ **Dimensionality no longer causes TypeError** (bug that user reported)

### What Now Works

1. **Power operations:**
   - `T**2` → `'kelvin ** 2'` ✓
   - `velocity**2` → `'meter ** 2 / second ** 2'` ✓
   - `x**0.5` → `'meter ** 0.5'` ✓

2. **Dimensionalization:**
   - `dimensionalise()` works with arrays ✓
   - No more `TypeError` about unknown `dimensionality=` parameter ✓

3. **Code quality:**
   - Internal import failures now **fail loud** ✓
   - Bugs no longer hidden by silent exception catching ✓

---

## Impact

### Physics Calculations

**Now correct:**
- Energy calculations: `E = m * v**2` (needs `v**2` units)
- Strain rate squared
- Thermal diffusivity (length² / time)
- All compound unit expressions

### Code Maintainability

**Improved:**
- Internal bugs surface immediately (no silent failures)
- API changes cause clear errors (not mysterious behavior)
- Debugging is possible (exceptions aren't swallowed)

---

## Lessons Learned

1. **Silent exception catching for internal imports is a bug magnet**
   - Only catch for external optional packages
   - Internal code should fail loudly

2. **Test API contracts**
   - `UnitAwareArray` signature should have been checked before passing `dimensionality=`
   - Would have caught the error immediately

3. **Existing code often has the solution**
   - `compute_expression_units()` always worked perfectly
   - Problem was just the wrong import

4. **Type hints would help**
   - `dimensionality=` parameter would have been caught by type checker
   - Consider adding type hints to constructors

---

## Next Steps

**Potential improvements:**
1. Audit other locations for silent exception catching
2. Add type hints to `UnitAwareArray.__new__()`
3. Add type hints to `dimensionalise()` and `non_dimensionalise()`
4. Consider adding unit tests for invalid parameters (should raise TypeError)

---

**Status:** Both bugs fixed, code quality improved, tests passing ✅
