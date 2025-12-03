# NumPy Array-to-Scalar Deprecation Fix

**Date**: 2025-01-11
**Status**: ✅ FIXED

## Problem

NumPy 1.25+ deprecated calling `float()` directly on arrays with `ndim > 0`:

```
DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated,
and will error in future. Ensure you extract a single element from your array before
performing this operation. (Deprecated NumPy 1.25.)
```

**Location**: `tests/test_0803_simple_workflow_demo.py:94` (and 3 other places in same file)

## Root Cause

`uw.function.evaluate()` returns numpy arrays even for single-point evaluations (typically shape `(1, 1, 1)` for scalar fields). Calling `float()` directly on these arrays triggers the deprecation warning.

## Solution

Replace `float(array)` with `np.asarray(array).item()` to properly extract the scalar value.

### Before
```python
if hasattr(temp_from_m, '_pint_qty'):
    temp_m_val = temp_from_m._pint_qty.magnitude.item()
else:
    temp_m_val = float(temp_from_m)  # ⚠️ Deprecated for arrays
```

### After
```python
if hasattr(temp_from_m, '_pint_qty'):
    temp_m_val = temp_from_m._pint_qty.magnitude.item()
else:
    temp_m_val = np.asarray(temp_from_m).item()  # ✅ Works for both arrays and scalars
```

## Changes Made

Fixed 7 occurrences in `test_0803_simple_workflow_demo.py`:

**Lines updated**:
- Line 89: `temp_km_val = np.asarray(temp_from_km).item()`
- Line 94: `temp_m_val = np.asarray(temp_from_m).item()`
- Line 99: `temp_model_val = np.asarray(temp_from_model).item()`
- Line 169: `survey_val = ... else np.asarray(temp_survey).item()`
- Line 170: `drill_val = ... else np.asarray(temp_drill).item()`
- Line 171: `model_val = ... else np.asarray(temp_model).item()`
- Line 192: `temp_um_val = ... else np.asarray(temp_um).item()`

## Why `np.asarray().item()` Works

1. **`np.asarray()`**: Converts input to numpy array (no-op if already array, converts scalar to 0-d array)
2. **`.item()`**: Extracts the single scalar value from array
   - Works for 0-d arrays: `np.array(5).item()` → `5`
   - Works for 1-element arrays: `np.array([[[5]]]).item()` → `5`
   - Raises error for multi-element arrays (which is what we want - fail fast)

## Testing

```bash
# Before fix:
pytest tests/test_0803_simple_workflow_demo.py -v
# → DeprecationWarning at line 94

# After fix:
pytest tests/test_0803_simple_workflow_demo.py -v
# → PASSED with no warnings ✅
```

## Best Practice Going Forward

When extracting scalar values from `uw.function.evaluate()` results:

```python
# ✅ RECOMMENDED: Handle both UWQuantity and plain arrays
if hasattr(result, '_pint_qty'):
    value = result._pint_qty.magnitude.item()  # UWQuantity
else:
    value = np.asarray(result).item()  # Plain array

# ❌ AVOID: Direct float() on potentially-array results
value = float(result)  # Will fail in future NumPy versions
```

## Related Issues

This pattern might exist in example scripts or documentation. When updating those files, use the same fix.

## Why This Matters

- **Future-proofing**: NumPy will make this an error (not just warning) in future versions
- **Clarity**: Explicitly extracting the scalar value is clearer than implicit conversion
- **Correctness**: `.item()` raises an error for multi-element arrays, catching bugs early

## No Other Occurrences Found

Searched the test suite for similar patterns - this was the only file with the issue.

## Summary

**Problem**: Calling `float()` on numpy arrays is deprecated
**Solution**: Use `np.asarray(result).item()` to extract scalar
**Status**: ✅ Fixed - test passes with no warnings
**Impact**: Future-proof, clearer code, prevents errors in NumPy 2.0+
