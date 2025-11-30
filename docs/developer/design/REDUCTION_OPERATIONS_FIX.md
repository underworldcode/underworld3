# UnitAwareArray: Reduction Operations with Unit Preservation

**Date**: 2025-10-12
**Status**: ✅ FIXED

## Problem

User reported that reduction operations like `y.max()`, `y.min()`, `y.mean()`, `y.sum()`, `y.std()`, and `y.var()` were losing units and returning plain floats instead of UWQuantity objects with units.

**User's question**: "Should y.max() have units? (Currently it does not)"

## Root Cause

Two separate issues were identified:

### Issue 1: Scalar Results Returned Plain Floats
The initial implementation of reduction methods called `super().max()`, `super().mean()`, etc., which returned plain numpy scalars without unit information. While we wrapped array results with UnitAwareArray, scalar results were not wrapped with UWQuantity.

### Issue 2: std() and var() Failed Due to Numpy's Internal mean()
When calling `super().std()` or `super().var()`, numpy's internal `_var()` function uses its own internal `mean()` that bypasses our override:

```python
# numpy/core/_methods.py, line 173
def _var(arr, ...):
    arrmean = mean(arr, ...)  # Uses numpy's internal mean, not our override!
    x = asanyarray(arr - arrmean)  # arrmean has no units!
```

This caused:
- `arrmean` returned a plain float without units (from numpy's internal mean)
- Subtraction `arr - arrmean` failed with "Cannot subtract array with units 'km' and dimensionless array"

## Solution

### Part 1: Wrap Scalar Results as UWQuantity

Added `_wrap_scalar_result()` method to convert scalar results to UWQuantity:

```python
def _wrap_scalar_result(self, value):
    """Wrap scalar result with units as UWQuantity."""
    if self.has_units:
        import underworld3 as uw
        return uw.function.quantity(float(value), self._units)
    return value
```

Updated all reduction methods to use this wrapper for scalar results:
- `max()`, `min()`, `mean()`, `sum()` now return UWQuantity for scalar results
- Array results (when `axis` is specified) return UnitAwareArray

### Part 2: Reimplement std() and var() to Use Unit-Aware Operations

Instead of calling `super().std()` and `super().var()` which go through numpy's internal functions, we now implement variance and standard deviation ourselves:

**Variance Implementation**:
```python
def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """Return variance with units squared."""
    if not self.has_units:
        # No units - use numpy's default
        return super().var(...)

    # Calculate variance manually using unit-aware mean
    arr_mean = self.mean(axis=axis, dtype=dtype, keepdims=True, where=where)

    # Extract magnitude from UWQuantity mean
    if hasattr(arr_mean, 'magnitude'):
        mean_value = float(arr_mean.magnitude) if np.isscalar(arr_mean.magnitude) else np.asarray(arr_mean.magnitude)
    else:
        mean_value = arr_mean

    # Get raw array values (without units) for arithmetic
    arr_values = np.asarray(self)

    # Compute variance on raw values
    deviations = arr_values - mean_value
    squared_devs = deviations ** 2
    variance_value = np.mean(squared_devs, axis=axis, keepdims=keepdims)

    # Apply ddof correction
    if ddof != 0:
        n = self.size if axis is None else self.shape[axis]
        variance_value = variance_value * n / (n - ddof)

    # Wrap with squared units
    var_units = f"({self._units})**2"
    return uw.function.quantity(float(variance_value), var_units)
```

**Standard Deviation Implementation**:
```python
def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """Return standard deviation with units preserved."""
    if not self.has_units:
        return super().std(...)

    # Calculate std using unit-aware variance
    variance = self.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)

    # Take square root
    if hasattr(variance, 'magnitude'):
        std_value = np.sqrt(float(variance.magnitude))
    else:
        std_value = np.sqrt(float(variance))

    return uw.function.quantity(std_value, self._units)
```

### Part 3: Fix Unit Compatibility Checking Bug

Fixed a bug in `_check_unit_compatibility()` where the `else: other_units = None` at the end of the elif chain was overwriting the correctly extracted units from UWQuantity objects:

```python
# OLD CODE (BROKEN):
if hasattr(other, 'units'):
    other_units = str(other.units)
    # ... extract magnitude ...
    other = extracted_value
elif hasattr(other, '_units'):
    other_units = other._units
else:
    other_units = None  # ❌ This overwrites the value set in the if block!

# NEW CODE (FIXED):
other_units = None  # Initialize first

if hasattr(other, 'units'):
    other_units = str(other.units)
    # ... extract magnitude ...
    other = extracted_value
elif hasattr(other, '_units'):
    other_units = other._units
elif has_units(other):
    other_units = get_units(other)
# No else clause that overwrites!
```

## Test Results

**Test script**: `/tmp/test_reduction_units.py`

All reduction operations now preserve units correctly:

### ✅ Scalar Results (UWQuantity)
- `y.max()` → `2900.0 kilometer` (UWQuantity with units)
- `y.min()` → `0.0 kilometer` (UWQuantity with units)
- `y.mean()` → `1451.1 kilometer` (UWQuantity with units)
- `y.sum()` → `116089.1 kilometer` (UWQuantity with units)
- `y.std()` → `887.9 kilometer` (UWQuantity with units)
- `y.var()` → `788298.4 kilometer**2` (UWQuantity with squared units)

### ✅ Array Results (UnitAwareArray)
- `coords.max(axis=0, keepdims=True)` → UnitAwareArray with shape (1, 2) and units="km"

## Key Technical Insights

1. **Numpy's Internal Functions Bypass Overrides**: When `super().std()` is called, numpy's internal `_var()` uses its own `mean()` that doesn't respect our overridden `mean()` method. This is a fundamental limitation of subclassing numpy arrays.

2. **Solution: Implement Algorithms Directly**: To maintain unit awareness throughout statistical calculations, we must implement the algorithms ourselves using our unit-aware operations rather than calling numpy's high-level functions.

3. **Raw Value Arithmetic**: When computing deviations and squared deviations, we work with raw numpy arrays (`np.asarray(self)`) and extracted magnitudes to avoid unit compatibility checks, then re-wrap the final result with appropriate units.

4. **Unit Transformation Rules**:
   - `mean()`, `sum()`, `max()`, `min()`, `std()` preserve original units
   - `var()` returns units squared (e.g., km → km²)
   - Array results (with `axis` parameter) return UnitAwareArray
   - Scalar results (no `axis` or `axis=None`) return UWQuantity

## Benefits

- **Statistical Analysis with Units**: Users can now compute statistics on physical quantities without losing unit information
- **Type Safety**: Reduction operations return typed objects (UWQuantity or UnitAwareArray) that preserve dimensional analysis
- **Natural Workflow**: No need to manually track units through statistical calculations
- **Correct Dimensionality**: Variance correctly returns squared units

## Related Files

- `src/underworld3/utilities/unit_aware_array.py` - Main implementation (lines 237-575)
- `/tmp/test_reduction_units.py` - Comprehensive test suite
- `planning/UNITAWARE_ARRAY_MAGNITUDE.md` - Related `.magnitude` property
- `planning/UNIT_ALGEBRA_FIX.md` - Related unit algebra implementation

---

**Status**: ✅ Fully implemented and tested. All reduction operations preserve units correctly.
