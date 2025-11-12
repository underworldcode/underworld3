# Lightweight UnitAwareArray (v2) Usage Audit
**Date:** 2025-01-07
**Purpose:** Document all uses before removal/replacement

## Summary

The lightweight `UnitAwareArray` from `function/unit_conversion.py` (lines 12-192) has **VERY LIMITED** usage in the codebase:

- **Only 2 import locations** (both in same file)
- **Only used internally** within `unit_conversion.py` and `ddt.py`
- **NOT used by** coordinates, variables, or any user-facing code
- **Safe to replace** with comprehensive UnitAwareArray from `utilities/unit_aware_array.py`

## All Usages Found

### 1. Internal Use in `unit_conversion.py`

**Location:** `src/underworld3/function/unit_conversion.py`

**Usage 1:** Lines 162-164 (within `non_dimensional_value()` method)
```python
if not dimensionality:
    # Already dimensionless, return as-is
    return UnitAwareArray(np.asarray(self), units="dimensionless", dimensionality={})
```

**Usage 2:** Line 184 (within `non_dimensional_value()` method)
```python
# Return UnitAwareArray with preserved dimensionality
return UnitAwareArray(nondim_values, units="dimensionless", dimensionality=dimensionality)
```

**Purpose:** Returns dimensionless arrays from `non_dimensional_value()` method with preserved dimensionality metadata.

**Replacement Strategy:** These return values are IMMEDIATELY consumed (not stored or passed around), so replacing with comprehensive UnitAwareArray will work fine.

### 2. Time-Stepping System in `ddt.py`

**Location:** `src/underworld3/systems/ddt.py`

**Import:** Lines 738, 768
```python
from underworld3.function.unit_conversion import UnitAwareArray
```

**Usage 1:** Lines 740-749 (Semi-Lagrangian node point calculation)
```python
v_with_units = UnitAwareArray(v_at_node_pts, units=v_units)
# Non-dimensionalize
v_nondim = uw.non_dimensionalise(v_with_units, model)
# Extract numpy array
if isinstance(v_nondim, UnitAwareArray):
    v_at_node_pts = np.array(v_nondim)
elif hasattr(v_nondim, "value"):
    v_at_node_pts = v_nondim.value
else:
    v_at_node_pts = v_nondim
```

**Usage 2:** Lines 770-779 (Semi-Lagrangian mid point calculation)
```python
v_with_units = UnitAwareArray(v_at_mid_pts, units=v_units)
# Non-dimensionalize
v_nondim = uw.non_dimensionalise(v_with_units, model)
# Extract numpy array
if isinstance(v_nondim, UnitAwareArray):
    v_at_mid_pts = np.array(v_nondim)
elif hasattr(v_nondim, "value"):
    v_at_mid_pts = v_nondim.value
else:
    v_at_mid_pts = v_nondim
```

**Purpose:** Temporarily wraps velocity arrays with units for non-dimensionalization, then immediately extracts plain numpy arrays.

**Pattern:**
1. Create lightweight UnitAwareArray
2. Pass to `uw.non_dimensionalise()`
3. Immediately extract plain numpy array
4. Discard the unit-aware wrapper

**Replacement Strategy:** This is a **very temporary** wrapper - units are added just to call `non_dimensionalise()`, then stripped immediately. Comprehensive UnitAwareArray will work perfectly here.

## Why Lightweight Version Was Created

Based on code analysis, the lightweight version was likely created for this specific pattern:
- **Minimal overhead** for temporary wrapping
- **Intentionally loses units** through numpy operations (comment lines 79-84)
- **Just metadata** - no operation preservation

However, this is **NOT a good enough reason** to keep it because:
1. The "temporary wrapper" use case is rare (only 2 locations)
2. The comprehensive version works fine for temporary wrapping too
3. Having two implementations creates confusion and fragility
4. The comprehensive version is already used by coordinates (user-facing)

## Replacement Plan

### Step 1: Update Imports

**File:** `src/underworld3/systems/ddt.py` (lines 738, 768)

**Change:**
```python
# OLD
from underworld3.function.unit_conversion import UnitAwareArray

# NEW
from underworld3.utilities.unit_aware_array import UnitAwareArray
```

### Step 2: Update Internal Returns

**File:** `src/underworld3/function/unit_conversion.py` (lines 164, 184)

**Change:** Update imports at top of file
```python
# Add near top of file (around line 10)
from underworld3.utilities.unit_aware_array import UnitAwareArray
```

No code changes needed - the class API is compatible!

### Step 3: Remove Lightweight Implementation

**File:** `src/underworld3/function/unit_conversion.py`

**Delete:** Lines 12-192 (entire lightweight UnitAwareArray class)

**Note:** Keep only the import at the top (added in Step 2)

### Step 4: Test

Run these tests to ensure nothing breaks:
```bash
# Test time-stepping system
pixi run -e default pytest tests/test_*ddt*.py -v

# Test non-dimensionalization
pixi run -e default pytest tests/test_*units*.py -v

# Test function evaluation
pixi run -e default pytest tests/test_*evaluate*.py -v
```

## Risk Assessment

**Risk Level:** ⚠️ **LOW-MEDIUM**

**Why Low:**
- Only 2 files use it
- Usage is very localized (temporary wrapping pattern)
- Comprehensive UnitAwareArray has compatible API
- Both versions have same basic properties (`.units`, `._units`)

**Why Medium (caution needed):**
- Time-stepping code (`ddt.py`) is critical for correctness
- Non-dimensionalization is unit-system-critical
- Need to verify that comprehensive version's richer functionality doesn't cause issues

**Mitigation:**
- Test semi-Lagrangian time-stepping thoroughly after change
- Verify non-dimensionalization produces identical results
- Check that immediate numpy conversion (lines 745, 775 in ddt.py) still works

## Testing Checklist

After replacement, verify:
- [ ] `ddt.py` imports work
- [ ] `unit_conversion.py` internal returns work
- [ ] Semi-Lagrangian time-stepping produces correct results
- [ ] `uw.non_dimensionalise()` produces identical results
- [ ] No performance regression (comprehensive version might be slightly slower)
- [ ] All units tests still pass

## Conclusion

**Recommendation:** ✅ **SAFE TO REPLACE**

The lightweight UnitAwareArray has minimal, localized usage. Replacing it with the comprehensive version is straightforward and low-risk. The main caution is to test time-stepping thoroughly, but the pattern used (temporary wrap → immediate unwrap) should work identically with the comprehensive version.

**Benefits of Replacement:**
1. Single implementation to maintain
2. Consistent behavior across codebase
3. If usage ever expands, comprehensive features already available
4. Closure property guaranteed everywhere
5. Less confusion for developers

**Next Action:** Execute the 4-step replacement plan above, then run the testing checklist.
