# Deprecated Methods Usage Audit - Phase 2
**Date:** 2025-01-07
**Purpose:** Identify all usages of `.to_units()` and `.to_nd()` before removal

## Summary

**`.to_units()` Method:**
- **Definition Locations:** 3 files (utilities/unit_aware_array.py, utilities/units_mixin.py, function/quantities.py)
- **Usage in Source:** 3 files (kdtree.py, ckdtree.pyx, enhanced_variables.py - documentation only)
- **Usage in Tests:** 2 files (test_0802_unit_aware_arrays.py, test_0630_mesh_units_demonstration.py)
- **Decision:** Replace all usages with `.to()` (Pint-compatible API)

**`.to_nd()` Method:**
- **Definition Location:** 1 file (utilities/dimensionality_mixin.py)
- **Usage in Source:** 0 files (no actual usage!)
- **Usage in Tests:** 1 file (test_0814_dimensionality_nondimensional.py - entire test file for this feature)
- **Decision:** Remove method and test file (broken symbolic approach)

---

## `.to_units()` Usage Details

### Definitions (3 locations)

1. **src/underworld3/utilities/unit_aware_array.py**
   - Has both `.to()` and `.to_units()` methods
   - `.to_units()` is an alias that calls `.to()`
   - **Action:** Remove `.to_units()` method

2. **src/underworld3/utilities/units_mixin.py**
   - Provides `.to_units()` for classes that mix it in
   - **Action:** Remove `.to_units()` method

3. **src/underworld3/function/quantities.py** (UWQuantity class)
   - Has `.to_units()` method
   - **Action:** Remove `.to_units()` method (keep only `.to()`)

### Source Code Usages (3 locations - need updates)

1. **src/underworld3/kdtree.py:113**
   ```python
   if hasattr(coords, "to_units"):
       coords_converted = coords.to_units(self.coord_units)
   ```
   - **Context:** KDTree coordinate unit conversion
   - **Action:** Change `to_units` → `to`

2. **src/underworld3/ckdtree.pyx:138**
   ```python
   if hasattr(coords, 'to_units'):
       coords_converted = coords.to_units(self.coord_units)
   ```
   - **Context:** Cython KDTree coordinate unit conversion
   - **Action:** Change `to_units` → `to`

3. **src/underworld3/discretisation/enhanced_variables.py:61**
   ```python
   density_gcc = density.to_units("g/cm^3")         # Unit conversion
   ```
   - **Context:** DOCUMENTATION COMMENT in example docstring
   - **Action:** Change `to_units` → `to` in documentation

### Test Usages (2 files - need updates)

1. **tests/test_0802_unit_aware_arrays.py:65**
   ```python
   length_mm = length_m.to_units("mm")
   ```
   - **Context:** Testing UnitAwareArray unit conversion
   - **Action:** Change `to_units` → `to`

2. **tests/test_0630_mesh_units_demonstration.py** (3 usages)
   - Line 117: `survey_mesh_km = survey_mesh.to_units("km")`
   - Line 181: `local_in_regional = regional_mesh.to_units("km")`
   - Line 182: `detail_in_meters = detail_mesh.to_units("m")`
   - **Context:** Demonstration of mesh unit conversion
   - **Action:** Change all `to_units` → `to`

---

## `.to_nd()` Usage Details

### Definition (1 location)

1. **src/underworld3/utilities/dimensionality_mixin.py** (lines 101-153)
   - Implements symbolic non-dimensionalization
   - **Problem:** Doesn't respect variable data operations (symbolic only)
   - **Action:** Remove entire method

### Source Code Usages

**None!** The method is defined but never used in actual source code.

### Test Usages (1 file - will be deleted)

1. **tests/test_0814_dimensionality_nondimensional.py** (entire file tests this method)
   - 4 test functions all use `.to_nd()`
   - Lines 89, 118, 145, 175, 176
   - **Action:** Delete entire test file (tests broken functionality)

### Documentation Usages (2 files - no action needed)

- `docs/examples/Dimensionality-Demo.py` - Demo file showing the concept
- Various planning documents mention `.to_nd()` in design discussions
- **Action:** No changes needed (historical documentation)

---

## Impact Analysis

### Breaking Changes

**None for end users!**
- `.to_units()` → `.to()` is just an API rename
- `.to_nd()` was never actually used in real code
- Only test files need updates

### Files to Modify

**Source Files (5):**
1. `src/underworld3/utilities/unit_aware_array.py` - Remove `.to_units()` method definition
2. `src/underworld3/utilities/units_mixin.py` - Remove `.to_units()` method definition
3. `src/underworld3/function/quantities.py` - Remove `.to_units()` method definition
4. `src/underworld3/kdtree.py` - Change `.to_units` → `.to` (2 locations: hasattr + call)
5. `src/underworld3/ckdtree.pyx` - Change `.to_units` → `.to` (2 locations)

**Mixin File (1):**
6. `src/underworld3/utilities/dimensionality_mixin.py` - Remove `.to_nd()` method (lines 101-153)

**Test Files (3):**
7. `tests/test_0802_unit_aware_arrays.py` - Change `.to_units` → `.to`
8. `tests/test_0630_mesh_units_demonstration.py` - Change `.to_units` → `.to` (3 locations)
9. `tests/test_0814_dimensionality_nondimensional.py` - **DELETE entire file**

**Documentation (1):**
10. `src/underworld3/discretisation/enhanced_variables.py` - Update docstring example

---

## Migration Plan

### Step 1: Update Source Code Usages
- Fix kdtree.py and ckdtree.pyx to use `.to()`
- Update enhanced_variables.py docstring

### Step 2: Update Test Files
- Fix test_0802_unit_aware_arrays.py
- Fix test_0630_mesh_units_demonstration.py

### Step 3: Remove Method Definitions
- Remove `.to_units()` from unit_aware_array.py
- Remove `.to_units()` from units_mixin.py
- Remove `.to_units()` from quantities.py
- Remove `.to_nd()` from dimensionality_mixin.py

### Step 4: Delete Outdated Test
- Delete test_0814_dimensionality_nondimensional.py

### Step 5: Validate
- Run full test suite to ensure no breakage
- Run units tests specifically (test_07*_units*.py, test_08*_*.py)

---

## Risk Assessment

**RISK: LOW**

**Rationale:**
1. `.to_units()` is just an alias for `.to()` - simple rename
2. `.to_nd()` has zero actual usage in source code
3. Only test files need updates (no user-facing code)
4. All usages are straightforward replacements
5. Tests will catch any issues immediately

**Mitigation:**
- Run comprehensive test suite after changes
- Focus on units tests (test_07*_units*.py, test_08*_*.py)
- Check that KDTree functionality still works (critical for swarm operations)

---

## Expected Test Results After Phase 2

**Before Phase 2:**
- Closure tests: 24/30 passing (80%)
- Units tests: Should all still pass

**After Phase 2:**
- Closure tests: 24/30 passing (80%) - unchanged
- Units tests: Should all still pass
- test_0814_dimensionality_nondimensional.py: DELETED (no longer runs)

**No test breakage expected** - all changes are mechanical replacements.
