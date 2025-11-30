# Test Reliability System Setup - 2025-11-15

## Summary

Implemented a comprehensive dual-classification system for tests that integrates the existing test levels (complexity) with new reliability tiers (trust level).

## What Was Accomplished

### 1. Fixed Critical JIT Compilation Bug ✅

**Problem**: UWQuantity constants with units (like `uw.quantity(1.0, "Pa*s")`) weren't being unwrapped to numeric values during JIT compilation, causing C compiler errors.

**Fix**:
- Modified `unwrap()` function in `src/underworld3/function/expressions.py` to properly respect `keep_constants=False` parameter
- Added enhanced debugging output in `src/underworld3/utilities/_jitextension.py` to show free symbols and their attributes
- **Result**: test_0818_stokes_nd.py now fully passing (all 5 tests)

**Files Modified**:
- `src/underworld3/function/expressions.py` (lines 277-316)
- `src/underworld3/utilities/_jitextension.py` (lines 414-440)
- `debug_stokes_jit.py` (fixed API misuse)

### 2. Designed Test Reliability Classification System ✅

**Dual Classification**:
1. **Test Levels** (existing, number prefix 0000-9999): Complexity/scope
   - Level 1 (0000-0499): Quick core tests (~2-5 min)
   - Level 2 (0500-0899): Intermediate tests (~5-10 min)
   - Level 3 (1000+): Physics/solver tests (~10-15 min)

2. **Reliability Tiers** (new, pytest markers): Trust level
   - Tier A: Production-ready, trusted for TDD
   - Tier B: Validated, use with caution
   - Tier C: Experimental, development only

**Key Principle**: Orthogonal dimensions - a test can be Level 2 (intermediate complexity) AND Tier A (production-ready) simultaneously.

### 3. Documentation Created ✅

**Core Documents**:
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md` - Complete system specification
- `docs/developer/TEST-CLASSIFICATION-2025-11-15.md` - Current status analysis
- `UNWRAPPING_BUG_FIX_2025-11-15.md` - JIT bug fix documentation
- Updated `CLAUDE.md` - Integrated system overview

**Infrastructure Files**:
- Updated `tests/pytest.ini` - Added tier_a, tier_b, tier_c markers
- Created `.claude/commands/test-tier-a.md` - Slash command for Tier A tests
- Created `.claude/commands/test-tier-ab.md` - Slash command for Tier A+B tests
- Created `.claude/commands/test-units-classify.md` - Slash command for classification

### 4. Integration with Existing Systems ✅

**Pixi Tasks** (in `pixi.toml`):
- `pixi run underworld-test [1|2|3|1,2,3]` - Run by test level
- Compatible with new tier markers

**Test Levels Script** (`scripts/test_levels.sh`):
- Already implements level-based testing
- Can be extended to support tier filtering in future

**Pytest Markers** (`tests/pytest.ini`):
- tier_a, tier_b, tier_c now available
- Usage: `pytest -m tier_a` or `pytest -m "tier_a or tier_b"`

## Current Test Status

### After JIT Unwrapping Fix

**Known Good**:
- ✅ test_0818_stokes_nd.py: All 5 tests PASSING

**Units Tests (test_07*_units*.py, test_08*_*.py)**:
- Total: 259 tests
- Passing: ~180 (before fix, likely more now)
- Failing: ~79 (needs current analysis)

**Categories to Classify**:
1. Comparison operators (test_0810): Feature status unclear
2. Reduction operations (test_0850-0852): Recently documented as passing, investigate breakage
3. Mesh variable ordering (test_0813): Should work per "No Batman" fix
4. Units propagation (test_0850_units_*): Advanced features, possibly incomplete
5. Poisson with units (test_0812): Integration test
6. Coordinate units (test_0815): Recently completed feature

## Next Steps

### Immediate (Today)

1. **Wait for test run to complete** 🔄 (running in background)
2. **Analyze current failures** - Categorize each into:
   - Tier B: Valid test, needs code fix
   - Tier C: Test/feature incomplete, mark xfail

3. **Mark high-confidence Tier A tests** - Start with:
   - test_0000-0499 (Level 1 core tests that pass)
   - test_0700_units_system.py (if passing)
   - test_1000-1050 (Level 3 established solvers that pass)

### Short-term (This Week)

1. **Apply tier markers to all tests**:
   ```python
   # Example for Tier A
   import pytest

   @pytest.mark.tier_a
   def test_basic_mesh_creation():
       \"\"\"Test basic mesh creation.\"\"\"
       ...

   # Example for Tier C with xfail
   @pytest.mark.tier_c
   @pytest.mark.xfail(reason="Comparison operators not fully implemented")
   def test_uwquantity_comparison():
       \"\"\"Test UWQuantity comparison operators.\"\"\"
       ...
   ```

2. **Fix or document each failure**:
   - Either: Fix the code to make test pass
   - Or: Mark test as xfail with clear reason
   - Or: Remove test if fundamentally wrong

3. **Update test_levels.sh** (optional): Add tier filtering support

### Medium-term (Next 2 Weeks)

1. **Promote test_0818_stokes_nd.py to Tier A**:
   - Monitor for consistent passing (1 week)
   - Add `@pytest.mark.tier_a`
   - Document promotion in commit message

2. **Review all regression tests** (test_06*):
   - Validate each test is correct
   - Mark appropriate tier
   - Critical for stability - priority for Tier A

3. **CI Integration**:
   - Set up Tier A as pre-merge CI check
   - Full Tier A+B for nightly builds
   - Document CI expectations

## Usage Examples

### Run Tests by Level (Existing)
```bash
# Quick core tests only
pixi run underworld-test 1

# Intermediate + Physics
pixi run underworld-test 2,3

# All tests
pixi run underworld-test
```

### Run Tests by Tier (New)
```bash
# Only production-ready tests (safe for TDD)
pixi run -e default pytest -m tier_a -v

# Production + validated tests (full validation)
pixi run -e default pytest -m "tier_a or tier_b" -v

# Exclude experimental tests
pixi run -e default pytest -m "not tier_c" -v
```

### Combined Filtering
```bash
# Level 2 tests, Tier A only (trusted intermediate tests)
pixi run -e default pytest tests/test_0[5-8]*py -m tier_a -v

# Level 3 tests, Tier A+B (all physics validation)
pixi run -e default pytest tests/test_1*py -m "tier_a or tier_b" -v
```

## Decision Matrix for Classification

| Condition | Test Level | Reliability Tier |
|-----------|------------|------------------|
| Core import/mesh test, stable, passing | Level 1 (0000-0499) | Tier A |
| Units integration, recently added, passing | Level 2 (0800-0899) | Tier B |
| Advanced units, feature incomplete, failing | Level 2 (0850-0899) | Tier C + xfail |
| Stokes solver, proven, passing | Level 3 (1010-1050) | Tier A |
| New solver variant, works but new | Level 3 (1000+) | Tier B |
| Future feature test, not implemented | Any level | Tier C + xfail |

## Key Principles

1. **Levels = Complexity**: What type of functionality is being tested
2. **Tiers = Trust**: How much we trust the test results
3. **Orthogonal**: A simple test can be experimental (Level 1, Tier C)
4. **Conservative Promotion**: Start at Tier C/B, earn Tier A over time
5. **Clear Communication**: xfail reasons must explain what's missing

## Benefits

1. **Prevents TDD Confusion**: Developers know which tests to trust (Tier A)
2. **Documents Maturity**: Clear progression from experimental to production
3. **Supports Development**: Can write tests for future features (Tier C)
4. **Reduces False Alarms**: Tier C failures expected, Tier A failures urgent
5. **Guides Effort**: Clear which tests need investigation (B) vs are known incomplete (C)
6. **Maintains Momentum**: Can add tests without breaking CI (mark as Tier C)

## Open Questions

### For User Review

1. **Comparison Operators**: Are UWQuantity comparison operators (<, >, ==, !=) intended to be fully functional? If not, mark tests as Tier C + xfail.

2. **Reduction Operations**: test_0850_comprehensive_reduction_operations.py was documented as "All Passing" in October. What changed? Should these be Tier B (investigate breakage) or Tier C (feature incomplete)?

3. **Units Propagation**: test_0850_units_propagation.py tests advanced units features. Are these complete or still in development? Determines Tier B vs Tier C.

4. **Mesh Variable Ordering**: test_0813_mesh_variable_ordering_regression.py tests the "No Batman" fix. CLAUDE.md says this is fixed. Why failing now? High priority investigation (should be Tier B→A).

## Implementation Checklist

- [x] Design test reliability classification system
- [x] Create comprehensive documentation
- [x] Update pytest.ini with markers
- [x] Update CLAUDE.md with integrated system
- [x] Create slash commands for tier-based testing
- [x] Fix critical JIT unwrapping bug
- [x] Document bug fix with technical details
- [ ] Analyze current test failures (in progress - test run ongoing)
- [ ] Classify all failing tests into Tiers B or C
- [ ] Mark all passing core tests as Tier A
- [ ] Apply xfail markers to Tier C tests with reasons
- [ ] Update test_levels.sh for tier support (optional)
- [ ] Set up CI using Tier A tests
- [ ] Promote test_0818_stokes_nd.py to Tier A after 1 week
