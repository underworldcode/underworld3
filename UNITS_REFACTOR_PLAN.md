# Units System Refactoring Plan
**Date:** 2025-01-07
**Status:** Planning - Pre-Refactor Documentation

## Context

The units system crashed mid-refactor during mathematical mixin replacement, leaving the codebase in a fragmented state with:
- Two competing UnitAwareArray implementations
- Partially integrated UnitAwareExpression architecture
- Broken `.to_nd()` method
- Inconsistent API (`.to()` vs `.to_units()`)
- Unclear closure properties

This document outlines the comprehensive refactoring plan to complete the work.

## Guiding Principles

### 1. Closure Property (MUST ENFORCE)
**Definition:** Unit-aware × Unit-aware → Unit-aware

All operations between unit-aware objects MUST preserve unit-awareness:
- `Temperature * Velocity[0]` → unit-aware expression
- `Temperature.diff(x)` → unit-aware derivative
- `2 * Temperature * Velocity[0]` → unit-aware expression
- `coord.max()` → unit-aware value
- etc.

**Why:** Without closure, unit-awareness becomes unreliable. Users can't compose operations freely without losing unit information unpredictably.

### 2. API Consistency (STANDARDIZE ON PINT)
- Use `.to(target_units)` NOT `.to_units()` (match Pint)
- Use `.magnitude` for raw values (match Pint)
- Use `.units` for unit strings (match Pint)

### 3. No Deprecation (CLEAN BREAKS)
Since code hasn't been widely released, we can make breaking changes rather than maintaining deprecated APIs.

### 4. Complete or Abort Partial Work
The UnitAwareExpression architecture is partially integrated. Either:
- **Complete** integration across all variable types, OR
- **Abort** and stick with current mixin approach

No half-finished states.

## Refactoring Steps

### Phase 1: Remove Duplicates (IMMEDIATE) ⚠️ HIGH PRIORITY

**Goal:** Single UnitAwareArray implementation

**Actions:**
1. Document all uses of lightweight UnitAwareArray ✅ DONE
2. Update imports in `ddt.py` (2 locations)
3. Update imports in `unit_conversion.py` (add import at top)
4. Delete lines 12-192 from `function/unit_conversion.py`
5. Run test suite to verify no breakage

**Files Modified:**
- `src/underworld3/systems/ddt.py` (lines 738, 768)
- `src/underworld3/function/unit_conversion.py` (add import, delete class)

**Testing:**
```bash
pixi run -e default pytest tests/test_*ddt*.py -v
pixi run -e default pytest tests/test_*units*.py -v
pixi run -e default pytest tests/test_0850_units_closure_comprehensive.py -v
```

**Risk:** LOW-MEDIUM (well-documented usage, simple replacement)

### Phase 2: API Standardization (MEDIUM PRIORITY)

**Goal:** Consistent `.to()` API everywhere

**Actions:**
1. Find all `.to_units()` methods
2. Replace with `.to()` or make `.to_units()` an alias
3. Update all call sites
4. Update documentation

**Files to Search:**
- `utilities/unit_aware_array.py` (has both `.to()` and `.to_units()`)
- `function/quantities.py` (UWQuantity)
- Any other unit-aware classes

**Testing:**
```bash
pixi run -e default pytest tests/test_*units*.py -v
pixi run -e default pytest tests/test_*quantities*.py -v
```

**Risk:** LOW (mostly renaming, can keep aliases temporarily)

### Phase 3: Remove Broken `.to_nd()` (MEDIUM PRIORITY)

**Goal:** Remove or replace symbolic non-dimensionalization

**Actions:**
1. Find all uses of `.to_nd()` method
2. Determine if any code actually uses it
3. Either:
   - **Remove** if unused, OR
   - **Replace** with proper array-based non-dimensionalization

**Files:**
- `utilities/dimensionality_mixin.py` (lines 101-153)
- Search for calls: `grep -r "\.to_nd\(" src/`

**Why Broken:** Returns SymPy expression `var.sym / scaling_coefficient` which doesn't respect variable data operations.

**Alternative:** Use `uw.non_dimensionalise(var.array)` pattern instead.

**Testing:**
```bash
pixi run -e default pytest tests/test_*nondimensional*.py -v
```

**Risk:** MEDIUM (need to verify not used in critical code)

### Phase 4: Complete Mathematical Mixin Replacement (HIGH EFFORT)

**Goal:** Finish integrating UnitAwareExpression architecture

**Current State:**
- `expression/unit_aware_expression.py` exists and is partially integrated
- `mathematical_mixin.py` uses it for component access (5 locations)
- Architecture is sound (4-layer: SymPy, Units, Math, Lazy)

**Decision Point:** Complete or Abort?

**Option A: Complete Integration**
1. Make all variable operations return UnitAwareExpression
2. Replace old mixins (units_mixin, dimensionality_mixin, mathematical_mixin)
3. Ensure closure property everywhere
4. Update all variable classes to use new architecture

**Benefits:**
- Clean architecture with clear separation
- Guaranteed closure property
- Future-proof design

**Effort:** HIGH (1-2 weeks)
**Risk:** MEDIUM-HIGH (requires extensive testing)

**Option B: Abort and Consolidate**
1. Remove `expression/unit_aware_expression.py`
2. Revert mathematical_mixin to use only old mixins
3. Document that current mixin approach is permanent
4. Fix closure property within current architecture

**Benefits:**
- Less work
- Known working state

**Effort:** LOW (few days)
**Risk:** LOW (reverting to known state)

**Recommendation:** ⚠️ **NEED USER DECISION**

Before proceeding, user should decide:
- Is the UnitAwareExpression architecture worth completing?
- Or should we stick with current mixin approach and make it work?

### Phase 5: Enforce Closure Property (CRITICAL)

**Goal:** Verify and enforce closure everywhere

**Actions:**
1. Run comprehensive test suite ✅ CREATED: `test_0850_units_closure_comprehensive.py`
2. Fix any failures
3. Document closure as a requirement in architecture docs
4. Add CI checks for closure property

**Test Coverage:**
- Variable × Variable
- Scalar × Variable
- Derivatives
- Division
- Component access
- Coordinates
- Addition/subtraction with unit checking
- Power operations
- Complex compound expressions
- UnitAwareArray arithmetic
- Evaluation results

**Testing:**
```bash
pixi run -e default pytest tests/test_0850_units_closure_comprehensive.py -v
```

**Risk:** MEDIUM (may reveal missing closure in some operations)

### Phase 6: Documentation and Examples

**Goal:** Clear documentation of unit system architecture

**Actions:**
1. Create `docs/developer/UNITS-ARCHITECTURE.md`
2. Document:
   - Closure property and why it matters
   - Which classes are unit-aware
   - How to use units in expressions
   - How derivatives handle units
   - Coordinate unit system
3. Update beginner tutorials to show best practices
4. Add troubleshooting guide

**Files:**
- New: `docs/developer/UNITS-ARCHITECTURE.md`
- Update: Relevant beginner tutorials
- Update: `CLAUDE.md` with final architecture

## Testing Strategy

### Before Any Changes
```bash
# Baseline - all tests should pass
pixi run -e default pytest tests/test_07*_units*.py -v
pixi run -e default pytest tests/test_08*_units*.py -v
```

### After Each Phase
```bash
# Units system tests
pixi run -e default pytest tests/test_07*_units*.py -v
pixi run -e default pytest tests/test_08*_units*.py -v

# New closure tests
pixi run -e default pytest tests/test_0850_units_closure_comprehensive.py -v

# Regression tests
pixi run -e default pytest tests/test_06*_regression.py -v

# Full test suite (final validation)
pixi run -e default pytest tests/ -v --tb=short
```

## Rollback Plan

If any phase causes significant problems:

1. **Git:** Each phase should be a separate commit
2. **Revert:** `git revert <commit-hash>`
3. **Document:** Why the revert was needed
4. **Re-plan:** Adjust strategy based on what failed

## Placeholder Commit (BEFORE STARTING)

**Title:** `docs: Units system audit and refactoring plan`

**Message:**
```
docs: Units system audit and refactoring plan

Comprehensive audit of units system reveals:
- Two competing UnitAwareArray implementations (one should be removed)
- Partial integration of UnitAwareExpression architecture (incomplete)
- Broken .to_nd() method using symbolic approach
- Inconsistent API (.to() vs .to_units())
- Unclear closure properties

Adds:
- UNITS_AUDIT.md - Complete audit findings
- LIGHTWEIGHT_UNITAWAREARRAY_USAGE.md - Documents all uses before removal
- UNITS_REFACTOR_PLAN.md - Comprehensive refactoring plan
- test_0850_units_closure_comprehensive.py - Test suite for closure property

Context:
Units system crashed mid-refactor during mathematical mixin replacement.
This documentation establishes baseline before systematic refactoring.

Next steps:
1. Remove duplicate UnitAwareArray (Phase 1)
2. Standardize API (Phase 2)
3. Decide: complete or abort mathematical mixin replacement (Phase 4)
4. Enforce closure property everywhere (Phase 5)

No code changes in this commit - documentation only.
```

**Files in Commit:**
- `UNITS_AUDIT.md` (not created yet - was interrupted)
- `LIGHTWEIGHT_UNITAWAREARRAY_USAGE.md` ✅
- `UNITS_REFACTOR_PLAN.md` ✅ (this file)
- `tests/test_0850_units_closure_comprehensive.py` ✅

## Success Criteria

Units system is considered "fixed" when:
1. ✅ Single UnitAwareArray implementation
2. ✅ Consistent API (`.to()` everywhere)
3. ✅ Closure property verified by comprehensive tests
4. ✅ All derivatives have proper units
5. ✅ All coordinates are unit-aware
6. ✅ Complex expressions maintain unit-awareness
7. ✅ Documentation clearly explains architecture
8. ✅ All tests pass (including new closure tests)

## Timeline Estimate

- **Phase 1 (Remove duplicates):** 1-2 days
- **Phase 2 (API standardization):** 2-3 days
- **Phase 3 (Remove .to_nd()):** 1-2 days
- **Phase 4 (Complete mixin replacement):** 7-14 days (if chosen)
  - OR Phase 4 (Abort mixin replacement): 2-3 days
- **Phase 5 (Enforce closure):** 3-5 days
- **Phase 6 (Documentation):** 2-3 days

**Total (if completing mixin):** 16-29 days (3-6 weeks)
**Total (if aborting mixin):** 11-18 days (2-4 weeks)

## Questions for User

Before proceeding, need decisions on:

1. **Mathematical Mixin Replacement:** Complete or Abort?
   - Complete = Better architecture, more work
   - Abort = Less work, stick with current approach

2. **Breaking Changes:** Acceptable to break API?
   - Yes = Can remove `.to_units()` entirely
   - No = Keep as deprecated alias

3. **Testing Thoroughness:** How much testing before release?
   - Minimal = Run existing tests
   - Standard = Run existing + new closure tests
   - Thorough = Add extensive integration tests

4. **Documentation Priority:** When to write docs?
   - Early = Write as we go (slower but clearer)
   - Late = Write after refactoring (faster but may forget details)
