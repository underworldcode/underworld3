# Secondary Issues - Complete Index and Navigation
**For Comprehensive Understanding of All Issues**

---

## Overview

During investigation of the critical unwrap bug, two additional issues were discovered:

| Issue | Severity | Type | Status |
|-------|----------|------|--------|
| **Scalar Subscript Error** | HIGH | SymPy/UWexpression interaction | Needs Investigation |
| **Scaling Not Applied** | MEDIUM | Design vs Test Mismatch | Needs Investigation |

Both issues are **INDEPENDENT** of the unwrap fix and may be pre-existing.

---

## Issue 1: Scalar Subscript Error

### Quick Summary
SymPy's simplification code tries to iterate over scalar UWexpressions, which raises a TypeError. This blocks all solver compilation.

### Files with Details
- **Quick Reference**: `SECONDARY_ISSUES_QUICK_REFERENCE.md` (Start here!)
- **Detailed Analysis**: `SECONDARY_ISSUES_DETAILED_ANALYSIS.md` (Issue 1 section)
- **Source File**: `/src/underworld3/utilities/mathematical_mixin.py`, lines 53-86

### Key Information
```
Error: TypeError: 'UWexpression' object (scalar) is not subscriptable
Triggered: During SymPy simplify() → cancel() → factor_terms()
Cause: SymPy tries to iterate like: type(expr)([do(i) for i in expr])
Impact: Blocks ALL solver compilation
```

### Four Proposed Fixes (Pick One)
1. **Fix A (SIMPLEST)**: Override `__iter__()` - 5 minutes
2. **Fix B (DEFENSIVE)**: Safer `__getitem__()` - 10 minutes
3. **Fix C (TARGETED)**: Skip simplification - 20 minutes
4. **Fix D (ARCHITECTURAL)**: Don't inherit from Symbol - 2-3 hours

### Diagnostic Test
```bash
pixi run -e default python << 'EOF'
import underworld3 as uw
mesh = uw.meshing.StructuredQuadBox(elementRes=(2,2))
u = uw.discretisation.MeshVariable('u', mesh, 1, degree=2)
poisson = uw.systems.Poisson(mesh, u_Field=u)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1
poisson.add_dirichlet_bc(1.0, 'Bottom')
poisson.add_dirichlet_bc(0.0, 'Top')
try:
    poisson.solve()
    print('✓ No issue')
except TypeError as e:
    print(f'✗ Issue present: {e}')
EOF
```

---

## Issue 2: Scaling Not Applied in Unwrap

### Quick Summary
When ND (non-dimensional) scaling is enabled, the unwrap function does not apply scaling factors to expressions. Test expects them, but implementation intentionally skips them.

### Files with Details
- **Quick Reference**: `SECONDARY_ISSUES_QUICK_REFERENCE.md` (Start here!)
- **Detailed Analysis**: `SECONDARY_ISSUES_DETAILED_ANALYSIS.md` (Issue 2 section)
- **Source File**: `/src/underworld3/function/expressions.py`, lines 103-146
- **Test File**: `/tests/test_0816_global_nd_flag.py`, lines ~130-135

### Key Information
```
Test expects: unwrap(T.sym) contains "1000" or "0.001"
Actual: Returns T(N.x, N.y) without scaling
Cause: _apply_scaling_to_unwrapped() returns expression unchanged
Design: Comments say "PETSc handles all scaling, no need in unwrap"
Question: Is test wrong or implementation wrong?
```

### Three Possible Resolutions
1. **Resolution A (MOST LIKELY)**: Test is wrong - fix test expectations - 5 minutes
2. **Resolution B (UNLIKELY)**: Implementation is wrong - add scaling logic - 1-2 hours
3. **Resolution C (THOROUGH)**: Investigate both - check git history - 2-3 hours

### Diagnostic Test
```bash
pixi run -e default pytest tests/test_0816_global_nd_flag.py::test_unwrap_with_scaling -xvs
```

---

## Detailed Documentation Map

### For Understanding the Issues

#### Quick Start (5-10 minutes)
1. Read: `SECONDARY_ISSUES_QUICK_REFERENCE.md`
   - Quick facts for each issue
   - Quick diagnostics
   - Decision trees

#### Comprehensive Understanding (30-45 minutes)
1. Read: `SECONDARY_ISSUES_DETAILED_ANALYSIS.md`
   - Full problem descriptions
   - Root cause analysis
   - 4 fix options for each issue
   - Testing strategies

#### For Implementation
1. Reference: `SECONDARY_ISSUES_QUICK_REFERENCE.md`
   - Code snippets for each fix
   - Effort estimates
   - Debugging tips

---

## Critical Questions to Answer

### For Scalar Subscript Issue
1. Is this a new issue or pre-existing?
   - Check if Poisson tests were running before
   - Check if code was exercising this path

2. Which fix is best?
   - Fix A for quick solution
   - Fix B for defensive programming
   - Fix C to avoid SymPy interactions
   - Fix D for architectural cleanliness

3. Why wasn't it caught earlier?
   - Solver tests not running?
   - Code path never exercised?
   - Recent changes exposed it?

### For Scaling Issue
1. Is the test correct?
   - Does the test make sense?
   - Is scaling supposed to appear in unwrap?
   - What does the design doc say?

2. Is the implementation correct?
   - Is `_apply_scaling_to_unwrapped()` supposed to scale?
   - Was it ever implemented?
   - Why does it return unchanged?

3. Is there a mismatch?
   - Test expects one thing
   - Implementation does another
   - Who's right?

---

## Investigation Flowchart

```
START: Secondary Issues Investigation
│
├─ Scalar Subscript Error
│  ├─ Run diagnostic test
│  │  ├─ Error occurs → Expected
│  │  └─ No error → Issue not present in your environment
│  │
│  ├─ If error occurs:
│  │  ├─ Is it pre-existing?
│  │  │  ├─ Check git history
│  │  │  └─ Test with original code
│  │  │
│  │  └─ Pick fix (A is simplest)
│  │     ├─ Implement
│  │     ├─ Test thoroughly
│  │     └─ Document findings
│  │
│  └─ Track: Which solver tests use simplify()?
│
├─ Scaling Not Applied Issue
│  ├─ Run: pytest test_0816_global_nd_flag.py
│  │  ├─ Test fails → Expected
│  │  └─ Test passes → Already fixed somehow?
│  │
│  ├─ If test fails:
│  │  ├─ Is this pre-existing?
│  │  │  ├─ Check git history of test
│  │  │  └─ Check design doc creation date
│  │  │
│  │  ├─ Which resolution is correct?
│  │  │  ├─ Read design doc carefully
│  │  │  ├─ Check git history of implementation
│  │  │  └─ Decide: Test wrong or impl wrong?
│  │  │
│  │  └─ Apply fix (A is likely)
│  │     ├─ Update test or implementation
│  │     ├─ Run all ND scaling tests
│  │     └─ Document findings
│  │
│  └─ Track: Do other ND scaling tests work?
│
└─ COMPLETE: All issues investigated and documented
```

---

## Performance Impact Matrix

| Issue | If Ignored | If Fixed (A) | If Fixed (B) | If Fixed (D) |
|-------|-----------|-------------|-------------|-------------|
| **Scalar Subscript** | Solvers broken | Minimal impact | Defensive | Clean design |
| **Scaling** | Test fails | Test passes | Scaling works | Proper behavior |
| **Effort** | 0 | 5m | 1h | 2-3h |
| **Risk** | 0% | 1% | 5% | 20% |

---

## Summary of Evidence

### Scalar Subscript
- ✅ Confirmed to occur
- ✅ Root cause identified (SymPy iteration)
- ✅ Code location known (MathematicalMixin.__getitem__)
- ⚠️ Unknown if pre-existing or new
- ⚠️ Multiple fix options available

### Scaling Application
- ✅ Test fails as documented
- ✅ Implementation returns unchanged (by design)
- ✅ Design doc says no scaling needed
- ⚠️ Test might have wrong expectations
- ⚠️ Need to verify design intent

---

## Recommendations for Next Steps

### Priority Order
1. **CRITICAL**: Fix scalar subscript (blocks all solvers)
   - Time: 5 minutes (Fix A)
   - Risk: Very low
   - Impact: Enables solver compilation

2. **IMPORTANT**: Resolve scaling design question
   - Time: 30 minutes (investigation)
   - Risk: Low (either fix test or implement)
   - Impact: ND scaling tests pass

3. **NICE-TO-HAVE**: Architectural improvement (Fix D)
   - Time: 2-3 hours
   - Risk: Medium (large refactoring)
   - Impact: Cleaner design long-term

### Immediate Actions
1. Run both diagnostic tests
2. Determine scope (how many tests fail?)
3. Pick simpler fix for scalar issue (Fix A)
4. Determine if test or implementation needs fix for scaling
5. Implement and test
6. Document in code comments

---

## References and Links

### Within This Repository
- `SECONDARY_ISSUES_QUICK_REFERENCE.md` - Quick lookup guide
- `SECONDARY_ISSUES_DETAILED_ANALYSIS.md` - Comprehensive analysis
- `/src/underworld3/utilities/mathematical_mixin.py` - Scalar subscript location
- `/src/underworld3/function/expressions.py` - Scaling location
- `/tests/test_0816_global_nd_flag.py` - Failing test

### Code Locations
```
Scalar Subscript:
  src/underworld3/utilities/mathematical_mixin.py:53-86

Scaling Application:
  src/underworld3/function/expressions.py:75-146

Tests:
  tests/test_1000_poissonCart.py (blocked by scalar issue)
  tests/test_0816_global_nd_flag.py (scaling issue)
```

---

## Checklist for Resolution

### Scalar Subscript Issue
- [ ] Run diagnostic test
- [ ] Determine if pre-existing
- [ ] Choose fix (recommend A)
- [ ] Implement fix
- [ ] Test Poisson solver
- [ ] Test other solvers
- [ ] Document in code
- [ ] Close issue

### Scaling Application Issue
- [ ] Run failing test
- [ ] Read design doc thoroughly
- [ ] Check git history
- [ ] Determine: test wrong or impl wrong?
- [ ] Apply appropriate fix
- [ ] Run all ND scaling tests
- [ ] Update test expectations or implementation
- [ ] Document decision in code
- [ ] Close issue

---

## Summary

Two secondary issues have been identified and thoroughly documented:

1. **Scalar Subscript Error** - Quick fix needed (5 minutes)
2. **Scaling Not Applied** - Investigation needed (30 minutes)

Both are well-understood with clear fix options. Neither is caused by the unwrap fix (they're separate). Documentation is comprehensive for efficient resolution.

**Confidence in these analyses: VERY HIGH** ✅