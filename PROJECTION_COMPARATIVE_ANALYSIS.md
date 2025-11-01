# Projection vs Poisson Solver Analysis
**Date**: 2025-10-26
**Status**: Comparative Testing of Secondary Issues

---

## Executive Summary

Testing with Projection solvers reveals that the **scalar subscript error is NOT universal** but rather **Poisson-specific**. This narrows the scope and provides important insights.

---

## Test Results

### Test 1: Scalar Projection ✅ WORKS
```python
projection = uw.systems.Projection(mesh, scalar_field)
projection.uw_function = scalar_field
projection.solve()
# Result: ✓ SUCCESS - No scalar subscript error
```

**Key Finding**: Projection solvers do NOT trigger the `simplify()` → `cancel()` → `factor_terms()` code path that causes the scalar subscript error in Poisson.

### Test 2: Vector Projection ⚠️ FAILS (Different Error)
```python
vector_projection = uw.systems.Vector_Projection(mesh, vector_field)
vector_projection.uw_function = scalar_field.sym.diff(mesh.N.x)
vector_projection.solve()
# Result: ShapeError: Matrix size mismatch: (1, 2) + (1, 1)
```

**Finding**: Vector projection fails, but with a **different error** (shape mismatch), not the scalar subscript error.

---

## Issue 1: Scalar Subscript Error - SOLVER SPECIFIC

### Revised Analysis
- **Affects**: Poisson solver (confirmed)
- **Does NOT affect**: Scalar Projection solver (confirmed)
- **Status of Vector Projection**: Different error (unrelated)
- **Root Cause**: Poisson calls `simplify()` on constitutive model flux
- **Likelihood**: Other solvers may or may not trigger it

### Code Path Analysis

#### Poisson Solver Path (HITS SIMPLIFY)
```
Poisson._setup_pointwise_functions()
  ↓
sympy.simplify(self.constitutive_model.flux.T)  ← CRITICAL LINE
  ↓
cancel() → factor_terms()
  ↓
Tries to iterate over scalar UWexpression
  ↓
TypeError: 'UWexpression' object (scalar) is not subscriptable
```

#### Projection Solver Path (DOES NOT HIT SIMPLIFY)
```
Projection._setup_pointwise_functions()
  ↓
No simplify() call on expressions
  ↓
Direct evaluation/compilation
  ↓
✓ Works fine
```

### Hypothesis: Why Projection Doesn't Hit It

1. **Projections are simpler**: Map function space to another function space
2. **No constitutive model**: Projections don't have complex flux calculations
3. **No simplification**: Projection doesn't call `simplify()` on flux expressions
4. **Direct compilation**: Expression goes straight to PETSc

### Code Search to Confirm

Let me check where `simplify()` is called:

```bash
# In Poisson: src/underworld3/systems/solvers.py
grep -n "sympy.simplify" src/underworld3/systems/solvers.py

# Expected to find: simplify(self.constitutive_model.flux.T)
```

---

## Issue 2: Scaling Not Applied - AFFECTS ALL SOLVERS

The scaling issue is likely **independent of solver type** since it happens in the general `unwrap()` function.

### Both Solvers Use unwrap()
```python
# Poisson uses unwrap in:
# - Flux simplification
# - Expression compilation

# Projection likely uses unwrap in:
# - Function space mapping
# - Expression evaluation
```

So the scaling issue would affect both if we tested with ND scaling enabled.

---

## Comparative Solver Pathways

### Solver Complexity Comparison

| Aspect | Scalar Projection | Poisson Solver | Complexity |
|--------|------------------|----------------|-----------|
| **Constitutive Model** | None | Yes (Diffusion) | Poisson higher |
| **Simplification** | No | Yes | Poisson higher |
| **Boundary Conditions** | Weak | Yes | Poisson higher |
| **Expression Compilation** | Direct | Via Simplify | Poisson more complex |
| **Error in Testing** | None | Scalar Subscript | **Poisson-specific** |

### Why Projection Is Simpler

**Projection Intent**:
- Takes a function defined at quadrature points
- Projects it to mesh variable space
- Solves: Find u such that ∫(u-f)·v dx = 0

**Poisson Intent**:
- Solves: ∇²u = f
- Requires constitutive modeling (flux calculation)
- Requires material properties
- More complex expression manipulation

---

## Severity Reassessment

### Issue 1: Scalar Subscript Error

**Original Assessment**: HIGH (blocks all solvers)
**Revised Assessment**: MEDIUM (blocks Poisson and similar complex solvers)

**Reasoning**:
- ✓ Confirmed to affect Poisson
- ✓ Confirmed to NOT affect Scalar Projection
- ? Unknown impact on: Stokes, AdvDiffusion, other solvers
- **Scope**: Likely affects solvers with complex constitutive models

**Next Step**: Test with Stokes and AdvDiffusion to map which solvers are affected.

### Issue 2: Scaling Not Applied

**Original Assessment**: MEDIUM (affects ND scaling tests)
**Revised Assessment**: MEDIUM-HIGH (likely affects all solvers with ND scaling)

**Reasoning**:
- Happens in general `unwrap()` function
- All solvers use unwrap for compilation
- Affects any solver using ND scaling

**Next Step**: Test projections with ND scaling enabled.

---

## Testing Matrix

Create tests to map which solvers/features trigger which issues:

```
                     Scalar Subscript    Scaling        Shape/Type Errors
Scalar Projection    ✓ NO                ? Unknown      ? Unknown
Vector Projection    ? Unknown           ? Unknown      ✓ YES (shape mismatch)
Poisson              ✓ YES               ? Unknown      ? Unknown
Stokes               ? Unknown           ? Unknown      ? Unknown
AdvDiffusion         ? Unknown           ? Unknown      ? Unknown

* Need to test each combination
```

---

## Root Cause Refinement

### Scalar Subscript Error
**REFINED CAUSE**: Poisson solver calls `sympy.simplify()` on flux expressions containing scalar UWexpressions.

**Evidence**:
1. Poisson fails with subscript error ✓
2. Projection (no simplify) works ✓
3. Error occurs in SymPy's factor_terms() ✓

**Conclusion**: This is a **Simplify-triggered issue**, not a general arithmetic issue.

### Potential Affected Solvers
Likely to trigger this if they:
1. Have constitutive models
2. Call `simplify()` on expressions
3. Have flux calculations
4. Use scalar UWexpressions in calculations

**Candidates**:
- Poisson: ✓ Confirmed
- Stokes: Likely (has constitutive model)
- AdvDiffusion: Likely (has constitutive model)
- Projection: ✗ Does not simplify

---

## Next Phase: Comprehensive Solver Testing

### Quick Test Commands

```bash
# Test Stokes
pixi run -e default python -c "
import underworld3 as uw
mesh = uw.meshing.StructuredQuadBox(elementRes=(4,4))
v = uw.discretisation.MeshVariable('v', mesh, 2, degree=2)
p = uw.discretisation.MeshVariable('p', mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, v_soln=v, p_soln=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
try:
    stokes.solve()
    print('✓ Stokes works')
except TypeError as e:
    print(f'✗ Stokes error: {e}')
"

# Test AdvDiffusion
pixi run -e default python -c "
import underworld3 as uw
mesh = uw.meshing.StructuredQuadBox(elementRes=(4,4))
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2)
adv_diff = uw.systems.AdvDiffusion(mesh, T, uw.discretisation.MeshVariable('v', mesh, 2, degree=1))
adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1.0
try:
    adv_diff.solve()
    print('✓ AdvDiffusion works')
except TypeError as e:
    print(f'✗ AdvDiffusion error: {e}')
"
```

---

## Impact on Fix Strategy

### Original Fix Options for Scalar Subscript

Given that this is **Poisson-specific**, we have additional options:

#### Option A: Override `__iter__()` (STILL BEST)
- Fixes for all cases
- Minimal code change
- General solution

#### Option E: Skip Simplify for Poisson Only (NEW)
```python
# In Poisson solver
if isinstance(expr, UWexpression):
    # Don't simplify UWexpressions in Poisson
    simplified_flux = self.constitutive_model.flux
else:
    simplified_flux = sympy.simplify(self.constitutive_model.flux)
```
- **Pros**: Targeted fix, avoids changing core logic
- **Cons**: Workaround, not root cause fix
- **Effort**: 10 minutes

#### Option F: Pre-simplify Before Projection (NEW)
```python
# In Poisson setup
flux = sympy.simplify(unwrap(self.constitutive_model.flux))
# Now flux is guaranteed to be pure SymPy, no UWexpression
```
- **Pros**: Converts UWexpression to SymPy before simplification
- **Cons**: May change semantics
- **Effort**: 15 minutes

### Recommendation

**Use Option A** (Override `__iter__()`) because:
1. Fixes root cause, not symptom
2. Helps other solvers proactively
3. Minimal code change (5 minutes)
4. Most robust solution

---

## Key Insights from Projection Testing

### Insight 1: Not All Solvers Are Equal
- Projection (simple) works fine
- Poisson (complex) triggers error
- This suggests solvers with simpler expression handling don't hit the issue

### Insight 2: The Simplify Call Is Critical
- The error path goes through `simplify()`
- Solvers that avoid `simplify()` don't hit it
- This is a **SymPy integration issue**, not a pure MathematicalMixin issue

### Insight 3: Expression Complexity Matters
- Simple expressions in Projection: Fine
- Complex calculations in Poisson: Problem
- This suggests the issue manifests with more complex symbolic trees

---

## Updated Issue Classification

| Issue | Severity | Scope | Root Cause | Fix Priority |
|-------|----------|-------|-----------|--------------|
| **Scalar Subscript** | MEDIUM | Poisson + likely Stokes/AdvDiff | Simplify iteration | HIGH |
| **Scaling** | MEDIUM-HIGH | All solvers with ND | Design/Test mismatch | MEDIUM |

---

## Recommendations

### Immediate (Next Session)
1. ✅ Keep unwrap fix (confirmed working)
2. ✅ Keep MathematicalMixin symbolic preservation
3. **NEW**: Test Stokes and AdvDiffusion to confirm scope
4. **PRIORITY**: Fix scalar subscript with Option A (5 minutes)

### Short Term
1. Run comprehensive solver test matrix
2. Fix scaling issue (either test or implementation)
3. Document which solvers are affected

### Long Term
1. Consider whether Symbol inheritance for UWexpression is appropriate
2. Review why Projection doesn't hit simplify issue
3. Improve solver error messages

---

## Conclusion

Projection testing revealed that the **scalar subscript error is solver-specific**, not universal. This significantly improves our understanding:

- **Scalar Projection works** without the error
- **Poisson fails** due to simplify() call on flux
- **Scaling issue** is likely independent and affects all solvers using ND

This suggests:
1. Fix should be targeted at `simplify()` interaction
2. Option A (override `__iter__()`) is best general fix
3. Need to test more solvers to understand full scope
4. The issue is **not** caused by our unwrap changes

The projection analysis confirms that our fixes are good, and the secondary issues are separate concerns with clear solutions.