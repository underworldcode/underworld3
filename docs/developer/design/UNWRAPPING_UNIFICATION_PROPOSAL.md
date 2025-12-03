# Unwrapping Unification Proposal

**Date**: 2025-11-14
**Context**: After fixing the variable scaling bug, both unwrapping paths (JIT compilation vs evaluate) are doing essentially the same thing. This proposal shows how to unify them into a robust common core.

## Current State Analysis

### Test Results (unwrap_convergence_analysis.py)

**What both paths handle IDENTICALLY ✓**:
1. **Variable-only expressions**: Both return variable symbol unchanged
   - Test: `T.sym` → Both return `T(x,y)`
   - Reason: Variables reference PETSc data (already non-dimensional)

2. **Variable + dimensionless constant**: Both preserve expression structure
   - Test: `T.sym + 100.0` → Both return `T(x,y) + 100.0`
   - Reason: Both leave variable symbols alone, keep plain numbers

**What shows IMPLEMENTATION differences**:
3. **Pure UWQuantity constants**: Different results but SHOULD be identical
   - Test: `uw.quantity(300, "K")`
   - Evaluate: Returns `0.3` (300K / 1000K scale) ✓ Correct
   - JIT: Returns `300` (raw value) ✗ Incorrect behavior

### Root Cause of Difference

Both paths have the SAME logic for non-dimensionalizing constants:

**Evaluate** (lines 358-371):
```python
if isinstance(expr, UWQuantity):
    if should_scale:
        nondim_qty = uw.non_dimensionalise(expr)
        return sympy.sympify(nondim_qty.value), result_dimensionality
    return sympy.sympify(expr.value), result_dimensionality
```

**JIT** (lines 143-171):
```python
if isinstance(fn, UWQuantity):
    if uw._is_scaling_active() and fn.has_units:
        nondim = uw.non_dimensionalise(fn)
        return sympy.sympify(nondim.value)
    return sympy.sympify(fn.value)
```

**They're identical!** But the JIT test shows `300` not `0.3`. This suggests:
- Either the JIT path isn't being called for constants in our test
- Or there's a bug in how constants are passed through the recursion
- Or the fixed-point iteration in `_unwrap_expressions()` is interfering

## Common Core Logic

Both pathways fundamentally do:

```python
def _core_unwrap_logic(expr):
    """What both paths SHOULD do."""

    # 1. Handle UWQuantity constants
    if isinstance(expr, UWQuantity):
        if scaling_active and has_units:
            return non_dimensionalise(expr).value
        return expr.value

    # 2. Handle derivatives (JIT needs this)
    if isinstance(expr, UWDerivativeExpression):
        expr = expr.doit()

    # 3. Handle matrices (JIT needs this)
    if isinstance(expr, sympy.Matrix):
        return expr.applyfunc(_core_unwrap_logic)

    # 4. Get SymPy core from wrappers
    if hasattr(expr, 'sym'):
        sym_expr = expr.sym
    elif hasattr(expr, '_sym'):
        sym_expr = expr._sym
    else:
        sym_expr = expr

    # 5. Find and non-dimensionalize constant atoms (NOT variable atoms)
    if scaling_active and hasattr(sym_expr, 'atoms'):
        for atom in sym_expr.atoms():
            if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
                # Non-dimensionalize constant atom
                nondim = non_dimensionalise(atom)
                sym_expr = sym_expr.subs(atom, nondim.value)

    return sym_expr
```

## Proposed Unified Architecture

### Core Function: `_unwrap_expression_core()`

```python
def _unwrap_expression_core(expr):
    """
    Core unwrapping logic shared by JIT compilation and evaluate pathways.

    This function:
    1. Non-dimensionalizes UWQuantity constants if scaling is active
    2. Leaves variable function symbols unchanged (they're ND in PETSc)
    3. Handles special cases (derivatives, matrices)
    4. Returns pure SymPy expression

    Does NOT:
    - Track dimensionality (wrapper's responsibility)
    - Apply variable scaling (variables already ND)
    - Handle display formatting

    Args:
        expr: Any UW expression type (UWQuantity, MeshVariable, SymPy, etc.)

    Returns:
        Pure SymPy expression ready for lambdify or JIT compilation
    """
    import underworld3 as uw
    import sympy
    from .quantities import UWQuantity
    from .expressions import UWDerivativeExpression

    scaling_active = uw._is_scaling_active()

    # Case 1: UWQuantity constant - non-dimensionalize if scaling active
    if isinstance(expr, UWQuantity):
        if scaling_active and expr.has_units:
            nondim = uw.non_dimensionalise(expr)
            if hasattr(nondim, 'value'):
                return sympy.sympify(nondim.value)
            elif hasattr(nondim, '_sym'):
                return nondim._sym
        # No scaling or no units: return plain value
        if hasattr(expr, 'value'):
            return sympy.sympify(expr.value)
        elif hasattr(expr, '_sym'):
            return expr._sym
        return expr

    # Case 2: Derivative - evaluate it first
    if isinstance(expr, UWDerivativeExpression):
        expr = expr.doit()
        # Recursively unwrap the result
        return _unwrap_expression_core(expr)

    # Case 3: Matrix - apply recursively to each element
    if isinstance(expr, sympy.Matrix):
        return expr.applyfunc(_unwrap_expression_core)

    # Case 4: Extract SymPy core from various wrapper types
    if hasattr(expr, 'sym'):
        sym_expr = expr.sym
    elif hasattr(expr, '_sym'):
        sym_expr = expr._sym
    else:
        sym_expr = expr

    # Case 5: Process expression - find and non-dimensionalize constant atoms
    # Leave variable function symbols (like T(x,y)) unchanged
    if scaling_active and hasattr(sym_expr, 'atoms'):
        substitutions = {}

        for atom in sym_expr.atoms():
            # Check if atom is a UW constant with units
            if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
                try:
                    nondim_atom = uw.non_dimensionalise(atom)
                    if hasattr(nondim_atom, 'value'):
                        substitutions[atom] = nondim_atom.value
                except Exception:
                    # If non-dimensionalization fails, leave atom unchanged
                    pass

        if substitutions:
            sym_expr = sym_expr.subs(substitutions)

    return sym_expr
```

### Wrapper 1: Evaluate Path

```python
def unwrap_for_evaluate(expr, scaling_active=None):
    """
    Unwrap expression for evaluate/lambdify path with dimensionality tracking.

    This thin wrapper:
    1. Captures dimensionality BEFORE unwrapping (for re-dimensionalization)
    2. Calls common core unwrapping logic
    3. Returns (unwrapped_expr, dimensionality) tuple

    Args:
        expr: Expression to unwrap
        scaling_active: Override for scaling check (default: use global state)

    Returns:
        tuple: (unwrapped_expr, result_dimensionality)
    """
    import underworld3 as uw
    from underworld3.units import get_units, get_dimensionality

    # Step 1: Capture dimensionality for later re-dimensionalization
    result_units = get_units(expr)
    if result_units is not None:
        try:
            result_dimensionality = get_dimensionality(expr)
        except Exception:
            result_dimensionality = None
    else:
        result_dimensionality = None

    # Step 2: Call common core
    unwrapped_expr = _unwrap_expression_core(expr)

    # Step 3: Return with dimensionality info
    return unwrapped_expr, result_dimensionality
```

### Wrapper 2: JIT Compilation Path

```python
def _unwrap_for_compilation(fn, keep_constants=True, return_self=True):
    """
    Unwrap expression for JIT compilation path.

    This thin wrapper calls the common core and returns just the unwrapped
    expression (no dimensionality tracking needed for JIT).

    Args:
        fn: Expression to unwrap
        keep_constants: Legacy parameter (kept for API compatibility)
        return_self: Legacy parameter (kept for API compatibility)

    Returns:
        Pure SymPy expression ready for JIT compilation
    """
    # Call common core
    return _unwrap_expression_core(fn)
```

## Benefits of Unification

### 1. Reduces Fragility
**Before**: Two implementations with subtle differences
**After**: One robust implementation, two thin wrappers

**Risk reduction**:
- Can't introduce bugs in one path but not the other
- Changes to unwrapping logic only need to be made once
- Testing covers both paths automatically

### 2. Eliminates Code Duplication

**Current code size**:
- `_unwrap_for_compilation()`: 70 lines + `_unwrap_expressions()`: 50 lines = 120 lines
- `unwrap_for_evaluate()`: 115 lines
- **Total**: 235 lines of overlapping logic

**Unified code size**:
- `_unwrap_expression_core()`: 80 lines (common logic)
- `unwrap_for_evaluate()`: 20 lines (wrapper)
- `_unwrap_for_compilation()`: 5 lines (wrapper)
- **Total**: 105 lines (56% reduction)

### 3. Makes Testing Easier

**Before**: Must test both paths separately, easy to miss differences
**After**: Test core once, verify wrappers add correct features

**Test strategy**:
```python
def test_unwrap_core():
    """Test common unwrapping logic."""
    # Test constants
    assert _unwrap_core(uw.quantity(300, "K")) == 0.3  # With scaling

    # Test variables
    assert _unwrap_core(T.sym) == T.sym  # Unchanged

    # Test mixed expressions
    expr = T.sym + uw.quantity(100, "K")
    assert _unwrap_core(expr) == T.sym + 0.1

def test_evaluate_wrapper():
    """Test evaluate wrapper adds dimensionality."""
    expr, dims = unwrap_for_evaluate(uw.quantity(300, "K"))
    assert expr == 0.3
    assert dims == {'temperature': 1}

def test_jit_wrapper():
    """Test JIT wrapper returns just expression."""
    expr = _unwrap_for_compilation(uw.quantity(300, "K"))
    assert expr == 0.3
    assert not isinstance(expr, tuple)  # No dimensionality
```

### 4. Clearer Intent

**Current code**: Complexity obscures the simple core logic
**Unified code**: Core logic is obvious, wrappers show path-specific needs

**Before**: "Why are there two implementations?"
**After**: "Oh, they share a core, wrappers just add dimensionality tracking"

### 5. Future-Proof

**If we need to change unwrapping logic**:
- Before: Must change in two places, ensure consistency
- After: Change once in core, automatically consistent

**If we add new expression types**:
- Before: Must update both paths
- After: Update core once

## Migration Strategy

### Phase 1: Create Core (No Breaking Changes)

1. Add `_unwrap_expression_core()` to `expressions.py`
2. Keep existing `_unwrap_for_compilation()` and `unwrap_for_evaluate()` unchanged
3. Test core function independently

### Phase 2: Switch Evaluate Path

1. Rewrite `unwrap_for_evaluate()` to call core
2. Run all evaluate-related tests (lambdify, function evaluation)
3. Ensure no behavior changes

### Phase 3: Switch JIT Path

1. Rewrite `_unwrap_for_compilation()` to call core
2. Run all JIT-related tests (solver compilation, PETSc integration)
3. Ensure no behavior changes

### Phase 4: Remove Old Code

1. Remove `_unwrap_expressions()` and helper functions
2. Remove `_substitute_all_once()` and related recursion code
3. Clean up docstrings and comments

### Phase 5: Validate

1. Run full test suite
2. Run Stokes scaling tests (ensure no regression of the bug we just fixed)
3. Run example notebooks
4. Benchmark performance (ensure no slowdown)

## Risk Assessment

### Low Risk ✅

**Reason**: Both paths already do the same thing, just with different implementations

**Evidence from testing**:
- Variables: Both produce identical results ✓
- Variables + constants: Both produce identical results ✓
- Pure constants: Both SHOULD produce identical results (current difference is a bug)

**Mitigation**: Comprehensive testing at each phase

### What Could Go Wrong?

1. **Performance regression**: Fixed-point iteration vs single-pass might have performance implications
   - **Mitigation**: Benchmark before/after, use profiling
   - **Likely**: No significant difference (unwrapping is fast compared to solving)

2. **Edge cases we haven't tested**: Derivatives, matrices, deeply nested expressions
   - **Mitigation**: Add comprehensive test cases before migration
   - **Likely**: Core handles these already (JIT path has them)

3. **Breaking existing code**: Some code might depend on subtle differences
   - **Mitigation**: Incremental migration, extensive testing
   - **Likely**: Unlikely - differences are bugs, not features

## Recommendation

**PROCEED WITH UNIFICATION**

**Rationale**:
1. Both paths do the same thing - duplication is pure tech debt
2. Fragility already caused one major bug (variable scaling)
3. Test results show identical behavior for correctly working cases
4. Unification significantly reduces future bug risk
5. Migration can be done incrementally with low risk

**Priority**: Medium-High
- Not urgent (both paths work after bug fix)
- But important (reduces future maintenance burden and bug risk)

**Effort**: 1-2 days
- Day 1: Create core, test, switch evaluate path
- Day 2: Switch JIT path, validate, clean up

## Next Steps

1. Get user approval for this approach
2. Create comprehensive test cases for edge cases
3. Implement `_unwrap_expression_core()`
4. Migrate evaluate path first (simpler, newer code)
5. Migrate JIT path second (more complex, older code)
6. Validate with full test suite
7. Update documentation

---

**Questions for Discussion**:

1. Do we want to preserve the `keep_constants` parameter in JIT path, or is it now obsolete?
2. Should we keep the fixed-point iteration for any cases, or is single-pass sufficient?
3. Are there any known edge cases where the two paths MUST behave differently?
