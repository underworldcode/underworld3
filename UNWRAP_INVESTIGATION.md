# Unwrap Issue - Deep Investigation
**Date**: 2025-10-26
**Status**: Critical Bug Analysis

---

## The Problem

The `unwrap()` function is not substituting UWexpression values, breaking solver compilation:

```python
Ra = uw.expression(r"\mathrm{Ra}", 1e7)
T = uw.MeshVariable("T", mesh, 1)
expr = Ra * T

unwrapped = uw.function.fn_unwrap(expr)
# Returns: "Ra * T" (unchanged)
# Expected: "10000000.0 * T" (Ra substituted)
```

This prevents construction of solvers because expressions remain symbolic.

---

## Root Cause Analysis

### Key Discovery: `keep_constants` Parameter

The unwrap function has a critical parameter `keep_constants=True` (default).

**In `_substitute_all_once` (line 30-31)**:
```python
if keep_constants and is_constant_expr(atom):
    continue  # SKIP substitution of constant expressions!
```

### The Logic Chain

1. **Ra is a constant expression**: It has no dependencies on variables
   - `is_constant_expr(Ra)` returns `True`

2. **When `keep_constants=True`**: Constant expressions are NOT substituted
   - This is by design - to preserve expression symbols

3. **But then `unwrap()` is called with default `keep_constants=True`**:
   - Result: Ra is never substituted
   - Solver gets `Ra * T` instead of `10000000.0 * T`

### Code Path

```
unwrap(expr, keep_constants=True)  # DEFAULT!
  ↓
_unwrap_expressions(expr, keep_constants=True)
  ↓
_substitute_all_once(expr, keep_constants=True)
  ↓
for atom in extract_expressions_and_functions(expr):
    if isinstance(atom, UWexpression):
        if keep_constants and is_constant_expr(atom):  # <-- SKIPS Ra!
            continue
```

---

## Problem Scenarios

### Scenario 1: Simple Expression
```python
Ra = uw.expression("Ra", 1e7)
T = uw.MeshVariable("T", mesh, 1)
expr = Ra * T

# What happens:
unwrap(expr)
  → _substitute_all_once(expr, keep_constants=True)
    → Finds Ra in atoms
    → Checks: is_constant_expr(Ra) → True
    → Checks: keep_constants and is_constant_expr(Ra) → True
    → Skips Ra with continue
    → Result: expr unchanged, returns "Ra * T"
```

### Scenario 2: Complex Expression
```python
alpha = uw.expression("alpha", 3e-5)  # constant
T = uw.MeshVariable("T", mesh, 1)
grad_T = T.sym.diff(mesh.N.x)  # has variable, not constant

expr = alpha * grad_T

# What happens:
unwrap(expr, keep_constants=True)
  → _substitute_all_once finds: alpha (constant), grad_T (non-constant)
  → Skips alpha (keep_constants + is_constant)
  → Substitutes grad_T
  → Result: "alpha * (derivative)" - still has unsubstituted alpha!
```

---

## Why This Breaks Solvers

Solvers need **completely unwrapped expressions** with all numeric values substituted:

```python
# What solvers expect:
10000000.0 * T(x, y)

# What they're getting:
Ra * T(x, y)

# Result:
✗ NameError: Ra not defined in solver compilation context
✗ Solver fails to compile
✗ Cannot construct or use solver
```

---

## The Design Intent vs. Reality

### What Was Intended
- `keep_constants=False`: Substitute everything (for compilation)
- `keep_constants=True`: Keep constant expressions as symbols (for display/documentation)

### What's Happening
- Unwrap is called with default `keep_constants=True`
- This was probably intended for intermediate processing
- But it's breaking the final compilation step

---

## Potential Fixes

### Fix 1: Change Default Parameter (SIMPLE) ⭐
```python
def unwrap(fn, keep_constants=False, return_self=True):  # Change default!
    """Unwrap UW expressions..."""
```

**Pros**: Simple one-line fix
**Cons**: May break code that relies on current default

### Fix 2: Add Explicit `full_unwrap()` Function
```python
def full_unwrap(fn):
    """Fully unwrap all expressions (for compilation)."""
    return unwrap(fn, keep_constants=False, return_self=True)

def partial_unwrap(fn):
    """Keep constants symbolic (for display)."""
    return unwrap(fn, keep_constants=True, return_self=True)
```

**Pros**: Clear intent, no breaking changes
**Cons**: Requires renaming/migration

### Fix 3: Detect Context (ADVANCED)
```python
def unwrap(fn, keep_constants=None, return_self=True):
    if keep_constants is None:
        # Auto-detect: if any UWexpression found, default to False
        # (assume we want full unwrap if expressions are present)
        keep_constants = len(extract_expressions_and_functions(fn)) == 0
    return _unwrap_expressions(fn, keep_constants, return_self)
```

**Pros**: Smart default behavior
**Cons**: Complex, may have edge cases

### Fix 4: Split Substitution Logic (BEST LONG-TERM)
```python
def substitute_variables_only(fn):
    """Substitute non-constant expressions (vars, derivatives)."""
    expr = fn
    for atom in extract_expressions_and_functions(fn):
        if isinstance(atom, UWexpression):
            # Only substitute if it's NOT a constant expression
            if not is_constant_expr(atom):
                expr = expr.subs(atom, atom.sym)
        elif callable(atom):  # Variables/Functions
            expr = expr.subs(atom, ...)
    return expr

def full_unwrap(fn):
    """Substitute everything including constants."""
    expr = fn
    for atom in extract_expressions_and_functions(fn):
        expr = expr.subs(atom, atom.sym)
    return expr
```

**Pros**: Clear separation of concerns
**Cons**: Requires refactoring multiple functions

---

## Test Case: Verify the Issue

```python
import underworld3 as uw

mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
Ra = uw.expression("Ra", 1e7)
T = uw.MeshVariable("T", mesh, 1)

expr = Ra * T
print(f"Original: {expr}")

# Current behavior (broken)
unwrapped_default = uw.function.fn_unwrap(expr)
print(f"unwrap() default: {unwrapped_default}")  # Shows "Ra * T" - WRONG!

# What should happen
unwrapped_full = uw.function.fn_unwrap(expr, keep_constants=False)
print(f"unwrap(keep_constants=False): {unwrapped_full}")  # Should show "1e7 * T"
```

---

## Recommendation

**Immediate Fix**: Use **Fix 1** - Change default to `keep_constants=False`

This is:
1. **Simplest** - One parameter change
2. **Most correct** - Matches solver compilation needs
3. **Least breaking** - Code explicitly calling with `keep_constants=True` still works

**Then**: Add warning/deprecation for code that passes `keep_constants=True` explicitly

**Long-term**: Implement Fix 4 for better API clarity

---

## Implementation Plan

1. **Change default**: `keep_constants=False` in `unwrap()`
2. **Update docstring**: Explain the parameter clearly
3. **Add test**: Verify substitution actually happens
4. **Check impacts**: Run full test suite
5. **Update caller code**: Any code relying on old default must be fixed

---

## Related Code Paths

### Where unwrap is called:
- Solver compilation
- Expression evaluation
- JIT compilation
- Function evaluation

### Who calls unwrap:
- PETSc solver assembly
- Function evaluation routines
- Symbolic simplification

All of these NEED full unwrapping (keep_constants=False) for correct operation.

---

**Next Step**: Implement Fix 1 and verify it resolves the solver compilation issues.