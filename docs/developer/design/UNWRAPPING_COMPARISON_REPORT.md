# Unwrapping Comparison Report: Evaluate vs JIT Compilation

**Date**: 2025-11-14
**Purpose**: Compare unwrapping logic between `evaluate()` path and JIT compilation path

## Summary

Both pathways now handle constants and variables consistently after the scaling fix, but they differ in:
1. **Return signature** (evaluate returns dimensionality info, JIT doesn't)
2. **Recursion strategy** (different approaches to flattening)
3. **Atom detection method** (evaluate uses hasattr, JIT uses extract_expressions_and_functions)

## Detailed Comparison

### 1. Entry Points

**JIT Compilation Path:**
- Function: `_unwrap_for_compilation(fn, keep_constants, return_self)`
- Location: `expressions.py:121-190`
- Called from: `_jitextension.py` via deprecated `unwrap()` alias
- Purpose: Strip ALL metadata for C code generation

**Evaluate Path:**
- Function: `unwrap_for_evaluate(expr, scaling_active)`
- Location: `expressions.py:295-409`
- Called from: `functions_unit_system.py:108`
- Purpose: Prepare expression for lambdify while tracking dimensionality

### 2. Return Signature

**JIT Compilation:**
```python
return result  # Just the unwrapped SymPy expression
```

**Evaluate:**
```python
return sym_expr, result_dimensionality  # Tuple: (expr, dims_dict)
```

The evaluate path tracks `result_dimensionality` so it can re-dimensionalize results after evaluation.

### 3. Constant (UWQuantity) Handling

**Both pathways handle this identically:**

```python
# Check if scaling active
if uw._is_scaling_active() and fn.has_units:
    nondim = uw.non_dimensionalise(fn)
    # Return non-dimensionalized value

# Otherwise return plain value
```

✅ **No difference** - Both non-dimensionalize constants when scaling is active

### 4. Variable Symbol Handling

**JIT Compilation:**
```python
# Step 1: Call _unwrap_expressions (recursive substitution)
result = _unwrap_expressions(fn, keep_constants, return_self)

# Step 2: Apply scaling (NOW DISABLED - returns expr unchanged)
if uw._is_scaling_active():
    result = _apply_scaling_to_unwrapped(result)
```

**Evaluate:**
```python
# Get SymPy core (no recursive substitution)
if hasattr(expr, 'sym'):
    sym_expr = expr.sym
else:
    sym_expr = expr

# Look for atoms with _pint_qty and substitute them
for atom in sym_expr.atoms():
    if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
        nondim_atom = uw.non_dimensionalise(atom)
        substitutions[atom] = nondim_atom.value
```

✅ **After scaling fix**: Both leave variable symbols unchanged
❓ **Question**: Evaluate scans atoms for `_pint_qty`, JIT uses `_unwrap_expressions` - different recursion strategies

### 5. Recursion Strategy

**JIT Compilation (via `_unwrap_expressions`):**
```python
def _unwrap_expressions(fn, keep_constants, return_self):
    expr = fn
    expr_s = _substitute_all_once(expr, keep_constants, return_self)

    # Keep substituting until no more changes
    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_all_once(expr, keep_constants, return_self)

    return expr
```

Calls `_substitute_all_once` which:
- Uses `extract_expressions_and_functions(fn)` to find UWexpression atoms
- Checks each atom for `_pint_qty` (lines 65-80)
- Non-dimensionalizes if scaling active
- Recursively substitutes until fixed point

**Evaluate:**
```python
# Single pass over atoms
if should_scale and hasattr(sym_expr, 'atoms'):
    substitutions = {}

    for atom in sym_expr.atoms():
        if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
            nondim_atom = uw.non_dimensionalise(atom)
            substitutions[atom] = nondim_atom.value

    if substitutions:
        sym_expr = sym_expr.subs(substitutions)
```

Single pass using SymPy's `.atoms()` method.

❓ **Question**: Why different approaches? JIT uses recursive fixed-point iteration, evaluate uses single-pass atom scan.

### 6. Atom Detection

**JIT Path:**
```python
for atom in extract_expressions_and_functions(fn):
    if isinstance(atom, UWexpression):
        if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
            # Non-dimensionalize
```

Uses `extract_expressions_and_functions()` - a custom function that finds UWexpression atoms.

**Evaluate Path:**
```python
for atom in sym_expr.atoms():
    if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
        # Non-dimensionalize
```

Uses SymPy's built-in `.atoms()` method.

❓ **Question**: Should both use the same atom detection mechanism?

### 7. UWDerivativeExpression Handling

**JIT Compilation:**
```python
if isinstance(fn, UWDerivativeExpression):
    result = fn.doit()  # Evaluate derivative immediately
```

**Evaluate:**
- No special handling for UWDerivativeExpression
- Relies on `.sym` property extraction

⚠️ **Potential difference**: JIT evaluates derivatives, evaluate might not

### 8. Matrix Handling

**JIT Compilation:**
```python
if isinstance(fn, sympy.Matrix):
    f = lambda x: _unwrap_expressions(x, keep_constants, return_self)
    result = fn.applyfunc(f)  # Apply recursively to each element
```

**Evaluate:**
- No special matrix handling
- Assumes expr is already extracted to SymPy form

❓ **Question**: Does evaluate handle Matrix inputs correctly?

## Key Findings

### ✅ Similarities After Scaling Fix

1. **Constant non-dimensionalization**: Both paths non-dimensionalize UWQuantity constants when scaling is active
2. **Variable preservation**: Both now leave variable symbols like p(x,y) unchanged (after the fix)
3. **Scaling check**: Both use `uw._is_scaling_active()` or equivalent

### ❓ Differences That May Matter

1. **Return type**: Evaluate returns `(expr, dimensionality)` tuple, JIT returns just `expr`
   - **Impact**: Evaluate can re-dimensionalize results, JIT strips all unit info

2. **Recursion depth**: JIT uses fixed-point iteration, evaluate uses single pass
   - **Impact**: JIT might handle deeply nested expressions differently
   - **Question**: Are there cases where this matters?

3. **Atom detection**: JIT uses `extract_expressions_and_functions()`, evaluate uses `.atoms()`
   - **Impact**: Might find different atoms
   - **Question**: Do they find the same set of constants to non-dimensionalize?

4. **Derivative handling**: JIT evaluates derivatives with `.doit()`, evaluate doesn't
   - **Impact**: Might affect expressions containing derivatives
   - **Question**: Does this cause different behavior?

5. **Matrix handling**: JIT has special matrix recursion, evaluate doesn't
   - **Impact**: Might affect vector/tensor expressions
   - **Question**: Does evaluate handle these correctly?

## Recommendations for Further Investigation

1. **Test atom detection equivalence**:
   - Create expression with nested constants: `2 * uw.quantity(5, "Pa") + uw.quantity(3, "K")`
   - Check if both paths find and non-dimensionalize the same atoms

2. **Test derivative handling**:
   - Create expression with derivative: `T.sym.diff(mesh.N.x)`
   - Check if both paths handle it correctly

3. **Test matrix expressions**:
   - Create matrix expression with velocity field
   - Check if both paths unwrap matrix elements correctly

4. **Test recursion depth**:
   - Create deeply nested expression: `A * (B + (C * D))`
   - Check if fixed-point iteration vs single-pass makes a difference

5. **Verify `extract_expressions_and_functions()` vs `.atoms()`**:
   - Compare what atoms each method finds for same expression
   - Ensure they find the same constants that need non-dimensionalization

## Conclusion

After the scaling fix, both paths handle the **core concern** correctly:
- ✅ Constants are non-dimensionalized
- ✅ Variables are NOT scaled (left as ND symbols)

However, there are **architectural differences** in:
- Recursion strategy (fixed-point vs single-pass)
- Atom detection (custom vs SymPy built-in)
- Special case handling (derivatives, matrices)

These differences may or may not matter depending on the complexity of expressions being unwrapped.

**No changes recommended at this time** - just documenting the differences for awareness.
