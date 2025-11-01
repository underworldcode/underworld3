# Secondary Issues - Detailed Analysis
**Date**: 2025-10-26
**Status**: Identified but not caused by unwrap fix

---

## Issue 1: Scalar Subscript Error
### Severity: HIGH (blocks all solver compilation)
### Cause: Unknown - needs investigation

---

## Problem Description

When attempting to compile a Poisson solver, SymPy's simplification code fails:

```
TypeError: 'UWexpression' object (scalar) is not subscriptable
  File "sympy/core/exprtools.py", line 1228, in do
    return type(expr)([do(i) for i in expr])
                              ^^^^
  File "underworld3/utilities/mathematical_mixin.py", line 59, in __getitem__
    raise TypeError(f"'{type(self).__name__}' object (scalar) is not subscriptable")
```

### Stack Trace Analysis

```
poisson.solve()
  ↓
SolverBaseClass._build()
  ↓
SNES_Scalar._setup_pointwise_functions()
  ↓
sympy.simplify(self.constitutive_model.flux.T)
  ↓
sympy.simplify → cancel() → factor_terms()
  ↓
SymPy iteration: type(expr)([do(i) for i in expr])
  ↓
Tries to iterate over UWexpression (treating it like a sequence)
  ↓
__getitem__() called on scalar UWexpression
  ↓
MathematicalMixin.__getitem__ raises TypeError for scalars
```

### Root Cause Hypothesis

**SymPy's internal code assumes that Symbol subclasses are iterable or support indexing.**

UWexpression inherits from SymPy's Symbol class:
```python
class UWexpression(MathematicalMixin, UWQuantity, uw_object, Symbol):
```

When SymPy's factorization code processes expressions, it does:
```python
type(expr)([do(i) for i in expr])  # Tries to iterate!
```

For normal Symbols, this fails silently or works. But UWexpression's `__getitem__` explicitly raises an error for scalars.

### Code Location

**File**: `/src/underworld3/utilities/mathematical_mixin.py`, lines 53-86

```python
def __getitem__(self, index):
    """Component access with proper bounds checking."""
    sym = self._validate_sym()

    # Check for scalar variables (no indexing allowed)
    if not hasattr(sym, "shape") or not hasattr(sym, "__getitem__"):
        raise TypeError(f"'{type(self).__name__}' object (scalar) is not subscriptable")
```

### When Does It Occur?

1. When building Poisson solver
2. When `self.constitutive_model.flux.T` is accessed
3. When sympy.simplify() is called on the flux
4. When SymPy's cancel() → factor_terms() tries to factorize

### Why Wasn't This Caught Before?

Likely because:
1. The solver wasn't being compiled/used before
2. The test suite may not have run full Poisson tests
3. The unwrap fix may have changed code paths being executed

### Potential Fixes

#### Option 1: Allow Iteration Over Scalars (PERMISSIVE)
```python
def __iter__(self):
    """Allow SymPy to iterate (returns empty iterator for scalars)."""
    sym = self._validate_sym()
    if hasattr(sym, "__iter__"):
        return iter(sym)
    else:
        return iter([])  # Empty iterator for scalars
```

**Pros**: Fixes the immediate error
**Cons**: May mask real issues, allows iteration of scalars

#### Option 2: Override `__getitem__` More Carefully (DEFENSIVE)
```python
def __getitem__(self, index):
    """Component access with proper bounds checking."""
    sym = self._validate_sym()

    # Allow SymPy-style access even for scalars
    if not hasattr(sym, "shape"):
        # Scalar - delegate to symbol, don't raise
        return sym[index]  # SymPy will raise if inappropriate

    # ... rest of vector/matrix handling
```

**Pros**: Delegates to SymPy's error handling
**Cons**: May allow unexpected indexing

#### Option 3: Prevent simplification on UWexpressions (TARGETED)
```python
# Before simplification:
if isinstance(expr, UWexpression):
    return expr  # Don't simplify UWexpressions
expr = sympy.simplify(expr)
```

**Pros**: Avoids the problematic code path
**Cons**: Loses simplification benefits

#### Option 4: Don't Inherit From Symbol (ARCHITECTURAL)
Make UWexpression NOT inherit from Symbol, but wrap a Symbol instead:

```python
class UWexpression(MathematicalMixin, UWQuantity, uw_object):
    def __init__(self, name, sym=None, ...):
        self._symbol = Symbol(name)  # Wrap, don't inherit
        self.sym = sym

    def _sympify_(self):
        return self._symbol  # Return the wrapped symbol
```

**Pros**: Clean separation, no inheritance issues
**Cons**: Large refactoring, may break other things

---

## Issue 2: Scaling Application Not Working

### Severity: MEDIUM (affects ND scaling tests)
### Cause: Unclear - needs investigation

---

## Problem Description

When unwrap is called with ND scaling enabled, the scaling factor should be applied:

```python
uw.use_nondimensional_scaling(True)
model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

T = uw.MeshVariable("T", mesh, 1, units="kelvin")
T.set_reference_scale(1000.0)

# Expect: T unwraps to something containing scaling (1/1000 or 0.001)
unwrapped = uw.unwrap(T.sym)
# Actual: Returns T unchanged without scaling

# Test assertion:
assert "1000" in str(unwrapped) or "0.001" in str(unwrapped)
# FAILS: String is just "T(N.x, N.y)"
```

### Test Code Location

**File**: `tests/test_0816_global_nd_flag.py`
**Lines**: Around 130-135

```python
def test_unwrap_with_scaling():
    """Test that unwrap() DOES scale when flag is True."""
    uw.reset_default_model()
    uw.use_nondimensional_scaling(True)  # Enable scaling

    model = uw.get_default_model()
    model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

    # ... setup ...

    unwrapped = uw.unwrap(T.sym)
    unwrap_str = str(unwrapped_scalar)
    assert (
        "1000" in unwrap_str or "0.001" in unwrap_str
    ), f"Unwrapped expression should contain scaling factor when flag=True: {unwrap_str}"
```

### Current Behavior

```
Expected: Matrix([[... * 0.001 * T(N.x, N.y) ...]])
Actual:   Matrix([[T(N.x, N.y)]])
```

### Root Cause Hypothesis

**The scaling context is not being applied during unwrap.**

Looking at `unwrap()` function in expressions.py (lines 94-100):

```python
# Apply scaling if context is active
import underworld3 as uw

if uw._is_scaling_active():
    result = _apply_scaling_to_unwrapped(result)

return result
```

And `_apply_scaling_to_unwrapped()` (lines 103-146):

```python
def _apply_scaling_to_unwrapped(expr):
    """Apply non-dimensional scaling to an unwrapped SymPy expression."""
    # ... lots of explanation ...
    return expr  # Return expression unchanged - PETSc handles all scaling
```

**The function returns the expression UNCHANGED!**

### Why Was This Designed This Way?

The docstring explains (lines 110-134):

```python
"""
IMPORTANT: This function does NOT scale MeshVariable symbols because PETSc
already stores non-dimensional values when ND scaling is active.

CRITICAL INSIGHT - Derivative Scaling:
======================================
Spatial derivatives do NOT need explicit scaling coefficients in unwrap!

Why? PETSc automatically handles derivative scaling:
- PETSc stores: v̂ = v/V₀ (non-dimensional velocity)
- PETSc computes derivatives in PHYSICAL coordinates (x, y in meters)
- Result: ∂v̂/∂x = ∂(v/V₀)/∂x = (1/V₀)∂v/∂x
...

Therefore, NO variable or derivative scaling needed in unwrap!
"""
```

### The Design Philosophy

The scaling happens at THREE layers:
1. **PETSc storage**: `var.data = var_dimensional / scaling_coefficient`
2. **User setters**: UWQuantity wrappers for source terms, BCs, etc.
3. **Array property**: `var.array` auto-scales for user-facing dimensional access

So the idea is: **unwrap should NOT scale** because PETSc handles it.

### Why The Test Fails

**The test expects scaling to appear in unwrapped expressions**, but the implementation intentionally doesn't add it.

This could mean:
1. **The test is wrong**: It expects something that shouldn't happen
2. **The implementation is wrong**: It should scale but doesn't
3. **The design changed**: Scaling used to be applied but was removed
4. **The test setup is wrong**: Scaling isn't actually enabled

### What Should Actually Happen?

Looking at the comment in `_apply_scaling_to_unwrapped()`:

> "Scaling happens at three separate layers... PETSc handles all scaling... NO variable or derivative scaling needed in unwrap!"

This suggests:
- **Variables like T should NOT be scaled in unwrap**
- **PETSc already stores them non-dimensional**
- **The test expectation may be incorrect**

### Potential Issues

#### Problem A: Test Is Testing Wrong Thing
The test expects:
```python
assert "1000" in unwrap_str or "0.001" in unwrap_str
```

But according to the design, unwrap should NOT scale. So the test should be:
```python
assert "1000" not in unwrap_str  # No scaling applied
assert "T" in unwrap_str          # Just the variable
```

#### Problem B: Scaling Not Being Applied Where It Should
If scaling IS supposed to be applied in some cases, but isn't:
- Check `_apply_scaling_to_unwrapped()` implementation
- Verify `uw._is_scaling_active()` works correctly
- Check if MeshVariable reference scales are being set

#### Problem C: Integration With Reference Scales
When `T.set_reference_scale(1000.0)` is called:
- Is this actually stored?
- Is it used during unwrap?
- Is it propagated to the scaling context?

### Code Location

**File**: `/src/underworld3/function/expressions.py`

**Function 1** (lines 94-100): Check if scaling applied in unwrap
```python
if uw._is_scaling_active():
    result = _apply_scaling_to_unwrapped(result)
```

**Function 2** (lines 103-146): Apply scaling (currently returns unchanged)
```python
def _apply_scaling_to_unwrapped(expr):
    """Apply non-dimensional scaling..."""
    return expr  # Unchanged!
```

### Related Code

**File**: `/src/underworld3/discretisation/discretisation_mesh_variables.py`
- `set_reference_scale()` method - sets scaling factor

**File**: `/src/underworld3/__init__.py`
- `use_nondimensional_scaling()` function
- `_is_scaling_active()` function
- `get_default_model()` and scaling context

### Potential Fixes

#### Option 1: Fix The Test (if design is correct)
```python
def test_unwrap_with_scaling():
    """Test that unwrap() correctly handles ND scaling context."""
    # ...setup...

    unwrapped = uw.unwrap(T.sym)
    unwrap_str = str(unwrapped_scalar)

    # PETSc handles scaling at storage level, not unwrap level
    # So unwrapped expression should just have T, no scaling factor
    assert "T" in unwrap_str, f"Expected variable symbol: {unwrap_str}"
    assert "1000" not in unwrap_str, f"Should not have scaling in unwrap: {unwrap_str}"
```

#### Option 2: Implement Scaling In Unwrap (if needed)
```python
def _apply_scaling_to_unwrapped(expr):
    """Apply non-dimensional scaling to unwrapped expression."""
    # Extract scaling factors for each variable
    # Apply them to the expression
    # Return scaled expression
    pass  # Needs implementation
```

#### Option 3: Investigate Actual Scaling Mechanism
- Verify `set_reference_scale()` is working
- Check if scaling context is being set correctly
- Trace through what `_is_scaling_active()` returns
- Understand when scaling SHOULD vs SHOULDN'T be applied

---

## Summary Table

| Issue | Severity | Cause | Impact | Fix Complexity |
|-------|----------|-------|--------|----------------|
| Scalar Subscript | HIGH | SymPy iteration over UWexpression | Blocks all solvers | MEDIUM |
| Scaling Application | MEDIUM | Design vs test mismatch | ND scaling tests fail | LOW-MEDIUM |

---

## Investigation Priorities

### Priority 1: Scalar Subscript (Blocks Everything)
Must fix before solvers can work at all.
1. Determine if it's pre-existing or new
2. Decide on fix strategy (Options 1-4)
3. Implement and test
4. Verify all solvers work

### Priority 2: Scaling Application (ND Scaling)
Affects non-dimensional scaling tests.
1. Verify the design intent (is scaling SUPPOSED to be applied?)
2. Check if test expectation is correct
3. Either fix test or fix implementation
4. Verify ND scaling works correctly

### Priority 3: Root Cause Analysis
Why weren't these caught earlier?
1. Were tests not being run?
2. Did recent changes expose these?
3. Are they pre-existing?

---

## Testing Strategy

### For Scalar Subscript
```bash
# Create minimal solver test
python -c "
import underworld3 as uw

# Try to create and solve simplest possible system
mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
u = uw.discretisation.MeshVariable('u', mesh, 1, degree=2)

# Try Poisson
poisson = uw.systems.Poisson(mesh, u_Field=u)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0
poisson.add_dirichlet_bc(1.0, 'Bottom')
poisson.add_dirichlet_bc(0.0, 'Top')

try:
    poisson.solve()
    print('✓ Poisson works')
except TypeError as e:
    if 'subscriptable' in str(e):
        print('✗ Scalar subscript error confirmed')
    else:
        print(f'✗ Different error: {e}')
except Exception as e:
    print(f'✗ Other error: {type(e).__name__}: {e}')
"
```

### For Scaling Application
```bash
# Test ND scaling with detailed output
python -c "
import underworld3 as uw

uw.use_nondimensional_scaling(True)
model = uw.get_default_model()
model.set_reference_quantities(temperature_diff=uw.quantity(1000, 'kelvin'))

print(f'Scaling active: {uw._is_scaling_active()}')

mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
T.set_reference_scale(1000.0)

print(f'T.sym: {T.sym}')

unwrapped = uw.function.fn_unwrap(T.sym)
print(f'Unwrapped: {unwrapped}')
print(f'Contains scaling: {\"1000\" in str(unwrapped) or \"0.001\" in str(unwrapped)}')
"
```

---

## Recommendations

1. **Do NOT assume these are caused by unwrap fix** - They may be pre-existing
2. **Investigate separately** - Each issue needs independent analysis
3. **Determine scope** - Run tests to see how many failures are related
4. **Prioritize by impact** - Scalar subscript blocks everything, scale issue is specific
5. **Document findings** - Whatever you discover, document thoroughly

