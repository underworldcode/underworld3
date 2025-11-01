# Secondary Issues - Quick Reference Guide
**For Quick Lookup and Debugging**

---

## Issue 1: Scalar Subscript Error

### Quick Facts
- **Error**: `TypeError: 'UWexpression' object (scalar) is not subscriptable`
- **Happens**: When solving Poisson or other systems
- **Triggered by**: SymPy's `simplify()` → `cancel()` → `factor_terms()`
- **Root**: SymPy tries to iterate over UWexpression as if it's iterable

### Stack Trace Pattern
```
poisson.solve()
  └─> SolverBaseClass._build()
        └─> _setup_pointwise_functions()
              └─> sympy.simplify(self.constitutive_model.flux.T)
                    └─> cancel() → factor_terms()
                          └─> type(expr)([do(i) for i in expr])  ← Tries to iterate!
                                └─> UWexpression.__getitem__()
                                      └─> Raises TypeError for scalars
```

### Where It Happens
**File**: `/src/underworld3/utilities/mathematical_mixin.py`, lines 53-86

```python
def __getitem__(self, index):
    if not hasattr(sym, "shape") or not hasattr(sym, "__getitem__"):
        raise TypeError(f"... is not subscriptable")  # ← This line!
```

### Quick Diagnostic
```bash
# Test if you hit this issue
pixi run -e default python -c "
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
    print(f'✗ Scalar subscript issue: {e}')
"
```

### 4 Proposed Fixes (Ranked by Preference)

#### Fix A: Override `__iter__()` (SIMPLEST) ⭐
```python
def __iter__(self):
    """Let SymPy iterate safely."""
    sym = self._validate_sym()
    if hasattr(sym, "__iter__"):
        return iter(sym)
    else:
        return iter([])
```
- **Pros**: One method, minimal change
- **Cons**: Allows iteration of scalars (unusual)
- **Effort**: 5 minutes

#### Fix B: Safer `__getitem__()` (DEFENSIVE)
```python
def __getitem__(self, index):
    sym = self._validate_sym()
    if not hasattr(sym, "shape"):
        # Don't raise, let SymPy decide
        return sym[index]
    # ... rest of code
```
- **Pros**: Delegates to SymPy
- **Cons**: Allows unexpected indexing
- **Effort**: 10 minutes

#### Fix C: Skip Simplification (TARGETED)
```python
# In solver setup:
if isinstance(expr, UWexpression):
    return expr
result = sympy.simplify(expr)
```
- **Pros**: Avoids problematic code path
- **Cons**: Loses simplification
- **Effort**: 20 minutes

#### Fix D: Don't Inherit From Symbol (ARCHITECTURAL)
```python
class UWexpression(...):
    def __init__(self, ...):
        self._symbol = Symbol(name)  # Wrap, don't inherit
```
- **Pros**: Clean design
- **Cons**: Large refactoring
- **Effort**: 2-3 hours

---

## Issue 2: Scaling Not Applied in Unwrap

### Quick Facts
- **Error**: Test expects scaling factors in unwrapped expressions
- **Actual**: Unwrap returns expressions without scaling
- **Location**: `test_0816_global_nd_flag.py::test_unwrap_with_scaling`
- **Question**: Is the test wrong or is the implementation wrong?

### What The Code Does
**File**: `/src/underworld3/function/expressions.py`, lines 103-146

```python
def _apply_scaling_to_unwrapped(expr):
    """Apply non-dimensional scaling..."""
    # ... lots of explanation ...
    return expr  # Returns UNCHANGED! ← This is intentional
```

### Design Intent (From Comments)
```
"Scaling happens at THREE layers:
1. PETSc storage layer (var.data = var / scale)
2. User setter layer (UWQuantity wrappers)
3. Array property layer (var.array auto-scales)

Therefore: NO variable or derivative scaling needed in unwrap!"
```

### The Test Expects
```python
unwrapped = uw.unwrap(T.sym)
assert "1000" in str(unwrapped) or "0.001" in str(unwrapped)
# Expects scaling factor to appear
```

### What Actually Happens
```python
uw.use_nondimensional_scaling(True)
model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))
T.set_reference_scale(1000.0)

unwrapped = uw.unwrap(T.sym)
# Returns: T(N.x, N.y) without any 1000 or 0.001
```

### Quick Diagnostic
```bash
# Check if this affects you
pixi run -e default pytest tests/test_0816_global_nd_flag.py::test_unwrap_with_scaling -xvs
```

### Root Cause Analysis
Need to determine:

1. **Is the test correct?**
   ```python
   # Does the test make sense?
   # If PETSc handles scaling at storage level,
   # then unwrap shouldn't scale
   ```

2. **Is the design correct?**
   ```python
   # The design doc says no scaling in unwrap
   # But test expects scaling
   # Who's right?
   ```

3. **Was scaling ever implemented?**
   ```bash
   git log -p src/underworld3/function/expressions.py | grep -A5 -B5 "return expr"
   # Check if it used to scale but was removed
   ```

### 3 Possible Resolutions

#### Resolution A: Test Is Wrong (MOST LIKELY)
Fix the test to match the design:
```python
def test_unwrap_with_scaling():
    """Test that unwrap() works with ND scaling context."""
    # ...setup...
    unwrapped = uw.unwrap(T.sym)
    # PETSc handles scaling at storage level
    # So unwrapped should just have the variable, no scale factor
    assert "T" in str(unwrapped)
    assert "1000" not in str(unwrapped)  # No scaling in unwrap
```
- **Effort**: 5 minutes
- **Confidence**: HIGH (matches design doc)

#### Resolution B: Implementation Is Wrong
Implement scaling in unwrap:
```python
def _apply_scaling_to_unwrapped(expr):
    """Actually apply scaling..."""
    # Extract scales for each variable
    # Multiply into expression
    return scaled_expr
```
- **Effort**: 1-2 hours
- **Confidence**: LOW (contradicts design doc)

#### Resolution C: Both Need Investigation
```bash
# Check history
git log --oneline -20 tests/test_0816_global_nd_flag.py
# Check design changes
git log --oneline -20 src/underworld3/function/expressions.py
# Run both old and new code paths
```
- **Effort**: 2-3 hours
- **Confidence**: MEDIUM

---

## Testing Checklist

### For Scalar Subscript Issue
- [ ] Can create a simple Poisson system?
- [ ] Does it compile/solve?
- [ ] Does the error occur during `simplify()`?
- [ ] Is it in `factor_terms()` specifically?
- [ ] Does it happen for all scalar UWexpressions?
- [ ] Does it happen for vector/matrix UWexpressions?

### For Scaling Issue
- [ ] Does `_is_scaling_active()` return True?
- [ ] Is `set_reference_scale()` storing the value?
- [ ] Is the scale accessible in unwrap context?
- [ ] What does the design doc really intend?
- [ ] Have test expectations changed?
- [ ] Does ND scaling work elsewhere?

---

## Debugging Tips

### For Scalar Subscript
```python
# Add debug output
import underworld3 as uw

# Patch simplify to see where it fails
original_simplify = uw.sympy.simplify

def debug_simplify(expr, *args, **kwargs):
    print(f"Simplifying: {type(expr).__name__} = {expr}")
    try:
        return original_simplify(expr, *args, **kwargs)
    except TypeError as e:
        print(f"ERROR: {e}")
        print(f"Expression: {expr}")
        raise

uw.sympy.simplify = debug_simplify

# Now run your test
poisson.solve()
```

### For Scaling Issue
```python
# Check what's happening
import underworld3 as uw

uw.use_nondimensional_scaling(True)
model = uw.get_default_model()
model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

print(f"Scaling active: {uw._is_scaling_active()}")
print(f"Model: {model}")
print(f"Model scaling: {getattr(model, '_scaling_context', 'Not found')}")

# Check reference scales
T = uw.discretisation.MeshVariable('T', mesh, 1, units='kelvin')
T.set_reference_scale(1000.0)

print(f"T reference scale: {getattr(T, '_reference_scale', 'Not found')}")

# Unwrap and check
unwrapped = uw.function.fn_unwrap(T.sym)
print(f"Unwrapped: {unwrapped}")
```

---

## Known Workarounds

### For Scalar Subscript
**Until fixed**, try avoiding `simplify()`:
```python
# Instead of letting solver simplify, pass pre-simplified expressions
poisson.f = 0  # Simple source, no simplification needed
```

### For Scaling
**Until resolved**, manually apply scaling if needed:
```python
# If you need scaling in unwrapped expressions
scale_factor = 1.0 / 1000.0
unwrapped = scale_factor * uw.function.fn_unwrap(T.sym)
```

---

## Related Code Locations

### Scalar Subscript Related
- `/src/underworld3/utilities/mathematical_mixin.py` (lines 53-86) - `__getitem__`
- `/src/underworld3/function/expressions.py` (lines 215-?) - `UWexpression` definition
- PETSc solver setup code - where simplify is called

### Scaling Related
- `/src/underworld3/function/expressions.py` (lines 75-146) - `unwrap()` and `_apply_scaling_to_unwrapped()`
- `/src/underworld3/__init__.py` - `use_nondimensional_scaling()`, `_is_scaling_active()`
- `/src/underworld3/discretisation/discretisation_mesh_variables.py` - `set_reference_scale()`
- `/tests/test_0816_global_nd_flag.py` - The test itself

---

## Decision Tree

### If Poisson Solve Fails:

```
Does error contain "subscriptable"?
├─ YES → Scalar subscript issue
│   ├─ Apply Fix A (Override __iter__) for quick fix
│   └─ Consider Fix B-D for better solution
│
└─ NO → Other error
    ├─ Check what the error is
    └─ Might be scaling issue or something else
```

### If ND Scaling Tests Fail:

```
Does "1000" or "0.001" appear in unwrapped?
├─ YES → Scaling is working, test might be checking wrong thing
│   └─ Verify test expectations
│
└─ NO → Scaling not applied
    ├─ Is design supposed to scale? Check comments
    ├─ If yes → Fix implementation (Resolution B)
    └─ If no → Fix test (Resolution A)
```

---

## Next Steps Priority

1. **FIRST**: Run the diagnostic tests (5 minutes)
2. **SECOND**: Determine which fixes apply (15 minutes)
3. **THIRD**: Implement simplest fix first (30 minutes)
4. **FOURTH**: Test comprehensively (1 hour)
5. **FIFTH**: Document findings (30 minutes)