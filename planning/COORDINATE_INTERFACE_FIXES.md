# Coordinate Interface Regression Fixes

**Date**: 2025-01-11
**Status**: ✅ FIXED - All tests passing

## Problem

After implementing `mesh.X` as a `CoordinateSystem` object (to add `.coords` and `.units` properties), we broke existing code that expected `mesh.X` to behave like a SymPy Matrix for mathematical operations.

### Test Failures

1. **test_0601_mesh_vector_calc.py** - Collection error
   - Line 326: `v31.sym.diff(v31.mesh.X)` failed
   - SymPy's `diff` method couldn't handle CoordinateSystem object

2. **test_1011_stokesSph.py** - 3 failures
   - Line 71: `unit_rvec = mesh.X / (radius_fn)` failed
   - Division operation not supported on CoordinateSystem

## Root Cause

When we changed `mesh.X` to return a `CoordinateSystem` object instead of a SymPy `Matrix`, existing code that performed mathematical operations on `mesh.X` broke because:

1. SymPy couldn't sympify the CoordinateSystem object
2. Arithmetic operations weren't defined
3. SymPy type-checking attributes were missing

## Solution

Made `CoordinateSystem` act as a transparent wrapper around the symbolic matrix `_X` for all mathematical operations while still providing the new `.coords` and `.units` properties.

### Changes to `coordinates.py`

**1. Constructor Guard** (lines 67-73)
```python
# Prevent SymPy from trying to construct CoordinateSystem from list
if isinstance(mesh, (list, tuple)) or not hasattr(mesh, 'r'):
    raise TypeError(...)
```

**2. SymPy Integration** (lines 402-413)
```python
def _sympify_(self):
    """Tell SymPy how to convert this object."""
    return self._X

def __sympy__(self):
    """Alternative SymPy conversion protocol."""
    return self._X
```

**3. SymPy Type Checking** (lines 415-441)
```python
@property
def is_symbol(self): return False

@property
def is_Matrix(self): return True

@property
def is_scalar(self): return False

@property
def is_number(self): return False

@property
def is_commutative(self): return self._X.is_commutative
```

**4. Attribute Delegation** (lines 443-457)
```python
def __getattr__(self, name):
    """Delegate SymPy-specific attributes to _X."""
    return getattr(self._X, name)
```

**5. Arithmetic Operations** (lines 459-494)
```python
def __add__(self, other): return self._X + other
def __radd__(self, other): return other + self._X
def __sub__(self, other): return self._X - other
def __rsub__(self, other): return other - self._X
def __mul__(self, other): return self._X * other
def __rmul__(self, other): return other * self._X
def __truediv__(self, other): return self._X / other
def __rtruediv__(self, other): return other / self._X
def __pow__(self, other): return self._X ** other
def __neg__(self): return -self._X
```

**6. Shape Property** (lines 396-398)
```python
@property
def shape(self):
    """Shape of the symbolic coordinate matrix."""
    return self._X.shape
```

### Key Design Insights

1. **Dual Nature**: CoordinateSystem acts as both:
   - A data container (`mesh.X.coords`, `mesh.X.units`)
   - A symbolic matrix (`mesh.X[0]`, `x, y = mesh.X`, `mesh.X / scalar`)

2. **Transparent Wrapper**: All mathematical operations delegate to `_X`, so existing code works unchanged

3. **SymPy Integration**: Multiple protocols (`_sympify_`, `__sympy__`, `__getattr__`) ensure SymPy can use CoordinateSystem transparently

4. **Type Checking**: SymPy checks many `is_*` attributes to determine how to handle objects - we provide appropriate values

## Testing Results

### Before Fixes
```
ERROR tests/test_0601_mesh_vector_calc.py - AttributeError
FAILED tests/test_1011_stokesSph.py::test_stokes_sphere[mesh0] - TypeError
FAILED tests/test_1011_stokesSph.py::test_stokes_sphere[mesh1] - TypeError
FAILED tests/test_1011_stokesSph.py::test_stokes_sphere[mesh2] - TypeError
```

### After Fixes
```
tests/test_0601_mesh_vector_calc.py: 28 passed, 1 xpassed ✅
tests/test_1011_stokesSph.py: 3 passed ✅
```

## Backward Compatibility Verified

All existing patterns continue to work:

```python
# Symbolic coordinate access (unchanged)
x, y = mesh.X
x_coord = mesh.X[0]

# Mathematical operations (unchanged)
unit_rvec = mesh.X / radius_fn
expr = mesh.X[0]**2 + mesh.X[1]**2

# SymPy operations (unchanged)
grad_p = p.sym.diff(mesh.X[0])
jac_v = v.sym.jacobian(mesh.X)

# NEW: Data and units access
coords = mesh.X.coords
units = mesh.X.units
```

## Lessons Learned

### Critical Rebuild Requirement
After modifying source files, ALWAYS run `pixi run underworld-build`! Underworld3 is installed as a package, so changes don't take effect until rebuilt.

### Wrapping SymPy Objects
When creating a wrapper around SymPy objects:
1. Implement `_sympify_()` and `__sympy__()`
2. Add SymPy type-checking properties (`is_symbol`, `is_Matrix`, etc.)
3. Delegate arithmetic operations to the wrapped object
4. Use `__getattr__` to delegate attribute access
5. Provide a guard in `__init__` to prevent SymPy reconstruction

### Test Early, Test Often
The tests caught these regressions immediately during collection (not even execution), showing how critical comprehensive test coverage is.

## Related Documents

- `COORDINATE_INTERFACE_DESIGN.md` - Original design rationale
- `COORDINATE_INTERFACE_STATUS.md` - Implementation status
- `COORDINATE_ACCESS_AUDIT.md` - Codebase audit
- `COORDINATE_MIGRATION_GUIDE.md` - Migration guide for users

## Next Steps

Now that backward compatibility is ensured:
1. ✅ Tests pass - regression fixed
2. ⏭️ Proceed with gradual migration (tutorials, examples)
3. ⏭️ Update documentation to show new interface
4. ⏭️ Eventually deprecate `mesh.data`

No urgent action needed - all existing code works correctly.
