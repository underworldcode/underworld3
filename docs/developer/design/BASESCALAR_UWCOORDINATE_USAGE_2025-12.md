# UWCoordinate Architecture (December 2025)

## Overview

This document describes the UWCoordinate implementation in Underworld3, which provides user-friendly coordinate objects (`mesh.X`) that work transparently with SymPy's differentiation and JIT code generation.

## Architecture Summary

**Key Design Decision (December 2025)**: UWCoordinate now **subclasses `BaseScalar`** instead of `Symbol`, with custom `__eq__` and `__hash__` methods that match the original N.x/N.y/N.z objects. This makes `sympy.diff()` work transparently with mesh coordinates.

### Coordinate Representations

1. **`mesh.N`** → SymPy `CoordSys3D` object
   - Raw SymPy vector coordinate system
   - Used for vector operations, basis vectors, `matrix_to_vector`

2. **`mesh.CoordinateSystem.N`** → `sympy.Matrix` of raw BaseScalars
   - Used internally for `.jacobian()` operations
   - Contains the original N.x, N.y, N.z objects

3. **`mesh.CoordinateSystem.X`** (or `mesh.X`) → `sympy.Matrix` of UWCoordinates
   - **User-facing coordinates** for expressions
   - Works transparently with `sympy.diff()` ✅
   - Works with JIT code generation ✅
   - Recognized by type in `unwrap_for_evaluate`

## UWCoordinate Implementation

```python
from sympy.vector.scalar import BaseScalar

class UWCoordinate(BaseScalar):
    """
    A Cartesian coordinate variable (x, y, or z).
    Subclasses BaseScalar for full SymPy differentiation compatibility.
    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None, mesh=None, axis_index=None):
        obj = BaseScalar.__new__(cls, index, system, pretty_str, latex_str)
        return obj

    def __init__(self, index, system, pretty_str=None, latex_str=None, mesh=None, axis_index=None):
        self._mesh = mesh
        self._axis_index = axis_index if axis_index is not None else index
        self._coord_name = pretty_str or f"x_{index}"
        # Cache the original BaseScalar for equality comparison
        self._original_base_scalar = system.base_scalars()[index]

    def __eq__(self, other):
        """Equal to the original BaseScalar (N.x, N.y, N.z)."""
        if other is self._original_base_scalar:
            return True
        if isinstance(other, UWCoordinate) and hasattr(other, '_original_base_scalar'):
            if other._original_base_scalar is self._original_base_scalar:
                return True
        return BaseScalar.__eq__(self, other)

    def __hash__(self):
        """Hash same as the original BaseScalar."""
        return hash(self._original_base_scalar)

    @property
    def _ccodestr(self):
        """Delegate C code string to the original BaseScalar."""
        return self._original_base_scalar._ccodestr

    def _ccode(self, printer, **kwargs):
        """C code representation for JIT compilation."""
        return self._ccodestr
```

### Key Features

1. **`__eq__` and `__hash__`**: Match the original N.x/N.y/N.z objects
   - This is what makes `sympy.diff()` work!
   - SymPy uses identity checking for differentiation variables
   - By matching equality/hash, UWCoordinate is "seen" as the same as the original

2. **`_ccodestr` property**: Delegates to original BaseScalar
   - The mesh sets `_ccodestr` on N.x, N.y, N.z for JIT compilation
   - UWCoordinate exposes the same attribute via property delegation

3. **`_ccode()` method**: Returns `_ccodestr` for SymPy's C code printer
   - SymPy's CCodePrinter looks for this method on symbols
   - Required for JIT code generation to work

## Usage Examples

### Differentiation (Now Works Directly!)

```python
import sympy
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(...)
x, y = mesh.X  # UWCoordinates

v = uw.discretisation.MeshVariable('V', mesh, num_components=2, degree=2)

# Direct sympy.diff NOW WORKS (December 2025):
dv_dx = sympy.diff(v.sym[0], x)  # Returns V_{0,0}(N.x, N.y)
dv_dy = sympy.diff(v.sym[0], y)  # Returns V_{0,1}(N.x, N.y)

# Also works with expressions built from coordinates:
r_squared = x**2 + y**2
dr2_dx = sympy.diff(r_squared, x)  # Returns 2*N.x
```

### JIT Code Generation (Works Transparently)

```python
# Expressions with UWCoordinates compile correctly:
result = uw.function.evaluate(x**2 + y**2, mesh.X.coords)
# JIT generates: pow(petsc_x[0], 2) + pow(petsc_x[1], 2)
```

### For Internal Code

When maximum clarity is needed (e.g., in solver code), you can still use raw BaseScalars:

```python
# Internal code pattern - use BaseScalars directly
N_x, N_y = mesh.CoordinateSystem.N[0], mesh.CoordinateSystem.N[1]
jacobian = velocity.sym.jacobian(mesh.CoordinateSystem.N)
```

## Deprecated Functions

### `uw.uwdiff()` - DEPRECATED

Since `sympy.diff()` now works directly with UWCoordinates, `uwdiff()` is deprecated:

```python
# OLD (deprecated):
result = uw.uwdiff(v.sym[0], y)  # Still works but emits DeprecationWarning

# NEW (preferred):
result = sympy.diff(v.sym[0], y)  # Works directly!
```

### `uw.function.derivative()` - Simplified

The `uw.function.derivative()` function has been simplified - it no longer needs to convert UWCoordinates since `sympy.diff()` handles them natively. It's still useful for handling UWexpression unwrapping.

## Historical Context

### Previous Architecture (Before December 2025)

UWCoordinate previously subclassed `sympy.Symbol`, which caused several issues:

1. **Differentiation failed**: `sympy.diff(expr, y)` returned 0 because `UWCoordinate('y')` was a different symbol than `N.y` in the expression

2. **Required workarounds**: Users had to use `uw.uwdiff()` or `uw.function.derivative()` instead of standard `sympy.diff()`

3. **Symbol identity issues**: SymPy's Symbol class relies on object identity, making it difficult to make UWCoordinate "look like" the original BaseScalar

### Solution: BaseScalar Subclass

The solution was suggested by Louis: subclass `BaseScalar` instead of `Symbol`, and implement `__eq__`/`__hash__` to match the original. This elegant approach:

- Makes UWCoordinate a first-class citizen in SymPy's type hierarchy
- Allows transparent differentiation
- Avoids monkey-patching SymPy internals
- Maintains backward compatibility

## Technical Details

### Why `__eq__` and `__hash__` Work

SymPy's differentiation checks if the differentiation variable appears in the expression using set membership and equality. By making:
- `UWCoordinate.__eq__(N.x)` return `True`
- `UWCoordinate.__hash__()` return `hash(N.x)`

The UWCoordinate is found in the expression's free symbols and differentiation proceeds correctly.

### JIT Code Path

1. Mesh initialization sets `_ccodestr` on raw BaseScalars:
   ```python
   self._N.x._ccodestr = "petsc_x[0]"
   ```

2. JIT extension sets `_ccode` method on BaseScalar type

3. When UWCoordinate appears in expression, its `_ccode()` method is called, which returns `_ccodestr` via property delegation

### Locations Still Using Raw BaseScalars

Some internal code continues to use `mesh.CoordinateSystem.N` directly:

- `constitutive_models.py` - Velocity gradient jacobians
- `petsc_generic_snes_solvers.pyx` - Solver assembly
- `maths/vector_calculus.py` - Strain tensors, gradients
- `systems/solvers.py` - Assembly operations

This is fine - both UWCoordinates and raw BaseScalars work interchangeably for differentiation now.

## References

- `coordinates.py` - UWCoordinate implementation
- `function/__init__.py` - `derivative()` function
- `discretisation_mesh.py` - Mesh coordinate setup
- `utilities/_jitextension.py` - JIT code generation
