# Template Expression Pattern in Underworld3 Solvers

## Summary

The **Template Expression Pattern** is a design pattern used in Underworld3 solvers to create persistent expression containers that preserve object identity for lazy evaluation while allowing their symbolic content to be updated dynamically.

## Problem Solved

Previously, solver properties like `F0`, `F1`, and `PF0` created NEW expression objects every time they were accessed:

```python
# OLD PATTERN (problematic)
@property
def F0(self):
    f0 = expression(  # Creates NEW expression each time!
        r"f_0 \left( \mathbf{u} \right)",
        -self.bodyforce.sym,
        "Force term"
    )
    return f0
```

This caused:
1. **Uniqueness warnings** - Each access created a duplicate named expression
2. **Object identity loss** - Python `id(solver.F0)` changed on each access
3. **Lazy evaluation issues** - Templates couldn't reliably reference sub-expressions
4. **Memory inefficiency** - Accumulating unused expression objects

## Solution: ExpressionProperty Descriptor

The `ExpressionProperty` descriptor creates expression containers ONCE and preserves their identity:

```python
# NEW PATTERN (correct)
class MySolver:
    F0 = ExpressionProperty(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: -self.bodyforce.sym,
        "Force term"
    )
```

## How It Works

1. **First Access**: Creates a persistent `UWexpression` container
2. **Subsequent Accesses**: Returns the SAME container (same Python id)
3. **Content Updates**: Only the `.sym` property changes when referenced values change
4. **Lazy Evaluation**: Templates can reliably reference sub-expressions

## Implementation Details

### ExpressionProperty Class

Located in `src/underworld3/utilities/_api_tools.py`:

```python
class ExpressionProperty:
    """
    Property descriptor for persistent UWexpression template containers.

    Parameters
    ----------
    name_template : str or callable
        LaTeX name for the expression
    sym_template : callable
        Function that returns the symbolic expression
    description : str
        Description of the expression
    """

    def __init__(self, name_template, sym_template, description, attr_name=None):
        self.name_template = name_template
        self.sym_template = sym_template
        self.description = description
        self.attr_name = attr_name

    def __get__(self, obj, objtype=None):
        # Check if expression already exists
        expr = getattr(obj, self.attr_name, None)

        if expr is None:
            # Create the expression ONCE
            expr = expression(
                name,
                self.sym_template(obj),
                self.description,
                _unique_name_generation=True
            )
            setattr(obj, self.attr_name, expr)
        else:
            # Update content if needed
            new_sym = self.sym_template(obj)
            if expr.sym != new_sym:
                expr.sym = new_sym

        return expr
```

### Usage in Solvers

All major solvers now use `ExpressionProperty`:

```python
class SNES_Stokes(SNES_Stokes_SaddlePt):
    # Template expressions with persistent identity
    F0 = ExpressionProperty(
        r"\mathbf{f}_0\left( \mathbf{u} \right)",
        lambda self: -self.bodyforce.sym,
        "Stokes pointwise force term"
    )

    F1 = ExpressionProperty(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: sympy.simplify(
            self.stress + self.penalty * self.div_u * sympy.eye(self.mesh.dim)
        ),
        "Stokes pointwise flux term"
    )

    PF0 = ExpressionProperty(
        r"\mathbf{h}_0\left( \mathbf{p} \right)",
        lambda self: sympy.simplify(sympy.Matrix((self.constraints))),
        "Pressure constraint term"
    )
```

## Benefits

1. **No Uniqueness Warnings**: Expressions created once, not repeatedly
2. **Preserved Identity**: `id(solver.F0)` remains constant
3. **Lazy Evaluation**: Templates can reliably reference sub-expressions
4. **Memory Efficient**: No accumulation of unused expressions
5. **Clean Syntax**: Declarative pattern at class level
6. **Automatic Updates**: Content updates when dependencies change

## Comparison

### Before (Property Pattern)
```python
class Solver:
    @property
    def F0(self):
        # Creates NEW expression every access
        return expression(name, value, desc)

# Problem:
solver = Solver()
id1 = id(solver.F0)  # e.g., 140234567
id2 = id(solver.F0)  # e.g., 140234789 (DIFFERENT!)
```

### After (ExpressionProperty Pattern)
```python
class Solver:
    F0 = ExpressionProperty(name, value_fn, desc)

# Solution:
solver = Solver()
id1 = id(solver.F0)  # e.g., 140234567
id2 = id(solver.F0)  # e.g., 140234567 (SAME!)
```

## Updated Solvers

The following solvers have been migrated to use `ExpressionProperty`:

- `SNES_Poisson` - F0, F1
- `SNES_Darcy` - F0, F1
- `SNES_Stokes` - F0, F1, PF0
- `SNES_VE_Stokes` - Inherits from Stokes
- `SNES_Projection` - F0, F1
- `SNES_Vector_Projection` - F0, F1
- `SNES_Tensor_Projection` - Uses scalar projection
- `SNES_AdvectionDiffusion` - Uses legacy pattern (time-dependent)
- `SNES_Diffusion` - Uses legacy pattern (time-dependent)
- `SNES_NavierStokes` - Uses legacy pattern (time-dependent)

Note: Time-dependent solvers still use the legacy pattern due to their complex interaction with `DuDt` and `DFDt` objects. These may be migrated in a future update.

## Related Components

### SymbolicProperty
A simpler descriptor for automatic unwrapping of symbolic objects:
- Used for: `uw_function`, `source_term`, etc.
- Auto-unwraps objects with `_sympify_()` protocol
- Doesn't create persistent containers

### UWexpression
The underlying expression class:
- Maintains unique names to avoid duplicates
- Supports lazy evaluation
- Has mutable `.sym` property for content updates

## Best Practices

1. **Use ExpressionProperty for solver template expressions** (F0, F1, PF0, etc.)
2. **Use SymbolicProperty for simple symbolic inputs** (uw_function, etc.)
3. **Lambda functions in sym_template** should capture dependencies properly
4. **Avoid creating expressions in properties** - use descriptors instead
5. **Test object identity** to ensure persistence is working

## Migration Guide

To migrate a solver from property pattern to ExpressionProperty:

1. Remove the `@property` decorator and method
2. Add `ExpressionProperty` at class level:
   ```python
   # OLD
   @property
   def F0(self):
       return expression(name, value, desc)

   # NEW
   F0 = ExpressionProperty(
       name,
       lambda self: value,
       desc
   )
   ```
3. Ensure lambda captures all needed dependencies
4. Test that object identity is preserved

## Technical Notes

- Expressions use `_unique_name_generation=True` to avoid conflicts
- The `.sym` property is updated lazily on access
- AttributeError handling prevents issues during initialization
- Weak references prevent circular dependencies
- Compatible with JIT compilation and PETSc solvers

## Future Work

- Migrate time-dependent solvers to use ExpressionProperty
- Consider caching strategies for expensive symbolic operations
- Extend pattern to other persistent symbolic objects
- Add debugging tools for tracking expression updates