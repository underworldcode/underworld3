# UWCoordinate Design Document (December 2025)

## Problem Statement

The current units/nondimensionalization system has a recurring architectural flaw: **coordinates (x, y, z) are indistinguishable from other symbolic objects**, leading to:

1. `has_units` flag checks scattered throughout arithmetic operators
2. Special-case branching in `unwrap_for_evaluate`
3. Bugs where expressions like `r - inner_radius` embed raw values (3000) instead of nondimensional values (3.0)
4. Repeated refactor cycles that add complexity without solving the root cause

## Root Cause

Coordinates are fundamentally different from quantities:

| Aspect | Quantity (e.g., `inner_radius`) | Coordinate (e.g., `x`) |
|--------|--------------------------------|------------------------|
| Has a value | Yes (3000 km) | No - it's a placeholder |
| Needs ND scaling | Yes (3000 → 3.0) | No - takes on input values |
| At evaluation | Substitute scaled constant | Substitute PETSc coord data |
| Similar to | Constants, parameters | MeshVariable |

But currently both are represented as SymPy Symbols (or UWexpression), so the code can't tell them apart without flag checks.

## Solution: UWCoordinate Class

Create a dedicated class for Cartesian coordinates that:

1. **Is recognizable by type** - `isinstance(sym, UWCoordinate)` replaces `has_units` checks
2. **Parallels MeshVariable** - `.sym` for symbolic, `.data` for numeric
3. **Is JIT compatible** - unwraps to BaseScalar for existing machinery
4. **Is NOT nondimensionalized** - it's a placeholder, not a value

## Design

### Class Definition

```python
# Location: underworld3/coordinates.py

class UWCoordinate(Symbol):
    """
    A Cartesian coordinate variable (x, y, or z).

    Parallels MeshVariable pattern:
    - .sym → BaseScalar (symbolic form for JIT)
    - .data → coordinate values from mesh (numeric)

    At evaluation, substituted with PETSc coordinate data.
    NOT nondimensionalized - it's a placeholder, not a quantity.

    Examples
    --------
    >>> x, y = mesh.X  # UWCoordinate objects
    >>> x.sym          # BaseScalar N.x for JIT
    >>> x.data         # numpy array of x-coordinates
    >>> r = sympy.sqrt(x**2 + y**2)  # Works in expressions
    """

    def __new__(cls, name, base_scalar, mesh, axis_index):
        instance = Symbol.__new__(cls, name)
        return instance

    def __init__(self, name, base_scalar, mesh, axis_index):
        self._base_scalar = base_scalar  # The SymPy BaseScalar (N.x, N.y, N.z)
        self._mesh = mesh
        self._axis_index = axis_index    # 0, 1, or 2
        self._name = name

    @property
    def sym(self):
        """
        BaseScalar for JIT/symbolic operations.

        This is what the evaluate/JIT system sees after unwrapping.
        """
        return self._base_scalar

    @property
    def data(self):
        """
        Coordinate values from mesh.

        Returns dimensional values if mesh has units, ND otherwise.
        Mirrors MeshVariable.data pattern.
        """
        return self._mesh.X.coords[:, self._axis_index]

    @property
    def mesh(self):
        """Parent mesh."""
        return self._mesh

    @property
    def axis(self):
        """Axis index (0=x, 1=y, 2=z)."""
        return self._axis_index

    def __repr__(self):
        return f"{self._name}"
```

### Integration: CoordinateSystem

Modify `CoordinateSystem.__init__` to create `UWCoordinate` objects instead of using raw BaseScalars:

```python
# In CoordinateSystem.__init__

# Create UWCoordinate objects for Cartesian coordinates
base_scalars = self.mesh.r  # The raw BaseScalars (N.x, N.y, N.z)

if self.mesh.cdim == 2:
    x = UWCoordinate("x", base_scalars[0], self.mesh, 0)
    y = UWCoordinate("y", base_scalars[1], self.mesh, 1)
    self._N = sympy.Matrix([[x, y]])
else:  # cdim == 3
    x = UWCoordinate("x", base_scalars[0], self.mesh, 0)
    y = UWCoordinate("y", base_scalars[1], self.mesh, 1)
    z = UWCoordinate("z", base_scalars[2], self.mesh, 2)
    self._N = sympy.Matrix([[x, y, z]])

# LaTeX forms for display
self._N[0]._latex_form = r"\mathrm{x}"
self._N[1]._latex_form = r"\mathrm{y}"
if self.mesh.cdim == 3:
    self._N[2]._latex_form = r"\mathrm{z}"
```

### Integration: unwrap_for_evaluate

Simplify the unwrap logic to use type-based dispatch:

```python
def unwrap_for_evaluate(expr, scaling_active=None):
    """
    Unwrap expression for evaluate/lambdify path.

    Type-based dispatch (no has_units flags):
    - UWCoordinate: unwrap to BaseScalar (placeholder, no scaling)
    - UWexpression: nondimensionalize via .data
    - UWQuantity: nondimensionalize via .data
    - BaseScalar/MeshVariable.sym: pass through unchanged
    """
    # ... existing preamble ...

    for sym in sym_expr.free_symbols:
        if isinstance(sym, UWCoordinate):
            # Coordinate placeholder - unwrap to BaseScalar, NO scaling
            substitutions[sym] = sym.sym

        elif isinstance(sym, UWexpression):
            # Expression wrapping a value - nondimensionalize
            if should_scale:
                substitutions[sym] = float(sym.data)
            else:
                substitutions[sym] = float(sym.value)

        elif isinstance(sym, UWQuantity):
            # Quantity - nondimensionalize
            if should_scale:
                substitutions[sym] = float(sym.data)
            else:
                substitutions[sym] = float(sym.value)

        # BaseScalars, Function objects, etc. pass through unchanged

    # ... rest of function ...
```

### Cleanup: Remove has_units Branching from Operators

With `UWCoordinate` recognizable by type, arithmetic operators can be simplified:

```python
# In UWexpression

def __sub__(self, other):
    """Subtraction - LAZY EVALUATION, delegate to Symbol."""
    # No need to check has_units - just keep symbols in tree
    return Symbol.__sub__(self, other)

def __add__(self, other):
    """Addition - LAZY EVALUATION, delegate to Symbol."""
    return Symbol.__add__(self, other)

# etc. for other operators
```

The `unwrap_for_evaluate` function handles all the type-specific logic in one place.

## What This Achieves

1. **Single point of truth** - type identity determines behavior, not flags
2. **Eliminates has_units branching** - no more scattered flag checks
3. **Nilpotent nondimensionalization** - call it once, it handles everything correctly
4. **Parallel to MeshVariable** - users understand the pattern
5. **JIT compatible** - BaseScalar unwrapping preserved

## Derived Coordinates (r, θ, φ)

Derived coordinates like `r = sqrt(x**2 + y**2)` remain as regular `UWexpression` objects:

```python
# In CoordinateSystem for CYLINDRICAL2D
x, y = self.N  # These are UWCoordinate objects
r = expression(R"r", sympy.sqrt(x**2 + y**2), "Radial Coordinate")
```

This is correct because:
- `r` is an expression built FROM coordinates
- It contains `UWCoordinate` symbols that will be unwrapped to BaseScalars
- No nondimensionalization needed for the expression itself

## Migration Path

1. Implement `UWCoordinate` class in `coordinates.py`
2. Modify `CoordinateSystem.__init__` to use it
3. Update `unwrap_for_evaluate` with type-based dispatch
4. Simplify arithmetic operators to pure delegation
5. Remove `has_units` checks where no longer needed
6. Run test suite to verify

## Testing

Key test case that should pass after implementation:

```python
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    ...
)

mesh = uw.meshing.Annulus(...)
r, th = mesh.CoordinateSystem.xR

inner_radius = uw.expression(r"r_i", uw.quantity(3000, "km"), "inner radius")
outer_radius = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
mantle_thickness = outer_radius - inner_radius

r_prime = (r - inner_radius) / mantle_thickness

# This should give values 0 to 1, not -889!
result = uw.function.evaluate(r_prime, mesh.X.coords)
assert 0 <= result.min() <= 0.1
assert 0.9 <= result.max() <= 1.0
```
