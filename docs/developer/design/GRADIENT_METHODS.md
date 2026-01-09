# Gradient Evaluation Methods in Underworld3

**Date:** 2026-01-08
**Status:** Dual method implementation complete

## Overview

Computing gradients of field variables is fundamental to many geodynamic applications:
- Strain rates from velocity fields
- Heat flux from temperature fields
- Stress from displacement fields
- Error estimation for mesh adaptation

This document describes the dual gradient evaluation methods available in UW3, their accuracy, computational cost, and appropriate use cases.

## Method Comparison

| Method | Name | Accuracy | Cost | Solve Required | Use Case |
|--------|------|----------|------|----------------|----------|
| Clement Interpolant | `"interpolant"` | O(h) | Low | No | Quick estimates, error indicators |
| L2 Projection | `"projection"` | O(h²) | Medium | Yes (cached) | High accuracy, constitutive relations |
| Direct Differentiation | N/A | Exact at quadrature | N/A | No | Weak form integration (internal) |

## Unified API

Both methods are accessible through the standard `uw.function.evaluate()` syntax:

```python
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(...)
x, y = mesh.X
T = uw.discretisation.MeshVariable("T", mesh, num_components=1, degree=1)

# Default: uses "projection" for PETSc path (O(h²) accurate)
dTdx = uw.function.evaluate(T.sym.diff(x), coords)

# Explicit interpolant method (O(h) accurate, faster)
dTdx_fast = uw.function.evaluate(T.sym.diff(x), coords, gradient_method="interpolant")

# Explicit projection method (O(h²) accurate)
dTdx_accurate = uw.function.evaluate(T.sym.diff(x), coords, gradient_method="projection")
```

## Method 1: Interpolant (Clement)

### Theory

The Clement interpolant [1] recovers continuous gradients from discontinuous (cell-wise) gradient data:

1. Compute gradient in each cell (piecewise constant)
2. For each vertex, average gradients from all cells sharing that vertex
3. Result: continuous P1 (linear) gradient field

This is an O(h) accurate approximation - error halves when mesh resolution doubles.

### Implementation

The interpolant method uses PETSc's `DMPlexComputeGradientClementInterpolant` internally, with a scratch DM to avoid polluting the mesh's field structure.

```python
# Via unified evaluate() API
dTdx = uw.function.evaluate(T.sym.diff(x), coords, gradient_method="interpolant")

# Or using the direct function
from underworld3.function.gradient_evaluation import evaluate_gradient
grad = evaluate_gradient(T, coords, method="interpolant")
```

### Advantages
- No linear solve required
- Fast computation (local averaging only)
- No additional fields added to mesh DM (uses scratch DM internally)
- Suitable for post-processing and visualization

### Limitations
- Only O(h) accurate (first order)
- Always produces P1 output regardless of source field degree
- Less accurate than L2 projection for smooth solutions

## Method 2: Projection (L2)

### Theory

Solve for the gradient that minimizes the L2 error:

Find **g** such that:
∫ **g** · **v** dΩ = ∫ ∇f · **v** dΩ  for all test functions **v**

This gives O(h²) accuracy for smooth solutions - error quarters when resolution doubles.

### Implementation

The projection method creates cached gradient variables and projectors on the mesh. Subsequent calls reuse these with warm-started solves (`zero_init_guess=False`).

```python
# Via unified evaluate() API (default for PETSc path)
dTdx = uw.function.evaluate(T.sym.diff(x), coords, gradient_method="projection")

# Caching: projection objects are stored on mesh._gradient_cache
# Second call reuses the projector with warm start
dTdx_2 = uw.function.evaluate(T.sym.diff(x), new_coords, gradient_method="projection")
```

### Caching Strategy

The projection method automatically caches:
1. **Gradient MeshVariables**: One per gradient component (dT/dx, dT/dy, etc.)
2. **Projector objects**: Ready to solve with `zero_init_guess=False` for warm-starting

Cache key: `_grad_proj_{variable_name}_c{component}_d{dim}`

When the source field values change, the projector's symbolic expression still references the correct source, so the next solve produces updated results.

### Advantages
- O(h²) accuracy for smooth solutions
- Optimal in L2 sense
- Caching eliminates setup overhead for repeated evaluations
- Warm-started solves for time-dependent problems

### Limitations
- Requires solving linear system (mass matrix)
- Creates additional mesh variables (cached on mesh)
- Higher computational cost than interpolant

## Convergence Comparison

Test function: f(x,y) = sin(πx) sin(πy)

| Cell Size | Interpolant | Projection | Ratio |
|-----------|-------------|------------|-------|
| 0.200     | 2.48e-01    | 1.25e-01   | 2.0×  |
| 0.100     | 7.91e-02    | 4.08e-02   | 1.9×  |
| 0.050     | 1.91e-02    | 7.09e-03   | 2.7×  |

**Observed Convergence Rates:**
- Interpolant: O(h^1.9) ≈ O(h²) on regular meshes (superconvergence)
- Projection: O(h^2.1) ≈ O(h²)

Note: On irregular meshes, the interpolant degrades to O(h) while projection maintains O(h²).

## Choosing a Method

### Use Interpolant (`"interpolant"`) when:
- Quick gradient estimates are needed
- Post-processing or visualization
- Error estimation for mesh adaptation
- O(h) accuracy is acceptable
- Avoiding solve overhead is important

### Use Projection (`"projection"`) when:
- High accuracy is required
- Gradient field will be reused multiple times (caching benefits)
- Smooth solution with O(h²) convergence expected
- Computational cost is not limiting

### Default Behavior

When `gradient_method=None`:
- **PETSc path** (`rbf=False`): defaults to `"projection"` (accurate)
- **RBF path** (`rbf=True`): derivatives not supported (raises error)

## API Reference

### `uw.function.evaluate(expr, coords, ..., gradient_method=None)`

Evaluate expression including derivatives at arbitrary coordinates.

**Parameters:**
- `expr`: SymPy expression, may include `T.sym.diff(x)` terms
- `coords`: Coordinates array, shape (n_points, dim)
- `gradient_method`: `"interpolant"`, `"projection"`, or `None` (default)

**Returns:**
- Evaluated array with appropriate shape

### `uw.function.evaluate_gradient(scalar_var, coords, method="interpolant")`

Direct gradient evaluation without going through the expression system.

**Parameters:**
- `scalar_var`: Scalar MeshVariable to differentiate
- `coords`: Coordinates array, shape (n_points, dim)
- `method`: `"interpolant"` or `"projection"`

**Returns:**
- Gradient array, shape (n_points, dim)

### `uw.function.compute_clement_gradient_at_nodes(scalar_var)`

Compute Clement gradient at mesh nodes only (no interpolation).

**Parameters:**
- `scalar_var`: Scalar MeshVariable to differentiate

**Returns:**
- Gradient array, shape (n_nodes, dim)

## Implementation Details

### Expression Detection

Derivatives are detected via `UnderworldAppliedFunctionDeriv` class:
- `meshvar`: weakref to source MeshVariable
- `component`: which component of the variable
- `diffindex`: derivative direction (0=x, 1=y, 2=z)

In `mesh_vars_in_expression()`, derivative expressions are collected into `derivfns` dict:
```python
derivfns[source_meshvar].append((deriv_expr, diffindex))
```

### Evaluation Pipeline

1. `evaluate()` in `functions_unit_system.py` receives expression
2. `evaluate_nd()` in `_function.pyx` calls `petsc_interpolate()`
3. `petsc_interpolate()` detects derivatives and calls `interpolate_gradients_at_coords()`
4. Gradients are computed (interpolant or projection) and substituted for derivative symbols
5. Final expression is evaluated via lambdify

### Scratch DM Approach (Interpolant)

The interpolant method uses an ephemeral "scratch DM" to avoid adding permanent fields:

1. Clone mesh DM (gets topology, 0 fields)
2. Add P1 linear FE field matching Clement output
3. Populate with gradient data
4. Interpolate to requested coordinates
5. Destroy scratch objects

### Gradient Cache (Projection)

The projection method stores cached objects on `mesh._gradient_cache`:

```python
mesh._gradient_cache[cache_name] = {
    'projectors': [(proj_var_d0, projector_d0), (proj_var_d1, projector_d1), ...],
    'source_var': scalar_var,
    'component': component,
}
```

## References

1. Clément, P. (1975). "Approximation by finite element functions using local regularization". RAIRO Analyse numérique, 9(R-2), 77-84.

2. PETSc Manual: DMPlex Finite Element Support

3. Zienkiewicz, O.C. and Zhu, J.Z. (1992). "The superconvergent patch recovery and a posteriori error estimates". Int. J. Numer. Methods Eng., 33, 1331-1382.

## Future Work

- Support for vector field gradients (velocity → strain rate tensor)
- Clear cache method when mesh adapts
- Parallel (MPI) verification for projection caching
