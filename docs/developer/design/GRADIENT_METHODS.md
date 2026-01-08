# Gradient Evaluation Methods in Underworld3

**Date:** 2026-01-07
**Status:** Implementation complete

## Overview

Computing gradients of field variables is fundamental to many geodynamic applications:
- Strain rates from velocity fields
- Heat flux from temperature fields
- Stress from displacement fields
- Error estimation for mesh adaptation

This document compares the available gradient evaluation methods in UW3, their accuracy, computational cost, and appropriate use cases.

## Method Comparison

| Method | Accuracy | Cost | Solve Required | Use Case |
|--------|----------|------|----------------|----------|
| Clement Interpolant | O(h) | Low | No | Quick estimates, error indicators, post-processing |
| L2 Projection | O(h²) | Medium | Yes (mass matrix) | High accuracy requirements |
| Direct Differentiation | Exact at quadrature | N/A | No | Weak form integration (internal) |

## Method 1: Clement Interpolant

### Theory

The Clement interpolant [1] recovers continuous gradients from discontinuous (cell-wise) gradient data:

1. Compute gradient in each cell (piecewise constant)
2. For each vertex, average gradients from all cells sharing that vertex
3. Result: continuous P1 (linear) gradient field

This is an O(h) accurate approximation - error halves when mesh resolution doubles.

### Implementation

```python
import underworld3 as uw

# Quick evaluation at arbitrary points (recommended)
grad = uw.function.evaluate_gradient(T, coords)

# Or just get values at mesh nodes
grad_nodes = uw.function.compute_clement_gradient_at_nodes(T)
```

### Convergence

Test function: f(x,y) = x² + y², exact gradient = (2x, 2y)

| Resolution | Max Error | Rate |
|------------|-----------|------|
| 8×8        | 1.77e-01  | -    |
| 16×16      | 8.84e-02  | 1.0  |
| 32×32      | 4.42e-02  | 1.0  |
| 64×64      | 2.21e-02  | 1.0  |

**Observation:** Error halves with each resolution doubling → O(h) convergence confirmed.

### Advantages
- No linear solve required
- Fast computation (local averaging only)
- No additional fields added to mesh DM (uses scratch DM internally)
- Suitable for post-processing and visualization

### Limitations
- Only O(h) accurate (first order)
- Requires scalar input (apply component-wise for vectors)
- Less accurate than L2 projection for smooth solutions

## Method 2: L2 Projection

### Theory

Solve for the gradient that minimizes the L2 error:

Find **g** such that:
∫ **g** · **v** dΩ = ∫ ∇f · **v** dΩ  for all test functions **v**

This gives O(h²) accuracy for smooth solutions - error quarters when resolution doubles.

### Implementation

```python
import underworld3 as uw

# Create gradient variable
grad_T = uw.discretisation.MeshVariable('grad_T', mesh, mesh.dim)

# Set up projection
projector = uw.systems.Projection(mesh, grad_T)
projector.uw_function = T.sym.diff(mesh.X)
projector.solve()

# Evaluate at points
result = uw.function.evaluate(grad_T.sym, coords)
```

### Advantages
- O(h²) accuracy for smooth solutions
- Optimal in L2 sense
- Works with any polynomial degree

### Limitations
- Requires solving linear system (mass matrix)
- Adds permanent field to mesh DM
- More setup code required
- Higher computational cost

## Method 3: Direct Differentiation (Weak Form)

### Theory

Within weak form integration, gradients are evaluated directly at quadrature points using the finite element basis function derivatives:

∇f|_q = Σᵢ fᵢ ∇Nᵢ(x_q)

This is exact at quadrature points (no approximation beyond FE interpolation).

### Implementation

This is used internally by solvers. Users access it via symbolic expressions:

```python
# In constitutive relations
strain_rate = sympy.derive_by_array(v.sym, mesh.X)

# In weak forms
stokes.constitutive_model.flux = viscosity * strain_rate
```

### Use Case
- Internal to PDE solvers
- Constitutive model definitions
- Not for post-processing (use Clement or L2 projection)

## Choosing a Method

### Use Clement Interpolant when:
- Quick gradient estimates are needed
- Post-processing or visualization
- Error estimation for mesh adaptation
- O(h) accuracy is acceptable
- Avoiding DM field pollution is important

### Use L2 Projection when:
- High accuracy is required
- Gradient field will be reused multiple times
- Smooth solution with O(h²) convergence expected
- Computational cost is not limiting

### Use Direct Differentiation when:
- Defining weak forms or constitutive models
- Gradients are needed at quadrature points only
- Working within solver framework

## API Reference

### `uw.function.evaluate_gradient(scalar_var, coords, method="clement")`

Evaluate gradient at arbitrary coordinates without polluting mesh DM.

**Parameters:**
- `scalar_var`: Scalar MeshVariable to differentiate
- `coords`: Coordinates array, shape (n_points, dim)
- `method`: Gradient method (currently "clement" only)

**Returns:**
- Gradient array, shape (n_points, dim)

### `uw.function.compute_clement_gradient_at_nodes(scalar_var)`

Compute Clement gradient at mesh nodes only (no interpolation).

**Parameters:**
- `scalar_var`: Scalar MeshVariable to differentiate

**Returns:**
- Gradient array, shape (n_nodes, dim)

## Implementation Details

### Scratch DM Approach

The `evaluate_gradient` function uses an ephemeral "scratch DM" to avoid adding permanent fields to the mesh:

1. Clone mesh DM (gets topology, 0 fields)
2. Add P1 linear FE field matching Clement output
3. Populate with gradient data
4. Interpolate to requested coordinates
5. Destroy scratch objects

This ensures the main mesh DM is unchanged after the call.

### PETSc Functions Used

- `DMPlexComputeGradientClementInterpolant`: Core Clement computation
- `dm.clone()`: Create scratch DM with topology
- `PETSc.FE().createDefault()`: Create P1 finite element
- `DMInterpolation`: Arbitrary point evaluation

## References

1. Clément, P. (1975). "Approximation by finite element functions using local regularization". RAIRO Analyse numérique, 9(R-2), 77-84.

2. PETSc Manual: DMPlex Finite Element Support

3. Zienkiewicz, O.C. and Zhu, J.Z. (1992). "The superconvergent patch recovery and a posteriori error estimates". Int. J. Numer. Methods Eng., 33, 1331-1382.

## Integration with `uw.function.evaluate(T.diff(x), coords)`

### Current Behavior

When you call `T.sym.diff(x)`, SymPy creates an `UnderworldAppliedFunctionDeriv` object:

```python
x, y = mesh.X
dTdx = T.sym.diff(x)  # Returns UnderworldAppliedFunctionDeriv instance
```

Currently, `uw.function.evaluate()` raises an error when encountering these:

```
RuntimeError: Derivative functions are not handled in evaluations,
a projection should be used first to create a mesh Variable.
```

This happens in `mesh_vars_in_expression()` at `expressions.py:1648-1652`.

### Proposed Integration

**Key Classes:**
- `UnderworldAppliedFunction`: Base class for mesh variable symbols
- `UnderworldAppliedFunctionDeriv`: Subclass for derivatives, with:
  - `meshvar`: weakref to source MeshVariable
  - `component`: which component of the variable
  - `diffindex`: derivative direction (0=x, 1=y, 2=z)

**Integration Point:** `mesh_vars_in_expression()` in `expressions.py`

Instead of raising an error, the function should:

1. **Collect derivative expressions** alongside regular mesh variables
2. **Group by source variable** - multiple derivatives of same variable share one gradient computation
3. **Return derivative info** for the evaluation pipeline

**Modified `mesh_vars_in_expression()` signature:**
```python
def mesh_vars_in_expression(expr):
    """
    Returns:
        tuple: (mesh, regular_vars, derivative_vars)
        - regular_vars: set of UnderworldAppliedFunction
        - derivative_vars: dict mapping source_var -> list of (deriv_expr, diffindex)
    """
```

**Evaluation Pipeline Changes:**

In `evaluate_nd()` or `functions_unit_system.py`:

```python
mesh, regular_vars, derivative_vars = mesh_vars_in_expression(expr)

if derivative_vars:
    # For each source variable with derivatives needed
    for source_var, deriv_list in derivative_vars.items():
        # Compute gradient once per source variable
        gradient_at_nodes = compute_clement_gradient_at_nodes(source_var)

        # Create scratch DM and interpolate to coords
        gradient_at_coords = _interpolate_gradient(mesh, gradient_at_nodes, coords)

        # Substitute derivative symbols with interpolated values
        for deriv_expr, diffindex in deriv_list:
            # gradient_at_coords[:, diffindex] gives ∂f/∂x_i values
            expr = expr.subs(deriv_expr, gradient_at_coords[:, diffindex])
```

### Implementation Complexity

**Moderate complexity** because:
1. The evaluation pipeline has multiple paths (RBF, DMInterpolation, pure sympy)
2. Need to handle mixed expressions (derivatives + regular vars)
3. Must work correctly with unit handling and coordinate transformations
4. Expression substitution requires care with SymPy's immutability

**Recommended Approach:**
1. Start with a specialized `evaluate_derivative()` function
2. Then integrate into main `evaluate()` as an optional path
3. Keep the error for unsupported cases (second derivatives, etc.)

### Alternative: Direct User API

A simpler intermediate step is the current `evaluate_gradient()`:

```python
# Instead of: uw.function.evaluate(T.diff(x), coords)  # Fails
# Use:
grad = uw.function.evaluate_gradient(T, coords)
dTdx = grad[:, 0]  # x-derivative
dTdy = grad[:, 1]  # y-derivative
```

This is explicit, clear, and already works.

## Future Work

- Full integration with `uw.function.evaluate(T.diff(x), coords)` syntax
- Support for vector field gradients (velocity → strain rate)
- Caching of scratch DM for repeated evaluations
- Parallel (MPI) support verification
