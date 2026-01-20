# Boundary Conditions on Curved Surfaces

**Getting accurate free-slip and Neumann conditions on non-planar boundaries**

When applying boundary conditions to curved or angled surfaces (ellipses, spheres, arbitrary geometries), the direction of the surface normal matters significantly. This guide explains the different approaches and their trade-offs.

---

## The Problem: Facet-Based Normals

Underworld3 provides `mesh.Gamma`, which gives the outward normal vector at boundary quadrature points. These normals are computed by PETSc from the mesh geometry.

**Key facts about `mesh.Gamma`:**
- The vectors ARE unit vectors (magnitude = 1)
- The direction is based on **mesh facet geometry**, not the true surface

For straight-edged boundaries (boxes, etc.), `mesh.Gamma` is exact. But for curved boundaries, the mesh is a piecewise-linear approximation of the curve, creating a "stair-step" pattern of normals:

```
True ellipse surface:     Mesh approximation:

    ╭───────╮                ╱─────╲
   ╱         ╲              ╱       ╲
  │    →→→    │            │  →→↗↗   │
  │   →→→→    │    vs      │ →→→↗↗   │
   ╲         ╱              ╲       ╱
    ╰───────╯                ╲─────╱

(smooth normals)          (step-function normals)
```

This can cause significant errors in free-slip boundary conditions.

---

## Three Approaches

### 1. Raw `mesh.Gamma` (Simplest)

Use the mesh-derived normals directly:

```python
Gamma = mesh.Gamma
penalty = 10000
stokes.add_natural_bc(penalty * Gamma.dot(v.sym) * Gamma, "Boundary")
```

**When to use:**
- Straight-edged boundaries (boxes, channels)
- Circular boundaries (normals are radial, so facet direction is correct)
- Quick prototyping where high accuracy isn't critical

**Accuracy:** ~25-30% error on elliptical boundaries


### 2. Projected Normals (Recommended for Curved Boundaries)

Project `mesh.Gamma` onto a continuous mesh variable, which interpolates and smooths the normals:

```python
import sympy

x, y = mesh.X
r = sympy.sqrt(x**2 + y**2)
unit_r = sympy.Matrix([x/r, y/r]).T

# Create variable to store projected normals
n_proj = uw.discretisation.MeshVariable("n_proj", mesh, mesh.dim, degree=2)

# Project mesh.Gamma with orientation correction
projection = uw.systems.Vector_Projection(mesh, n_proj)
projection.uw_function = sympy.Matrix([[0] * mesh.dim])

# Ensure consistent outward orientation
orientation = sympy.sign(unit_r.dot(mesh.Gamma))
projection.add_natural_bc(mesh.Gamma * orientation, "Outer")
projection.add_natural_bc(mesh.Gamma * orientation, "Inner")
projection.solve()

# Normalize to unit vectors
import numpy as np
mag = np.sqrt(np.sum(n_proj.data**2, axis=1, keepdims=True))
n_proj.data[:] /= mag

# Use in boundary condition
stokes.add_natural_bc(penalty * n_proj.sym.dot(v.sym) * n_proj.sym, "Boundary")
```

**When to use:**
- Curved boundaries where you don't have an analytical formula
- Complex geometries from CAD/imaging data
- When you want good accuracy without deriving surface equations

**Accuracy:** ~0.1% error compared to analytical normals

**Why it works:** The projection solves a weak-form problem that naturally smooths the discontinuous facet normals into a continuous field. The finite element basis functions interpolate between facets, approximating the true surface direction.


### 3. Analytical Normals (Most Accurate)

Derive the surface normal from the mathematical definition of the boundary:

```python
import sympy

x, y = mesh.X
a, b = 1.5, 1.0  # Ellipse semi-axes

# For ellipse: (x/a)² + (y/b)² = 1
# Normal = gradient of level set function
normal = sympy.Matrix([2*x/a**2, 2*y/b**2]).T
unit_normal = normal / sympy.sqrt(normal.dot(normal))

# Use in boundary condition
stokes.add_natural_bc(penalty * unit_normal.dot(v.sym) * unit_normal, "Outer")
```

**When to use:**
- Known geometric shapes (ellipses, superellipses, spheroids)
- Highest accuracy requirements
- Benchmarking and validation

**Accuracy:** Exact (limited only by mesh resolution and quadrature)


---

## Common Surface Normal Formulas

For a surface defined implicitly by $f(x, y) = c$, the outward normal is:

$$\hat{n} = \frac{\nabla f}{|\nabla f|}$$

### Ellipse

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$$

```python
# Outward normal (pointing away from center)
normal = sympy.Matrix([2*x/a**2, 2*y/b**2]).T
unit_normal = normal / sympy.sqrt(normal.dot(normal))
```

### Superellipse

$$\left|\frac{x}{a}\right|^n + \left|\frac{y}{b}\right|^n = 1$$

```python
# For n > 2 (rounded rectangle)
n_exp = 4  # Example
normal = sympy.Matrix([
    n_exp * sympy.sign(x) * sympy.Abs(x/a)**(n_exp-1) / a,
    n_exp * sympy.sign(y) * sympy.Abs(y/b)**(n_exp-1) / b
]).T
unit_normal = normal / sympy.sqrt(normal.dot(normal))
```

### Sphere (3D)

$$x^2 + y^2 + z^2 = R^2$$

```python
x, y, z = mesh.X
r = sympy.sqrt(x**2 + y**2 + z**2)
unit_normal = sympy.Matrix([x/r, y/r, z/r]).T
```

### Ellipsoid (3D)

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$$

```python
x, y, z = mesh.X
normal = sympy.Matrix([2*x/a**2, 2*y/b**2, 2*z/c**2]).T
unit_normal = normal / sympy.sqrt(normal.dot(normal))
```

---

## Experimental Comparison

We tested the three approaches on an elliptical annulus (ellipticity = 1.5) with a free-slip boundary condition:

| Approach | Error vs Analytical |
|----------|---------------------|
| Raw `mesh.Gamma` | 26.88% |
| Projected normals | 0.06% |
| Analytical | 0% (reference) |

The projected normals provide **99.8% improvement** over raw `mesh.Gamma` for curved boundaries.

---

## When Accuracy Matters

The error in boundary normals affects:

1. **Free-slip conditions**: Incorrect normals allow spurious tangential flow
2. **Heat flux (Neumann) conditions**: Wrong direction for flux specification
3. **Traction boundary conditions**: Force direction errors
4. **Stress computations**: Errors in computed surface stresses

For many geodynamics applications, the ~25% error from raw normals may be acceptable for exploratory work but problematic for:
- Benchmark comparisons
- Convergence studies
- Quantitative predictions

---

## Angled Boundaries

For boundaries that aren't aligned with coordinate axes (e.g., a tilted fault plane), the same principles apply:

```python
# Fault plane: ax + by + c = 0
a_coef, b_coef = 1.0, 0.5  # Normal direction coefficients
normal = sympy.Matrix([a_coef, b_coef]).T
unit_normal = normal / sympy.sqrt(normal.dot(normal))
```

For complex geometries where the orientation varies spatially, the projection approach is often the most practical.

---

## Tips for Success

1. **Always normalize**: Analytical formulas need explicit normalization
2. **Check orientation**: Ensure normals point outward (use `sign(r.dot(normal))`)
3. **Verify visually**: Plot the normal field to catch errors
4. **Start with projection**: It's robust and doesn't require deriving formulas
5. **Use analytical for validation**: Compare projected normals against analytical when possible

---

## Further Reading

- [Custom Mesh Creation](custom-meshes.md) — Creating elliptical and complex meshes
- [Stokes Ellipse Example](../examples/fluid_mechanics/intermediate/Ex_Stokes_Ellipse_Cartesian.py) — Complete worked example

---

*Last updated: January 2026*
