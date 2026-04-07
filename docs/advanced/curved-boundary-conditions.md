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

## Four Approaches

### 1. Nitsche Free-Slip (Recommended)

Nitsche's method provides a variationally consistent alternative to penalty
that is insensitive to the penalty magnitude and gives optimal convergence:

```python
stokes.add_nitsche_bc("Upper", gamma=10)
```

The method automatically constructs penalty, consistency (stress flux),
symmetry, and pressure coupling terms. The `gamma` parameter is dimensionless
and mesh-independent — `gamma=10` works for P2 elements regardless of
resolution or viscosity.

**Prescribed normal velocity:**
```python
stokes.add_nitsche_bc("Inlet", g=1.0, gamma=10)
```

**Custom constraint direction** (e.g., fault normal different from surface normal):
```python
fault_normal = sympy.Matrix([0.6, 0.8])
stokes.add_nitsche_bc("Fault", direction=fault_normal, gamma=10)
```

**When to use:**
- Free-slip on any geometry (boxes, annuli, spherical shells)
- Spherical shell models where penalty is fragile
- When you don't want to tune a penalty parameter
- Basal shear constraints with custom direction

**Accuracy:** Optimal convergence rate. On a Cartesian box test, Nitsche at
`gamma=10` gives 0.08% velocity error vs the essential BC solution (penalty
at 1e4 gives 0.15%).


### 2. Penalty Free-Slip (Simple but Fragile)

Use the mesh-derived normals directly with a penalty parameter:

```python
Gamma = mesh.Gamma
penalty = 10000
stokes.add_natural_bc(penalty * Gamma.dot(v.sym) * Gamma, "Boundary")
```

**When to use:**
- Quick prototyping where high accuracy isn't critical
- When Nitsche is not yet available for your solver type

**Limitations:**
- Penalty must be tuned: too small → loose constraint, too large → ill-conditioning
- On spherical shells, penalty can become unstable at moderate resolution
- ~25-30% error on elliptical boundaries when using raw facet normals


### 3. Projected Normals (For Curved Boundaries with Penalty)

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


### 4. Analytical Normals (Most Accurate)

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

We tested the approaches on an elliptical annulus (ellipticity = 1.5) with a free-slip boundary condition:

| Approach | Error vs Analytical | Notes |
|----------|---------------------|-------|
| Penalty + raw `mesh.Gamma` | 26.88% | Penalty-sensitive |
| Penalty + projected normals | 0.06% | Requires projection solve |
| Penalty + analytical normals | 0% (reference) | Requires surface formula |
| Nitsche (default) | ~0.1% | No penalty tuning needed |

Nitsche is recommended for most use cases — it matches the accuracy of
projected normals without requiring a separate projection solve or penalty
tuning.

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

## Internal Boundaries

```{warning}
Nitsche BCs are designed for **exterior** boundaries only. Do not use
`add_nitsche_bc` on internal boundary labels (e.g., `"Internal"` from
`BoxInternalBoundary` or `AnnulusInternalBoundary`).
```

On an internal surface, PETSc evaluates boundary residuals from **both**
adjacent cells with opposite normals. The Nitsche consistency and pressure
terms flip sign between the two sides and partially cancel, injecting
a spurious residual that degrades the solution. Testing on an annulus
with an internal boundary showed that Nitsche produced *worse* constraint
enforcement than no constraint at all.

**For internal boundary constraints, continue using the penalty approach:**

```python
Gamma = mesh.Gamma
stokes.add_natural_bc(penalty * Gamma.dot(v.sym) * Gamma, "Internal")
```

The penalty term is quadratic in the normal (`n × n`), so both sides
reinforce correctly regardless of normal orientation.

A proper Nitsche formulation for internal surfaces would require an
interior penalty (IP/SIP) method with explicit jump and average operators
across the interface — accessing the solution from both adjacent cells.
PETSc's current pointwise callbacks do not provide cross-cell access,
so this would require a different assembly strategy. This is an area
for future development, particularly for embedded impermeable surfaces
in 3D spherical models where penalty sensitivity becomes problematic.

---

## Tips for Success

1. **Start with Nitsche**: `stokes.add_nitsche_bc("Upper", gamma=10)` — no penalty tuning needed
2. **For penalty BCs, always normalize**: Analytical formulas need explicit normalization
3. **Check orientation**: Ensure normals point outward (use `sign(r.dot(normal))`)
4. **Verify visually**: Plot the normal field to catch errors
5. **Use analytical normals for validation**: Compare against exact surface geometry when possible
6. **Custom constraint direction**: Use `direction=` parameter when the constraint
   direction differs from the surface normal (faults, basal shear)

---

## Further Reading

- [Custom Mesh Creation](custom-meshes.md) — Creating elliptical and complex meshes
- [Stokes Ellipse Example](../examples/fluid_mechanics/intermediate/Ex_Stokes_Ellipse_Cartesian.py) — Complete worked example
- Sime & Wilson (2020), [arXiv:2001.10639](https://arxiv.org/abs/2001.10639) — Nitsche free-slip for geodynamics

---

*Last updated: April 2026*
