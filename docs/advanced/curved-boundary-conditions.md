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

## Five Approaches

### 1. Rotated Strong Free-Slip (Recommended)

```python
stokes.add_rotated_freeslip_bc(0, "Upper")
```

The velocity at each boundary node is rotated into a (normal, tangential)
frame and the normal component is fixed, as an essential condition would fix a
Cartesian component on a flat wall. The value comes first: `0` is free slip, and
a scalar expression or field prescribes the wall-normal velocity.

What it gives:

- `v·n = 0` at every boundary node to machine precision, on curved and deformed
  boundaries, with nothing to tune.
- The constraint reaction is the wall normal traction, read with
  `stokes.boundary_normal_traction("Upper")` or written as topography with
  `stokes.dynamic_topography(...)`.

The default normal is the measure-weighted node normal of the mesh,
`mesh.boundary_normal(boundary)`: the average of the facet normals at each node,
weighted by facet length (area in 3-D). It is the normal consistent with what the
solver integrates. **On a faceted curved wall, leave the normal to Underworld3.**
Passing the surface's own normal with `normal=` adds a condition the discrete
wall slip cannot meet where the facets are of unequal length: on the elliptical
annulus below, rotated free slip with the ellipse's own normal is 43.7 % and
44.6 % from the reference at cell sizes 0.1 and 0.05, with a large constant
pressure, while the default normal is 0.19 % at both (issue #1168, open). On a
circle, where the facets on a wall are of equal length, the two normals give the
same answer.

**Accuracy.** On the annulus with an exact solution (Kramer et al. 2021), the
error of the reaction against the exact wall normal stress is 6.8e-3, 3.3e-3,
2.1e-3 and 1.1e-3 at cell sizes 0.15, 0.10, 0.075 and 0.05, and the flow through
the wall is 5e-11 to 1e-10, the solver's floor (Underworld technical note
2026-016). On the Cartesian box in the table under approach 2, at cell size 1/8,
the rotated answer differs from the essential one by 2e-8, the solver tolerance
being 1e-6 (rerun, November 2026).

**When to use:** free slip on any exterior boundary that carries the same kind
of condition for the whole run. This is the default choice.
Details are in the developer page
[Rotated strong free-slip](../developer/subsystems/rotated-freeslip.md).


### 2. Nitsche (For Conditions That Change During the Model)

A rotated constraint is a hard (essential) condition, so it cannot be ramped from
one kind of condition to another during the run, for example from a fixed
velocity to a traction. Nitsche's method imposes the condition weakly, as boundary
terms in the weak form, whose weight can change. Use it for a condition that must
evolve in time:

```python
stokes.add_nitsche_bc(0.0, "Upper")
```

The method adds a penalty term, the consistency (stress flux) and symmetry
terms, and the pressure coupling in the continuity equation.

**Choosing `gamma`.** The penalty coefficient is `gamma * viscosity / h`, with
`h` the local cell size, so `gamma` is dimensionless: the coefficient follows the
viscosity and the cell size by construction. A larger `gamma` holds the wall
more tightly. The flow through the wall falls as `gamma` rises, but it is a poor
guide: choose `gamma` against the quantity you need, which is usually the stress
at the wall, and check that the answer does not move when `gamma` changes. The
Nitsche numbers of technical note 2026-016 are being corrected for #1145 (see
the note below), so none is quoted here. The examples here leave `gamma` at its
default.

**Curved walls that are not circles.** Nitsche is not accurate on such a wall
today. On the elliptical annulus below it is 35.2 % and 39.5 % from the
reference at the default `gamma` (cell sizes 0.1 and 0.05), 46.9 % and 46.2 % at
`gamma = 1000`, with a large constant pressure in both, and 19.5 % and 11.3 %
with the ellipse's own normal passed as `normal=` (issue #1168, open). Use
rotated free slip with its default normal on such a wall.

**Prescribed normal velocity:**
```python
stokes.add_nitsche_bc(1.0, "Inlet")
```

**Constraint along a direction that is not the wall normal** (e.g. basal shear
on an exterior boundary; see *Internal Boundaries* below for why not on an
internal label):
```python
shear_direction = sympy.Matrix([0.6, 0.8])
stokes.add_nitsche_bc(0.0, "Bottom", direction=shear_direction)
```

**When to use:**
- A boundary condition that switches on or off, or changes kind, during the run
- A constraint along a direction that is not the wall normal (basal shear)

**Accuracy.** On a Cartesian box (buoyancy-driven flow, free slip on the top and
bottom, P2/P1, cell size 1/8) the velocity differs from the essential-condition
answer by the amounts below, and the wall-normal velocity it leaves is given
relative to the largest speed (rerun, November 2026, on the code with the
correction of #1145, described in the note after the table):

| Condition | Difference from essential | Wall-normal velocity |
|---|---|---|
| Rotated free slip | 2e-8 | 0 |
| Nitsche, `gamma = 10` | 0.20 % | 7.0e-3 |
| Nitsche, default `gamma = 12.5` | 0.12 % | 4.3e-3 |
| Nitsche, `gamma = 100` | 0.010 % | 3.3e-4 |
| Nitsche, `gamma = 1000` | 0.0010 % | 3.3e-5 |
| Penalty, coefficient 1e4 | 0.15 % | 1.5e-3 |

```{note}
Before November 2026 the pressure coupling of `add_nitsche_bc` in the
continuity equation was not assembled (issue #1072, corrected in #1145). A
Stokes answer with a Nitsche condition normal to a wall, computed before that
correction, should be recomputed.
```


### 3. Penalty Free-Slip (Simple, and Only With the Node Normal)

```python
n = mesh.boundary_normal("Boundary")     # measure-weighted node normal
penalty = 10000
stokes.add_natural_bc(penalty * n.dot(v.sym) * n, "Boundary")
```

**Use `mesh.boundary_normal(boundary)`, not `mesh.Gamma`.** A penalty written
against the per-facet normal asks a node shared by two facets to satisfy two
different constraints, which on a two-component velocity leaves nothing: push
the coefficient up and the boundary freezes. Measured on an annulus with an
exact solution (Kramer et al. 2021), coefficient `1e6`, cell 0.15 → 0.035:

| normal | leak `u·n` | velocity error | surface stress error |
|---|---|---|---|
| `mesh.Gamma` (facet) | 1e-5 | 0.60, flat under refinement | 0.21 → 0.26, growing |
| `mesh.boundary_normal` (node) | 3e-5 | 1.0e-2 → 4.9e-4 | 2.4e-2 → 1.4e-3 |

The facet-normal row does not converge, and the leak cannot see it: at `1e3` that
penalty leaks 3e-2 and gets the surface stress right to 2e-3, while at `1e8` it
leaks 1e-7 and is 26% wrong. This is the classical over-constraint that the
*consistent* normal was introduced to avoid (Engelman, Sani & Gresho 1982), and
`mesh.boundary_normal` is that normal.

**When to use:**
- Quick prototyping where high accuracy isn't critical
- A solver that has neither the rotated nor the Nitsche condition

**Limitations:**
- Penalty must be tuned: too small → loose constraint, too large → ill-conditioning
- On spherical shells, penalty can become unstable at moderate resolution
- Check the answer, not just the leak: a constraint that is satisfied is not
  evidence that the solution is right


### 4. Projected Normals (Superseded by `mesh.boundary_normal`)

`mesh.boundary_normal(boundary)` assembles the measure-weighted node normal
directly, tracks mesh deformation, and is correct in parallel, so the recipe
below is kept for reference rather than recommended. Project `mesh.Gamma` onto a
continuous mesh variable, which interpolates and smooths the normals:

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

**Accuracy:** on the elliptical annulus below, the penalty with projected normals
is 0.10 % and 0.023 % from the reference at cell sizes 0.1 and 0.05 (rerun,
November 2026). `mesh.boundary_normal` gives 0.19 % and 0.045 % (#1168).

**Why it works:** The projection solves a weak-form problem that naturally smooths the discontinuous facet normals into a continuous field. The finite element basis functions interpolate between facets, approximating the true surface direction.


### 5. Analytical Normals (Penalty Only)

For the penalty condition, the surface normal can be derived from the
mathematical definition of the boundary:

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
- A penalty condition on a known geometric shape (ellipses, superellipses,
  spheroids), for example as a reference in a benchmark

**Accuracy:** on the elliptical annulus below it agrees with the penalty with
`mesh.boundary_normal` to 0.19 % and 0.045 % at cell sizes 0.1 and 0.05.

```{warning}
This applies to the penalty only. Do not pass the analytical normal to
`add_rotated_freeslip_bc(..., normal=...)` on a faceted curved wall: there it
makes the answer worse, not better (approach 1, issue #1168).
```


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

The elliptical annulus of the worked example (outer semi-axes 1.5 and 1, inner
circle of radius 0.5, radial body force `cos(4 theta)`, P2/P1, free slip on both
walls). There is no exact solution for this geometry; the reference is the
penalty (coefficient 1e4) with the ellipse's own normal. The measure is the
relative L2 difference of the nodal velocity from the reference. Measured in
November 2026 on two machines independently, which agree to the printed figures
(table in issue #1168; the projected-normal row was measured on one of them only):

| Wall condition | Cell 0.1 | Cell 0.05 |
|---|---|---|
| Rotated free slip, default normal | 0.19 % | 0.19 % |
| Penalty, `mesh.boundary_normal` | 0.19 % | 0.045 % |
| Penalty, projected normals | 0.10 % | 0.023 % |
| Penalty, `mesh.Gamma` (facet) | 32.6 % | 15.1 % |
| Nitsche, `normal=` the ellipse's own | 19.5 % | 11.3 % |
| Nitsche, default | 35.2 % | 39.5 % |
| Nitsche, `gamma = 1000` | 46.9 % | 46.2 % |
| Rotated free slip, `normal=` the ellipse's own | 43.7 % | 44.6 % |

The first three agree with each other to 0.2 % or better. The facet-normal
penalty and Nitsche with the ellipse's own normal close as the cells shrink. The
last three do not close and carry a large constant pressure; their cause is open
(issue #1168).

---

## When Accuracy Matters

The error in boundary normals affects:

1. **Free-slip conditions**: Incorrect normals allow spurious tangential flow
2. **Heat flux (Neumann) conditions**: Wrong direction for flux specification
3. **Traction boundary conditions**: Force direction errors
4. **Stress computations**: Errors in computed surface stresses

For many geodynamics applications, the error from raw facet normals (32.6 % and 15.1 % on the ellipse above) may be acceptable for exploratory work but problematic for:
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
enforcement than no constraint at all. (That test predates the correction
of the Nitsche pressure coupling, #1145 of November 2026, and has not been
repeated since; the reason given here does not depend on it.)

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

1. **Start with rotated free slip**: `stokes.add_rotated_freeslip_bc(0, "Upper")` — nothing to tune; keep `add_nitsche_bc` for a condition that changes during the run
2. **For penalty BCs, always normalize**: Analytical formulas need explicit normalization
3. **Check orientation**: Ensure normals point outward (use `sign(r.dot(normal))`)
4. **Verify visually**: Plot the normal field to catch errors
5. **Leave the normal to Underworld3 for rotated free slip**: on a faceted curved wall a given `normal=` makes the answer worse (issue #1168); analytical normals are for penalty references
6. **Custom constraint direction**: Use `direction=` parameter when the constraint
   direction differs from the surface normal (faults, basal shear)

---

## Further Reading

- [Custom Mesh Creation](custom-meshes.md) — Creating elliptical and complex meshes
- [Stokes Ellipse Example](../examples/fluid_mechanics/intermediate/Ex_Stokes_Ellipse_Cartesian.py) — Complete worked example
- [Rotated strong free-slip](../developer/subsystems/rotated-freeslip.md) — How the rotated constraint and its reaction work
- Underworld technical note 2026-016, *Free-slip boundary conditions on curved boundaries* — The annulus comparison quoted on this page (its Nitsche numbers are being corrected for #1145)
- Sime & Wilson (2020), [arXiv:2001.10639](https://arxiv.org/abs/2001.10639) — Nitsche free-slip for geodynamics

---

*Last updated: November 2026*
