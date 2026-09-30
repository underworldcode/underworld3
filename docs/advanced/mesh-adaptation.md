---
title: "Mesh Adaptation"
---

# Adaptive Mesh Refinement

Underworld3 provides **metric-based adaptive mesh refinement (AMR)** through integration with MMG/PETSc. This enables dynamic refinement near features of interest (faults, boundaries, steep gradients) while keeping element counts manageable.

```{note}
AMR features require the `amr` environment with custom PETSc build. See [Environment Setup](#environment-setup) below.
```

## Quick Start

```python
import underworld3 as uw

# Create initial mesh
mesh = uw.meshing.StructuredQuadBox(elementRes=(20, 20), minCoords=(0, 0), maxCoords=(1, 1))

# Create a refinement metric based on a field gradient
temperature = uw.discretisation.MeshVariable("T", mesh, 1)
# ... initialize temperature field ...

# Refine where gradients are steep
metric = uw.adaptivity.metric_from_gradient(
    temperature,
    h_min=0.01,    # Fine mesh where gradient is high
    h_max=0.1,     # Coarse mesh where gradient is low
)

# Adapt the mesh
mesh.remesh(metric)
```

## Core Concepts

### What is Metric-Based Adaptation?

In metric-based AMR, you specify a **target edge length** (h-field) at each point in the mesh. The adaptation algorithm then modifies the mesh to achieve these targets:

- **Smaller h** → finer mesh (more elements)
- **Larger h** → coarser mesh (fewer elements)

The metric tensor encodes these targets in a form the adaptation algorithm understands:

$$M = h^{-2} \cdot I$$

where $h$ is the target edge length and $I$ is the identity matrix.

### How Adaptation Works

The algorithm iteratively modifies the mesh so that all edges have **unit length in metric space**:

$$\mathbf{e}^T M \mathbf{e} \approx 1$$

for each edge vector $\mathbf{e}$. Edges that are too long get subdivided; regions with edges too short may be coarsened.

```{tip}
**Key insight**: Higher metric values produce finer mesh. If you want 10× refinement, the metric values should be ~100× larger (since $M \propto 1/h^2$).
```

### Two families of mesh adaptation

UW3 offers **two complementary** ways to put resolution where it is
needed:

| | `mesh.remesh(...)` (this page) | `uw.meshing.node_redistribution(...)` |
|---|---|---|
| Mechanism | **Re-mesh** (MMG): insert/remove/retriangulate | **Redistribute** the existing nodes (move only) |
| Node budget | *Changes* — targets an **absolute** edge length `h` | **Fixed** — relative redistribution to a target *density* |
| Topology | New mesh, **variables reset** (must transfer) | **Unchanged** — variables, DOFs, partition preserved |
| Grading reach | Strong (can add nodes → ~10×) | Capped by the node count (~1.5–2×) |
| Cell shape | Isotropic (`M = h⁻²I`) | **Anisotropic** — cells aligned to the feature |
| Cost | Re-mesh + full variable transfer | A few cheap SPD elliptic solves (no re-mesh) |
| Parallel | MMG re-partition | O(N), GAMG-parallelisable, no transfer |

Use `mesh.remesh` when you need a genuinely finer mesh (more
elements) and can afford to rebuild the problem. Use the
**node-snuggling** redistribution when you want to *reshape* the
existing mesh toward a feature every timestep cheaply, keeping the
topology (and all fields) intact — see the **Node redistribution**
section below.

---

## Metric Creation Functions

Underworld3 provides several ways to create adaptation metrics:

### From Field Gradients

Refine where a field is changing rapidly:

```python
metric = uw.adaptivity.metric_from_gradient(
    field,              # MeshVariable (scalar)
    h_min=0.005,        # Edge length where gradient is highest
    h_max=0.05,         # Edge length where gradient is lowest
    profile="linear",   # or "smoothstep", "power"
)
```

**Use cases:**
- Temperature-driven adaptation (refine thermal boundary layers)
- Strain rate adaptation (refine shear zones)
- Composition gradients (refine material interfaces)

**Profiles:**
- `"linear"`: Direct mapping from gradient to h
- `"smoothstep"`: Smooth S-curve transition
- `"power"`: $h \propto |\nabla\phi|^{-1/2}$ for error equidistribution

### From Indicator Fields

Refine based on any scalar indicator:

```python
metric = uw.adaptivity.metric_from_field(
    indicator,          # MeshVariable (scalar)
    h_min=0.01,         # Edge length where indicator is high
    h_max=0.1,          # Edge length where indicator is low
    invert=False,       # If True, high indicator → coarse mesh
)
```

**Use cases:**
- Error estimates (refine high-error regions)
- Phase fields (refine at interfaces)
- Distance fields (refine near surfaces)

### From Surface Distance

For embedded surfaces (faults, horizons), use the Surface class:

```python
# Create and discretize surface
fault = uw.meshing.Surface("fault", mesh, fault_points)
fault.discretize()

# Create metric that refines near the fault
metric = fault.refinement_metric(
    h_near=0.01,        # Fine mesh near fault
    h_far=0.1,          # Coarse mesh far away
    width=0.2,          # Transition zone width
    profile="linear",   # or "smoothstep", "gaussian"
)
```

### Direct h-field

If you have a custom h-field (array of target edge lengths):

```python
# Compute h-values however you like
h_values = my_custom_h_field(mesh)  # Shape: (n_nodes,)

# Convert to metric
metric = uw.adaptivity.create_metric(mesh, h_values)
```

---

## Controlling Element Count

### The Challenge

Refining near features increases element count. To maintain roughly constant total elements:

| Region | Strategy |
|--------|----------|
| Near feature | Refine (small h) |
| Far from feature | Coarsen (large h) |

The small refined region is compensated by the large coarsened region.

### Practical Approach

1. **Start coarser**: Use initial mesh with ~2× larger cells than your final uniform target
2. **Set h_far slightly larger than initial**: Allows coarsening far from features
3. **Set h_near for desired refinement**: Ratio `h_far/h_near` gives refinement factor

**Example**: 10× refinement near fault, constant element count

```python
params = uw.Params(
    # Initial mesh: 0.5 km cells (coarser than 0.25 km uniform target)
    uw_cell_size_km = 0.5,

    # Adaptation: 10× refinement near fault
    uw_adapt_h_near_km = 0.05,   # 10× finer than initial
    uw_adapt_h_far_km = 0.9,     # Slight coarsening elsewhere
    uw_adapt_width_km = 1.5,     # Transition zone
)
```

### Element Count Table

| h_near/h_far | Refinement Factor | Metric Ratio |
|--------------|-------------------|--------------|
| 0.5 | 2× | 4× |
| 0.1 | 10× | 100× |
| 0.05 | 20× | 400× |
| 0.01 | 100× | 10,000× |

---

## Complete Example: Fault-Adaptive Shear Box

```python
import numpy as np
import underworld3 as uw

# Units
u = uw.scaling.units

# Create initial mesh (coarser to allow refinement room)
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(40, 20),
    minCoords=(0, 0),
    maxCoords=(20, 10),  # 20 km × 10 km domain
)

# Define fault as a horizontal line at y = 5 km
fault_points = np.array([
    [5.0, 5.0],
    [15.0, 5.0],
])

# Create surface
fault = uw.meshing.Surface("fault", mesh, fault_points)
fault.discretize()

# Create refinement metric
# 10× refinement near fault, coarsen elsewhere
metric = fault.refinement_metric(
    h_near=0.05,    # ~50 m elements near fault
    h_far=1.0,      # ~1 km elements far from fault
    width=2.0,      # 2 km transition zone
    profile="linear",
)

# Adapt mesh
mesh.remesh(metric)

# Check result
print(f"Adapted mesh: {mesh.X.coords.shape[0]} nodes")

# Now set up physics on the adapted mesh...
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
# ... continue with Stokes solve, etc.
```

---

## Variable Handling After Adaptation

### Variables Are Reset

After `mesh.remesh()`, all variables on the old mesh become invalid. Variables on the new mesh start uninitialized.

**For analytical initialization:**
```python
# Before adaptation
T_init = sympy.sin(x) * sympy.exp(-y)

# After adaptation - reinitialize from expression
T.array[:, 0, 0] = uw.function.evaluate(T_init, mesh.X.coords)
```

**For data transfer (checkpoint/restart):**
```python
# Create variable on new mesh
T_new = uw.discretisation.MeshVariable("T", new_mesh, 1)

# Transfer from old mesh
uw.adaptivity.mesh2mesh_meshVariable(T_old, T_new)
```

### Surface Variables

Surface variables (e.g., fault friction) are automatically marked stale after adaptation and recomputed lazily when accessed.

---

## Troubleshooting

### Memory Errors

```
## Error: unable to allocate larger solution.
## Check the mesh size or increase maximal authorized memory with the -m option.
```

**Solutions:**
- Use more moderate refinement ratios
- Start with coarser initial mesh
- Increase available memory

### Slow Adaptation

Adaptation time scales with:
- Initial element count
- Refinement ratio
- Boundary complexity

**Tips:**
- 2D adaptation is typically fast (seconds)
- 3D can take minutes for complex geometries
- Consider adaptive time-stepping (adapt every N steps, not every step)

### Boundary Issues

If boundaries are lost or corrupted after adaptation:
- Check that boundary labels are defined correctly
- Verify boundaries enum matches mesh boundaries
- The stacking/unstacking should handle this automatically

---

## Environment Setup

AMR requires a custom PETSc build with MMG support:

```bash
# Install the AMR environment
pixi install -e amr

# Build custom PETSc (takes ~1 hour)
pixi run -e amr petsc-build

# Verify installation
pixi run -e amr python -c "import underworld3; print('AMR ready')"
```

**Included libraries:**
- **MMG** (mmg2d, mmg3d): Metric-based mesh adaptation
- **ParMMG**: Parallel mesh adaptation
- **Pragmatic**: Alternative adaptation library

---

## API Reference

### Core Functions

| Function | Purpose |
|----------|---------|
| `mesh.remesh(metric)` | Re-mesh (in place) using metric field |
| `uw.adaptivity.create_metric(mesh, h)` | Convert h-field to metric |
| `uw.adaptivity.metric_from_gradient(field, ...)` | Metric from field gradient |
| `uw.adaptivity.metric_from_field(indicator, ...)` | Metric from indicator field |
| `uw.adaptivity.mesh2mesh_meshVariable(old, new)` | Transfer variable data |

### Surface Methods

| Method | Purpose |
|--------|---------|
| `surface.discretize()` | Create triangle mesh from points |
| `surface.distance` | MeshVariable with signed distance |
| `surface.refinement_metric(...)` | Create distance-based metric |

---

## Mathematical Details

For the mathematically inclined, see the [Developer Design Document](../developer/design/ADAPTIVE_MESHING_DESIGN.md) which covers:

- Metric tensor formulation
- Dimension-independent derivation
- Boundary label stacking algorithm
- Gradient computation (Clement interpolant)
- Convergence properties

---

## Node redistribution — the snuggling mover

When you want to concentrate resolution on an evolving feature
**every timestep** without re-meshing — keeping the topology and
all field data intact — use **node redistribution** instead of
`mesh.remesh`. The purposeful spelling is
`uw.meshing.node_redistribution(mesh, metric)` (equivalently the
mesh-controlled method `mesh.redistribute_nodes(metric)`); it names
the capability — the algorithm behind it (the Huang–Kamenski MMPDE
mover) is an implementation detail documented on the machinery,
`smooth_mesh_interior`:

```python
import underworld3 as uw
from underworld3.meshing import (
    node_redistribution, metric_density_from_gradient)

# ... mesh + a temperature field T after some solve ...

# Relative target DENSITY from |∇T| (the fixed-node-budget
# analogue of metric_from_gradient: same percentile-window idea,
# but ρ is a *density*, not an absolute h — there is no node
# budget to spend on an absolute size).
rho = metric_density_from_gradient(mesh, T, amp=8.0)

# Move the nodes to that metric (topology / DOFs / variables
# all preserved — no transfer needed).
node_redistribution(mesh, rho, boundary_slip=True)
```

Node redistribution is implemented for **2D simplex (triangle)
meshes**; other mesh types (quad/hex, 3D, manifolds) raise an
honest `NotImplementedError` stating what exists. To *add*
resolution locally instead of moving it, use the nested
adapt-on-top: `child = mesh.adapt(metric, max_levels=2)` — no
`engine=` needed (the graded newest-vertex-bisection engine is the
default on 2D meshes; `engine=` remains as an advanced selector).

```{tip}
**`method="mmpde"` is the default mover** (since this release): the
variational moving-mesh adaptation of Huang & Kamenski. It is
matrix-free (no PETSc solve — small per-cell dense algebra plus a
parallel `Vec` assembly), provably non-folding, currently 2D
(triangle meshes; the method itself is dimension-general but the
3D discretization is not implemented), and genuinely
*clusters and aligns* to an **anisotropic tensor** metric. It is
both the most capable and the most straightforward to reason about,
which is why it is now the default. Pass a **scalar** density (as
above; it is promoted to the isotropic tensor `ρ·I`) or a `d×d`
**tensor** metric (e.g. {py:func}`fault_metric_tensor`, or a
`grad T`-aligned boundary-layer tensor) to get true thin-across /
long-along refinement. Full design + derivation:
{doc}`/developer/design/anisotropic-mmpde-mover`.

The earlier movers — `"spring"` (volumetric equant-cell smoother),
`"ma"` (isotropic Monge–Ampère), `"ot"` (linear OT-improvement
step) and `"anisotropic"` (decoupled-Winslow tensor smoother) —
were **retired in 2026-07**: `"mmpde"` with a scalar metric
reproduces their isotropic equidistribution, and with a tensor
metric it clusters and aligns where they could not. The retired
spellings now raise a `ValueError` pointing here.

Key `mmpde` knobs (via `method_kwargs`): `p` (functional exponent,
1.5–2), `theta` (Huang alignment/equidistribution balance, 1/3),
`step_frac` (per-node move cap, 0.2), `tol` (scale-relative
convergence exit), `metric_eval` (`"rbf"` default — fast baked
metric interpolation). `boundary_slip=True` lets boundary nodes
slide tangentially (needed for surface-reaching features).
```

`metric_density_from_gradient` builds
$\rho = 1 + \mathrm{amp}\cdot t$, $t = \mathrm{clip}\big((|\nabla
T| - g_{lo})/(g_{hi}-g_{lo}),0,1\big)$ with $g_{lo},g_{hi}$ the
lo/hi percentiles of $|\nabla T|$ — deliberately the same shape as
{py:func}`underworld3.adaptivity.metric_from_gradient`, so the
*intent* you express is identical whichever family you choose.

```{important}
This is a **gradient** metric: it resolves where the field
*changes* (boundary layers, fronts, plume edges), and is
isotropic-coarse at a smooth *peak* ($\nabla\rho=0$) — it
deliberately de-refines a feature's core. For core resolution a
curvature (Hessian) metric is the (future) tool. It also does
**not** beat the fixed node-count cap — for a *separable* feature
the explicit 1-D OT is exact and cheaper; the mover earns its keep
on general non-separable features and on cell-alignment / quality
(it never produces slivers).
```

The full mathematical derivation (including the retired OT /
Monge–Ampère and Winslow movers, kept as an R&D record) is in
{doc}`/developer/design/mesh-adaptation-formulation`; operational
detail in {doc}`/developer/subsystems/mesh-metric-redistribution`;
the dated R&D log in
`docs/developer/design/ma-newton-cofactor-exploration.md`.

---

## References

1. MMG Platform: https://www.mmgtools.org/
2. Alauzet, F. "Metric-based anisotropic mesh adaptation" (2010)
3. PETSc DMPlex: https://petsc.org/main/manual/dmplex/

---

## Examples

Complete working examples are available in the examples directory:

- `docs/examples/fluid_mechanics/intermediate/shear_box_2d_fault_adaptive.py`
- `docs/examples/utilities/advanced/MeshRefine-AdaptiveMetric-Static.py`
- `docs/examples/utilities/advanced/MeshRefine-AdaptiveMetric-Dynamic.py`

These demonstrate real-world usage patterns including:
- Fault-based refinement
- Dynamic adaptation during time-stepping
- Image-based mesh density control
