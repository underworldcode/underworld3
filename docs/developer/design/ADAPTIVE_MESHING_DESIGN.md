# Adaptive Mesh Refinement Design Document

**Status**: Implemented
**Date**: January 2026
**Authors**: Louis Moresi, Claude Code

---

## Overview

Underworld3 provides adaptive mesh refinement (AMR) capabilities through integration with MMG/PETSc mesh adaptation algorithms. This document describes the design, implementation, and mathematical foundations of the system.

## Architecture

### Component Overview

```{mermaid}
flowchart TB
    subgraph API["User-Facing API"]
        direction LR
        A1["mesh.adapt(metric)"]
        A2["surface.refinement_metric()"]
        A3["uw.adaptivity.metric_from_*()"]
    end

    subgraph Metric["Metric System"]
        direction LR
        M1["create_metric()"]
        M2["metric_from_gradient()"]
        M3["metric_from_field()"]
    end

    subgraph PETSc["PETSc/MMG Integration"]
        direction LR
        P1["mesh_adapt_meshVar()"]
        P2["_dm_stack_bcs()"]
        P3["_dm_unstack_bcs()"]
    end

    subgraph External["External Libraries"]
        direction LR
        E1["MMG (mmg2d, mmg3d)"]
        E2["ParMMG"]
        E3["PETSc DMPlex"]
    end

    API --> Metric
    Metric --> PETSc
    PETSc --> External
```

### Key Files

| File | Purpose |
|------|---------|
| `src/underworld3/adaptivity.py` | Core adaptation functions and metric creation |
| `src/underworld3/meshing/surfaces.py` | Surface class with `refinement_metric()` method |
| `src/underworld3/discretisation/discretisation_mesh.py` | `mesh.adapt()` method |

---

## Metric Tensor Mathematics

### Fundamental Relationship

For isotropic mesh adaptation, MMG/PETSc uses a metric tensor:

$$M = h^{-2} \cdot I$$

where:
- $h$ is the target edge length
- $I$ is the identity matrix (2×2 in 2D, 3×3 in 3D)

This relationship is **dimension-independent** — the same formula applies in 2D and 3D because the metric defines edge lengths, not areas or volumes.

### How Adaptation Works

The adaptation algorithm seeks to make all edges have **unit length in metric space**:

$$\mathbf{e}^T M \mathbf{e} = 1$$

for edge vector $\mathbf{e}$. When an edge is longer than the target (in metric space), it gets subdivided. When shorter, elements may be merged.

### Key Insight: Metric Drives to Unity

During refinement, the mesh metric at each point is driven toward unity. This means:

- **Higher metric values** → finer mesh (smaller elements)
- **Lower metric values** → coarser mesh (larger elements)

The relationship between metric value $M$ and element density is:
- $M = 1/h^2$, so $h = 1/\sqrt{M}$
- Doubling $M$ reduces edge length by $\sqrt{2}$
- 100× larger $M$ gives 10× smaller edges

### Metric Value Interpretation

The metric values are **dimensionless** and relate to target edge lengths:

| h-value | Metric (M = 1/h²) | Element Size Relative to h=1 |
|---------|-------------------|------------------------------|
| 1.0 | 1.0 | Reference |
| 0.5 | 4.0 | 2× finer |
| 0.1 | 100.0 | 10× finer |
| 0.01 | 10,000.0 | 100× finer |
| 2.0 | 0.25 | 2× coarser |

---

## Metric Creation Functions

### create_metric(mesh, h_values, name)

The core function that converts an h-field (target edge lengths) to metric values:

```python
def create_metric(mesh, h_values, name=None):
    """Convert h-field to metric tensor format."""
    metric = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)
    with mesh.access(metric):
        # M = 1/h² (isotropic, dimension-independent)
        metric.data[:, 0] = 1.0 / (h_values ** 2)
    return metric
```

### metric_from_gradient(field, h_min, h_max, ...)

Creates adaptation metric from the gradient of a scalar field:

- **Steep gradients** → small h → large metric → finer mesh
- **Smooth regions** → large h → small metric → coarser mesh

Uses Clement interpolant for gradient computation:
```python
gradient_at_nodes = uw.function.compute_clement_gradient_at_nodes(field)
grad_mag = np.sqrt(np.sum(gradient_at_nodes ** 2, axis=1))
```

Profiles available:
- `"linear"`: Direct mapping from gradient to h
- `"smoothstep"`: S-curve transition (C¹ continuous)
- `"power"`: h ∝ |∇φ|^(-1/2) for error equidistribution

### metric_from_field(indicator, h_min, h_max, ...)

General-purpose metric from any indicator field:

- Error estimates
- Phase fields (refine at interfaces)
- Distance fields
- Material composition

### Surface.refinement_metric(h_near, h_far, width, ...)

Specialized for surface-based refinement:

```python
fault = uw.meshing.Surface("fault", mesh, fault_points)
metric = fault.refinement_metric(
    h_near=0.005,   # Fine mesh near surface
    h_far=0.05,     # Coarse mesh far away
    width=0.1,      # Transition zone
    profile="linear"
)
mesh.adapt(metric)
```

---

## Boundary Label Handling

PETSc's `DMPlexAdaptMetric` only interpolates **one boundary label** during adaptation. Underworld handles this transparently by using a unified `UW_Boundaries` label at mesh creation time, where different boundaries are distinguished by numerical values rather than separate label names.

This means boundary conditions are preserved automatically through adaptation — users don't need to manage label stacking manually.

---

## Element Count Control

### The Challenge

Refining near features (faults, boundaries) increases element count. To maintain roughly constant total elements, we need to **coarsen far from features**.

### Strategy

```
Total elements ≈ (refined region elements) + (coarse region elements)
              ≈ (small area × high density) + (large area × low density)
```

For a fault that affects ~10% of the domain with 10× refinement:
- Refined region: 0.1 × 100 = 10 units
- Coarse region: 0.9 × 0.5 = 0.45 units (if coarsened 2×)
- Total: ~10.45 vs original 1.0 × 1 = 1.0

To compensate, coarsen the far-field:
- Use `h_far > h_original`
- The refined region is small, so slight coarsening elsewhere compensates

### Empirical Calibration

In practice, metric values require empirical calibration:

| Parameter | Purpose | Typical Range |
|-----------|---------|---------------|
| `h_near` | Target edge length near feature | 0.1× to 0.01× of domain scale |
| `h_far` | Target edge length far from feature | 1× to 2× of original cell size |
| `width` | Transition zone width | 2× to 5× h_far |

**Example** (from shear_box_2d_fault_adaptive.py):
```python
params = uw.Params(
    uw_cell_size_km = 0.5,       # Initial mesh: coarser to allow refinement room
    uw_adapt_h_near_km = 0.05,   # 10× finer than initial
    uw_adapt_h_far_km = 0.9,     # Slight coarsening to compensate
    uw_adapt_width_km = 1.5,     # Transition zone
)
```

---

## Variable Handling After Adaptation

### The Problem

After mesh adaptation, variables on the old mesh cannot be used directly — the nodes have changed.

### Solution: Lazy Recomputation

Variables are marked "stale" after adaptation and recomputed lazily when accessed:

```python
class Surface:
    def _mark_all_proxies_stale(self):
        """Mark all variable proxies as needing recomputation."""
        for var in self._variables.values():
            var._proxy_stale = True
```

For MeshVariables, users must either:
1. Reinitialize from analytical functions
2. Use `mesh2mesh_meshVariable()` to transfer data

### mesh2mesh_meshVariable()

Transfers data between meshes using a swarm intermediary:

```python
def mesh2mesh_meshVariable(from_var, to_var):
    """Transfer MeshVariable data via swarm interpolation."""
    # Create temporary swarm on new mesh
    # Interpolate old field to swarm positions
    # Project swarm data to new mesh variable
```

---

## Performance Considerations

### Adaptation Cost

Mesh adaptation with MMG is typically:
- **2D**: Fast (seconds for 100k elements)
- **3D**: Slower (minutes for complex geometries)

The cost scales with:
- Initial element count
- Refinement ratio (h_max/h_min)
- Boundary complexity

### Memory Requirements

MMG requires memory for both old and new meshes during adaptation. For very large meshes or aggressive refinement, memory can be a bottleneck.

Error message to watch for:
```
## Error: unable to allocate larger solution.
## Check the mesh size or increase maximal authorized memory with the -m option.
```

**Solution**: Use more moderate refinement ratios or start with coarser initial mesh.

### Parallel Considerations

- `ParMMG` provides parallel adaptation
- Requires PETSc built with ParMMG support
- Domain decomposition is handled automatically

---

## Environment Requirements

AMR features require the `amr` environment with custom PETSc build:

```bash
# Install AMR environment
pixi install -e amr

# Build custom PETSc (takes ~1 hour)
pixi run -e amr petsc-build

# Run with AMR support
pixi run -e amr python my_adaptive_script.py
```

The custom PETSc includes:
- MMG (2D and 3D mesh adaptation)
- ParMMG (parallel adaptation)
- Pragmatic (alternative adaptation library)

---

## API Reference Summary

### Core Functions

```python
# Create metric from h-field
metric = uw.adaptivity.create_metric(mesh, h_values)

# Create metric from gradient
metric = uw.adaptivity.metric_from_gradient(field, h_min, h_max)

# Create metric from indicator
metric = uw.adaptivity.metric_from_field(indicator, h_min, h_max)

# Adapt mesh
new_mesh = mesh.adapt(metric)

# Transfer variable data
uw.adaptivity.mesh2mesh_meshVariable(old_var, new_var)
```

### Surface-Based Refinement

```python
# Create surface from points
surface = uw.meshing.Surface("fault", mesh, points)
surface.discretize()

# Get distance field
dist = surface.distance  # MeshVariable with signed distance

# Create refinement metric
metric = surface.refinement_metric(
    h_near=0.01,    # Fine near surface
    h_far=0.1,      # Coarse far away
    width=0.2,      # Transition zone
    profile="linear"
)

# Adapt
mesh.adapt(metric)
```

---

## References

1. MMG Platform documentation: https://www.mmgtools.org/
2. Alauzet, F. "Metric-based anisotropic mesh adaptation" (2010)
3. PETSc DMPlex documentation: https://petsc.org/main/manual/dmplex/
4. ParMMG: https://github.com/MmgTools/ParMmg

---

## Related Documents

- `docs/advanced/mesh-adaptation.md` — User guide
- `docs/api/adaptivity.md` — API reference
- `docs/developer/design/GEOGRAPHIC_COORDINATE_SYSTEM_DESIGN.md` — Coordinate handling for geographic meshes
- `docs/developer/design/mesh-geometry-audit.md` — Mesh geometry system

---

*Last updated: January 2026*
