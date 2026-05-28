# Fault refinement — the simplification

```{note}
Design note, 2026-05-28. Captures the convergence after the
feature/elliptic-ma fault-meshing work: one mover, one metric form, one
slip, 2D *and* 3D. The pieces this collapses (the anisotropic tensor
mover and the analytic-Eulerian per-segment machinery) remain present
for the moment but are scheduled for deprecation.
```

## The recipe

```python
import sympy, underworld3 as uw

rho_T = uw.meshing.metric_density_from_gradient(mesh, T, metric_choice="arc-length")
rho_F = uw.meshing.fault_comb_metric(mesh, faults, cell_size=dx, n_across=N)

uw.meshing.smooth_mesh_interior(
    mesh, method="ma",
    metric=[(rho_T, 1.0), (rho_F, w_F)],        # composable list (max-on-excess)
    boundary_slip=True,                          # generic topology slip
    method_kwargs=dict(n_outer=1, n_picard=25))  # single-shot
```

One mover (single-shot Monge–Ampère), one metric form (scalar density), one
composition operator (weighted max on the excess), one slip (topology-based
vertex normals). Works in **2D and 3D**, on Cartesian boxes, annulus,
sphere, polyhedra, curved surfaces.

## Why each piece

### Single-shot MA

`smooth_mesh_interior(method="ma", n_outer=1)` is the Caffarelli-clean
Monge–Ampère map: one solve, untangled by construction, no
outer-iteration compounding, nothing to tune. **No metric-rebuild
requirement** at `n_outer=1` — the target metric is evaluated once, on the
undeformed mesh, so a precomputed distance *field* is bit-equivalent to an
analytic one and the Eulerian/convection question never arises. (At
`n_outer>1` the mover re-queries the metric on the deformed mesh, so a
frozen field would convect — use an analytic metric there, or stick with
single-shot.)

### Scalar comb metric

`fault_comb_metric(mesh, faults, cell_size=dx, n_across=N)` places narrow
teeth at `d = 0, dx, 2 dx, …` from each fault's distance field.
Equidistribution drops a node row at each tooth → evenly-spaced rows ⇒ a
band of `~ N` roughly-uniform cells across each fault, **with the `d=0`
tooth pinning a row on the fault line** (so close faults centre to
~0.0002 — better than h-adapt with `mesh.adapt`).

For 2D faults the per-segment min-distance is analytic. For curved or
**3D triangulated** fault surfaces (`FaultSurface.compute_distance_field`,
kdtree-based), the comb is built directly on the precomputed distance
**field** — segment-count-independent JIT cost, and the natural input
for 3D where analytic point-to-triangulated-surface distance is hard.

### Composable list of metrics

`smooth_mesh_interior(metric=[(m_i, w_i), …])` composes internally via

$$\rho_{\text{combined}}(x) = 1 + \max_i\, w_i\,\big(\rho_i(x) - 1\big)$$

— "refine wherever any feature demands it," with weights scaling each
feature's demand cleanly. Scalar densities compose by `max` trivially;
metric *tensors* would need Alauzet metric intersection (much more
involved) — another reason scalar-MA is the convergence point.

### Generic topology-based tangent slip

`_boundary_vertex_normals(mesh)` computes outward unit normals at each
boundary vertex *geometrically* from the cell coordinates (boundary
facets identified topologically, normals area-weighted averaged). It
classifies each vertex as **face-slip** (all incident facet normals
within ~15° of the average — slides tangentially) or **pinned**
(corners, 3D edges between faces). Works on **any** simplicial mesh.

This replaces the old `Gamma_P1`-based slip, which evaluated PETSc's
`petsc_n` quadrature symbol at *vertices* (undefined off boundary
quadrature points) — radial mesh classes worked around it by
redefining `Gamma` as the analytic radial unit vector, but Cartesian
got garbage normals and was silently pinned.

### Dimension-general MA

`_winslow_elliptic` is now dimension-general (bit-identical at `cdim=2`):

* **Normalisation `c`** branches on the source's leading term:
  `c = 1/⟨b^{-1/2}⟩²` for the 2D convex radical, `c = 1/⟨b^{-1}⟩` for
  the 3D simple Picard. Wrong `c` made the source non-zero-mean and the
  pure-Neumann φ-Poisson unsolvable (the constant nullspace fixes
  *solution* ambiguity, not *RHS* inconsistency) — the actual cause of
  the previous 3D failure.

* **3D source**: `f_src = tr(H_s) + g − det(I+H_s)`
  (`H_s` symmetrised), restoring the 2×2 principal-minor terms the old
  `(g−1) − det(H)` dropped in 3D. Reduces to the 2D simple-Picard form
  exactly.

* **Tet signed-volume backtrack**: `_tri_cells` returns `None` for tets,
  so 3D previously had no anti-tangle guard. Added `_tet_cells` +
  `_signed_volumes` and a tet branch in the backtrack.

Validated on a 3D slab and spherical-shell adapt (refines toward the
feature, 0 inverted tets) and a 3D disk fault (the recipe above).

## What this collapses

The following remain in the codebase for the moment but are scheduled for
deprecation once external users have migrated:

| Component | Replaced by |
|---|---|
| `_winslow_anisotropic` (anisotropic tensor mover) | single-shot MA + comb |
| `fault_metric_tensor` (analytic 2×2 supplied tensor) | `fault_comb_metric` |
| `_winslow_anisotropic.supplied_D` entry point | (no need — comb is scalar) |
| Per-segment analytic min-distance for curved faults | `Surface.distance` / `FaultSurface.compute_distance_field` |
| Ring-projection slip on annulus + geometric box-slip | topology-based generic slip |

The `fault_metric` facade keeps `method="anisotropic"` and `method="adapt"`
(MMG) for the moment as documented alternatives — the recommended default
is `method="ma"`.

## Honest limits

* **Budget cap**: `r-adapt` (any mover, including MA) redistributes a *fixed*
  set of nodes — `cell_size` in `fault_comb_metric` is a *target*, not a
  guarantee. The realised cell sizes are roughly `~1.5–2.5×` finer than the
  base mesh per feature. To honour an absolute `cell_size`, use
  `mesh.adapt` (MMG) via `fault_metric(method="adapt")` — but that *adds*
  nodes (topology changes, disturbing particle workflows).

* **Composed multi-feature budgets compete**: composing gradient(T) with a
  fault sends a fixed budget over two extended demands. Weights tune
  *who* wins; the base mesh resolution controls the absolute resolution
  each can reach.

* **Multi-iteration metric convection**: at `n_outer>1` the MA mover
  re-queries the target metric on the deformed mesh. Analytic metrics
  re-evaluate correctly (Eulerian); a frozen *field* metric (the field
  comb) convects and degrades. The recommended single-shot recipe
  sidesteps this entirely.

* **3D MA is the simple Picard, not a convex branch**: it converges
  cleanly on gentle metrics (validated on the slab, sphere shell, and
  disk fault) but could be fragile on very strong/sharp ones. The 2D
  convex-branch (BFO) path stays in place at `cdim=2`.

## Migration

For users of the now-deprecated paths:

* `smooth_mesh_interior(method="anisotropic", supplied_D=M, ...)` →
  `smooth_mesh_interior(method="ma", metric=fault_comb_metric(...))`
  (or via the list-of-metrics composition).

* `fault_metric_tensor` → `fault_comb_metric` (or `fault_metric(method="ma", ...)`).

* Hand-built `sympy.Max(...)` composition → pass `metric=[m1, m2, …]`
  to `smooth_mesh_interior`.

* Custom box-face slip code → just enable `boundary_slip=True`; the
  generic slip handles any geometry.

## References

* `src/underworld3/meshing/surfaces.py` — `fault_metric_tensor`,
  `fault_comb_metric`, `fault_metric`, `compose_metrics`.
* `src/underworld3/meshing/smoothing.py` — `_winslow_elliptic` (now
  dimension-general), `smooth_mesh_interior(metric=[...])`.
* `src/underworld3/meshing/_ot_adapt.py` — `_boundary_facets`,
  `_boundary_vertex_normals`, generic `_build_slip_projector`.
* `tests/test_0762_fault_metric_tensor.py` — 17 tier-A tests locking
  the new layer.
