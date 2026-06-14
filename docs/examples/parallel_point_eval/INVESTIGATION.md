# Parallel point-evaluation on a 2-manifold mesh — investigation

**Status (2026-05-22):** investigation, decision, prototype, and
end-to-end validation. Worktree `feature/parallel-point-eval` carries
all patches (uncommitted — implementation will be a separate PR).

---

## The spec

Given a numpy array of `N` global coordinates in 3-space — each of
which lives in a cell owned by some rank in a `P`-rank MPI job —
return the value of an expression at each of those coordinates back
to the *calling* rank. The coordinate array has shape `(N, cdim)`,
where `cdim` is the embedding dimension (3 for a 2-manifold in 3-space).

This is the gateway that `SemiLagrangian_DDt.update_pre_solve`
(`systems/ddt.py:1972, 2046`) needs in order to do SL trace-back on
a manifold mesh.

## Final architecture (validated)

Three layered pieces, all gated on `dim != cdim` so volume meshes are
completely unaffected:

**1. Eight `self.dim → self.cdim` reshapes / shape-checks** in
`swarm.py`, `_function.pyx`, `discretisation_mesh.py`. These were
hardcoded `self.dim` that should have been `self.cdim` from the
start; no-op on volume meshes.

**2. `SwarmVariable.__init__` VECTOR vtype branch honors caller's
`size`.** Previously `num_components = mesh.dim` regardless of the
explicit `size` argument; on a manifold this silently downgraded the
coord field to bs=dim. Fix: when `size == mesh.cdim`, use cdim.
~10 lines.

**3. Generalised edge-perpendicular construction at the two control-
point sites** (`_mark_faces_inside_and_out`,
`_mark_local_boundary_faces_inside_and_out`). In 2-D the perpendicular
is `(-edge_y, edge_x)` (i.e. z-hat × edge). On a 2-manifold the
implicit z-hat is replaced by the explicit cell normal:
`cell_normal × edge_vector`. Same algorithm, same kdtree, same sign-
flip — one extra cross-product per face. ~25 lines.

**4. Navigation-only auxiliary DM** (`_nav_dm`). On manifold meshes,
clone the main DM and apply `distributeOverlap(1)` *only on the clone*.
The clone is used solely to build the navigation kdtree, the in-cell
control points, and the cell-owner mask. The solver / FE assembly
DM stays non-overlapped because PETSc's stock FE assembly
(`DMPlexSNESComputeResidualFEM`) integrates over all local cells and
sums via LocalToGlobal+ADD_VALUES, which double-counts contributions
at the partition seam on an overlapped DM. Keeping FE off the
overlap path preserves accuracy; keeping navigation on the overlap
path resolves the orphan/contested-cells problem at partition seams.
~120 lines (DM clone + nav-coord array + threading `nav_dm` and
`nav_coords` through the four navigation methods).

**5. Ownership filter** (`_get_owned_cells_mask`). Even with the nav
DM populated, an in-cell test that accepts ghost cells creates a
contested-cell problem — multiple ranks claim the same query point.
Filter the in-cell test to accept only owned cells; rejection on
the receiver-side then drives the existing migrate iteration to
route to the actual owner. Uses the nav DM's PointSF leaves to
identify ghost cells. ~15 lines.

**6. Halo-edge filter** on boundary-face detection. With overlap,
the outer edge of the partition halo has faces whose unique bounding
cell is a ghost — these are not real domain boundaries, but the
naive `getJoin(face).shape[0] == 1` test classifies them as such.
Filter to faces whose single bounding cell is *owned* locally. ~6 lines.

## Validation results

`probe_real_uw3_path.py` — `uw.function.global_evaluate` on
`SphericalManifold(cellSize=0.3)` through the actual UW3 pipeline:

| Configuration | Result |
|---|---|
| 1-rank | 25/25 finite, max\|err\| = 1.56e-2 (FE-interp accuracy ✓) |
| 2-rank | rank 0/1: 25/25 finite, max\|err\| = 1.6e-2 / 2.2e-2 ✓ |

`probe_ownership_resolution.py` — 60 surface queries on the same
mesh, classified by how many ranks claim each:

| Configuration | unique owner | contested | orphan |
|---|---|---|---|
| 2-rank (final) | **60/60** | 0 | 0 |

`probe_manifold_solver.py` — Helmholtz solver
`(-Δ_S + I) T = z` on `SphericalManifold(cellSize=0.2)`, P2 elements,
analytic `T = z/3`:

| Configuration | rel L2 |
|---|---|
| 1-rank | 7.7e-3 (FE-interp accuracy ✓) |
| 2-rank (final) | rank 0/1: 7.7e-3 / 6.3e-3 ✓ |

`probe_volume_regression.py` — 2-D / 3-D `UnstructuredSimplexBox`:

| Configuration | Result |
|---|---|
| 1- and 2-rank 2-D | 30/30 finite, max\|err\| = 1.3e-3 ✓ |
| 1- and 2-rank 3-D | 30/30 finite, max\|err\| = 8.9e-3 ✓ |

Tier-A test suite (`test_0001_meshes`, `test_0003_swarm_variable_constraints`,
`test_0110_basic_swarm`, `test_0111_swarm_lifecycle`,
`test_0100_backward_compatible_data`, `test_0101_kdtree`):
**36/36 pass.**

## Why a navigation-only DM, not solver overlap

We tested overlap on the solver DM. The Helmholtz solve converged
to a rel L2 error of 0.10 — 15× worse than serial — because PETSc's
stock `DMPlexSNESComputeResidualFEM` integrates over
`DMPlexGetAllCells_Internal` (all local cells, including ghosts) and
LocalToGlobal-with-ADD_VALUES sums contributions for shared DOFs from
*both* the owner-rank and the ghost-receiver-rank, double-counting.

PETSc's own SNES tutorials never use overlap on the solver DM — only
on the *preconditioner* (`-pc_asm_overlap`, `-pc_asm_dm_subdomains`).
That matches this finding: PETSc-FE-on-overlap requires extra
configuration (cell-ownership label + filtered assembly) that isn't
exposed cleanly in petsc4py and isn't demonstrated in any tutorial.

The two-DM split — main DM non-overlapped for the solver, clone
overlapped for navigation — keeps both paths simple and preserves
FE accuracy. The navigation DM is a clone, not a separate mesh: it
shares vertex coordinates with the main DM where they overlap, and
adds the partition halo as additional cells with their own coords.

## Why not the smaller alternatives

The investigation considered five candidate primitives. All lose to
the chosen architecture:

- **PetscSF "swarm-lite":** ~150 LoC for a replacement transport
  whose ownership-lookup primitive is the same kdtree as the existing
  pipeline. No correctness or scaling improvement.
- **PETSc PIC swarm fix** (`swarm.c:1634` use `DMGetCoordinateDim`):
  orthogonal — UW3 uses BASIC swarms for evaluation. Independent MR
  for any future PIC-on-manifold use case.
- **DMInterpolation parallel:** O(N·P) communication, wrong scaling.
  Also routes through `DMLocatePoints` (currently very slow).
- **VecScatter:** strict superset of PetscSF complexity.
- **Per-rank-mean centroid kdtree only (no overlap):** results in 1/25
  partition-boundary points returning wild (O(0.3) magnitude error)
  on 2-rank manifold. ~4% miss rate for SL trace-back, unacceptable.
- **Solver-DM overlap:** double-counts FE assembly contributions at
  partition seam (validated empirically — 15× regression in
  Helmholtz error).

## What "works" means now

- **`global_evaluate`** on a 2-rank `SphericalManifold` returns the
  correct value at every query, to FE-interpolation accuracy.
- **Steady Helmholtz / Poisson solver** on the same mesh returns the
  analytic answer to FE-interpolation accuracy.
- **Time-dependent diffusion** via iterated Helmholtz (backward-Euler
  per step): the Y_10 mode decays as `exp(-2κt)` to within FE-interp
  accuracy on a 2-rank parallel run. Identical to serial within
  numerical noise. No vector solver involved — all the time-stepping
  goes through `SNES_Scalar` which is already cdim-clean.
- **Volume meshes** are completely unaffected — no overlap, no nav
  DM, no behaviour change.
- **Volume-mesh tier-A regression suite** passes unchanged
  (`test_0001_meshes`, `test_0110_basic_swarm`, `test_0101_kdtree`,
  `test_1010_stokesCart`: 36/36 pass).

## What doesn't yet work (follow-up PR scope)

- **`AdvDiffusionSLCN` end-to-end on a manifold**: the SLCN scheme's
  flux-history term `DFDt` is a projection of `-κ∇T` (a cdim-component
  vector) onto a mesh variable. It currently uses
  `SNES_Vector_Projection` (a `SNES_Vector` subclass), which has
  pre-manifold `mesh.dim`-vs-`mesh.cdim` assumptions throughout its
  F0/F1 shape checks, Jacobian loops, and PETSc-FE attachment.
  A partial cdim plumbing (lines 2502 and 2745 of
  `petsc_generic_snes_solvers.pyx`) gets the solver further but
  hits a Vec-size mismatch downstream — the audit is incomplete.

- **The architectural unlock for that follow-up** is
  `SNES_MultiComponent_Projection`. The flux projection is block-
  diagonal across components (no cross-coupling), so a
  multi-component scalar-style projection with `n_components = cdim`
  is mathematically the right fit. `SNES_MultiComponent` currently
  has an explicit `raise ValueError("currently assumes mesh.cdim ==
  mesh.dim")` guard (`petsc_generic_snes_solvers.pyx:3336`), but
  the guard reads as a precaution rather than a load-bearing
  invariant — `n_components` is already independent of `mesh.dim`
  in the class. Lifting the guard, swapping the projection class in
  `SemiLagrangian_DDt` for the manifold VECTOR case, and verifying
  the resulting flux history would close the SLCN-on-manifold story.

- **Stokes on a manifold** — same root cause as SLCN: needs
  `SNES_Stokes_SaddlePt` / `SNES_Vector` cdim plumbing. The vector
  velocity field on a 2-manifold has cdim=3 components (embedded
  with implicit tangency); the existing solvers assume dim=2 for
  the vector size. Same audit as SLCN; would naturally land in the
  same follow-up PR.

For each of these, the cdim audit needs to classify each `mesh.dim`
site as either **tangent-space** (keep `dim`, e.g. FE element
topology, number of spatial derivative directions in the tangent
plane) or **embedded-vector** (move to `cdim`, e.g. vector component
count, gradient shape in 3-space). On volume meshes these are equal
so all changes are no-op there.

## Open follow-ups (separate PRs)

- **Bounded-manifold test.** All `SphericalManifold` testing here is
  on a *closed* manifold (no boundary curve). The bounded case
  (`extract_surface` with a partial label, or a gmsh-built spherical
  cap, or eventually a real geographic patch) exercises the
  Site B generalised perpendicular construction and the
  near-boundary kdtree branch in `points_in_domain`. Must be added
  before merging.
- **Mesh adaptation interaction.** The nav DM is built once at mesh
  construction. If `mesh.adapt()` or coord changes invalidate the
  navigation indices, the nav DM also needs rebuilding. Already
  has cache invalidation hooks via `_mesh_version`; verify the
  nav-DM rebuild path is exercised.
- **PETSc MR for `swarm.c:1634`** — independent of this work.
- **DMPlex point-location performance** — separate investigation.
- **Issue #197** (`extract_region` broken on `development`) — outstanding.

## Files in this directory

- `INVESTIGATION.md` — this document.
- `prototype_scatter_gather.py` — clean-room demo of the
  scatter-evaluate-gather pattern on a volume mesh. Architectural
  sketch.
- `probe_real_uw3_path.py` — `uw.function.global_evaluate` on
  `SphericalManifold` through the actual UW3 pipeline. The headline
  regression probe.
- `probe_volume_regression.py` — `uw.function.global_evaluate` on
  2-D and 3-D `UnstructuredSimplexBox` meshes; confirms no
  behaviour change on volume meshes.
- `probe_ownership_resolution.py` — classifies surface queries as
  unique-owner / contested / orphan to verify the navigation logic.
- `probe_orphan_trace.py` — per-orphan diagnostic showing closest
  local cell on each rank, ownership status, and in-cell-test result.
- `probe_internal_call.py` — compares public `points_in_domain`
  output against the underlying `_get_closest_local_cells_internal`
  return; used to isolate the halo-edge boundary-face bug.
- `probe_meshvar_under_overlap.py` — confirms MeshVariable `data`
  contract is unchanged by overlap.
- `probe_manifold_solver.py` — Helmholtz solver smoke test on
  `SphericalManifold`; quantifies the FE-overlap interaction and
  the two-DM-split fix.

Run the headline probes:

```
mpirun -n 2 python docs/examples/parallel_point_eval/probe_real_uw3_path.py
mpirun -n 2 python docs/examples/parallel_point_eval/probe_ownership_resolution.py
mpirun -n 2 python docs/examples/parallel_point_eval/probe_manifold_solver.py
mpirun -n 2 python docs/examples/parallel_point_eval/probe_volume_regression.py
```
