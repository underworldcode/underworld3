---
title: "Persisting the FMG mesh hierarchy across checkpoints"
status: "Implemented (2026-06-11, commit 3cd73cde) — FMG coarse-hierarchy sidecar restore in discretisation_mesh.py"
---

# Persisting the geometric-multigrid hierarchy across checkpoints

## Problem

A mesh built with `refinement=N` carries a geometric refinement hierarchy in
`mesh.dm_hierarchy`, which the Stokes/scalar/vector solvers use for geometric
**Full Multigrid (FMG)** — the anisotropy-robust preconditioner of choice on
adapted meshes. But the checkpoint load path (`_from_plexh5`) reconstructs only
the single saved DMPlex, so a **reloaded mesh has `dm_hierarchy = [dm]`** (one
level) and FMG silently falls back to GAMG after a restart. This note records the
design that restores the hierarchy on reload, and the experiments that shaped it.

## Why store the coarse levels (not reconstruct, not refine-from-label)

Several approaches were prototyped and rejected:

- **Reconstruct the coarse mesh from a per-node "level" label.** The coarse
  *topology* is recoverable bit-exact from labelled nodes (the all-midpoint
  "central" fine cells map 1:1 to coarse cells). **But** a coarse DMPlex built
  from scratch (`createFromCellList`) does not reproduce the original's internal
  cone-orientation / edge ordering, and PETSc's nested multigrid interpolator
  needs the **canonical `refine()` numbering**. Splicing a reconstructed coarse
  in throws `PETSc has generated inconsistent data` (err 77).
- **Store only the coarsest level + `refine()` back up on reload.** Works for the
  hierarchy, but then the rebuilt fine has to be reconciled (numbering + field
  data) with the saved deformed mesh — fragile.

The winning insight: a **loaded** coarse DM preserves the canonical numbering
(`topologyLoad` is faithful), so it can be `setCoarseDM`-linked directly under the
working fine and FMG just works — **no refine, no node-moving, no reconstruction**.
This is exactly the live `clone_dm_hierarchy` pattern with *load* swapped for
*clone*. Validated: `refine(stored L0) == saved fine` bit-exact in 2D and 3D, and a
reloaded hierarchy drives `pc_type=mg` to convergence.

## On-disk format: a single coarsest sidecar

PETSc's `HDF5_PETSC` `DMView` writes to fixed top-level groups (`/topology`,
`/geometry`, `/labels`) — it is **not namespaced by DM name**. Writing a second
DMPlex into the same file (PETSc viewer append, *or* an h5py-injected subgroup
that the PETSc reader then ignores) corrupts the file (a reload BUS-errors). So
the hierarchy is stored in **one extra single-DM file** beside the main
checkpoint, holding only the **coarsest** level:

```
mymesh.h5                  # the working/fine mesh (unchanged, fully compatible)
mymesh.hierarchy.L0.h5     # coarsest level only
```

The intermediate coarse levels are not stored — on reload they are rebuilt by
`refine()`-ing the coarsest `N-1` times (they come back canonically numbered,
which is all the co-located nested interpolation needs). The main file's
`metadata` group gains `hierarchy_coarse_levels = N-1` (the refinement depth).
Old checkpoints (attribute absent) and plain meshes (no hierarchy) write no
sidecar and reload exactly as before.

## Reload and the link-free working `dm`

On reload the coarse levels are loaded and spliced:
`dm_hierarchy = [L0, …, L_{N-2}, fine]`, linked with `setCoarseDM`. One subtlety:
the mesh's **working `self.dm` must be a link-free clone** of the finest level
(mirroring the `refinement` construction branch). If `self.dm` itself carries a
coarse-DM link, `mesh.update_lvec()`'s `createFieldDecomposition` recurses into
the 0-field coarse levels and fails (`requested fields 1 > DM fields 0`). The
linked hierarchy lives in `dm_hierarchy`; the solver clones it
(`clone_dm_hierarchy`) for its own multigrid setup.

## Parallel: co-location via the Simple partitioner

Works in serial **and** parallel through the same reload path. The hazard in
parallel is that the coarse sidecars and the fine reload on **independent
partitions**; linking incompatibly-partitioned levels sends the interpolator into
a cross-rank point-location spin (observed before the fix: np=2, rank 0 at 99% CPU
indefinitely, rank 1 idle).

The fix needs no custom partition math. The fine carries the **canonical
refinement numbering** — coarse cell `c`'s children are fine cells
`c·numSubcells + r`, laid out contiguously right after `c`. So if the fine *and*
every coarse level are distributed with PETSc's **Simple** partitioner (equal
contiguous splits of `[0, Ncells)`), the fine split at `k·Nf/p` lines up with the
coarse split at `k·Nc/p` (since `Nf = numSubcells·Nc`): **each rank's coarse cells
and their fine children land on the same rank.** The multigrid interpolation is
then rank-local — no cross-partition communication, no hang — and the levels are a
genuine per-rank refinement, so the exact **nested** interpolator applies (the fine
levels are flagged via `DMPlexSetRegularRefinement`).

Trade-off: hierarchy meshes reload with a Simple (contiguous) partition rather than
the default graph partition. For refinement meshes the canonical ordering is
reasonably coherent, and field reload is coordinate-matched (partition-agnostic),
so correctness is unaffected; partition-quality tuning can come later. Plain
(non-hierarchy) meshes are untouched — they keep the default partitioner.

## Implementation

All in `src/underworld3/discretisation/discretisation_mesh.py`:

- `_hierarchy_sidecar_name()` — sidecar path convention.
- `Mesh.write()` — writes `metadata/hierarchy_coarse_levels` and one sidecar
  holding the coarsest level (collective).
- `Mesh.__init__` `.h5` branch — loads the coarsest sidecar and rebuilds the
  intermediate coarse levels by `refine()` (serial and parallel), stashing the
  list on `self._sidecar_coarse_levels`.
- `Mesh.__init__` hierarchy section — distributes fine + coarse with the Simple
  partitioner (co-location), splices them under the working dm, flags the fine
  levels as regular refinements, re-establishes the link-free clone.
- `petsc_dm_{set,get}_regular_refinement` in `cython/petsc_discretisation.pyx` —
  wraps `DMPlexSetRegularRefinement` (not exposed by petsc4py) so reloaded levels
  take the exact nested interpolation path.

No new user-facing surface: the same `Mesh(file)` reload transparently restores the
hierarchy when the checkpoint has one, and behaves exactly as before when it does
not.

Tests: `tests/test_0004_checkpoint_fmg_hierarchy.py`.
