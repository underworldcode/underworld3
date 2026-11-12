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

## Parallel: distribute the coarsest, then refine

The reload rebuilds the hierarchy the way a fresh `refinement` mesh is built:
the coarsest level from the sidecar is distributed, and every finer level is
made from it by a local `refine()`. `refine()` never moves points across ranks,
so each coarse cell and all of its children are on one rank, and the nested
interpolation is rank-local at any np. (Distributing independently loaded levels
misaligns them, and the interpolator spins in cross-rank point location.)

## Matching the rebuilt fine mesh to the file (#690)

The rebuilt fine level is the saved fine mesh, numbered and distributed
differently, and at its *reference* geometry: refinement of the coarsest level,
with no refinement callback and no mesh motion. Two things must be carried from
the file onto it: the saved coordinates, and the SF from the file's points to
the mesh's points, through which `read_checkpoint` reloads fields by section.

Until #690 was fixed the saved coordinates were paired with a serial refinement
of the sidecar **by array index**. That holds only when the file was written in
serial: a mesh written on two or more ranks reloaded, on every rank count, with
cells joined to the wrong vertices (the integral of 1 came back negative). The
mesh SF was also left as the file-to-loaded SF composed with the SF of the
*coarse* distribute, so section field reloads failed on more ranks than wrote.

The reload now matches points (`_match_file_points`):

- **Vertices by reference position.** `Mesh.write` stores each fine vertex's
  reference coordinates in the mesh file, as a vertex vector
  `uw_reference_coordinates` (`Mesh._view_reference_coordinates`). They are
  computed by refining the coarsest level exactly as the reload does, so the
  match is exact however far the saved mesh has moved. A file written before
  this has no such vector; its saved coordinates stand in, which matches any
  mesh that has not moved by a large fraction of a cell.
- **Every other point by its closure vertices**, keyed by their file ids. This
  is also the check: a wrong vertex match leaves cells with no partner.
- The matched file ids give the new mesh SF directly, and the saved coordinates
  are stamped on at the end of `__init__`.

If any point is unmatched (an old file of a moved mesh), the reload warns and
falls back to reloading the fine mesh **without** its hierarchy, the same path
as an unrefined mesh, which is always correct. Solvers then use their default
preconditioner instead of FMG. A corrupt mesh is never returned.

The vertex and closure tables are gathered on every rank, as the earlier
coordinate broadcast was; a distributed match would be needed for very large
meshes.

## Implementation

All in `src/underworld3/discretisation/discretisation_mesh.py`:

- `_hierarchy_sidecar_name()`: the sidecar path convention.
- `Mesh.write()`: writes `metadata/hierarchy_coarse_levels`, the reference
  coordinates (`_view_reference_coordinates`) and one sidecar holding the
  coarsest level (collective).
- `Mesh._load_dm_from_file()` (`.h5` branch): loads the fine mesh with its SF
  and the coarsest sidecar.
- `Mesh._splice_hierarchy_from_sidecar()`: distributes the coarsest, refines it,
  matches the rebuilt fine to the file (`_load_reference_coordinates`,
  `_match_file_points`), and sets the link-free working `dm` and the mesh SF; or
  returns False for the flat fallback.

No new user-facing surface: the same `Mesh(file)` reload transparently restores the
hierarchy when the checkpoint has one, and behaves exactly as before when it does
not.

Tests: `tests/test_0004_checkpoint_fmg_hierarchy.py`,
`tests/test_0690_refined_checkpoint_cross_np.py` (written on one rank count,
read on another) and `tests/parallel/test_0690_refined_checkpoint_reload_parallel.py`.
