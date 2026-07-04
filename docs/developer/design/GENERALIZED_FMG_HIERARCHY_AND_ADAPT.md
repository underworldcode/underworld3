# Generalized FMG hierarchy + adapt-on-top

Status: design (hardening the custom-prolongation prototype). 2026-06-28.

## Context

UW3's geometric FMG (`Mesh(refinement=N)` → `dm_hierarchy`, `preconditioner="fmg"`)
requires a **uniform nested** hierarchy with PETSc's canonical nested interpolation.
That couples multigrid to uniform refinement and rules out local adaptation.

A prototype established that we can drive geometric multigrid with a prolongation we
**build ourselves** (barycentric or local-RBF) and install via `PC.setMGInterpolation`,
with Galerkin RAP forming the coarse operators. Validated:
- FMG-equivalent convergence on non-nested *and* nested hierarchies (SolCx velocity
  block, η-jump 1e6: 8 iters, vs GAMG 157 — geometric MG where FMG cannot otherwise exist);
- a 5-level hierarchy (3 uniform + 2 SBR-targeted at the jump) solves at 8 iters;
- local SBR refinement is a ~4-line stock-petsc4py call (no MMG): mark cells in a label,
  `dm_plex_transform_type=refine_sbr`, `dm.adaptLabel("name")` — conforming, on-rank.

**Load-bearing correctness lesson:** essential BCs must be applied at **every** level,
transfers mapping reduced→reduced. PETSc native FMG does this automatically (each level's
`PetscSection` carries the Dirichlet constraints); custom-P from raw coordinates must
replicate it. Exact nesting makes the omission fatal (coarse boundary-normal DOFs coincide
with BC-removed fine DOFs → zero columns → singular coarse operator).

## Two-layer architecture

### Layer 1 — generalized FMG hierarchy (adapter-agnostic)
**The existing `refinement=N` uniform hierarchy STANDS** and is the FMG base. Layer 1
generalizes the *transfer machinery* so the hierarchy may contain levels that are **not**
uniform refinements, by carrying **supplied prolongations** instead of relying on canonical
nesting.

Layer 1 knows nothing about *how* a level was produced. Its contract is a sequence of
levels, each `(dm, prolongation_to_next)`, plus:
- **BC-per-level reduction** — transfers map reduced→reduced (the invariant above), derived
  from the owning solver's essential BCs via each level DM's global section.
- **Transfer builders** — pluggable: `nested` (PETSc canonical, for uniform levels),
  `barycentric` (FE-exact, needs a triangulation), `rbf` (local kNN, scattered-point
  robust, reuses the swarm-proxy RBF kernel). Default per level: `nested` where the level
  is a genuine uniform refinement, else `barycentric`.
- **Installation** — set the transfers on the velocity-block (Stokes) or top (scalar/vector)
  PCMG via `setMGInterpolation`, before first `PCSetUp`, with `KSPSetDMActive(OPERATOR,False)`
  and `pc_mg_galerkin=both`. Re-attach nullspaces (pressure / rigid-body) after rebuild.

Home: `src/underworld3/utilities/custom_mg.py` (grow the committed module).

Generality requirement: Layer 1's interface must admit **any** future level source
(MMG/parmmg, edge/face-point injection, knockout-as-coarsening, independent meshes) and
**any** transfer builder. No SBR-specific assumptions in Layer 1.

### Layer 2 — adapt-on-top (first concrete application)
On top of the uniform FMG base, add refined levels via the **existing adaptive-meshing
pattern**: choose an **adapter** + a **metric**, call **periodically**; field
transfer/redistribution handled by the existing remesh machinery
(`remesh_with_field_transfer`, `on_remesh`, `mesh.adapt`).

This pass's adapter is **SBR refine-on-top**:
- metric → mark cells (e.g. |∇field| threshold, or distance-to-feature) → `adaptLabel`
  with `refine_sbr` → a new finest level on top of the current finest;
- **non-load-balancing** (on-rank, no redistribution) — which **bounds the number of
  extra levels** (imbalance grows with depth); the adapter exposes/limits this;
- registers `(new level DM, custom-P transfer)` into the Layer-1 hierarchy.

Home: an adapter consistent with `adaptivity.py` / `mesh.adapt(metric, adapter=...)`,
calling into Layer 1. Future adapters (load-balancing MMG, etc.) plug in the same way.

## Parallel strategy (parallel from the start)

The supported parallel path is the **nested, co-partitioned** hierarchy:
- The uniform base is co-partitioned by construction (`dm.refine()` preserves partition);
  SBR is on-rank → levels stay co-located (a fine cell's parent coarse cell is on the
  same rank). So each rank builds its block of `P` from its **local** coarse cells —
  point-location is rank-local; only a thin ghost layer at partition boundaries.
- **BC-per-level reduction is parallel-correct for free**: it rides the DM global section
  (`localToGlobal` gives the reduced global ordering across ranks).
- `P` is assembled as an MPIAIJ matrix (fine local rows; coarse local + ghost columns).

**Non-nested / independent-mesh custom-P** in parallel (cross-rank point location) is now
supported via a **cross-partition transfer** (`_build_crosspart_transfer`,
`_gather_coarse_cloud`). When the coarse and fine meshes are partitioned independently, a
fine leaf on rank *r* can sit in a coarse cell owned by rank *s*, so the rank-local builder
either misses it (nearest-DOF fallback — wrong) or leaves a coarse DOF with no fine image
(zero column). The fix exploits the fact that a coarse MG level is, by definition, **small**:

- **All-gather the coarse node cloud** (coords + each node/component's GLOBAL reduced column
  index, `-1` for a BC-constrained DOF), deduplicated by rounded coordinate (ghost copies are
  bit-identical). Every rank then holds the *full* coarse mesh.
- Each rank locates its **owned** fine nodes against that full cloud → point location spans
  partitions. Columns are the coarse global reduced indices (off-rank columns are fine for
  MPIAIJ); constrained coarse DOFs stay as barycentric vertices but drop from the columns
  (reduced→reduced). Fine rows stay rank-local.

`CustomMGHierarchy(..., cross_partition=...)` / `set_custom_fmg(..., cross_partition=...)`:
`"auto"` (default) builds the rank-local co-partitioned transfer first and rebuilds a level
cross-partition **only if it has zero columns** (the signature of a cross-partition miss) —
so the validated nested/adapt path stays on the fast rank-local builder bit-for-bit, while
non-nested tails are fixed automatically. `True` forces cross-partition (use when a coarse
level is known non-nested and might mis-locate without producing a zero column); `False`
forces the rank-local path. Validated: an independent (non-co-partitioned) coarse box tail
converges in the same iteration count as serial and matches a GAMG reference to ~1e-8 at
np2 and np4.

Non-load-balancing SBR → bound the added refinement depth (configurable cap); document the
imbalance/level trade-off.

### Operator-faithful finest reduced map
The finest transfer's row space **must** equal the assembled operator's space (PCMG Galerkins
`PᵀAP` against the real operator). The finest reduced map is read from the DM global section,
which is that space only **after** `snes.setUp()` finalizes it — critical on an `adapt()`
child, whose section can otherwise be read before finalization and disagree with the operator
(a rectangular finest transfer → bare PETSc error 60 in the PtAP). `CustomMGHierarchy.build`
calls `snes.setUp()` before reading the finest map and asserts its size against the assembled
operator (`_assert_finest_matches_operator`), failing with an actionable message instead. This
removed the earlier defensive skip of semi-Lagrangian advection-diffusion on adapt children —
such solves now install custom-P geometric MG and match a default-preconditioner solve.

## Correctness invariants (must hold, all levels)
1. BCs applied at every level; transfers reduced→reduced (no zero columns).
2. Each prolongation reproduces constants (row-sums = 1 — partition of unity).
3. `pc_mg_galerkin=both` (coarse operators = PᵀAP; UW installs no coarse residual/Jacobian).
4. Nullspaces (constant-pressure / rigid-body) re-attached after any rebuild.
5. No silent fallback: if a level lacks a valid transfer, error (don't degrade to GAMG silently).

## Test matrix
- Hierarchy build: uniform-only; uniform + N SBR levels; nesting + label survival.
- Per-level reduction: zero-column count == 0 on all transfers (scalar + free-slip vector).
- Transfer: partition-of-unity; barycentric vs rbf; nested-exact vs general.
- Solve: FMG + V-cycle converge on nested and non-nested (serial and np2/4); match
  uniform-FMG iters; beat GAMG; regression — existing FMG/GAMG paths unchanged.
- Parallel (np>1): co-located transfers, reduced sections, FMG convergence; bounded levels.

## Phased roadmap
1. **Layer 1 core** — generalized hierarchy + auto BC-per-level reduction + transfer builders
   (nested/barycentric/rbf) + install into PCMG; scalar + free-slip vector; serial-validated,
   designed parallel-correct (rank-local construction).
2. **Stokes integration** — `set_custom_mg`/hierarchy on the velocity block; nullspace
   re-attach; guards; SolCx end-to-end.
3. **Parallel validation** — np>1 tests; co-location/ghost handling; reduced-section
   correctness; fix the known rank-local-accumulator pitfalls.
4. **Layer 2 adapter** — SBR refine-on-top following `mesh.adapt` (metric→mark→refine→register),
   non-load-balancing, bounded depth, field transfer via existing remesh.
5. **Tests + design doc finalize + tier classification.**

Later (not this pass; do not block): dynamic per-step reallocation loop; efficiency
(transfer caching, exact-nested combinatorial P, operator reuse); load-balancing adapters;
knockout-as-coarsening as an alternative Layer-2 strategy.

## Status (2026-06-29)

**Layer 1 is an independent, working capability** — arbitrary coarse grids
(nested *or* non-nested, uniform *or* SBR-refined) → geometric FMG via custom-P
with BC-per-level reduction. It does not depend on Layer 2.

Landed (committed, tested — `test_1014/1015/1016/1017` serial 20 +
`tests/parallel/test_1017_custom_mg_parallel_mpi.py` np2):
- `CustomMGHierarchy`, `set_custom_fmg`, `sbr_refine`/`sbr_refine_where`;
- automatic BC-per-level reduction + zero-column guard;
- Stokes velocity-block injection; leak-free per-level reduction (copyDS, no
  factory); parallel (np>1) nested co-partitioned transfers;
- validated: scalar jump-coeff Poisson 5-level (3 uniform + 2 SBR) 3 FMG iters
  vs GAMG 46; SolCx velocity block 6 iters vs GAMG ~198, np=1/2/4.

**Current scope:** scalar / single-field-vector **and** Stokes velocity-block;
**serial and parallel** — nested co-partitioned (fast rank-local build) **and
non-nested** (cross-partition transfer, `cross_partition="auto"`, np2/4 matches
serial). Hardening steps 1–4 are complete; what remains is test
tier-classification + undrafting PR #290.

**Solver-family coverage** — the install keys off the PC topology, so one of two
branches covers every solver that consumes the mesh (all `solve()` overrides
delegate to a hooked base `solve()`; the `inject_custom_mg` hook lives in
`SNES_Scalar.solve`, `SNES_Vector.solve`, `SNES_Stokes_SaddlePt.solve`):

| Solver family | Branch (`field_id`) | Serial | Parallel |
|---|---|---|---|
| `SNES_Scalar` (Poisson, Darcy, Projection, AdvDiff, Diffusion) | top-level (`None`) | ✓ | ✓ |
| `SNES_Vector` (Vector_Projection, displacement) | top-level (`None`) | ✓ | ✓ |
| `SNES_Stokes` / VE / NavierStokes | velocity block (`0`) | ✓ | ✓ |
| `SNES_Stokes_Constrained` (in-saddle multipliers) | velocity block (`0`) | ✓ | **skip** ¹ |

¹ The constrained solver is **not parallel-safe today** — it segfaults at np>1
*independently of custom-P*, in the interior-multiplier section reduction
(`_constrain_interior_multipliers_in_section`; **issue #291**; the canonical
`test_1062_constrained_solcx` also segfaults at np=2 under plain GAMG; workaround
`_reduce_interior_multiplier = False`). custom-P on the constrained velocity block
works in serial; the parallel test auto-enables once #291 is fixed.

Hardening steps (all complete as of 2026-06-29):
1. ~~**Stokes / saddle-point** (velocity-block injection).~~ **DONE** (2026-06-29).
   `set_custom_fmg(..., field_id=0)` drives custom-P geometric MG on the velocity
   sub-block. The sub-PC is unreachable until the monolithic Jacobian is assembled,
   so `_install_velocity_block_transfers` forces a Jacobian assembly
   (`computeFunction`+`computeJacobian` at the zero guess; `max_it=0` fallback),
   reaches the velocity sub-PC, `reset`s it and rebuilds a **fresh PCMG** from our P
   (mechanism A — mirrors the proven standalone recipe and sidesteps the
   `MatProductReplaceMats` live-swap bug; the Galerkin-off + `MatPtAP` path remains
   the documented fallback), then re-attaches the coupled Stokes nullspace. Wired
   into `SNES_Stokes_SaddlePt.solve` as a guarded no-op. Validated on SolCx
   (η-jump 1e6, 3-level nested, in-solver): velocity block 6 MG iters vs GAMG 198,
   solution matches GAMG to 1.5e-9 (`test_1017`). NOTE: free-slip velocity
   rigid-body modes on `A_vv` + coarse ops are **not yet** handled (reusing
   `_attach_stokes_nullspace` covers the SolCx pressure-nullspace case only) — a
   Phase-2.5 follow-up.
2. ~~**Drop the factory**.~~ **DONE** (2026-06-29). Each coarse level's
   BC-constrained reduced map is derived directly from its DM via
   `_coarse_reduced_map`: clone the coarse DM, `copyFields` + `copyDS` from the
   finest solver, `createDS`. The DS carries UW's exact essential-BC definitions
   and is topology-independent, so it constrains any coarse mesh sharing the
   solver's boundary labels — validated byte-identical to the old factory path,
   leak-free (no SNES / JIT). `set_custom_fmg` / `build` no longer take a
   `level_solver_factory`.
3. ~~**Parallel (np>1)**.~~ **DONE** (2026-06-29). The hierarchy path builds
   parallel-correct transfers on the nested co-partitioned hierarchy: each rank
   builds its block of `P` rank-locally (ghost-inclusive coarse coords → every
   owned fine node lands in a local coarse simplex, verified 0 misses np=2/4),
   the reduced global numbering rides the DM global section
   (`_level_dof_layout` scatters owned global indices out via `globalToLocal` —
   constrained DOFs `-1`, ghosts resolve to the owner's global), and transfers
   assemble as MPIAIJ (owned fine rows, global coarse cols incl. off-rank;
   constrained coarse DOFs drop → reduced→reduced). Parallel zero-column guard
   via `Pᵀ·1` + allreduce. Validated np=1/2/4: scalar Poisson 4 iters + Stokes
   SolCx velocity block 6 iters, matching the GAMG reference and each other
   across rank counts (`tests/parallel/test_1017_custom_mg_parallel_mpi.py`). The
   **legacy** finest-only path (`set_custom_mg` / `_reduce_to_global`) stays
   serial-only and raises loudly at np>1.
4. ~~**Non-nested (np>1)**.~~ **DONE** (2026-07-03). A non-nested coarse tail
   (`set_custom_fmg([independent coarse mesh])`) is partitioned independently of
   the fine mesh, so a fine leaf on rank *r* may sit in a coarse cell owned by
   rank *s* — the rank-local build (step 3) then produces zero columns (23 at
   np2, 48 at np4, caught loudly by the zero-column guard). Fix = a
   **cross-partition transfer** (`_gather_coarse_cloud` + `_build_crosspart_transfer`):
   all-gather the (small) coarse node cloud — coords + each node/component's
   global reduced column index (`-1` = constrained) — dedup by coordinate, then
   every rank locates its **owned** fine nodes against the **full** coarse mesh.
   Columns are coarse global reduced indices (MPIAIJ handles off-rank); fine rows
   stay rank-local. Routed via `cross_partition` on `CustomMGHierarchy` /
   `set_custom_fmg`: `"auto"` (default) uses the fast rank-local build and only
   rebuilds a level cross-partition when it has zero columns — so the validated
   nested/adapt path stays **bit-identical** on the fast builder, non-nested tails
   are fixed automatically; `True`/`False` force either path. Validated np=1/2/4:
   a non-nested coarse-tail custom-P FMG converges in the same 6 iters as serial
   and matches a GAMG reference to ~1e-8
   (`test_1017_custom_mg_parallel_mpi.py::test_parallel_custom_fmg_nonnested`).

## Explicit non-goals for this pass
- Knockout (shown to pay full-fine assembly; structural value only) — not pursued now.
- Dynamic time-stepping convection integration — later phase.
