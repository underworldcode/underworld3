# Layer 2 — SBR adapt-on-top (mesh-owned custom-P hierarchy)

Status: **Phase 1 implemented (2026-06-29)** — design below; see "Implementation"
at the end for the as-built API and what landed. Builds on Layer 1
(`docs/developer/design/GENERALIZED_FMG_HIERARCHY_AND_ADAPT.md`,
`utilities/custom_mg.py`, PR #290). Branch `feature/adapt-on-top`.

## Context

Layer 1 gave us geometric multigrid on **arbitrary** (nested *or* non-nested,
uniform *or* SBR-refined) hierarchies via custom-built prolongations + Galerkin RAP,
with BC-per-level reduction, for every solver family (scalar / vector / Stokes
velocity block), serial and parallel. Layer 2 is the **first concrete application**:
locally refine the mesh where the solution needs it, **on top of** a static uniform
base, and have **every solver that uses the mesh** drive geometric MG on the result.

The defining property (L.M.): **this is adapt / re-adapt, not node movement.** There
is *no node translation and no cumulative refinement* — fundamentally unlike MMPDE.
Each adapt **discards** the previous refined level(s) and **re-marks from the static
base**. The refined top is therefore a pure function of the current metric, fully
rebuildable and fully described by its marker set.

## Model

```
  base finest  ── SBR(marker_t) ─▶  refined finest at step t      (discard at t+1)
  base finest  ── SBR(marker_{t+1}) ─▶ refined finest at step t+1
```

- **Base** = the existing `Mesh(refinement=N)` uniform hierarchy. **Static for the
  whole run** (built once, never moves, sidecar-reconstructable). It supplies the MG
  coarse levels.
- **Refined top** = up to `max_levels` SBR levels applied to the **base finest**,
  marked from an isotropic metric, capped by a node budget. Transient: discarded and
  rebuilt each adapt. SBR cannot coarsen — re-marking from base IS the coarsening.
- The solver operator lives on the **refined finest**; the MG hierarchy is
  `[base L0 … base finest] + [SBR level(s)]`.

Because nodes do not move, the inter-adapt field transfer is plain **Eulerian REMAP**
(evaluate the old finest field at the new finest DOF coords) — no ALE / CARRY
semantics. (Contrast MMPDE, where nodes move with the material and history needs ALE
carry; not applicable here.)

## API — integrate into `mesh.adapt`

A new **nested, on-rank adapter mode** alongside today's MMG path:

```python
mesh.adapt(metric, adapter="sbr", max_levels=2, node_budget=None)  # new (this work)
mesh.adapt(metric)                                                 # = adapter="mmg" (today)
```

- `metric` — the existing isotropic metric interface (`adaptivity.create_metric`,
  `metric_from_gradient`, `metric_from_field`): a scalar MeshVariable carrying
  `M = 1/h²` (target edge length h). Reused unchanged.
- `adapter="sbr"` — nested skeleton-based refinement on top of the base finest.
  `adapter="mmg"` (default) keeps today's topology-changing/redistributing behaviour.
- `max_levels` — cap on SBR depth (bounds the non-load-balanced imbalance).
- `node_budget` — cap on added DOFs: mark the highest-metric cells first until the
  budget is hit (so refinement concentrates where the metric is largest).

Marking: convert the metric to a per-cell target h, mark cells whose current size
exceeds target (SBR `adaptLabel`/`refine_sbr`, via `custom_mg.sbr_refine_where`).

## Mesh-owned hierarchy — all solvers consume it

Layer 1 registers custom-P **per solver** (`solver._custom_mg` via
`set_custom_fmg`). Layer 2 moves the hierarchy's home to the **mesh** so every solver
on that mesh consumes the same refined hierarchy with no per-solver call — directly
realising "force the adaptivity into the mesh; all solvers consume it".

- The mesh holds the current `[static coarse meshes] + refined finest` and the
  per-level transfer builders.
- A solver's existing custom-P injection (`inject_custom_mg`, already wired into every
  base `solve()`) learns to pick up a **mesh-owned** hierarchy if present (in addition
  to a solver-set one). `field_id` is inferred per solver topology (0 for the Stokes
  velocity block, None for scalar/vector) exactly as today.
- `mesh.adapt(adapter="sbr")` updates the mesh-owned hierarchy and invalidates
  registered solvers (`is_setup=False`, the existing mechanism) so each rebuilds on
  the new finest at next `solve()`.

## Field transfer

Reuse the existing remesh machinery — **no new transfer code**:
- The adapt wraps the SBR move in the existing var-transfer path
  (`remesh_with_field_transfer` / the `mesh.adapt` var-reset+`global_evaluate`
  REMAP). Eulerian REMAP is correct here (no node translation).
- `global_evaluate` is swarm-migration based → **parallel-safe and partition-agnostic**
  for the field values, independent of the mesh partition.
- Operator `on_remesh` hooks fire as usual; SLCN/DuDt history transfers per its
  policy. (Re-adapt ⇒ REMAP fallback is the right semantics, not ALE carry.)

## No redistribution — a CORRECTNESS requirement, not just cost

Adapted layers are **on-rank**; load balancing is not required and the imbalance is
accepted (bounded by `max_levels`). But redistribution must be actively prevented,
because **custom-P's parallel path requires the finest to stay co-partitioned with the
coarse levels** (rank-local point location: every owned fine node lands in a *local*
coarse simplex). Redistributing the refined finest would diverge its partition from the
static coarse tail and **break parallel custom-P** — not merely slow it down.

Guards:
- Wrap the SBR'd DM with **`distribute=False`**; never call `DMPlexDistribute` /
  redistribute on adapted layers (the MMG path's `redistribute` flag must not reach the
  `sbr` path).
- `dm.refine()` (base) and `adaptLabel`/SBR (top) both preserve partition by
  construction; the only redistribution risk is the Mesh-wrap of the new finest →
  assert per-rank ownership of the static coarse tail is unchanged after an adapt.

## Data model — static coarse tail, transient fine head

- **Coarse levels: allocate once, reuse forever.** Uniform + static ⇒ DMs, sections,
  coordinates and their (nested) transfers never change. Small (coarse uniform) ⇒
  negligible standing memory, zero per-adapt cost.
- **Only the finest is transient:** each adapt frees the old finest's field vecs + top
  transfer and allocates new ones (new DOF count). Memory churn is confined to one
  level.
- **Transfer caching (efficiency lever):** coarse→coarse transfers are constant →
  build once, cache. Only the **top** transfer (static coarse finest → new SBR finest)
  is rebuilt per adapt. Base-level transfers, being genuine uniform refinements, can use
  PETSc **nested** interpolation (exact, cheap, cached); the SBR-top uses **barycentric**
  custom-P. (`CustomMGHierarchy.build` currently rebuilds all levels — add a per-level
  cache keyed on level identity.)
- **Galerkin coarse operators still recompute** each solve (PᵀAP from the new fine A —
  the physics/coefficients changed); intrinsic, not avoidable by keeping coarse grids.
  But the transfers feeding RAP are mostly cached.

## Checkpointing (designed here; implemented in a follow-up)

Adapted meshes are restartable by storing **markers, not meshes** — consistent with the
existing FMG sidecar philosophy and exploiting SBR determinism:

- **Base**: existing sidecar (coarsest DM + refine count) → reconstruct the static
  uniform hierarchy bit-identically (canonical `refine()` numbering).
- **Adapted levels**: store **one cell-marker label per SBR level**, in that level's
  (deterministic) cell numbering. SBR (`adaptLabel`/`refine_sbr`) is deterministic given
  the marker ⇒ replaying rebuilds each refined level bit-identically. Markers are tiny
  (a label/IS) ⇒ checkpoint size stays dominated by field data.
- **Fields**: stored on the finest as today.
- **Reload**: base sidecar → re-refine uniform → replay SBR markers per level → load
  fields onto the reconstructed finest.

Robustness note: **custom-P sidesteps the `err77` canonical-nested-numbering fragility
that bit native-FMG checkpoint reconstruction** ([[project_fmg_checkpoint_hierarchy]]).
Our transfers are built from *coordinates*, not PETSc parent-child maps, so reconstruction
only needs the finest's coordinates/topology to match — far more forgiving than native
nested interp. Store the realized marker (not the metric), since the marker is what
produced the current mesh.

## Correctness invariants (Layer 2)

1. Base hierarchy is immutable for the run; only the SBR top changes.
2. Adapted layers are on-rank — **no redistribution** (partition of the static coarse
   tail is invariant across adapts).
3. Re-adapt is non-cumulative: each adapt re-marks from the base finest; previous SBR
   levels are discarded (no node translation, no accumulation).
4. Field transfer is Eulerian REMAP via `global_evaluate` (parallel-safe).
5. All Layer-1 invariants hold per level (BCs at every level, transfers reduced→reduced,
   partition-of-unity, `pc_mg_galerkin=both`, nullspaces re-attached, no silent GAMG
   fallback).

## Phased plan

1. **Live adapt path (this increment)** — `mesh.adapt(metric, adapter="sbr",
   max_levels, node_budget)`: metric→mark→SBR(`distribute=False`)→wrap→mesh-owned
   custom-P hierarchy→REMAP field transfer→solver auto-pickup→invalidate. Validate a
   solve converges via custom-P on the refined mesh and a moving-feature re-adapt loop
   carries fields; serial then np=2 (co-partitioned, no redistribute).
2. **Transfer caching** — cache static coarse-level transfers; rebuild only the top.
3. **Checkpointing** — marker-sidecar store + reconstruct (per the scheme above).
4. **Driver/example + tests + tier classification.**

Non-goals this pass: cumulative refinement / node movement (explicitly out — that's
MMPDE's domain); load-balancing the adapted layers; MMG-path changes.

## Prototype validation (2026-06-29, before touching mesh.adapt)

The full mechanic is de-risked in a study script
(`~/+Simulations/layer2_adapt_on_top_study/proto_sbr_adapt_on_top.py`) using **only
Layer-1 code** (`sbr_refine_where` + `set_custom_fmg`), serial **and** np=2:

- **A. SBR adapt-on-top + custom-P** — static base (refinement=2, finest 512 cells)
  → SBR-refine near a feature (→1023) → wrap → custom-P hierarchy
  `[L0,L1,base-finest,SBR-finest]` → Poisson solve: **pc=mg, 4 levels, 4 iters**,
  matches GAMG to 2.4e-8.
- **B. REMAP field transfer** base-finest → SBR-finest via `global_evaluate`: rel err
  vs analytic 3.9e-4 (P2 interpolation floor).
- **C. Re-adapt (non-cumulative)** — feature moves 0.7→0.5→0.35; each step discards the
  SBR top, re-marks from the **same static base finest**, re-solves (4 iters, pc=mg),
  carries the field (~5e-4). Base finest stays 512 cells (unchanged) throughout.
- **np=2 (no-redistribute correctness)** — each rank SBR-refines its OWN cells
  (rank0 256→502, rank1 256→521; on-rank imbalance accepted), the refined finest stays
  **co-partitioned with the base coarse levels**, so parallel custom-P works (pc=mg,
  4 iters, matches GAMG). REMAP + re-adapt loop parallel-safe.

Conclusion: the design holds end-to-end. Remaining work is wiring it into
`mesh.adapt(adapter="sbr")` with the mesh-owned hierarchy + solver auto-pickup
(no per-solver `set_custom_fmg`), then transfer caching, then checkpointing.

## Implementation (Phase 1, 2026-06-29)

The live-adapt path is wired into the mesh API. The naming settled differently
from the early `adapter=` sketch above: `adapt` now **returns a child**, and the
old in-place MMG `adapt` was **renamed `remesh`**.

**`mesh.adapt(metric, max_levels=2, node_budget=None, builder="barycentric",
adapter="sbr", verbose=False) -> child`** (`discretisation_mesh.py`,
`_adapt_sbr`):
- Marks base-finest cells whose characteristic size `h ≈ (dim!·vol)^(1/dim)`
  exceeds the metric target `1/√M` at the centroid; up to `max_levels` SBR passes;
  `node_budget` keeps the highest-metric cells first (approximate DOF cap).
- Metric sampled by `uw.function.evaluate` (serial) / `global_evaluate` (np>1,
  partition-agnostic). SBR via `custom_mg.sbr_refine` (`distribute=False`, on-rank).
- Wraps the refined finest as a child: `child.parent = self`,
  `child._relationship_kind = "refinement"`, registered in
  `parent._registered_children`. The base mesh is untouched (re-adapt is
  non-cumulative).
- The child owns the static coarse tail `child._custom_mg_coarse_meshes`
  (`= parent._coarse_level_meshes()`, built once and cached) + `_custom_mg_builder`.

**Solver auto-pickup** (`custom_mg.maybe_inject_custom_mg`, called from the four
solve hooks in `petsc_generic_snes_solvers.pyx`): when a solver's `_custom_mg` is
unset but its mesh carries `_custom_mg_coarse_meshes`, it lazily builds
`CustomMGHierarchy([*coarse, solver.mesh], field_id=…)` (0 for the Stokes
velocity block, None for scalar/vector) — so *every* solver on an adapted mesh
drives geometric MG with no per-solver `set_custom_fmg`. A solver-set hierarchy
still wins.

**`copy_into` / `add_into`** (`enhanced_variables.py`) dispatch on the child kind
via `mesh._refine_prolongate` / `_refine_restrict`: parent→child is FE-exact
barycentric custom-P (serial) or `global_evaluate` REMAP (np>1); child→parent is
nearest-node injection (serial) / `global_evaluate` (np>1). Submesh children keep
the existing `subpoint_is` restrict/prolongate.

**`mesh.remesh(metric)`** is the renamed in-place MMG path (byte-identical body).
`mesh.adapt(adapter="mmg")` is a deprecation shim → `DeprecationWarning` + forwards
to `remesh`. The MMG regression tests (`test_0810/0830`, `ptest_0763`) were
migrated to `remesh`.

**Validation**: `tests/test_0835_sbr_adapt_on_top.py` (7 tests, tier_b) + the study
script `~/+Simulations/layer2_adapt_on_top_study/validate_mesh_adapt_integration.py`
(serial + np=2, ALL OK): child link + finer; solver auto-pickup pc=mg 4 iters ==
GAMG; copy_into both directions; non-cumulative re-adapt; node_budget localises;
shim warns. SBR needs no MMG/pragmatic PETSc build (unlike `remesh`).

**Not yet** (follow-ups, unchanged from the phased plan): transfer caching
(reuse cached coarse-level transfers; rebuild only the SBR top); marker-sidecar
checkpointing (`child._adapt_markers` already stashes the per-level markers);
chaining `adapt` on an already-adapted child; load-balancing the adapted layers.
