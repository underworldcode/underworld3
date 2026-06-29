# Layer 2 — SBR adapt-on-top (mesh-owned custom-P hierarchy)

Status: design (2026-06-29). Builds on Layer 1
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
