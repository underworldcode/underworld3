# Lagged reference-mesh SL history (and mixed-mesh evaluation)

**Status:** design / plan (2026-06). Supersedes the ALE `v_mesh`-pulse + CARRY
approximation for the semi-Lagrangian history on a moving mesh. Two-phase plan:
**Phase 1 — mixed-mesh expression evaluation** (the enabling primitive,
implement and prove first); **Phase 2 — lagged rolling-clone SL history**
(the architecture built on Phase 1).

Related: [REMESH_FIELD_TRANSFER_DESIGN](REMESH_FIELD_TRANSFER_DESIGN.md),
[in_memory_checkpoint_design](in_memory_checkpoint_design.md),
[submesh-solver-architecture](submesh-solver-architecture.md),
[STRESS_EQUILIBRIUM_FREESURFACE](STRESS_EQUILIBRIUM_FREESURFACE.md).

## Problem

On a moving mesh (free surface, or adaptation) the semi-Lagrangian update needs
the *old* field at the departure point. At a **receding** free surface
(downwellings) that point lies in the layer the surface just vacated — which the
*new* mesh does not cover. Sampling the new-mesh FE field there extrapolates the
P3 polynomial → ±300 node-level speckles → blow-up (Ra=1e5 annulus, ~step 20).

The current code approximates "old field on old geometry" *without* keeping a
copy: history vars are `CARRY`'d onto the new mesh and the trace-back subtracts
`v_mesh = Δx/dt` (`ddt.py` `SemiLagrangian.update_pre_solve` /
`_activate_ale_for_traceback`). This is the approximation that fails:

- The order-1 advected field has its `v_mesh` correction **gated off** at `i=0`
  (the band-aid re-record path, `ddt.py:2447`), so its foot lands at the *old*
  surface — in the vacated layer.
- The annulus `return_coords_to_bounds` (`meshing/annulus.py:507`) clamps to the
  **construction-time circular radius**, not the deformed surface — so receded
  feet aren't recognised as outside and aren't clamped.

Confirmed empirically (`~/+Simulations/fs_convection_goal4/smoke_geomclamp.py`):
a geometry-aware clamp **delays** the blow-up ~2 steps and halves the early
overshoot but does **not** cure it — because the departure *value* is wrong, not
just the geometry. Clamp/monotone band-aids are dead ends; `monotone` does run on
the FE path (`functions_unit_system.py:854`) but fails via neighbour-corruption
feedback once a speckle seeds.

## Target architecture: lagged rolling-clone

Keep a **frozen clone of the previous-step mesh** (geometry + the primitive
fields needed for history). The SL trace-back samples *that* clone:
`global_evaluate(history_expr_on_clone, departure_coords)`. The departure point
sits in the receded layer, which the **old** mesh covers — so the sample is a
real interpolated value: no clamp, no extrapolation, no speckle. This is the
*exact* computation the `v_mesh`/CARRY apparatus only approximates.

### Invariants

1. **One clone, any scheme order.** Every trace-back is a single-Δt reach from
   the current mesh to the previous-step mesh. BDF2 reaches 2Δt as two composed
   single-Δt traces across successive steps (`psi_star[1]` is last step's
   already-traced `psi_star[0]`). History *depth* lives in the **stack of
   fields**, not in a stack of meshes. The clone is a **rolling** one-step-behind
   copy, refreshed at the start of each step.
2. **Shared partition.** The clone must share topology *and* the exact parallel
   partition / local numbering (PETSc `DMClone`-style: shared DMPlex + `PetscSF`,
   **independent** coordinate vector). Then refreshing the clone's geometry each
   step is a **local memcpy of coordinates, no global scatter**. (Sampling at
   launch points is `global_evaluate` — parallel-routed and partition-agnostic;
   shared partition lets it reuse routing.)
3. **The clone stores primitives, not derived tensors.** Store `T` (and `V`, and
   stress for VE) at the lagged time; **recompute** gradients / fluxes / stress
   at full FE accuracy on the clone. This retires the lossy projected histories
   (`DFDt` flux history, the `psi_star_flat` tensor view, the VE stress
   projection) and their projection diffusion.
4. **Mesh-homogeneous history; the mixing guard defines the clone set.** Each
   history term is an expression evaluated **entirely on the clone**; the only
   thing crossing the mesh boundary is a value **array at coordinates**, never a
   symbolic expression spanning two DMs. The evaluator's existing
   refuse-mixed-mesh guard is the *enforcement*: if a history build trips it,
   that names a current-mesh field that must be cloned to the lagged mesh. Time-
   splitting (BDF/AM) already separates terms by time level, so "no mixed mesh"
   and "no mixed time level in one expression" become the same discipline.
5. **Split the implicit `self.mesh`.** Today `update_pre_solve` assumes one mesh
   everywhere (`get_closest_cells`, `_centroids`, `return_coords_to_bounds`, the
   `global_evaluate` calls). The refactor makes two roles explicit:
   *foot computation* (where did the material come from) on the **current** mesh;
   *history sampling* (value/flux there) on the **lagged** clone. Every
   evaluation takes its mesh **from the variable/expression**, not a default.
6. **Unifies free-surface and adaptation.** The lagged clone is the SL reference
   regardless of *how* nodes moved (free surface, mmpde, OT). Same code path —
   the original goal of this work.

## Phase 1 — mixed-mesh expression evaluation (implement first)

The enabling primitive, and the efficiency-critical piece. Two implementation
strategies; Phase 1 likely needs both, chosen per term:

- **(a) Clone-the-primitives** → expression is homogeneous on the clone. Simple
  per-eval; costs memory + per-step field copy. Good default for history terms.
- **(b) True mixed-mesh evaluate** → evaluate each meshVariable on *its own* DM
  at the shared physical coordinates, then combine the per-mesh arrays pointwise.
  No field cloning; the harder/efficiency-critical path.

### Semantics of (b)

For an expression `E` whose meshVariable atoms belong to meshes `{M_a, M_b, …}`,
at physical points `X`: value at `x_k` = `E(var_a(x_k|M_a), var_b(x_k|M_b), …)`.
Pointwise; derivatives are local to each variable's own mesh, so they're fine.

### Mechanism

1. Partition the (unwrapped) expression's atoms by owning mesh.
2. For each mesh `M_i`, point-locate `X` and interpolate the needed
   variable/derivative components → array `A_i`.
3. Substitute the `A_i` into the symbolic combination and evaluate pointwise
   (existing numpy/lambdify path).

### Efficiency (the crux)

- **Per-mesh location cache.** Point location dominates. Cache per mesh, keyed on
  coord-hash (cf. `_dminterpolation_cache`, see
  [data-access](../subsystems/data-access.md)); reuse across all history levels
  evaluated at the same `X` within one step.
- **Shared-partition fast path.** When `M_b` is a partition-sharing clone of
  `M_a`, a point owned by rank *r* on `M_a` is owned by rank *r* on `M_b` — one
  routing, two local interpolations; no second scatter. This is *why* the clone
  shares the partition.
- **Reuse the locator across the BDF stack** (all `psi_star[i]` share `X`).

### API

- Keep refuse-mixed-mesh as the **default** (accidental mixing still caught).
- Add an **explicit opt-in** (e.g. `evaluate(..., allow_mixed_mesh=True)` or a
  dedicated `multi_mesh_evaluate`) for intentional history evaluation.
- Caveat to verify up front: `global_evaluate` of **derivative / vector**
  expressions on a *non-current* mesh has a known wrinkle on some branches
  (`scalar_component` mismatch). The gradient-recompute path depends entirely on
  this — confirm/fix it as the first Phase-1 task.

### Tests

Correctness vs per-mesh manual eval; the shared-partition fast path
(bit-identical, reduced routing); parallel (np>1); derivative/vector exprs across
meshes.

## Phase 2 — lagged rolling-clone SL history

1. **Clone primitive:** `DMClone`-with-shared-partition + independent coords;
   rolling refresh (local coord + field copy at step start). Reuse the in-memory
   snapshot toolkit if it already provides same-partition state hold.
2. **DDt refactor:** split foot-mesh (current) vs history-mesh (clone); sample
   history via Phase-1 evaluation; recompute fluxes/gradients homogeneously on
   the clone.
3. **Retire** the `CARRY` + `v_mesh`-pulse + `i=0`-gate machinery for the trace-
   back (scope what else still depends on `CARRY` before deleting).
4. **Validate:** the free-surface smoke test reaches approximate equilibrium
   (T∈[0,1], no speckle); adaptation regression unchanged; bit-identical on a
   non-moving mesh.

## Open questions / risks

- Derivative/vector `global_evaluate` on a non-current mesh (Phase-1 task 1).
- What else depends on `CARRY` / the ALE pulse (swarm advection? VE?).
- Memory/perf of the rolling clone at scale; locator rebuild vs reuse.
- VE: the stress history needs lagged `V` (and `∇V`) on the clone — confirm the
  clone primitive set covers the constitutive expression's full free-variable
  set (the mixing guard enumerates it).

## Relationship to landed work

- **PR #246** (capability gate + `Mesh.deform` + `ephemeral_coords` + SL-field
  `CARRY` auto-stamp) made coordinate mutation foolproof and proved the failure
  is *not* transfer-incoherence — it set up this design. The `CARRY` auto-stamp
  is part of the apparatus this design will eventually retire.
- The **big question** this unblocks: SL height field vs full node movement +
  mmpde. The lagged-clone approach makes SL correct at a moving boundary; the
  comparison is then about cost/robustness, not correctness.
