# Lagged reference-mesh SL history (and mixed-mesh evaluation)

**Status:** design / plan (2026-06). **Core mechanism PROVEN by serial prototype
(2026-06-18)** — the old-frame reach-back (below) holds T∈[0,1] for 40 steps on
the Ra=1e5 free-surface annulus (surface to h_max≈0.1, ~3.4 cells) where the
stock ALE blows up by s28. Prototype: `~/+Simulations/fs_convection_goal4/
oldframe_prototype.py` (serial; a 2nd identical Annulus stands in for the clone,
moved to the previous step's coords each step). What remains is the *robust*
implementation. Supersedes the ALE `v_mesh`-pulse + CARRY approximation for the
semi-Lagrangian history on a moving mesh. Two-phase plan: **Phase 1 — mixed-mesh
expression evaluation** (the enabling primitive, implement and prove first);
**Phase 2 — lagged rolling-clone SL history** (the architecture built on Phase 1).

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

### Old-frame reach-back (the key reformulation — 2026-06-18)

The current ALE folds mesh motion into an apparent velocity `v_mesh = Δx/dt` and
reaches back on the *new* mesh: `x_dep = x_new − (V_new − v_mesh)·dt`. This is
lossy in two independent ways, both confirmed experimentally (the buoyancy-
*decoupled* `kinematic_fs` run blows up under free-slip — see "(B)" in
[[project_deform_keystone_carry_field]]):

* **Mesh-motion is re-derived by approximation, then interpolated.** We *know*
  node *i* was at `x_old[i]` exactly. Folding that into a linear-in-dt velocity
  and interpolating the *new* mesh to recover it injects error every deform —
  in the pure-mesh-motion limit (`V=0`) the exact answer is just the carried
  nodal value `psi_star[i]`, but the code interpolates `x_new ± Δx` instead.
* **`V_new` is the wrong velocity, temporally and spatially.** It is the
  *end-of-step* velocity on the *new* mesh; it neither transported the material
  over `[t,t+dt]` nor lives on the mesh that brackets the departure point. At a
  disequilibrium free surface `Δx_n ≠ V_n·dt`, so `(V_new − v_mesh)` keeps a
  spurious normal component that pushes the foot off the surface.

**Reformulation:** carry the *old state* on the clone — old coords (the clone
geometry), old `T` (`psi_star`), **and old `V`** — and do the whole characteristic
in the old frame on the old mesh:

```
x_mid = x_old[i] − 0.5·dt·V_old(x_old[i])
x_dep = x_old[i] −     dt·V_old(x_mid)        # RK2, all evals on the OLD mesh
T*    = T_old(x_dep)                          # representable: old mesh covers old domain
```

No `v_mesh`, no fold. Mesh motion is **exact** (V=0 → returns `psi_star[i]` with
no interpolation); only physical advection is approximate, evaluated where the
point is always representable. Carrying `V_old` is one extra field copy (as cheap
as `T`) and removes the temporal inconsistency of `V_new`. Strictly-2nd-order
(space-time-midpoint) sampling would mix `V_old` (old mesh) and `V_new` (new
mesh) — genuinely mixed-mesh, i.e. Phase 1 — but **old-frame RK2 with `V_old`
alone** is already consistent, needs no mixed-mesh machinery, and is the first
target. NB the surface velocity used to build the clone/step must be the
re-solved post-deform V (the driver already re-solves Stokes after `deform()`
and before advection — verified), so `V_old` for step *n+1* is step *n*'s
converged field.

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

## Implementation plan (decided 2026-06-18)

Decisions after the prototype proved the mechanism. Two stages: **Stage 0 lands a
correct (if inefficient) implementation; Stage 1 makes it efficient.** Crucially,
Stage 0 needs **no new `global_evaluate`/mixed-mesh machinery** — that rewrite is
large and must NOT ride on this work.

### Stage 0 — correct & landable: old-frame reach-back via ephemeral restore
- **Old-mesh realisation = (a) ephemeral restore on the working mesh.** Each step
  stash `coords_old` + the history already lives in `psi_star` (CARRY). For the
  reach-back, `with mesh.ephemeral_coords(): mesh._deform_mesh(coords_old)`,
  `global_evaluate(psi_star[i].sym, x_dep)` on that restored geometry, restore to
  new coords. Clean (uses the real mesh + existing evaluate), inefficient (two
  `nuke_coords_and_rebuild` per reach-back). This is the validation+landing path.
- **No old V, no second mesh, no clone, no mixed-mesh eval.** The prototype proved
  the foot computed with `V_new` (current mesh, RK2) + sampling on old geometry is
  stable. Old V is an accuracy refinement only — deferred.
- **Foot vs history split** in `SemiLagrangian.update_pre_solve`: foot =
  `x_new − dt·V_new(x_new − ½dt·V_new)` on the **current** mesh; history value =
  old field on the **old** geometry. Drop `_activate_ale_for_traceback`,
  `_pending_v_mesh_disp`, the `i=0` gate for this path.
- **CN check DONE (2026-06-18): CN θ=0.5 + old-frame `DuDt` + stock `DFDt` + BDF1
  is STABLE** (T∈[0,1], 40 steps, h_max→0.092, Nu→90) — `oldframe_cn_dudtonly.log`.
  So **only the advective `DuDt` needs old-frame**; the diffusive flux history
  (`DFDt`) is fine under stock ALE (dissipative/low-amplitude). Stage 0 = one
  patch. **BDF2 also DONE** (CN θ=0.5 + order=2 + old-frame `DuDt` + stock `DFDt` → stable,
  T∈[0,1], 40 steps, Nu→94 — `oldframe_cn_bdf2.log`): the "one clone, any order"
  invariant holds — both history levels trace to the same 1-step foot on the old
  geometry (new `psi_star[1]`=old `psi_star[0]` traced; new `psi_star[0]`=current
  field traced). **Plan fully informed on the accuracy axis: CN + BDF1 + BDF2 all
  validated.** DFDt-stock-OK was at Ra=1e5 κ=1; higher-Ra may need DFDt old-frame
  too (same machinery — cheap option, not default).
- **Scope: free-surface / SL-deform path.** Leave the *working* interior-node
  adaptation path (its `v_mesh`/`on_remesh` route) **untouched** — do not risk a
  regression in a working subsystem. (Unifying adaptation onto old-frame is real
  and desirable — see Stage 1 — but gated behind the adaptation regression suite.)
- **Retire aggressively where safe (tech-debt paydown).** Remove genuinely-dead
  code the old-frame path obsoletes; keep only what the still-living adaptation
  path needs. Full removal of `CARRY`/`v_mesh`/`i=0`-gate waits on Stage 1.
- **Validation gates:** free-surface Ra=1e5 res-20 holds T∈[0,1] under CN+BDF1 and
  CN+BDF2 (matching the BE prototype's stability, now with the accurate scheme);
  adaptation regression bit-comparable; serial first, then parallel.

### Stage 1 — efficient & unified (deferred; note the lever exists)
- **(c) `DMClone` shared-SF + independent coords** replaces the ephemeral restore:
  refresh the lagged geometry by local coord memcpy (no `nuke_coords_and_rebuild`),
  sample via `global_evaluate` on the clone. Requires Phase-1 mixed-mesh eval
  (the `global_evaluate`-on-a-non-current-DM machinery) — the large piece kept out
  of Stage 0.
- **Unify adaptation onto old-frame** (regression-gated), then fully retire
  `CARRY` + `v_mesh` + `on_remesh` ALE branch + `i=0` gate.
- Optional accuracy: carry old V for temporally-consistent / 2nd-order-midpoint
  reach-back (needs mixed-mesh V — `V_old` on clone + `V_new` on working mesh).
- **E (cost/cadence)** decisions (locator rebuild amortisation, refresh ordering)
  belong here, once the algorithm is fixed.

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

## Stage 0 — landed (2026-06-18): old-frame reach-back, cost/benefit

Stage 0 (a) — *ephemeral old-geometry restore on the working mesh* — is
implemented and validated as the production cure for the high-Ra free-surface
SL blow-up. It is the minimal subset of this design: no clone, no mixed-mesh
evaluator. This section records what it costs, what it leaves on the table, and
the recommendation on whether to proceed to the rolling clone (Stage 1 (c)) now.

### What landed

`SemiLagrangian.old_frame_traceback` (opt-in, default `False`; forwarded through
`AdvDiffusionSLCN(..., old_frame_traceback=True)` to the **advective `DuDt`
only** — the diffusive `DFDt` keeps stock ALE, validated sufficient at Ra=1e5).
Three hooks in `systems/ddt.py`:

1. `on_remesh` stashes the pre-move geometry (`_oldframe_X = ctx.old_X`, earliest
   since the last solve) **instead of** a `v_mesh` displacement, and leaves
   `psi_star` `CARRY`'d.
2. `update_pre_solve` computes the departure foot from the **physical** velocity
   (no `v_mesh`; `_ale_active` is `False`), records `psi_star[0]` by **direct
   nodal carry** (not a re-evaluate on the deformed mesh — see below), skips the
   new-mesh bounds clamp, and samples `psi_star` inside
   `with mesh.ephemeral_coords(): mesh._deform_mesh(_oldframe_X); global_evaluate(...)`.
3. The one-step stash is consumed at the end of `update_pre_solve`.

**De-risked:** a variable's `.data` is bit-identical through a
`_deform_mesh(old) → _deform_mesh(new)` round-trip (`nuke_coords_and_rebuild`
rebuilds the DS / DM / DOF-coordinate caches but never touches the solution
`Vec`), and `_deform_mesh` invalidates `_dminterpolation_cache` + the evaluation
cache — so the ephemeral sample is correct and not stale. The clone (Stage 1) is
therefore **not required** for correctness; it is an efficiency / capability
play.

**Load-bearing detail:** the first real-API run blew up *like the baseline*
despite old-frame being active, because the standard `store_result` path
**re-records** `psi_star[0]` by evaluating `psi_fn` on the *deformed* mesh at
centroid-shifted nodes — injecting boundary-layer interpolation error that grows
with `h_max` and then rides the old-geometry sample. Recording the history by a
**direct nodal carry** (reusing the parallel `_record_psi_star_from_field_data`
path) restores the prototype's exact behaviour. This is the "store primitives,
not re-derived values" principle of invariant 3, in miniature.

### Validation (Ra=1e5 annulus, res 20, free surface, CN)

| run | worst T over 30 steps |
|-----|-----------------------|
| baseline (stock ALE) | `[-311, +333]` — blow-up by ~step 20 |
| old-frame, **SLCN** (BDF1 + CN flux, θ=0.5) | `[0.000, 1.000]` |
| old-frame, **SL-BDF2** (BDF2 + flux at n+1, θ=1) | `[0.000, 1.000]` |

`h_max` reaches ~0.067 (surface deformed ~2.3 cells), Nu climbs 12→55 — vigorous,
not over-diffused. Old-frame is **order-agnostic** — it changes only *where* the
history is sampled, so both the canonical SLCN and SL-BDF2 schemes hold T∈[0,1].
(The two scheme knobs must be paired: SLCN = BDF1 time-difference + Crank-Nicolson
flux; SL-BDF2 = BDF2 time-difference + flux implicit at n+1 — a BDF2 stencil with a
CN flux is *not* a consistent 2nd-order scheme. See
[`docs/advanced/semi-lagrangian-time-integration.md`](../../advanced/semi-lagrangian-time-integration.md)
and Bonaventura et al., 2021.) Unit regression: `tests/test_0855_oldframe_sl_traceback.py`
(no-op on static mesh bit-identical; geometry restored + stash consumed; samples
on old geometry not new; bounded on a moving mesh; BDF2) — passes serial and
np=2; the existing `ddt` suite is unchanged (opt-in, default off).

### Cost of Stage 0 (a)

Measured on the Ra=1e5 res-20 free-surface loop (`oldframe_overhead_bench.py`):

- **`adv_diff.solve`: +166 ms/step (+8 %)** over baseline (2033 → 2199 ms/step),
  from **+2 `nuke_coords_and_rebuild` per solve** (deform-to-old + restore) for an
  order-1 `DuDt`. Order *N* wraps the sample per history level → `2N` rebuilds /
  step (a hoist to one ephemeral block per step is an obvious Stage-0
  optimisation, deferred).
- **Hidden churn not in that 8 %:** `_deform_mesh` unconditionally sets
  `is_setup = False` on *every* registered solver, so each ephemeral round-trip
  also forces the **Stokes** DM/assembly to rebuild on its next solve — even
  though the working mesh returns to bit-identical coordinates. This is the real
  Stage-0 tax at scale (Stokes assembly ≫ a `nuke`), and it is **entirely
  avoidable**: a guard that skips the `is_setup` invalidation when
  `ephemeral_coords` restores identical coordinates would remove it without the
  clone. Recommended as the first Stage-0 follow-up if the tax bites.

### What Stage 0 leaves on the table (the Stage-1 (c) clone)

The rolling clone (DMClone, shared partition, independent coords, refreshed by
local memcpy; sampled via the **mixed-mesh evaluator**, Phase 1) would:

- **Eliminate both costs above** — the working mesh never moves, so no per-step
  `nuke` and no solver-rebuild churn; the clone refresh is a local coord memcpy.
- **Enable old `V` on the clone** → second-order *temporal* reach-back (the
  current foot uses only the current velocity).
- **Recompute fluxes / gradients at full FE accuracy** on the clone, retiring the
  lossy projected histories (`DFDt`, `psi_star_flat`, VE stress projection).

Its build cost is the large piece deliberately excluded from Stage 0: the
mixed-mesh `global_evaluate` (per-mesh point location + pointwise combine,
shared-partition fast path) and **verifying derivative/vector `global_evaluate`
on a non-current DM** (the `scalar_component` wrinkle, invariant 5 / Phase-1 task
1).

### Recommendation

**Ship Stage 0 (a) now; defer the clone (Stage 1 (c)).** Stage 0 delivers the
correctness cure with ~8 % on the advection solve and zero new infrastructure,
and (Task 2 below) is mesh-agnostic — it covers interior-node adaptation on the
same code path. The clone's payoff is *efficiency and second-order/flux
capability*, not correctness, and it costs the whole mixed-mesh evaluator. Build
it when one of these triggers fires, not before:

1. The Stokes-rebuild churn is shown to dominate a production run **and** the
   cheap `is_setup` guard above does not remove it; or
2. a feature needs old `V` (second-order temporal reach-back) or full-accuracy
   flux/stress recompute on a moving mesh (VE on a free surface / adapting mesh);
   or
3. the per-step `nuke` cost becomes material at higher order or resolution.

Until then the v_mesh/`CARRY`/i=0-gate apparatus stays as the default path; old-
frame is the opt-in cure. Retiring that apparatus is gated on unifying the
adaptation path onto old-frame (Task 2) and is tracked separately.

### Task 2 — old-frame through interior-node adaptation

_(see `~/+Simulations/fs_convection_goal4/oldframe_adapt_comparison.py`;
no-slip Ra=1e5 res-16 annulus convection, periodic interior adaptation, old-frame
vs the current `v_mesh` path; gentle `refinement=1.3` — aggressive refinement
destabilises the **base** convection regardless of transfer mode, so it is not a
fair test)._

Old-frame is mesh-agnostic: `on_remesh` stashes the old geometry for *any* node
move (free surface or mmpde/OT interior adaptation), so the same code path
applies. The comparison (30 steps, 7 adaptation events each):

| mover | v_mesh worst T | old-frame worst T | final Nu (vm / of) | worst q (vm / of) |
|-------|----------------|-------------------|--------------------|-------------------|
| `follow_metric` (anisotropic) | **[-6.97, +4.27]** | **[-0.51, +1.07]** | 16.4 / 16.0 | 0.778 / 0.783 |
| `OT_adapt` | [-0.014, 1.000] | [0.000, 1.000] | 21.8 / 21.7 | 0.762 / 0.762 |
| `smooth_mesh_interior` | [0, 1] (n_adapts=0) | [0, 1] (n_adapts=0) | 32.3 / 32.1 | 0.826 / 0.826 |

**Conclusion: old-frame MATCHES or BEATS the current `v_mesh` path with no
regression.** For `follow_metric`'s anisotropic interior motion it is
*substantially* better bounded — `v_mesh` overshoots to `[-7, +4]` (the same
new-mesh-frame fold that fails at the free surface), while old-frame holds
`≈[0, 1]` — at equal Nu and equal-or-better mesh quality. For `OT_adapt` (whose
internal reset + FE-remap already keeps the history clean) both are bounded, old-
frame marginally better. `smooth_mesh_interior` on a uniform mesh is a no-op
(equant cells → no move); the near-identical Nu confirms old-frame does not
perturb a no-adaptation run.

So old-frame is the **single SL reference for both free surface and adaptation**
(invariant 6), and is the path to eventually retiring the `v_mesh`/`CARRY`/i=0-
gate apparatus. That retirement is held back only by sequencing/benchmarking
discipline (it is the default path today and old-frame is opt-in), not by any
correctness gap — and is tracked as the unification follow-up.

