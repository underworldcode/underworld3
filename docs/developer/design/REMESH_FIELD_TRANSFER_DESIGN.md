# Field transfer on mesh adaptation — design

Status: **design / spec for implementation.** Phase-1 (REMAP) not yet built.
A working **band-aid** is in `ddt.py` (see §7). This doc is the entry point for
the implementation session.

## 1. Problem

When an adaptive mover repositions mesh nodes, every field defined on the mesh
must be brought onto the new node layout. Today this is done **by user code**
(the harness hand-remaps `T` after calling the mover), which structurally
cannot transfer variables the user does not know exist — solver history
(`DuDt.psi_star`), projection/Hessian work-vars, RBF proxies, etc.

The concrete failure (parallel adaptive `AdvDiffusionSLCN`, see memory
`project_slcn_psistar_seam_remap`): a 30–50% temperature spike at np-partition
seams, traced to the SLCN history `psi_star` being re-recorded by a rank-local
`evaluate` of the field **at its own (on-vertex) node coordinates**, which
mis-locates at a process boundary. Root cause is *hidden-variable transfer*,
not the φ-solve / remap of `T` / monotone clamp / implicit solve (all
exonerated by a long isolation; see the memory).

**Goal:** move field transfer out of user code and into the adapt operation,
with a per-variable policy so hidden variables fail *safe* (transferred
needlessly) rather than *silently wrong* (stale).

## 2. Transfer-semantics taxonomy — four kinds

| policy | post-adapt value = | for |
|---|---|---|
| **REMAP** (default) | old field re-sampled at the new node positions | any Eulerian stored quantity; *all* DuDt history |
| **CARRY** | unchanged DOF value (belongs to node/material identity) | genuinely Lagrangian fields (nodes move *with* material) |
| **REINIT** | nothing — recomputed from source before next use | stateless work-vars (gradient/Hessian projections, RBF proxies) |
| **ALE** | CARRY + a lazy `v_mesh` correction (§6) | history of a material derivative (adv-diff, NS, VE), as an opt-in refinement of REMAP |

**REMAP is the safe universal default** — literally "the previous values are
those at the original mesh points," correct to first order or better for any
Eulerian quantity. ALE is a refinement, never forced.

### 2a. CRITICAL: DuDt history is REMAP, never REINIT

`DuDt.psi_star` is **persistent, unrecoverable state**, not a recomputable
work-var. For order ≥ 2 the older terms (`psi_star[1]`, …) are *accumulated
upstream history* — each is the field traced back along the velocity field of
an **earlier** step. Once Stokes overwrites `V` with the new step's velocity
and the older solution is gone, those terms cannot be rebuilt. The
shift-history step (`psi_star[i] = φ·psi_star[i-1] + (1-φ)·psi_star[i]`,
ddt.py ~2035) reads **every** term, so **all** must be correctly transferred
before `update_pre_solve`.

⇒ history terms get REMAP (or ALE). **Never REINIT.** REINIT is strictly for
variables that hold no memory and are recomputed from scratch each solve.

The order-1 case *looks* recomputable (the re-record rebuilds `psi_star[0]`
from `u_Field`) — which is why the band-aid (§7) sufficed at θ=1 — but that is
a coincidence of order 1 and must not be generalised.

## 3. Interface — per-variable policy + operator hook

Two levels, because ALE/history transfer is not expressible per-variable
(it moves a field *and* its history coherently under one mesh velocity —
"T and Tdot together"):

1. **`MeshVariable.remesh_policy`** — enum `REMAP` (default) / `CARRY` /
   `REINIT` for standalone fields and the common case. The **framework**
   stamps this on hidden vars it creates (projections/proxies → `REINIT`);
   the **user** only flags their own `CARRY` fields. Default-unflagged ⇒
   REMAP ⇒ a forgotten variable fails safe.
2. **Operator `on_remesh(ctx)` hook** — solvers / `DuDt`s register one and
   handle their *own* field+history set coherently (the SLCN `DuDt` decides
   REMAP-all-history vs ALE-carry+`v_mesh`). Operator-managed vars are marked
   so the generic per-variable pass defers to the hook.

**The adapt op** (a new `mesh.adapt` / inside `smooth_mesh_interior` /
`OT_adapt` / `follow_metric`) becomes the single owner of the transfer dance
the harness does by hand today:

```
snapshot old coords + old field data (for REMAP/ALE vars)
move nodes to new_X   (the mover; see §5)
for each registered var: apply remesh_policy
    REMAP  -> var.data = interp(old field) at new positions   # robust, see §4
    CARRY  -> leave
    REINIT -> leave / mark stale (recomputed on next use)
for each registered operator: operator.on_remesh(ctx)         # ALE etc.
```

`ctx` (a `RemeshContext`) carries: old coords, new coords, total displacement
`Δx`, a bound interpolator, `dt`, and a scratch slot where operators stash
`v_mesh`.

### 3a. REMAP is robust by construction

REMAP evaluates the **old** field at the **new** node positions. Those are
generic *interior* points of the old mesh, so `global_evaluate` locates them
fine — this is exactly the `T`-remap that already works and that FIXED_DEFORM
proved clean (np4-vs-serial 7e-4). It is **not** the degenerate "evaluate a
field at its own vertices" that the band-aid had to dodge (that is the
per-step re-record, a different operation). So the general REMAP does not
inherit the band-aid's on-vertex fragility.

Implementation: snapshot old field on the old mesh; after the move, do the
proven *deform-back → evaluate-at-new → deform-forward* dance **once for all
REMAP vars together** (not per-var, not per-deform).

## 4. Authority for the policy

- `MeshVariable.remesh_policy` default = **REMAP**.
- The **framework** stamps the hidden vars it creates:
  `Vector_Projection`/`Hessian`/gradient targets → `REINIT`; swarm proxies →
  `REINIT` (re-projected from the swarm — but note §8 staleness gap).
- A `DuDt`/solver stamps its history vars and registers `on_remesh`.
- The **user** flags only their own Lagrangian fields → `CARRY`.

## 5. Audit: deferring the update for multi-shot movers (OT n_outer≥12)

Findings (smoothing.py, discretisation_mesh.py):

- **`_deform_mesh` does NOT touch field data** (discretisation_mesh.py:2001).
  It writes coords, calls `nuke_coords_and_rebuild`, rebuilds the `_coords`
  view, sets every registered solver `is_setup=False`, invalidates the
  evaluation hash + DMInterpolation cache, bumps `_topology_version`, syncs
  submeshes. **Field `.data` is untouched.** ⇒ field transfer is *already*
  one-shot and external to the mover; there is no per-inner-deform field
  remap to eliminate.
- Each mover (`_winslow_elliptic` ~1570, `_winslow_equidistribute` ~1861,
  `_winslow_anisotropic` ~2639) calls `_deform_mesh` **once per outer step**
  (inside the `for outer in range(n_outer)` loop). So OT pays
  `nuke_coords_and_rebuild` **×n_outer**.
- `nuke_coords_and_rebuild` (discretisation_mesh.py:1757) per call:
  `clearDS`/`createDS`; `createCoordinateSpace` + `dm_force_coordinate_field`;
  **kd-tree rebuild** (`_build_kd_tree_index`); `_get_mesh_sizes` (centroids /
  radii / search_lengths); `copyDS`; invalidate normals + face control points;
  and **refill the coord cache for every registered variable**
  (`for _var in self.vars: self._get_coords_for_var(_var)`, line 1890).
- The mover's inner loop *consumes* the rebuilt state: it calls
  `uw.function.evaluate(metric, Dcoords)` (needs the **kd-tree** of the
  just-deformed mesh) and `solver.solve()` (needs the **DS**) every outer step.
  ⇒ the kd-tree + DS rebuild are **genuinely needed per inner step** and
  cannot be naively deferred.

**So the deferral opportunity is narrower than "defer the rebuild":**

- The **avoidable waste** is the per-variable coord-cache refill (line 1890):
  it refills *all* registered vars (T, V, P, every `psi_star`, …) on every
  inner deform, but the mover only needs its **own work-vars** (metric, φ,
  gradients). Refilling the user fields' caches n_outer× is pure waste.
- That refill is a **parallel-correctness BUGFIX (#130)**: it was made eager
  to avoid a subset-of-ranks collective deadlock in `_get_coords_for_basis`
  (lazy rbf refill). So any deferral must keep collective access correct —
  defer the *non-mover* vars' refill to a single full refill at sweep exit
  (safe, since user fields aren't accessed mid-sweep), and refill only the
  mover's work-vars inner-loop.

**Options for point 2 (ranked):**

1. **Scope the per-var refill (low risk, real win).** Give
   `nuke_coords_and_rebuild` an optional "active vars" set; during a mover
   sweep refill only the mover's work-vars; do one full refill at the end.
   Saves `(n_outer−1) × n_user_vars` refills. No context manager strictly
   needed — the mover already owns its sweep.
2. **Light vs full rebuild split.** Separate "geometry rebuild" (DS + coords,
   needed per inner step) from "search/eval rebuild" (kd-tree, centroids,
   per-var cache). But the mover's inner `evaluate(metric,…)` needs the
   kd-tree, so this only helps if the metric eval is reworked to not need the
   kd-tree (e.g. evaluate-at-own-nodes as a data read) — larger change.
3. **`mesh.batch_deform()` context manager.** Suppresses the non-mover-var
   refill + defers a single full rebuild of the search/eval state to exit.
   Cleanest API but must respect the #130 collective constraint and ensure no
   stale kd-tree/centroid read happens mid-sweep (the mover *does* read them,
   so "defer the kd-tree" is unsafe — the CM can only defer the per-var cache
   + field-side state, i.e. option 1 wrapped in a CM).

**Recommendation:** option 1 (scoped refill), optionally surfaced as a context
manager later. MA is one-shot so it is unaffected either way.

## 6. ALE = CARRY + lazy `v_mesh` correction

ALE leaves the history on its (now-moved) nodes — **CARRY, no re-interpolation,
no interpolation diffusion** — and corrects for the *arbitrary* mesh motion via
`v_mesh = Δx_total/dt` entering the **next** step's characteristic: the SL
trace-back runs along `(v − v_mesh)` so it samples the carried history at the
correct relative-upstream point. The correction is **lazy** (applied at the
next solve, not baked in at adapt) and `v_mesh` is a **one-step pulse** (zero
again until the next adapt). It must use the **total** displacement across the
whole mover sweep (`X_final − X_orig`), not per inner deform.

- The `v_mesh` correction is exactly what distinguishes ALE from plain CARRY:
  pure CARRY is only correct when nodes move *with the material*; a re-meshing
  adapt is arbitrary motion, so the correction restores correctness.
- First-order equal to REMAP (built-in cross-check: validate ALE ≈ REMAP on a
  smooth case, then trust it on the steep one), but avoids re-interpolating the
  *unrecoverable* higher-order history each adapt — the whole point.
- Fits the deferred-update theme: CARRY is the zero-cost thing during the
  sweep; the `v_mesh` correction is accumulated and applied once on consumption.

**ALE is a DDt property, scoped to the advective flavors** (`SemiLagrangian`,
`Lagrangian`), **not** per-solver. The material-derivative + characteristic
logic lives in the DDt; put it there and it works uniformly for adv-diff,
Navier–Stokes, and VE-Stokes-via-`DFDt` with the solver doing nothing special.
The adapt op supplies `v_mesh`. The `Eulerian` DDt has no advection ⇒ REMAP.
A reset/re-mesh adapt (large discontinuous displacement) is not ALE-able ⇒
REMAP.

## 7. The band-aid (already landed) and its relationship to the true fix

`ddt.py` `SemiLagrangian._record_psi_star_from_field_data()` + a guarded call:
under MPI, when `psi_fn` is a single mesh-variable component on this mesh, the
per-step "record current field into `psi_star[0]`" copies the field's nodal
data **directly** instead of evaluating it at its own (on-vertex) coords.
Serial keeps the validated shifted-evaluate path **bit-identical** (verified
6.66e-16); result: parallel adaptive adv-diff 0.137 → 6e-4.

Relationship to the true fix:
- It is **orthogonal** to REMAP: it fixes the *per-step re-record*, not the
  *adapt-time transfer*. **Keep it** — it is the correct parallel-safe
  re-record, and it is what later lets the (recomputable, order-1) `psi_star[0]`
  avoid a redundant REMAP. It is **order-1 only** (touches no `psi_star[1+]`);
  higher order needs REMAP of the full history stack regardless.
- A cleanup later: consider promoting the direct nodal copy to the standard
  re-record (un-gate from MPI) after re-validating serial — it is *more*
  correct than the shifted-evaluate, not just a band-aid.

## 8. Validation gaps / cautions to carry into the build

- **Order-2 untested.** The harness only ran θ=1 / order-1, so higher-order
  history transfer is unexercised. Build a 2nd-order adaptive case and verify
  the full `psi_star` stack transfers (it is the load-bearing path for §2a).
- **Swarm proxy staleness.** `SwarmVariable._proxy_stale` (swarm.py) is a lazy
  flag **not set by `_deform_mesh`**. Under adaptation a proxy `MeshVariable`
  may read stale on the new mesh. Proxies should be `REINIT` (re-project from
  the swarm) *and* the adapt must trigger `_proxy_stale = True` (or the
  re-projection) — otherwise a different silent-staleness bug.
- **Vector/tensor DuDt** (NS / VE flux history) under parallel adaptation is
  not covered by the band-aid and must be validated under Phase-1 REMAP.
- **Solver stability is paramount**: any change touching the DS rebuild /
  solver assembly path needs the tier-A/B suite + a bit-identical serial probe.

## 9. Phasing

1. **Phase 1 — REMAP in the adapt op.** Policy enum + framework stamping +
   adapt op owns snapshot/move/transfer (all REMAP, incl. full history stack).
   Replaces the harness's manual transfer; makes parallel adaptation correct
   for *all* hidden vars and all DDt orders. Keep the band-aid. Add the scoped
   per-var refill (§5 option 1). Validate: parallel adapt sweep np4-vs-serial
   (target ≤ no-adapt baseline), order-2 case, tier-A serial bit-identical.
2. **Phase 2 — ALE on the advective DDt.** `on_remesh` hook: CARRY history +
   stash total-`Δx` `v_mesh` for the next `(v − v_mesh)` trace-back. Validate
   against Phase-1 REMAP via the first-order agreement; benefits adv-diff +
   VE-Stokes; removes the interpolation-diffusion cost on 12-shot OT.

## File map

| file | site | role |
|---|---|---|
| `discretisation/discretisation_mesh.py` | `_deform_mesh` @2001, `nuke_coords_and_rebuild` @1757 (per-var refill @1890), `vars` registry @3082 | coords/cache rebuild; scoped-refill change (§5) |
| `meshing/smoothing.py` | movers @1570/1861/2639 (per-outer `_deform_mesh`); `smooth_mesh_interior` @2683; `OT_adapt`, `follow_metric` | adapt op owns transfer; mover sweep refill scope |
| `systems/ddt.py` | `SemiLagrangian` (history @119, shift @2035, re-record @2085, band-aid `_record_psi_star_from_field_data`); flavors @98/108/119/146 | history policy = REMAP/ALE; `on_remesh` hook |
| `systems/solvers.py` | `AdvDiffusionSLCN`, NS, VE (`DuDt`/`DFDt` @997/1354) | use DDt; nothing solver-specific for ALE |
| `swarm.py` | `_proxy_stale` @309, `_update` @1019/2129 | proxy REINIT + adapt-triggered staleness (§8) |
