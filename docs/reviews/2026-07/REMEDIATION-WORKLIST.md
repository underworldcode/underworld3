# Remediation Worklist — July 2026 Quality Campaign (Cross-Dimension Synthesis)

**Status**: worklist v1, 2026-07-03 — synthesized from all six audit dimensions
**Base**: `development` @ `1d003481` (audit worktree, campaign index at `e848d131`)
**Sources**: `LOOSE-ENDS-AUDIT.md` (LE-01…26), `API-CONSISTENCY-REVIEW.md` (API-01…12),
`READABILITY-REVIEW.md` (READ-01…118), `SWARM-SUBSYSTEM-REVIEW.md` (SWARM-01…24),
`DOCS-STANDARDS-COHERENCE.md` (DOC-01…08), `BRANCH-TRIAGE-LEDGER.md`.

Abbreviation: `pyx` = `src/underworld3/cython/petsc_generic_snes_solvers.pyx`.

## Overview

This is the single ranked worklist for the remediation waves. Every actionable
finding from the six review documents is assigned to exactly one track below;
duplicates across dimensions are merged (the merge is noted on the row). Within
each wave, items are ranked **severity first, then effort** (S before M before L).

Tracks:

- **Track 0 — Bug fixes** (added by this synthesis): confirmed correctness defects
  the audits found behind TODOs/skips/docstrings. These do not fit the five
  cleanup waves and should ship as small individual PRs, generally *before* the
  wave that touches the same file. 19 items.
- **Wave A — deletions & dead code**: 27 items.
- **Wave B — internal deprecated-pattern migration**: 6 items (~41 call sites).
- **Wave C — API harmonization with deprecation shims**: 13 items.
- **Wave D — readability rewrites** (incl. the flagged benchmarked `.pyx` sub-wave D2): 94 items.
- **Wave E — docs alignment**: 11 items.
- **Branch-track** (parallel): 6 ordered actions covering ~80 branches/worktrees.
- **Follow-on**: 8 items (swarm modernization refactor, guardrails, deferred solver work).

Ground rules carried from the campaign: no `pyx` numerics without separate
benchmarking; API changes only via deprecation shims (no hard breaks); `tier_a`
green pre/post each wave, `tier_a or tier_b` before merge, np2/np4 for anything
touching swarm/migration/partition-sensitive code. **Unverified findings
(LE-13…26, API-10…12, READ-15…118, SWARM-15…24, DOC-08) must pass the same
adversarial verification as the verified tables before their fix is applied** —
they are scheduled here but each carries that gate implicitly.

Eighteen items need the maintainer's judgment before (or as part of) execution;
they are marked `[D#]` on their rows and collected in
**"Decisions needed from Louis"** at the end.

## Changes Made

None — planning document only. Each remediation PR will cite the finding ID(s)
it resolves and the track item below.

## System Architecture

How the tracks interlock (the cross-wave dependency map):

1. **Track 0 before Wave B in `swarm.py`/`ddt.py`** — the swarm bug fixes
   (BF-02…BF-08) touch the same functions Wave B migrates; fix first, migrate
   second, so the migration diff is against correct code.
2. **Wave A before Wave B for the swarm package** — deleting `swarms/pic_swarm.py`
   (WA-02) removes 18 of the ~41 deprecated-pattern sites from Wave B's scope.
3. **Wave B before Wave D in the same file** — the `ddt.py` rewrites
   (READ-17/18/19/20) and any swarm-file restructuring must wait for the
   access-pattern migration (WB-01/WB-02) so each PR stays reviewable.
4. **Wave C before Wave E's call-site sweep** — the BC arg-order shim (WC-01/02)
   must land before the ~1,370-site docs/tests sweep (WE-09).
5. **Wave E internal ordering** — authority map (WE-01) → style-guide rewrite
   (WE-02) → docstring-queue regeneration (WE-03) → docstring wave (WE-04),
   per DOC-04/01/02/05.
6. **Branch-track before file waves where branches collide** —
   `bugfix/yield-homotopy` (LAND) conflicts with `pyx` and
   `constitutive_models.py`: land it before Wave A/D `pyx` edits (WA-05…09,
   D-pyx groups). `bugfix/custom-mg-parallel` (KEEP_ACTIVE) extends
   `custom_mg.py` by +262 lines: Wave C's WC-04 and the D-custom_mg group wait
   for or coordinate with it. The `feature/elliptic-ma` EXTRACT (smoothing.py
   parallel allgather fix) should be rescued before the D-smoothing group edits
   the same function.
7. **Follow-on gates** — SWARM-12's lazification is *blocked* until BF-08
   (pre-solve proxy refresh) exists; the swarm self-validating-cache refactor
   (FO-01) subsumes BF-02's tactical fix but does not replace it.

## Track 0 — Bug fixes (small individual PRs, most-severe first)

| # | Finding(s) | Sev | Eff | Location | Action | Deps / notes |
|---|-----------|-----|-----|----------|--------|--------------|
| BF-01 | LE-01 | High | S | `function/_function.pyx:845` | Fix dead ternary so rbf-path derivatives use each expression's own component; regression test `v[1].diff(x)` with `rbf=True` | Standalone |
| BF-02 | SWARM-01, SWARM-02 (+SWARM-17 fold after verify) | High | S | `swarm.py:3451` | Move `_invalidate_canonical_data()` (+ kd-tree drop, proxy-stale mark) ahead of the migrate() no-move early return; add invalidation in `add_particles_with_global_coordinates` and (verify) `populate()` | Before WB-01; np2/np4 gate |
| BF-03 | LE-02 | High | M | `cython/petsc_maths.pyx:303` | Rewrite `CellWiseIntegral.evaluate()` against `mesh.dm`+`getDS()` (the `Integral` pattern); remove xfails `test_0501:182,195`; note the #171 caveat | Standalone |
| BF-04 | SWARM-06 | High | M | `systems/ddt.py:3197,3281,3549,3638` | Rewrite Lagrangian history component writes via `_data_layout` modern interface; add advecting-history tests for both classes | Counts toward Wave B for those lines |
| BF-05 | SWARM-04 | High | M | `swarm.py:480` | Separate "suppress migration" from "suppress PETSc sync" inside `migration_disabled()`; flush pending writes on exit; repair vacuous `ptest_0755:322-329` | np2 gate |
| BF-06 | SWARM-07 | High | M | `swarm.py:1457` | Empty-rank strategy: guard KDTree on 0 particles; starved-rank proxies untouched + warn (not silent zeros) | np4 test, all particles on one rank |
| BF-07 | SWARM-03 | High | M | `swarm.py:3114` | Deferred/context-exit migration for modern coordinate writes (docstring already promises it); never per-write (collective deadlock) | **[D12]**; after BF-02/BF-05 |
| BF-08 | LE-03 = SWARM-05 | High | M | `swarm.py:1075` | Stale-proxy hole (#215 Bug 3): single eager proxy refresh at solve entry; gate on `test_0006_memory_leak` | **[D11]**; unblocks FO-01's SWARM-12 lazification |
| BF-09 | READ-01 | High | S | `meshing/smoothing.py:3124` | 3D MMPDE `NameError`: implement `_signed_volumes` OR `NotImplementedError` + 2D-only docstring; level_1 test either way | **[D8]**; before D-smoothing group |
| BF-10 | LE-06 | Med | M | `tests/test_0813...:32` + `utilities/mathematical_mixin.py:795` | (a) Interim: rewrite the three Batman regression tests units-free (restores DM-corruption coverage now); (b) implement `UnitAwareDerivativeMatrix.__mul__`/`__neg__`, then re-unitize | (a) immediate; (b) units subsystem |
| BF-11 | LE-07 | Med | M | `swarm.py:665-706` | Align SwarmVariable array-view reductions to MeshVariable per-component-tuple contract; unskip `test_0850:32/72/335` | **[D6]** return-type behaviour change |
| BF-12 | LE-08 | Med | M | `function/functions_unit_system.py:306` | Quantity-valued coordinate lists in `evaluate()`: coerce, or document-unsupported and rewrite `test_0812` with supported forms | **[D7]** (units-family decision) |
| BF-13 | LE refuted-appendix #2 | Med | S | `meshing/cartesian.py:536` | `BoxInternalBoundary` rank>0 `UnboundLocalError`: bind/broadcast `boundaries`/`boundary_normals` on all ranks; then revisit the six `test_0502` MPI skips | One-liner + np2 test |
| BF-14 | SWARM-11 (fix part) | Med | S | `swarm.py:4885` | Keyword-explicit `super().__init__` so `verbose` stops landing in `recycle_rate` | Class fate is **[D5]** |
| BF-15 | READ-46 | Med | S | `systems/ddt.py:3167,3519` | Fix `_object_viewer` AttributeError (`self.psi` → `psi_fn`; guard `dt_physical`); delete Eulerian's dead commented copy | Verify first (unverified table) |
| BF-16 | SWARM-16 | Med | M | `swarm.py:4649` | Substep advection: particles crossing partitions evaluated on wrong rank from substep 2 — verify, then migrate (or global-evaluate) between substeps | Verify first; np2 trajectory test |
| BF-17 | SWARM-19 | Med | S | `swarm.py:3810` | `save()` writes different coordinate systems per IO branch — verify, then unify on model-unit coordinates | Verify first; both-branch round-trip test |
| BF-18 | READ-43 | Med | S | `discretisation/discretisation_mesh.py:3689` | Deprecated `points` setter silently discards the NDArray_With_Callback wrapper: fix to in-place write or remove the setter (it is already deprecated) | Verify first; mark `TODO(BUG)` immediately in Wave D if deferred |
| BF-19 | LE-04 | High | L | `tests/parallel/test_1017...:129` | #291 (Stokes_Constrained np>1 segfault): keep skip as-is; keep on the release-blocking list — fix is owned outside the cleanup waves | **[D17]**; on fix, unskip + drop serial-only restriction |

## Wave A — deletions & dead code (one or few PRs; behaviour-neutral by construction)

Gate: full `tier_a` identical pre/post; `tier_a or tier_b` before merge; `./uw build`
with `rm -rf build/` for any `pyx` touch. Maintainer signs off deletions per batch.

| # | Finding(s) | Sev | Eff | Location | Action | Deps / notes |
|---|-----------|-----|-----|----------|--------|--------------|
| WA-01 | LE-12 | Med | S | `discretisation/persistence.py` | Delete module + `__init__.py:209` import; update CLAUDE.md Key Files entry | **[D1]** delete vs warn-on-import shim |
| WA-02 | SWARM-09 (+LE-17 subsumed) | Med | S | `swarms/pic_swarm.py` | Delete the never-installed 1,534-line module + breadcrumbs (`swarm.py:70, 2449-2455`) | **[D4]**; after WA-03 decision; removes 18 Wave-B sites |
| WA-03 | SWARM-08 | Med | M | `swarm.py:3360, 4744` | Recycle/streak feature: excise `recycle_rate>1` machinery + docstring claims (or port working logic → moves to Track 0) | **[D4]**; either way add a guard/test |
| WA-04 | LE-09 = READ-44 | Med | S | `systems/ddt.py:2750-2785` | Delete both `if 0 and preserve_moments` blocks, `self.I`, and access remnants; `preserve_moments=True` raises `NotImplementedError`; test it | |
| WA-05 | READ-21 | Med | S | `pyx:1383,1386,1466,2022` | `raise("string")` → `ValueError`/`TypeError` with the same message (error-path only) | After yield-homotopy lands (pyx conflicts) |
| WA-06 | READ-14 | Med | S | `pyx:3884, 7140` | Delete `if True:` constant guards, dedent — byte-identical | Same pyx PR as WA-05/07 |
| WA-07 | LE-13 + READ-24 + LE-15 | Med | M | `pyx:4026-4066, 5148-5168, 2766-2778, 2684, 3624, 5119, 6355, 6470, 2746, 3216-3227, ...` | Delete fossil commented-out blocks and dead assignments; KEEP the 3-line clearDS/createDS rationale and one-line intent notes | Comment-only per pyx rule; rebuild + tier_a |
| WA-08 | READ-70 | Low | S | `pyx:1` | Delete 4 unused imports (`xmlrpc.client.Boolean` et al.) | Same pyx PR |
| WA-09 | READ-23 (deletion part) | Med | S | `pyx:5620` | Delete empty 'robust'/'fast' `pass` branches (ValueError on unknown); comment the kaskade/additive divergence — **do not change option values** | |
| WA-10 | READ-31 | Med | S | `discretisation_mesh.py:4492,4606,5544,5584` | Delete four dead methods (zero callers) | Verify first |
| WA-11 | READ-32 | Med | S | `discretisation_mesh.py:5699` | Delete dead-AND-broken `meshVariable_mask_from_label` | Verify first |
| WA-12 | READ-33 + READ-34 | Med | S | `discretisation_mesh.py:4548, 1063, 527, 695, 711` | Delete commented face-normal block, `if False:` chains, scratch | Verify first |
| WA-13 | READ-77 | Low | S | `discretisation_mesh.py:8,11,18,878` | Delete dead imports; de-duplicate CoordSys3D import | |
| WA-14 | READ-60 + READ-61 | Med | S | `discretisation/remesh.py:190,200` | Delete dead `_remap_one_var` and `_new_coord_cache` | |
| WA-15 | READ-115 | Low | S | `remesh.py:168` | Delete unreachable else-branch and impossible `if var is None:` guard; half-line fail-safe note | |
| WA-16 | SWARM-21 | Low | S | `swarm.py:4641,4682-4700,1988,1994` | Remove/gate unconditional hot-path prints; delete dead `corrector`/`evalf` signature params (commented implementation) | Param removal via WC-12-style shim if public |
| WA-17 | SWARM-22 = LE-14 | Low | S | `swarm.py:1235,1285,1342,1391,961-967,1136,1499,1184-1200,1242-1263` | Delete no-op sync TODO stubs, `_rbf_reduce_to_meshVar`, legacy/enhanced-array pass-throughs (retire `test_0530` with them), commented blocks | `sync=` kwarg itself deprecates via WC-12 |
| WA-18 | LE-16 | Low | S | `cython/petsc_discretisation.pyx:248-285` | Delete dead triple-quoted `petsc_dm_get_periodicity`; planning-file note if still wanted | |
| WA-19 | READ-68, READ-85, READ-86, READ-88, READ-93, READ-95 | Low | S | smoothing/ddt/solvers dead locals & stale comment blocks | One batch PR of small deletions | Verify each anchor |
| WA-20 | LE-18 | Low | S | `tests/test_quantities_simplified.py` | Delete (imports a nonexistent module); port any uncovered cases to real `quantities` tests first | |
| WA-21 | LE-19 | Low | S | `tests/test_0754:166,288; test_0756:164` | Delete UnitAwareExpression-era skips; keep the neighbouring correct xfails | |
| WA-22 | LE-20 | Low | S | `tests/test_0750_meshing_follow_metric.py:269,324,344` | Delete (or keep with reasons as-is — they already point at surviving coverage) | |
| WA-23 | LE-10 | Med | M | `tests/test_0620, test_0630` | Delete `test_0630`; rewrite `test_0620` to assert the *implemented* mesh-units semantics | **[D7]** |
| WA-24 | LE-11 | Med | M | 16 skip markers across 7 units test files | One decision for the `coord_units`/quantity-coordinate placeholder family: consolidate into one labelled aspirational module, or delete + record the proposal in `docs/developer/design/` | **[D7]**; overlaps BF-12 |
| WA-25 | LE refuted-appendix #1 | Low | S | `tests/parallel/ptest_0762:75` | Remove stale skip (bug #151 fixed 2026-04); re-run at np≥2 | |
| WA-26 | LE-22 | Low | S | `pixi.toml` / setup deps, `kdtree.py:8` | Remove unused `pykdtree` dependency (macOS OpenMP crash hazard); keep module as import point; full macOS test run | **[D13]** |
| WA-27 | DOC-06 | Med | S | `docs/developer/design/` | `git mv` 24 scripts + 5 PNGs to `design/experiments/exp-integrator/`; delete 6 `.trace.txt`; relocate `_repro_dminterp_multifield_bug.py`; update ~10 doc references | **[D16]**; link fixes verified by WE docs build |

## Wave B — internal migration off deprecated access patterns (~41 sites)

Gate: np2/np4 parallel tests for every swarm/ddt touch; `tier_a` pre/post.
Do after Track-0 swarm fixes and WA-02 (which deletes pic_swarm's 18 sites).

| # | Finding(s) | Sev | Eff | Scope | Action | Deps / notes |
|---|-----------|-----|-----|-------|--------|--------------|
| WB-01 | SWARM-13 (part) | Med | M | `swarm.py` — 13 live `with …access(…)` sites (1161, 2381, 3339, 3345, 3376, 3628, 3725, 4719, 4767, 4791, 4949, 4972, 4975) | Migrate to direct `.data` access | After BF-02…BF-08 |
| WB-02 | SWARM-13 (part) | Med | M | `systems/ddt.py` — 7-9 sites | Migrate; BF-04 already converts the Lagrangian blocks | Before READ-17…20 (Wave D same-file) |
| WB-03 | briefing inventory | Med | S | `utilities/adaptivity.py` — 4 sites | Migrate | |
| WB-04 | briefing inventory | Low | S | `pyx` — 7 deprecated-pattern refs | Verify: most are commented one-liners (delete in WA-07); migrate any live ones as comment-safe edits only | pyx rule applies |
| WB-05 | briefing inventory | Med | M | internal `mesh.data` refs (~14) | Migrate to `mesh.X.coords` | |
| WB-06 | SWARM-13 (part) | Low | S | `swarm.py:4317-4456` | Delete `_legacy_access` (zero callers) once WB-01 lands | |

## Wave C — API harmonization (shims only; zero-cost when unused)

Every shim lands with the two-test pattern (old signature = identical result +
exactly one DeprecationWarning; new signature = zero warnings). Conventions
C2–C9 from `API-CONSISTENCY-REVIEW.md` are adopted; **C1 is superseded by the
maintainer decision of 2026-07-04** (recorded in `UW3_STYLE_CHARTER.md` §6): the
ORIGINAL value-first order `add_<kind>_bc(value, boundary, ...)` is canonical, and
the NEWER boundary-first methods migrate to it.

| # | Finding | Sev | Eff | Action | Deps / notes |
|---|---------|-----|-----|--------|--------------|
| WC-01 | API-01 | High | M | Value-first arg order (per D2, decided 2026-07-04): migrate `add_nitsche_bc`/`add_rotated_freeslip_bc`/`add_constraint_bc` to `(value, boundary, ...)` with shims for their current boundary-first signatures; legacy trio already conforms | **[D2 DECIDED]** shim kept indefinitely pending contrary ruling; WE-09 sweep now targets only the newer methods' call sites |
| WC-02 | API-02 | Med | S | ONE datum name (`value`) with `conds=`/`g=` aliases; ONE direction convention; finish `components=` deprecation | **[D3]** name choice; same edit as WC-01 |
| WC-03 | API-04 | Med | S | `consistent_jacobian` → validated property `{False, True, "continuation"}`, falsy→False, else ValueError; NumPy docstring from the `pyx:71-91` comment | Bug-fix-flavoured (invalid values currently silently select Newton) |
| WC-04 | API-05 | Med | S | `SolverBaseClass.set_custom_fmg(...)` method (lazy-import pattern); deprecate `set_custom_mg`; unify on `builder=` | Coordinate with `bugfix/custom-mg-parallel` (BT); np2/np4 |
| WC-05 | API-06 | Med | S | Export `rotated_bc`/`boundary_flux`/`custom_mg` from `utilities/__init__`; add `BoundingSurface` + `register_*_surfaces` to `meshing/__init__`/`__all__` | Pure additions |
| WC-06 | API-07 | Med | S | `uw.quantity` = THE factory; `create_quantity` warns one cycle; expose `uw.UWQuantity`; update `test_0640` same PR | **[D14]** |
| WC-07 | API-08 | Med | S | `SNES_Poisson.__init__` reordered to `(mesh, u_Field, degree, verbose)` with `type(third) is bool` legacy shim + positional regression test | |
| WC-08 | API-03 | Med | S | Align `SNES_Vector.add_nitsche_bc` signature with the Stokes variant (`normal=`; `mask=` → clear NotImplementedError if unsupported) | Signature/docs only |
| WC-09 | API-09 | Med | S | Rename free function `boundary_flux_to_field` → `boundary_flux_field` (alias one cycle); document `scale = -1/buoyancy_scale` relationship — do NOT alias the parameters | |
| WC-10 | API-10 | Low | S | `SNES_Vector` `u_Field=None` contract: auto-create like `SNES_Scalar` or raise at construction | Verify first; pairs with READ-22 |
| WC-11 | API-11 | Low | M | `units=` mesh-constructor kwarg: settle live-vs-deprecated, fix self-contradictory docstrings, thread through remaining constructors if live | **[D7]**; can ride a units wave |
| WC-12 | SWARM-22 (kwarg part) | Low | S | Deprecate the no-op `sync=` kwarg on the four pack/unpack methods via keyword shim | With WA-17 |
| WC-13 | READ-65 | Low | S | Drop dead `relax`/`step_frac` params; `n_sweeps` → `max_cg_iters` with one deprecation cycle | With D-smoothing group |

## Wave D — readability rewrites (grouped by file; one PR per group)

Rule of thumb: pure code motion / rename / dedent, mechanically verified
(bit-identical outputs where the source review specifies). `D-doc` rows are
comment/docstring-only. All unverified rows re-verify at their remediation base.

### D-smoothing (`meshing/smoothing.py`) — after BF-09; rescue elliptic-ma EXTRACT first

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-01 | READ-02 | High | M | Delete the ~100-line inline duplicate of `_build_M_tensor()`; call the closure once pre-loop |
| D-02 | READ-05 | Med | M | Extract `_backtracked_move(area_floor=0.0)` + `_cap_step_to_edge_fraction` (×3 / ×2 copies; default keeps older movers bit-identical) |
| D-03 | READ-06 | Med | M | Rename the four mis-prefixed `_winslow_*` movers (keep `_winslow_anisotropic`); sweep scripts/docs or alias one cycle |
| D-04 | READ-07 | Med | M | Module-top `_MPI` import + `_global_min/max/sum/mean` helpers (kills ~19 inline imports / 43 allreduce sites) |
| D-05 | READ-11 | Med | S | Accept `resolution_ratio=None` explicitly, then remove `**_ignored` and warn on unknown kwargs |
| D-06 | READ-08 | Med | S | D-doc: rewrite the stale module docstring (one paragraph per mover; mmpde recommended-not-default) |
| D-07 | READ-09 | Med | S | D-doc: delete dead amp-inversion lines; document the envelope branch; add `'arc-length'` to option lists |
| D-08 | READ-10 | Med | S | D-doc: fix `smooth_mesh_interior` method lists to include `mmpde` |
| D-09 | READ-12 | Med | S | D-doc: document the real 5-key `mesh_metric_mismatch` return dict |
| D-10 | READ-66 | Low | S | `_warm_start_krylov` + `_solver_wiring` helpers; rename `_zig` |
| D-11 | READ-67 | Low | S | Extract duplicated radial/tangential displacement reweighting |
| D-12 | READ-69 | Low | S | Extract `_mean_edge_length` (×2) |
| D-13 | READ-04 | Med | L | LAST: split into `meshing/smoothing/` package; MUST re-export the cross-module private names (`_edge_pairs`, `_tri_cells`, `_pinned_mask`, …) |

### D-pyx-safe (naming/docs/dead-code only, per campaign rule) — after yield-homotopy lands

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-14 | READ-13 | Med | S | Rename `_maybe_install_snes_update` (banned hedging prefix) — but see BT-02: the unmerged `feature/snes-update-callbacks` tip already does this rename; cherry-pick instead of re-doing |
| D-15 | READ-22 | Med | S | Explicit if/else for `u_Field=None` in Scalar/Vector constructors (pairs with WC-10) |
| D-16 | READ-29 | Med | S | Dedent the byte-identical 12-line Picard/else solve tail to run once; fix the misleading comment |
| D-17 | READ-71 | Low | S | Fix F1-guard error message (says F0) |
| D-18 | READ-72 | Low | S | One class-level SNES convergence-reason table; both consumers format from it |
| D-19 | READ-73 | Low | S | One `_nondimensional_time` helper (×6 copies) |
| D-20 | READ-76 | Low | S | Rename `dim`→`cdim` locals; fix stale line-number comment |
| D-21 | READ-74 | Low | S | D-doc: delete or `TODO(DESIGN):`-ify unresolved editorial musings |
| D-22 | READ-30 (comment part) | Med | S | D-doc: comment the hardcoded `snes_max_it = 50` clobber now; behaviour change deferred to D2 **[D18]** |

### D2 — benchmarked `.pyx` structural sub-wave (NOT Wave-D-safe; own benchmark protocol) **[D10]**

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-23 | READ-03 | High | L | Extract four-way-duplicated setup/BC/solve helpers, **parameterizing the 2-vs-2 essential-BC label divergence** (`str(boundary)` vs `"UW_Boundaries"`) unless [D10] unifies it deliberately |
| D-24 | READ-25 | Med | M | Extract explicit-index Jacobian-construction helpers (Vector/MultiComponent) |
| D-25 | READ-26 | Med | M | Extract shared bd-residual/jacobian wiring |
| D-26 | READ-27 | Med | M | Extract `_gather_state_for_residual` / `_split_local_residual_by_field` |
| D-27 | READ-28 | Med | S | Align MultiComponent's missing `_current_jit_cache_key` with siblings (document first) |
| D-28 | READ-75 | Low | M | Extract monitor-toggle / GAMG-defaults option bundles (×4 / ×3) |

### D-mesh (`discretisation/discretisation_mesh.py`)

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-29 | READ-15 | High | L | Split the ~830-line `Mesh.__init__` into 8 named private methods (pure code motion) |
| D-30 | READ-16 | High | M | One module-level coords-update callback (the `__init__` version) used at all 3 diverged sites — coordinate with the `bugfix/deform-cache-invalidation` EXTRACT (`_mesh_version` gap, same territory) |
| D-31 | READ-36 | Med | M | Extract `_surviving_labels` with the safe getValueIS-first idiom (one copy uses the hard-abort-prone pattern) |
| D-32 | READ-37 | Med | S | Delete the stale inline KDTree vertex map; call the already-fixed `_build_vertex_map()` |
| D-33 | READ-38 | Med | M | Extract shared teardown/reinit/invalidate helpers for `_re_extract_from_parent` vs `adapt()` |
| D-34 | READ-39 | Med | M | Extract `_facet_outward_unit_normal` (×2 four-way dispatch) |
| D-35 | READ-40 | Med | M | `view()`: delete dead gathers; extract variable/boundary table printers |
| D-36 | READ-35 | Med | S | Collapse the duplicated length-scale try/except; narrow the bare excepts |
| D-37 | READ-41 | Med | S | Fix `all_edges_IS_dm` NameError-risk + misnaming |
| D-38 | READ-80 | Low | M | Rename the single-letter locals in `quality()` |
| D-39 | READ-79 | Low | S | De-triplicate the `_test_if_points_in_cells_internal` loop |
| D-40 | READ-82 | Low | S | Fix un-imported `Dict` annotation |
| D-41 | READ-83 | Low | S | Fix inverted no-op guards |
| D-42 | READ-84 | Low | S | Drop the forward-compat `level` param (or state "always 0 today") |
| D-43 | READ-42 | Med | S | D-doc: replace the UW2 `_legacy_access` docstring example |
| D-44 | READ-78 | Low | S | D-doc: fix the self-contradictory `tol > 0` docstring |
| D-45 | READ-81 | Low | S | D-doc: state the sanctioned reason at each `except Exception: pass` in the refresh paths |

### D-ddt (`systems/ddt.py`) — after WB-02

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-46 | READ-17 | High | L | Decompose the 515-line `update_pre_solve` into intention-named helpers |
| D-47 | READ-18 | High | M | One `_to_nondim_ndarray` helper for the ×6 unwrap dance |
| D-48 | READ-19 | High | M | Extract `_velocity_nd_at` for the near-identical node/midpoint velocity blocks |
| D-49 | READ-20 | High | L | Shared `_DDtBase`/module helpers for the ~250-line quintuplicated boilerplate |
| D-50 | READ-45 | Med | S | D-doc + one-line change: delete stale theta comment; pass `self.theta` on restore **[D9]** |
| D-51 | READ-47 | Med | S | Replace nested bare-except dispatch with explicit branch + narrowed except |
| D-52 | READ-87 | Low | S | Rename the leaked-loop-index work-variable name |
| D-53 | READ-90 | Low | S | Hoist redundant inline imports |
| D-54 | READ-92 | Low | S | Narrow the `register_remesh_hook` excepts + one-line comments |
| D-55 | READ-89 | Low | S | D-doc: replace drifted line-number comments with method names |
| D-56 | READ-91 | Low | S | D-doc: document (or remove) `Symbolic`'s interface-parity-only params |

### D-solvers (`systems/solvers.py`)

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-57 | READ-48 | Med | S | Rename `_maybe_install_auto_gauge` (banned prefix) |
| D-58 | READ-49 | Med | M | `_SmoothingLengthMixin` for the triplicated properties |
| D-59 | READ-50 | Med | M | Extract `_nondimensionalise_timestep`; narrow the bare `except: pass` |
| D-60 | READ-51 | Med | M | Extract `_global_max_diffusivity` / `_centroid_velocities_nd` (×4 / ×3) |
| D-61 | READ-52 | Med | M | Collapse `_apply_unit_aware_scaling` to a single path; specific exceptions |
| D-62 | READ-53 | Med | S | `projection_problem_description` double-counts smoothing/penalty: grep-first delete, else deprecate + fix assignment **[D9]** |
| D-63 | READ-54 | Med | M | Extract `_invalidate_solution_cache`; consider a transient-solve template |
| D-64 | READ-96 | Low | S | `expression` lambda → def with docstring |
| D-65 | READ-97 | Low | S | Initialise `_prev_effective_order` in `__init__` |
| D-66 | READ-98 | Low | S | Delete shadow imports |
| D-67 | READ-99 | Low | S | Define `CM_is_setup` once on the base; handle the pre-assignment case |
| D-68 | READ-94 | Low | S | D-doc: document `percentile` in `estimate_dt` |

### D-rotated_bc (`utilities/rotated_bc.py`)

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-69 | READ-56 | Med | S | Call `_zero_rows_local` at its three hand-inlined sites |
| D-70 | READ-57 | Med | S | Extract `_boundary_spec` normalizer |
| D-71 | READ-58 | Med | M | Extract the line search from the 177-line nonlinear solve; group destroys |
| D-72 | READ-55 | Med | S | D-doc: rewrite the stale "Development version / productizes prototypes" opening |
| D-73 | READ-100 | Low | S | Module constants for field ids |
| D-74 | READ-101 | Low | S | Move pressure-datum search into the LU branch; `TODO(BUG)` the parallel-unsafety |
| D-75 | READ-102 | Low | S | Plain sympy import (drop impossible-state try/except) |
| D-76 | READ-104 | Low | M | Split semicolon-chained PETSc lifecycle lines |
| D-77 | READ-105 | Low | S | One module-top `mpi` import |
| D-78 | READ-103 | Low | S | D-doc: rename `info`→`solve_result`; document return keys |

### D-custom_mg (`utilities/custom_mg.py`) — wait for `bugfix/custom-mg-parallel` landing

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-79 | READ-59 | Med | M | `LevelLayout` NamedTuple replacing magic `lay[3]` indices |
| D-80 | READ-106 | Low | S | Extract `_clone_dm_with_solver_discretisation` |
| D-81 | READ-107 | Low | S | Extract the serial zero-column guard beside the parallel one |
| D-82 | READ-108 | Low | S | Narrow the bare except around `getNumFields()` |
| D-83 | READ-109 | Low | S | Rename terse parallel-path parameters to match serial vocabulary |
| D-84 | READ-111 | Low | S | Split semicolon-packed lines |
| D-85 | READ-113 | Low | S | Rename `sub` prefix-string collision |
| D-86 | READ-110 | Low | S | D-doc: comment the RBF r=0 clamp |
| D-87 | READ-112 | Low | S | D-doc: lifespan marker on the legacy finest-only path |

### D-remesh (`discretisation/remesh.py`)

| # | Finding | Sev | Eff | Action |
|---|---------|-----|-----|--------|
| D-88 | READ-62 | Med | S | Hoist `REMESH_MONOTONE` env knob to a documented module constant |
| D-89 | READ-63 | Med | M | One `_write_var_data` helper naming the sanctioned swallow (×4 copies) |
| D-90 | READ-114 | Low | S | Rename `_snapshot_remap_data` → `_snapshot_var_data` |
| D-91 | READ-116 | Low | S | Rename `_remap_var_set` to its public alias; delete the alias line |
| D-92 | READ-117 | Low | S | Drop redundant comprehension tail; enumerate the two scratch-key conventions |
| D-93 | READ-64 | Med | S | D-doc: fix the wrong re-entrancy comment |
| D-94 | READ-118 | Low | S | D-doc: state the best-effort `on_remesh` hook contract (or decide to warn unconditionally) |

## Wave E — docs alignment (ordering matters: WE-01 → WE-02 → WE-03 → WE-04)

| # | Finding | Sev | Eff | Action | Deps |
|---|---------|-----|-----|--------|------|
| WE-01 | DOC-04 | Med | S | Adopt the one-governing-doc-per-topic authority map; repoint `CLAUDE.md:324`; record the table in `docs/developer/index.md` | First |
| WE-02 | DOC-01 | High | M | Rewrite the Style Guide's four stale normative sections (NumPy/RST docstrings, MyST `.md`, `mesh.X.coords`, drop Quarto front matter); also fix the non-running `swarm.data += …` "Preferred" example (SWARM-13 evidence) | After WE-01 |
| WE-03 | DOC-02 | High | S | Regenerate `review_queue.md` via `scripts/docstring_sweep.py`; add to release checklist | Before WE-04 |
| WE-04 | DOC-05 | Med | M | Docstring wave on the *verified* gaps: `uw.function.evaluate` params, `Swarm.advection` ×2, the checkpoint trio | After WE-03 |
| WE-05 | DOC-03 | Med | M | Backfill the changelog "May – early July" (~10-14 grouped entries); add the release-checklist sweep | |
| WE-06 | DOC-07 | Med | M | Status headers on the ~16 unmarked design docs (per-doc git verification before stamping) | |
| WE-07 | DOC-08 | Low | S | Fix `mesh-adaptation.md:247` + the `mesh.adapt()` docstring Examples (deprecated patterns) | |
| WE-08 | API-12 | Low | S | Convert `units.py` public docstrings Google→NumPy (`docstring_sweep.py` assists) | |
| WE-09 | API-01/02 sweep | Med | M | Update call sites of the NEWER methods (nitsche/rotated/constraint) to value-first order; the ~1,370 legacy-order sites already conform (per D2 decision) | After WC-01/02 **[D2 DECIDED]** |
| WE-10 | LE-23 | Low | S | Update the `disk_snapshot.py` "Phase 1 (this commit)" header (phases 2/3 shipped) | |
| WE-11 | SWARM-24 | Med | S | Interim: banner on the misleading 26-line `swarm-system.md` stub pointing at the swarm review; real doc is FO-01's deliverable | |

## Branch-track (parallel; per the ledger's execution protocol)

| # | Action | Scope | Deps / notes |
|---|--------|-------|--------------|
| BT-01 | **Safety pushes first** (pushing is not "touching") | Unpushed locally-unique commits: `bugfix/custom-mg-parallel`, `feature/adaptive-convection` (13 ahead of origin), `feature/adapt-on-top` (no remote), `bugfix/yield-homotopy` (local-only) | Immediate; zero risk |
| BT-02 | **EXTRACT rescues** (11 branches) — commit/PR the named unique content before anything is deleted | blog-posts revision+figures; cetz-figures figure set; elliptic-ma `mover=`/allgather fix/scripts; exp-integrator-freesurface paper draft; fault-convection reimplementation spec; gradient-plasticity spike (commit-to-preserve, do NOT PR); petsc-cell-hint tip + build-petsc.sh fix; snes-update-callbacks tip `b82acea7` (cherry-pick — also resolves READ-13/D-14); vep-two-stokes `ViscoPlasticExplicitElastic` + post-mortem; deform-cache-invalidation `_mesh_version` gap (rework as issue/small PR — feeds D-30); fault-system-workflow H2Ex examples (onto product-system) | **[D15]**; blocks deletion batches |
| BT-03 | **LAND PRs** (4) | `feature/adapt-on-top`; `bugfix/yield-homotopy` (expect pyx/constitutive conflicts — land before Wave A/D pyx edits); `worktree-product-system` (commit dirty polish, fold adaptive-convection's `_run.py` fix, coordinate workflows-package ownership); `feature/parallel-point-eval` tip | **[D15]** landing order/ownership |
| BT-04 | **Deletion batches** with archive tags (`git tag archive/<branch>` + push tags), re-verifying merge-ancestry + clean worktree per entry | (a) bulk 40 REMOVE_WORKTREE_ONLY; (b) investigated 8 REMOVE_WORKTREE_ONLY; (c) ARCHIVE_DELETE risk-none (15); (d) ARCHIVE_DELETE risk-low; (e) EXTRACT branches after BT-02 verified on development | **[D15]** per-batch sign-off; never key on worktree dir names (6 misnamed) |
| BT-05 | **Follow-up triage** of the two un-triaged items | `bugfix/rotated-freeslip-schur-pc` (local, clean) and `origin/bugfix/stokes-constrained-parallel` (holds the OPEN release-blocking Item2, 0.4% 3D velocity) | **[D17]** |
| BT-06 | **KEEP_ACTIVE guard list** — exclude from all cleanup | `feature/numpy2-support`, `feature/quality-audit-2026-07`, `bugfix/custom-mg-parallel`, `feature/adaptive-convection` | Standing |

## Follow-on (post-wave workstreams)

| # | Item | Scope |
|---|------|-------|
| FO-01 | **Swarm modernization design doc + refactor** (this campaign's dimension-5 deliverable) | Self-validating canonical cache (SWARM-10, generation-stamp discipline), migration trigger matrix (SWARM-03/18), shared array-view refactor for BOTH swarm and mesh variables (SWARM-14, incl. unifying the three `_data_layout` copies and deleting dead `_array_cache`), rank-local RBF seam behaviour (SWARM-15), `_get_map` stale-cache trap (SWARM-23), KDTree copy-in-`__cinit__` question (SWARM-20), checkpoint/restore fidelity audit; replaces the SWARM-24 stub with a real subsystem doc |
| FO-02 | **SWARM-12 safe parts** (can ship early) | `sym_1d` staleness check + `np.add.at` vectorization (bit-identical guard); lazification of `IndexSwarmVariable._update()` stays BLOCKED until BF-08 lands |
| FO-03 | **Guardrails: UW3 Style Charter + mechanical CI gates** | no-`maybe_`/hedging-name lint; deprecated-pattern scanner over src+docs in CI; shim-warning tests as the Wave-B/C regression net; `docstring_sweep` + changelog sweep on the release checklist; black/format check promotion; commented-out-code and bare-`except: pass` review checklist items (READ cross-cutting patterns) |
| FO-04 | **Partition-seam natural-BC fix, option (a)** (LE-05) | Partition-independent manual boundary-load assembly; the `pyx:2545` TODO(BUG) record stays until then |
| FO-05 | **Issue #291 fix** (BF-19) | Stokes_Constrained np>1 interior-multiplier section reduction; then unskip `test_1017` + serial-only restrictions |
| FO-06 | **Stokes_Constrained Item2** (0.4% 3D velocity, gauge-independent) | Release-blocking; lives on `origin/bugfix/stokes-constrained-parallel` (BT-05) |
| FO-07 | **Verification pass service** | Adversarial verification of every unverified finding (LE-13…26, API-10…12, READ-15…118, SWARM-15…24) at each wave's remediation base, before its fix is applied |
| FO-08 | **Units-family feature work** (if [D7] decides "implement") | `coord_units` / quantity-coordinate evaluate / UnitAwareArray returns, replacing the placeholder tests deleted in WA-23/24 |

## Decisions needed from Louis

1. **persistence.py fate (LE-12 / WA-01)** — delete `discretisation/persistence.py` outright, or leave a one-release warn-on-import shim? (Either way the CLAUDE.md Key Files entry is updated.)
2. **BC argument-order migration (API-01 / WC-01, WE-09)** — **DECIDED 2026-07-04**: the ORIGINAL value-first order `(value, boundary)` is canonical (most used in examples/benchmarks; matches previous major versions). The newer boundary-first methods migrate to it with shims. Remaining sub-decision: confirm the canonical datum name (`conds` implied vs `value`) — see #3.
3. **Canonical BC value-parameter name (API-02 / WC-02)** — `value` (proposed, self-documenting) vs `g` (zero-churn alternative); shim mechanics are identical either way.
4. **Recycle/streak swarms (SWARM-08/09 / WA-02, WA-03)** — port the working recycle logic from the dead `swarms/pic_swarm.py` into `Swarm`, or excise `recycle_rate > 1` and its docstring claims; `pic_swarm.py` (1,534 never-installed lines) is deleted either way.
5. **NodalPointSwarm fate (SWARM-11 / BF-14)** — keep with a smoke test, or deprecate; it has zero remaining instantiation sites and a positional-argument bug.
6. **Swarm reduction return types (LE-07 / BF-11)** — approve the behaviour change aligning SwarmVariable array-view reductions to MeshVariable's per-component-tuple contract (callers relying on scalar returns from multi-component reductions will see tuples).
7. **Units-family scope decision (LE-08, LE-10, LE-11, API-11 / BF-12, WA-23, WA-24, WC-11)** — one decision for the family: implement quantity-valued-coordinate `evaluate()` and the `coord_units` feature set, or declare them unsupported, delete the ~16 placeholder skips, delete `test_0630`, rewrite `test_0620` to the implemented mesh-units semantics, and settle whether the Cartesian-only `units=` constructor kwarg is live or deprecated (its docstrings currently contradict each other).
8. **3D MMPDE (READ-01 / BF-09)** — implement `_signed_volumes` (restoring the docstring's d=3 claim) or raise `NotImplementedError` and make the docstring 2D-only.
9. **Two one-line behavioural corrections found by the readability audit (READ-45, READ-53 / D-50, D-62)** — (a) pass `self.theta` when restoring SemiLagrangian state instead of the hard-coded 0.5; (b) stop double-counting the smoothing/penalty terms in `SNES_Vector_Projection.projection_problem_description` (delete if unused). Both are flagged rather than folded silently into Wave D.
10. **Benchmarked `.pyx` structural sub-wave (READ-03/25/26/27/28/75, READ-30 behaviour part / D2)** — approve running it at all, and rule on the 2-vs-2 essential-BC label divergence (`str(boundary)` in Scalar/SaddlePt vs `"UW_Boundaries"` in Vector/MultiComponent): preserve it deliberately, or unify it — unification is a solver-behaviour change needing its own validation.
11. **Stale-proxy fix shape (LE-03 = SWARM-05 / BF-08, issue #215 Bug 3)** — approve a single eager proxy refresh at solve entry (the #216-consistent design), gated on the memory-leak test, rather than per-access refresh.
12. **Automatic migration on modern coordinate writes (SWARM-03 / BF-07)** — approve deferred/context-exit migration semantics (the class docstring already promises automatic migration; per-write migration is collective-deadlock-prone and is ruled out).
13. **pykdtree dependency removal (LE-22 / WA-26)** — packaging/environment change: remove the unused dependency whose OpenMP runtime can crash macOS beside PETSc.
14. **`uw.create_quantity` deprecation (API-07 / WC-06)** — approve `uw.quantity` as THE factory, `create_quantity` warned for one cycle, and `UWQuantity` exposed at top level (`test_0640` currently pins `create_quantity` as public).
15. **Branch-track sign-offs (BT-02/03/04)** — per the ledger's execution protocol: (a) approve the rescue plan for the 14 HIGH-risk rows (work existing in exactly one place); (b) set the landing order and ownership for `feature/adapt-on-top`, `bugfix/yield-homotopy`, and the workflows package (`worktree-product-system` vs `feature/adaptive-convection` carry deliberate duplicates — who lands it); (c) sign off each deletion batch (bulk 40, investigated 8, ARCHIVE_DELETE none/low, EXTRACT-after-rescue).
16. **Design-directory cleanup deletions (DOC-06 / WA-27)** — approve deleting the 6 `.trace.txt` solver logs outright and moving the 29 experiment scripts/PNGs to `design/experiments/exp-integrator/`.
17. **Release-blocker confirmation (LE-04/#291, Stokes_Constrained Item2 / BF-19, BT-05, FO-05/06)** — confirm both stay on the release-blocking list and assign owners; neither is fixable inside the cleanup waves.
18. **`snes_max_it = 50` clobber (READ-30 / D-22)** — the solve path silently overwrites user `petsc_options` each solve: approve changing it to respect user settings (behaviour change, D2 benchmarked sub-wave) or keep the behaviour and only document it.

## Testing Instructions

Per-wave gates (campaign ground rules) plus the finding-specific validation
recorded in each source review's own Testing Instructions section:

- **Every wave**: `./uw build` in the wave worktree; `pytest -m "level_1 and tier_a"`
  green pre/post; `pytest -m "tier_a or tier_b"` before merge.
- **Track 0 / anything touching swarm, migration, or partition-sensitive code**:
  `mpirun -np 2` and `-np 4` parallel suites; each bug fix lands with the named
  regression test from its source review (the unskip IS the regression test where
  a skip masked the bug).
- **Wave A**: behaviour-neutral by construction — full tier_a identical pre/post;
  `pyx` touches force `rm -rf build/` + rebuild (stale `build/src/*.c` trap);
  confirm via `strings <.so> | grep <marker>` where applicable.
- **Wave B**: `/check-patterns` over `src/` shows the migrated sites gone;
  shim-warning tests (Wave C) later enforce that internal code no longer
  exercises deprecated paths.
- **Wave C**: two-test shim pattern per item (old signature identical + exactly one
  DeprecationWarning; new signature zero warnings under `simplefilter("error")`).
- **Wave D**: pure-motion claims verified mechanically (bit-identical outputs per
  READABILITY-REVIEW Testing Instructions §2–8); D2 additionally requires the
  benchmark protocol and bit-for-bit BC-behaviour comparison at np1/np2/np4.
- **Wave E**: `pixi run docs-build` clean; `/check-patterns` over `docs/`;
  regenerated docstring queue sanity checks (DOC-02/05 cross-validation).
- **Branch-track**: re-verify `merge-base --is-ancestor` (or squash content-diff)
  and clean `status --porcelain` per entry at execution time; archive tag before
  every delete; push tags.

## Known Limitations

- Line numbers throughout are pinned to `development@1d003481`; every wave
  re-verifies its anchors at its own base. Finding IDs are the stable reference.
- The majority of Wave D rows (READ-15…118) and several rows in other waves are
  **unverified findings** — scheduled here as a triage queue, each gated on the
  FO-07 verification pass before its fix is applied.
- Item counts double-assign nothing: where two dimensions found the same defect
  the row lists all IDs and lives in one track only.
- The Branch-track verdicts drift with repository state; the ledger's execution
  protocol (re-verification per batch) is mandatory, not advisory.
- Wave sizing is intentionally uneven: Wave D is ~94 rows but mostly S-effort
  grouped by file into ~9 PRs; Wave B is 6 rows but its np-parallel gates make
  it the riskiest cleanup wave.

## Sign-Off

| Reviewer | Role | Status |
|----------|------|--------|
| Louis Moresi | Maintainer | Pending review — 18 decision items above require explicit sign-off |
| Claude (audit synthesis session) | Author | Complete 2026-07-03 |

*Underworld development team with AI support from Claude Code.*
