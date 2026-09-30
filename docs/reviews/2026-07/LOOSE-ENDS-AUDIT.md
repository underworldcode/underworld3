# Loose-Ends Audit — July 2026 Quality Campaign (Dimension 1)

**Status**: audit complete; findings adversarially verified 2026-07-03 (where marked)
**Base**: `development` @ `1d003481` (audit worktree, campaign index at `e848d131`)
**Scope**: TODO/TODO(BUG) markers, stubs, skipped/xfailed tests, disabled logic
(`if 0`, commented-out code) across `src/` and `tests/`.

Abbreviation used throughout: `pyx` = `src/underworld3/cython/petsc_generic_snes_solvers.pyx`.

## Overview

June 2026's ~132 commits left a large but **highly bimodal** population of loose
ends. At one extreme, several `TODO(BUG)` comments are among the best engineering
records in the codebase (the partition-seam natural-BC note at `pyx:2545` documents
three tested-and-rejected workarounds and three fix options). At the other, a
one-line dead ternary (`comp = 0 if ... == 1 else 0`, `_function.pyx:845`) silently
returns wrong derivative values for every multi-component variable on the rbf
evaluation path — a real correctness bug hiding behind a `TODO` that reads like a
note-to-self.

The skipped-test population divides cleanly into three families:

1. **Skips masking real, reproducible bugs** — the skip is honest, the bug is live
   (LE-04, LE-06, LE-07, LE-08). Two were reproduced live in this worktree's own
   built environment during verification.
2. **Aspirational placeholders** — ~30+ skips across the units test suite
   (`test_0620/0630/0730/0800/0803/0804/0811`) test interfaces that were *proposed*
   and never implemented, or implemented with different semantics. They provide
   false coverage: several now pass vacuously rather than skip (LE-10, LE-11).
3. **Legitimately-conditional markers** — correctly-annotated xfails guarding
   tracked open bugs (LE-26) or acknowledged fragile tests (LE-21). These should
   stay.

Every `file:line` cited below was read directly in this worktree; line numbers are
exact at `1d003481`. Twelve findings (LE-01 … LE-12) are adversarially verified;
fourteen (LE-13 … LE-26) had their evidence lines personally read by the author but
did not go through the full adversarial pass. Two claims were **refuted** during
verification and are recorded in the appendix so the same false leads are not
re-found — one of them (the `ptest_0762` skip) turned out to be *stale* rather than
masking anything: the bug it cites was fixed in April 2026 (issue #151, commit
`453e5063`).

## Changes Made

None — audit only. Proposed changes are classified per finding and are scheduled
for Wave A (deletions & dead code), the bug-fix track (real bugs found behind
TODO/skips), and the dimension-5 swarm review (API-inconsistency evidence).

## System Architecture

What this dimension's survey revealed about the affected subsystems, for the
maintainer:

**Function evaluation (`src/underworld3/function/`)** has two derivative-evaluation
paths: an accurate L2-projection path (`_function.pyx:973`) and a fast Clement/rbf
path (`_clement_to_work_variable`, reached via `rbf=True, force_l2=False`,
`_function.pyx:959-969`). The rbf path computes per-component Clement gradients
correctly and stores them keyed by `(source_var, component)` (lines 832-841), and
each derivative expression carries its correct data-column index
(`diffcls.component = data_loc`, line 845 in the registration block at 843-847) —
but the retrieval loop discards it (LE-01). The evaluation gateway for units
(`functions_unit_system.py`) accepts quantity *values* but not quantity-valued
*coordinates* in list form (LE-08); a whole family of unit-aware-coordinate
features (`coord_units`, `UnitAwareArray` returns) exists only as test-suite
placeholders (LE-11).

**Diagnostics (`src/underworld3/cython/petsc_maths.pyx`)** — `CellWiseIntegral`
still carries the PR #172 clone-DM pattern that was reverted out of `Integral`
(`88807c26`): a cloned DM with a single fresh P1 field is fed a global vector
packed for the *multi-field* mesh layout, over-counting by ~2× (LE-02; reproduced
live: `CellWiseIntegral(mesh, fn=1.0)` on the unit square sums to 2.0). It is
exported to users at `src/underworld3/maths/__init__.py:56`.

**Swarm subsystem (`src/underworld3/swarm.py`, `swarms/pic_swarm.py`)** is the
densest cluster of loose ends and feeds directly into the dimension-5 review.
Proxy staleness is handled *lazily by design* — the #216 fix (commit `af537d56`)
deliberately kept invalidation lazy to avoid an O(100 MiB) eager-refresh memory
regression (comment at `swarm.py:2699-2707`), which leaves one documented hole:
solvers that read the proxy's DM directly, without touching `.sym`, consume stale
data (LE-03). The array-view reduction interface diverges from MeshVariable's
contract (scalar vs per-component-tuple returns, LE-07). Four identical
`if sync: pass` placeholders make a threaded-through `sync=` parameter a silent
no-op (LE-14).

**Solvers (`pyx`)** — the loose ends here are mostly *records*, not bugs: the
`pyx:2545` block is a high-value account of a genuine parallel assembly defect
(≈0.027% natural-BC load under-count at partition seams) and its dead-end
workarounds (LE-05). The remainder is pre-refactor commented-out code (old setup
flow, superseded `clearDS`/`createDS` block whose 3-line rationale — "destroys
field registrations" — is worth keeping, old f0/F1 constructions, debug prints;
LE-13, LE-15). Per the campaign ground rule, only comment-level changes are
proposed for this file.

**Units test suite** — the largest false-coverage surface. The proposed
`mesh.units` interface was implemented with *different* semantics (model-owned
coordinate units; a conflicting `units=` parameter warns and is ignored,
`discretisation_mesh.py:274, 282-301`), so `test_0620`'s try/except-skip tests
now partly pass vacuously and partly skip — both outcomes test nothing (LE-10).
Sixteen skip markers across seven files guard the never-implemented
`coord_units`/quantity-coordinate features (LE-11); `UnitAwareExpression` tests
outlived the architecture that removed the class (LE-19). Separately, the units
system has one real regression-coverage casualty: all three mesh-variable-ordering
("Batman") regression tests die in `Projection.solve()` because
`UnitAwareDerivativeMatrix` defines no `__mul__`/`__neg__` and sympy's
`DomainMatrix.scalarmul` bypasses its `_sympy_()` protocol — so a DM-state
corruption guard has zero active coverage (LE-06).

**Time-derivative machinery (`systems/ddt.py`)** exposes a documented constructor
parameter (`preserve_moments`) whose entire implementation is dead behind
`if 0 and ...` (LE-09) — a user setting it gets a silent no-op.

**Compat shims** — `discretisation/persistence.py` is a pure re-export shim with
three "TODO: Implement" stubs for features that live nowhere else; its only
importer is `__init__.py:209` (LE-12).

## Findings

### Verified findings (adversarially checked)

Ranked most-severe-first. All dispositions respect the campaign ground rules (no
solver numerics, no hard API breaks).

| ID | Sev | Effort | Location | Category | Summary |
|----|-----|--------|----------|----------|---------|
| LE-01 | high | S | `src/underworld3/function/_function.pyx:845` | todo-real-bug | Dead ternary always selects component 0: rbf-path derivative evaluation of multi-component variables returns the component-0 gradient for every component |
| LE-02 | high | M | `src/underworld3/cython/petsc_maths.pyx:303` | todo-real-bug | `CellWiseIntegral` clone-DM/section layout mismatch over-counts ~2× (reproduced: fn=1.0 on unit square → 2.0); user-facing export |
| LE-03 | high | M | `src/underworld3/swarm.py:1075` | todo-real-bug | Stale proxy DM after swarm write: lazy refresh fires only on `.sym` access; solvers reading the proxy DM directly get stale data (issue #215 Bug 3, deferred by #216) |
| LE-04 | high | L | `tests/parallel/test_1017_custom_mg_parallel_mpi.py:129` | skipped-test-masking-bug | Skip masks issue #291: Stokes_Constrained segfaults at np>1 in the interior-multiplier section reduction, independently of custom-P |
| LE-05 | medium | L | `src/underworld3/cython/petsc_generic_snes_solvers.pyx:2545` | todo-real-bug | Partition-seam natural-BC under-count (~0.027% load, ~0.1% velocity in augmented Stokes_Constrained); TODO(BUG) block is a high-value engineering record |
| LE-06 | medium | M | `tests/test_0813_mesh_variable_ordering_regression.py:32` | skipped-test-masking-bug | Three skips kill ALL "Batman" DM-corruption regression coverage; root cause is `UnitAwareDerivativeMatrix * NegativeOne` TypeError in the Projection residual template (reproduced live) |
| LE-07 | medium | M | `tests/test_0850_comprehensive_reduction_operations.py:32` | skipped-test-masking-bug | Swarm array-view reductions return scalars where MeshVariable returns per-component tuples (reproduced live); 2 skips mask it, 1 skip is stale, 1 is blocked only by the xfailed missing global `std()` |
| LE-08 | medium | M | `tests/test_0812_poisson_with_units.py:25` | skipped-test-masking-bug | Three skips mask a real gap: `evaluate()` at quantity-valued `[(x, y)]` coordinates raises TypeError in `non_dimensionalise` (reproduced live); substantive Poisson-with-units coverage lost |
| LE-09 | medium | S | `src/underworld3/systems/ddt.py:2750` | disabled-logic | Documented `preserve_moments` parameter is a silent no-op — entire implementation dead behind `if 0 and ...` (lines 2750, 2760) plus "TODO: DELETE" access-pattern remnants |
| LE-10 | medium | M | `tests/test_0620_mesh_units_interface.py:64` | skipped-test-obsolete | 11 try/except-skip tests of a PROPOSED mesh-units interface; actual semantics differ (model-owned units) so several now pass vacuously — false coverage either way. `test_0630` (4 decorator skips) is a pure demonstration of the superseded proposal |
| LE-11 | medium | M | `tests/test_0804_backward_compatibility_units.py:17` | skipped-test-obsolete | 16 skip markers across 7 files are placeholders for the never-implemented `coord_units`/`UnitAwareArray`-return/quantity-coordinate feature family |
| LE-12 | medium | S | `src/underworld3/discretisation/persistence.py:51` | stub-verdict | Pure backward-compat re-export shim + three "TODO: Implement" stubs; only importer is `__init__.py:209`; module name promises functionality that does not exist |

#### LE-01 — rbf-path multi-component derivatives are wrong (`_function.pyx:845`)

Evidence (read at `1d003481`):

```python
comp = 0 if source_var.num_components == 1 else 0  # TODO: handle multi-component
grad = gradient_at_nodes[(source_var, comp)]  # shape (n_nodes, dim)
```

Both ternary branches are `0`. Per-component Clement gradients are correctly
computed and stored keyed `(source_var, c)` (lines 832-841), and each
`UnderworldAppliedFunctionDeriv` carries the correct data-column index
(`diffcls.component = data_loc`, registration block 843-847). Only retrieval is
broken: `v[1].diff(x)` with `rbf=True` returns the component-0 gradient. The L2
path (`:973`) is unaffected.

**Fix**: in the loop at 848-854, key by each expression's own component —
`grad = gradient_at_nodes[(source_var, deriv_expr.component)]` — and delete the
ternary. Add a regression test evaluating `v[1].diff(x)` with `rbf=True` on a
vector MeshVariable. rbf-path only; no solver numerics.

#### LE-02 — CellWiseIntegral ~2× over-count (`petsc_maths.pyx:303`)

The TODO(BUG) at 303-309 is accurate: lines 310-319 clone `mesh.dm`, attach a
fresh single-P1-field DS, and hand `DMPlexComputeCellwiseIntegralFEM` a global
vector packed for `mesh.dm`'s multi-field layout (`:298`). Reproduced live in the
worktree env: `CellWiseIntegral(mesh, fn=1.0)` on the unit square sums to 2.0
(control `Integral` = 1.0). The over-count reproduces even with a constant
integrand, so the corruption is the DM/section layout mismatch generally, not
only wrong-offset DOF reads. History: `57ec0176` (PR #172) introduced the pattern,
`88807c26` reverted it from `Integral`, `d65fe9a2` added the TODO + the two xfails
at `tests/test_0501_integrals.py:182,195`.

**Fix**: rewrite `evaluate()` to integrate against `mesh.dm` + `mesh.dm.getDS()`
(the current `Integral` pattern), then remove the two xfails. Caveat: this
reinherits the issue-#171 linear-time-growth behaviour `Integral` has — acceptable
for correctness parity, note it in the fix.

#### LE-03 — stale swarm proxy on DM-level access (`swarm.py:1075`)

`_update()` (1060-1073) only marks `_proxy_stale=True`; `_update_proxy_if_stale()`
is called in production only from `.sym`/`.sym_1d` (1627, 1643; same pattern at
2203). `Mesh.update_lvec()` (`discretisation_mesh.py:2924-2960`) reads `var.vec`
for every mesh variable including swarm proxies, with no staleness check, and no
solver-entry refresh exists. The reproducer tests work around it manually
(`tests/test_0112_swarm_add_particles.py:110, 821`). Issue #215 is **closed**
(by PR #216) with Bug 3 explicitly deferred — cite it as "closed; Bug 3 deferred",
not as an open issue.

**Fix constraint**: hooking the refresh into every DM-level access is exactly the
eager pattern #216 rejected (O(100 MiB) regression — `swarm.py:2699-2707`,
`tests/test_0006_memory_leak.py`). The maintainer-consistent variant is a single
eager refresh at solve entry (iterate swarm-proxied aux variables before
assembly). Verify against #215's reproducer and the memory-leak test.

#### LE-04 — skip masks release-relevant #291 segfault (`test_1017...mpi.py:129`)

The skip (129-135) is legitimate and self-un-blocking ("auto-enables once #291 is
fixed"), but the underlying defect is user-facing breakage of a shipped-on-
`development` solver: Stokes_Constrained segfaults at np=2 with plain GAMG
(canonical `test_1062_constrained_solcx` also segfaults), isolated to
`_constrain_interior_multipliers_in_section` (`pyx:5014` default-on flag, `:6925`
definition, `:7345` call site). Issue #291 is open.

**Disposition**: keep the skip and wording; keep #291 on the campaign's
release-blocking list; on fix, remove this skip and the sibling serial-only
restriction.

#### LE-05 — partition-seam natural-BC record (`pyx:2545-2583`)

Verbatim-verified TODO(BUG) block above the natural-BC registration loop (`:2584`):
interior facets with one local support cell at partition seams lose non-owned
closure DOFs at global assembly (~0.027% load under-count → ~0.1% velocity in
augmented Stokes_Constrained); documents three rejected cheap workarounds and
three fix options. Introduced by `09e8c734` (PR #265). Cross-ref to
`discretisation_mesh.py:758-761` (non-overlapped assembly DM) is real.

**Disposition**: keep — high-value engineering record. Track fix option (a)
(partition-independent manual boundary-load assembly) as a campaign follow-on.
Note: the `test_0502_boundary_integrals.py` MPI skips and issue #291 are
*adjacent territory*, not evidence of this bug (see refuted-claims appendix).

#### LE-06 — Batman regression coverage hole (`test_0813_mesh_variable_ordering_regression.py:32,111,179`)

All three tier_b tests die at `proj.solve()` (the Poisson solve succeeds) with
`TypeError: unsupported operand type(s) for *: UnitAwareDerivativeMatrix and
NegativeOne` — reproduced live by bypassing the skips. Fires in the Projection
residual template `(self.u.sym - self.uw_function) * self.uw_weighting_function`
(`systems/solvers.py:2648`); `UnitAwareDerivativeMatrix`
(`utilities/mathematical_mixin.py:795-1016`) defines no `__mul__`/`__neg__` and
sympy's `DomainMatrix` element-wise path bypasses `_sympy_()`. This is the only
test file covering the "batman" DM-corruption regression — zero active coverage.

**Fix (two parts)**: (1) implement/route `UnitAwareDerivativeMatrix.__mul__`/
`__neg__` (units subsystem) — the repair point is the DomainMatrix
scalar-multiplication path; (2) until then, rewrite the three tests *without
units* so the ordering/no-Batman coverage is restored immediately (the units
dependency — `units="metre"` at test setup — is incidental to what they guard).

#### LE-07 — swarm reduction-interface skips (`test_0850_comprehensive_reduction_operations.py`)

Measured live: `SimpleSwarmArrayView.max/min/mean/sum/std` (`swarm.py:665-706`)
return scalars for vector variables; `SimpleMeshArrayView`
(`discretisation_mesh_variables.py:2287-2360`) returns per-component tuples. Per-
skip triage: **line 32** and **line 72** mask this real bug (line 72 additionally
needs `vtype=uw.VarType.TENSOR` — requiring explicit vtype for ambiguous component
counts is *shared, deliberate* design, not a SwarmVariable bug: MeshVariable
raises the identical ValueError, `swarm.py:197-200`); **line 335** is stale — its
body passes today as written; **line 268** is blocked only by the legitimately-
xfailed missing global `MeshVariable.std()` (xfails at 163, 294, 313 are correct
unimplemented-feature markers). The file header (13-19) already declares the
implementation wrong.

**Fix**: align the SwarmVariable array-view reductions to the MeshVariable
per-component-tuple contract, then unskip 32/72; unskip 335 as-is. Note this is a
return-type behaviour change for callers relying on scalar returns from
multi-component swarm reductions. Feed into the dimension-5 swarm review.

#### LE-08 — quantity-coordinate evaluate gap (`test_0812_poisson_with_units.py:25,106,183`)

Reproduced live: Poisson with quantity BCs converges, then
`uw.function.evaluate(T.sym, [(x_qty, y_qty)])` raises
`TypeError: Cannot non-dimensionalise object of type <class 'list'>` for both
`pint.Quantity` and `UWQuantity`. Mechanism: `functions_unit_system.py:306-309`
passes the raw list to `uw.non_dimensionalise()`, whose protocol chain
(`units.py:643-878`) has no list/tuple-of-quantity handling. Adjacent xfails:
`test_0750_unit_aware_interface_contract.py:116,195,249`.

**Fix**: coerce lists/tuples of UWQuantity in the evaluate() gateway, then unskip;
or declare the format unsupported, document it, and rewrite the tests with
supported coordinate forms so the Poisson-with-units coverage is regained.

#### LE-09 — `preserve_moments` silent no-op (`ddt.py:2750, 2760`)

Documented at 1390-1391 ("Use moment-preserving projection (experimental)"),
accepted at 1464, stored at 1480; both implementation blocks are behind
`if 0 and self.preserve_moments and ...` — unreachable by short-circuit. Inner
"TODO: DELETE" commented-out `mesh.access` blocks at 2768-2770 and 2777-2781.
No caller in src/tests/docs ever passes `preserve_moments=True`.

**Fix (Wave A)**: delete both `if 0` blocks and the remnants (git remembers);
make `preserve_moments=True` raise
`NotImplementedError("preserve_moments is not currently implemented")` rather
than removing the parameter (avoids the API break); update the docstring.

#### LE-10 / LE-11 — aspirational units-test placeholders

LE-10: `test_0620` (11 skip sites, e.g. `:64`) tests a "PROPOSED" interface whose
real implementation differs — `Mesh.__init__` accepts `units=`
(`discretisation_mesh.py:274`) but the model owns coordinate units and conflicts
warn-and-ignore (`:282-301`). Because `mesh.units` now exists, several tests pass
vacuously instead of skipping — false coverage either way. `test_0630` carries 4
unconditional decorator skips ("Demonstrates proposed interface - not yet
implemented"). **Fix**: delete `test_0630`; rewrite `test_0620` to assert the
*implemented* semantics (units accepted, model precedence, warning on conflict).

LE-11: 16 skip markers, all unimplemented-feature placeholders for the same
family: exact reason "coord_units parameter not implemented" ×7
(`test_0730:91,117`; `test_0803_simple_workflow_demo:21`;
`test_0803_units_workflow_integration:24,262,328`; `test_0804:17`); sibling
"planned feature" reasons in `test_0800:58,111,162` (UnitAwareArray return;
`points_in_domain`/`get_closest_cells` with Pint coordinates) and
`test_0811:35,59,86,111,134,155` (evaluate with quantity-coordinate lists —
overlaps LE-08). `evaluate()` (`functions_unit_system.py:789-805`) has no
`coord_units` kwarg; no design doc commits to one. **Fix**: one decision for the
family — consolidate into a single clearly-labelled aspirational module if the
feature is on the roadmap, else delete all placeholders and record the proposal
in `docs/developer/design/`.

#### LE-12 — `persistence.py` verdict: delete

54-line module: re-export of `EnhancedMeshVariable`/`create_enhanced_mesh_variable`
(lines 40-43) + three "TODO: Implement" stubs (51-53); docstring admits the
2025-01-13 move and the misleading name. Only importer: `__init__.py:209`.
Deletion is behaviour-neutral (`EnhancedMeshVariable` is independently exported
via `discretisation/__init__.py:21` and `__init__.py:605`); the Symbol
Disambiguation note is already in
`docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md` (line 365). **Fix**:
delete module + import; update the CLAUDE.md Key Files entry; optionally leave a
one-release warn-on-import shim if external imports are a concern.

### Unverified findings (evidence lines read by the author; not adversarially checked)

| ID | Sev | Effort | Location | Category | Summary & disposition |
|----|-----|--------|----------|----------|-----------------------|
| LE-13 | medium | M | `pyx:4026` | dead-code | Stale pre-refactor remnants across the solver file: old setup flow (4026-4038), superseded clearDS/createDS block (4052-4066 — keep its 3-line "destroys field registrations" rationale), old f0/F1 constructions (2684-2688, 3624-3631), debug prints (2746-2747), commented `mesh.access` one-liners (3016, 3057, 4044), dead attribute assignments (3216-3227, 2334, 3280). Wave A: delete (comment-only change per pyx ground rule); tier_a green pre/post since a pyx touch forces a rebuild |
| LE-14 | low | S | `src/underworld3/swarm.py:1235` | todo-stale | Four identical `TODO: Add parallel sync logic here if needed` over `if sync: pass` (1235, 1285, 1342, 1391) — the `sync=` argument is a silent no-op; speculative API. Delete the blocks; drop the parameter or document that DMSwarm getField/restoreField handles sync. Feed to dimension 5 |
| LE-15 | low | S | `pyx:2766` | dead-code | Ten-line commented-out sketch of natural-BC flux Jacobians under the intent note at 2764 ("perhaps a different user-interface altogether is required for flux-like bcs"). Keep the one-line intent comment, delete the code block, record the idea in planning if still wanted |
| LE-16 | low | S | `src/underworld3/cython/petsc_discretisation.pyx:248` | dead-code | Content-free `## Todo !` introducing `petsc_dm_get_periodicity` preserved inside a module-level triple-quoted string (250-285) — can never run, no explanatory note. Wave A: delete; planning-file entry if DM periodicity reading is still wanted |
| LE-17 | low | S | `src/underworld3/swarms/pic_swarm.py:326` | dead-code | Commented-out `setLocalSizes` (with 6-line "existing values are wrong" explanation) + lost-particle debug remnant (~387-390). Delete both; condense to one line: do not pre-set local sizes — `insertPointUsingCellDM` sizes the swarm. Fold into the swarm cleanup wave |
| LE-18 | low | S | `tests/test_quantities_simplified.py:21` | skipped-test-obsolete | Module-level skip: imports `underworld3.function.quantities_simplified`, "which does not exist"; has never tested anything. Delete; first port any uncovered cases (multiplication-order, `.data` non-dimensionalisation) to tests against the real `quantities` module |
| LE-19 | low | S | `tests/test_0754_arithmetic_closure_complete.py:166` | skipped-test-obsolete | Skips citing "UnitAwareExpression class not implemented - feature replaced by simplified units architecture" (0754:166,288; 0756:164) test a class the authoritative redesign removed. Delete; re-express still-relevant closure properties against the simplified API (the neighbouring xfails at 0754:109,146 already do and should stay) |
| LE-20 | low | S | `tests/test_0750_meshing_follow_metric.py:269` | skipped-test-obsolete | Three `strict=False` xfails (269, 324, 344) mark capabilities of the pre-merge elliptic-ma MA mover that the development merge deliberately dropped/replaced. **Corrected during this audit** — the briefing's description (list-of-(metric,weight) composition; pointer to `fault_comb_metric`) does not match the file: actual reasons are (269) a #202 field-transfer skip-threshold nudge, a skip-optimisation only; (324) alignment capture now validated via the OT mover (test_0760); (344) boundary slip now via `smooth_mesh_interior(method='mmpde', slip_surfaces=...)` (test_0855). Disposition unchanged in spirit: delete (269 optionally re-tuned) or keep only with reasons as-is — they already point at the surviving coverage |
| LE-21 | low | M | `tests/test_1100_AdvDiffCartesian.py:98` | skipped-test-broken | xfail on the mesh0 param is a broken test, not a product bug: the file's own header says "not a great test" — step-function IC unrepresentable on the FE mesh, tolerance tuned to the legacy RBF-smoothed boundary path; the corrected in-cell/FE routing (more accurate) exceeds the stale atol. Rework with a representable IC + FE-derived tolerance, then drop the xfail; until then the annotation is correct |
| LE-22 | low | S | `src/underworld3/kdtree.py:8` | todo-unimplemented-feature | TODO(CLEANUP) accurate: pykdtree unused (module re-exports the ckdtree/nanoflann backend) yet remains a dependency whose OpenMP runtime causes fatal double-init crashes on macOS beside PETSc/numpy; tracked in planning (Active, 2026-02-13). Wave A: remove pykdtree from pixi.toml/setup deps, keep this module as the import point, delete the TODO |
| LE-23 | low | S | `src/underworld3/checkpoint/disk_snapshot.py:29` | stub-verdict | Verdict on the "empty stub groups": KEEP — they are the documented phased on-disk format (`write_snapshot_skeleton` at 161 stubs groups with `filled_by` attrs at 192; called by `write_snapshot` at 275/292; exercised by test_0010). Only defect: header still says "Phase 1 (this commit)" (line 29) while phases 2/3a/3b are implemented below (`filled_by` set at 356, 375, 423). Update the docstring only |
| LE-24 | low | S | `src/underworld3/meshing/smoothing.py:2033` | comment-block-verdict | Verdict on the 18 comment blocks ≥10 lines (589-4489, incl. the redistribution-regimes note at 2033 and the envelope-ansatz derivation at 4025): ALL are prose documenting physics/design intent — none is commented-out code. Same for ddt.py's large blocks except the LE-09 `if 0` region. Keep all (the campaign's documented-intent carve-out); optionally promote the two longest derivations to docs/ if dimension 4 flags file length |
| LE-25 | low | S | `src/underworld3/discretisation/discretisation_mesh.py:2584` | todo-legitimate-note | Residual marker triage — legitimate, keep: TODO(parallel) at 2584 and `functions_unit_system.py:721/757`; TODO(follow-up) at 2818 (`_pinned_mask` face-label limitation); "for now" heuristics in `units.py:430/1110/1602`, `swarm.py:1109-1148/2306`; feature/deprecation notes in `model.py:404/701/3788/3910`; rename notes (`discretisation_mesh_variables.py:1133/2809`, `discretisation_mesh.py:4409`); minor notes (`adaptivity.py:862`, `geometry_tools.py:5`, `segmented.py:218`, `__init__.py:656`, `_jitextension.py:924`, `solvers.py:3995`). Cosmetic-only: fix the `TODO: DECPRECATE` typo at `pyx:1450` and normalise marker spelling if a style-charter wave lands |
| LE-26 | low | S | `tests/test_1064_constrained_spherical_shell_response.py:227` | skipped-test-masking-bug | Two strict xfails correctly document that LU sub-solves in the constrained velocity\|[p,h] fieldsplit do not reproduce the validated Nitsche/default velocity response — a tracked open solver bug adjacent to the Stokes_Constrained 3D velocity discrepancy, not a test defect. Keep exactly as-is (`strict=True` surfaces a silent fix as XPASS); cross-reference the reason string so both are triaged together |

## Testing Instructions

How to validate the eventual fixes, per the campaign gates (tier_a green pre/post
each wave; `tier_a or tier_b` before merge; np2/np4 for swarm/migration touches).

**Bug fixes behind verified findings** (each unskip is itself the regression test):

- **LE-01**: new test — vector MeshVariable with known analytic field; assert
  `uw.function.evaluate(v[1].diff(x), coords, rbf=True)` matches the analytic
  ∂v₁/∂x and *differs* from ∂v₀/∂x. Must fail before the fix, pass after. Also
  confirm the L2 path (`rbf=False`) is byte-identical pre/post.
- **LE-02**: `CellWiseIntegral(mesh, fn=1.0)` on the unit square must sum to 1.0;
  remove the two xfails at `tests/test_0501_integrals.py:182,195` and confirm they
  pass. Compare against `Integral` for a non-trivial multi-field integrand.
- **LE-03**: run issue #215's reproducer (a Projection consuming a swarm proxy
  after a swarm write, *without* touching `.sym`); remove the manual
  `_update_proxy_if_stale()` workarounds at
  `tests/test_0112_swarm_add_particles.py:110,821` and confirm green. Gate on
  `tests/test_0006_memory_leak.py` to prove no eager-refresh memory regression.
  Run np2/np4 (swarm code).
- **LE-04/#291**: fix is out of this dimension's scope; on fix, remove the skip at
  `test_1017...mpi.py:129` and the serial-only restriction, then
  `mpirun -n 2 pytest tests/test_1062_constrained_solcx.py` must pass.
- **LE-06**: after implementing `UnitAwareDerivativeMatrix.__mul__`/`__neg__`,
  remove the three skips in `test_0813_mesh_variable_ordering_regression.py` — all
  three (including the created-BEFORE-solve control) must pass. Interim: the
  units-free rewrites must pass immediately.
- **LE-07**: after aligning swarm reductions to the tuple contract, unskip lines
  32/72 (adding `vtype=uw.VarType.TENSOR` at the line-72 test) and unskip 335
  as-is; the xfails at 163/294/313 stay until global `std()` lands.
- **LE-08**: unskip the three tests in `test_0812_poisson_with_units.py`; also run
  the adjacent xfails at `test_0750_unit_aware_interface_contract.py:116,195,249`
  looking for XPASS.

**Wave A deletions** (LE-09, LE-12 … LE-20 deletions, LE-13, LE-22): behaviour-
neutral by construction — full `pytest -m "tier_a"` pre/post must be identical;
`pytest -m "tier_a or tier_b"` before merge. For LE-13 (pyx comments) a rebuild is
forced: `./uw build` then tier_a. For LE-09 add one test asserting
`preserve_moments=True` raises NotImplementedError. For LE-12 confirm
`import underworld3` and `from underworld3.discretisation import MeshVariable`
still work and grep shows no remaining `persistence` importer. For LE-22, a full
test run on macOS after removing pykdtree from the dependency set (the crash it
guards against is load-order dependent).

**Test-suite hygiene** (LE-10/11/18/19/20): deletions/rewrites — run the affected
files plus `pytest -m "tier_a or tier_b"`; for rewritten `test_0620`, assert the
implemented semantics (units accepted, model precedence, `UserWarning` on
conflict — `discretisation_mesh.py:282-301`).

**Stale-skip removal from the refuted appendix**: un-skip
`tests/parallel/ptest_0762_read_timestep_swarm_routed.py:75` and run at np≥2; the
#151 fix is already covered by
`tests/parallel/test_0790_swarm_write_timestep_mpi.py`.

## Known Limitations

- **LE-13 … LE-26 are not adversarially verified.** Every anchor line was read
  directly by the author in this worktree, but the surrounding claims (e.g. "no
  other caller", history assertions) were not independently re-derived for these
  fourteen. One briefing claim in this group was materially wrong and is corrected
  in place (LE-20).
- Line numbers are exact at `development` @ `1d003481` only; any wave that lands
  earlier shifts them.
- Live reproductions (LE-02, LE-06, LE-07, LE-08) ran in this worktree's own built
  environment (site-packages verified against src); they were not repeated on a
  second platform or in parallel.
- Skip/xfail counts for the units placeholder family (LE-11) were grep-verified at
  this ref but the family boundary is a judgement call — a few "planned feature"
  markers elsewhere may belong to it.
- Overlaps deliberately deferred: swarm API inconsistencies (LE-03, LE-07, LE-14,
  LE-17) feed dimension 5; file-length/readability observations (LE-24) feed
  dimension 4; the BC/API-surface issues are dimension 3's territory.
- GitHub issue states (#151 closed, #215 closed, #291 open) were checked 2026-07-03
  and may move.

## Refuted claims (do not re-find)

| Claimed finding | Why it is wrong |
|-----------------|-----------------|
| `tests/parallel/ptest_0762_read_timestep_swarm_routed.py:75` — "skip masks an untracked parallel `Swarm.write_timestep` hang" | The hang is GitHub issue #151, **closed 2026-04-29**, fixed by commit `453e5063` (an ancestor of `1d003481`): both `Swarm.save` (`swarm.py:3768-3804`) and `SwarmVariable.save` (`swarm.py:1861-1892`) carry `BUGFIX(#151)` allgather/global-shape/per-rank-slab fixes for exactly the hypothesised collective-create root cause. Dedicated np≥2 regression coverage exists (`tests/parallel/test_0790_swarm_write_timestep_mpi.py`). The only real defect is that the **skip itself is stale** (authored on a branch parallel to the fix, never re-validated) — remove it and re-run at np≥2; do NOT open a new issue. |
| `tests/test_0502_boundary_integrals.py:141` — "six MPI skips are the same partition-seam assembly family as `pyx:2545`; add an np=2 characterisation test of the known wrong value" | Wrong diagnosis. The underlying bug is an **UnboundLocalError in the `BoxInternalBoundary` constructor**: `boundaries`/`boundary_normals` are bound only inside the `if uw.mpi.rank == 0:` gmsh block (`meshing/cartesian.py:536`, bindings ~547-548 / ~644-648) while `Mesh(...)` at ~896-901 runs on all ranks — rank>0 raises before any mesh exists, exactly as the test file's own comment (115-117) says. The `pyx:2545` bug concerns natural-BC *residual* scatter and its comment notes the pure integral is machine-precision identical in parallel. The skips also do not remove all parallel internal-boundary coverage (Annulus/spherical internal tests at 240-293, 325 carry no MPI skip; test_0502 is not in the parallel CI suite — `scripts/test_levels.sh:179,190`, `scripts/test.sh:99`). A "known wrong value" characterisation test is impossible: at np=2 the constructor raises. The real fix is a one-liner mesh-construction bug (broadcast/bind the Enum on all ranks), unrelated to `pyx:2545`. |

## Sign-Off
| Role | Name | Date | Status |
|------|------|------|--------|
| Maintainer | Louis Moresi | 2026-07-05 | Pending review |
| Author | Claude (audit session) | 2026-07-03 | Complete |



- **Author**: loose-ends audit (dimension 1), 2026-07-03, on the audit worktree
  at `development` @ `1d003481`.
- **Verification**: LE-01 … LE-12 adversarially verified (independent re-read +
  live reproduction where noted); LE-13 … LE-26 author-read anchors only; two
  claims refuted and recorded above.
- **Maintainer sign-off**: pending (L. Moresi) — required before Wave A deletions
  execute and before LE-07's return-type behaviour change is scheduled.