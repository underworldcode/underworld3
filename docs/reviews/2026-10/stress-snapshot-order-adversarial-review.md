# Adversarial review: the stress history's snapshot order

**Branch**: `bugfix/backward-nodes-snapshot-order` (base `development` 262a2968).
**Commits reviewed**: d77bc9d2 (semi-Lagrangian refresh after the trace-back),
06bd7139 (test 1076 renumbered 1102), 0732e839 (Eulerian flavours: the lapsed
snapshot request, refresh after the carry), and the review-fix commit that
follows them.
**Reviewers**: two independent read-only passes (correctness, parallel and
determinism; docs, Charter, CI reach and contracts), 2026-10-22, plus our own
probes on hyperion. One round.

## What changed

The commit projection of a viscoelastic stress history reads `psi_star[0]`
through a frozen copy, the source snapshot. The semi-Lagrangian flavour
refreshed that copy before the trace-back, so the stored stress was rebuilt
from the field that had not moved (d77bc9d2). The Eulerian flavours refreshed
in the same wrong order, but Stokes asked for the snapshot before their
projection existed and the request lapsed, so their commit read `psi_star[0]`
live and their results were not affected. `Eulerian.enable_source_snapshot`
now builds the projection first; `update_pre_solve` carries the history
through one hook, `_carry_history` (split correction, or the SUPG transport),
then refreshes.

## Evidence

- test_1102, written first, unfixed code: eulerian fails `source reads
  psi_star[0]`; split test fails `enable_source_snapshot did nothing`.
- Request fixed, old order: `test_the_stored_stress_rotates_with_the_flow[eulerian]`
  fails `unmoved > 1.0` at 5.38e-05; the snapshot identity fails at 0.0247
  (eulerian) and 0.125 (split).
- Both fixed: test_1102 6 passed serial, 6 passed at np 2.
- Snapshot minus `psi_star[0]` at every commit, four steps of a rotating flow:
  0.0 for semi_lagrangian and for eulerian (probe `probe_snapshot_at_commit.py`).
- test_1102 table re-measured, unchanged: 0.285/1.45, 0.291/1.55, 0.086/1.55.
- test_1052 + test_1059: 46 passed; no pin moved.
- Wide serial set (transcript, describe, adjoint, restore, capability guides,
  SUPG plugin, rotating transport, test_1102): 258 passed, 1 xpassed
  (test_1100 boxmesh, a known `strict=False` xfail).
- `check_test_coverage.py`: 355 files reachable. `check_deprecated_patterns.py`: clean.

## Findings and what we did

- **Docstring history was wrong** (`ddt.py`, `enable_source_snapshot`). We had
  written that the snapshot was introduced because the projection was implicit
  in its own target. `git show 5b3548c8^` shows the projection already solved
  into the flat variable. Fixed: the docstring states how it works and the one
  order constraint; no history.
- **Unreachable commit citations**: 8f2b0dd, 31abad1, 380ab4b are rebased
  duplicates; the reachable fix is 5b3548c8. Fixed in test_1052.
- **"Until 2026-10-18"** dated an unmerged fix. Fixed: names d77bc9d2 on this
  branch, and `development` once it merges.
- **Defect-era cylinder results stated as fact** in the stress-transport table
  and conformation paragraph. Fixed: both point to the warning.
- **test_1052**: two docstrings still claimed the uniform shear box catches the
  refresh order or depends on the snapshot. Fixed.
- **Loose bound**: `rotated < 0.4` against eulerian 0.086. Fixed: per-history
  bounds 0.35 / 0.35 / 0.11.
- **Source check not unwrapped** (Charter: unwrap before atoms); the frozen test
  ran `store_result=True`, which Stokes does not. Fixed both.
- **Split test passed on the buggy column** (`_apply_split_advection` corrects
  yy with the xy entry for a tensor). Fixed: the test loads and checks xx only.
- **Stale snapshot on a `store_result=True` record** (semi-Lagrangian). The
  projection fallback reads the snapshot, which now holds the previous step's
  carried history. Not reached: a probe shows the record takes the `evaluate`
  route, which reads `psi_star[0]` live (error 3.3e-3 against the commit, 3.0
  against the stale value), and no caller enables the snapshot with
  `store_result=True`. Marked TODO(BUG) at the fallback, not fixed.
- **`.subs` misses `psi_star[0]` inside a UWexpression** (at yield,
  `E_eff_inv_II`). Pre-existing; no effect while snapshot = `psi_star[0]` at the
  commit. Marked TODO(BUG), not fixed.
- **`_apply_split_advection` wrong for tensors.** Pre-existing. Marked TODO(BUG).
- `docs/docstrings/inventory.json` still holds the old docstring; generated.

## Attacks that failed

- `_setup_projections` is idempotent; nothing outside `ddt.py` reads the
  projection, the flat variable or the snapshot.
- `psi_fn` is the zeros placeholder when the projection is built early; the
  setter rebuilds the substituted source when Stokes rewires it.
- Removing `EulerianSUPG.update_pre_solve` changes nothing: dt was
  non-dimensionalised twice (idempotent), `_advection_mode` is "assembled".
- Parallel: the carry condition is rank-independent; MeshVariable creation is
  unconditional and collective; the refresh copies local data after sync. np 2
  passes.
- Variable names come from the global instance counter: unique and identical
  on every rank. Checkpoint names (`psi_star`) unchanged.
- Integration-point, forward, Lagrangian and symbolic histories never set the
  multi-component flag: no snapshot, no order defect.
- The 1076 to 1102 rename leaves no stale reference; 1102 is free in
  `tests/parallel`; `test_110*py` runs it.
- No capability guide is newly stale beyond `transport-schemes.md`, updated.

## Issues adjudicated

| issue | verdict | evidence |
|---|---|---|
| #735 nodal stress ~50% high at a no-slip wall | **Still live** | measured while the nodal history did not carry its stored stress; d77bc9d2 changes the nodal flavour in exactly this regime. Re-measure the cylinder table (a campaign, not a hyperion job) before ruling. |
| #742 `psi_star[0]` holds different levels on Stokes and NavierStokesSLCN | **Still live** | this branch does not touch the NavierStokesSLCN record convention. |
| #749 Eulerian stress history blows up at a shear-wave reflection | **Still live** | the eulerian commit reads the same values before and after (snapshot = live `psi_star[0]`, 0.0); its numerics are unchanged. |

## Merge conditions

- The review-fix commit is on the branch, and test_1102 passes serial and at
  np 2 after it.
- The PR body states #735, #742, #749 stay open, and why.
