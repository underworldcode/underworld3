# Adversarial review — wall-clock guard and sub-solve work gauge

Branch `feature/solver-wallclock-guard` vs `development`.

Method: two independent adversarial reviewers, each given one dimension and instructed to
break the change rather than praise it. Both were told to report only findings they had
verified against the source; both went further and verified empirically, one by driving
standalone petsc4py to establish PETSc's callback ordering, the other by neutering parts
of the implementation and observing which tests still passed.

**Verdict: not ready as submitted.** Four major defects in the new code, two false claims
in its documentation, and a test suite that a reviewer showed would pass with several
parts of the feature broken. All are addressed on the branch; the two findings that
measurement did not sustain are recorded as such rather than quietly fixed.

---

## MAJOR — the deadline was disabled for the first preconditioner application of *every* solve

`begin_solve()` left the clock at infinity, and only outer-KSP iteration 0 armed it. But
PETSc's default for GMRES — the Stokes outer solver — is **left** preconditioning, and
`KSPInitialResidual` applies the preconditioner *before* iteration 0. The reviewer
established the ordering directly:

```
outer=gmres  pc_side=LEFT :  sub0_mon(0), sub0_mon(1), sub1_mon(0), sub1_mon(1), outer_mon(0), ...
outer=fgmres pc_side=RIGHT:  outer_mon(0), sub0_mon(0), ...
```

So one complete velocity-block solve plus one pressure-block solve ran entirely outside
the budget — on every solve, not just the first. That is precisely the grind the guard
exists to bound. Both the docstring and the design note described this as a first-solve
limitation caused by the blocks not yet existing; the blocks exist from solve 2 onward,
and the clock was simply not running.

**Fixed**: the clock starts in `begin_solve()`, and outer iteration 0 restarts it per
Newton step.

## MAJOR — the premise that forced a hand-rolled convergence test was false

Both the code and the design note asserted that "a custom convergence test replaces the
default — petsc4py exposes no way to chain to it". petsc4py ships
`KSP.addConvergenceTest(fn, prepend=True)`, which composes the custom test with the
*native* one, whatever it is configured to be.

Verified here before acting on it:

| | iterations | reason |
|---|---|---|
| uninstrumented | 30 | CONVERGED_RTOL |
| prepended test returning `ITERATING` | 30 | CONVERGED_RTOL |
| prepended test returning `DIVERGED_BREAKDOWN` at it 3 | 3 | DIVERGED_BREAKDOWN |

Every divergence the reviewer then listed was a self-inflicted consequence of
reimplementing rather than prepending: four dropped `KSPConvergedDefault` behaviours
(`-ksp_converged_maxits`, the non-zero-initial-guess residual convention, the norm-type
early return, `-ksp_min_it`), and a NaN guard written as `rnorm != rnorm` that let an
infinite residual through to be reported as `CONVERGED_RTOL` — silent false convergence,
in exactly the inf/inf viscoplastic regime the guard targets.

**Fixed**: `_default_convergence` is deleted. The guard returns `ITERATING` and PETSc
decides. This also makes `unguard()` exact rather than approximate — PETSc offers no way
to remove an added test, so it stays installed and goes inert, which *is* the
uninstrumented behaviour, instead of being swapped for a freshly created default context
that would have lost any options-configured flags.

## MAJOR — the instrumentation re-opened the leak `BUGFIX(#157)` was written to close

`SNES.getKSP()` and `PC.getFieldSplitSubKSP()` both increment the reference count. The
instrumentation held those entries until the *next* solve, so a rebuild
(`is_setup=False`, remesh, adapt) did not free the old KSP, PC, coarse operators or
multigrid hierarchy: they survived precisely while the new hierarchy was being built.

The comment at the destroy site records that this exact retention "at Gadi scale … push[ed]
past memory limits" for `SNES_Tensor_Projection.solve()` cycling six components in 3D —
a loop that drives `is_setup=False` on every component.

**Fixed**: the instrumentation releases its references before the SNES is destroyed.

## MAJOR — `guard()`'s rotated-free-slip refusal was bypassed by call order

Found independently by both reviewers, and demonstrated live: `guard(...)` then
`add_rotated_freeslip_bc(...)` armed without error, and the subsequent solve took the
rotated path — which never reaches the instrumentation — and reported
`deadline_expired=False, converged=True`. The guard was silently inert: "looks like
protection and is not", which is the exact state the `NotImplementedError` exists to
prevent.

**Fixed**: re-checked at solve time as well as at arm time; the test now covers both
orders.

## MINOR — `preonly` and `richardson` outer KSPs are changed by being monitored

Both are gated on `ksp->numbermonitors` in PETSc: `KSPSolve_PREONLY` adds a norm, a
matvec and a second norm purely to have something to report, and `KSPRICHARDSON` gives up
the fused `PCApplyRichardson` path. The gauge attached a monitor to the outer KSP of every
solver unconditionally, and `model.py` ships `preonly` presets.

**Fixed**: both types are skipped. Neither iterates, so there was nothing to count or
bound — it was all cost and no information.

---

## The test suite was the sharpest hit

The second reviewer neutered parts of the implementation and recorded which tests still
passed. Four separate holes:

- **`unguard()` was untested.** Making the removal a no-op left 7/7 passing. The test
  passed only because disarming *also* reset the deadline, so the solve converged — it
  never observed which convergence test was installed.
- **No work parity assertion.** Tightening the guard's `rtol` by 1e-4 left 7/7 passing.
  The healthy-solve test compared field values only, so any change confined to the
  sub-blocks — which cannot move the answer, only the cost — was invisible. Against
  "Solver Stability is Paramount" that is the wrong thing to be blind to.
- **`wall_per_step`'s per-Newton-step meaning was never exercised.** The fixture was
  constant-viscosity, hence linear, hence `nl_its == 1` on every solve in the file — so
  per-step and per-solve budgets were indistinguishable. That is the entire meaning of
  the only parameter the feature takes.
- **The parallel test was dead code.** Not listed in `mpi_runner.sh`, referenced by no CI
  workflow, and not pytest-collectable (`ptest_*` does not match `test_*`). The serial
  suite did not cover the reduction either — replacing the `Allreduce` with a rank-local
  read left 7/7 passing. So the hazard the design note calls "specific and severe" had
  zero automated coverage.

Also: the rotated-BC test corrupted the module-scoped fixture (`add_rotated_freeslip_bc`
sets `is_setup=False`), so reordering the file made the *gauge* test fail — a failure
pointing at the wrong culprit. And tests left the solver armed at a microsecond budget if
they failed part-way, cascading into everything after.

**Fixed**: cost parity is asserted alongside answer parity; the per-step semantics is
exercised on a nonlinear model by measuring the solve's total cost and then setting a
budget only a per-*solve* interpretation would blow; the first-solve lower-bound contract
and the arm-before-first-solve path are covered; the rotated tests use their own solvers;
a fixture guarantees disarming. The parallel test is registered in `mpi_runner.sh`.

The suite is marked **tier_b, not tier_a**. Charter §8 reserves tier A for hardened tests;
these are new and have never run in CI.

---

## Findings that measurement did NOT sustain

Recorded rather than silently fixed, because the reasoning was sound and only the
conclusion was wrong.

**"A retry or continuation stage is handed a fresh budget after an expiry."** Mechanically
true — `deadline_expired` latches while `open_step()` could re-arm — and a real concern,
since a deadline exit is a negative reason and therefore *triggers* a warm-start retry.
But PETSc prevents it independently: once the sub-preconditioner is marked failed, the
next `snes.solve` in the same call bails before reaching an iteration. Measured with the
expiry latch removed, on both the retry and the continuation path: 34 clock ticks against
32 with it, and `converged=False` either way. The latch is kept — relying on PC-failure
propagation is implicit rather than guaranteed — but it is belt-and-braces, and no test
claims to prove otherwise.

**"PETSc handles in `_ksps` could be reused after an object is freed, aliasing a new
KSP."** Checked and clean: the retained reference keeps the address occupied, so no freed
handle can alias. The same fact is what caused the memory-retention finding above — the
aliasing safety was being paid for in memory.

---

## Verified clean

- `Allreduce(MPI.IN_PLACE, flag, MPI.MAX)` on the outer KSP's communicator is correct, and
  the fieldsplit blocks live on that same communicator.
- `DIVERGED_BREAKDOWN` returned at outer iteration 0 is safe (`KSPGMRESBuildSoln` handles
  `it-1 == -1` explicitly).
- The rotated path builds its own report, so `sub` / `deadline_expired` default rather
  than carrying stale values.
- `ksp.getTolerances()` returns `(rtol, atol, divtol, max_it)` — a different order from
  SNES — and was read correctly.
- The fake clock intercepts every clock read the guard makes: `time.monotonic` appears
  exactly twice in the diff, both via the module global.
- A non-fieldsplit solver (Poisson) reports `sub == {}`, and the guard still fires on its
  outer KSP.
- The `DIVERGED_BREAKDOWN`-not-`DIVERGED_MAX_IT` claim is load-bearing *and* caught by the
  suite: swapping the reason back fails the "bites inside the blocks" test.
- The parallel claim is true: the ptest passes at np=2 and hangs past a 120 s limit when
  the reduction is replaced by a rank-local read.

## Known limitations, documented rather than hidden

- The injected-clock tests can catch a guard that never fires, not one that fires too
  eagerly — with a fake clock, PETSc's real work costs zero ticks, so expiry is guaranteed
  for any budget. The real-clock generous-budget test covers the other direction.
- `SubSolveReport.complete` is set conservatively: `False` whenever the block was attached
  mid-solve, even in a right-preconditioned configuration where the count would in fact be
  exact.
- `solve_report.sub` keys come from the split names, so the block-constrained Stokes path
  yields `"0"` / `"1"` rather than `"velocity"` / `"pressure"`. Documented; read the keys.
