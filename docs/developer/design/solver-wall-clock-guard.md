# Bounding a solver grind: the wall-clock guard and the sub-solve work gauge

**Status**: implemented. `solver.guard(wall_per_step=...)`, `solver.unguard()`,
`solve_report.sub`, `solve_report.deadline_expired`.
Implementation: `src/underworld3/systems/solver_health.py`.

## The problem

A viscoplastic Stokes solve near its conditioning floor does not fail. It *grinds* — for
hours, inside a single Newton step, with no error and no sign of stopping. Any study that
sweeps a parameter plane hits this immediately: the interesting cells are exactly the
unreachable ones, and an unreachable cell costs unbounded time.

The obvious defences do not work, and both were measured before this was written.

**An iteration cap does not bound wall time.** Capping nonlinear iterations, outer Krylov
iterations, or inner iterations all leave the same hole: below the conditioning floor a
single Newton step grinds *within one outer Krylov iteration*. The counter the cap watches
never advances, so the cap never fires. Measured: 3814 s inside one outer iteration.

Nor can an iteration cap be tuned around the problem. Tight caps terminate quickly but
classify wrongly in *both* directions — a continuation march settles at its starting
parameter and reports success (a false rescue), while a step that would have converged in
fifteen iterations is called unreachable at ten (a false failure). Loose caps classify
correctly and run past ten minutes a station. The number of iterations a feasible step
needs is not knowable in advance, which is precisely why a *time* budget is the right
independent guard.

**A Python timer does not fire.** `signal.alarm` schedules a handler, but Python runs
signal handlers only between bytecodes, and during the grind control is inside PETSc's C
code the entire time. Measured: a 90 s alarm still unfired at 10 minutes.

So the only code that runs during a grind is PETSc's own. The guard has to live there.

## The construction

Three pieces, each chosen for a reason that was measured rather than assumed.

**Where the clock starts.** The outer KSP runs exactly once per Newton step, so its
iteration 0 is the natural place to restart the budget. No SNES hook is needed — the
original design proposed resetting the deadline from a custom SNES convergence test, but
that would mean reimplementing `SNESConvergedDefault` (petsc4py exposes no way to chain to
it) for no gain.

**Where the deadline is checked.** In convergence tests on the outer KSP *and on every
fieldsplit sub-KSP*. The sub-KSPs are where the granularity is: in a representative solve
the outer KSP performed 1 iteration while the velocity block performed 285, so only the
sub-KSP tests can interrupt a grind at anything finer than "the whole Newton step".

**Which diverged reason is returned.** This is the part that is easy to get wrong, and the
earlier prototype did. PETSc's `KSPCheckSolve` deliberately *exempts* `KSP_DIVERGED_ITS`
from marking the preconditioner as failed — truncating an inner solve at its iteration cap
is normal behaviour, not an error. A guard that returns `DIVERGED_MAX_IT` from a sub-KSP
therefore does nothing at all: measured, the outer solve absorbed 36 truncated velocity
solves and reported CONVERGED. Returning `DIVERGED_BREAKDOWN` marks the sub-preconditioner
failed, which surfaces as `DIVERGED_PC_FAILED` at the outer KSP and `DIVERGED_LINEAR_SOLVE`
at the SNES.

Because a custom convergence test *replaces* the default rather than composing with it,
the guard also carries the ordinary rtol / atol / divtol semantics itself. Tolerances are
re-read every iteration, because Eisenstat–Walker rewrites the outer rtol before each
Newton step.

### Parallel

Wall clocks are not synchronised across ranks, so "has the budget run out" is exactly the
kind of question that splits a communicator — and a PETSc convergence test that returns
different verdicts on different ranks leaves the ranks on different code paths, so the
next collective deadlocks. The expiry decision is therefore reduced over the solver's
communicator before it is acted on. This is well posed because Krylov iterations are
globally synchronised, so every rank reaches the same check the same number of times.

The reduction is one integer per iteration, against a multigrid cycle, and every iteration
already pays for at least one norm reduction — so it does not change the communication
character of the solve.

`tests/parallel/ptest_0203_wallclock_guard_parallel.py` provokes the hazard deliberately
with deliberately skewed per-rank clocks. With the reduction it passes at np=2 and np=4;
with the reduction removed it deadlocks (confirmed at a 90 s limit).

The extra reductions are paid only on the guarded path — an unarmed solver installs no
convergence tests at all, so the default solve is untouched.

### What the guard does not cover

**Rotated free-slip.** That path runs its own Krylov loop outside `self.snes`
(`utilities/rotated_bc.py`), which the deadline cannot reach. `guard()` raises there
rather than pretending, matching `estimate_difficulty()`.

**A `preonly` outer KSP.** With no outer iterations there is no convergence test to start
the clock, so the guard is inert. A direct solve is not the thing this guard exists to
bound, but it is worth knowing that it is silent rather than protective.

**The first preconditioner application.** On a solver's **first** solve the fieldsplit
blocks do not exist when the solve begins:
they are created by `PCSetUp`, which runs inside the first `KSPSolve`. The guard attaches
to them at the earliest reachable moment — the outer KSP's iteration 0 — which is *after*
the preconditioner has been applied once. So one preconditioner application on the first
solve is outside the deadline. Every solve after that is fully covered, because the blocks
persist.

A driver that cares should do one cheap solve before arming the guard, which it usually
does anyway to pay for the JIT compile.

## The work gauge

`solve_report.ksp_its` counts outer Krylov iterations. Under Eisenstat–Walker that is about
one per Newton step, so it is not a measure of cost — using it as a work axis understates a
Stokes solve by two orders of magnitude. Measured on a small box: outer 1, velocity block
285, pressure block 9.

`solve_report.sub` records iterations and applications per fieldsplit block. For the
velocity block under `pc_type=mg` that iteration count *is* the multigrid cycle count. It
is collected by monitors, which observe and never decide, so it cannot change a solve.

The same first-solve limitation applies, and the report says so: `SubSolveReport.complete`
is `False` when the block was instrumented part-way through the solve, meaning `its` is a
lower bound. Measured, the first solve reports about half the true count. Every later solve
is exact.

Cost of leaving the gauge always on: 624 monitor callbacks in a 280 ms solve, no measurable
change in wall time (medians 282 ms instrumented against 284 ms not, run alternately;
run-to-run noise is larger than the difference).

## Usage

```python
stokes.guard(wall_per_step=150.0)
stokes.solve()
if stokes.solve_report.deadline_expired:
    ...                    # too hard at these parameters — back off a continuation step
stokes.unguard()

cycles = stokes.solve_report.sub["velocity"].its       # the honest work axis
```

A guard exit is a report, not an exception: the current iterate is left in the fields, so a
continuation driver can revert or retry from it.

## Related

- `docs/developer/design/nonlinear-solver-homotopy-warmstart.md` — the δ-continuation this
  guard exists to make sweepable.
- `docs/developer/design/solver-strategies-catalogue.md` — the solver configurations whose
  cost this measures.
