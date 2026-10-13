r"""Sub-solve work gauges and the wall-clock guard for SNES-based solvers.

Two facilities, deliberately separate, both attached to the solver's own PETSc objects.

**The gauge** (always on, observation only). A nonlinear Stokes solve reports about one
*outer* KSP iteration per Newton step, because Eisenstat--Walker loosens the outer
tolerance to whatever the current Newton step deserves. The work is really done inside
the ``fieldsplit_velocity_`` sub-KSP, one multigrid cycle per iteration, and inside the
``fieldsplit_pressure_`` sub-KSP, one *velocity solve* per iteration. So
``solve_report.ksp_its`` is not a measure of cost. ``solve_report.sub`` is: iterations
and applications per fieldsplit block, which is what "how expensive was this solve"
actually means.

**The guard** (opt-in, changes termination). ``solver.guard(wall_per_step=...)`` bounds
the wall-clock time of each Newton step. Nothing outside PETSc can do this:

* an iteration cap does not bound wall time, because below the conditioning floor a
  *single* Newton step grinds inside one outer KSP iteration -- the outer count never
  advances, so no cap on it can fire;
* a Python ``signal.alarm`` does not fire either, because Python runs signal handlers
  only between bytecodes and control is inside PETSc's C code the whole time. Measured:
  a 90 s alarm sat unfired at 10 minutes.

The only code that runs during the grind is PETSc's own convergence test, so that is
where the deadline lives. It is reset at the start of every outer KSP solve -- which is
once per Newton step -- and checked inside the sub-KSP tests, where iterations are
frequent enough to give seconds of granularity.

**Which diverged reason** matters, and is not obvious. PETSc's ``KSPCheckSolve``
deliberately *exempts* ``KSP_DIVERGED_ITS`` from marking the preconditioner as failed --
truncating an inner solve at its iteration cap is normal, not an error. So a guard that
returns ``DIVERGED_MAX_IT`` from a sub-KSP silently does nothing: measured, the outer
solve absorbed 36 truncated velocity solves and still reported CONVERGED. Returning
``DIVERGED_BREAKDOWN`` marks the sub-preconditioner failed, which surfaces at the outer
KSP as ``DIVERGED_PC_FAILED`` and at the SNES as ``DIVERGED_LINEAR_SOLVE``.

**The guard does not replace PETSc's convergence test, it runs in front of it.**
``KSP.addConvergenceTest(..., prepend=True)`` composes with whatever native test the KSP
is configured with, so returning ``ITERATING`` hands the decision straight back to
``KSPConvergedDefault`` -- with its options-configured context intact
(``-ksp_converged_maxits``, the non-zero-initial-guess residual convention, the
norm-type early return, and the rest). Verified: a prepended test that always returns
``ITERATING`` gives bit-identical iteration count and reason to an uninstrumented solve.
A disarmed guard is therefore genuinely inert rather than approximately inert, which a
hand-rolled reimplementation of the default test could never be.

See ``docs/developer/design/solver-wall-clock-guard.md``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Marking the sub-preconditioner failed is the whole point -- see the module docstring.
_DEADLINE_REASON = PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN
#: A solve that stopped because it ran out of iterations, not because it
#: converged. PETSc's KSPCheckSolve deliberately does NOT treat this as a
#: failure for a sub-KSP, which is exactly why it goes unnoticed.
_CAPPED_REASON = PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT

# KSP types whose behaviour changes when a monitor is attached (both are gated on
# ksp->numbermonitors in PETSc): preonly.c computes norms it would otherwise skip, and
# rich.c abandons the fused PCApplyRichardson path. Neither iterates, so there is
# nothing here for the gauge to count or the guard to bound.
_MONITOR_SENSITIVE_KSPS = ("preonly", "richardson")


@dataclass(frozen=True)
class SubSolveReport:
    """Work done by one fieldsplit sub-solve over a single outer solve.

    Attributes
    ----------
    name
        Block name taken from the sub-KSP's options prefix -- ``"velocity"`` and
        ``"pressure"`` for the Stokes saddle-point solver. A solver that builds its
        splits by DM field index instead (the block-constrained Stokes path) names them
        ``"0"``, ``"1"``, ...; read the keys rather than assuming them.
    its
        Iterations summed over every application of this sub-solve. For the velocity
        block under ``pc_type=mg`` this is the multigrid cycle count, i.e. the honest
        work axis.
    applications
        Number of times the sub-solve ran during the outer solve.
    capped
        How many of those applications ended at the block's iteration cap
        (``KSP_DIVERGED_MAX_IT``) rather than at its tolerance. **Non-zero means
        the block did not solve.** It matters far more than it looks: the Schur
        operator ``S = -B A^-1 B^T`` is applied *through* the velocity solve, so
        a capped velocity block hands the pressure Krylov an operator that moves
        between applications, and no Krylov method converges against that. Every
        expensive pathology measured in #625 was a cap — the pressure block, the
        velocity block beneath it, and the velocity block again under
        augmentation. Measured: raising only ``fieldsplit_velocity_ksp_max_it``
        from 200 to 5000 took one solve from 976 s to 25.6 s with an identical
        answer.
    complete
        ``False`` when the block was instrumented *during* this solve rather than
        before it, which happens on a solver's very first solve: the fieldsplit blocks
        do not exist until ``PCSetUp`` has run, and that runs inside the first
        ``KSPSolve``. Everything the preconditioner did before then is not in the
        count, so treat ``its`` as a lower bound. Every later solve is exact -- so a
        driver that wants exact work should do one cheap solve first, which it usually
        does anyway to pay for the JIT compile.
    """

    name: str
    its: int
    applications: int
    complete: bool = True
    capped: int = 0

    def __str__(self) -> str:
        return (f"{self.name}: {self.its} its / {self.applications} applications"
                + (f" — {self.capped} AT THE ITERATION CAP" if self.capped else "")
                + ("" if self.complete else " (lower bound)"))


def _block_name(ksp):
    """Readable block name from a sub-KSP options prefix.

    Prefixes look like ``Solver_6_fieldsplit_velocity_``; the part that identifies the
    block is the tail after the last ``fieldsplit_``.
    """
    prefix = ksp.getOptionsPrefix() or ""
    tail = prefix.rsplit("fieldsplit_", 1)[-1]
    return tail.strip("_") or prefix.strip("_") or "sub"


class _InstrumentedKSP:
    """One KSP the instrumentation is attached to, and everything it needs.

    Holds the iteration counters (fed by a monitor, which cannot affect the solve) and
    the deadline-checking convergence test, which runs *in front of* the KSP's native
    test rather than replacing it.
    """

    def __init__(self, ksp, name, instrumentation, is_outer, mid_solve):
        self.ksp = ksp
        self.name = name
        self.its = 0
        self.applications = 0
        self.capped = 0
        self.complete = not mid_solve
        self._instrumentation = instrumentation
        self.is_outer = is_outer
        self._test_installed = False
        ksp.setMonitor(self._monitor)
        # The REASON, not an iteration count compared against the cap: a solve
        # that happens to converge on its last permitted iteration is converged,
        # and counting would call it capped. A post-solve hook is the only place
        # the per-application reason can be read before the next solve resets it.
        ksp.setPostSolve(self._after_application)

    def reset(self):
        self.its = 0
        self.applications = 0
        self.capped = 0
        self.complete = True          # attached before this solve began

    def report(self):
        return SubSolveReport(name=self.name, its=self.its,
                              applications=self.applications,
                              complete=self.complete, capped=self.capped)

    def _after_application(self, ksp, rhs, x):
        """Record whether this application ran out of iterations."""
        if ksp.getConvergedReason() == _CAPPED_REASON:
            self.capped += 1

    def _monitor(self, ksp, its, rnorm):
        # PETSc calls the monitor once per iteration including iteration 0, which
        # reports the initial residual rather than work done -- so a call at 0 opens a
        # new application and every later call is one iteration.
        if its == 0:
            self.applications += 1
            if self.is_outer:
                # The outer KSP runs once per Newton step, so its iteration 0 is the
                # natural place to start that step's clock -- no SNES hook needed. The
                # sub-KSPs do not exist until PCSetUp has run, and PCSetUp runs inside
                # KSPSolve, so this is the earliest they can be reached.
                self._instrumentation.open_step()
                self._instrumentation.attach_sub_ksps(ksp, mid_solve=True)
        else:
            self.its += 1

    def _test(self, ksp, its, rnorm):
        if self.is_outer and its == 0:
            # PETSc does not fix the order of the monitor and the convergence test at
            # iteration 0, so whichever runs first opens the step. The second call
            # re-reads the clock and pushes the deadline out by the gap between them --
            # microseconds against a multigrid cycle, and never in the unsafe direction.
            self._instrumentation.open_step()
            self._instrumentation.attach_sub_ksps(ksp, mid_solve=True)
        if self._instrumentation.deadline_passed():
            self._instrumentation.deadline_expired = True
            return _DEADLINE_REASON
        # Hand the decision back to the KSP's own test, whatever it is configured to be.
        return PETSc.KSP.ConvergedReason.ITERATING

    def install_test(self):
        """Prepend the deadline test, once and for all.

        PETSc allows exactly one ``addConvergenceTest`` per KSP and offers no way to
        take it off again, so the test is installed on first arming and left in place;
        ``disarm()`` makes it inert instead of removing it. That is a true restore
        rather than an approximate one: with no budget set the test returns ``ITERATING``
        immediately and the KSP's native test decides exactly as it would have.
        """
        if self._test_installed:
            return
        self.ksp.addConvergenceTest(self._test, prepend=True)
        self._test_installed = True


class SolverInstrumentation:
    """A solver's PETSc instrumentation: the sub-solve gauge, and the guard when armed.

    One instance per solver, created on first use. It re-attaches itself to whatever
    SNES the solver currently holds, because a setup-dirtying change (remesh, adapt, a
    rebuilt discretisation) replaces the SNES and drops everything attached to the old
    one.
    """

    def __init__(self):
        self.wall_per_step = None            # None => guard disarmed
        self.deadline_expired = False        # latched for the whole of one solve
        self._ksps: Dict[int, _InstrumentedKSP] = {}      # keyed by PETSc handle
        self._outer_handle = None
        self._expires = math.inf
        self._comm = None
        self._flag = np.zeros(1, dtype=np.int32)

    # ---------------------------------------------------------------- arming

    @property
    def armed(self):
        return self.wall_per_step is not None

    def arm(self, wall_per_step):
        """Set the per-Newton-step wall-clock budget, in seconds."""
        budget = float(wall_per_step)
        if not budget > 0.0:
            raise ValueError(
                f"wall_per_step must be a positive number of seconds (got "
                f"{wall_per_step!r}); call unguard() to remove the deadline."
            )
        self.wall_per_step = budget
        for entry in self._ksps.values():
            entry.install_test()

    def disarm(self):
        self.wall_per_step = None
        self._expires = math.inf
        # The prepended tests stay installed -- PETSc has no way to remove them -- but
        # with no budget they return ITERATING and the native test decides, which is
        # exactly the uninstrumented behaviour.

    def release(self):
        """Drop every PETSc reference, keeping the arming state.

        petsc4py increments the reference count of a KSP handed out by ``getKSP`` or
        ``getFieldSplitSubKSP``, so holding these entries keeps a destroyed solver's KSP,
        PC and whole multigrid hierarchy alive. A solver that rebuilds in a loop would
        then carry two hierarchies at once -- the leak BUGFIX(#157) was written to close.
        Called from the solver when it tears its SNES down.
        """
        self._ksps.clear()
        self._outer_handle = None

    # ------------------------------------------------------------ attachment

    def begin_solve(self, snes):
        """Prepare instrumentation for a solve that is about to start.

        Attaches to the current SNES's outer KSP if it is one we have not seen, and
        clears the per-solve counters. Called from the solver's single solve funnel, so
        a recreated SNES is picked up automatically.
        """
        self.deadline_expired = False

        outer = snes.getKSP()
        if outer.getType() in _MONITOR_SENSITIVE_KSPS:
            # These two change what they DO when a monitor is attached: KSPSolve_PREONLY
            # adds a norm, a matvec and a second norm purely to have something to report,
            # and KSPRICHARDSON gives up the fused PCApplyRichardson path. Neither has
            # outer iterations to bound or to count, so instrumenting them would be all
            # cost and no information.
            return
        if self._outer_handle not in (None, outer.handle):
            # A rebuilt solver (remesh, adapt, new discretisation) carries a new SNES,
            # and everything attached to the old one went with it. Start over rather
            # than accumulate blocks that can never report again.
            self._ksps.clear()
        self._outer_handle = outer.handle

        for entry in self._ksps.values():
            entry.reset()
        self._attach(outer, name="outer", is_outer=True, mid_solve=False)
        # Start the clock HERE, not at the outer KSP's first iteration. Under left
        # preconditioning -- PETSc's default for GMRES, which is the Stokes outer solver
        # -- KSPInitialResidual applies the preconditioner BEFORE iteration 0, so a
        # clock that only started at iteration 0 would leave one full velocity solve
        # plus one pressure solve outside the budget on every single solve. Measured:
        # the sub-block monitors fire before the outer monitor for pc_side LEFT.
        self.open_step()

    def attach_sub_ksps(self, ksp, mid_solve):
        """Attach to the fieldsplit blocks of ``ksp``'s preconditioner.

        Only ever called from inside a solve: the blocks do not exist until ``PCSetUp``
        has run, and asking for them earlier is a PETSc error rather than an empty list.
        They persist across solves, so this attaches once and then does nothing.
        """
        pc = ksp.getPC()
        if pc.getType() != "fieldsplit":
            return                                        # no sub-solves to account for
        for sub in pc.getFieldSplitSubKSP():
            self._attach(sub, name=_block_name(sub), is_outer=False,
                         mid_solve=mid_solve)

    def _attach(self, ksp, name, is_outer, mid_solve):
        if ksp.handle in self._ksps:
            return
        entry = _InstrumentedKSP(ksp, name, self, is_outer, mid_solve)
        self._ksps[ksp.handle] = entry
        if is_outer:
            self._comm = ksp.getComm().tompi4py()
        if self.armed:
            entry.install_test()

    # ------------------------------------------------------------ the deadline

    def open_step(self):
        """Start the clock for one Newton step.

        Once the deadline has fired it stays fired for the rest of the solve, so a
        solve that runs the SNES more than once -- the Picard-to-Newton continuation, or
        a warm-start retry after divergence -- cannot hand each attempt a fresh full
        budget.

        MEASURED: PETSc already prevents this on its own. After a deadline exit the
        sub-preconditioner is marked failed, and the next ``snes.solve`` bails before it
        reaches an iteration, so neither a retry nor a continuation stage re-armed even
        with the latch removed (tick counts 32 against 34). The latch is kept because
        that protection is an implicit consequence of PC-failure propagation rather than
        a guarantee, and because it makes the intended semantics explicit -- but it is
        belt-and-braces, and there is deliberately no test claiming to prove otherwise.
        """
        if not self.armed or self.deadline_expired:
            return
        self._expires = time.monotonic() + self.wall_per_step

    def deadline_passed(self):
        """Has the budget run out? The answer must be identical on every rank.

        A convergence test that returns different reasons on different ranks leaves the
        ranks on different PETSc code paths and the next collective deadlocks. Wall
        clocks are not synchronised across ranks, so the local answer is reduced over
        the KSP's own communicator. That costs one integer reduction per iteration,
        against a multigrid cycle -- and every iteration already pays for at least one
        norm reduction, so it does not change the communication character of the solve.
        """
        if self.deadline_expired:
            return True            # latched, and latched identically on every rank
        if not self.armed:
            return False
        local = 1 if time.monotonic() > self._expires else 0
        if self._comm is None or self._comm.size == 1:
            return bool(local)
        self._flag[0] = local
        self._comm.Allreduce(MPI.IN_PLACE, self._flag, op=MPI.MAX)
        return bool(self._flag[0])

    # ---------------------------------------------------------------- readout

    def sub_reports(self):
        """Per-block work for the solve that just finished.

        The outer KSP is excluded: its count is already ``solve_report.ksp_its``.
        """
        return {
            entry.name: entry.report()
            for entry in self._ksps.values()
            if not entry.is_outer and entry.applications > 0
        }
