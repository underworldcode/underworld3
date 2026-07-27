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


@dataclass(frozen=True)
class SubSolveReport:
    """Work done by one fieldsplit sub-solve over a single outer solve.

    Attributes
    ----------
    name
        Block name taken from the sub-KSP's options prefix -- ``"velocity"`` or
        ``"pressure"`` for the Stokes saddle-point solver.
    its
        Iterations summed over every application of this sub-solve. For the velocity
        block under ``pc_type=mg`` this is the multigrid cycle count, i.e. the honest
        work axis.
    applications
        Number of times the sub-solve ran during the outer solve.
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

    def __str__(self) -> str:
        return (f"{self.name}: {self.its} its / {self.applications} applications"
                + ("" if self.complete else " (lower bound)"))


def _block_name(ksp):
    """Readable block name from a sub-KSP options prefix.

    Prefixes look like ``Solver_6_fieldsplit_velocity_``; the part that identifies the
    block is the tail after the last ``fieldsplit_``.
    """
    prefix = ksp.getOptionsPrefix() or ""
    tail = prefix.rsplit("fieldsplit_", 1)[-1]
    return tail.strip("_") or prefix.strip("_") or "sub"


def _default_convergence(ksp, its, rnorm, state):
    """Reproduce ``KSPConvergedDefault``'s verdict for this iteration.

    A custom convergence test *replaces* the default -- petsc4py exposes no way to chain
    to it -- so a guard has to carry the ordinary rtol / atol / divtol semantics itself,
    or a healthy KSP would never converge. Tolerances are read fresh each iteration
    because Eisenstat--Walker rewrites the outer rtol before every Newton step.
    """
    R = PETSc.KSP.ConvergedReason
    if its == 0:
        state["r0"] = rnorm if rnorm > 0.0 else 1.0
    rtol, atol, divtol, _max_it = ksp.getTolerances()
    if rnorm != rnorm:                                    # NaN
        return R.DIVERGED_NANORINF
    if rnorm <= atol:
        return R.CONVERGED_ATOL
    if rnorm <= rtol * state["r0"]:
        return R.CONVERGED_RTOL
    if rnorm >= divtol * state["r0"]:
        return R.DIVERGED_DTOL
    return R.ITERATING


class _InstrumentedKSP:
    """One KSP the instrumentation is attached to, and everything it needs.

    Holds the iteration counters (fed by a monitor, which cannot affect the solve) and,
    when the guard is armed, the deadline-checking convergence test that replaces the
    default one.
    """

    def __init__(self, ksp, name, instrumentation, is_outer, mid_solve):
        self.ksp = ksp
        self.name = name
        self.its = 0
        self.applications = 0
        self.complete = not mid_solve
        self._instrumentation = instrumentation
        self.is_outer = is_outer
        self._state = {"r0": 1.0}
        ksp.setMonitor(self._monitor)

    def reset(self):
        self.its = 0
        self.applications = 0
        self.complete = True          # attached before this solve began

    def report(self):
        return SubSolveReport(name=self.name, its=self.its,
                              applications=self.applications, complete=self.complete)

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
            # PETSc does not fix the order of the monitor and the convergence test, so
            # whichever runs first at iteration 0 opens the step. Both are idempotent.
            self._instrumentation.open_step()
            self._instrumentation.attach_sub_ksps(ksp, mid_solve=True)
        if self._instrumentation.deadline_passed():
            self._instrumentation.deadline_expired = True
            return _DEADLINE_REASON
        return _default_convergence(ksp, its, rnorm, self._state)

    def install_test(self):
        self.ksp.setConvergenceTest(self._test)

    def remove_test(self):
        self.ksp.setConvergenceTest(None)                 # restores KSPConvergedDefault


class SolverInstrumentation:
    """A solver's PETSc instrumentation: the sub-solve gauge, and the guard when armed.

    One instance per solver, created on first use. It re-attaches itself to whatever
    SNES the solver currently holds, because a setup-dirtying change (remesh, adapt, a
    rebuilt discretisation) replaces the SNES and drops everything attached to the old
    one.
    """

    def __init__(self):
        self.wall_per_step = None            # None => guard disarmed
        self.deadline_expired = False
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
        for entry in self._ksps.values():
            entry.remove_test()

    # ------------------------------------------------------------ attachment

    def begin_solve(self, snes):
        """Prepare instrumentation for a solve that is about to start.

        Attaches to the current SNES's outer KSP if it is one we have not seen, and
        clears the per-solve counters. Called from the solver's single solve funnel, so
        a recreated SNES is picked up automatically.
        """
        self.deadline_expired = False
        self._expires = math.inf

        outer = snes.getKSP()
        if self._outer_handle not in (None, outer.handle):
            # A rebuilt solver (remesh, adapt, new discretisation) carries a new SNES,
            # and everything attached to the old one went with it. Start over rather
            # than accumulate blocks that can never report again.
            self._ksps.clear()
        self._outer_handle = outer.handle

        for entry in self._ksps.values():
            entry.reset()
        self._attach(outer, name="outer", is_outer=True, mid_solve=False)

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
        """Start the clock for one Newton step."""
        if self.armed:
            self._expires = time.monotonic() + self.wall_per_step
        else:
            self._expires = math.inf

    def deadline_passed(self):
        """Has the budget run out? The answer must be identical on every rank.

        A convergence test that returns different reasons on different ranks leaves the
        ranks on different PETSc code paths and the next collective deadlocks. Wall
        clocks are not synchronised across ranks, so the local answer is reduced over
        the KSP's own communicator. That costs one integer reduction per iteration,
        against a multigrid cycle -- and every iteration already pays for at least one
        norm reduction, so it does not change the communication character of the solve.
        """
        if self._expires == math.inf:
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
