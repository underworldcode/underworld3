"""Sub-solve work gauge and the wall-clock guard (``systems/solver_health.py``).

Written against the ways the feature could be fake:

* the gauge could be the outer Krylov count under a new name — so it is asserted to see
  work the outer count cannot, and to admit when its count is only a lower bound;
* the guard could fail to stop a grind, or stop a healthy solve. Both are checked, and
  the healthy-solve check compares *iteration counts* as well as the answer: the guard
  runs a test in front of PETSc's own, and a test that perturbed convergence would move
  the cost long before it moved the answer;
* ``wall_per_step`` could mean per *solve* rather than per Newton step, which one linear
  solve cannot distinguish — so the re-arming is exercised on a nonlinear model;
* ``unguard()`` could leave the guard half-attached.

The tests that must prove *where* the deadline bites drive an injected clock rather than
a wall-clock budget. A budget in seconds cannot express "expire during the block solves"
on an unknown machine: too large and the solve finishes, too small and it expires at the
outer iteration before the blocks are reached. Its limitation is stated where it is used.

Tier B, not A: these are new, and Charter §8 reserves tier A for tests that have been
through review and use in anger.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import solver_health

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


class _TickingClock:
    """A clock that advances one tick per reading.

    Makes the deadline expire after a known number of convergence-test checks instead of
    after an interval of real time, so the test asserts the same thing on a fast machine
    and a slow one. Note what this cannot do: because PETSc's real work costs zero ticks,
    expiry is guaranteed for any budget — these tests can catch a guard that never fires,
    not one that fires too eagerly. The real-clock test below covers that direction.
    """

    def __init__(self, tick=1.0):
        self.tick = tick
        self.now = 0.0

    def monotonic(self):
        self.now += self.tick
        return self.now


def _stokes(tag, viscosity):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3
    )
    v = uw.discretisation.MeshVariable(f"U{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1, continuous=True)

    solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = viscosity(v)
    x, y = mesh.X
    solver.bodyforce = sympy.Matrix(
        [0.0, -sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]
    )
    for wall in ("Top", "Bottom", "Left", "Right"):
        solver.add_dirichlet_bc((0.0, 0.0), wall)
    # Guard exits are reports, not errors — a driver reads them and steps back.
    solver.petsc_options["snes_error_if_not_converged"] = "false"
    return solver, v


@pytest.fixture(scope="module")
def linear_box():
    """A small, well-conditioned LINEAR Stokes solve — one Newton step."""
    solver, v = _stokes("g", lambda v: 1.0)
    solver.solve()        # first solve: pays the JIT, and attaches the instrumentation
    yield solver, v
    solver.unguard()


@pytest.fixture
def guarded(linear_box):
    """Hands back the shared solver and guarantees it is disarmed afterwards.

    Without this a test that fails part-way leaves the module-scoped solver armed at a
    microsecond budget, and every later test in the file fails for the wrong reason.
    """
    solver, v = linear_box
    try:
        yield solver, v
    finally:
        solver.unguard()


def test_sub_solve_gauge_sees_work_the_outer_count_cannot(guarded):
    """``ksp_its`` counts outer Krylov iterations, which Eisenstat--Walker collapses to
    about one per Newton step. The multigrid cycles are in the velocity block."""
    solver, _ = guarded
    solver.solve(zero_init_guess=True)
    report = solver.solve_report

    assert report.converged
    assert "velocity" in report.sub, "no velocity block work recorded"
    assert "pressure" in report.sub, "no pressure block work recorded"
    # The point of the gauge: the outer count is not the cost. Two orders of magnitude
    # apart in practice; one order of magnitude is a safe floor for the assertion.
    assert report.sub["velocity"].its > 10 * report.ksp_its
    assert report.sub["velocity"].applications >= 1
    # Attached before this solve began, so the count is exact rather than a lower bound.
    assert report.sub["velocity"].complete


def test_first_solve_admits_its_count_is_a_lower_bound():
    """A fresh solver cannot see the fieldsplit blocks until PETSc has set up the
    preconditioner, part-way through its first solve. The count that results is short —
    measured, by about half — so the report must say so rather than pass it off as
    exact."""
    solver, _ = _stokes("lb", lambda v: 1.0)
    solver.solve()
    first = solver.solve_report.sub["velocity"]
    solver.solve(zero_init_guess=True)
    second = solver.solve_report.sub["velocity"]

    assert not first.complete, "the first solve claimed an exact count it cannot have"
    assert second.complete
    assert second.its > first.its, (
        "the first solve should undercount; if it does not, the 'lower bound' contract "
        "is either wrong or no longer needed"
    )


def test_generous_budget_changes_neither_the_answer_nor_the_cost(guarded):
    """The guard runs a test in front of PETSc's own. If that composition perturbed
    convergence at all, the iteration counts would move first — long before the answer
    did, and invisibly for anything confined to the sub-blocks."""
    solver, v = guarded
    solver.unguard()
    solver.solve(zero_init_guess=True)
    unguarded = np.asarray(v.array).copy()
    before = solver.solve_report
    assert before.converged

    solver.guard(wall_per_step=600.0)
    solver.solve(zero_init_guess=True)
    after = solver.solve_report

    assert after.converged
    assert not after.deadline_expired
    drift = np.abs(np.asarray(v.array) - unguarded).max()
    assert drift < 1.0e-6, f"guarded answer differs from unguarded by {drift:.3e}"
    # Cost parity — the sharp assertion. An armed-but-unexpired guard must hand every
    # convergence decision straight back to PETSc.
    assert after.nl_its == before.nl_its
    assert after.ksp_its == before.ksp_its
    assert after.sub["velocity"].its == before.sub["velocity"].its
    assert after.sub["pressure"].its == before.sub["pressure"].its


def test_exhausted_budget_stops_the_solve_and_reports_it(guarded):
    """A budget too small to finish must stop the solve, and must not claim success.

    Reporting matters as much as stopping: a solve cut short that still said
    ``converged`` would silently corrupt a continuation driver, which reads exactly this
    to decide whether a parameter station is reachable.
    """
    solver, _ = guarded
    solver.guard(wall_per_step=1.0e-6)
    solver.solve(zero_init_guess=True)
    report = solver.solve_report

    assert report.deadline_expired, "the wall-clock deadline never fired"
    assert not report.converged, "a solve cut short by the deadline reported success"
    assert report.reason_str == "DIVERGED_LINEAR_SOLVE"


def test_deadline_bites_inside_the_fieldsplit_blocks(guarded, monkeypatch):
    """The claim the design rests on: the deadline is honoured *below* the outer Krylov
    iteration, where an iteration cap has nothing to count.

    The clock expires after roughly fifty convergence-test checks. The outer test
    contributes only a couple of those, so expiry necessarily happens inside the block
    solves — and the assertions confirm it: the outer iteration never completed
    (``ksp_its == 0``, so no cap on it could have fired) while the velocity block had
    already run many multigrid cycles.
    """
    solver, _ = guarded
    monkeypatch.setattr(solver_health, "time", _TickingClock(tick=1.0))

    solver.guard(wall_per_step=50.0)          # fifty ticks == fifty checks
    solver.solve(zero_init_guess=True)
    report = solver.solve_report

    assert report.deadline_expired
    assert not report.converged
    assert report.ksp_its == 0, "the outer Krylov iteration completed; nothing to prove"
    assert report.sub["velocity"].its >= 10, (
        "the velocity block did no real work before the deadline fired, so this does "
        "not show the deadline reaching inside the blocks"
    )


def test_the_budget_is_per_newton_step_not_per_solve(monkeypatch):
    """``wall_per_step`` is the only parameter the feature takes, and a linear solve
    cannot tell its two possible meanings apart — one Newton step is one solve.

    On a nonlinear model, run the solve once to learn how many clock ticks it costs in
    total, then re-run it with a budget of 90% of that. A per-SOLVE budget expires; a
    per-STEP budget does not, because no single Newton step costs that much. So the
    solve converging *is* the proof that the deadline restarted, and the tick count
    confirms it really did outlive one budget's worth.
    """
    solver, _ = _stokes("ps", lambda v: 1.0 + 5.0 * v.sym.dot(v.sym))
    solver.solve()
    assert solver.solve_report.nl_its >= 2, (
        "fixture is not nonlinear enough to distinguish per-step from per-solve"
    )

    clock = _TickingClock(tick=1.0)
    monkeypatch.setattr(solver_health, "time", clock)
    solver.guard(wall_per_step=1.0e9)          # never expires: just count the ticks
    try:
        solver.solve(zero_init_guess=True)
        assert not solver.solve_report.deadline_expired
        total = clock.now

        clock.now = 0.0
        budget = 0.9 * total
        solver.guard(wall_per_step=budget)
        solver.solve(zero_init_guess=True)
    finally:
        solver.unguard()

    report = solver.solve_report
    assert not report.deadline_expired, (
        f"a budget of {budget:.0f} ticks expired, so it was being applied to the whole "
        f"solve rather than restarted for each of its {report.nl_its} Newton steps"
    )
    assert report.converged
    assert clock.now > budget, (
        f"the solve cost only {clock.now:.0f} ticks against a {budget:.0f}-tick budget, "
        "so it never outlived one budget and proves nothing about restarting"
    )


def test_a_guard_armed_before_the_first_solve_still_fires():
    """The documented usage is ``guard(...)`` then ``solve()`` on a fresh solver, where
    the KSP the guard must attach to does not exist yet."""
    solver, _ = _stokes("cold", lambda v: 1.0)
    solver.guard(wall_per_step=1.0e-6)
    try:
        solver.solve()
    finally:
        solver.unguard()
    assert solver.solve_report.deadline_expired
    assert not solver.solve_report.converged


def test_unguard_leaves_the_solver_exactly_as_it_was(guarded):
    """The deadline must come off completely — otherwise a driver that guards one probe
    poisons every later solve on the same solver, in cost if not in answer."""
    solver, _ = guarded
    solver.unguard()
    solver.solve(zero_init_guess=True)
    never_guarded = solver.solve_report

    solver.guard(wall_per_step=1.0e-6)
    solver.solve(zero_init_guess=True)
    assert solver.solve_report.deadline_expired

    solver.unguard()
    solver.solve(zero_init_guess=True)
    after = solver.solve_report
    assert after.converged
    assert not after.deadline_expired
    assert after.nl_its == never_guarded.nl_its
    assert after.ksp_its == never_guarded.ksp_its
    assert after.sub["velocity"].its == never_guarded.sub["velocity"].its


def test_guard_rejects_a_meaningless_budget(guarded):
    solver, _ = guarded
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            solver.guard(wall_per_step=bad)


def test_guard_refuses_the_rotated_free_slip_path_whichever_order():
    """Rotated free-slip runs its own Krylov loop outside ``self.snes``, so the deadline
    cannot reach it. Refusing is the point — a guard that attaches and never fires looks
    like protection and is not.

    Both orders matter: arming the guard first and adding the BC afterwards used to slip
    through the arm-time check and produce exactly that silent no-op.
    """
    solver, _ = _stokes("rot1", lambda v: 1.0)
    solver.add_rotated_freeslip_bc(0, "Top")
    with pytest.raises(NotImplementedError, match="rotated free-slip"):
        solver.guard(wall_per_step=10.0)

    other, _ = _stokes("rot2", lambda v: 1.0)
    other.guard(wall_per_step=10.0)
    other.add_rotated_freeslip_bc(0, "Top")
    with pytest.raises(NotImplementedError, match="rotated free-slip"):
        other.solve()
