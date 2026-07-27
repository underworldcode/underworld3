"""Sub-solve work gauge and the wall-clock guard (``systems/solver_health.py``).

These tests are written against the two ways the feature could be fake:

* the gauge could be the outer Krylov count under a new name, which would be useless —
  so it is asserted to see work the outer count cannot;
* the guard could fail to stop a grind, or could stop a healthy solve. Both are checked,
  and the healthy-solve check compares the *answer* against an unguarded solve, because
  the guard replaces PETSc's convergence tests and a sloppy replacement would quietly
  move where the solve stops.

The test that has to prove the guard bites *inside* the fieldsplit blocks — the claim
the whole design rests on — drives a fake clock rather than a wall-clock budget. A
budget in seconds cannot express "expire during the block solves" on an unknown machine:
too large and the solve finishes, too small and it expires at the outer iteration
before the blocks are ever reached.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import solver_health


class _TickingClock:
    """A clock that advances one tick per reading.

    Makes the deadline expire after a known number of convergence-test checks instead
    of after an interval of real time, so the test asserts the same thing on a fast
    machine and a slow one.
    """

    def __init__(self, tick=1.0):
        self.tick = tick
        self.now = 0.0

    def monotonic(self):
        self.now += self.tick
        return self.now


@pytest.fixture(scope="module")
def stokes_box():
    """A small, well-conditioned Stokes solve.

    Deliberately easy: these tests are about the instrumentation, and a genuinely hard
    solve would make them slow and their timing fragile.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3
    )
    v = uw.discretisation.MeshVariable("Ug", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pg", mesh, 1, degree=1, continuous=True)

    solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    solver.bodyforce = sympy.Matrix(
        [0.0, -sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]
    )
    for wall in ("Top", "Bottom", "Left", "Right"):
        solver.add_dirichlet_bc((0.0, 0.0), wall)
    # Guard exits are reports, not errors — a driver reads them and steps back.
    solver.petsc_options["snes_error_if_not_converged"] = "false"

    solver.solve()          # first solve: pays the JIT, and attaches the instrumentation
    return solver, v


@pytest.mark.level_1
@pytest.mark.tier_a
def test_sub_solve_gauge_sees_work_the_outer_count_cannot(stokes_box):
    """``ksp_its`` counts outer Krylov iterations, which Eisenstat--Walker collapses to
    about one per Newton step. The multigrid cycles are in the velocity block."""
    solver, _ = stokes_box
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


@pytest.mark.level_1
@pytest.mark.tier_a
def test_generous_budget_leaves_a_healthy_solve_alone(stokes_box):
    """The guard replaces PETSc's convergence tests. If the replacement were wrong the
    solve would stop somewhere else — so compare the answer, not just the reason."""
    solver, v = stokes_box
    solver.unguard()
    solver.solve(zero_init_guess=True)
    unguarded = np.asarray(v.array).copy()
    assert solver.solve_report.converged

    solver.guard(wall_per_step=600.0)
    solver.solve(zero_init_guess=True)
    guarded_report = solver.solve_report
    solver.unguard()

    assert guarded_report.converged
    assert not guarded_report.deadline_expired
    drift = np.abs(np.asarray(v.array) - unguarded).max()
    assert drift < 1.0e-6, f"guarded answer differs from unguarded by {drift:.3e}"


@pytest.mark.level_1
@pytest.mark.tier_a
def test_exhausted_budget_stops_the_solve_and_reports_it(stokes_box):
    """A budget too small to finish must stop the solve, and must not claim success.

    Reporting matters as much as stopping: a solve cut short that still said
    ``converged`` would silently corrupt a continuation driver, which reads exactly
    this to decide whether a parameter station is reachable.
    """
    solver, _ = stokes_box
    solver.guard(wall_per_step=1.0e-6)
    solver.solve(zero_init_guess=True)
    report = solver.solve_report
    solver.unguard()

    assert report.deadline_expired, "the wall-clock deadline never fired"
    assert not report.converged, "a solve cut short by the deadline reported success"
    assert report.reason_str == "DIVERGED_LINEAR_SOLVE"


@pytest.mark.level_1
@pytest.mark.tier_a
def test_deadline_bites_inside_the_fieldsplit_blocks(stokes_box, monkeypatch):
    """The claim the design rests on: the deadline is honoured *below* the outer Krylov
    iteration, where an iteration cap has nothing to count.

    The clock is set to expire after roughly fifty convergence-test checks. The outer
    test contributes only a couple of those, so expiry necessarily happens inside the
    block solves — and the assertions confirm it: the outer iteration never completed
    (``ksp_its == 0``, so no cap on it could have fired) while the velocity block had
    already run many multigrid cycles.
    """
    solver, _ = stokes_box
    clock = _TickingClock(tick=1.0)
    monkeypatch.setattr(solver_health, "time", clock)

    solver.guard(wall_per_step=50.0)          # fifty ticks == fifty checks
    solver.solve(zero_init_guess=True)
    report = solver.solve_report
    solver.unguard()

    assert report.deadline_expired
    assert not report.converged
    assert report.ksp_its == 0, "the outer Krylov iteration completed; nothing to prove"
    assert report.sub["velocity"].its >= 10, (
        "the velocity block did no real work before the deadline fired, so this does "
        "not show the deadline reaching inside the blocks"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_unguard_restores_normal_convergence(stokes_box):
    """The deadline must be removable — otherwise a driver that guards one probe would
    poison every later solve on the same solver."""
    solver, _ = stokes_box
    solver.guard(wall_per_step=1.0e-6)
    solver.solve(zero_init_guess=True)
    assert solver.solve_report.deadline_expired

    solver.unguard()
    solver.solve(zero_init_guess=True)
    report = solver.solve_report
    assert report.converged
    assert not report.deadline_expired


@pytest.mark.level_1
@pytest.mark.tier_a
def test_guard_rejects_a_meaningless_budget(stokes_box):
    solver, _ = stokes_box
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            solver.guard(wall_per_step=bad)


@pytest.mark.level_1
@pytest.mark.tier_a
def test_guard_refuses_the_rotated_free_slip_path(stokes_box):
    """Rotated free-slip runs its own Krylov loop outside ``self.snes``, so the deadline
    cannot reach it. Refusing is the point: a guard that attaches and never fires looks
    like protection and is not."""
    solver, _ = stokes_box
    solver.add_rotated_freeslip_bc(0, "Top")
    try:
        with pytest.raises(NotImplementedError, match="rotated free-slip"):
            solver.guard(wall_per_step=10.0)
    finally:
        solver._rotated_freeslip_bcs.clear()
