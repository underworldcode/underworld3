"""A warm start is held to the same convergence target as a cold start of the same problem.

PETSc's nonlinear test is relative to the residual of the guess a solve starts from, so a
warm start near the answer was asked for ``rtol`` of an already small residual and kept
iterating. Every solve now measures one reference, ``F_ref = max(||F(x0)||, ||F(0)||)``:
``F(0)`` is what a cold start would have measured. It stops at the first iterate inside
``T = max(rtol * F_ref, atol)``. Design:
docs/developer/design/CONVERGENCE_REFERENCE_POLICY_2026-11.md.

On each native route, a nonlinear problem is solved cold, its forcing is changed by 1 %,
and it is solved again warm. Each solve must:
- end inside T, computed here from its own first residual and ``F(0)``, which this file
  evaluates through the SNES;
- end at the FIRST iterate inside T. On development the warm solve's target was rtol of
  its own small first residual, so it continued past T.

A solve with memory is the exception. A time step starts from the last step's answer and
carries it through its history, so ``F(0)`` measures the whole field and rtol of it lets a
step stop before it has made its change. Such an equation declares a zero reference, and a
step is measured against its own first residual, ``T = max(rtol * ||F(x0)||, atol)``.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

RTOL = 1.0e-3


def _residual_at_zero(solver):
    """||F(0)||, the residual a cold start of the current problem would have measured."""
    x = solver.snes.getSolution().duplicate()
    x.set(0.0)
    F = x.duplicate()
    solver.snes.computeFunction(x, F)
    norm = F.norm()
    x.destroy()
    F.destroy()
    return norm


def _check_stops_at_the_first_iterate_inside_the_target(solver, label, reference=None):
    history, _its = solver.snes.getConvergenceHistory()
    history = np.asarray(history, dtype=float)
    if reference is None:
        reference = _residual_at_zero(solver)
    target = RTOL * max(history[0], reference)
    assert solver.snes.getConvergedReason() > 0, (label, solver.snes.getConvergedReason())
    assert history[-1] <= target, (label, history, target)
    assert np.all(history[:-1] > target), (
        f"{label}: continued past the first iterate inside the target", history, target)


def _power_law_stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    v = uw.discretisation.MeshVariable("V0028", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0028", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    edot = mesh.vector.strain_tensor(v.sym)
    eII = sympy.sqrt(sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2) + edot[0, 1] ** 2)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        (sympy.Float(0.01) + eII) ** sympy.Rational(-2, 3))
    amplitude = uw.expression(r"A_{0028}", 1.0, "forcing amplitude")
    x, _y = mesh.X
    stokes.bodyforce = amplitude * sympy.Matrix([[0, sympy.cos(np.pi * x)]])
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.tolerance = RTOL
    return stokes, amplitude


def _nonlinear_poisson():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    T = uw.discretisation.MeshVariable("T0028", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    grad = mesh.vector.gradient(T.sym)
    poisson.constitutive_model.Parameters.diffusivity = 1.0 + 10.0 * grad.dot(grad)
    amplitude = uw.expression(r"S_{0028}", 10.0, "source amplitude")
    x, y = mesh.X
    poisson.f = amplitude * sympy.sin(np.pi * x) * sympy.sin(np.pi * y)
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.tolerance = RTOL
    return poisson, amplitude


@pytest.mark.parametrize("build", [_power_law_stokes, _nonlinear_poisson],
                         ids=["saddle_point", "scalar"])
def test_a_warm_start_stops_at_the_cold_start_target(build):
    solver, amplitude = build()
    solver.solve(zero_init_guess=True)
    _check_stops_at_the_first_iterate_inside_the_target(solver, "cold")
    amplitude.sym = 1.01 * float(amplitude.sym)
    solver.solve(zero_init_guess=False)
    _check_stops_at_the_first_iterate_inside_the_target(solver, "warm, forcing changed by 1 %")


def _nonlinear_advection_diffusion():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    v = uw.discretisation.MeshVariable("U0028", mesh, 2, degree=1)
    T = uw.discretisation.MeshVariable("C0028", mesh, 1, degree=2)
    adv_diff = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    grad = mesh.vector.gradient(T.sym)
    adv_diff.constitutive_model.Parameters.diffusivity = 1.0 + grad.dot(grad)
    adv_diff.add_dirichlet_bc(0.0, "Left")
    adv_diff.add_dirichlet_bc(0.0, "Right")
    v.array[:, 0, 0] = 0.05
    x, y = mesh.X
    T.array = uw.function.evaluate(sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y), T.coords)
    adv_diff.tolerance = RTOL
    return adv_diff


def test_a_time_step_is_measured_against_its_own_first_residual():
    adv_diff = _nonlinear_advection_diffusion()
    assert adv_diff._carries_memory()
    # A step short against the diffusion time changes the field little: ||F(x0)|| is
    # about 8 and ||F(0)|| about 260. The step's first residual is the reference. Under
    # max(||F(x0)||, ||F(0)||) every step stops near 0.02, above rtol of its own first
    # residual.
    for step in range(3):
        adv_diff.solve(zero_init_guess=False, timestep=1.0e-4)
        _check_stops_at_the_first_iterate_inside_the_target(
            adv_diff, f"time step {step}", reference=0.0)


def test_the_saddle_point_route_keeps_eisenstat_walker():
    # Eisenstat-Walker is PETSc's KSP pre-solve on this route. The floor is composed
    # with the KSP's convergence test, so EW still sets each linear solve's rtol, and
    # the first is EW's rtol_0. Set as a pre-solve, the floor replaced EW's.
    stokes, _amplitude = _power_law_stokes()
    stokes.solve(zero_init_guess=True)
    assert stokes.snes.getUseEW()
    rtols = []

    def _record(ksp, its, rnorm):
        if its == 0:
            rtols.append(ksp.getTolerances()[0])

    stokes.snes.getKSP().setMonitor(_record)
    stokes.solve(zero_init_guess=True)
    stokes.snes.getKSP().cancelMonitor()
    assert len(rtols) >= 2, rtols
    assert rtols[0] == pytest.approx(stokes.snes.getParamsEW()["rtol_0"]), rtols


def test_a_linear_tolerance_the_solver_did_not_derive_is_not_loosened():
    # The floor replaces only the ksp_rtol that ``tolerance`` derives. A linear rtol
    # set by the user, or by the library for a solve it wants tight, stays in force.
    derived, _ = _nonlinear_poisson()
    derived.solve(zero_init_guess=True)
    assert derived._convergence_reference["linear_floor"] == pytest.approx(
        0.1 * derived._convergence_reference["target"])
    own, _ = _nonlinear_poisson()
    own.petsc_options["ksp_rtol"] = 1.0e-9
    own.solve(zero_init_guess=True)
    assert own._convergence_reference["linear_floor"] is None
