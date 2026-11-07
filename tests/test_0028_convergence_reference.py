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


def _check_stops_at_the_first_iterate_inside_the_target(solver, label):
    history, _its = solver.snes.getConvergenceHistory()
    history = np.asarray(history, dtype=float)
    target = RTOL * max(history[0], _residual_at_zero(solver))
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
