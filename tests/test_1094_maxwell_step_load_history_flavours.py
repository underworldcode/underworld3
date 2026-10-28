"""A Maxwell block under a step traction: the creep benchmark, per history flavour.

Plane strain, incompressible, a bar fixed at x = 0 and pulled by a constant
normal traction sigma at x = L with the long sides free: the deviatoric law
gives epsilon_xx = sigma / (4 mu) at once (the elastic jump) and a steady
creep rate d(epsilon_xx)/dt = sigma / (4 eta) thereafter, so the end velocity
settles to sigma L / (4 eta). ETD-1 holds the strain rate constant over a
step and integrates the stress ODE exactly for it, so with no history the
first step's velocity is sigma L / (4 eta (1 - e^-x)), x = dt mu / eta: the
jump spread over the step (0.946 of jump-over-dt plus creep at x = 1/8).
Every flavour of the stress history must reproduce this
(the history is the stress itself; with a velocity this small the transport
between steps is negligible and the flavours must agree with each other).

Written when the semi-Lagrangian and Eulerian flavours disagreed by a quarter
on the second step of the box fault model (2026-10-17).
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

L, SIGMA, MU, ETA, DT = 3.0, 0.01, 1.0, 1.0, 0.125


def _end_velocities(flavour, nsteps=4, tag="a"):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(L, 1.0),
                                              cellSize=0.25, regular=False, qdegree=3)
    v = uw.discretisation.MeshVariable(f"v_{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"p_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = flavour
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="none")
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])
    stokes.tolerance = 1e-8
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_natural_bc(sympy.Matrix([[-SIGMA, 0.0]]), "Right")      # a positive value pushes in
    out = []
    pts = np.array([[L - 0.01, 0.5]])
    for k in range(nsteps):
        stokes.solve(timestep=DT, zero_init_guess=(k == 0))
        out.append(float(np.asarray(uw.function.evaluate(v.sym, pts)).reshape(-1, 2)[0, 0]))
    return np.array(out)


@pytest.mark.parametrize("flavour", ["semi_lagrangian", "eulerian", "integration_point", "none"])
def test_creep_rate_and_elastic_jump(flavour):
    u = _end_velocities(flavour, tag=flavour[:4])
    creep = SIGMA * L / (4 * ETA)
    x = DT * MU / ETA
    first = creep / (1.0 - np.exp(-x))        # ETD-1's exact first step from a zero history
    assert abs(u[0] - first) < 0.08 * first, (flavour, u, first)
    # later steps: the creep rate, within the mesh's few per cent
    assert np.all(np.abs(u[1:] - creep) < 0.08 * creep), (flavour, u, creep)


def test_flavours_agree():
    us = {f: _end_velocities(f, nsteps=3, tag="g" + f[:3]) for f in ("semi_lagrangian", "eulerian", "none")}
    a = us["semi_lagrangian"]
    for f in ("eulerian", "none"):
        assert np.allclose(a, us[f], rtol=0.02), us
