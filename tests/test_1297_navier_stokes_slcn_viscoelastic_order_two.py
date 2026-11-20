"""NavierStokesSLCN with a viscoelastic model: the momentum flux is the model's
stress, and the stress history is built at the model's order (UNFILED 39).

The solver averaged its flux history with Adams-Moulton weights, (5/12, 8/12,
-1/12) at order 2. With a viscoelastic model that history is the stress, which the
model's BDF update has already integrated in time, so the average was a second
integration on top, and at order 2 it diverged: on the shear box below the stress
at the origin came out 2.95 away from Stokes at step 30, and on a box with a stiff
inclusion the velocity 277. The solver's own order also overwrote the model's,
so a BDF-2 model given to a solver of order 1 ran at order 1. Stokes is the
reference: the composed NavierStokes agrees with it to 3e-8 on the shear box.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

DT, STEPS = 0.1, 30
POINTS = np.array([[0.0, 0.0], [0.2, 0.0], [-0.5, 0.2], [0.6, -0.3], [0.3, 0.25]])


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _box(family, inclusion=False, solver_order=2, model_order=2):
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8), minCoords=(-1.0, -0.5),
                                        maxCoords=(1.0, 0.5))
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    if family == "stokes":
        s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    else:
        s = uw.systems.NavierStokesSLCN(mesh, v, p, rho=1.0e-6, order=solver_order)
        s.bodyforce = sympy.Matrix([[0.0, 0.0]])
    s.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        s.Unknowns, order=model_order, integrator="bdf")
    cm = s.constitutive_model
    cm.Parameters.shear_viscosity_0 = (
        1.0 + 9.0 * sympy.exp(-((x - 0.2) ** 2 + y ** 2) / 0.04) if inclusion else 1.0)
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = DT
    s.add_dirichlet_bc((0.5, 0.0), "Top")
    s.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    s.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    s.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    s.tolerance = 1.0e-8
    return v, s


def _run(v, s):
    stress, velocity = [], []
    for _ in range(STEPS):
        s.solve(timestep=DT, zero_init_guess=False)
        level_0 = s.Unknowns.DFDt.psi_star[0].sym
        stress.append(float(np.asarray(uw.function.evaluate(level_0[0, 1], POINTS[:1])).reshape(-1)[0]))
        velocity.append(np.asarray(uw.function.evaluate(v.sym, POINTS)).reshape(-1))
    return np.array(stress), np.array(velocity)


def test_the_order_two_stress_follows_stokes_on_the_shear_box():
    reference, _ = _run(*_box("stokes"))
    stress, _ = _run(*_box("slcn"))
    # the stress at the origin, every step; measured 5.3e-5 at step 30
    assert np.abs(stress - reference).max() < 1.0e-4 * np.abs(reference).max(), (stress, reference)


def test_the_order_two_velocity_stays_bounded_with_a_stiff_inclusion():
    _, reference = _run(*_box("stokes", inclusion=True))
    _, velocity = _run(*_box("slcn", inclusion=True))
    # measured 3.2e-2 at step 30 (NavierStokes against Stokes: 4.8e-2 here)
    error = np.abs(velocity - reference).max(axis=1) / np.abs(reference).max()
    assert error.max() < 0.1, error


def test_the_model_keeps_its_order_and_the_history_is_built_at_it():
    _v, s = _box("slcn", solver_order=1, model_order=2)
    assert s.constitutive_model.order == 2
    assert s.Unknowns.DFDt.order == 2
