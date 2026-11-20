"""After a solve, a viscoelastic stress history's level 0 holds the stress that
solve produced, in every solver family (#742).

The analytic Maxwell shear box: simple shear at rate 1 of a Maxwell element
(eta = G = 1), ETD-1, five solves of 0.1. ETD-1 is exact for constant forcing,
so the stress after the fifth solve is 1 - exp(-0.5) = 0.393469. Stokes and the
composed NavierStokes committed it after the solve; NavierStokesSLCN recorded
the flux at the next step's start instead, so its level 0 held the stress one
solve old, 1 - exp(-0.4) = 0.329681, and a diagnostic or checkpoint read after a
solve was a step stale. It now commits after the solve as the others do.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

DT, STEPS = 0.1, 5
EXACT = 1.0 - np.exp(-STEPS * DT)          # 0.393469
STALE = 1.0 - np.exp(-(STEPS - 1) * DT)    # 0.329681


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _shear_box(family):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8), minCoords=(-1.0, -0.5),
                                        maxCoords=(1.0, 0.5))
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    if family == "stokes":
        solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    elif family == "navier_stokes":
        solver = uw.systems.NavierStokes(mesh, v, p, rho=1.0e-6, order=1)
    else:
        solver = uw.systems.NavierStokesSLCN(mesh, v, p, rho=1.0e-6, order=1)
        solver.bodyforce = sympy.Matrix([[0.0, 0.0]])
    solver.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        solver.Unknowns, order=1, integrator="etd")
    cm = solver.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = DT
    solver.add_dirichlet_bc((0.5, 0.0), "Top")
    solver.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    solver.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    solver.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    solver.tolerance = 1.0e-8
    return solver


@pytest.mark.parametrize("family", ["stokes", "navier_stokes", "navier_stokes_slcn"])
def test_level_0_holds_the_stress_the_solve_produced(family):
    solver = _shear_box(family)
    for _ in range(STEPS):
        solver.solve(timestep=DT, zero_init_guess=False)
    level_0 = solver.Unknowns.DFDt.psi_star[0].sym
    sxy = float(np.asarray(uw.function.evaluate(level_0[0, 1], np.array([[0.0, 0.0]]))).reshape(-1)[0])
    assert abs(sxy - EXACT) < 2.0e-3, (family, sxy, EXACT, STALE)


def test_navier_stokes_slcn_refuses_a_stress_transport_it_does_not_honour():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    solver = uw.systems.NavierStokesSLCN(mesh, v, p, rho=1.0, order=1)
    assert solver.stress_transport == "semi_lagrangian"
    solver.stress_transport = "semi_lagrangian"
    with pytest.raises(ValueError, match="NavierStokes"):
        solver.stress_transport = "integration_point"
