"""A stress history survives a snapshot and restore.

Run the Maxwell shear box for six steps, snapshot the model, run six more and
record the stress; restore, run the same six again: the stress must be the
same to round-off. For every history that carries the stress. A history
whose bookkeeping (step history, initialisation flag, launch values) were not
captured would either re-initialise from zero or carry the wrong step.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]   # solves: not level 1


def _shear_box(transport):
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    eta = G = 1.0
    speed, height, width, dt = 0.5, 1.0, 2.0, 0.1
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8), minCoords=(-width / 2, -height / 2),
                                       maxCoords=(width / 2, height / 2))
    v = uw.discretisation.MeshVariable(f"U_r_{transport}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_r_{transport}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.constitutive_model.Parameters.shear_modulus = G
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((speed, 0.0), "Top")
    stokes.add_dirichlet_bc((-speed, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-10
    return orchestration_model, stokes, dt


def _stress_at_origin(stokes):
    return float(np.asarray(uw.function.evaluate(stokes.DFDt.psi_star[0].sym[0, 1], np.array([[0.0, 0.0]]))).reshape(-1)[0])


@pytest.mark.parametrize("transport", ["semi_lagrangian", "integration_point", "forward"])
def test_a_restored_history_continues_where_it_left_off(transport):
    orchestration_model, stokes, dt = _shear_box(transport)
    for _ in range(6):
        stokes.solve(timestep=dt, zero_init_guess=False)
    snap = orchestration_model.save_state()
    for _ in range(6):
        stokes.solve(timestep=dt, zero_init_guess=False)
    straight = _stress_at_origin(stokes)
    straight_field = np.array(stokes.DFDt.psi_star[0].data)
    orchestration_model.load_state(snap)
    for _ in range(6):
        stokes.solve(timestep=dt, zero_init_guess=False)
    assert abs(_stress_at_origin(stokes) - straight) < 1.0e-12
    assert np.allclose(np.asarray(stokes.DFDt.psi_star[0].data), straight_field, rtol=0.0, atol=1.0e-12)
    # and it is a real twelve-step state, not a re-initialised six-step one
    assert abs(straight - (1.0 - 1.1 ** -12)) < 1.0e-6
