"""The forward stress history in parallel: np >= 2 equals serial.

The Maxwell shear box with the forward history under a flow with both velocity
components (a rigid rotation added to the shear), so arrivals cross partition
seams whichever way the box is cut. The stress at the origin after twenty
steps must be the serial BDF-1 value 1 - 1.1**-20 = 0.85136, the same number
the integration-point history gives.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.mpi(min_size=2), pytest.mark.timeout(600)]


@pytest.mark.mpi(min_size=2)
def test_the_forward_history_gives_the_serial_stress_on_every_rank():
    eta = G = 1.0
    speed, height, width, dt = 0.5, 1.0, 2.0, 0.1
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8), minCoords=(-width / 2, -height / 2),
                                       maxCoords=(width / 2, height / 2))
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "forward"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.constitutive_model.Parameters.shear_modulus = G
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((speed, 0.0), "Top")
    stokes.add_dirichlet_bc((-speed, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-8
    relocated = 0
    for _ in range(20):
        stokes.solve(timestep=dt, zero_init_guess=False)
        relocated += stokes.DFDt._n_relocated
    origin = np.array([[0.0, 0.0]])
    value = float(np.asarray(uw.function.global_evaluate(stokes.DFDt.psi_star[0].sym[0, 1], origin)).reshape(-1)[0])
    assert abs(value - 0.85136) < 2.0e-5, value
    # the seam must actually have been crossed for this to test the parallel path
    assert uw.mpi.comm.allreduce(relocated, op=uw.MPI.SUM) > 0
