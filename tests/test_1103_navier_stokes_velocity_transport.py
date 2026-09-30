"""The Navier-Stokes momentum carried by each transport: on the grid (SUPG) or by
each semi-Lagrangian scheme, chosen by ``velocity_transport``.

A lid-driven cavity at Reynolds number 100 (lid speed 1, viscosity 0.01,
density 1), first order, ten steps of 0.05 from rest: the momentum is advected,
so the velocity history is doing work. There is no closed form; each scheme is
held to its own recorded value. The forward integration-point history fits a
linear polynomial per cell and refuses the P2 velocity.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

POINTS = np.array([[0.5, 0.75], [0.3, 0.5]])
# BASELINES: horizontal velocity at POINTS after ten steps (2026-09-29)
CAVITY_U = {
    "eulerian": (-0.1290241, -0.0546760),
    "backward_nodes": (-0.1296780, -0.0552098),
    "backward_integration_points": (-0.1288954, -0.0548034),
    "forward_nodes": (-0.1295085, -0.0550723),
}


def lid_driven_cavity(transport, steps=10, dt=0.05, cell_size=1.0 / 12):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=cell_size, qdegree=3, regular=False)
    v = uw.discretisation.MeshVariable("U_cav", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P_cav", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1, velocity_transport=transport)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 0.01
    ns.add_dirichlet_bc((1.0, 0.0), "Top")
    for wall in ("Bottom", "Left", "Right"):
        ns.add_dirichlet_bc((0.0, 0.0), wall)
    ns.bodyforce = sympy.Matrix([[0.0, 0.0]])
    ns.tolerance = 1.0e-8
    for _ in range(steps):
        ns.solve(timestep=dt)
    values = np.asarray(uw.function.global_evaluate(v.sym[0], POINTS)).reshape(-1)
    return type(ns.DuDt).__name__, values


@pytest.mark.parametrize("transport", list(CAVITY_U))
def test_each_velocity_history_gives_its_recorded_cavity_flow(transport):
    uw.reset_default_model()
    kind, values = lid_driven_cavity(transport)
    expected_kind = ("EulerianSUPG" if transport == "eulerian"
                     else "".join(w.capitalize() for w in transport.split("_")) + "SemiLagrangian")
    assert kind == expected_kind
    assert np.allclose(values, CAVITY_U[transport], atol=1.0e-6), (transport, values)
    # the schemes differ by their transport error, about a percent here
    assert np.allclose(values, CAVITY_U["eulerian"], rtol=2.0e-2), (transport, values)


def test_the_forward_integration_point_fit_refuses_a_p2_velocity():
    uw.reset_default_model()
    with pytest.raises(NotImplementedError, match="degree must be 1"):
        lid_driven_cavity("forward_integration_points", steps=0)


def test_the_forward_schemes_refuse_order_two():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
    v = uw.discretisation.MeshVariable("U_o2", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P_o2", mesh, 1, degree=1)
    with pytest.raises(NotImplementedError, match="order must be 1"):
        uw.systems.NavierStokes(mesh, v, p, order=2, velocity_transport="forward_nodes")


def test_the_former_solver_names_still_work_and_say_what_replaces_them():
    uw.reset_default_model()
    with pytest.warns(FutureWarning, match="velocity_transport='backward_nodes'"):
        assert uw.systems.NavierStokesSLCN is uw.systems.solvers.SNES_NavierStokes
    with pytest.warns(FutureWarning, match="transport='backward_nodes'"):
        assert uw.systems.AdvDiffusionSLCN is uw.systems.solvers.SNES_AdvectionDiffusion
