"""The forward-from-nodes history on the rotating diffusing Gaussian.

Launched from the field's own nodes and from a lattice inside every element
(the field's interpolant is a polynomial there, so both are known exactly),
carried one step forward, fitted per cell at the field's degree and read back at
the nodes. Selected by transport="forward_nodes" on the SLCN advection-diffusion solver, half a
revolution, kappa 0.01, dt 0.02, P2 temperature, against the test_1101 fixture:
the backward nodal history gives 2.52e-2 on the disc and 6.43e-2 on the box
(where the flow crosses all four walls).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from test_1101_advdiff_swarm_rotating_gaussian import SIGMA, KAPPA, T_END, EXACT_PEAK, _disc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _run(mesh, walls):
    x, y = mesh.X
    sol = uw.analytic.RotatingGaussian(mesh, sigma=SIGMA, centre_radius=0.5, omega=1.0, diffusivity=KAPPA)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sol.at(0.0), T.coords).reshape(-1)
    V = sympy.Matrix([[-y, x]])
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V, order=1, transport="forward_nodes")
    assert type(adv.DuDt).__name__ == "ForwardNodesSemiLagrangian"
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = KAPPA
    for wall in walls:
        adv.add_dirichlet_bc(0.0, wall)
    dt = 0.02
    nsteps = int(round(T_END / dt)); dt = T_END / nsteps
    for _ in range(nsteps):
        adv.solve(timestep=dt)
    return float(sol.error(sol.at(T_END), T, norm="integral")), float(np.asarray(T.data).max())


def test_forward_from_nodes_on_the_disc():
    uw.reset_default_model()
    err, peak = _run(_disc(24), ("Upper",))
    assert abs(err - 0.017804) < 1.0e-4, err               # BASELINE (2026-09-27)
    assert abs(peak - EXACT_PEAK) < 0.002, (peak, EXACT_PEAK)


def test_forward_from_nodes_holds_the_box():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
                                             cellSize=2.0 / 24, qdegree=3, regular=False)
    err, peak = _run(mesh, ("Left", "Right", "Top", "Bottom"))
    assert abs(err - 0.065672) < 1.0e-4, err               # BASELINE (2026-09-27)
    assert abs(peak - EXACT_PEAK) < 0.005, (peak, EXACT_PEAK)


# The composed solver with each transport on the same fixture. Hard baselines: the
# stored-level flux is rebuilt from the carried field (an integration-point store
# through its continuous snapshot), so the numbers are the composed solver's own,
# not the SLCN solver's.
COMPOSED_DISC = {
    "eulerian": 0.017242,
    "backward_nodes": 0.020781,
    "backward_integration_points": 0.021344,      # theta 0.5: the snapshot stand-in
    "forward_nodes": 0.017335,
}


def _run_composed(mesh, walls, transport):
    x, y = mesh.X
    sol = uw.analytic.RotatingGaussian(mesh, sigma=SIGMA, centre_radius=0.5, omega=1.0, diffusivity=KAPPA)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sol.at(0.0), T.coords).reshape(-1)
    adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=sympy.Matrix([[-y, x]]), order=1, transport=transport)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = KAPPA
    for wall in walls:
        adv.add_dirichlet_bc(0.0, wall)
    dt = 0.02
    nsteps = int(round(T_END / dt)); dt = T_END / nsteps
    for _ in range(nsteps):
        adv.solve(timestep=dt)
    return float(sol.error(sol.at(T_END), T, norm="integral")), float(np.asarray(T.data).max()), adv


@pytest.mark.parametrize("transport", list(COMPOSED_DISC))
def test_the_composed_advdiffusion_carries_each_transport_on_the_disc(transport):
    uw.reset_default_model()
    err, peak, adv = _run_composed(_disc(24), ("Upper",), transport)
    assert abs(err - COMPOSED_DISC[transport]) < 1.0e-4, (transport, err)      # BASELINE (2026-10-05)
    assert abs(peak - EXACT_PEAK) < 0.003, (peak, EXACT_PEAK)
    assert adv.transport == transport
    if transport != "eulerian":
        with pytest.raises(ValueError, match="eulerian"):
            adv.supg_weight = 0.0
        assert np.isfinite(adv.estimate_dt())                 # the cell-crossing time, not inf


def test_the_composed_advdiffusion_refuses_the_linear_forward_fit_for_a_quadratic_field():
    uw.reset_default_model()
    with pytest.raises(NotImplementedError, match="degree must be 1"):
        _run_composed(_disc(24), ("Upper",), "forward_integration_points")
