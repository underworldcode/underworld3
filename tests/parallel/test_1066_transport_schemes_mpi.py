"""Every stress and advection history in parallel: np >= 3 equals serial.

Stress: the turned-over Maxwell box of test_1062 (a shear modulus varying in x,
two counter-rotating cells between no-slip walls, an unstructured mesh), so
the stress is non-uniform and the flow crosses seams of either orientation.
Advection: a rotating Gaussian in a square box (the flow crosses every wall),
P2, a quarter turn, with each of the four semi-Lagrangian value histories.
Momentum: the lid-driven cavity of test_1103, with each velocity history. The values at two points after the
run must be the serial ones. At least three ranks: two ranks meet only along
one seam, while three or more also meet at points, where an arrival can be
handed to either of two other ranks.

The value histories that apply an inflow value are given one, so the inflow
detection runs across seams too. At np 4 the backward integration-point stress
history has three departure points that the parallel evaluator used to strand
and fill by rbf extrapolation (7.5e-5 before the 2026-09-27 fix): this file is
the regression test for global_evaluate's containment round as well.
"""
import numpy as np
import pytest
import sympy

import sys
from pathlib import Path

import underworld3 as uw
from test_1062_forward_stress_history_mpi import turned_over_maxwell_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_1103_navier_stokes_velocity_transport import CAVITY_U, lid_driven_cavity  # noqa: E402

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.mpi(min_size=3), pytest.mark.timeout(1200)]

# BASELINES: the serial values at the points of each test (2026-09-27)
STRESS_XY = {
    "backward_nodes": (-0.1001236, 0.1000386),
    "backward_integration_points": (-0.1166604, 0.0682694),
    "forward_integration_points": (-0.1190800, 0.0702081),
    "forward_nodes": (-0.1190115, 0.0701718),
    "lagrangian": (-0.1214269, 0.0672075),
    "eulerian": (-0.1186881, 0.0699410),
    # the global projection of the forward histories (2026-10-05)
    "forward_integration_points:global": (-0.1189135, 0.0709336),
    "forward_nodes:global": (-0.1190056, 0.0679288),
}
ADVECTED_T = {
    "backward_nodes": (0.7259154, 0.0751478),
    "backward_integration_points": (0.7533908, 0.0739654),
    "forward_integration_points": (0.6347829, 0.0836136),
    "forward_nodes": (0.7694358, 0.0721413),
    "forward_integration_points:global": (0.6278635, 0.0857784),
    "forward_nodes:global": (0.7855832, 0.0711712),
}
# np 3, 4 and 6 give the serial values to the 7 figures recorded: the history
# projections are converged to 1e-10 and no stage depends on which rank, or
# which of two cells, a point is assigned to. A misrouted or dropped value
# costs 1e-5 or more.
ATOL = 1.0e-6
T_POINTS = np.array([[0.0, 0.5], [-0.2, 0.35]])


def rotating_gaussian(transport, steps=16, dt=np.pi / 32):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3, regular=False)
    x, y = mesh.X
    transport, _, reconstruction = transport.partition(":")      # "forward_nodes:global"
    degree = 1 if transport == "forward_integration_points" else 2
    T = uw.discretisation.MeshVariable("T_rg", mesh, 1, degree=degree)
    T.array[:, 0, 0] = uw.function.evaluate(
        sympy.exp(-((x - 0.5) ** 2 + y ** 2) / (2 * 0.1 ** 2)), T.coords).reshape(-1)
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=sympy.Matrix([[-y, x]]), order=1,
                                      transport=transport)
    if reconstruction:
        adv.DuDt.reconstruction = reconstruction
    if adv.DuDt.applies_inflow_value:
        adv.DuDt.inflow_value = sympy.Matrix([[0.0]])
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-3
    for wall in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, wall)
    adv.tolerance = 1.0e-10
    for _ in range(steps):
        adv.solve(timestep=dt)
    return np.asarray(uw.function.global_evaluate(T.sym[0], T_POINTS)).reshape(-1)


@pytest.mark.parametrize("transport", [
    pytest.param(t, marks=pytest.mark.xfail(
        uw.mpi.size >= 4, strict=False, reason="#797: the particle history differs from "
        "serial by 1.2e-5 at np 4 and 6 (np 3 matches)"))
    if t == "lagrangian" else t for t in STRESS_XY])
def test_every_stress_history_gives_the_serial_stress_on_every_rank(transport):
    uw.reset_default_model()
    _kind, values, relocated = turned_over_maxwell_box(transport)
    if transport.startswith("forward_integration"):
        assert relocated > 0      # the seams were crossed, so the exchange ran
    assert np.allclose(values, STRESS_XY[transport], atol=ATOL), (transport, values)


@pytest.mark.parametrize("transport", list(ADVECTED_T))
def test_every_value_history_gives_the_serial_field_on_every_rank(transport):
    uw.reset_default_model()
    values = rotating_gaussian(transport)
    assert np.allclose(values, ADVECTED_T[transport], atol=ATOL), (transport, values)


@pytest.mark.parametrize("transport", list(CAVITY_U))
def test_every_velocity_history_gives_the_serial_flow_on_every_rank(transport):
    uw.reset_default_model()
    _kind, values = lid_driven_cavity(transport)
    assert np.allclose(values, CAVITY_U[transport], atol=ATOL), (transport, values)
