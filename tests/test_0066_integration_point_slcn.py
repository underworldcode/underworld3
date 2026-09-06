"""Semi-Lagrangian history at the integration points.

Two properties: the value each slot carries is the snapshot evaluated
exactly at the traced departure point (the floor: for a P2 field and a
uniform velocity the sample is exact to round-off, for one and for two
segments), and on a rotating Gaussian the scheme is at least as accurate as
the nodal SLCN it replaces and keeps the peak better.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_slots_are_exact_departure_point_values():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    f = lambda X: 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2 + X[:, 0] * X[:, 1]
    T.data[:, 0] = f(np.asarray(T.coords))
    v = np.array([1.0, 0.5])
    V = sympy.Matrix([[v[0], v[1]]])
    dt = 0.1

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V, degree=2, order=2)
    assert all(ps.is_integration_point for ps in ddt.psi_star)

    ddt.update_pre_solve(dt)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - v * dt
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert inside.sum() > 100
    assert np.abs(ddt.psi_star[0].data[inside, 0] - f(foot[inside])).max() < 1e-12

    # Second slot: two segments back, sampled from the older snapshot.
    ddt.update_post_solve(dt)
    ddt.update_pre_solve(dt)
    foot2 = X - v * 2 * dt
    inside2 = (foot2 > 0.0).all(1) & (foot2 < 1.0).all(1)
    assert np.abs(ddt.psi_star[1].data[inside2, 0] - f(foot2[inside2])).max() < 1e-12

    # Negative control: the nodal scheme's slot is an interpolant of an
    # interpolant, so the same check on it does not hold to round-off.
    Tn = uw.discretisation.MeshVariable("Tn", mesh, 1, degree=2)
    Tn.data[:, 0] = f(np.asarray(Tn.coords))
    nodal = uw.systems.ddt.SemiLagrangian(mesh, Tn, V, uw.VarType.SCALAR, degree=2, continuous=True, order=2)
    nodal.update_pre_solve(dt)
    nodal.update_post_solve(dt)
    nodal.update_pre_solve(dt)
    Xn = np.asarray(nodal.psi_star[1].coords)
    footn = Xn - v * 2 * dt
    insiden = (footn > 0.0).all(1) & (footn < 1.0).all(1)
    assert np.abs(nodal.psi_star[1].data[insiden, 0] - f(footn[insiden])).max() > 1e-12


def _rotating_gaussian(mesh, kind, dt, nsteps):
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    x0, sig = 0.5, 0.12
    gauss = lambda X, cx, cy: np.exp(-((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2) / (2 * sig ** 2))
    T = uw.discretisation.MeshVariable(f"T_{kind}", mesh, 1, degree=2)
    T.data[:, 0] = gauss(np.asarray(T.coords), x0, 0.0)
    if kind == "ip":
        DuDt = uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V, degree=2, order=1)
        adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V, DuDt=DuDt, order=1)
    else:
        adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V, order=1)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1e-9
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)
    for _ in range(nsteps):
        adv.solve(timestep=dt)
    ang = nsteps * dt
    exact = gauss(np.asarray(T.coords), x0 * np.cos(ang), x0 * np.sin(ang))
    E = uw.discretisation.MeshVariable(f"E_{kind}", mesh, 1, degree=2)
    E.data[:, 0] = T.data[:, 0] - exact
    l2 = np.sqrt(uw.maths.Integral(mesh, E.sym[0] ** 2).evaluate())
    return l2, T.data[:, 0].max()


def test_undersampled_rule_is_refused():
    """P2 history on a qdegree-2 triangle mesh: 6 points for 6 local dofs.
    That configuration blows up at small Courant number, so it is refused."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=2)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    V = sympy.Matrix([[1.0, 0.0]])
    with pytest.raises(RuntimeError, match="oversampled"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V, degree=2)
    # P1 on the same rule is 2x oversampled and accepted.
    T1 = uw.discretisation.MeshVariable("T1", mesh, 1, degree=1)
    uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T1, V, degree=1)


@pytest.mark.level_2
def test_rotating_gaussian_beats_nodal_slcn():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1, -1), maxCoords=(1, 1), cellSize=0.08, qdegree=3
    )
    dt, nsteps = 0.1, 16
    l2_nodal, peak_nodal = _rotating_gaussian(mesh, "nodal", dt, nsteps)
    l2_ip, peak_ip = _rotating_gaussian(mesh, "ip", dt, nsteps)
    assert l2_ip <= l2_nodal
    assert peak_ip >= peak_nodal
    assert l2_ip < 0.02
