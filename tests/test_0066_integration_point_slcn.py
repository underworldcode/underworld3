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
        adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V, DuDt=DuDt, order=1)
    else:
        adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V, order=1)
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
    from mpi4py import MPI
    peak = uw.mpi.comm.allreduce(float(T.data[:, 0].max()), op=MPI.MAX)  # global, not rank-local
    return l2, peak


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


def _unsteady_uniform_flow_check(kind, vform="var"):
    """Uniform velocity that changes linearly in time, v(t) = a + b t. The
    exact foot for the interval [t1, t1 + dt] is x - dt (a + b (t1 + dt/2)).
    With the mid-time velocity extrapolated from v(t1) and v(t0) the trace
    reproduces it; with v(t1) alone the foot is off by b dt^2 / 2."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    T = uw.discretisation.MeshVariable(f"T_{kind}", mesh, 1, degree=2)
    v_var = uw.discretisation.MeshVariable(f"v_{kind}", mesh, mesh.dim, degree=2)
    f = lambda X: 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2 + X[:, 0] * X[:, 1]
    T.data[:, 0] = f(np.asarray(T.coords))
    a, b, dt = np.array([1.0, 0.5]), np.array([2.0, -1.0]), 0.1
    # V_fn is symbolic by design: any expression of the variable must work.
    # "ramp": a constant that changes between the two steps; the cached
    # previous velocity must carry the OLD value (substituting snapshots of
    # the variables into the expression would read the new one).
    c = uw.expression(r"c_{ramp}", 1.0, "ramping factor")
    V_fn, factor = {"var": (v_var, 1.0), "neg": (-v_var.sym, -1.0), "half": (v_var.sym / 2, 0.5),
                    "ramp": (c * v_var.sym, 1.0)}[vform]

    if kind == "ip":
        ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V_fn, degree=2, order=1)
    else:
        ddt = uw.systems.ddt.SemiLagrangian(mesh, T, V_fn, uw.VarType.SCALAR, degree=2, continuous=True, order=1)

    v_var.data[...] = a + b * 0.0
    ddt.update_pre_solve(dt)
    ddt.update_post_solve(dt)
    if vform == "ramp":
        # v(t1) = 1.5 (a + b dt) through the constant; v(t0) = a. Extrapolated
        # mid-time velocity 1.5 v(t1) - 0.5 v(t0) = 2.25 (a + b dt) - 0.5 a.
        c.sym = 1.5
        v_var.data[...] = a + b * dt
        v_mid = 2.25 * (a + b * dt) - 0.5 * a
        v_naive = 1.5 * (a + b * dt)
    else:
        v_var.data[...] = a + b * dt
        v_mid = factor * (a + b * (dt + 0.5 * dt))
        v_naive = factor * (a + b * dt)
    ddt.update_pre_solve(dt)

    X = np.asarray(ddt.psi_star[0].coords)
    exact_foot = X - dt * v_mid
    naive_foot = X - dt * v_naive
    inside = (exact_foot > 0.02).all(1) & (exact_foot < 0.98).all(1) & (naive_foot > 0.02).all(1) & (naive_foot < 0.98).all(1)
    assert inside.sum() > 100
    got = np.asarray(ddt.psi_star[0].data[:, 0])
    return np.abs(got[inside] - f(exact_foot[inside])).max(), np.abs(got[inside] - f(naive_foot[inside])).max()


@pytest.mark.parametrize("vform", ["var", "neg", "half", "ramp"])
@pytest.mark.parametrize("kind", ["ip", "nodal"])
def test_midtime_velocity_makes_the_trace_second_order(kind, vform):
    err_exact, err_naive = _unsteady_uniform_flow_check(kind, vform)
    # The trace is exact to round-off. Integration-point: the tolerance
    # covers the evaluator, which returns one foot in ~2000 up to a few 1e-5
    # off (a locator edge case shared by evaluate and global_evaluate). Nodal:
    # the scheme traces from nodes nudged 0.1 % toward the cell centroid
    # and stores at the unnudged node, an error of order 0.001 h |grad psi|
    # per step (2e-4 here) that the integration-point scheme does not have.
    assert err_exact < (1e-4 if kind == "ip" else 1e-3)
    # Negative control: the foot from v^n alone is b dt^2/2 away, which for
    # this quadratic field is a visible difference.
    assert err_naive > 1e-3


@pytest.mark.parametrize("config", ["order2", "theta1", "cn"])
def test_composed_advdiffusion_reachability(config):
    """The composed uw.systems.AdvDiffusion (#688) takes the history as its
    transport manager. With no spatial term on the old level (BDF2, or
    theta = 1) the integration-point history runs there and matches the SLCN
    solver; with the Crank-Nicolson flux (theta = 0.5) the old level is
    differentiated, which a delta field cannot supply, and the JIT guard
    refuses with a clear message."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1, -1), maxCoords=(1, 1), cellSize=0.1, qdegree=3
    )
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    gauss = lambda X: np.exp(-((X[:, 0] - 0.5) ** 2 + X[:, 1] ** 2) / (2 * 0.12 ** 2))
    order, theta = {"order2": (2, 1.0), "theta1": (1, 1.0), "cn": (1, 0.5)}[config]

    def run(solver_cls, kwargs):
        T = uw.discretisation.MeshVariable(f"T_{config}_{solver_cls.__name__}", mesh, 1, degree=2)
        T.data[:, 0] = gauss(np.asarray(T.coords))
        D = uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V, degree=2, order=order, theta=theta)
        adv = solver_cls(mesh, u_Field=T, V_fn=V, DuDt=D, order=order, **kwargs)
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
        adv.constitutive_model.Parameters.diffusivity = 1e-9
        for b in ("Left", "Right", "Top", "Bottom"):
            adv.add_dirichlet_bc(0.0, b)
        for _ in range(8):
            adv.solve(timestep=0.1)
        return np.asarray(T.data[:, 0]).copy()

    if config == "cn":
        with pytest.raises(RuntimeError, match="integration-point"):
            run(uw.systems.AdvDiffusion, {})
        return
    kw = {"theta": theta} if order == 1 else {}
    T_composed = run(uw.systems.AdvDiffusion, kw)
    T_slcn = run(uw.systems.AdvDiffusionSLCN, {})
    # Same history, same time derivative; the solvers differ only in how the
    # (negligible) diffusion is applied, so the fields agree closely.
    assert np.abs(T_composed - T_slcn).max() < 5e-3


def test_value_and_flux_histories_share_one_characteristic_trace():
    """The SLCN solver's value and flux histories follow the same velocity
    from the same nodes: one trace per step (two velocity evaluations for
    the RK2 segment), the flux history served from the cache. With theta=1
    the old-level flux is never read, so the flux history is not traced at
    all and the trace records one velocity level per step."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=2)
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    for theta, hits in ((0.5, 1), (1.0, 0)):
        T = uw.discretisation.MeshVariable(f"T{int(theta * 10)}", mesh, 1, degree=2)
        T.data[:, 0] = np.exp(-((np.asarray(T.coords) - 0.5) ** 2).sum(1) / 0.02)
        adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V, order=1, theta=theta)
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
        adv.constitutive_model.Parameters.diffusivity = 1e-3
        for b in ("Left", "Right", "Top", "Bottom"):
            adv.add_dirichlet_bc(0.0, b)
        trace = adv._characteristics
        assert trace is not None
        assert adv.DuDt.characteristics is trace and adv.DFDt.characteristics is trace
        assert not adv.DuDt._owns_characteristics
        adv.solve(timestep=0.05)
        n0, h0 = trace.n_velocity_evaluations, trace.n_cache_hits
        adv.solve(timestep=0.05)
        # per step: one RK2 segment = 2 velocity evaluations, plus the one
        # evaluation that records v^{n-1} at the nodes
        assert trace.n_velocity_evaluations - n0 == 2, trace.n_velocity_evaluations - n0
        assert trace.n_cache_hits - h0 == hits
        assert adv._flux_history_is_read() == (theta < 1.0)
        if theta == 1.0:
            assert not adv.DFDt._history_initialised


def test_private_trace_when_a_manager_stands_alone():
    """A manager used without a solver owns its trace and delimits its own
    steps; the mid-time velocity becomes available after the first step."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=2)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]
    D = uw.systems.ddt.SemiLagrangian(mesh, T.sym, sympy.Matrix([[1.0, 0.0]]), vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    tr = D.characteristics
    assert D._owns_characteristics
    assert tr.midtime_expr() == tr.V_matrix()           # nothing recorded yet
    D.update_pre_solve(0.1)
    assert tr.level_valid(1)
    assert tr.midtime_expr() != tr.V_matrix()           # 1.5 v^n - 0.5 v^{n-1}
    assert tr.n_velocity_evaluations == 2
