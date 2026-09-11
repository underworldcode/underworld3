"""Semi-Lagrangian history at the integration points.

Three properties. The value each slot carries is the snapshot evaluated
exactly at the traced departure point (the floor: for a P2 field and a
uniform velocity the sample is exact to round-off, for one and for two
segments). On a rotating Gaussian the scheme is at least as accurate as the
nodal SLCN it replaces and keeps the peak better. And a history may be a
vector or a tensor, which stores one dof per INDEPENDENT component -- a
symmetric tensor in 2-D is 2x2 symbolically and three columns in storage, and
getting that packing wrong transposes a stress silently.
"""

import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems.ddt import _storage_components

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
def test_rotating_gaussian_ip_accuracy():
    """Contract: the integration-point trace resolves the rotating Gaussian.

    An absolute bound against the known solution, with no rival method in it.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1, -1), maxCoords=(1, 1), cellSize=0.08, qdegree=3
    )
    l2_ip, _ = _rotating_gaussian(mesh, "ip", 0.1, 16)
    assert l2_ip < 0.02, f"integration-point trace L2 error {l2_ip:.3e}"


# tier_c overrides the module-level tier_a for this test alone: it compares two
# transport managers, so it can fail because one of them got better.
@pytest.mark.level_2
@pytest.mark.tier_c
def test_rotating_gaussian_ip_against_nodal_characterisation():
    """Characterisation: the integration-point trace is not worse than nodal.

    This compares two METHODS, so it can fail because the code improved — a
    better nodal SLCN would break it, and that is good news. Tier C: a failure
    demands an explanation, not a revert. It is NOT the justification for the
    integration-point path; `test_rotating_gaussian_ip_accuracy` asserts that
    against the known solution.

    The relationship is sensitive to the Courant number, the quadrature degree
    and the element size, so it is a characterisation of this fixture
    (cellSize=0.08, dt=0.1, 16 steps) rather than a general claim. Compare
    `project_integration_point_proxy_pic_lip`, where the bulk diagnostics were
    identical while the interface answer was not.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1, -1), maxCoords=(1, 1), cellSize=0.08, qdegree=3
    )
    dt, nsteps = 0.1, 16
    l2_nodal, peak_nodal = _rotating_gaussian(mesh, "nodal", dt, nsteps)
    l2_ip, peak_ip = _rotating_gaussian(mesh, "ip", dt, nsteps)

    print(f"L2: ip={l2_ip:.4e} nodal={l2_nodal:.4e}; "
          f"peak: ip={peak_ip:.4f} nodal={peak_nodal:.4f}")
    explain = ("If the nodal path improved, explain it and re-characterise; "
               "do not revert to make this pass.")
    assert l2_ip <= l2_nodal, f"ip {l2_ip:.3e} > nodal {l2_nodal:.3e}. {explain}"
    assert peak_ip >= peak_nodal, (
        f"ip peak {peak_ip:.4f} < nodal {peak_nodal:.4f}. {explain}")


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


# ---------------------------------------------------------------------------
# Vector and tensor histories
# ---------------------------------------------------------------------------

V_UNIFORM = np.array([1.0, 0.5])
DT = 0.1


def _velocity():
    return sympy.Matrix([[V_UNIFORM[0], V_UNIFORM[1]]])


# Fields chosen to lie in the P2 space, so the sample at the departure point is
# exact to round-off, and with every component DISTINCT so that a packing or
# transposition error cannot hide.
def _scalar_field(X):
    return 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2


def _vector_field(X):
    return np.column_stack([
        1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] ** 2,
        -2.0 + X[:, 0] * X[:, 1] + X[:, 1] ** 2,
    ])


def _tensor_entries(X):
    """The 2x2 symbolic entries, keyed by (i, j)."""
    xx = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
    yy = -2.0 + X[:, 1] + 0.5 * X[:, 1] ** 2
    xy = 0.25 + X[:, 0] * X[:, 1] - 0.5 * X[:, 0] ** 2
    return {(0, 0): xx, (1, 1): yy, (0, 1): xy, (1, 0): xy}


def _pack(entries, columns):
    return np.column_stack([entries[ij] for ij in columns])


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vtype,dim,expected",
    [
        (uw.VarType.SCALAR, 2, [(0, 0)]),
        (uw.VarType.VECTOR, 2, [(0, 0), (0, 1)]),
        (uw.VarType.SYM_TENSOR, 2, [(0, 0), (1, 1), (0, 1)]),
        (uw.VarType.SYM_TENSOR, 3, [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]),
    ],
)
def test_the_storage_order_is_what_the_symbol_reconstructs(vtype, dim, expected):
    """Pin the column -> (i, j) convention against the variable itself.

    If the storage order ever changes, this fails here rather than quietly
    transposing a stress history somewhere downstream.
    """
    if dim == 2:
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.4, qdegree=3)
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.5, qdegree=3,
        )
    var = uw.discretisation.MeshVariable(f"sv{vtype.value}{dim}", mesh,
                                         vtype=vtype, degree=1)
    columns = _storage_components(vtype, tuple(var.sym.shape))
    assert columns == expected
    assert len(columns) == var.num_components

    # a distinct marker per column, read back through the symbol
    with uw.synchronised_array_update():
        for c in range(var.num_components):
            var.data[:, c] = 10.0 * (c + 1)
    point = np.full((1, mesh.dim), 0.5)
    got = np.asarray(uw.function.evaluate(var.sym, point)).reshape(var.sym.shape)
    for c, (i, j) in enumerate(columns):
        assert got[i, j] == pytest.approx(10.0 * (c + 1)), (c, i, j, got)


def test_a_vector_history_holds_the_departure_point_values():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    U = uw.discretisation.MeshVariable("Uv", mesh, vtype=uw.VarType.VECTOR, degree=2)
    with uw.synchronised_array_update():
        U.data[...] = _vector_field(np.asarray(U.coords))

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, U, _velocity(), vtype=uw.VarType.VECTOR, degree=2, order=2)
    assert ddt.num_components == 2
    assert all(ps.is_integration_point for ps in ddt.psi_star)
    assert ddt.bdf().shape == (1, 2)

    ddt.update_pre_solve(DT)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - V_UNIFORM * DT
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert inside.sum() > 100
    got = np.asarray(ddt.psi_star[0].data)[inside]
    assert np.abs(got - _vector_field(foot[inside])).max() < 1e-12

    # two segments back, from the older snapshot
    ddt.update_post_solve(DT)
    ddt.update_pre_solve(DT)
    foot2 = X - V_UNIFORM * 2 * DT
    inside2 = (foot2 > 0.0).all(1) & (foot2 < 1.0).all(1)
    got2 = np.asarray(ddt.psi_star[1].data)[inside2]
    assert np.abs(got2 - _vector_field(foot2[inside2])).max() < 1e-12


def test_a_symmetric_tensor_history_transports_every_component():
    """Three independent components, all different, so a packing error or a
    transposed off-diagonal cannot pass."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    S = uw.discretisation.MeshVariable("St", mesh, vtype=uw.VarType.SYM_TENSOR,
                                       degree=2)
    columns = _storage_components(uw.VarType.SYM_TENSOR, (2, 2))
    with uw.synchronised_array_update():
        S.data[...] = _pack(_tensor_entries(np.asarray(S.coords)), columns)

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, S, _velocity(), vtype=uw.VarType.SYM_TENSOR, degree=2, order=1)
    assert ddt.num_components == 3
    assert ddt._components == columns
    assert ddt.bdf().shape == (2, 2)

    ddt.update_pre_solve(DT)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - V_UNIFORM * DT
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert inside.sum() > 100
    expected = _pack(_tensor_entries(foot[inside]), columns)
    got = np.asarray(ddt.psi_star[0].data)[inside]
    assert np.abs(got - expected).max() < 1e-12

    # and the symbol reads back as the right 2x2 matrix, off-diagonal included
    point = foot[inside][0].reshape(1, -1)
    entries = _tensor_entries(point)
    sym = np.asarray(
        uw.function.evaluate(ddt.psi_star[0].sym, X[inside][0].reshape(1, -1))
    ).reshape(2, 2)
    assert sym[0, 1] == pytest.approx(entries[(0, 1)][0], abs=1e-10)
    assert sym[1, 0] == pytest.approx(sym[0, 1])
    assert sym[0, 0] == pytest.approx(entries[(0, 0)][0], abs=1e-10)
    assert sym[1, 1] == pytest.approx(entries[(1, 1)][0], abs=1e-10)
    assert abs(sym[0, 0] - sym[1, 1]) > 0.1        # the components are distinct


def test_a_scalar_history_is_unchanged():
    """The generalisation must not move the scalar answer."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=3)
    T = uw.discretisation.MeshVariable("Ts", mesh, 1, degree=2)
    with uw.synchronised_array_update():
        T.data[:, 0] = _scalar_field(np.asarray(T.coords))

    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, T, _velocity(), degree=2, order=1)
    assert ddt.num_components == 1
    assert ddt.bdf().shape == (1, 1)

    ddt.update_pre_solve(DT)
    X = np.asarray(ddt.psi_star[0].coords)
    foot = X - V_UNIFORM * DT
    inside = (foot > 0.0).all(1) & (foot < 1.0).all(1)
    assert np.abs(
        np.asarray(ddt.psi_star[0].data)[inside, 0] - _scalar_field(foot[inside])
    ).max() < 1e-12


@pytest.mark.parametrize("vtype", [uw.VarType.VECTOR, uw.VarType.SYM_TENSOR])
def test_the_history_symbol_participates_in_expressions(vtype):
    """A shaped history has to be usable, not merely storable: its symbol goes
    where a mesh variable's symbol goes."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=3)
    var = uw.discretisation.MeshVariable(f"Pe{vtype.value}", mesh, vtype=vtype,
                                         degree=2)
    columns = _storage_components(vtype, tuple(var.sym.shape))
    with uw.synchronised_array_update():
        var.data[...] = 2.0
    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, var, _velocity(), vtype=vtype, degree=2, order=1)
    ddt.update_pre_solve(DT)

    star = ddt.psi_star[0].sym
    trace = sum(star[i, i] for i in range(star.shape[0]))
    assert float(uw.maths.Integral(mesh, trace).evaluate()) == pytest.approx(
        2.0 * star.shape[0], rel=1e-8)

    # the second invariant of the difference is a legitimate weak-form term
    expr = (star - ddt.bdf()).T * (star - ddt.bdf())
    assert expr.shape[0] == star.shape[1]
    assert len(columns) == ddt.num_components


def test_the_refusal_is_gone_but_the_rule_check_is_not():
    """An undersampled quadrature rule is still refused, whatever the shape:
    a delta field cannot carry more values than the rule has points."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=2)
    U = uw.discretisation.MeshVariable("Ur", mesh, vtype=uw.VarType.VECTOR, degree=2)
    with pytest.raises(RuntimeError, match="qdegree|rule|oversample"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, U, _velocity(), vtype=uw.VarType.VECTOR, degree=2, order=1)


def test_the_storage_map_follows_the_shape_not_the_mesh_dimension():
    """On a manifold the topological and embedding dimensions differ — a
    spherical surface is dim 2, cdim 3 — and the variable sizes its storage by
    the embedding one, which is what ``.sym`` is shaped by. The map therefore
    reads the tensor dimension off the shape and never touches the mesh.
    """
    assert len(_storage_components(uw.VarType.SYM_TENSOR, (2, 2))) == 3
    assert len(_storage_components(uw.VarType.SYM_TENSOR, (3, 3))) == 6
    assert _storage_components(uw.VarType.VECTOR, (1, 3)) == [(0, 0), (0, 1), (0, 2)]


def test_a_vtype_that_does_not_match_psi_fn_is_refused():
    """And refused BEFORE any variable is allocated: a mesh variable created
    and then abandoned leaves its field on the DM (#1058)."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=3)
    before = len(mesh.vars)

    with pytest.raises(ValueError, match="psi_fn has shape"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, sympy.Matrix([[1.0]]), _velocity(),
            vtype=uw.VarType.VECTOR, degree=2, order=1)

    with pytest.raises(ValueError, match="psi_fn has shape"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, sympy.Matrix([[1.0, 2.0]]), _velocity(),
            vtype=uw.VarType.SYM_TENSOR, degree=2, order=1)

    assert len(mesh.vars) == before, "a refused history left variables behind"


def test_the_shape_guard_is_on_the_setter_not_only_the_constructor():
    """A solver reassigns ``DFDt.psi_fn = flux.T`` on every setup, so a guard
    that lives only in ``__init__`` is absent from the one path that is
    actually driven. A wrong shape there does not raise: the component writer
    reads ``psi_fn[i, j]`` for the slots it already has, so a larger matrix is
    silently TRUNCATED."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=3)
    S = uw.discretisation.MeshVariable("Sg", mesh, vtype=uw.VarType.SYM_TENSOR,
                                       degree=2)
    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(
        mesh, S, _velocity(), vtype=uw.VarType.SYM_TENSOR, degree=2, order=1)

    with pytest.raises(ValueError, match="psi_fn has shape"):
        ddt.psi_fn = sympy.eye(3)                 # would truncate to 3 columns
    with pytest.raises(ValueError, match="psi_fn has shape"):
        ddt.psi_fn = sympy.Matrix([[1.0]])        # would IndexError later

    ddt.psi_fn = sympy.Matrix([[1.0, 2.0], [2.0, 3.0]])   # the right shape
    assert tuple(ddt.psi_fn.shape) == (2, 2)


def test_a_full_tensor_is_not_accepted_as_a_symmetric_one():
    """SYM_TENSOR and TENSOR share a symbolic shape and differ in storage
    width (3 against 4 in 2-D), so the shape check alone cannot separate them
    and the mismatch used to surface as a bare broadcast error naming neither
    vtype."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=3)
    full = uw.discretisation.MeshVariable("Tg", mesh, vtype=uw.VarType.TENSOR,
                                          degree=2)
    with pytest.raises(ValueError, match="stores 4 components"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, full, _velocity(), vtype=uw.VarType.SYM_TENSOR,
            degree=2, order=1)


def test_an_asymmetric_psi_fn_under_sym_tensor_says_so():
    """A symmetric history stores the upper triangle, so an asymmetric psi_fn
    loses its lower entries. It used to do that in silence, and the nodal
    SemiLagrangian silently keeps the OTHER triangle (see the TODO(BUG) on
    that class), so a user swapping one for the other would get a different
    answer with no diagnostic either way."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.3, qdegree=3)
    x, y = mesh.X
    asymmetric = sympy.Matrix([[1 + x, 2 + y], [100.0, 3 + x * y]])

    with pytest.warns(UserWarning, match="not symmetric"):
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, asymmetric, _velocity(), vtype=uw.VarType.SYM_TENSOR,
            degree=2, order=1)

    # a symmetric one is silent
    symmetric = sympy.Matrix([[1 + x, 2 + y], [2 + y, 3 + x * y]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, symmetric, _velocity(), vtype=uw.VarType.SYM_TENSOR,
            degree=2, order=1)
