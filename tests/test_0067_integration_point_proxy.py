"""Swarm proxy directly at the integration points.

``SwarmVariable(..., proxy_location="integration_points")`` reconstructs the
particle field at the mesh integration points into an
``IntegrationPointVariable`` and the weak form reads it there with no basis
interpolation (the Ellipsis / Underworld PIC-LIP mapping). Checked here:

- a material step carried by particles is reproduced far more sharply at
  the integration points than through the nodal proxy;
- the proxy has no gradient and the JIT refuses one;
- ``Lagrangian_Swarm`` accepts the option and its history slots become
  integration-point proxies that reproduce a linear field exactly;
- a symmetric-tensor swarm variable gets a multi-component proxy.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=2)


def test_material_step_is_sharper_at_integration_points():
    mesh = _mesh()
    x, y = mesh.X
    swarm = uw.swarm.Swarm(mesh)
    M_n = uw.swarm.SwarmVariable("Mn", swarm, 1, proxy_degree=1, proxy_location="nodes")
    M_q = uw.swarm.SwarmVariable("Mq", swarm, 1, proxy_location="integration_points")
    swarm.populate(fill_param=3)
    assert M_q._meshVar.is_integration_point and not M_n._meshVar.is_integration_point

    X = np.asarray(swarm._particle_coordinates.data)
    step = (X[:, 0] < 0.5).astype(float)
    with uw.synchronised_array_update():
        M_n.data[:, 0] = step
        M_q.data[:, 0] = step

    # The exact step at the integration points, as an integration-point field.
    S = uw.discretisation.IntegrationPointVariable("S", mesh)
    S.data[:, 0] = (np.asarray(S.coords)[:, 0] < 0.5).astype(float)

    err_n = uw.maths.Integral(mesh, (M_n.sym[0] - S.sym[0]) ** 2).evaluate()
    err_q = uw.maths.Integral(mesh, (M_q.sym[0] - S.sym[0]) ** 2).evaluate()
    # Both proxies are nonzero-error (the interface sits between particles),
    # and the integration-point proxy is sharper by a clear margin.
    assert err_q > 0.0 and err_n > 0.0
    assert err_q < 0.5 * err_n, (err_q, err_n)

    # The integration-point proxy carries the step within one cell: away
    # from the interface it is exactly 0 or 1 at every integration point.
    Xq = np.asarray(M_q._meshVar.coords)
    far = np.abs(Xq[:, 0] - 0.5) > 0.1
    vals = np.asarray(uw.function.evaluate(M_q.sym[0], Xq)).reshape(-1)   # own points, through the symbol
    assert np.allclose(vals[far], (Xq[far, 0] < 0.5).astype(float), rtol=0, atol=1e-14)

    # No gradient: the JIT refuses a derivative of the proxy symbol.
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    proj = uw.systems.solvers.SNES_Projection(mesh, T)
    proj.uw_function = M_q.sym[0].diff(x)
    with pytest.raises(RuntimeError, match="integration-point"):
        proj.solve()


def test_lagrangian_swarm_history_at_integration_points():
    mesh = _mesh()
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords) @ np.array([1.0, 2.0])   # x + 2y
    swarm = uw.swarm.Swarm(mesh)
    lag = uw.systems.ddt.Lagrangian_Swarm(
        swarm=swarm, psi_fn=T.sym, vtype=uw.VarType.SCALAR, degree=1, continuous=True,
        order=1, proxy_location="integration_points",
    )
    swarm.populate(fill_param=3)
    lag.update_pre_solve(dt=0.1)

    slot = lag.psi_star[0]
    assert slot._meshVar.is_integration_point
    # Particle values are the field at the particles ...
    Xp = np.asarray(swarm._particle_coordinates.data)
    assert np.allclose(np.asarray(slot.data)[:, 0], Xp[:, 0] + 2.0 * Xp[:, 1], atol=1e-10)
    # ... and the proxy at the integration points reproduces the linear field
    # (order-1 RBF is linear-exact), read straight from the slot's symbol.
    Xq = np.asarray(slot._meshVar.coords)
    got = np.asarray(uw.function.evaluate(slot.sym[0], Xq)).reshape(-1)
    assert np.allclose(got, Xq[:, 0] + 2.0 * Xq[:, 1], atol=1e-8)
    # bdf() is built from the integration-point symbols.
    assert slot._meshVar.sym[0] in lag.bdf()[0].free_symbols or lag.bdf()[0].has(slot._meshVar.sym[0])


def test_tensor_swarm_variable_gets_a_multicomponent_proxy():
    mesh = _mesh()
    swarm = uw.swarm.Swarm(mesh)
    S = uw.swarm.SwarmVariable("S", swarm, vtype=uw.VarType.SYM_TENSOR, proxy_location="integration_points")
    swarm.populate(fill_param=2)
    Nq = len(np.asarray(mesh.integration_rule.getData()[1]))
    c0, c1 = mesh.dm.getHeightStratum(0)
    assert S._meshVar.is_integration_point
    assert S._meshVar.data.shape == ((c1 - c0) * Nq, 3)
    with uw.synchronised_array_update():
        S.data[:, 0] = 1.0; S.data[:, 1] = 2.0; S.data[:, 2] = 3.0
    # Read through the symbol (which refreshes the proxy): the symmetric
    # tensor [[c0, c2], [c2, c1]] at every integration point.
    Xq = np.asarray(S._meshVar.coords)
    vals = np.asarray(uw.function.evaluate(S.sym, Xq)).reshape(len(Xq), 2, 2)
    assert np.allclose(vals, [[1.0, 3.0], [3.0, 2.0]])


@pytest.mark.level_2
def test_layered_couette_interface_on_edges_is_exact_at_integration_points():
    """Two viscosity layers (1 and 1e3) carried by particles, interface on mesh
    edges, top-driven layer flow. The exact velocity is piecewise linear and
    lies in the P2 space, so the only error is the proxy's representation of
    the step: the integration-point proxy is exact to solver tolerance, the
    nodal proxy (a node on the interface averages both materials) is not."""
    h, eta_top = 0.5, 1.0e3
    results = {}
    for proxy in ("nodes", "integration_points"):
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=2, regular=True)
        v = uw.discretisation.MeshVariable(f"v_{proxy}", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable(f"p_{proxy}", mesh, 1, degree=1)
        swarm = uw.swarm.Swarm(mesh)
        if proxy == "nodes":
            M = uw.swarm.SwarmVariable("M", swarm, 1, proxy_degree=1)
        else:
            M = uw.swarm.SwarmVariable("M", swarm, 1, proxy_location="integration_points")
        swarm.populate(fill_param=3)
        X = np.asarray(swarm._particle_coordinates.data)
        with uw.synchronised_array_update():
            M.data[:, 0] = (X[:, 1] > h).astype(float)
        eta = 1.0 + (eta_top - 1.0) * sympy.Max(0, sympy.Min(1, M.sym[0]))
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
        stokes.add_dirichlet_bc((1.0, 0.0), "Top")
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
        stokes.tolerance = 1e-8
        stokes.solve()
        A = 1.0 / (h + (1.0 - h) / eta_top)
        Xv = np.asarray(v.coords)
        vx_exact = np.where(Xv[:, 1] < h, A * Xv[:, 1], A * h + A / eta_top * (Xv[:, 1] - h))
        results[proxy] = np.abs(np.asarray(v.data[:, 0]) - vx_exact).max()
    assert results["integration_points"] < 1e-5
    assert results["nodes"] > 1e-2          # the control: the nodal proxy smears the step


# ---------------------------------------------------------------------------
# proxy_location="cells": a least-squares polynomial per cell
# ---------------------------------------------------------------------------


def _jittered_swarm(mesh, fill, seed=0, **var_kwargs):
    """A populated swarm with its lattice jittered so cells hold uneven particle counts."""
    swarm = uw.swarm.Swarm(mesh)
    M = uw.swarm.SwarmVariable("M", swarm, 1, **var_kwargs)
    swarm.populate(fill_param=fill)
    rng = np.random.default_rng(seed + uw.mpi.rank)
    X = np.array(swarm._particle_coordinates.data, copy=True)
    X += rng.uniform(-1, 1, X.shape) * 0.3 * 0.1 / fill
    X = np.clip(X, 1e-6, 1 - 1e-6)
    with uw.synchronised_array_update():
        swarm._particle_coordinates.data[...] = X
    swarm.migrate()
    return swarm, M


def test_cells_proxy_reproduces_polynomials_and_has_a_gradient():
    mesh = _mesh()
    x, y = mesh.X
    swarm, M = _jittered_swarm(mesh, 4, proxy_location="cells", proxy_degree=2)
    assert not M._meshVar.continuous and M._meshVar.degree == 2
    X = np.asarray(swarm._particle_coordinates.data)
    quad = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 4.0 * X[:, 0] * X[:, 1] - X[:, 1] ** 2
    with uw.synchronised_array_update():
        M.data[:, 0] = quad
    quad_sym = 1 + 2 * x - 3 * y + 4 * x * y - y ** 2
    # Read through the weak form: the assembler evaluates the discontinuous
    # proxy at the rule, where the fit must be exact.
    err = uw.maths.Integral(mesh, (M.sym[0] - quad_sym) ** 2).evaluate()
    assert err < 1e-14, err
    # Fifteen particles per cell: no cell dropped below Nb + 2 = 8 after the
    # jitter, so every cell took its own P2 fit.
    assert M._cell_projector.n_thin == 0
    # The proxy has a gradient (unlike the integration-point proxy).
    gerr = uw.maths.Integral(mesh, (M.sym[0].diff(x) - (2 + 4 * y)) ** 2).evaluate()
    assert gerr < 1e-12, gerr


def test_cells_proxy_light_swarm_stays_linear_exact():
    """Three particles per cell (below the degree-2 fit's own threshold): every
    cell takes the linear patch fit and a linear field is still exact."""
    mesh = _mesh()
    x, y = mesh.X
    swarm, M = _jittered_swarm(mesh, 1, proxy_location="cells", proxy_degree=2)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        M.data[:, 0] = 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1]
    err = uw.maths.Integral(mesh, (M.sym[0] - (1 + 2 * x + 3 * y)) ** 2).evaluate()
    assert err < 1e-14, err
    assert M._cell_projector.n_thin > 0


def test_cells_proxy_material_step_is_sharp_and_bounded_at_cell_edges():
    """A step on x = 0.5 with the mesh regular so cell edges lie on it: the
    per-cell fit is exactly 0 or 1 in every cell (no overshoot, unlike the RBF)."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=2, regular=True)
    x, y = mesh.X
    swarm = uw.swarm.Swarm(mesh)
    M = uw.swarm.SwarmVariable("M", swarm, 1, proxy_location="cells", proxy_degree=1)
    swarm.populate(fill_param=3)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        M.data[:, 0] = (X[:, 0] < 0.5).astype(float)
    S = uw.discretisation.IntegrationPointVariable("S", mesh)
    S.data[:, 0] = (np.asarray(S.coords)[:, 0] < 0.5).astype(float)
    err = uw.maths.Integral(mesh, (M.sym[0] - S.sym[0]) ** 2).evaluate()
    assert err < 1e-14, err


def test_cells_proxy_vector_variable_and_lagrangian_swarm():
    mesh = _mesh()
    swarm = uw.swarm.Swarm(mesh)
    V = uw.swarm.SwarmVariable("V", swarm, vtype=uw.VarType.VECTOR, proxy_location="cells", proxy_degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords) @ np.array([1.0, 2.0])
    lag = uw.systems.ddt.Lagrangian_Swarm(
        swarm=swarm, psi_fn=T.sym, vtype=uw.VarType.SCALAR, degree=1, continuous=False,
        order=1, proxy_location="cells",
    )
    swarm.populate(fill_param=3)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        V.data[:, 0] = X[:, 0]
        V.data[:, 1] = 2.0 * X[:, 1]
    x, y = mesh.X
    err = uw.maths.Integral(mesh, (V.sym[0] - x) ** 2 + (V.sym[1] - 2 * y) ** 2).evaluate()
    assert err < 1e-14, err
    lag.update_pre_solve(dt=0.1)
    slot = lag.psi_star[0]
    assert not slot._meshVar.continuous
    err = uw.maths.Integral(mesh, (slot.sym[0] - (x + 2 * y)) ** 2).evaluate()
    assert err < 1e-14, err


def test_lagrangian_swarm_flip_update_keeps_particle_values_for_a_resolved_field():
    """FLIP read-back: the particle takes the mesh INCREMENT (solution minus
    the proxy the mesh saw). For a field the proxy resolves exactly the
    increment is zero, so the particle values are untouched by a solve that
    reproduces the field; PIC would re-sample them."""
    mesh = _mesh()
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.data[:, 0] = np.asarray(T.coords) @ np.array([1.0, 2.0]) + 0.5
    swarm = uw.swarm.Swarm(mesh)
    lag = uw.systems.ddt.Lagrangian_Swarm(
        swarm=swarm, psi_fn=T.sym, vtype=uw.VarType.SCALAR, degree=2, continuous=False,
        order=1, step_averaging=1, proxy_location="cells", particle_update="flip",
    )
    swarm.populate(fill_param=3)
    lag.update_pre_solve(dt=0.1)
    Xp = np.asarray(swarm._particle_coordinates.data)
    before = np.array(lag.psi_star[0].data[:, 0], copy=True)
    assert np.allclose(before, Xp @ np.array([1.0, 2.0]) + 0.5, atol=1e-10)
    # Perturb the particle values by a sub-cell "residual" the P2 proxy cannot
    # hold, then run the post-solve update with the mesh field unchanged.
    rng = np.random.default_rng(1)
    noise = 1e-3 * rng.standard_normal(before.shape[0])
    with uw.synchronised_array_update():
        lag.psi_star[0].data[:, 0] = before + noise
    lag.update_pre_solve(dt=0.1)          # proxy refits from the noisy particles
    lag.update_post_solve(dt=0.1)
    after = np.asarray(lag.psi_star[0].data[:, 0])
    # solution (T, exact linear) - proxy (linear + fit of the noise): the
    # residual survives up to the part of the noise the fit absorbed.
    assert np.abs(after - (before + noise)).max() < 5e-3
    assert np.abs(after - before).max() > 1e-4     # PIC would have given `before` back
    with pytest.raises(ValueError, match="particle_update"):
        uw.systems.ddt.Lagrangian_Swarm(swarm=swarm, psi_fn=T.sym, vtype=uw.VarType.SCALAR,
                                        degree=1, continuous=False, proxy_location="cells", particle_update="xx")


def test_lagrangian_swarm_history_is_sampled_before_the_first_move():
    """The first advection samples the history at the launch positions. Left
    to the first solve, the sampling would see the landed positions and the
    first step would transport nothing (a one-step lag)."""
    mesh = _mesh()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]                  # psi = x
    swarm = uw.swarm.Swarm(mesh)
    lag = uw.systems.ddt.Lagrangian_Swarm(
        swarm=swarm, psi_fn=T.sym, vtype=uw.VarType.SCALAR, degree=2, continuous=False,
        order=1, proxy_location="cells",
    )
    swarm.populate(fill_param=2)
    assert not lag._history_initialised
    X_before = np.array(swarm._particle_coordinates.data, copy=True)
    swarm.advection(sympy.Matrix([[0.1, 0.0]]), 0.5, order=2)   # every particle moves +0.05 in x
    X_after = np.asarray(swarm._particle_coordinates.data)
    assert lag._history_initialised
    kept = np.abs(X_after[:, 0] - X_before[:, 0] - 0.05) < 1e-12  # particles not returned to bounds
    vals = np.asarray(lag.psi_star[0].data[:, 0])
    assert np.allclose(vals[kept], X_before[kept, 0], atol=1e-10)   # launch positions ...
    assert not np.allclose(vals[kept], X_after[kept, 0], atol=1e-3) # ... not landing positions
