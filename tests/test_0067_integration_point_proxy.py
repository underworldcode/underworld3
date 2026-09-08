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
