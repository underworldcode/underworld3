"""Three tensor-shaped paths that raised (probe pass, development 01dec8d2).

* #995: ``SemiLagrangian(..., vtype=MATRIX)`` built its history stores without
  the MATRIX shape, so their constructor failed on ``None``.
* #984: ``SwarmVariable.read_timestep`` put the file's values into
  ``self.array[:, 0, :]``, which is the stored layout only for scalars and
  vectors: a tensor raised "could not broadcast".
* #1000: ``global_evaluate`` of ONE component with ``monotone="clamp"`` built
  the bound for every component of the variable and failed to reshape.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)


@pytest.mark.tier_b
def test_a_matrix_semi_lagrangian_history_holds_the_field_it_tracks():
    mesh = _mesh()
    M = uw.discretisation.MeshVariable("M1165", mesh, (2, 3), vtype=uw.VarType.MATRIX, degree=1)
    v = uw.discretisation.MeshVariable("v1165", mesh, 2, degree=2)
    X = np.asarray(M.coords)
    for i in range(2):
        for j in range(3):
            M.array[:, i, j] = (1 + i) * X[:, 0] + (1 + j) * X[:, 1]
    history = uw.systems.ddt.SemiLagrangian(mesh, M.sym, v.sym, vtype=uw.VarType.MATRIX,
                                            degree=1, continuous=True)
    assert history.psi_star[0].shape == (2, 3)
    history.initialise_history()
    assert np.abs(np.asarray(history.psi_star[0].array) - np.asarray(M.array)).max() < 1e-12


@pytest.mark.tier_b
@pytest.mark.parametrize("vtype", [uw.VarType.SYM_TENSOR, uw.VarType.TENSOR])
def test_a_tensor_swarm_variable_round_trips_through_write_and_read_timestep(vtype, tmp_path):
    mesh = _mesh()
    swarm = uw.swarm.Swarm(mesh)
    t = uw.swarm.SwarmVariable(f"t1165_{vtype.name}", swarm, vtype=vtype, proxy_degree=1)
    swarm.populate(fill_param=2)
    stored = np.asarray(t.data)
    values = np.arange(stored.size, dtype=float).reshape(stored.shape) / stored.size
    t.data[...] = values
    swarm.write_timestep("sw1165", "sw", swarmVars=[t], outputPath=str(tmp_path), index=0)
    t.data[...] = 0.0
    t.read_timestep("sw1165", "sw", t.name, 0, outputPath=str(tmp_path))
    # the same swarm, so every particle finds its own saved point
    assert np.abs(np.asarray(t.data) - values).max() == 0.0


@pytest.mark.tier_b
@pytest.mark.parametrize("vtype", [uw.VarType.TENSOR, uw.VarType.SYM_TENSOR, uw.VarType.VECTOR])
def test_clamping_one_component_returns_that_component(vtype):
    """A field linear in x, y is exact in P1 and lies inside its neighbour
    range at interior points, so the clamped value is the exact one."""
    mesh = _mesh()
    var = uw.discretisation.MeshVariable(f"c1165_{vtype.name}", mesh, vtype=vtype, degree=1)
    X = np.asarray(var.coords)
    rows, cols = var.sym.shape
    for i in range(rows):
        for j in range(cols):
            if vtype == uw.VarType.SYM_TENSOR and j < i:
                continue
            var.array[:, i, j] = (1 + i + 2 * j) * X[:, 0] + (2 - i) * X[:, 1]
            if vtype == uw.VarType.SYM_TENSOR:
                var.array[:, j, i] = var.array[:, i, j]
    pts = np.array([[0.31, 0.42], [0.63, 0.27], [0.5, 0.5]])
    i, j = (0, 1) if cols > 1 and rows > 1 else (0, cols - 1)
    out = uw.function.global_evaluate(var.sym[i, j], pts, monotone="clamp")
    exact = (1 + i + 2 * j) * pts[:, 0] + (2 - i) * pts[:, 1]
    assert np.abs(np.asarray(out).ravel() - exact).max() < 1e-10
