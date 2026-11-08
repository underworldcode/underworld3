"""``global_evaluate`` and the RBF route return each component of a variable from its
own stored column.

``_data_layout`` stores a MATRIX by column. ``global_evaluate`` passes every result,
whatever its source, through a MATRIX swarm variable. Its ``array`` wrote by row and
read by column, so every non-symmetric 2-D result came back transposed:
``[[1, 2], [3, 4]]`` read ``[[1, 3], [2, 4]]``. That includes a TENSOR variable, a
velocity gradient and any non-symmetric expression, not only a MATRIX variable.

The RBF route, which ``global_evaluate`` takes for a point no rank holds, looked a
component up by its position in the symbol rather than its stored column. That
transposes a MATRIX variable. A TENSOR and a symmetric tensor store by row, so on
that route alone they were right.

Semi-Lagrangian history samples its field through ``global_evaluate``, so a
non-symmetric TENSOR history held the transpose at every step, and it is written
back one stored column at a time by ``ddt._write_evaluated``.
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(cellSize=0.25, regular=True, qdegree=3)


@pytest.mark.parametrize("matrix", [[[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]],
                         ids=["square", "non_square"])
def test_global_evaluate_returns_a_non_symmetric_expression_in_its_components(mesh, matrix):
    name = f"one0784_{len(matrix[0])}"
    one = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)
    one.array[:, 0, 0] = 1.0
    expression = sympy.Matrix(matrix) * one.sym[0]
    values = np.asarray(uw.function.global_evaluate(expression, np.array([[0.3, 0.3]])))[0]
    assert np.allclose(values, matrix, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("shape, vtype", [((2, 2), uw.VarType.MATRIX), ((2, 3), uw.VarType.MATRIX),
                                          ((3, 2), uw.VarType.MATRIX), ((2, 2), uw.VarType.TENSOR)])
def test_global_evaluate_returns_a_variable_in_its_components(mesh, shape, vtype):
    """The TENSOR is not a control: ``global_evaluate`` transposed it as well."""
    name = f"ge_{vtype.name}_{shape[0]}{shape[1]}"
    var = uw.discretisation.MeshVariable(name, mesh, shape, degree=1, vtype=vtype)
    expected = np.array([[1.0 + i + 2.0 * j for j in range(shape[1])] for i in range(shape[0])])
    for i in range(shape[0]):
        for j in range(shape[1]):
            var.array[:, i, j] = expected[i, j]
    values = np.asarray(uw.function.global_evaluate(var.sym, np.array([[0.3, 0.3]])))[0]
    assert np.allclose(values, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("shape, vtype", [((2, 2), uw.VarType.MATRIX), ((2, 3), uw.VarType.MATRIX),
                                          ((2, 2), uw.VarType.TENSOR), ((2, 2), uw.VarType.SYM_TENSOR)])
def test_the_rbf_path_finds_each_component_in_its_stored_column(mesh, shape, vtype):
    """``rbf_evaluate`` read the column at the component's position in the symbol,
    not its stored column, so it transposed a MATRIX where the finite element route
    did not. The TENSOR and symmetric tensor were right on this route."""
    name = f"rbf_{vtype.name}_{shape[1]}"
    var = uw.discretisation.MeshVariable(name, mesh, shape, degree=1, vtype=vtype)
    symmetric = vtype == uw.VarType.SYM_TENSOR
    expected = np.array([[1.0 + i + j if symmetric else 1.0 + i + 2.0 * j
                          for j in range(shape[1])] for i in range(shape[0])])
    for i in range(shape[0]):
        for j in range(shape[1]):
            var.array[:, i, j] = expected[i, j]
    values = np.asarray(uw.function.evaluate(var.sym, np.array([[0.3, 0.3]]), rbf=True))[0]
    assert np.allclose(values, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("vtype", [uw.VarType.TENSOR, uw.VarType.SYM_TENSOR])
def test_semi_lagrangian_history_keeps_every_component_of_a_tensor(mesh, vtype):
    """With zero velocity the trace-back is the identity, so one history step must
    hold the field exactly, in every stored column."""
    velocity = uw.discretisation.MeshVariable(f"Vel_{vtype.name}", mesh, 2, degree=2,
                                              vtype=uw.VarType.VECTOR)
    velocity.data[:] = 0.0
    field = uw.discretisation.MeshVariable(f"G_{vtype.name}", mesh, (2, 2), degree=1, vtype=vtype)
    expected = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 4.0,
                (1, 0): 3.0 if vtype == uw.VarType.TENSOR else 2.0}
    data = np.array(field.data)
    for (i, j), value in expected.items():
        data[:, field._data_layout(i, j)] = value
    field.data[:] = data

    history = uw.systems.ddt.SemiLagrangian(mesh, field.sym, velocity.sym, vtype=vtype,
                                             degree=1, continuous=True, order=1)
    history.update_pre_solve(0.05)

    stored = np.asarray(history.psi_star[0].data)
    worst = {}
    for (i, j), value in expected.items():
        column = stored[:, history.psi_star[0]._data_layout(i, j)]
        local = float(np.abs(column - value).max(initial=0.0))
        worst[f"{i}{j}"] = uw.mpi.comm.allreduce(local, op=MPI.MAX)
    assert all(e < 1.0e-10 for e in worst.values()), worst


def test_a_matrix_valued_derivative_under_rbf_is_refused(mesh):
    """The RBF route recovers a derivative into one scalar work variable, so a
    matrix-valued derivative came back ``(N, 1, 1)``: ``evaluate`` broadcast its first
    entry and ``global_evaluate`` raised IndexError. It is refused by name; a scalar
    derivative still goes this route, and is exact on a linear field."""
    field = uw.discretisation.MeshVariable("T_rbf0784", mesh, 1, degree=1)
    field.data[:, 0] = 3.0 * np.asarray(field.coords)[:, 0]
    x, y = mesh.X
    gradient = sympy.Matrix([[field.sym[0].diff(x), field.sym[0].diff(y)]])
    points = np.array([[0.3, 0.3], [0.6, 0.2]])

    for call in (lambda: uw.function.evaluate(gradient, points, rbf=True),
                 lambda: uw.function.global_evaluate(gradient, points, mode="fast")):
        with pytest.raises(NotImplementedError, match="rbf"):
            call()

    slope = np.asarray(uw.function.evaluate(field.sym[0].diff(x), points, rbf=True)).reshape(-1)
    assert np.allclose(slope, 3.0, rtol=1e-12, atol=0.0), slope
