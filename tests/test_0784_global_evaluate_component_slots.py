"""``global_evaluate`` and the RBF route return each component of a variable from its
own stored column.

``_data_layout`` stores a MATRIX by column. ``global_evaluate`` passed its result
through a MATRIX swarm variable's ``array``, whose write went by row and whose read
went by column, so a non-symmetric result came back transposed:
``[[1, 2], [3, 4]]`` read ``[[1, 3], [2, 4]]``. The RBF route, which
``global_evaluate`` takes for a point no rank holds, looked a component up by its
position in the symbol rather than its stored column, which transposes a MATRIX too.
A TENSOR and a symmetric tensor store by row, so neither was affected.
"""
import numpy as np
import pytest
import sympy

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
    """The TENSOR is the control: stored by row, it was never transposed."""
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
    did not. The TENSOR and symmetric tensor are controls."""
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
