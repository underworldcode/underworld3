"""``global_evaluate`` returns the expression's value, however small, on every
rank (UNFILED 42; the serial cases are in tests/test_1295_*).

The evaluators used to add :math:`10^{-18} x y` to every expression to make a
matrix's entries alike under lambdify; on a value of order :math:`10^{-19}`
that term was most of the answer. ``T`` holds :math:`x + 2y` on a degree-1
variable, so the exact value at any point inside the mesh is known; each rank
asks for its own share of the points.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.mpi(min_size=2),
              pytest.mark.timeout(300, method="thread")]

POINTS = np.array([[0.5, 0.5], [0.25, 0.75], [0.9, 0.1], [0.1, 0.2], [0.6, 0.3]])


def test_small_values_on_every_rank():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    T = uw.discretisation.MeshVariable("T1295p", mesh, 1, degree=1)
    coords = np.asarray(T.coords)
    T.array[:, 0, 0] = coords[:, 0] + 2.0 * coords[:, 1]
    x, y = mesh.X
    mine = POINTS[uw.mpi.rank::uw.mpi.size]
    field = uw.function.global_evaluate(sympy.Float(1.0e-19) * T.sym[0], mine)
    np.testing.assert_allclose(field[:, 0, 0], 1.0e-19 * (mine[:, 0] + 2.0 * mine[:, 1]),
                               rtol=1e-12, atol=0)
    coordinate = uw.function.global_evaluate(sympy.Float(1.0e-30) * x, mine)
    np.testing.assert_allclose(coordinate[:, 0, 0], 1.0e-30 * mine[:, 0], rtol=1e-12, atol=0)
    mixed = uw.function.global_evaluate(
        sympy.Matrix([[sympy.Float(1.0e-19) * T.sym[0], sympy.Integer(4)]]), mine)
    assert mixed.shape == (len(mine), 1, 2)
    np.testing.assert_array_equal(mixed[:, 0, 1], np.full(len(mine), 4.0))
