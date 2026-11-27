"""``evaluate`` returns the expression's value, however small (UNFILED 42).

The evaluators used to add ``zero_matrix(shape)`` to every expression, an entry
of :math:`\\varepsilon x y` with :math:`\\varepsilon = 10^{-18}`
(``uw.maths.functions.vanishing``), so that a matrix mixing constant and
coordinate-dependent entries became one uniform array under lambdify. On a
value of order :math:`10^{-18}` that term is the answer: ``evaluate(1e-19*T)``
came out 3.08 times too large at (0.5, 0.5), and a model-unit energy of about
:math:`10^{-18}` (J/mol under unit scaling) was wrong by a factor of three.

The baselines are exact: ``T`` holds the linear field :math:`x + 2y` on a
degree-1 variable, so its interpolant at a point inside the mesh is that value
to round-off, and a constant or a coordinate is its own value. The shapes are
those of the expression: ``(n, 1, 1)`` for a scalar, ``(n, rows, cols)`` for a
matrix, constant entries broadcast over the points.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]
# evaluate is rank-local (a point outside a rank's cells is extrapolated), so
# these cases are serial; the parallel ones are in tests/parallel/test_1295_*.

POINTS = np.array([[0.5, 0.5], [0.25, 0.75], [0.9, 0.1], [0.1, 0.2]])


@pytest.fixture(scope="module")
def linear_field():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    T = uw.discretisation.MeshVariable("T1295", mesh, 1, degree=1)
    coords = np.asarray(T.coords)
    T.array[:, 0, 0] = coords[:, 0] + 2.0 * coords[:, 1]
    return mesh, T


def _exact(points, scale=1.0):
    return scale * (points[:, 0] + 2.0 * points[:, 1])


@pytest.mark.parametrize("which", ["evaluate", "global_evaluate"])
def test_a_small_multiple_of_a_field_is_its_value(linear_field, which):
    mesh, T = linear_field
    evaluate = getattr(uw.function, which)
    values = evaluate(sympy.Float(1.0e-19) * T.sym[0], POINTS)
    assert values.shape == (len(POINTS), 1, 1)
    np.testing.assert_allclose(values[:, 0, 0], _exact(POINTS, 1.0e-19), rtol=1e-12, atol=0)


@pytest.mark.parametrize("which", ["evaluate", "global_evaluate"])
def test_a_small_multiple_of_a_coordinate_is_its_value(linear_field, which):
    mesh, T = linear_field
    evaluate = getattr(uw.function, which)
    x, y = mesh.X
    values = evaluate(sympy.Float(1.0e-30) * x, POINTS)
    np.testing.assert_allclose(values[:, 0, 0], 1.0e-30 * POINTS[:, 0], rtol=1e-12, atol=0)


@pytest.mark.parametrize("constant", [sympy.Integer(3), sympy.Rational(1, 3),
                                      sympy.Float(2.5e-20)])
def test_a_constant_is_broadcast_over_the_points(linear_field, constant):
    mesh, T = linear_field
    values = uw.function.evaluate(constant, POINTS)
    assert values.shape == (len(POINTS), 1, 1)
    np.testing.assert_array_equal(values[:, 0, 0], np.full(len(POINTS), float(constant)))


def test_a_matrix_mixing_constants_and_fields_keeps_its_shape(linear_field):
    mesh, T = linear_field
    x, y = mesh.X
    vector = sympy.Matrix([[sympy.Float(1.0e-19) * T.sym[0], sympy.Integer(4)]])
    values = uw.function.evaluate(vector, POINTS)
    assert values.shape == (len(POINTS), 1, 2)
    np.testing.assert_allclose(values[:, 0, 0], _exact(POINTS, 1.0e-19), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(values[:, 0, 1], np.full(len(POINTS), 4.0))

    tensor = sympy.Matrix([[x, sympy.Integer(0)], [sympy.Rational(1, 2), sympy.Float(1e-25) * y]])
    values = uw.function.evaluate(tensor, POINTS)
    assert values.shape == (len(POINTS), 2, 2)
    np.testing.assert_allclose(values[:, 0, 0], POINTS[:, 0], rtol=1e-15, atol=0)
    np.testing.assert_array_equal(values[:, 0, 1], np.zeros(len(POINTS)))
    np.testing.assert_array_equal(values[:, 1, 0], np.full(len(POINTS), 0.5))
    np.testing.assert_allclose(values[:, 1, 1], 1e-25 * POINTS[:, 1], rtol=1e-12, atol=0)


def test_a_constant_matrix_is_broadcast_over_the_points(linear_field):
    mesh, T = linear_field
    values = uw.function.evaluate(sympy.Matrix([[1, 2], [3, 4]]), POINTS)
    assert values.shape == (len(POINTS), 2, 2)
    np.testing.assert_array_equal(values, np.broadcast_to([[1.0, 2.0], [3.0, 4.0]],
                                                          (len(POINTS), 2, 2)))


def test_an_expression_in_a_parameter_only_is_broadcast(linear_field):
    mesh, T = linear_field
    t = uw.function.expression(r"t_{1295}", sympy.Float(1.0e-20), "a time")
    values = uw.function.evaluate(sympy.Integer(3) * t, POINTS)
    assert values.shape == (len(POINTS), 1, 1)
    np.testing.assert_allclose(values[:, 0, 0], np.full(len(POINTS), 3.0e-20), rtol=1e-15,
                               atol=0)
