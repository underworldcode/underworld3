"""``global_evaluate`` of a derivative at a point just outside the mesh gives the
serial value at any rank count (#999).

At np >= 2 the points no rank holds go to an out-of-domain fallback, which
evaluated the original expression, derivatives still in it, by RBF (Clement
recovery) where serial projects: ``dVx/dx`` at an outside point read 2.000,
1.830 and 1.834 at np 1, 2 and 3, and a matrix-valued derivative raised
NotImplementedError. The derivatives are now resolved once, on every rank, into
a projected work variable before the split, as ``evaluate_nd`` does serially.
Points far outside are refused (#786); these are within the mesh's largest cell.
"""
import numpy as np
import pytest

import underworld3 as uw
from serial_reference import emit, serial_reference

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread")]

POINTS = np.array([[0.3, 0.4], [0.6, 0.7], [1.03, 0.5], [0.5, -0.03]])


def _values():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2)
    x, y = mesh.X
    V = uw.discretisation.MeshVariable("V_999", mesh, 2, degree=2)
    c = np.asarray(V.coords_nd)
    V.data[:, 0] = c[:, 0] ** 2
    V.data[:, 1] = c[:, 0] * c[:, 1]
    grad = np.asarray(uw.function.global_evaluate(V.sym.jacobian(mesh.X[:2]), POINTS))
    dvx = np.asarray(uw.function.global_evaluate(V.sym[0].diff(x), POINTS)).reshape(-1)
    return grad.reshape(len(POINTS), 2, 2), dvx


def test_a_derivative_just_outside_matches_serial():
    grad, dvx = _values()
    reference = serial_reference(__file__, "derivatives")
    expected = np.asarray(reference["values"])
    got = np.concatenate([grad.reshape(-1), dvx])
    assert grad.shape == (len(POINTS), 2, 2)
    # The projection's iterative solve converges to about 1e-6 relative, which
    # depends on the partition; RBF against projection differed by about 8%.
    assert got == pytest.approx(expected, rel=1e-4, abs=1e-6), (got, expected)


if __name__ == "__main__":
    # The single-rank child of test_a_derivative_just_outside_matches_serial
    # (see serial_reference): `python <thisfile> derivatives`.
    import sys
    assert sys.argv[1:] == ["derivatives"], sys.argv
    _grad, _dvx = _values()
    emit(np.concatenate([_grad.reshape(-1), _dvx]), [len(POINTS), 0])
