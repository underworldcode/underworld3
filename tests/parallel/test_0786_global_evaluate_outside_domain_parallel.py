"""``global_evaluate`` refuses points far outside the mesh on every rank (#786).

The refusal counts the far points across ranks, so a rank whose own queries are
all inside still raises with the others, and no rank is left in a collective.
Points inside the domain, including those owned by another rank, are answered.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(240, method="thread")]


@pytest.fixture(scope="module")
def field():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.125
    )
    T = uw.discretisation.MeshVariable("T_786p", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords_nd)[:, 0]
    return T


@pytest.mark.skipif(uw.mpi.size < 2, reason="needs at least two ranks")
def test_a_far_point_on_one_rank_raises_on_every_rank(field):
    pts = np.array([[0.25, 0.5], [0.75, 0.5]])
    if uw.mpi.rank == 0:
        pts = np.vstack([pts, [[6.0e4, -3.0e4]]])
    with pytest.raises(ValueError, match="outside"):
        uw.function.global_evaluate(field.sym[0], pts)
    assert uw.mpi.comm.allgather(True) == [True] * uw.mpi.size


@pytest.mark.skipif(uw.mpi.size < 2, reason="needs at least two ranks")
def test_points_inside_are_answered_wherever_they_are_owned(field):
    pts = np.array([[0.1, 0.1], [0.9, 0.9], [1.0, 0.5], [0.5, 0.0]])
    vals = np.asarray(uw.function.global_evaluate(field.sym[0], pts)).ravel()
    assert vals == pytest.approx(pts[:, 0], abs=1e-12)


@pytest.mark.parametrize("kind", ["scalar_derivative", "gradient"])
def test_a_derivative_at_a_far_point_is_refused_at_any_rank_count(kind):
    """#999's case: P2 V = (x^2, xy), one of three points outside the mesh. It used
    to raise NotImplementedError at np >= 2 for the gradient, and give 2.000, 1.830
    and 1.834 at np 1, 2 and 3 for dVx/dx at the outside point. Refused with the
    same error at every rank count now."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
    )
    x, y = mesh.X
    V = uw.discretisation.MeshVariable(f"V_999_{kind}", mesh, 2, degree=2)
    c = np.asarray(V.coords_nd)
    V.data[:, 0] = c[:, 0] ** 2
    V.data[:, 1] = c[:, 0] * c[:, 1]
    expr = V.sym[0].diff(x) if kind == "scalar_derivative" else V.sym.jacobian(mesh.X[:2])
    pts = np.array([[0.3, 0.4], [0.6, 0.7], [1.5, 0.5]])
    with pytest.raises(ValueError, match="outside"):
        uw.function.global_evaluate(expr, pts)
    assert uw.mpi.comm.allgather(True) == [True] * uw.mpi.size
