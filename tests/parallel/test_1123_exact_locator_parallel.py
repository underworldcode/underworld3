"""On #1123's sheared 3-D mesh, every point inside the mesh is owned by some rank at
any rank count, and evaluated exactly in serial.

Development's 50-centroid search missed the containing cell on this mesh: in
parallel 6 to 52 inside points were owned by no rank (measured by H1 on #1175) and
came back from extrapolation. The exact locator walks to the containing cell.

Inside is decided without the library's locator: each rank tests every point
against every one of its own cells by barycentric coordinates in numpy, and the
answers are reduced over ranks.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1123_exact_locator_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _linear(P):
    return 2 * P[:, 0] + 3 * P[:, 1] - P[:, 2]


def _inside_some_local_cell(mesh, P):
    dm = mesh.dm
    cs, ce = dm.getHeightStratum(0)
    vs, ve = dm.getDepthStratum(0)
    if ce == cs:
        return np.zeros(P.shape[0], dtype=bool)
    X = np.asarray(mesh._coords).reshape(-1, 3)
    V = X[np.array([[q - vs for q in dm.getTransitiveClosure(c)[0] if vs <= q < ve]
                    for c in range(cs, ce)])]
    Ti = np.linalg.inv(np.transpose(V[:, 1:, :] - V[:, :1, :], (0, 2, 1)))
    inside = np.zeros(P.shape[0], dtype=bool)
    for start in range(0, P.shape[0], 256):
        Q = P[start:start + 256]
        lam = np.einsum("cij,ncj->nci", Ti, Q[:, None, :] - V[None, :, 0, :])
        smallest = np.minimum(lam.min(axis=2), 1.0 - lam.sum(axis=2))
        inside[start:start + 256] = (smallest >= 0.0).any(axis=1)
    return inside


def test_every_inside_point_is_owned_and_exact():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0., 0., 0.), maxCoords=(1., 1., 1.),
                                             cellSize=1 / 6, qdegree=3)
    X = np.asarray(mesh.X.coords).copy()
    A = np.array([[1.0, 1.2, 0.3], [0.1, 0.7, 0.5], [0.0, 0.2, 0.5]])
    Y = X @ A.T
    Y[:, 0] += 0.03 * np.sin(2 * np.pi * X[:, 1])
    Y[:, 1] += 0.03 * np.sin(2 * np.pi * X[:, 2])
    Y[:, 2] += 0.03 * np.sin(2 * np.pi * X[:, 0])
    mesh.deform(Y)
    T = uw.discretisation.MeshVariable("T1123p", mesh, 1, degree=1)
    T.array[:, 0, 0] = _linear(np.asarray(T.coords))

    # points 1e-8 to 1e-4 off every interior edge midpoint, the same on every rank
    dm = mesh.dm
    es, ee = dm.getDepthStratum(1)
    vs = dm.getDepthStratum(0)[0]
    Xv = np.asarray(mesh._coords).reshape(-1, 3)
    mids = np.array([0.5 * (Xv[dm.getCone(e)[0] - vs] + Xv[dm.getCone(e)[1] - vs])
                     for e in range(es, ee)]).reshape(-1, 3)
    mids = np.unique(np.round(np.vstack(uw.mpi.comm.allgather(mids)), 13), axis=0)
    rng = np.random.default_rng(3)
    P = np.vstack([mids + d * (lambda n: n / np.linalg.norm(n, axis=1)[:, None])(rng.normal(size=mids.shape))
                   for d in (1e-8, 1e-6, 1e-4)])

    inside = uw.mpi.comm.allreduce(_inside_some_local_cell(mesh, P).astype(np.int64), op=MPI.SUM) > 0
    value, extrapolated = uw.function.global_evaluate(T.sym[0], P, check_extrapolated=True)
    value = np.asarray(value).ravel()
    extrapolated = np.asarray(extrapolated).ravel().astype(bool)
    unowned = int((inside & extrapolated).sum())
    assert unowned == 0, f"{unowned} inside points owned by no rank (filled by extrapolation)"

    # Exact values are asserted serially. In parallel a point within the face
    # tolerance of a partition seam can be evaluated by the rank whose cell does
    # not contain it (up to 1.1e-3 here at np 4): the seam claim, left open on
    # #1175 for the follow-up that claims by exact containment first.
    if uw.mpi.size == 1:
        error = float(np.abs(value - _linear(P))[inside].max(initial=0.0))
        assert error < 1e-10, error


def test_a_rank_with_no_cells_locates_nothing_instead_of_raising():
    """On a simplex mesh a rank with no cells gets an empty exact locator: it
    returns -1 for every point. Development raised there ("An error was
    encountered attempting to find the closest cells") from the empty kd-tree."""
    if uw.mpi.size < 3:
        pytest.skip("a shell partition leaves a rank empty from np 3 (empty_rank_fixtures)")
    from empty_rank_fixtures import cells_along_a_side_leaving_one_rank_empty, right_triangle_of_cells

    n = cells_along_a_side_leaving_one_rank_empty(uw.mpi.size)
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(n, uw.mpi.size - 1), degree=1, qdegree=2)
    P = np.random.default_rng(0).uniform(0.0, 0.5, size=(50, 2))
    message = ""
    try:
        cells = np.asarray(mesh._robust_owning_cells(P))
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        cells = np.full(P.shape[0], -1)
    messages = uw.mpi.comm.allgather(message)
    assert not any(messages), messages
    owned = uw.mpi.comm.allreduce((cells >= 0).astype(np.int64), op=MPI.SUM)
    assert (owned >= 1).all(), "a point inside the triangle is owned by no rank"
