"""The exact simplex locator: barycentric containment and a face-crossing walk (#1123).

On a strongly sheared 3-D simplex mesh the old locator tested only the 50 cells
with the nearest centroids, and the containing cell can rank farther down: points
1e-8 to 1e-4 from interior edges were classified exterior and filled by RBF, wrong
by up to 0.18 for a field the element represents exactly (10 to 18 of about 1260
points at each distance). The walk crosses the face a point fails worst until a
cell contains it, so the containing cell is found whatever its centroid's rank.

A point several cells contain gets the lowest-numbered local cell in the star of
the entity it lies on, whatever cell the walk started from.

The truth here is a brute-force barycentric test over every cell, in numpy.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _cells_and_vertices(mesh):
    dm = mesh.dm
    cs, ce = dm.getHeightStratum(0)
    vs, ve = dm.getDepthStratum(0)
    X = np.asarray(mesh._coords).reshape(-1, mesh.cdim)
    cellv = np.array([[q - vs for q in dm.getTransitiveClosure(c)[0] if vs <= q < ve]
                      for c in range(cs, ce)])
    return X, cellv


def _min_barycentric(P, V):
    """Smallest barycentric coordinate of each point in each simplex (n, ncell)."""
    d = P.shape[1]
    T = np.transpose(V[:, 1:, :] - V[:, :1, :], (0, 2, 1))
    Ti = np.linalg.inv(T)
    lam = np.einsum("cij,ncj->nci", Ti, P[:, None, :] - V[None, :, 0, :])
    return np.minimum(lam.min(axis=2), 1.0 - lam.sum(axis=2))


@pytest.fixture(scope="module")
def sheared_box():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0., 0., 0.), maxCoords=(1., 1., 1.),
                                             cellSize=1 / 6, qdegree=3)
    X = np.asarray(mesh.X.coords).copy()
    A = np.array([[1.0, 1.2, 0.3], [0.1, 0.7, 0.5], [0.0, 0.2, 0.5]])
    Y = X @ A.T
    Y[:, 0] += 0.03 * np.sin(2 * np.pi * X[:, 1])
    Y[:, 1] += 0.03 * np.sin(2 * np.pi * X[:, 2])
    Y[:, 2] += 0.03 * np.sin(2 * np.pi * X[:, 0])
    mesh.deform(Y)
    T = uw.discretisation.MeshVariable("T1123", mesh, 1, degree=1)
    T.array[:, 0, 0] = _linear(np.asarray(T.coords))
    return mesh, T


def _linear(P):
    return 2 * P[:, 0] + 3 * P[:, 1] - P[:, 2]


@pytest.mark.parametrize("distance", [1e-8, 1e-6, 1e-4])
def test_points_near_interior_edges_are_exact(sheared_box, distance):
    if uw.mpi.size > 1:
        pytest.skip("serial: the truth is a brute-force test over the whole mesh")
    mesh, T = sheared_box
    dm = mesh.dm
    es, ee = dm.getDepthStratum(1)
    X, cellv = _cells_and_vertices(mesh)
    vs = dm.getDepthStratum(0)[0]
    mids = np.array([0.5 * (X[dm.getCone(e)[0] - vs] + X[dm.getCone(e)[1] - vs])
                     for e in range(es, ee)])
    rng = np.random.default_rng(3)
    d = rng.normal(size=mids.shape)
    P = mids + distance * d / np.linalg.norm(d, axis=1)[:, None]

    inside = _min_barycentric(P, X[cellv]).max(axis=1) >= 0.0
    value, exterior = uw.function.evaluate(T.sym[0], P, check_extrapolated=True)
    error = np.abs(np.asarray(value).ravel() - _linear(P))
    exterior = np.asarray(exterior).ravel()

    assert not exterior[inside].any(), (
        f"{int(exterior[inside].sum())} points inside the mesh were classified exterior")
    assert error[inside].max() < 1e-10, error[inside].max()


def test_a_point_on_a_vertex_gets_the_lowest_cell_of_its_star():
    if uw.mpi.size > 1:
        pytest.skip("serial: local cell numbers")
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8)
    locator = mesh._simplex_locator()
    assert locator is not None
    dm = mesh.dm
    cs, ce = dm.getHeightStratum(0)
    vs, ve = dm.getDepthStratum(0)
    X = np.asarray(mesh._coords).reshape(-1, 2)
    expected = np.array([min(q for q in dm.getTransitiveClosure(v, useCone=False)[0] if cs <= q < ce)
                         for v in range(vs, ve)])
    cells = np.asarray(mesh._robust_owning_cells(X))
    assert np.array_equal(cells, expected)
    # whatever the walk starts from
    for start in (np.zeros(len(X), dtype=np.int64), np.full(len(X), ce - cs - 1, dtype=np.int64)):
        assert np.array_equal(locator.locate(X, start) + cs, expected)


def test_the_walk_finds_every_contained_point_on_a_non_convex_boundary(sheared_box):
    """On #1123's mesh the most violated face of a cell near the wavy boundary can
    be a boundary face although the point is inside, further along the wall. The
    walk crosses the most violated face that has a neighbour instead, and finds
    every contained point from its start cell alone (it lost 7 of 2e5 when it left
    through the boundary)."""
    if uw.mpi.size > 1:
        pytest.skip("serial: local cells")
    mesh, _ = sheared_box
    locator = mesh._simplex_locator()
    X, cellv = _cells_and_vertices(mesh)
    rng = np.random.default_rng(9)
    cells = rng.integers(0, cellv.shape[0], 200000)
    P = np.einsum("nk,nkd->nd", rng.dirichlet(np.ones(4), cells.shape[0]), X[cellv][cells])
    _, nearest = mesh._index.query(P, k=1, sqr_dists=False)
    start = mesh._indexMap[np.asarray(nearest).reshape(-1)] - mesh.dm.getHeightStratum(0)[0]
    found, _, _ = locator.walk(P, start)
    assert (found >= 0).all(), f"{int((found < 0).sum())} contained points not found by the walk"
