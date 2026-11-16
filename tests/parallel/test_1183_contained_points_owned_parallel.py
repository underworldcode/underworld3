"""Every point a local cell contains is owned by that rank, on #1123's sheared mesh (#1183).

The face-crossing walk, its restarts from the nearest centroids and the
face-tolerance search could all miss a contained point on a strongly sheared mesh:
2 per million at np 2 on the old gmsh fixture (H1's probe on #1175). The exact
fallback tests every local cell whose bounding box holds a point the walk did not
place, so none is missed.

Those 2 per million sat in near-degenerate cells of one gmsh build (worst volume
ratio 0.027 at the old wave amplitude 0.03, which can turn a cell over on another
gmsh build). No valid fixture reproduces them reliably: on every non-tangling shear
tried, gmsh or structured, the walk misses none. The regression is therefore tested
directly: with the walk forced to miss every point, the fallback must find them all
(before #1190 the 50-centroid search lost 1200 to 1427 of 2e4 on this fixture).

Each rank samples 5e5 points inside its own cells (a quarter within 1e-9 of a face,
edge or vertex), so each is contained by a local cell by construction.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1183_contained_points_owned_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

import sheared_box_fixture
from underworld3.utilities._plex_topology import closure_points

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def test_every_contained_point_is_owned_on_the_sheared_mesh():
    mesh = sheared_box_fixture.sheared_box()

    cs, ce = mesh.dm.getHeightStratum(0)
    n = ce - cs
    rng = np.random.default_rng(100 + uw.mpi.rank)
    N = 1000000 // uw.mpi.size
    if n:
        Xl = np.asarray(mesh._coords).reshape(-1, 3)
        V = Xl[mesh._coord_rows_for_points(
            mesh.dm, closure_points(mesh.dm, np.arange(cs, ce), 0).ravel())].reshape(n, 4, 3)
        cells = rng.integers(0, n, N)
        w = rng.dirichlet(np.ones(4), N)
        near = slice(0, N // 4)
        w[near] = np.where(w[near] < 0.3, 1e-9 * rng.random(w[near].shape), w[near])
        w /= w.sum(axis=1, keepdims=True)
        P = np.einsum("nk,nkd->nd", w, V[cells])
    else:
        P = np.zeros((0, 3))

    owned = np.asarray(mesh._robust_owning_cells(P))
    lost = uw.mpi.comm.allreduce(int((owned < 0).sum()), op=MPI.SUM)
    assert lost == 0, f"{lost} contained points owned by no rank"


def test_the_fallback_grid_is_capped_on_a_graded_mesh():
    """Most cells packed into a corner of a large box, as on a mesh refined
    toward a fault: each axis maps [0, 0.9] onto [0, 0.01] (monotone, so no cell
    turns over), so about 73 % of the cells are about 1e-3 across in a box of
    extent 1. An uncapped grid at the median cell size would hold about 8e7 bins
    on one rank. The grid is capped at the local cell count, and every contained
    point is still found."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0., 0., 0.), maxCoords=(1., 1., 1.),
                                             cellSize=1 / 8, qdegree=2)
    X = np.asarray(mesh.X.coords).copy()
    mesh.deform(np.where(X < 0.9, X * (0.01 / 0.9), 0.01 + (X - 0.9) * (0.99 / 0.1)))

    locator = mesh._simplex_locator()
    if locator.n_cells == 0:
        # still takes part in the premise's reduction below
        assert uw.mpi.comm.allreduce(0, op=MPI.MAX) > 0
        return
    cs = mesh.dm.getHeightStratum(0)[0]
    Xl = np.asarray(mesh._coords).reshape(-1, 3)
    V = Xl[mesh._coord_rows_for_points(
        mesh.dm, closure_points(mesh.dm, np.arange(cs, cs + locator.n_cells), 0).ravel())
    ].reshape(locator.n_cells, 4, 3)
    rng = np.random.default_rng(5 + uw.mpi.rank)
    cells = rng.integers(0, locator.n_cells, 20000)
    P = np.einsum("nk,nkd->nd", rng.dirichlet(np.ones(4), cells.shape[0]), V[cells])

    found = locator.cells_containing(P)
    # The premise holds on the ranks that hold the packed corner (all of it in
    # serial); a rank with only the large outer cells needs no cap.
    over = locator.grid_bins_uncapped - locator.n_cells
    assert uw.mpi.comm.allreduce(over, op=MPI.MAX) > 0, "premise: no rank would exceed the cap"
    assert locator.grid_bins <= locator.n_cells, (locator.grid_bins, locator.n_cells)
    assert (found >= 0).all(), f"{int((found < 0).sum())} contained points not found"


def test_the_fallback_finds_every_point_the_walk_misses():
    """#1183's regression, on a valid mesh: with the walk forced to miss every
    point, each contained point must still be found, and in a cell containing
    it. On #1123's sheared mesh the 50-nearest-centroid search that was the only
    fallback before #1190 misses thousands of them; the bin grid finds all.
    (The 2 per million the walk itself missed lived in near-degenerate cells of
    one gmsh build, which no valid fixture reproduces reliably.)"""
    mesh = sheared_box_fixture.sheared_box()
    locator = mesh._simplex_locator()
    cs, ce = mesh.dm.getHeightStratum(0)
    n = ce - cs
    rng = np.random.default_rng(31 + uw.mpi.rank)
    if n:
        Xl = np.asarray(mesh._coords).reshape(-1, 3)
        V = Xl[mesh._coord_rows_for_points(
            mesh.dm, closure_points(mesh.dm, np.arange(cs, ce), 0).ravel())].reshape(n, 4, 3)
        cells = rng.integers(0, n, 20000)
        P = np.einsum("nk,nkd->nd", rng.dirichlet(np.ones(4), cells.shape[0]), V[cells])
    else:
        P = np.zeros((0, 3))

    def misses(points, start, restarts=(), return_exit=False):
        lost = np.full(np.asarray(points).shape[0], -1, dtype=np.int64)
        return (lost, lost.copy()) if return_exit else lost

    locator.locate = misses
    found = np.asarray(mesh._robust_owning_cells(P))
    lost = uw.mpi.comm.allreduce(int((found < 0).sum()), op=MPI.SUM)
    assert lost == 0, f"{lost} contained points not found once the walk missed them"
    if n:
        lam = locator.barycentric(P, found - cs)
        assert lam.min() >= -1e-12, lam.min()
