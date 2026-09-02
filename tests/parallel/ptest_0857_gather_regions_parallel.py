"""The region gather behind placement (:func:`place_surface._gather_regions`).

Each marked region's star and layer go to one rank of their own; regions
that touch are merged; a region already interior to a rank is left where
it is (#670). Deterministic assertions: the region ids and owners are
gathered and identical on every rank. Run:

    mpirun -np 3 python -m pytest tests/parallel/ptest_0857_gather_regions_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.line_cut import _coords
from underworld3.utilities.place_surface import (_gather_regions,
                                                 _shared_point_flags)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(300)]


def _box():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.12, regular=False, qdegree=1)


def _vertex_ids(mesh, rule):
    """Chart-length region ids from a rule on vertex coordinates."""
    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]        # DM vertex order, as the placer reads it
    ids = np.zeros(pEnd - pStart, dtype=np.int32)
    ids[vS - pStart: vE - pStart] = rule(X)
    return ids


def _same_everywhere(comm, value):
    return all(v == value for v in comm.allgather(value))


def test_two_far_regions_keep_their_own_owners():
    comm = uw.mpi.comm
    mesh = _box()
    ids = _vertex_ids(mesh, lambda X: np.where(
        X[:, 0] < 0.15, 1, np.where(X[:, 0] > 0.85, 2, 0)))
    new, n_region, n_moved, owner, canon = _gather_regions(mesh.dm, ids)
    assert _same_everywhere(comm, (n_region, n_moved, owner, canon))
    assert canon == {1: 1, 2: 2}, "two regions a domain apart were merged"
    assert set(owner) == {1, 2}
    assert n_region > 0 and n_moved <= n_region
    # the moved cells are exactly those a region claimed away from its rank
    cS, cE = new.getHeightStratum(0)
    n_cells = int(comm.allreduce(cE - cS, op=MPI.SUM))
    cS0, cE0 = mesh.dm.getHeightStratum(0)
    assert n_cells == int(comm.allreduce(cE0 - cS0, op=MPI.SUM))


def test_touching_regions_are_merged():
    comm = uw.mpi.comm
    mesh = _box()
    ids = _vertex_ids(mesh, lambda X: np.where(
        X[:, 0] < 0.45, 1, np.where(X[:, 0] < 0.55, 2, 0)))
    _new, n_region, _n_moved, owner, canon = _gather_regions(mesh.dm, ids)
    assert _same_everywhere(comm, (owner, canon))
    assert canon == {1: 1, 2: 1}, canon
    assert set(owner) == {1}


def test_an_interior_region_is_not_moved():
    """Mark one vertex per rank, deep inside that rank's own cells: every
    region's star and layer are interior, so nothing moves at all. An
    elongated box, so each rank's piece is many cells deep and the vertex
    farthest from any shared point has a two-cell neighbourhood of its
    own."""
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(4.0, 1.0, 1.0),
        cellSize=0.15, regular=False, qdegree=1)
    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    # shared = roots AND leaves: a seam vertex this rank owns is a root,
    # absent from its own leaf list, and its star is mostly elsewhere
    shared = np.asarray(_shared_point_flags(dm)).astype(bool)
    shared_v = np.flatnonzero(shared[vS - pStart: vE - pStart])
    X = _coords(dm)[: vE - vS]        # DM vertex order, as the placer reads it
    ids = np.zeros(pEnd - pStart, dtype=np.int32)
    if len(X):
        if shared_v.size:
            d = np.min(np.linalg.norm(
                X[:, None, :] - X[shared_v][None, :, :], axis=2), axis=1)
        else:
            d = np.linalg.norm(X - X.mean(axis=0), axis=1)
        v = int(np.argmax(d))
        ids[v + vS - pStart] = comm.rank + 1
    _new, n_region, n_moved, owner, canon = _gather_regions(dm, ids)
    assert _same_everywhere(comm, (n_region, n_moved, owner, canon))
    assert n_region > 0
    assert n_moved == 0, (n_moved, owner, canon)
    for k, r in owner.items():
        assert r == k - 1, (owner, "a region's owner is not the rank "
                            "that marked it")
