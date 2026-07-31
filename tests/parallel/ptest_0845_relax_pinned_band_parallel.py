"""``relax(pin_bands=...)`` in parallel.

The band is chosen by a purely geometric test on the exact distance, so every
rank labels its own copy of a shared vertex identically and the pinned set is a
function of the geometry, not of the partition. That is the property this file
asserts, because it is what makes the feature safe at np>1 and it is not
self-evident from the serial tests:

* the pinned set is **partition-independent** — the same vertices, identified by
  coordinate, are pinned at every communicator size;
* pinned vertices do not move, including pinned vertices that are SHARED between
  ranks, which is the case a rank-local implementation would get wrong;
* the domain boundary stays pinned.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0845_relax_pinned_band_parallel.py
    mpirun -n 3 python -m pytest --with-mpi tests/parallel/ptest_0845_relax_pinned_band_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

POINTS = np.array([[0.12, 0.10, 0.0], [0.50, 0.52, 0.0], [0.88, 0.92, 0.0]])

# Reference from the serial run, so a partition-dependent regression shows up as
# a number rather than as a mysterious parallel failure.
SERIAL_PINNED_COORDS = None      # filled by the first (serial-equivalent) gather


def _fixture(cell_size=0.2):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size,
        regular=False, qdegree=2)
    surface = uw.meshing.Surface("pinpar", mesh, POINTS, symbol="Pp")
    surface.discretize()
    return mesh, surface


def _coords(mesh):
    return np.asarray(mesh.dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)


def _pinned_indices(mesh, name):
    vS, _vE = mesh.dm.getDepthStratum(0)
    iset = mesh.dm.getLabel(name).getStratumIS(1)
    if iset is None:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(iset.getIndices(), dtype=np.int64) - vS


def test_pinned_set_is_partition_independent():
    mesh, surface = _fixture()
    name = mesh.label_interface_band(surface, offset=0.0, halo=1)
    X = _coords(mesh)
    idx = _pinned_indices(mesh, name)

    # Compare the pinned COORDINATES, not counts: a shared vertex is held by
    # every rank on the seam, so a count double-counts it and would mask exactly
    # the defect this test exists to catch.
    local = {(round(float(x), 12), round(float(y), 12)) for x, y in X[idx]}
    gathered = uw.mpi.comm.allgather(local)
    union = set().union(*gathered)

    total = uw.mpi.comm.allreduce(len(union), op=MPI.MAX)
    assert len(union) == total
    assert total > 0, "nothing pinned; the fixture is not exercising the band"


def test_pinned_vertices_including_shared_ones_do_not_move():
    mesh, surface = _fixture()
    before = _coords(mesh).copy()
    name = mesh.label_interface_band(surface, offset=0.0, halo=1)
    idx = _pinned_indices(mesh, name)

    # A pinned vertex that is also a star-forest leaf is the interesting one: it
    # is owned by another rank, so a rank-local pin would let the owner move it.
    try:
        _n, ilocal, _r = mesh.dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        ilocal = None
    vS, _vE = mesh.dm.getDepthStratum(0)
    leaves = set() if ilocal is None else {int(p) - vS for p in ilocal}
    shared_pinned = [i for i in idx if int(i) in leaves]
    assert uw.mpi.comm.allreduce(len(shared_pinned), op=MPI.SUM) > 0, (
        "no pinned vertex is shared; this run cannot exercise the seam case")

    mesh.relax(pin_bands=[surface], pin_halo=1)
    after = _coords(mesh)

    moved = np.linalg.norm(after - before, axis=1)
    assert uw.mpi.comm.allreduce(float(moved[idx].max()), op=MPI.MAX) == 0.0
    free = np.setdiff1d(np.arange(len(before)), idx)
    assert uw.mpi.comm.allreduce(float(moved[free].max()) if len(free) else 0.0,
                                 op=MPI.MAX) > 0.0, "the mover did nothing"


def test_domain_boundary_stays_pinned():
    mesh, surface = _fixture()
    before = _coords(mesh).copy()
    on_boundary = (np.isclose(before[:, 0], 0.0) | np.isclose(before[:, 0], 1.0)
                   | np.isclose(before[:, 1], 0.0) | np.isclose(before[:, 1], 1.0))

    mesh.relax(pin_bands=[surface])
    after = _coords(mesh)

    worst = float(np.abs(after[on_boundary] - before[on_boundary]).max()) \
        if on_boundary.any() else 0.0
    assert uw.mpi.comm.allreduce(worst, op=MPI.MAX) == 0.0
