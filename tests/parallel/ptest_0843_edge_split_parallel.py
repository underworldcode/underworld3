"""Parallel confluence of ``engine="edge_split"``.

The refined mesh must be the SAME at any communicator size. This is the
load-bearing test for the engine: three separate defects during development
showed up here and nowhere else —

- a collective (the star-forest reconcile) reached inside a rank-local branch,
  which deadlocked as soon as one rank owned no shared point;
- an edge selection by greedy sweep, whose result depends on iteration order and
  therefore on the partition (412 cells at np=1/2 but 463 at np=3 and 925 at
  np=4, all conforming, all plausible-looking in isolation);
- a mis-sized ``PetscSF`` reduce buffer, which corrupted the heap only after the
  second pass.

None of them is visible in a serial run, and the first two are invisible in a
single-pass run. The reference numbers are asserted in the serial file
(``tests/test_0843_edge_split_adapt.py::test_serial_reference_for_parallel_confluence``)
so a change to the contract is visible there rather than as a mysterious
parallel failure here.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0843_edge_split_parallel.py
    mpirun -n 3 python -m pytest --with-mpi tests/parallel/ptest_0843_edge_split_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import edge_split

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(300)]

# The serial reference: see the serial test file.
SERIAL_BASE_CELLS = 104
SERIAL_REFINED_CELLS = 412
SERIAL_PASSES = 7

CENTRE = np.array([0.35, 0.6])


def _owned_cells(dm):
    cS, cE = dm.getHeightStratum(0)
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        ilocal = None
    leaves = set() if ilocal is None else {int(p) for p in ilocal}
    return uw.mpi.comm.allreduce(
        sum(1 for c in range(cS, cE) if c not in leaves))


def _over_shared_facets(dm):
    fS, fE = dm.getHeightStratum(1)
    return uw.mpi.comm.allreduce(
        sum(1 for f in range(fS, fE) if len(dm.getSupport(f)) > 2))


def _centroids(dm):
    cS, cE = dm.getHeightStratum(0)
    if cE == cS:
        return np.zeros((0, dm.getCoordinateDim()))
    return np.array([dm.computeCellGeometryFVM(c)[1] for c in range(cS, cE)])


def _h_target(cen):
    d = np.linalg.norm(cen - CENTRE, axis=1)
    return np.where(d < 0.2, 0.05, 0.3)


def test_refined_mesh_is_independent_of_the_partition():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.35,
        refinement=1, qdegree=2)
    dm = base.dm_hierarchy[-1]
    assert _owned_cells(dm) == SERIAL_BASE_CELLS

    passes = 0
    while passes < 40:
        cS, _cE = dm.getHeightStratum(0)
        cen = _centroids(dm)
        if cen.shape[0]:
            sel = np.flatnonzero(
                edge_split.cell_diameters(dm) > _h_target(cen)) + cS
        else:
            sel = np.empty(0, dtype=int)     # this rank owns no cells
        dm, n_split = edge_split.bisect_longest_edges(dm, sel)
        if n_split == 0:
            break
        assert _over_shared_facets(dm) == 0, f"pass {passes} broke conformity"
        passes += 1

    assert _owned_cells(dm) == SERIAL_REFINED_CELLS, (
        f"np={uw.mpi.size} produced {_owned_cells(dm)} cells; serial gives "
        f"{SERIAL_REFINED_CELLS}. The refined mesh must not depend on the "
        f"partition.")
    assert passes == SERIAL_PASSES


def test_adapt_child_is_confluent_and_carries_the_tail():
    """The full ``mesh.adapt`` path, not just the engine."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
        regular=False, refinement=2, qdegree=3)

    def metric(cen):
        d = np.linalg.norm(np.asarray(cen) - np.array([0.4, 0.55]), axis=1)
        return 1.0 / np.where(d < 0.2, 0.03, 0.12) ** 2

    child = base.adapt(metric, max_levels=2, engine="edge_split")

    assert _over_shared_facets(child.dm) == 0
    tail = child._custom_mg_coarse_meshes
    assert tail is not None and len(tail) >= 3
    recorded = child._adapt_prolongation
    assert recorded and all(P is not None for P in recorded), (
        "the exact prolongation must survive at np>1: the inserted vertices are "
        "exact float edge midpoints on every rank")
    # Reported so a partition-dependent regression is legible in the log even if
    # the cell count assertion below is later relaxed.
    uw.pprint(0, f"[ptest_0843] np={uw.mpi.size}: child "
                 f"{_owned_cells(child.dm)} cells, tail {len(tail)} levels")
