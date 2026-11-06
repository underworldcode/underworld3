"""The nested-transfer helpers give every rank the same verdict (#961).

``nvb.nested_cell_parents`` and ``nvb.nested_prolongation_from_dms`` return ``None`` when
the map or the transfer cannot be recovered, and their callers then take a fallback that
is collective (the geometric transfer builder). Each used to decide on its own rank's
cells: one rank returned ``None`` and the others an array, so the ranks went down
different builders (ptest_0844_reconnect hung at np 2 and 4, #961). The verdict is now
reduced over all ranks. On every rank:

- a vertex transfer spoiled on rank 0 only gives ``None`` for the cell-parent map;
- the transfer recovered from a regular refinement, and its cell-parent map, are found;
- a rank that owns no fine vertices returns an empty transfer, not ``None``, so it does
  not veto the others.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0874_nested_transfer_verdict_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import nvb

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def _coarse_and_fine():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2, regular=True, qdegree=2)
    coarse = base.dm
    return coarse, coarse.refine()


def test_a_transfer_spoiled_on_one_rank_gives_no_map_on_any():
    coarse, fine = _coarse_and_fine()
    transfer = nvb.nested_prolongation_from_dms(coarse, fine)
    found = uw.mpi.comm.allgather(transfer is not None)
    parents = nvb.nested_cell_parents(coarse, fine, transfer) if all(found) else None
    good = uw.mpi.comm.allgather(parents is not None)
    spoiled = None
    if all(found):
        rows, cols, vals = transfer
        if uw.mpi.rank == 0 and cols.size:
            # rank 0's fine vertices now refer, alternately, to two coarse vertices at
            # opposite corners of its patch, which share no coarse cell: no fine cell there
            # has a parent at all
            X = coarse.getCoordinatesLocal().array.reshape(-1, 2)
            a, b = int(np.argmin(X.sum(axis=1))), int(np.argmax(X.sum(axis=1)))
            cols = np.where(np.arange(cols.size) % 2 == 0, a, b).astype(cols.dtype)
        spoiled = nvb.nested_cell_parents(coarse, fine, (rows, cols, vals))
    verdicts = uw.mpi.comm.allgather(spoiled is None)
    assert all(found) and all(good), (found, good)
    assert all(verdicts), verdicts


def test_a_rank_with_nothing_to_explain_returns_an_empty_transfer():
    coarse, fine = _coarse_and_fine()
    local = nvb._nested_prolongation_local(coarse, fine)
    vs, ve = fine.getDepthStratum(0)
    shapes = uw.mpi.comm.allgather(None if local is None else (ve - vs, len(local[0])))
    assert all(sh is not None for sh in shapes), shapes
