"""Parallel embedding of a thin volume (:func:`place_surface.place_thin_volume`).

The mechanism is place_sheet's, inherited whole: the assembly is meshed once
(rank 0) and broadcast so every rank marks against the identical skin; the
zone's region is gathered onto one rank; the carve and the annular hole fill
run there; every rank rebuilds collectively. The gathered region is
rank-interior, so nothing the surgery touches is shared and the topology is
partition-independent by construction.

Deterministic assertions only (the test_0842 lesson): the info dict identical
on every rank, labels of the advertised kinds and sizes, volume conserved by
the routine's own gate, and refusals collective — a rank-local raise is a
hang at np>=3 and the happy path cannot see that defect class at all. Run:

    mpirun -np 2 python -m pytest tests/parallel/ptest_0855_place_thin_volume_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

CROSS = [np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                   [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]]),
         np.array([[0.3, 0.5, 0.3], [0.7, 0.5, 0.3],
                   [0.7, 0.5, 0.7], [0.3, 0.5, 0.7]])]


def _owned_label_count(dm, name, value):
    pStart, _pEnd = dm.getChart()
    leaves = np.zeros(dm.getChart()[1] - pStart, dtype=bool)
    try:
        _n, ilocal, _ir = dm.getPointSF().getGraph()
        if ilocal is not None and len(ilocal):
            leaves[np.asarray(ilocal, dtype=np.int64) - pStart] = True
    except (ValueError, TypeError):
        pass
    n = 0
    if dm.hasLabel(name) and dm.getLabel(name).getStratumSize(value) > 0:
        for p in dm.getLabel(name).getStratumIS(value).getIndices():
            if not leaves[int(p) - pStart]:
                n += 1
    return int(uw.mpi.comm.allreduce(n, op=MPI.SUM))


def test_parallel_embedding_matches_the_serial_contract():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.09, regular=False, qdegree=2)
    new, info = place_thin_volume(mesh.dm, CROSS, width=0.045,
                                  label="Zone", label_value=5)

    # The routine gates volume, Euler and the constraint surfaces itself,
    # collectively; asserting the info here proves the gates ran and agreed.
    assert info["n_zone_cells"] > 0
    assert _owned_label_count(new, "Zone", 5) == info["n_zone_cells"]
    assert _owned_label_count(new, "Zone_skin", 5) == info["n_skin_faces"]

    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)


def test_refusals_are_collective():
    """A zone through the wall must refuse IDENTICALLY on every rank."""
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.15, regular=False, qdegree=2)
    low = np.array([[0.3, 0.3, 0.08], [0.7, 0.3, 0.08],
                    [0.7, 0.7, 0.08], [0.3, 0.7, 0.08]])
    message = None
    try:
        place_thin_volume(mesh.dm, [low], width=0.05, label="Bad")
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"
