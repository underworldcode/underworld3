"""Parallel removal (:func:`place_surface.remove_embedded`) — the lifecycle
on a distributed mesh, with no redistribution ever.

The removal gathers the object's star exactly as the placements do, carves
and refills on one rank, and every rank rebuilds collectively — the rest of
the mesh never moves. The object's geometry is allgathered from its labels
before marking, so removal works whatever the current distribution (the
object may have been scattered by a checkpoint reload).

Deterministic assertions only: labels globally empty, volume conserved (the
routine's own collective gate — asserting the info proves the gates ran and
agreed), the in-call validity battery (check_faces at every rank count), and
refusals collective. Cell counts after a refill are NOT pinned: the fill is
gmsh's, and its output depends on shell node ordering, which the gather
sets. Run:

    mpirun -np 2 python -m pytest tests/parallel/ptest_0856_remove_embedded_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.place_surface import (place_thin_volume,
                                                 remove_embedded)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

PATCH = np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                  [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]])


def test_a_distributed_zone_is_removed_cleanly():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.11, regular=False, qdegree=2)
    zoned, _ = place_thin_volume(mesh.dm, [PATCH], width=0.045,
                                 label="Zone", label_value=5)
    cleared, info = remove_embedded(zoned, "Zone", label_value=5)

    assert info["n_removed_cells"] > 0 and info["n_filled_cells"] > 0
    for name in ("Zone", "Zone_skin"):
        left = (cleared.getLabel(name).getStratumSize(5)
                if cleared.hasLabel(name) else 0)
        assert int(comm.allreduce(left, op=MPI.SUM)) == 0

    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)


def test_removal_refusals_are_collective():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.2, regular=False, qdegree=2)
    message = None
    try:
        remove_embedded(mesh.dm, "Ghost", label_value=3)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"
