"""Generators that convert their gmsh file inside a rank-0 block build in parallel (#953).

``SegmentedSphericalShell``, ``SegmentedSphericalBall`` and ``AnnulusWithSpokes`` read
their gmsh file on rank 0 alone, on ``COMM_SELF``, because they edit the DM before it
is distributed. The conversion ended with a barrier on the world communicator, which
only rank 0 reached: at two ranks the constructor never returned.

Each mesh must build, and hold the cells the same generator gives on one rank. That
count is computed here, in this environment, by a single-rank child (see
``serial_reference``), so it does not depend on which gmsh built the mesh.
"""
import sys

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

from serial_reference import emit, serial_reference

# The timeout is the regression's failure: without it a hang blocks the CI job.
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

GENERATORS = {
    "segmented_shell": lambda: uw.meshing.SegmentedSphericalShell(cellSize=0.5, numSegments=3),
    "segmented_ball": lambda: uw.meshing.SegmentedSphericalBall(cellSize=0.5, numSegments=3),
    "annulus_with_spokes": lambda: uw.meshing.AnnulusWithSpokes(cellSizeOuter=0.2, cellSizeInner=0.2),
}


def _owned_cells(mesh):
    owned = np.asarray(mesh.dm.getCellNumbering().getIndices()) >= 0
    return uw.mpi.comm.allreduce(int(np.count_nonzero(owned)), op=MPI.SUM)


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_a_generator_that_converts_on_rank_zero_builds_in_parallel(kind):
    cells = _owned_cells(GENERATORS[kind]())
    reference = serial_reference(__file__, kind, timeout=200)
    assert cells == int(reference["values"][0])


if __name__ == "__main__":
    # the single-rank child of the parallel run: `python <this file> <kind>`
    _cells = _owned_cells(GENERATORS[sys.argv[1]]())
    emit([_cells], [_cells, 0.0])
