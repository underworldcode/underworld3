"""A 2-D ladder band that outcrops at a low angle is placed in parallel.

The ribbon partitions the add-on for its bands before it places them, and the
placed mesh inherits the partition's provenance only if the placement moved no
cell. The band's cap runs along the wall past the end of the band's path, so a
partition measured from the path leaves part of the cap's gather outside the
region it keeps on one rank, and the ribbon is then refused as not rank-local
(#1172).

The base's cells are dealt to the ranks by cell number, not by a partitioner,
so a cell left outside the kept region is on another rank on any machine. The
base has one level under it: the provenance, and with it the rank-local check,
is recorded only for a mesh with a level below.

Asserted: the ribbon returns, and its mesh carries the partition's provenance
on every rank. The serial companion
(``tests/test_1097_band_partition_holds_the_placement.py``) compares the two
regions on one rank; this file runs the ribbon's own call of the partition.
"""
import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import place_surface

# Tier C until a parallel run has shown it to fail on a partition measured
# from the path.
pytestmark = [pytest.mark.mpi(min_size=4), pytest.mark.level_2, pytest.mark.tier_c,
              pytest.mark.timeout(900)]

N_COARSE = 17               # coarse cells along a side; one refinement halves them
RUNG = 0.5 / N_COARSE       # the band's rung and the refined base's cell size
WIDTH = 2.0 * RUNG
CLEARANCE = 0.7
ALL_BOUNDARIES_VALUE = 1001


def _unit_square_dealt_by_cell_number(n):
    """The unit square in ``2 n^2`` triangles, cell ``c`` on the rank a
    multiplicative hash of ``c`` names. Collective."""
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    coords = np.zeros((0, 2))
    cells = np.zeros((0, 3), dtype=PETSc.IntType)
    if rank == 0:
        at = lambda i, j: j * (n + 1) + i
        coords = np.array([[i / n, j / n] for j in range(n + 1) for i in range(n + 1)])
        cells = np.asarray(
            [tri for j in range(n) for i in range(n)
             for tri in ([at(i, j), at(i + 1, j), at(i, j + 1)],
                         [at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])],
            dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True, comm=comm)
    part = dm.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    if rank == 0:
        home = (np.arange(len(cells), dtype=np.int64) * 2654435761 % 2**32) % size
        shares = [np.flatnonzero(home == r) for r in range(size)]
        part.setShellPartition(
            size, [len(s) for s in shares], [int(c) for s in shares for c in s])
    else:
        part.setShellPartition(size, [0] * size, [])
    dm.distribute()
    # a mesh made from a bare DM has the name All_Boundaries and no such label (#1138)
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)
    return dm


def base_mesh():
    return uw.discretisation.Mesh(
        _unit_square_dealt_by_cell_number(N_COARSE), refinement=1, qdegree=2)


def low_angle_outcrop(dip_degrees):
    dip = np.radians(dip_degrees)
    top = np.array([0.3, 1.0 - RUNG])
    along = np.array([np.cos(dip), -np.sin(dip)])
    return top + (np.arange(12) * RUNG)[:, None] * along


@pytest.mark.parametrize("dip_degrees", [15.0, 20.0])
def test_outcropping_ladder_band_is_placed_on_the_partitioned_addon(dip_degrees):
    mesh, _info = place_surface.place_fault_ribbon_2d(
        base_mesh(), [("f", low_angle_outcrop(dip_degrees))], WIDTH, margin_rings=1,
        clearance=CLEARANCE, split=False, mesher="ladder")
    recorded = getattr(mesh, "_addon_parent", None) is not None
    assert uw.mpi.comm.allreduce(int(recorded)) == uw.mpi.size
    assert uw.mpi.comm.allreduce(mesh.dm.getStratumSize("Band", 71)) > 0
