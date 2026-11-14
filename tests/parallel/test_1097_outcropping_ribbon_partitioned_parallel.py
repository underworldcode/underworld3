"""A 2-D ladder band that outcrops at a low angle is placed in parallel.

The ribbon partitions the add-on for its bands before it places them, and the
placed mesh inherits the partition's provenance only if the placement moved no
cell. The band's cap runs along the wall past the end of the band's path, so a
partition measured from the path can leave part of the cap's gather on another
rank, and the ribbon is then refused as not rank-local (#1172).

Asserted: the ribbon returns, and its mesh carries the partition's provenance.
The serial companion (``tests/test_1097_band_partition_holds_the_placement.py``)
compares the two regions on one rank; this file is the one that runs the
ribbon's own call of the partition.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import place_surface

# Tier C until a parallel run has shown it to fail on a partition measured
# from the path.
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_c,
              pytest.mark.timeout(900)]

RUNG = 0.03                 # the band's rung and the base cell size
WIDTH = 2.0 * RUNG
CLEARANCE = 0.7


def _low_angle_outcrop(dip_degrees):
    dip = np.radians(dip_degrees)
    top = np.array([0.3, 1.0 - RUNG])
    along = np.array([np.cos(dip), -np.sin(dip)])
    return top + (np.arange(12) * RUNG)[:, None] * along


@pytest.mark.parametrize("dip_degrees", [15.0, 20.0, 25.0])
def test_outcropping_ladder_band_is_placed_on_the_partitioned_addon(dip_degrees):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=RUNG, qdegree=2)
    mesh, _info = place_surface.place_fault_ribbon_2d(
        base, [("f", _low_angle_outcrop(dip_degrees))], WIDTH, margin_rings=1,
        clearance=CLEARANCE, split=False, mesher="ladder")
    assert mesh._addon_partition is not None
    n_band = uw.mpi.comm.allreduce(
        mesh.dm.getStratumSize("Band", 71))
    assert n_band > 0
