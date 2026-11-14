"""A split fault's sides slip the same on any partition (#1103 review).

``boundary_slip`` drops boundary facets that sit on a partition seam (a facet
between two ranks' cells looks like a boundary facet on both). Identified by
coordinates, a split fault's Plus and Minus replicas, which coincide by design,
looked like a seam pair and were dropped even on one rank: with FltPlus and
FltMinus as explicit slip surfaces, no vertex slipped in parallel where serial
slips the fault's interior vertices. Seam facets are now found through the
point star forest. Checked: the number of vertices that slip (moved back onto
their surface after an offset) equals serial's on any rank count. This file
carries no ``mpi(min_size=2)`` mark: it runs serially and in the parallel
batches.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

TRACE = np.array([[0.25, 0.30], [0.75, 0.62]])
SERIAL_SLIPPING = 26        # measured serially on this fixture (2026-11-13)


def _slipping_vertices():
    from underworld3.meshing.smoothing import _owned_vertex_mask

    box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                            cellSize=0.08)
    mesh = box.add_fault([("Flt", TRACE)])
    ref = np.asarray(mesh.X.coords, dtype=float).copy()
    _pinned, project = mesh.boundary_slip(("FltPlus", "FltMinus"), reference_coords=ref)
    offset = ref + np.array([0.002, 0.003])
    moved = project(offset.copy())
    owned = np.asarray(_owned_vertex_mask(mesh.dm), dtype=bool)
    slipped = owned & (np.abs(moved - offset).max(axis=1) > 1e-12)
    return uw.mpi.comm.allreduce(int(slipped.sum()))


def test_a_split_faults_sides_slip_as_in_serial():
    count = _slipping_vertices()
    assert count == SERIAL_SLIPPING, count
