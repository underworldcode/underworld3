"""BoxInternalBoundary returns a point that left the box onto the wall it left by,
changing only that coordinate.

Its ``return_coords_to_bounds`` set every coordinate of an outside point to the
bound it had crossed (``coords[x00s, :] = minCoords[0]``), and in 3-D it tested
the y column against the z bounds. The other box meshes change only the
coordinate that is out of bounds; StructuredQuadBox returns it onto the wall.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.mark.parametrize("dim", [2, 3])
def test_only_the_coordinate_out_of_bounds_returns_onto_its_wall(dim):
    uw.reset_default_model()
    lo, hi = (0.0, 0.0, 0.0)[:dim], (2.0, 1.0, 3.0)[:dim]     # distinct bounds per axis
    mesh = uw.meshing.BoxInternalBoundary(
        elementRes=(4, 4, 4)[:dim], zelementRes=(2, 2), minCoords=lo, maxCoords=hi,
        zintCoord=0.5 * hi[-1], degree=1, qdegree=2)
    inside = 0.5 * np.array(hi)
    cases = []
    for axis in range(dim):
        for wall, outward in ((lo[axis], -1.0), (hi[axis], 1.0)):
            p = inside.copy()
            p[axis] = wall + outward * 0.25
            cases.append((axis, wall, p))
    returned = mesh.return_coords_to_bounds(np.array([p for *_, p in cases]))
    for (axis, wall, p), q in zip(cases, returned):
        expected = p.copy()
        expected[axis] = wall
        assert np.array_equal(q, expected), (axis, wall, p, q)
    # a point inside is left where it is
    assert np.array_equal(mesh.return_coords_to_bounds(inside[None, :].copy())[0], inside)
