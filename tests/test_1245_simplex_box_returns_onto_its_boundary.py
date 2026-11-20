"""An unstructured simplex box returns a point onto its boundary, continuously.

``return_coords_to_bounds`` used to move a point that is outside the box to 1e-3
inside it and leave a point inside where it was: a 1e-3 jump at the wall, so
two points a round-off apart, either side of a wall, came back 1e-3 apart. A
semi-Lagrangian foot on a free-slip wall (where v.n is round-off) landed on
either side of that jump. StructuredQuadBox returns a point onto the boundary;
the simplex box now does the same. This changes where a swarm particle that
left the box is put back: onto the wall, not 1e-3 inside it.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

ROUND_OFF = 1.0e-15


@pytest.mark.parametrize("dim", [2, 3])
def test_a_point_a_round_off_outside_returns_onto_the_wall(dim):
    uw.reset_default_model()
    lo, hi = (0.0,) * dim, (2.0, 1.0, 1.0)[:dim]
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=lo, maxCoords=hi,
                                             cellSize=0.5 if dim == 3 else 0.25)
    inside = 0.5 * np.array(hi)
    points = []
    for axis in range(dim):
        for wall, outward in ((lo[axis], -1.0), (hi[axis], 1.0)):
            for offset in (ROUND_OFF, -ROUND_OFF, 0.25):      # outside, inside, well outside
                p = inside.copy()
                p[axis] = wall + outward * offset
                points.append((axis, wall, offset, p))
    coords = np.array([p for *_, p in points])
    returned = mesh.return_coords_to_bounds(coords.copy())
    for (axis, wall, offset, p), q in zip(points, returned):
        if offset > 0:                                   # it was outside: onto the wall
            assert q[axis] == wall, (axis, wall, offset, q)
        else:                                            # it was inside: where it was
            assert np.array_equal(q, p), (axis, wall, offset, q)
        others = [k for k in range(dim) if k != axis]
        assert np.array_equal(q[others], p[others])
