"""A refined SegmentofSphere keeps its walls on spheres about its centroid (#949,
item 4).

The refinement callback snapped the wall vertices onto spheres about the origin,
not about ``centroid``. With the centroid at (0, 0, 0.3), refinement moved the
Lower vertices 0.25 to 0.55 from the centroid against a wall at 0.547 (the
hierarchy guard held some back and warned). The callback now snaps about the
centroid; at rest and at the origin nothing changes.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

R_INNER, R_OUTER = 0.547, 1.0


@pytest.mark.parametrize("centroid", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.3)],
                         ids=["at the origin", "off the origin"])
def test_the_refined_walls_lie_on_their_spheres_about_the_centroid(centroid):
    from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

    mesh = uw.meshing.SegmentofSphere(radiusOuter=R_OUTER, radiusInner=R_INNER,
                                      cellSize=0.25, refinement=1, centroid=centroid)
    X = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
    c = np.asarray(centroid)
    for wall, radius in (("Upper", R_OUTER), ("Lower", R_INNER)):
        rows = petsc_dm_find_labeled_points_local(mesh.dm, wall)
        assert len(rows) > 0, wall
        d = np.linalg.norm(X[rows] - c, axis=1)
        assert np.abs(d - radius).max() < 1e-12 * radius, (wall, d.min(), d.max())
