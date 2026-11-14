"""The facet restore on one rank (#1176).

The restore takes the nearest point on the mesh's boundary facets. To keep that
affordable once the facets are those of the whole mesh, the search is bounded: only
facets whose centroid is within the nearest centroid's distance plus the largest
facet radius are tested. That bound is exact, so the nearest point it gives is the
one the search over every facet gives, bit for bit and with the same facet on a
tie. Checked here on a 2-D and a 3-D boundary against the search over every facet.

And H1's fixture from #1176 on one rank: an annulus moved by deform, walls at 0.45
and 0.90. A point inside is returned unchanged; a point beyond the outer wall lands
on it.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.discretisation.discretisation_mesh import (
    _facets_that_can_hold_the_nearest_point)
from underworld3.meshing.smoothing.graph import _nearest_on_facets_2d, _nearest_on_facets_3d

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.mark.parametrize("dim", [2, 3])
def test_the_bounded_search_finds_the_nearest_point_the_full_search_finds(dim):
    if dim == 2:
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.05)
        nearest_on = _nearest_on_facets_2d
    else:
        mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25)
        nearest_on = _nearest_on_facets_3d
    mesh.deform(0.9 * np.asarray(mesh.X.coords, dtype=float))
    fpts, _ = mesh._boundary_facet_geometry()
    rng = np.random.default_rng(1176)
    pts = rng.uniform(-1.1, 1.1, (2000, dim))
    # points on the facets themselves, where ties between facets are likeliest
    vertices = fpts.reshape(-1, dim)
    pts = np.vstack([pts, vertices[rng.choice(len(vertices), 200, replace=False)]])

    centroids = np.ascontiguousarray(fpts.mean(axis=1))
    tree = uw.kdtree.KDTree(centroids)
    bounded = np.array([nearest_on(p[None], fpts[c])[0] for p, c in zip(
        pts, _facets_that_can_hold_the_nearest_point(pts, fpts, centroids, tree))])
    full = nearest_on(pts, fpts)

    assert np.array_equal(bounded, full)


def test_inside_points_stay_and_outside_points_reach_the_wall():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    mesh.deform(0.9 * np.asarray(mesh.X.coords, dtype=float))
    assert mesh.return_coords_to_bounds == mesh._facet_return_coords_to_bounds
    theta = 0.1 + 2.0 * np.pi * np.arange(32) / 32
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    inside = np.vstack([0.47 * ring, 0.75 * ring, 0.88 * ring])

    restored_inside = np.asarray(mesh.return_coords_to_bounds(inside.copy()))
    restored_beyond = np.asarray(mesh.return_coords_to_bounds(0.95 * ring))

    assert np.array_equal(restored_inside, inside)
    # within the largest chord's sagitta of the 0.90 circle (cell size 0.09 there)
    sagitta = 0.9 - np.sqrt(0.9 ** 2 - 0.09 ** 2)
    r = np.linalg.norm(restored_beyond, axis=1)
    assert np.all(r <= 0.9) and np.all(r >= 0.9 - 2.0 * sagitta)
