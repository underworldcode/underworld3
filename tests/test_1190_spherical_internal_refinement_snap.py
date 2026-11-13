"""Refinement keeps the internal sphere of a shell on its sphere (#949 item 3).

``SphericalShellInternalBoundary`` embeds a sphere of radius ``radiusInternal``
between its walls. Refinement places the new vertices of every surface on the
chords of the coarse one; the refinement callback moves the new vertices of
``Lower`` and ``Upper`` back onto their spheres, and must do the same for
``Internal``, so that the refined mesh can declare that sphere.

Contracts, on every level of the refinement hierarchy:

- every vertex of each of the three surfaces lies on its sphere, to 64 machine
  epsilon relative to the radius (on the base, 41521d4f, before the callback moved
  ``Internal``: 2.9e-02 off at refinement 1, on a 96-thread Linux
  workstation, 2026-11-14);
- the mesh declares the ``Internal`` sphere: ``analytic_surface("Internal")``
  returns radius ``radiusInternal``;
- the region labels still tile each shell exactly: the cells labelled ``Inner``
  fill the volume between the ``Lower`` and ``Internal`` polyhedra, and ``Outer``
  the volume between ``Internal`` and ``Upper``. A cell turned inside out by the
  snap would overlap its neighbours and the sum of cell volumes would exceed the
  volume between the surfaces. Measured with the snap: 2.2e-16 relative at most
  (refinement 1 and 2, Linux, 2026-11-14); a mutant that pushes one interior vertex
  through the Internal sphere after the callback gives 3.2e-03.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

pytestmark = [pytest.mark.level_1]

SHELL = dict(radiusOuter=1.0, radiusInternal=0.8, radiusInner=0.5, cellSize=0.3)
RADII = {"Lower": 0.5, "Internal": 0.8, "Upper": 1.0}
ON_SURFACE = 64 * np.finfo(float).eps


@pytest.fixture(scope="module", params=[1, 2], ids=["refinement1", "refinement2"])
def refined_shell(request):
    return uw.meshing.SphericalShellInternalBoundary(**SHELL, refinement=request.param)


def _vertex_coordinates(dm):
    return dm.getCoordinatesLocal().array.reshape(-1, 3)


def _cell_tetrahedra(dm, cells):
    from underworld3.utilities._plex_topology import closure_points, section_offsets
    vs, ve = dm.getDepthStratum(0)
    rows = section_offsets(dm.getCoordinateSection(), np.arange(vs, ve)) // 3
    return _vertex_coordinates(dm)[rows[closure_points(dm, cells, 0) - vs]]


def _enclosed_volume(dm, label, value):
    """Volume inside the closed polyhedron of the facets labelled ``label``. The
    polyhedron is star-shaped about the origin (its vertices lie on a sphere about
    it), so the volume is the sum, over facets, of the tetrahedra from the origin."""
    fs, fe = dm.getHeightStratum(1)
    facets = np.asarray(dm.getStratumIS(label, value).getIndices())
    facets = facets[(facets >= fs) & (facets < fe)]
    triangles = _cell_tetrahedra(dm, facets)
    return np.abs(np.einsum("ij,ij->i", triangles[:, 0],
                            np.cross(triangles[:, 1], triangles[:, 2]))).sum() / 6


def _region_volume(dm, label, value):
    cs, ce = dm.getHeightStratum(0)
    cells = np.asarray(dm.getStratumIS(label, value).getIndices())
    cells = cells[(cells >= cs) & (cells < ce)]
    tets = _cell_tetrahedra(dm, cells)
    return np.abs(np.linalg.det(tets[:, 1:] - tets[:, :1])).sum() / 6


@pytest.mark.tier_b
def test_every_level_keeps_each_surface_on_its_sphere(refined_shell):
    for level, dm in enumerate(refined_shell.dm_hierarchy):
        radius = np.linalg.norm(_vertex_coordinates(dm), axis=1)
        for label, expected in RADII.items():
            vertices = petsc_dm_find_labeled_points_local(dm, label)
            assert len(vertices) > 0
            off = np.max(np.abs(radius[vertices] - expected)) / expected
            assert off <= ON_SURFACE, (
                f"level {level}: {label} vertices are up to {off:.3e} of the radius "
                f"off the sphere of radius {expected}")


@pytest.mark.tier_b
def test_the_refined_shell_declares_its_internal_sphere(refined_shell):
    surface = refined_shell.analytic_surface("Internal")
    assert surface.kind == "radial" and surface.interior
    assert surface.radius == 0.8
    assert np.array_equal(surface.centre, np.zeros(3))


@pytest.mark.tier_b
def test_the_regions_fill_the_space_between_their_surfaces(refined_shell):
    dm = refined_shell.dm
    lower = _enclosed_volume(dm, "Lower", 11)
    internal = _enclosed_volume(dm, "Internal", 12)
    upper = _enclosed_volume(dm, "Upper", 13)
    for region, value, expected in (("Inner", 101, internal - lower),
                                    ("Outer", 102, upper - internal)):
        filled = _region_volume(dm, region, value)
        assert abs(filled - expected) <= ON_SURFACE * expected, (
            f"{region}: cells fill {filled!r}, the surfaces enclose {expected!r}")
