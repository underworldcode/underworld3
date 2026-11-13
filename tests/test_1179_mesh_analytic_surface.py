"""A mesh says which of its boundaries is an exact circle, sphere or plane.

Methods that use an analytic shortcut on a wall (its exact radius and centre) read
the surface the mesh registered for that boundary, and refuse when there is none.
The registration is the declaration: nothing is measured here. A raw
``mesh.deform`` releases every surface, so the declaration does not outlive a move
that may have taken the wall off it.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


@pytest.mark.tier_b
def test_an_annulus_declares_both_walls_as_circles_about_the_origin():
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2)
    for label, radius in (("Lower", 0.5), ("Upper", 1.0)):
        surface = mesh.analytic_surface(label)
        assert surface.kind == "radial" and not surface.interior
        assert surface.radius == radius
        assert np.array_equal(surface.centre, np.zeros(2))


@pytest.mark.tier_b
def test_an_internal_boundary_is_declared_as_an_interior_circle():
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusInner=0.5, radiusInternal=0.75, radiusOuter=1.0,
        cellSize_Inner=0.2, cellSize_Internal=0.2, cellSize_Outer=0.2)
    surface = mesh.analytic_surface("Internal")
    assert surface.kind == "radial" and surface.interior and surface.radius == 0.75


@pytest.mark.tier_b
def test_a_box_declares_its_faces_as_planes():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), cellSize=0.25)
    top = mesh.analytic_surface("Top")
    assert top.kind == "plane"
    assert np.array_equal(np.abs(top.normal), np.array([0.0, 1.0]))
    assert top.point[1] == 1.0


@pytest.mark.tier_b
def test_a_boundary_with_no_registered_surface_is_refused_by_name():
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2)
    with pytest.raises(ValueError, match="no analytic surface is registered for boundary 'Centre'"):
        mesh.analytic_surface("Centre")


@pytest.mark.tier_b
def test_a_deformed_mesh_no_longer_declares_its_walls():
    """``deform`` may have moved the wall, so it releases the declaration, even
    when (as here) the coordinates it was given are the ones the mesh had."""
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2)
    mesh.deform(np.array(mesh.X.coords))
    with pytest.raises(ValueError, match="'Upper' was released"):
        mesh.analytic_surface("Upper")


@pytest.mark.tier_b
def test_a_surface_that_is_not_analytic_is_refused():
    from underworld3.meshing.bounding_surface import BoundingSurface

    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2)
    segments = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    mesh.register_tangent_slip_provider(
        "Upper", BoundingSurface(mesh, "Upper", "facet", reference_facets=segments))
    with pytest.raises(ValueError, match="'Upper' is registered as 'facet'"):
        mesh.analytic_surface("Upper")


@pytest.mark.tier_b
def test_a_boundary_is_named_by_its_label_or_by_its_enum_member():
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2)
    assert mesh.analytic_surface(mesh.boundaries.Upper) is mesh.analytic_surface("Upper")


@pytest.mark.tier_b
def test_a_solid_disc_declares_no_inner_circle():
    """A disc has no inner wall: a circle of radius zero is a point, and the mesh
    holds no ``Lower`` boundary to lie on it."""
    mesh = uw.meshing.Annulus(radiusInner=0.0, radiusOuter=1.0, cellSize=0.2)
    assert mesh.analytic_surface("Upper").radius == 1.0
    with pytest.raises(ValueError, match="no analytic surface is registered for boundary 'Lower'"):
        mesh.analytic_surface("Lower")


@pytest.mark.tier_b
def test_a_refined_shell_does_not_declare_its_internal_sphere():
    """Refinement places the new vertices of the embedded surface on the chords
    of the coarse one (#949 item 3), so the refined mesh must not declare the
    sphere. The unrefined mesh is on it, and declares it."""
    shell = dict(radiusOuter=1.0, radiusInternal=0.8, radiusInner=0.5, cellSize=0.3)
    unrefined = uw.meshing.SphericalShellInternalBoundary(**shell)
    assert unrefined.analytic_surface("Internal").radius == 0.8

    refined = uw.meshing.SphericalShellInternalBoundary(**shell, refinement=1)
    with pytest.raises(ValueError, match="'Internal' was released: refinement"):
        refined.analytic_surface("Internal")
    assert refined.analytic_surface("Upper").radius == 1.0
