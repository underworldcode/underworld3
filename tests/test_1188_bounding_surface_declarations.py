"""A declared wall surface is kept true or released by every coordinate writer.

The surface a mesh registers for a boundary is its declaration that the boundary
is that circle, sphere or plane (``mesh.analytic_surface``). The declaration is
authoritative, so every operation that writes coordinates must either keep the
wall on the surface by construction, or release the declaration.

The first group checks the operations that keep it: the wall's vertices are on
the declared surface afterwards. The operations put them there by projection, so
the bound is rounding, 64 machine epsilon relative to the radius (measured: at
most 6e-16). That distance is measured here, in the tests, and nowhere at run
time.

The second group checks the operations that cannot keep it: the declaration is
released, and ``analytic_surface`` refuses.
"""
import warnings

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing.graph import _pinned_mask

pytestmark = [pytest.mark.level_1]

ON_SURFACE = 64 * np.finfo(float).eps


def _distance_off(mesh, label, surface):
    """The largest distance of the vertices of boundary ``label`` of ``mesh`` from
    the radial ``surface``, relative to its radius, over every rank."""
    on_label = _pinned_mask(mesh.dm, (label,))
    X = np.asarray(mesh.X.coords, dtype=float)[on_label]
    local = float(np.max(np.abs(np.linalg.norm(X - surface.centre, axis=1) - surface.radius),
                         initial=0.0))
    count = uw.mpi.comm.allreduce(int(on_label.sum()), op=MPI.SUM)
    assert count > 0, f"no vertices on {label}"
    return uw.mpi.comm.allreduce(local, op=MPI.MAX) / surface.radius


def _radial_metric(mesh, name):
    field = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)
    r = np.linalg.norm(np.asarray(field.coords), axis=1)
    field.array[:, 0, 0] = 1.0 + 4.0 * np.exp(-((r - 1.0) / 0.1) ** 2)
    return field.sym[0]


# ----- operations that keep the declaration ---------------------------------


def _moved_on(mesh, label, before):
    """The largest displacement of the vertices of boundary ``label`` from
    ``before``, over every rank."""
    on_label = _pinned_mask(mesh.dm, (label,))
    shift = np.asarray(mesh.X.coords, dtype=float)[on_label] - before[on_label]
    return uw.mpi.comm.allreduce(
        float(np.max(np.linalg.norm(shift, axis=1), initial=0.0)), op=MPI.MAX)


@pytest.mark.tier_b
def test_the_mover_with_slip_keeps_both_annulus_walls_on_their_circles():
    """The walls' own vertices slide (measured 6.5e-4 on Upper and 4.2e-4 on Lower,
    2026-11-14) and are restored onto the circles."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    before = np.array(mesh.X.coords)
    uw.meshing.smooth_mesh_interior(
        mesh, metric=_radial_metric(mesh, "SlipMetric"), method="mmpde", slip_surfaces=True,
        method_kwargs=dict(n_outer=6, step_frac=0.2, tol=5.0e-3))
    for label in ("Lower", "Upper"):
        assert _moved_on(mesh, label, before) > 1.0e-4, f"{label} did not slide"
        assert _distance_off(mesh, label, mesh.analytic_surface(label)) < ON_SURFACE


@pytest.mark.tier_b
def test_plain_smoothing_holds_the_pinned_walls_where_they_were():
    """Without a metric the smoother pins every boundary it is not told to free,
    so the walls keep their declaration because no wall vertex moves."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    before = np.array(mesh.X.coords)
    uw.meshing.smooth_mesh_interior(mesh)
    moved = uw.mpi.comm.allreduce(
        float(np.max(np.abs(np.asarray(mesh.X.coords) - before), initial=0.0)), op=MPI.MAX)
    assert moved > 1.0e-4, "the smoother moved nothing"
    for label in ("Lower", "Upper"):
        assert _moved_on(mesh, label, before) == 0.0
        assert _distance_off(mesh, label, mesh.analytic_surface(label)) < ON_SURFACE


@pytest.mark.tier_b
@pytest.mark.parametrize("build, labels", [
    (lambda: uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25, refinement=2),
     ("Lower", "Upper")),
    (lambda: uw.meshing.AnnulusInternalBoundary(
        radiusOuter=1.0, radiusInternal=0.75, radiusInner=0.5, cellSize_Inner=0.25,
        cellSize_Internal=0.25, cellSize_Outer=0.25, refinement=2),
     ("Lower", "Internal", "Upper")),
    (lambda: uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5,
                                       refinement=1),
     ("Lower", "Upper")),
], ids=["annulus", "annulus_internal_boundary", "spherical_shell"])
def test_constructor_refinement_keeps_each_declared_surface(build, labels):
    mesh = build()
    for label in labels:
        assert _distance_off(mesh, label, mesh.analytic_surface(label)) < ON_SURFACE


@pytest.mark.tier_b
def test_adapt_puts_each_generation_on_the_parent_surfaces():
    """An adapted child does not carry the parent's declarations (it refuses), but
    its wall vertices are snapped onto the parent's surfaces. The metric refines
    at the outer wall, so the child has wall vertices the parent did not: left on
    the parent's chords they would be 1.8e-3 of the radius inside the circle."""
    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25, refinement=1,
                              qdegree=2)

    def metric(centroids):
        d = np.abs(np.linalg.norm(np.asarray(centroids), axis=1) - 1.0)
        return 1.0 / np.minimum(np.sqrt(0.05 ** 2 + (2.0 * d) ** 2), 0.3) ** 2

    child = base.adapt(metric, max_levels=2)
    on_wall = lambda mesh: uw.mpi.comm.allreduce(
        int(_pinned_mask(mesh.dm, ("Upper",)).sum()), op=MPI.SUM)
    assert on_wall(child) > on_wall(base), "adapt added no vertex on the outer wall"
    for label in ("Lower", "Upper"):
        assert _distance_off(child, label, base.analytic_surface(label)) < ON_SURFACE


# ----- operations that release it --------------------------------------------


@pytest.mark.tier_b
def test_slip_on_a_surface_passed_as_free_releases_that_surface_only():
    """``{label: False}`` slides the label's vertices and does not restore them."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    mesh.boundary_slip({"Upper": False, "Lower": True})
    with pytest.raises(ValueError, match="'Upper' was released"):
        mesh.analytic_surface("Upper")
    assert mesh.analytic_surface("Lower").radius == 0.5


@pytest.mark.tier_b
def test_a_mover_run_with_a_free_surface_leaves_it_released_and_the_other_wall_declared():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    uw.meshing.smooth_mesh_interior(
        mesh, metric=_radial_metric(mesh, "FreeSlipMetric"), method="mmpde",
        slip_surfaces={"Upper": False, "Lower": True},
        method_kwargs=dict(n_outer=6, step_frac=0.2, tol=5.0e-3))
    with pytest.raises(ValueError, match="'Upper' was released"):
        mesh.analytic_surface("Upper")
    assert _distance_off(mesh, "Lower", mesh.analytic_surface("Lower")) < ON_SURFACE


@pytest.mark.tier_b
def test_refinement_that_leaves_wall_vertices_off_the_surface_releases_that_wall():
    """On this cubed sphere the callback would invert cells at a few Lower vertices,
    which therefore stay where refinement put them, inside the wall (#949). How many
    depends on the host's triangulation: three, 4e-2 of the radius inside, on Linux;
    one, 8e-2 inside, on macOS."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mesh = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.547, numElements=3,
                                      simplex=True, refinement=1)
    assert "Lower" in mesh._refinement_left_off_surface, (
        "on this host's triangulation refinement held back no Lower vertex of this "
        "cubed sphere, so the test has nothing to release: choose another fixture")
    with pytest.raises(ValueError, match="'Lower' was released"):
        mesh.analytic_surface("Lower")
    assert _distance_off(mesh, "Upper", mesh.analytic_surface("Upper")) < ON_SURFACE


@pytest.mark.tier_b
@pytest.mark.parametrize("with_metric, pinned, drifting", [
    (False, ["Upper"], ["Lower"]),
    (False, [], ["Lower", "Upper"]),
    (True, ["Upper"], ["Lower"]),
], ids=["smoother_upper_pinned", "smoother_nothing_pinned", "metric_mover_upper_pinned"])
def test_a_mover_releases_each_wall_it_neither_pins_nor_restores(with_metric, pinned, drifting):
    """A wall left out of ``pinned_labels`` moves with the interior (measured: the
    inner wall 13 % of its radius off its circle), so it is no longer declared.
    A pinned wall stays declared, and on its circle."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    metric = _radial_metric(mesh, f"DriftMetric{len(pinned)}") if with_metric else None
    uw.meshing.smooth_mesh_interior(mesh, metric=metric, pinned_labels=pinned)
    for label in drifting:
        with pytest.raises(ValueError, match=f"'{label}' was released: a mesh mover"):
            mesh.analytic_surface(label)
    for label in pinned:
        assert _distance_off(mesh, label, mesh.analytic_surface(label)) < ON_SURFACE


@pytest.mark.tier_b
def test_restoring_slip_on_a_released_surface_warns_that_nothing_is_restored():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    mesh.boundary_slip({"Upper": False, "Lower": True})
    with pytest.warns(UserWarning, match="'Upper'.*released.*not restored"):
        mesh.boundary_slip(True)


@pytest.mark.tier_b
def test_restoring_slip_after_a_direct_write_follows_the_wall_until_it_is_declared_again():
    """A direct write takes the outer wall 1 % of its radius off its circle. The
    write released the declaration, so a restoring slip leaves the wall where the
    write put it and warns, naming the way to declare it again (development, before
    the release, put it back on the constructor's circle). Declared again, the next
    restoring slip puts it on the circle."""
    from underworld3.meshing.bounding_surface import BoundingSurface

    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    X = np.array(mesh.X.coords, dtype=float)
    upper = _pinned_mask(mesh.dm, ("Upper",))
    r = np.linalg.norm(X, axis=1)
    ripple = 1.0 + 0.01 * np.cos(3.0 * np.arctan2(X[:, 1], X[:, 0])) * (r - 0.5) / 0.5
    mesh.X.coords[...] = X * ripple[:, None]

    def upper_off(coords):
        return float(np.max(np.abs(np.linalg.norm(coords[upper], axis=1) - 1.0)))

    with pytest.warns(UserWarning, match="'Upper'.*register_tangent_slip_provider"):
        slid = mesh.project_to_slip_surface(np.array(mesh.X.coords), True)
    assert upper_off(slid) > 0.9e-2

    mesh.register_tangent_slip_provider(
        "Upper", BoundingSurface(mesh, "Upper", "radial", centre=(0.0, 0.0), radius=1.0))
    restored = mesh.project_to_slip_surface(np.array(mesh.X.coords), {"Upper": True})
    assert upper_off(restored) < ON_SURFACE


@pytest.mark.tier_b
def test_the_restore_warning_does_not_name_an_interior_surface():
    """An interior surface (the Internal circle) never slides: its vertices are
    pinned whatever the slip request, so its release is not a restore dropped."""
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=1.0, radiusInternal=0.75, radiusInner=0.5, cellSize_Inner=0.2,
        cellSize_Internal=0.2, cellSize_Outer=0.2)
    mesh.bounding_surfaces["Internal"].release("released by the test")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh.boundary_slip(True)
    named = [str(w.message) for w in caught if "'Internal'" in str(w.message)]
    assert not named, named


@pytest.mark.tier_b
def test_a_write_to_the_coordinates_of_a_mesh_without_variables_releases_every_surface():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    mesh.X.coords[...] = 1.01 * np.array(mesh.X.coords)
    for label in ("Lower", "Upper"):
        with pytest.raises(ValueError, match=f"'{label}' was released"):
            mesh.analytic_surface(label)


@pytest.mark.tier_b
def test_after_a_direct_write_points_return_to_the_walls_the_mesh_now_has():
    """The annulus is scaled to 0.9 of its size, so it spans radii 0.45 to 0.90. A
    point at radius 0.95 is outside and returns to the outer wall (its facets, so
    within a chord's sagitta, 1.3e-3, of 0.90); a point at radius 0.47 is inside
    and stays. The circles of the constructor, 0.5 and 1.0, give neither answer."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    mesh.X.coords[...] = 0.9 * np.array(mesh.X.coords)
    points = np.array([[0.95, 0.0], [0.0, 0.47]])
    returned = np.asarray(mesh.return_coords_to_bounds(points.copy()))
    assert 0.898 < np.linalg.norm(returned[0]) <= 0.9 * (1.0 + ON_SURFACE)
    assert np.array_equal(returned[1], points[1])


@pytest.mark.tier_b
def test_a_second_release_changes_nothing():
    """The refusal keeps naming the surface that was declared and the first reason."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    mesh.deform(np.array(mesh.X.coords))
    mesh.deform(np.array(mesh.X.coords))
    with pytest.raises(ValueError, match="the radial surface of boundary 'Upper' was released"):
        mesh.analytic_surface("Upper")

    lower = mesh.bounding_surfaces["Lower"]
    lower.release("a second reason")
    with pytest.raises(ValueError, match="the radial surface of boundary 'Lower' was released: "
                                         "the mesh has been deformed"):
        mesh.analytic_surface("Lower")


@pytest.mark.tier_b
def test_a_mesh_reloaded_from_a_file_declares_nothing(tmp_path_factory):
    """A saved mesh comes back through ``uw.discretisation.Mesh`` and no constructor,
    so nothing registers a surface for it, whether or not it had moved."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    mesh.X.coords[...] = 1.01 * np.array(mesh.X.coords)
    folder = uw.mpi.comm.bcast(str(tmp_path_factory.mktemp("reload")) if uw.mpi.rank == 0 else None,
                               root=0)
    mesh.write(f"{folder}/moved.mesh.h5")
    reloaded = uw.discretisation.Mesh(f"{folder}/moved.mesh.h5")
    assert dict(reloaded.bounding_surfaces) == {}
    with pytest.raises(ValueError, match="no analytic surface is registered for boundary 'Upper'"):
        reloaded.analytic_surface("Upper")
