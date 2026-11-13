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


@pytest.mark.tier_b
@pytest.mark.parametrize("with_metric", [True, False], ids=["metric_mover", "plain_smoothing"])
def test_the_mover_with_slip_keeps_both_annulus_walls_on_their_circles(with_metric):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    before = np.array(mesh.X.coords)
    metric = _radial_metric(mesh, f"SlipMetric{int(with_metric)}") if with_metric else None
    uw.meshing.smooth_mesh_interior(
        mesh, metric=metric, method="mmpde", slip_surfaces=True,
        method_kwargs=dict(n_outer=6, step_frac=0.2, tol=5.0e-3))
    moved = uw.mpi.comm.allreduce(
        float(np.max(np.abs(np.asarray(mesh.X.coords) - before), initial=0.0)), op=MPI.MAX)
    assert moved > 1.0e-4
    for label in ("Lower", "Upper"):
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
    its wall vertices are snapped onto the parent's surfaces."""
    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25, refinement=1,
                              qdegree=2)

    def metric(centroids):
        d = np.abs(np.linalg.norm(np.asarray(centroids), axis=1) - 0.75)
        return 1.0 / np.minimum(np.sqrt(0.05 ** 2 + (2.0 * d) ** 2), 0.3) ** 2

    child = base.adapt(metric, max_levels=2)
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
    """On this cubed sphere the callback would invert cells at three Lower vertices,
    which therefore stay where refinement put them, 4e-2 inside the wall (#949)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mesh = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.547, numElements=3,
                                      simplex=True, refinement=1)
    with pytest.raises(ValueError, match="'Lower' was released"):
        mesh.analytic_surface("Lower")
    assert _distance_off(mesh, "Upper", mesh.analytic_surface("Upper")) < ON_SURFACE


@pytest.mark.tier_b
def test_a_write_to_the_coordinates_of_a_mesh_without_variables_releases_every_surface():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    mesh.X.coords[...] = 1.01 * np.array(mesh.X.coords)
    for label in ("Lower", "Upper"):
        with pytest.raises(ValueError, match=f"'{label}' was released"):
            mesh.analytic_surface(label)


@pytest.mark.tier_b
def test_a_second_release_changes_nothing():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    mesh.deform(np.array(mesh.X.coords))
    mesh.deform(np.array(mesh.X.coords))
    with pytest.raises(ValueError, match="the radial surface of boundary 'Upper' was released"):
        mesh.analytic_surface("Upper")


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
