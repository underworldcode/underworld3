"""Smoke tests for the parallel-safe mesh radius accessors.

Verifies that :meth:`Mesh.get_min_radius`, :meth:`Mesh.get_mean_radius`,
and :meth:`Mesh.get_max_radius` return sensible values in serial.
A matching MPI smoke test lives in tests/parallel/.
"""
import pytest

pytestmark = pytest.mark.level_1


def _check_radii_ordering(mesh):
    rmin = mesh.get_min_radius()
    rmean = mesh.get_mean_radius()
    rmax = mesh.get_max_radius()
    assert rmin > 0.0, f"min radius must be positive, got {rmin}"
    assert rmin <= rmean <= rmax, (
        f"expected min <= mean <= max, got "
        f"min={rmin}, mean={rmean}, max={rmax}")


def test_mesh_radii_accessors_annulus():
    import underworld3 as uw
    mesh = uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5,
        cellSize=1.0 / 16, qdegree=3)
    _check_radii_ordering(mesh)


def test_mesh_radii_accessors_box():
    import underworld3 as uw
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16)
    _check_radii_ordering(mesh)


def test_mesh_radii_accessors_are_collective():
    """The @uw.collective_operation decorator must be present so a
    selective_ranks() misuse fails fast rather than deadlocking."""
    import underworld3 as uw
    mesh = uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5,
        cellSize=1.0 / 16, qdegree=3)
    for name in ("get_min_radius", "get_mean_radius",
                 "get_max_radius"):
        fn = getattr(mesh, name)
        assert getattr(fn, "_is_collective", False), (
            f"{name} must be decorated with @uw.collective_operation "
            f"so misuse inside selective_ranks() is caught early.")
