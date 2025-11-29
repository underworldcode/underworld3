"""
Test that meshVariable.coords returns properly dimensionalised coordinates.

This test validates the fix for the bug where T.coords was returning non-dimensional
values [0-1] but claiming units of "meters", causing a factor of 1e6 error for a
1000 km domain.
"""

import pytest
import numpy as np
import underworld3 as uw


def test_variable_coords_no_units():
    """Test that coords work correctly without reference quantities."""
    uw.reset_default_model()

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )

    # Create variables with different degrees
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

    # Without units, coords should be non-dimensional [0-1]
    assert T.coords.min() >= 0.0
    assert T.coords.max() <= 1.0
    assert p.coords.min() >= 0.0
    assert p.coords.max() <= 1.0

    # mesh.X.coords should also be dimensionless
    assert mesh.X.coords.min() == 0.0
    assert mesh.X.coords.max() == 1.0


def test_variable_coords_with_units():
    """Test that coords are properly dimensionalised when using reference quantities."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Set reference quantities for 1000 km domain
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )

    # Create variable with units
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # Check that coords are properly dimensionalised to meters
    coords = T.coords

    # Verify units are present
    assert hasattr(coords, "units")
    assert coords.units == "meter"

    # Verify coordinate bounds are in physical units (meters, not [0-1])
    # 1000 km = 1e6 meters
    expected_max = 1e6
    assert np.isclose(np.asarray(coords).min(), 0.0, rtol=1e-5)
    assert np.isclose(np.asarray(coords).max(), expected_max, rtol=1e-5)

    # Verify mesh.X.coords matches
    assert np.isclose(np.asarray(mesh.X.coords).min(), 0.0, rtol=1e-5)
    assert np.isclose(np.asarray(mesh.X.coords).max(), expected_max, rtol=1e-5)


def test_variable_coords_different_degrees():
    """Test that different degree variables have different DOF locations."""
    uw.reset_default_model()
    model = uw.get_default_model()

    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )

    # Create variables with different degrees
    T_deg2 = uw.discretisation.MeshVariable("T2", mesh, 1, degree=2, units="kelvin")
    p_deg1 = uw.discretisation.MeshVariable("p1", mesh, 1, degree=1, units="pascal")

    # Different degrees should have different numbers of DOFs
    # Degree 2: (2*4+1)^2 = 81 points in 2D
    # Degree 1: (4+1)^2 = 25 points in 2D
    assert T_deg2.coords.shape[0] == 81
    assert p_deg1.coords.shape[0] == 25

    # Both should have proper dimensional scaling
    # coords.min() returns UWQuantity - extract magnitude for comparison
    assert np.isclose(T_deg2.coords.min().magnitude, 0.0, rtol=1e-5)
    assert np.isclose(T_deg2.coords.max().magnitude, 1e6, rtol=1e-5)
    assert np.isclose(p_deg1.coords.min().magnitude, 0.0, rtol=1e-5)
    assert np.isclose(p_deg1.coords.max().magnitude, 1e6, rtol=1e-5)


def test_variable_coords_consistency_with_mesh():
    """Test that variable coords are consistent with mesh coordinate bounds."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Use a different domain size to verify scaling is general
    domain_size = 2900  # km (mantle depth)
    model.set_reference_quantities(
        domain_depth=uw.quantity(domain_size, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # Get coordinate bounds
    mesh_min = mesh.X.coords.min(axis=0)
    mesh_max = mesh.X.coords.max(axis=0)
    var_min = T.coords.min(axis=0)
    var_max = T.coords.max(axis=0)

    # Variable coords should be within mesh bounds
    assert np.all(var_min >= mesh_min)
    assert np.all(var_max <= mesh_max)

    # And should match the expected physical scale (domain_size km = domain_size * 1e3 meters)
    expected_scale = domain_size * 1e3
    assert np.allclose(mesh_max, expected_scale, rtol=1e-5)
    assert np.allclose(var_max, expected_scale, rtol=1e-5)


if __name__ == "__main__":
    test_variable_coords_no_units()
    test_variable_coords_with_units()
    test_variable_coords_different_degrees()
    test_variable_coords_consistency_with_mesh()
    print("All tests passed!")
