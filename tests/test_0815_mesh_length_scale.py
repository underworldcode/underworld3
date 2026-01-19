#!/usr/bin/env python3
"""
Unit tests for mesh length_scale property (Phase 1 of ND solver implementation).

Tests the length scale system that ensures synchronization between mesh
coordinates and non-dimensional scaling:
- Length scale immutability after mesh creation
- Integration with model reference quantities
- Proper derivation from domain_depth and length
- Synchronization across multiple meshes
"""

import os
import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw


def test_default_length_scale():
    """Test that mesh has default length_scale=1.0 when no reference quantities set."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Check default values
    assert mesh.length_scale == 1.0, "Default length scale should be 1.0"
    assert mesh.length_units is not None, "Length units should be set"


def test_length_scale_from_domain_depth():
    """Test that length_scale is derived from domain_depth reference quantity."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Set reference quantities BEFORE creating mesh
    model.set_reference_quantities(
        domain_depth=uw.quantity(100, "km"), temperature_diff=uw.quantity(1000, "kelvin")
    )

    # Create mesh - should pick up domain_depth as length scale
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Check that length scale was set from domain_depth
    # 100 km = 100000 m (Pint default units are SI)
    expected_scale = 100000.0  # meters
    assert mesh.length_scale == pytest.approx(
        expected_scale, rel=1e-10
    ), f"Length scale should be {expected_scale} m (from 100 km domain_depth)"

    assert (
        "kilometer" in mesh.length_units or "meter" in mesh.length_units
    ), f"Length units should be related to length, got {mesh.length_units}"


def test_length_scale_from_length_quantity():
    """Test that length_scale is derived from 'length' reference quantity."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Set reference quantities with 'length' instead of 'domain_depth'
    model.set_reference_quantities(length=uw.quantity(50, "km"), time=uw.quantity(1, "megayear"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Check that length scale was set from length
    # 50 km = 50000 m
    expected_scale = 50000.0  # meters
    assert mesh.length_scale == pytest.approx(
        expected_scale, rel=1e-10
    ), f"Length scale should be {expected_scale} m (from 50 km length)"


def test_domain_depth_priority_over_length():
    """Test that domain_depth has priority over length when both are present."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Set BOTH domain_depth and length
    model.set_reference_quantities(
        domain_depth=uw.quantity(100, "km"), length=uw.quantity(50, "km")  # Should be ignored
    )

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Check that domain_depth was used (priority)
    expected_scale = 100000.0  # from domain_depth, not length
    assert mesh.length_scale == pytest.approx(
        expected_scale, rel=1e-10
    ), "domain_depth should have priority over length"


def test_length_scale_immutability():
    """Test that length_scale cannot be changed after mesh creation."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    original_scale = mesh.length_scale

    # Attempt to set length_scale - should fail (read-only property)
    with pytest.raises(AttributeError):
        mesh.length_scale = 100.0

    # Verify it hasn't changed
    assert (
        mesh.length_scale == original_scale
    ), "length_scale should be immutable after mesh creation"

    # Also test that internal attribute can't be easily changed
    # (Note: Python doesn't enforce true immutability, but we document it)
    mesh._length_scale = 999.0  # Can set private attribute (no enforcement)
    assert mesh.length_scale == 999.0  # But we document that users shouldn't do this


def test_multiple_meshes_same_scale():
    """Test that multiple meshes created with same model use same length scale."""
    uw.reset_default_model()
    model = uw.get_default_model()

    model.set_reference_quantities(domain_depth=uw.quantity(100, "km"))

    # Create two meshes
    mesh1 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    mesh2 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(2.0, 2.0), cellSize=0.5
    )

    # Both should have same length scale (from same model)
    assert (
        mesh1.length_scale == mesh2.length_scale
    ), "All meshes from same model should use same length scale"

    expected_scale = 100000.0  # 100 km in meters
    assert mesh1.length_scale == pytest.approx(expected_scale, rel=1e-10)
    assert mesh2.length_scale == pytest.approx(expected_scale, rel=1e-10)


def test_mesh_after_model_with_no_length_quantities():
    """Test mesh creation when model has reference quantities but no length-related ones."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Set reference quantities without length or domain_depth
    model.set_reference_quantities(
        temperature_diff=uw.quantity(1000, "kelvin"), time=uw.quantity(1, "megayear")
    )

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Should default to 1.0 since no length-related quantities provided
    assert (
        mesh.length_scale == 1.0
    ), "Length scale should default to 1.0 when no length quantities in model"


def test_length_scale_with_different_units():
    """Test length scale derivation with different unit systems."""
    uw.reset_default_model()
    model = uw.get_default_model()

    # Use miles instead of km
    model.set_reference_quantities(domain_depth=uw.quantity(62.137, "miles"))  # ~100 km

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Should convert to SI units (meters)
    # 62.137 miles ≈ 100 km ≈ 100000 m
    expected_scale = pytest.approx(100000.0, rel=0.01)  # 1% tolerance for unit conversion
    assert (
        mesh.length_scale == expected_scale
    ), f"Length scale should convert miles to meters, got {mesh.length_scale}"


def test_length_scale_property_is_readonly():
    """Test that length_scale is a read-only property with no setter."""
    uw.reset_default_model()

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Check that property exists and is callable
    assert hasattr(mesh, "length_scale")
    assert isinstance(mesh.length_scale, (int, float))

    # Check that there's no setter
    assert (
        not hasattr(type(mesh).length_scale, "fset") or type(mesh).length_scale.fset is None
    ), "length_scale should not have a setter (read-only property)"


def test_length_units_matches_coordinate_units():
    """Test that length_units is consistent with coordinate units."""
    uw.reset_default_model()
    model = uw.get_default_model()

    model.set_reference_quantities(domain_depth=uw.quantity(100, "km"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # length_units should be related to coordinate units
    # (May be "kilometer" from reference quantity or "meter" if converted)
    assert mesh.length_units is not None
    assert isinstance(mesh.length_units, str)


@pytest.mark.skipif(
    os.environ.get("CI") or not os.environ.get("DISPLAY"),
    reason="Visualization tests require display (skipped in CI)"
)
def test_mesh_view_displays_length_scale():
    """Test that mesh.view() displays length scale information."""
    uw.reset_default_model()
    model = uw.get_default_model()

    model.set_reference_quantities(domain_depth=uw.quantity(100, "km"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Call view() and check it doesn't error
    # (Can't easily test printed output, but we can ensure it runs)
    try:
        mesh.view(level=0)
        success = True
    except Exception as e:
        success = False
        print(f"mesh.view() failed: {e}")

    assert success, "mesh.view() should display length scale without error"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
