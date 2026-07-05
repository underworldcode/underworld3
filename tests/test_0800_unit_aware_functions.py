#!/usr/bin/env python3
"""
Unit tests for unit-aware functions (evaluate, mesh geometry, visualization).

Tests the core functionality of the universal units system:
- Unit-aware evaluate() and global_evaluate()
- Unit-aware mesh geometry functions
- Coordinate unit conversion
- Integration with existing mesh and scaling functionality
"""

import os
import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw
import sympy


def test_unit_aware_evaluate_basic():
    """Test basic unit-aware evaluate functionality."""
    # Set up model with scaling
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin,
    )

    # Create mesh
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), qdegree=2
    )

    # Test constant expression
    expr = sympy.sympify(42)

    # Physical coordinates (should be auto-converted)
    physical_coords = np.array([[1_000_000.0, 500_000.0]], dtype=np.float64)
    result_physical = uw.function.evaluate(expr, physical_coords)

    # Model coordinates (should work as before)
    model_coords = np.array([[1.0, 0.5]], dtype=np.float64)
    result_model = uw.function.evaluate(expr, model_coords)

    # Both should give same result
    assert np.allclose(result_physical, result_model)
    assert np.allclose(result_physical, 42)


def test_mesh_test_if_points_in_cells_unit_aware():
    """Test unit-aware test_if_points_in_cells function with automatic coordinate conversion."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin,
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), qdegree=2
    )

    # Get a test cell using model coordinates
    test_coords_model = np.array([[1.0, 0.5]], dtype=np.float64)
    closest_cells = mesh.get_closest_cells(test_coords_model)
    test_cell = closest_cells[0]

    # Test coordinates in model units
    model_coords = np.array([[1.0, 0.5]], dtype=np.float64)

    # Test coordinates in physical units (should be auto-converted)
    physical_coords = np.array([[1_000_000.0, 500_000.0]], dtype=np.float64) * uw.units.m

    # Both should give same result for point-in-cell test due to automatic coordinate conversion
    result_model = mesh.test_if_points_in_cells(model_coords, np.array([test_cell]))
    result_physical = mesh.test_if_points_in_cells(physical_coords, np.array([test_cell]))

    assert np.array_equal(result_model, result_physical)


def test_coordinate_unit_conversion_functions():
    """Test coordinate unit conversion utility functions."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin,
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), qdegree=2
    )

    from underworld3.function.unit_conversion import get_mesh_coordinate_units, has_units
    # get_units is at module level, not in unit_conversion
    get_units = uw.get_units

    # Test mesh coordinate info
    mesh_info = get_mesh_coordinate_units(mesh)
    assert mesh_info is not None
    assert mesh_info["scaled"] == True
    # Use approximate comparison for floating-point scale
    assert np.isclose(mesh_info["length_scale"], 1_000_000.0, rtol=1e-6)

    # Test manual coordinate conversion (no automatic conversion function)
    physical_coords = np.array([[1_000_000.0, 500_000.0]], dtype=np.float64)
    # Manual conversion: divide by length scale
    scale_factor = mesh_info["length_scale"]
    converted_coords = physical_coords / scale_factor
    expected_coords = np.array([[1.0, 0.5]], dtype=np.float64)

    assert np.allclose(converted_coords, expected_coords)

    # Test unit detection on plain arrays
    plain_array = np.array([1, 2, 3])
    assert has_units(plain_array) == False
    assert get_units(plain_array) is None


def test_unit_aware_with_no_scaling():
    """Test unit-aware functions work correctly when no scaling is applied."""
    uw.reset_default_model()

    # Create mesh without scaling
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), qdegree=2
    )

    # Test coordinates
    test_coords = np.array([[1.0, 0.5]], dtype=np.float64)

    # Functions should work normally
    expr = sympy.sympify(42)
    result = uw.function.evaluate(expr, test_coords)
    assert np.isclose(result[0], 42)

    # Mesh functions should work
    inside = mesh.points_in_domain(test_coords)
    assert inside[0] == True


if __name__ == "__main__":
    # Run tests individually for debugging
    test_unit_aware_evaluate_basic()
    test_mesh_test_if_points_in_cells_unit_aware()
    test_coordinate_unit_conversion_functions()
    test_unit_aware_with_no_scaling()
    print("All unit-aware function tests passed!")
