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
        mantle_temperature=1500 * uw.units.kelvin
    )

    # Create mesh
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
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


@pytest.mark.skip(reason="UnitAwareArray return type not implemented - planned feature for evaluate()")
def test_unit_aware_evaluate_coordinate_expressions():
    """Test unit-aware evaluate with coordinate-dependent expressions."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
    )

    # Expression using physical coordinate symbols (scaled mesh coordinates)
    x_phys, y_phys = mesh.X  # Physical coordinates (automatically scaled)
    expr_phys = x_phys  # Simple x coordinate expression

    # Expression using model coordinate symbols
    x_model, y_model = mesh.N.x, mesh.N.y  # Model coordinates
    expr_model = x_model  # Simple x coordinate expression

    # Test point - coordinates provided in model units for evaluation
    model_coords = np.array([[1.0, 0.5]], dtype=np.float64)  # 1.0, 0.5 model units

    # Evaluate physical expression with model coords (physical expression is scaled automatically)
    result_phys = uw.function.evaluate(expr_phys, model_coords)

    # Evaluate model expression with model coords
    result_model = uw.function.evaluate(expr_model, model_coords)

    # Physical expression should return scaled coordinate value (in physical units)
    # 1.0 model unit * 1000 km scale = 1_000_000.0 m
    assert np.isclose(result_phys[0], 1_000_000.0, rtol=1e-6)

    # Model expression should return model coordinate value with units
    # Both should be UnitAwareArray objects with length units
    assert hasattr(result_model, '_units'), f"Expected UnitAwareArray with units, got {type(result_model)}"
    assert np.isclose(result_model[0], 1.0, rtol=1e-6)
    result_units_str = str(result_model._units) if result_model._units else ""
    assert "kilometer" in result_units_str or "meter" in result_units_str or "km" in result_units_str or "m" in result_units_str


def test_mesh_points_in_domain_unit_aware():
    """Test points_in_domain function with automatic coordinate conversion."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
    )

    # Test points in model coordinates
    model_points = np.array([
        [1.0, 0.5],   # Center (should be inside)
        [0.5, 0.25],  # Quarter (should be inside)
        [2.5, 0.5],   # Outside
    ], dtype=np.float64)

    # Test points in physical coordinates (should be auto-converted)
    physical_points = np.array([
        [1_000_000.0, 500_000.0],    # Center in meters
        [500_000.0, 250_000.0],     # Quarter in meters
        [2_500_000.0, 500_000.0],   # Outside in meters
    ], dtype=np.float64) * uw.units.m

    # Both should give same results due to automatic coordinate conversion
    result_model = mesh.points_in_domain(model_points)
    result_physical = mesh.points_in_domain(physical_points)

    # Results should be the same since coordinates represent same physical locations
    assert np.array_equal(result_model, result_physical)
    assert result_model[0] == True   # Center inside
    assert result_model[1] == True   # Quarter inside
    assert result_model[2] == False  # Outside


def test_mesh_get_closest_cells_unit_aware():
    """Test unit-aware get_closest_cells function with automatic coordinate conversion."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
    )

    # Test coordinates in model units
    model_coords = np.array([[1.0, 0.5]], dtype=np.float64)

    # Test coordinates in physical units (should be auto-converted)
    physical_coords = np.array([[1_000_000.0, 500_000.0]], dtype=np.float64) * uw.units.m

    # Both should find same closest cells due to automatic coordinate conversion
    cells_model = mesh.get_closest_cells(model_coords)
    cells_physical = mesh.get_closest_cells(physical_coords)

    assert np.array_equal(cells_model, cells_physical)


def test_mesh_test_if_points_in_cells_unit_aware():
    """Test unit-aware test_if_points_in_cells function with automatic coordinate conversion."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
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
        mantle_temperature=1500 * uw.units.kelvin
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
    )

    from underworld3.function.unit_conversion import (
        get_mesh_coordinate_units,
        has_units,
        get_units
    )

    # Test mesh coordinate info
    mesh_info = get_mesh_coordinate_units(mesh)
    assert mesh_info is not None
    assert mesh_info['scaled'] == True
    assert mesh_info['length_scale'] == 1_000_000.0

    # Test manual coordinate conversion (no automatic conversion function)
    physical_coords = np.array([[1_000_000.0, 500_000.0]], dtype=np.float64)
    # Manual conversion: divide by length scale
    scale_factor = mesh_info['length_scale']
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
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(2.0, 1.0),
        qdegree=2
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
    test_unit_aware_evaluate_coordinate_expressions()
    test_mesh_points_in_domain_unit_aware()
    test_mesh_get_closest_cells_unit_aware()
    test_mesh_test_if_points_in_cells_unit_aware()
    test_coordinate_unit_conversion_functions()
    test_unit_aware_with_no_scaling()
    print("All unit-aware function tests passed!")