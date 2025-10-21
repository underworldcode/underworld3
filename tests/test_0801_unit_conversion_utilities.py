#!/usr/bin/env python3
"""
Unit tests for unit conversion utility functions.

Tests the helper functions for easy unit conversion:
- convert_quantity_units()
- detect_quantity_units()
- make_dimensionless()
- convert_array_units()
- has_units() and get_units()
- add_units()
"""

import os
import pytest
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw
import underworld3.function as fn


def test_detect_quantity_units():
    """Test detect_quantity_units function."""
    # Test with UWQuantity
    length_qty = uw.function.quantity(500, 'km')
    length_info = fn.detect_quantity_units(length_qty)

    assert length_info['has_units'] == True
    assert length_info['units'] == 'kilometer'
    assert length_info['is_dimensionless'] == False
    assert length_info['unit_type'] == 'UWQuantity'

    # Test with plain array
    plain_array = np.array([1, 2, 3])
    plain_info = fn.detect_quantity_units(plain_array)

    assert plain_info['has_units'] == False
    assert plain_info['units'] is None
    assert plain_info['is_dimensionless'] == False
    assert plain_info['unit_type'] == 'none'


def test_convert_quantity_units():
    """Test convert_quantity_units function."""
    # Create test quantity
    length_km = uw.function.quantity(5, 'km')

    # Convert to meters
    length_m = fn.convert_quantity_units(length_km, 'm')

    # Check conversion
    assert hasattr(length_m, '_pint_qty')
    assert length_m._pint_qty.magnitude == 5000
    assert str(length_m._pint_qty.units) == 'meter'

    # Test with plain array (should return as-is)
    plain_array = np.array([1, 2, 3])
    result = fn.convert_quantity_units(plain_array, 'm')
    assert np.array_equal(result, plain_array)


def test_convert_array_units():
    """Test convert_array_units function."""
    # Test array conversion
    distances_km = np.array([1, 5, 10])
    distances_m = fn.convert_array_units(distances_km, 'km', 'm')

    expected_m = np.array([1000, 5000, 10000])
    assert np.allclose(distances_m, expected_m)

    # Test with temperature
    temps_c = np.array([0, 100, 200])
    temps_k = fn.convert_array_units(temps_c, 'degC', 'K')

    expected_k = np.array([273.15, 373.15, 473.15])
    assert np.allclose(temps_k, expected_k)


def test_make_dimensionless():
    """Test make_dimensionless function."""
    # Set up model with reference quantities
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    # Test length
    length_qty = uw.function.quantity(500, 'km')
    length_dimensionless = fn.make_dimensionless(length_qty, model)

    assert hasattr(length_dimensionless, '_pint_qty')
    assert np.isclose(length_dimensionless._pint_qty.magnitude, 0.5)  # 500/1000
    assert str(length_dimensionless._pint_qty.units) == 'dimensionless'

    # Test temperature
    temp_qty = uw.function.quantity(750, 'K')
    temp_dimensionless = fn.make_dimensionless(temp_qty, model)

    assert np.isclose(temp_dimensionless._pint_qty.magnitude, 0.5)  # 750/1500
    assert str(temp_dimensionless._pint_qty.units) == 'dimensionless'


def test_has_units_and_get_units():
    """Test has_units and get_units functions."""
    # Test with UWQuantity
    length_qty = uw.function.quantity(100, 'km')

    assert fn.has_units(length_qty) == True
    # get_units() returns pint.Unit objects, not strings
    assert str(fn.get_units(length_qty)) == 'kilometer'

    # Test with plain array
    plain_array = np.array([1, 2, 3])

    assert fn.has_units(plain_array) == False
    assert fn.get_units(plain_array) is None

    # Test with dimensionless quantity
    dimensionless_qty = uw.function.quantity(5, 'dimensionless')

    assert fn.has_units(dimensionless_qty) == True
    # get_units() returns pint.Unit objects, not strings
    assert str(fn.get_units(dimensionless_qty)) == 'dimensionless'


def test_add_units():
    """Test add_units function."""
    # Test adding units to plain array
    temps_array = np.array([273, 373, 473])
    temps_with_units = fn.add_units(temps_array, 'K')

    # Check that units were added correctly
    assert hasattr(temps_with_units, '_pint_qty')
    assert str(temps_with_units._pint_qty.units) == 'kelvin'
    assert np.array_equal(temps_with_units._pint_qty.magnitude, temps_array)

    # Verify unit detection works on result
    unit_info = fn.detect_quantity_units(temps_with_units)
    assert unit_info['has_units'] == True
    assert unit_info['units'] == 'kelvin'
    assert unit_info['unit_type'] == 'UWQuantity'


def test_auto_convert_to_mesh_units():
    """Test auto_convert_to_mesh_units function."""
    # Set up model with scaling
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

    # Test coordinate conversion - coordinates are already in model units
    # Model units: mesh goes from 0.0 to 2.0 in x, 0.0 to 1.0 in y
    model_coords_input = np.array([[1.0, 0.5]], dtype=np.float64)
    mesh_coords = fn.auto_convert_to_mesh_units(model_coords_input, mesh)

    expected_coords = np.array([[1.0, 0.5]], dtype=np.float64)
    assert np.allclose(mesh_coords, expected_coords)


def test_convert_evaluation_result():
    """Test convert_evaluation_result function."""
    # Create test quantity
    velocity = uw.function.quantity(100, 'cm/year')

    # Convert to different units
    velocity_ms = fn.convert_evaluation_result(velocity, 'm/s')

    # Check conversion (100 cm/year to m/s)
    expected_ms = 100 * 0.01 / (365.25 * 24 * 3600)  # cm/year to m/s
    assert np.isclose(velocity_ms._pint_qty.magnitude, expected_ms, rtol=1e-6)
    assert 'meter / second' in str(velocity_ms._pint_qty.units)


def test_utility_functions_with_edge_cases():
    """Test utility functions with edge cases."""
    # Test with zero values
    zero_length = uw.function.quantity(0, 'km')
    zero_converted = fn.convert_quantity_units(zero_length, 'm')
    assert zero_converted._pint_qty.magnitude == 0

    # Test with negative values (use simpler units for edge case testing)
    negative_length = uw.function.quantity(-10, 'm')
    negative_mm = fn.convert_quantity_units(negative_length, 'mm')
    assert np.isclose(negative_mm._pint_qty.magnitude, -10000)

    # Test with very small numbers
    small_length = uw.function.quantity(1e-9, 'km')
    small_m = fn.convert_quantity_units(small_length, 'm')
    assert np.isclose(small_m._pint_qty.magnitude, 1e-6)


def test_unit_compatibility_errors():
    """Test that incompatible unit conversions raise appropriate errors."""
    length_qty = uw.function.quantity(100, 'km')

    # This should work (length to length)
    length_m = fn.convert_quantity_units(length_qty, 'm')
    assert length_m._pint_qty.magnitude == 100000

    # Test that dimensionless conversion fails when no appropriate scale
    uw.reset_default_model()
    model = uw.get_default_model()
    # Don't set reference quantities

    with pytest.raises(ValueError):
        fn.make_dimensionless(length_qty, model)


if __name__ == "__main__":
    # Run tests individually for debugging
    test_detect_quantity_units()
    test_convert_quantity_units()
    test_convert_array_units()
    test_make_dimensionless()
    test_has_units_and_get_units()
    test_add_units()
    test_auto_convert_to_mesh_units()
    test_convert_evaluation_result()
    test_utility_functions_with_edge_cases()
    test_unit_compatibility_errors()
    print("All unit conversion utility tests passed!")