#!/usr/bin/env python3
"""
Unit tests for UnitAwareArray - Integration of units with NDArray_With_Callback.

Tests the integration of the universal units system with array data structures:
- Unit tracking and propagation through operations
- Unit compatibility checking and automatic conversion
- Preservation of callback functionality
- Integration with existing UW3 unit conversion utilities
"""

import os
import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw
from underworld3.utilities import UnitAwareArray, create_unit_aware_array, zeros_with_units


def test_unit_aware_array_basic_creation():
    """Test basic creation of UnitAwareArray."""
    # Create array with units
    length = UnitAwareArray([1, 2, 3], units="m")

    assert length.has_units == True
    assert length.units == "m"
    assert np.array_equal(length, [1, 2, 3])

    # Create array without units
    dimensionless = UnitAwareArray([1, 2, 3])

    assert dimensionless.has_units == False
    assert dimensionless.units is None


def test_unit_aware_array_arithmetic_operations():
    """Test arithmetic operations with unit checking."""
    length = UnitAwareArray([1, 2, 3], units="m")
    time = UnitAwareArray([1, 2, 3], units="s")

    # Scalar multiplication preserves units
    doubled_length = length * 2
    assert doubled_length.units == "m"
    assert np.array_equal(doubled_length, [2, 4, 6])

    # Same units can be added
    more_length = UnitAwareArray([4, 5, 6], units="m")
    total_length = length + more_length
    assert total_length.units == "m"
    assert np.array_equal(total_length, [5, 7, 9])

    # Incompatible units raise error
    with pytest.raises(ValueError, match="Cannot add arrays with incompatible units"):
        length + time


def test_unit_aware_array_unit_conversion():
    """Test unit conversion functionality."""
    length_m = UnitAwareArray([1, 2, 3], units="m")

    # Convert to different units
    length_mm = length_m.to("mm")
    assert length_mm.units == "millimeter"
    assert np.allclose(np.asarray(length_mm), [1000, 2000, 3000])

    # Test automatic conversion in operations
    length_km = UnitAwareArray([0.001, 0.002, 0.003], units="km")

    # With auto_convert enabled, this should work
    total = length_m + length_km
    assert total.units == "m"
    assert np.allclose(np.asarray(total), [2, 4, 6])  # 1m + 1m, 2m + 2m, 3m + 3m


def test_unit_aware_array_callback_preservation():
    """Test that callback functionality is preserved."""
    callback_calls = []

    def test_callback(array, change_info):
        callback_calls.append(
            {
                "operation": change_info["operation"],
                "units": array.units,
                "new_value": change_info["new_value"],
            }
        )

    # Create array with callback
    velocity = UnitAwareArray([1, 2, 3], units="m/s")
    velocity.set_callback(test_callback)

    # Modify array - should trigger callback
    velocity[0] = 10

    assert len(callback_calls) == 1
    assert callback_calls[0]["operation"] == "setitem"
    assert callback_calls[0]["units"] == "m/s"
    assert callback_calls[0]["new_value"] == 10


def test_unit_aware_array_numpy_methods():
    """Test that numpy methods preserve units."""
    data = UnitAwareArray([[1, 2], [3, 4]], units="kg")

    # Test reshape
    reshaped = data.reshape(4)
    assert reshaped.units == "kg"
    assert reshaped.shape == (4,)

    # Test transpose
    transposed = data.transpose()
    assert transposed.units == "kg"
    assert transposed.shape == (2, 2)

    # Test copy
    copied = data.copy()
    assert copied.units == "kg"
    assert np.array_equal(np.asarray(copied), np.asarray(data))

    # Test view
    viewed = data.view()
    assert viewed.units == "kg"


def test_unit_aware_array_convenience_functions():
    """Test convenience functions for creating unit-aware arrays."""
    # Test zeros_with_units
    zeros_array = zeros_with_units((3, 2), units="Pa")
    assert zeros_array.units == "Pa"
    assert zeros_array.shape == (3, 2)
    assert np.all(zeros_array == 0)

    # Test create_unit_aware_array
    created_array = create_unit_aware_array([1, 2, 3], units="N")
    assert created_array.units == "N"
    assert np.array_equal(created_array, [1, 2, 3])


def test_unit_aware_array_unit_checking_toggle():
    """Test enabling/disabling unit checking."""
    length = UnitAwareArray([1, 2, 3], units="m")
    time = UnitAwareArray([1, 2, 3], units="s")

    # With checking enabled (default), incompatible addition fails
    with pytest.raises(ValueError):
        length + time

    # Disable unit checking
    length.unit_checking = False

    # Now incompatible addition works (but result units may be incorrect)
    result = length + time
    assert np.array_equal(result, [2, 4, 6])


def test_unit_aware_array_edge_cases():
    """Test edge cases and error handling."""
    # Test division by zero with units
    velocity = UnitAwareArray([1, 2, 0], units="m/s")
    time = UnitAwareArray([1, 0, 1], units="s")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = velocity / time
        # Should preserve the operation despite division by zero
        assert hasattr(result, "units")

    # Test conversion to invalid units
    length = UnitAwareArray([1, 2, 3], units="m")
    with pytest.raises(ValueError):
        length.to("invalid_unit")


def test_unit_aware_array_integration_with_uw_quantities():
    """Test integration with UW3 quantity system."""
    # Create UW quantity
    length_qty = uw.function.quantity(5, "km")

    # Create UnitAwareArray from quantity
    length_array = UnitAwareArray([1, 2, 3], units="km")

    # Test that units are compatible
    assert length_array.units == "km"

    # Test conversion using UW3 system
    length_m = length_array.to("m")
    assert np.allclose(np.asarray(length_m), [1000, 2000, 3000])


def test_unit_aware_array_with_scaling():
    """Test UnitAwareArray with model scaling context."""
    # Set up model with scaling
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin,
    )

    # Create unit-aware arrays
    physical_coords = UnitAwareArray([[1_000_000, 500_000]], units="m")

    # Test that arrays work with scaled coordinate systems
    assert physical_coords.has_units == True
    assert physical_coords.units == "m"

    # Convert to km
    coords_km = physical_coords.to("km")
    assert np.allclose(np.asarray(coords_km), [[1000, 500]])


def test_unit_aware_array_mathematical_operations():
    """Test mathematical operations preserve unit relationships."""
    # Create arrays with different units
    distance = UnitAwareArray([100, 200, 300], units="m")
    time = UnitAwareArray([10, 20, 30], units="s")

    # Division should work but result units are simplified
    # Note: Full Pint integration would give proper "m/s" units
    velocity = distance / time
    assert np.allclose(np.asarray(velocity), [10, 10, 10])

    # Multiplication
    area_factor = UnitAwareArray([2, 3, 4], units="m")
    # Note: This would ideally result in "m^2" units with full Pint integration
    area = distance * area_factor
    assert np.allclose(np.asarray(area), [200, 600, 1200])


def test_unit_aware_array_repr_and_str():
    """Test string representations include unit information."""
    velocity = UnitAwareArray([1, 2, 3], units="m/s")

    # Test __repr__ includes units
    repr_str = repr(velocity)
    assert "units='m/s'" in repr_str

    # Test __str__ includes units
    str_repr = str(velocity)
    assert "[m/s]" in str_repr


if __name__ == "__main__":
    # Run tests individually for debugging
    test_unit_aware_array_basic_creation()
    test_unit_aware_array_arithmetic_operations()
    test_unit_aware_array_unit_conversion()
    test_unit_aware_array_callback_preservation()
    test_unit_aware_array_numpy_methods()
    test_unit_aware_array_convenience_functions()
    test_unit_aware_array_unit_checking_toggle()
    test_unit_aware_array_edge_cases()
    test_unit_aware_array_integration_with_uw_quantities()
    test_unit_aware_array_with_scaling()
    test_unit_aware_array_mathematical_operations()
    test_unit_aware_array_repr_and_str()
    print("All UnitAwareArray tests passed!")
