"""
Test power operations on UWQuantity objects.

This test suite validates the fix for the power operation bug where
L0**3 was incorrectly returning 'meter' instead of 'meter ** 3'.

The bug was caused by __pow__() using the vestigial _units_backend
attribute (which is never set) instead of the modern _pint_qty approach.

Test Coverage:
- Integer powers (squared, cubed, etc.)
- Fractional powers (square root, etc.)
- Negative powers (inverse operations)
- Power vs multiplication equivalence
- Unit conversion after power operations
- Edge cases (zero, one)
"""

import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import underworld3 as uw


def test_integer_powers():
    """Test integer power operations."""
    L0 = uw.quantity(2900, "km").to("m")

    # Test L0**2
    L0_squared = L0**2
    assert str(L0_squared.units) == "meter ** 2", f"Expected 'meter ** 2', got '{L0_squared.units}'"
    assert abs(L0_squared.value - (2900000.0**2)) < 1e-6

    # Test L0**3 (the original bug case)
    L0_cubed = L0**3
    assert str(L0_cubed.units) == "meter ** 3", f"Expected 'meter ** 3', got '{L0_cubed.units}'"
    assert abs(L0_cubed.value - (2900000.0**3)) < 1e10

    # Test L0**4
    L0_fourth = L0**4
    assert str(L0_fourth.units) == "meter ** 4", f"Expected 'meter ** 4', got '{L0_fourth.units}'"


def test_fractional_powers():
    """Test fractional power operations."""
    L0 = uw.quantity(2900, "km").to("m")

    # Square root
    L0_sqrt = L0**0.5
    assert str(L0_sqrt.units) == "meter ** 0.5", f"Expected 'meter ** 0.5', got '{L0_sqrt.units}'"
    assert abs(L0_sqrt.value - (2900000.0**0.5)) < 1e-6

    # Cube root
    L0_cbrt = L0 ** (1.0 / 3.0)
    # Pint truncates floating point precision in unit strings
    assert "meter ** 0.333" in str(L0_cbrt.units), f"Expected 'meter ** 0.333...', got '{L0_cbrt.units}'"


def test_negative_powers():
    """Test negative power operations (inverse)."""
    L0 = uw.quantity(2900, "km").to("m")

    # Inverse (1/L0 = L0**-1)
    L0_inv = L0 ** (-1)
    assert str(L0_inv.units) == "1 / meter", f"Expected '1 / meter', got '{L0_inv.units}'"
    assert abs(L0_inv.value - (1.0 / 2900000.0)) < 1e-12

    # Inverse squared
    L0_inv_sq = L0 ** (-2)
    assert (
        str(L0_inv_sq.units) == "1 / meter ** 2"
    ), f"Expected '1 / meter ** 2', got '{L0_inv_sq.units}'"


def test_power_multiplication_equivalence():
    """Test that power and repeated multiplication give same results."""
    L0 = uw.quantity(2900, "km").to("m")

    # L0**2 should equal L0 * L0
    pow_result = L0**2
    mult_result = L0 * L0
    assert pow_result.units == mult_result.units
    assert abs(pow_result.value - mult_result.value) < 1e-6

    # L0**3 should equal L0 * L0 * L0 (original bug report)
    pow_result = L0**3
    mult_result = L0 * L0 * L0
    assert (
        pow_result.units == mult_result.units
    ), f"Power: {pow_result.units}, Mult: {mult_result.units}"
    assert abs(pow_result.value - mult_result.value) < 1e10


def test_unit_conversion_after_power():
    """Test that power operations preserve unit conversion capability."""
    L0 = uw.quantity(2900, "km")

    # Square the quantity
    L0_squared = L0**2
    assert str(L0_squared.units) == "kilometer ** 2"

    # Convert to m^2
    L0_squared_m = L0_squared.to("meter ** 2")
    assert str(L0_squared_m.units) == "meter ** 2"
    assert abs(L0_squared_m.value - (2900000.0**2)) < 1e-6


def test_power_of_different_units():
    """Test power operations on various unit types."""
    # Test with velocity
    v = uw.quantity(5, "cm/year").to("m/s")
    v_squared = v**2
    assert "meter ** 2" in str(v_squared.units)
    assert "second ** 2" in str(v_squared.units) or "second ** -2" in str(v_squared.units)

    # Test with temperature (no powers, just checking it doesn't break)
    T = uw.quantity(3000, "K")
    T_squared = T**2
    assert "kelvin ** 2" in str(T_squared.units)

    # Test with pressure-like quantity
    P = uw.quantity(1e21, "Pa")
    P_squared = P**2
    assert "pascal ** 2" in str(P_squared.units)


def test_edge_cases():
    """Test edge case powers."""
    L0 = uw.quantity(2900, "km").to("m")

    # Power of 0 should give dimensionless 1
    L0_zero = L0**0
    assert abs(L0_zero.value - 1.0) < 1e-12
    # Note: Pint returns dimensionless for x**0

    # Power of 1 should preserve value and units
    L0_one = L0**1
    assert str(L0_one.units) == "meter"
    assert abs(L0_one.value - 2900000.0) < 1e-6


def test_power_with_composite_dimensions():
    """Test power operations on quantities with composite dimensions."""
    # Create a quantity with composite dimensions (e.g., m^2/s)
    diffusivity = uw.quantity(1.0, "meter**2 / second")

    # Square it
    diff_squared = diffusivity**2
    assert "meter ** 4" in diff_squared.units
    assert "second" in diff_squared.units


def test_rayleigh_number_calculation():
    """
    Test the specific Rayleigh number calculation from Notebook 14
    that revealed the bug.
    """
    # Physical parameters from the notebook
    L0 = uw.quantity(2900, "km").to("m")
    rho0 = uw.quantity(3300, "kg/m^3")
    alpha = uw.quantity(3e-5, "1/K")
    g = uw.quantity(10, "m/s^2")
    DeltaT = uw.quantity(3000, "K")
    eta0 = uw.quantity(1e21, "Pa*s")
    kappa = uw.quantity(1e-6, "m^2/s")

    # This calculation should work with L0**3 (was buggy before)
    numerator = rho0 * alpha * g * DeltaT * L0**3
    denominator = eta0 * kappa

    Ra_quantity = numerator / denominator

    # Ra should be dimensionless
    assert (
        Ra_quantity.units is None or str(Ra_quantity.units) == "dimensionless"
    ), f"Ra should be dimensionless, got: {Ra_quantity.units}"

    # Ra should be on the order of 1e7 for these parameters
    Ra_value = float(Ra_quantity.value)
    assert 1e6 < Ra_value < 1e8, f"Ra = {Ra_value:.3e} is outside expected range [1e6, 1e8]"


def test_power_preserves_pint_qty():
    """Test that power operations preserve the _pint_qty attribute."""
    L0 = uw.quantity(2900, "km").to("m")

    # Verify input has _pint_qty
    assert hasattr(L0, "_pint_qty"), "Input should have _pint_qty"
    assert L0._has_pint_qty, "Input should have _has_pint_qty=True"

    # Test that result also has _pint_qty
    L0_cubed = L0**3
    assert hasattr(L0_cubed, "_pint_qty"), "Result should have _pint_qty"
    assert L0_cubed._has_pint_qty, "Result should have _has_pint_qty=True"

    # Verify the _pint_qty has correct units
    assert str(L0_cubed._pint_qty.units) == "meter ** 3"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
