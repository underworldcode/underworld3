"""
Test suite for UWQuantity comparison operators.

Tests the implementation of comparison operators (__lt__, __le__, __gt__, __ge__, __eq__, __ne__)
for UWQuantity objects with proper unit handling and dimensional analysis.
"""

import pytest
import underworld3 as uw


class TestUWQuantityComparison:
    """Test comparison operators for UWQuantity objects."""

    def test_equality_same_units(self):
        """Test equality comparison with same units."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(1.0, "m")
        assert q1 == q2

    def test_equality_unit_conversion(self):
        """Test equality comparison with unit conversion."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(100.0, "cm")
        assert q1 == q2

    def test_inequality(self):
        """Test inequality comparison."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(2.0, "m")
        assert q1 != q2

    def test_less_than(self):
        """Test less than comparison."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(2.0, "m")
        assert q1 < q2
        assert not (q2 < q1)

    def test_greater_than(self):
        """Test greater than comparison."""
        q1 = uw.quantity(2.0, "m")
        q2 = uw.quantity(1.0, "m")
        assert q1 > q2
        assert not (q2 > q1)

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(2.0, "m")
        assert q1 <= q2
        assert q1 <= q1
        assert not (q2 <= q1)

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        q1 = uw.quantity(2.0, "m")
        q2 = uw.quantity(1.0, "m")
        assert q1 >= q2
        assert q1 >= q1
        assert not (q2 >= q1)

    def test_cross_unit_comparison(self):
        """Test comparison across units with automatic conversion."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(50.0, "cm")
        assert q1 > q2
        assert q2 < q1

    def test_dimensionless_quantities(self):
        """Test comparison of dimensionless quantities."""
        q1 = uw.quantity(1.0)
        q2 = uw.quantity(2.0)
        assert q1 < q2
        assert q2 > q1

    def test_incompatible_units_strict_comparison_error(self):
        """Test that strict comparisons raise error for incompatible units."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(1.0, "s")
        with pytest.raises(ValueError, match="Cannot compare __lt__"):
            q1 < q2
        with pytest.raises(ValueError, match="Cannot compare __gt__"):
            q1 > q2

    def test_incompatible_units_equality_returns_false(self):
        """Test that equality returns False for incompatible units."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(1.0, "s")
        assert q1 != q2
        assert not (q1 == q2)

    def test_compare_with_units_to_scalar_error(self):
        """Test error when comparing quantity with units to plain scalar."""
        q1 = uw.quantity(1.0, "m")
        with pytest.raises(TypeError, match="Cannot compare"):
            q1 > 1.0
        with pytest.raises(TypeError, match="Cannot compare"):
            q1 < 1.0

    def test_dimensionless_with_scalar(self):
        """Test comparing dimensionless quantity with plain scalar."""
        q1 = uw.quantity(1.0)
        assert q1 < 2.0
        assert q1 <= 1.0
        assert not (q1 > 2.0)

    def test_temperature_threshold_example(self):
        """Practical example: temperature threshold comparison."""
        mantle_T = uw.quantity(1300, "K")
        melt_T = uw.quantity(1400, "K")
        surface_T = uw.quantity(273.15, "K")

        assert mantle_T < melt_T
        assert mantle_T > surface_T
        assert surface_T < mantle_T < melt_T

    def test_plate_velocity_limits_example(self):
        """Practical example: plate velocity limits."""
        plate_velocity = uw.quantity(5, "cm/year")
        max_velocity = uw.quantity(100, "mm/year")
        min_velocity = uw.quantity(2, "cm/year")

        assert plate_velocity <= max_velocity
        assert plate_velocity >= min_velocity
        assert min_velocity <= plate_velocity <= max_velocity

    def test_velocity_unit_conversion(self):
        """Test velocity comparison with multiple unit systems."""
        q1 = uw.quantity(1.0, "m/s")
        q2 = uw.quantity(3.6, "km/hour")
        assert q1 == q2

    def test_pressure_comparison(self):
        """Test pressure comparison across different scales."""
        q1 = uw.quantity(1000, "Pa")
        q2 = uw.quantity(1, "kPa")
        assert q1 == q2

        q3 = uw.quantity(1, "GPa")
        assert q3 > q1


class TestUWQuantityComparisonEdgeCases:
    """Test edge cases for comparison operators."""

    def test_zero_quantities(self):
        """Test comparison with zero quantities."""
        q1 = uw.quantity(0.0, "m")
        q2 = uw.quantity(0.0, "m")
        q3 = uw.quantity(1.0, "m")
        assert q1 == q2
        assert q1 < q3
        assert q3 > q1

    def test_negative_quantities(self):
        """Test comparison with negative quantities."""
        q1 = uw.quantity(-1.0, "m")
        q2 = uw.quantity(1.0, "m")
        assert q1 < q2
        assert q1 < 0

    def test_small_difference_comparison(self):
        """Test comparison with very small differences."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(1.0 + 1e-10, "m")
        # Should be slightly different due to floating point
        assert q1 <= q2

    def test_comparison_transitivity(self):
        """Test that comparison operators satisfy transitivity."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(2.0, "m")
        q3 = uw.quantity(3.0, "m")

        # Transitivity: a < b and b < c implies a < c
        assert q1 < q2 and q2 < q3 and q1 < q3
        assert q1 <= q2 and q2 <= q3 and q1 <= q3

    def test_comparison_reflexivity(self):
        """Test that comparison operators satisfy reflexivity."""
        q = uw.quantity(1.0, "m")
        assert q == q
        assert q <= q
        assert q >= q

    def test_comparison_symmetry(self):
        """Test symmetry of equality."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(100.0, "cm")
        assert (q1 == q2) == (q2 == q1)

    def test_very_large_differences(self):
        """Test comparison with very large differences."""
        q1 = uw.quantity(1e-20, "m")
        q2 = uw.quantity(1e20, "m")
        assert q1 < q2
        assert q2 > q1


class TestUWQuantityComparisonUnits:
    """Test comparison operators with various unit systems."""

    def test_si_length_comparison(self):
        """Test SI length unit comparisons."""
        q1 = uw.quantity(1.0, "m")
        q2 = uw.quantity(1000, "mm")
        q3 = uw.quantity(0.001, "km")
        assert q1 == q2 == q3

    def test_time_comparison(self):
        """Test time unit comparisons."""
        q1 = uw.quantity(60, "s")
        q2 = uw.quantity(1, "minute")
        assert q1 == q2

    def test_geological_time_comparison(self):
        """Test geological time scale comparison."""
        q1 = uw.quantity(1e6, "year")
        q2 = uw.quantity(1, "Myr")
        assert q1 == q2

    def test_energy_comparison(self):
        """Test energy unit comparisons."""
        q1 = uw.quantity(1000, "J")
        q2 = uw.quantity(1, "kJ")
        assert q1 == q2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
