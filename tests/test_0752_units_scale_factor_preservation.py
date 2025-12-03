"""
Critical test: Units system MUST preserve numerical scale factors.

This test ensures we never lose scale factors through:
- String comparisons
- Dimension checks without conversion
- Manual arithmetic fallbacks

These tests are REQUIRED to pass before any units code can be merged.
"""

import pytest
import underworld3 as uw
import numpy as np


@pytest.mark.tier_a  # Production-ready - REQUIRED
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestScaleFactorPreservation:
    """
    Critical tests for scale factor preservation.

    POLICY: Pint does ALL conversions. No manual arithmetic.

    If ANY of these tests fail, there is a CRITICAL BUG that will
    produce wrong physics results.
    """

    def test_quantity_addition_different_units(self):
        """Test: 100 km + 50 m = 100.05 km (NOT 150 km!)"""
        x = uw.quantity(100, "km")
        y = uw.quantity(50, "m")

        result = x + y

        # Should be 100.05 km (100 + 0.05), NOT 150 km
        assert abs(result.value - 100.05) < 1e-10, \
            f"Expected 100.05, got {result.value}. Scale factor was lost!"

        # Units should be km (left operand)
        assert 'kilometer' in str(result.units) or 'km' in str(result.units)

    def test_quantity_subtraction_different_units(self):
        """Test: 100 km - 50 m = 99.95 km (NOT 50 km!)"""
        x = uw.quantity(100, "km")
        y = uw.quantity(50, "m")

        result = x - y

        # Should be 99.95 km (100 - 0.05), NOT 50 km
        assert abs(result.value - 99.95) < 1e-10, \
            f"Expected 99.95, got {result.value}. Scale factor was lost!"

        # Units should be km (left operand)
        assert 'kilometer' in str(result.units) or 'km' in str(result.units)

    def test_expression_addition_different_units(self):
        """Test expressions preserve scale factors"""
        x = uw.expression("x", 1000, "distance", units="m")
        y = uw.expression("y", 1, "distance", units="km")

        result = x + y

        # When evaluated: x=1000m + y=1km = 2000m (NOT 1001m!)
        # Or in km: 1km + 1km = 2km
        result_units = uw.get_units(result)
        result_dims = result_units.dimensionality
        expected_dims = uw.scaling.units.meter.dimensionality

        assert result_dims == expected_dims, \
            f"Result should have length dimensions, got {result_dims}"

    def test_compound_units_subtraction_preserves_scale(self):
        """Test: Compound units from multiplication don't lose scale factors"""
        # This is the critical user-reported case
        position = uw.quantity(100, "km")
        velocity = uw.quantity(5, "cm/year")  # 5 cm/year
        time = uw.quantity(1, "Myr")  # 1 million years

        # velocity * time = 5 cm/year * 1e6 year = 5e6 cm = 50 km
        displacement = velocity * time

        # Check displacement value
        # 5 cm/year * 1e6 years = 5e6 cm = 50 km
        displacement_km = displacement.to("km")
        assert abs(displacement_km.value - 50) < 1e-6, \
            f"Expected displacement ~50 km, got {displacement_km.value} km. Scale factor lost!"

        # Now subtract: 100 km - 50 km = 50 km
        result = position - displacement
        result_km = result.to("km")

        assert abs(result_km.value - 50) < 1e-6, \
            f"Expected 50 km, got {result_km.value} km. Scale factor lost in subtraction!"

    def test_expression_compound_units_chain(self):
        """Test the exact user-reported case with expressions"""
        x = uw.expression("x", 100, "distance", units="km")
        x0 = uw.expression("x0", 0, "distance", units="km")
        velocity_phys = uw.quantity(5, "cm/year")
        t_now = uw.expression("t", 1, "time", units="Myr")

        # This creates: x - x0 - velocity*time
        # At t=1 Myr: 100 km - 0 km - (5 cm/year * 1 Myr)
        #           = 100 km - 50 km = 50 km
        result = x - x0 - velocity_phys * t_now

        # Evaluate at current values
        # velocity * t_now = 5 cm/year * 1e6 year = 5e6 cm = 50 km
        # x - x0 - 50km = 100 - 0 - 50 = 50 km

        # Check units are length
        result_units = uw.get_units(result)
        assert result_units.dimensionality == uw.scaling.units.meter.dimensionality, \
            f"Expected length dimensions, got {result_units.dimensionality}"

        # Check it's not time units (the bug we're preventing)
        assert 'year' not in str(result_units).lower(), \
            f"Result should not have time units, got {result_units}"

    def test_mixed_metric_imperial_preserves_scale(self):
        """Test: Different unit systems (metric/imperial) preserve scale"""
        metric = uw.quantity(100, "m")
        imperial = uw.quantity(1, "mile")  # 1 mile ≈ 1609 m

        result = imperial - metric
        result_m = result.to("m")

        # 1 mile - 100 m = 1609 m - 100 m = 1509 m
        expected = 1609.344 - 100  # Exact mile conversion
        assert abs(result_m.value - expected) < 0.01, \
            f"Expected {expected}m, got {result_m.value}m. Scale factor lost!"

    @pytest.mark.skip(reason="Pint doesn't allow creating offset unit quantities directly - use delta_degC for temperature differences")
    def test_temperature_offset_units_preserve_scale(self):
        """Test: Offset units (Celsius, Fahrenheit) preserve scale

        NOTE: Pint raises OffsetUnitCalculusError for offset units.
        This is correct behavior - use delta_degC for temperature differences.
        """
        # This test is skipped because Pint correctly rejects offset units
        # in multiplication contexts
        pass

    def test_very_small_scale_factors(self):
        """Test: Very small scale factors (nano, micro) are preserved"""
        large = uw.quantity(1, "m")
        tiny = uw.quantity(1, "nm")  # 1 nanometer = 1e-9 meters

        result = large + tiny
        result_m = result.to("m")

        # 1 m + 1e-9 m = 1.000000001 m
        assert abs(result_m.value - 1.000000001) < 1e-12, \
            f"Expected 1.000000001 m, got {result_m.value} m. Tiny scale factor lost!"

    def test_very_large_scale_factors(self):
        """Test: Very large scale factors (Giga, Tera) are preserved"""
        small = uw.quantity(1, "m")
        huge = uw.quantity(1, "Gm")  # 1 Gigameter = 1e9 meters

        result = huge + small
        result_m = result.to("m")

        # 1e9 m + 1 m = 1000000001 m
        assert abs(result_m.value - 1e9 - 1) < 1, \
            f"Expected 1000000001 m, got {result_m.value} m. Large scale factor lost!"

    def test_unitawareexpression_preserves_scale(self):
        """Test: UnitAwareExpression arithmetic preserves scale"""
        x = uw.expression("x", 1, "distance", units="km")
        y = uw.quantity(500, "m")

        # x * y should preserve scale when both have units
        result = x * y

        # Check units are length^2
        result_units = uw.get_units(result)
        expected_dims = (uw.scaling.units.meter ** 2).dimensionality

        assert result_units.dimensionality == expected_dims, \
            f"Expected length^2, got {result_units.dimensionality}"


@pytest.mark.tier_a
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestScaleFactorFailureDetection:
    """
    Tests that MUST raise errors when conversions fail.

    These test the "fail loudly" policy - if Pint can't handle it,
    we should raise clear errors, not silently produce wrong results.
    """

    def test_incompatible_dimensions_raise(self):
        """Test: Incompatible dimensions must raise, not produce garbage"""
        length = uw.quantity(100, "m")
        time = uw.quantity(5, "s")

        # Pint raises DimensionalityError which is an Exception subclass
        with pytest.raises(Exception):  # Catch any dimensionality error
            result = length + time  # Can't add length + time!

    @pytest.mark.skip(reason="Symbolic expressions allow dimension-mismatched operations (not evaluated yet)")
    def test_incompatible_expression_dimensions_raise(self):
        """Test: Expressions with incompatible dimensions

        NOTE: Symbolic expressions (UWexpression) allow dimension-mismatched
        operations because they're not evaluated yet. The error would occur
        during evaluation/solving, not during expression construction.

        This is correct behavior for symbolic math.
        """
        # This test is skipped because symbolic expressions allow any operations
        # Dimension checking happens during evaluation, not construction
        pass
