"""
Test for chained subtraction units bug.

CRITICAL BUG DETECTED:
    x - x0 - velocity * t

Should return length units, but returns time units instead.

This tests the specific user-reported regression.
"""

import pytest
import underworld3 as uw
import pint

@pytest.mark.tier_a
@pytest.mark.level_1
class TestSubtractionChainUnits:
    """Test that chained subtraction preserves correct units."""

    def test_simple_subtraction_chain(self):
        """Test: length - length - length = length"""
        x = uw.quantity(100, "km")
        x0 = uw.quantity(50, "km")
        dx = uw.quantity(10, "km")

        result = x - x0 - dx

        # Should have length units
        result_units_str = str(result.units)
        assert 'kilometer' in result_units_str or 'km' in result_units_str
        assert 'year' not in result_units_str
        assert 'megayear' not in result_units_str

    def test_subtraction_with_velocity_time_product(self):
        """Test: position - position0 - velocity*time = position (all lengths)"""
        x = uw.quantity(100, "km")
        x0 = uw.quantity(50, "km")
        velocity = uw.quantity(5, "cm/year")
        t = uw.quantity(1, "Myr")

        displacement = velocity * t

        # First check displacement has correct units (should be length)
        displacement_dims = displacement.units.dimensionality
        expected_length_dims = uw.scaling.units.meter.dimensionality
        assert displacement_dims == expected_length_dims, \
            f"Displacement should have length dimensions, got {displacement_dims}"

        # Now test the full chain
        result = x - x0 - displacement

        # Should have length units, NOT time units
        result_units_str = str(result.units)
        result_dims = result.units.dimensionality

        assert result_dims == expected_length_dims, \
            f"Result should have length dimensions, got {result_dims}"
        assert 'year' not in result_units_str.lower(), \
            f"Result should not contain 'year', got {result_units_str}"
        assert 'megayear' not in result_units_str.lower(), \
            f"Result should not contain 'megayear', got {result_units_str}"

    def test_expression_subtraction_chain(self):
        """Test the exact user-reported case with expressions.

        Note: When velocity*time is computed, Pint may return compound units
        like 'centimeter * megayear / year' which are dimensionally correct
        (the time components cancel to give length). What matters is:
        1. The result has LENGTH dimensionality
        2. The .to() method can convert to any length unit

        Checking the string representation is unreliable because Pint doesn't
        automatically simplify compound units.
        """
        x = uw.expression("x", 100, "distance", units="km")
        x0_at_start = uw.expression("x0", 50, "distance", units="km")
        velocity_phys = uw.quantity(5, "cm/year")
        t_now = uw.expression("t", 1, "time", units="Myr")

        # This is the exact user case
        result = x - x0_at_start - velocity_phys * t_now

        # Should have length dimensionality
        result_units = uw.get_units(result)
        result_dims = result_units.dimensionality
        expected_length_dims = uw.scaling.units.meter.dimensionality

        assert result_dims == expected_length_dims, \
            f"Result should have length dimensions, got {result_dims} from units {result_units}"

    def test_left_associativity_preservation(self):
        """Test that subtraction preserves length dimensionality through chained operations.

        Note: When mixing units (km and m), SymPy's internal ordering may result in
        either unit being returned. What matters is that:
        1. The result has LENGTH dimensionality (not time, etc.)
        2. The .to() method can convert between compatible units

        This is by design - specific unit choice is a minor detail when dimensionality
        is correct, and .to() methods handle conversion.
        """
        # (x - x0) - velocity*t
        # Step 1: x - x0 = length (dimensionality preserved)
        # Step 2: length - length = length (dimensionality preserved)

        x = uw.expression("x", 100, "distance", units="km")
        x0 = uw.expression("x0", 50, "distance", units="m")  # Different units!
        velocity = uw.quantity(5, "cm/year")
        t = uw.expression("t", 1, "time", units="Myr")

        expected_length_dims = uw.scaling.units.meter.dimensionality

        # First subtraction - should have length dimensionality
        step1 = x - x0
        result_1_units = uw.get_units(step1)
        result_1_dims = result_1_units.dimensionality
        assert result_1_dims == expected_length_dims, \
            f"First step should have length dimensionality, got {result_1_dims} from units {result_1_units}"

        # Second subtraction - should still have length dimensionality
        step2 = step1 - velocity * t
        result_2_units = uw.get_units(step2)
        result_2_dims = result_2_units.dimensionality
        assert result_2_dims == expected_length_dims, \
            f"Second step should have length dimensionality, got {result_2_dims} from units {result_2_units}"

        # Note: We don't check string representation because velocity*time produces
        # compound units like 'cm * Myr / year' which are dimensionally correct (length)
        # but contain time-related strings. Dimensionality check above is sufficient.
