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
        """Test the exact user-reported case with expressions."""
        x = uw.expression("x", 100, "distance", units="km")
        x0_at_start = uw.expression("x0", 50, "distance", units="km")
        velocity_phys = uw.quantity(5, "cm/year")
        t_now = uw.expression("t", 1, "time", units="Myr")

        # This is the exact user case
        result = x - x0_at_start - velocity_phys * t_now

        # Should have length units
        result_units = uw.get_units(result)
        result_dims = result_units.dimensionality
        expected_length_dims = uw.scaling.units.meter.dimensionality

        assert result_dims == expected_length_dims, \
            f"Result should have length dimensions, got {result_dims} from units {result_units}"

        result_units_str = str(result_units)
        assert 'megayear' not in result_units_str.lower(), \
            f"Result should not contain 'megayear', got {result_units_str}"
        assert 'year' not in result_units_str.lower(), \
            f"Result should not contain 'year', got {result_units_str}"

    def test_left_associativity_preservation(self):
        """Test that subtraction is left-associative and preserves first operand units."""
        # (x - x0) - velocity*t
        # Step 1: x - x0 = length (in x's units)
        # Step 2: length - length = length (in first operand's units)

        x = uw.expression("x", 100, "distance", units="km")
        x0 = uw.expression("x0", 50, "distance", units="m")  # Different units!
        velocity = uw.quantity(5, "cm/year")
        t = uw.expression("t", 1, "time", units="Myr")

        # First subtraction
        step1 = x - x0
        result_1_units = uw.get_units(step1)
        assert 'kilometer' in str(result_1_units) or 'km' in str(result_1_units), \
            f"First step should preserve x's units (km), got {result_1_units}"

        # Second subtraction
        step2 = step1 - velocity * t
        result_2_units = uw.get_units(step2)
        assert 'kilometer' in str(result_2_units) or 'km' in str(result_2_units), \
            f"Second step should preserve step1's units (km), got {result_2_units}"
