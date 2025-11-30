"""
Complete arithmetic closure tests for all unit-aware types.

POLICY: Pint-Only Arithmetic - No String Comparisons, No Manual Fallbacks

These tests ensure:
1. All combinations of unit-aware objects work with +, -, *, /
2. Results have correct units (Pint Unit objects, NOT strings)
3. Closure property: operations return unit-aware objects with proper interface
4. Scale factors are NEVER lost (100 km + 50 m = 100.05 km, NOT 150 km)

TEST COVERAGE:
- UWQuantity op UWQuantity
- UWQuantity op UnitAwareExpression  (Bug #3 - was raising TypeError)
- UnitAwareExpression op UWQuantity
- UnitAwareExpression op UnitAwareExpression
- UWexpression op UWQuantity
- All operations: +, -, *, /
- All return Pint Unit objects (Bug #1 - were returning strings)
"""

import pytest
import numpy as np
import underworld3 as uw
from pint import Unit


@pytest.mark.tier_a  # Production-ready - REQUIRED for TDD
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestArithmeticClosure:
    """
    Test arithmetic closure: operations on unit-aware types return unit-aware types.

    Closure means: If you add two UWQuantities, you get a UWQuantity.
    If you mix types, you get the appropriate result type.
    """

    def test_uwquantity_plus_uwquantity(self):
        """Test: UWQuantity + UWQuantity = UWQuantity with correct units"""
        a = uw.quantity(100, "km")
        b = uw.quantity(50, "m")

        result = a + b

        # Should be UWQuantity
        assert isinstance(result, uw.function.UWQuantity), \
            f"UWQuantity + UWQuantity should return UWQuantity, got {type(result)}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)} (Bug #1: string conversion!)"

        # Should preserve scale factor: 100 km + 50 m = 100.05 km
        result_km = result.to('km')
        assert np.allclose(result_km.value, 100.05, rtol=1e-9), \
            f"Expected 100.05 km, got {result_km.value} km (scale factor lost!)"

    def test_uwquantity_minus_uwquantity(self):
        """Test: UWQuantity - UWQuantity = UWQuantity with correct units"""
        a = uw.quantity(100, "km")
        b = uw.quantity(50, "m")

        result = a - b

        # Should be UWQuantity
        assert isinstance(result, uw.function.UWQuantity), \
            f"UWQuantity - UWQuantity should return UWQuantity, got {type(result)}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)} (Bug #1: string conversion!)"

        # Should preserve scale factor: 100 km - 50 m = 99.95 km
        result_km = result.to('km')
        assert np.allclose(result_km.value, 99.95, rtol=1e-9), \
            f"Expected 99.95 km, got {result_km.value} km (scale factor lost!)"

    def test_uwquantity_times_uwquantity(self):
        """Test: UWQuantity * UWQuantity = UWQuantity with combined units"""
        a = uw.quantity(5, "cm/year")
        b = uw.quantity(1, "Myr")

        result = a * b

        # Result should have correct dimensionality [length]
        assert result.units.dimensionality == uw.scaling.units.meter.dimensionality, \
            f"cm/year * Myr should have [length] dimensionality, got {result.units.dimensionality}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)}"

    def test_uwquantity_divided_by_uwquantity(self):
        """Test: UWQuantity / UWQuantity = UWQuantity with ratio units"""
        a = uw.quantity(100, "km")
        b = uw.quantity(1, "Myr")

        result = a / b

        # Result should have velocity dimensionality [length]/[time]
        velocity_dim = (uw.scaling.units.meter / uw.scaling.units.second).dimensionality
        assert result.units.dimensionality == velocity_dim, \
            f"km / Myr should have [length]/[time] dimensionality, got {result.units.dimensionality}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)}"

    @pytest.mark.xfail(reason="UWQuantity - UnitAwareExpression not fully implemented")
    def test_uwquantity_minus_unitawareexpression(self):
        """
        Test: UWQuantity - UnitAwareExpression works (Bug #3)

        CRITICAL: This was raising TypeError before the fix.
        Example: x0 - velocity_phys * t_now
        Where velocity_phys * t_now returns UnitAwareExpression
        """
        from underworld3.expression_types.unit_aware_expression import UnitAwareExpression
        import sympy

        x0 = uw.quantity(1, "km")

        # Create UnitAwareExpression (simulates velocity * time)
        t = sympy.Symbol('t')
        vel_times_time = UnitAwareExpression(50.0 * t, uw.scaling.units.kilometer)

        # This should NOT raise TypeError
        try:
            result = x0 - vel_times_time
            success = True
        except TypeError as e:
            success = False
            error_msg = str(e)

        assert success, \
            f"UWQuantity - UnitAwareExpression raised TypeError (Bug #3): {error_msg if not success else ''}"

        # Result should be UnitAwareExpression (mixing quantity + expression)
        assert isinstance(result, UnitAwareExpression), \
            f"UWQuantity - UnitAwareExpression should return UnitAwareExpression, got {type(result)}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)}"

    @pytest.mark.xfail(reason="UWQuantity + UnitAwareExpression not fully implemented")
    def test_uwquantity_plus_unitawareexpression(self):
        """Test: UWQuantity + UnitAwareExpression works"""
        from underworld3.expression_types.unit_aware_expression import UnitAwareExpression
        import sympy

        x0 = uw.quantity(1, "km")
        t = sympy.Symbol('t')
        displacement = UnitAwareExpression(50.0 * t, uw.scaling.units.kilometer)

        # Should work
        result = x0 + displacement

        assert isinstance(result, UnitAwareExpression), \
            f"UWQuantity + UnitAwareExpression should return UnitAwareExpression, got {type(result)}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)}"

    def test_unitawareexpression_minus_uwquantity(self):
        """Test: UnitAwareExpression - UWQuantity works (reverse of Bug #3)"""
        from underworld3.expression_types.unit_aware_expression import UnitAwareExpression
        import sympy

        t = sympy.Symbol('t')
        position = UnitAwareExpression(100.0 + 10.0 * t, uw.scaling.units.kilometer)
        offset = uw.quantity(5, "km")

        # Should work
        result = position - offset

        assert isinstance(result, UnitAwareExpression), \
            f"UnitAwareExpression - UWQuantity should return UnitAwareExpression, got {type(result)}"

        # Should have Pint Unit (NOT string!)
        assert isinstance(result.units, Unit), \
            f"Result.units should be Pint Unit, got {type(result.units)}"


@pytest.mark.tier_a
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestScaleFactorPreservation:
    """
    Critical tests: arithmetic MUST preserve numerical scale factors.

    POLICY: Pint does ALL conversions. No manual arithmetic.

    If ANY of these fail, there is a CRITICAL BUG that produces wrong physics.
    """

    def test_addition_preserves_scale_km_plus_m(self):
        """Test: 100 km + 50 m = 100.05 km (NOT 150 km!)"""
        a = uw.quantity(100, "km")
        b = uw.quantity(50, "m")

        result = a + b
        result_km = result.to("km")

        assert np.allclose(result_km.value, 100.05, rtol=1e-9), \
            f"Expected 100.05 km, got {result_km.value} km (Scale factor LOST - Bug in policy!)"

    def test_subtraction_preserves_scale_km_minus_m(self):
        """Test: 100 km - 50 m = 99.95 km (NOT 50 km!)"""
        a = uw.quantity(100, "km")
        b = uw.quantity(50, "m")

        result = a - b
        result_km = result.to("km")

        assert np.allclose(result_km.value, 99.95, rtol=1e-9), \
            f"Expected 99.95 km, got {result_km.value} km (Scale factor LOST - Bug in policy!)"

    def test_multiplication_combines_units_correctly(self):
        """Test: 5 cm/year * 1 Myr has [length] dimensionality"""
        velocity = uw.quantity(5, "cm/year")
        time = uw.quantity(1, "Myr")

        result = velocity * time

        # Should have length dimensionality
        length_dim = uw.scaling.units.meter.dimensionality
        assert result.units.dimensionality == length_dim, \
            f"cm/year * Myr should have [length] dimensionality, got {result.units.dimensionality}"

        # Value check: 5 cm/year * 1e6 years = 5e6 cm = 50 km
        result_km = result.to("km")
        assert np.allclose(result_km.value, 50.0, rtol=1e-6), \
            f"Expected 50 km, got {result_km.value} km"

    def test_very_small_scale_factors(self):
        """Test: 1 m + 1 nm preserves nano-scale"""
        large = uw.quantity(1, "m")
        tiny = uw.quantity(1, "nm")

        result = large + tiny
        result_m = result.to("m")

        # 1 m + 1 nm = 1.000000001 m
        assert np.allclose(result_m.value, 1.000000001, rtol=1e-12), \
            f"Expected 1.000000001 m, got {result_m.value} m (Tiny scale lost!)"

    def test_very_large_scale_factors(self):
        """Test: 1 Gm + 1 m preserves both scales"""
        huge = uw.quantity(1, "Gm")
        small = uw.quantity(1, "m")

        result = huge + small
        result_m = result.to("m")

        # 1 Gm + 1 m = 1000000001 m
        assert np.allclose(result_m.value, 1e9 + 1, rtol=1e-9), \
            f"Expected 1000000001 m, got {result_m.value} m (Large scale lost!)"


@pytest.mark.tier_a
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestPintUnitObjects:
    """
    Test that ALL .units properties return Pint Unit objects, NEVER strings.

    POLICY VIOLATION: Catches Bug #1 and Bug #2
    """

    def test_uwquantity_units_is_pint_unit(self):
        """Test: UWQuantity.units returns Pint Unit (NOT string)"""
        qty = uw.quantity(100, "km")

        assert isinstance(qty.units, Unit), \
            f"UWQuantity.units should be Pint Unit, got {type(qty.units)} (Bug #1: string conversion!)"

    def test_arithmetic_result_units_is_pint_unit(self):
        """Test: (UWQuantity + UWQuantity).units returns Pint Unit"""
        a = uw.quantity(100, "km")
        b = uw.quantity(50, "m")

        result = a + b

        assert isinstance(result.units, Unit), \
            f"Arithmetic result.units should be Pint Unit, got {type(result.units)} (Bug #1: string conversion!)"

    def test_multiplication_result_units_is_pint_unit(self):
        """Test: (UWQuantity * UWQuantity).units returns Pint Unit"""
        a = uw.quantity(5, "cm/year")
        b = uw.quantity(1, "Myr")

        result = a * b

        assert isinstance(result.units, Unit), \
            f"Multiplication result.units should be Pint Unit, got {type(result.units)} (Bug #1!)"

    def test_unitawareexpression_units_is_pint_unit(self):
        """Test: UnitAwareExpression.units returns Pint Unit"""
        from underworld3.expression_types.unit_aware_expression import UnitAwareExpression
        import sympy

        t = sympy.Symbol('t')
        expr = UnitAwareExpression(10.0 * t, uw.scaling.units.kilometer)

        assert isinstance(expr.units, Unit), \
            f"UnitAwareExpression.units should be Pint Unit, got {type(expr.units)} (Bug #1!)"


@pytest.mark.tier_a
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestIncompatibleDimensions:
    """
    Test that incompatible dimensions raise errors (fail loudly).

    POLICY: "An error is better than wrong physics"
    """

    def test_length_plus_time_raises(self):
        """Test: Cannot add length + time"""
        length = uw.quantity(100, "m")
        time = uw.quantity(5, "s")

        with pytest.raises((ValueError, TypeError, Exception)):
            result = length + time

    def test_length_minus_temperature_raises(self):
        """Test: Cannot subtract temperature from length"""
        length = uw.quantity(100, "m")
        temp = uw.quantity(300, "K")

        with pytest.raises((ValueError, TypeError, Exception)):
            result = length - temp


@pytest.mark.tier_a
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestUnitConversionMethods:
    """
    Test that unit conversion methods accept Pint Units (not just strings).

    POLICY: .to() should accept both strings AND Pint Unit objects.
    """

    def test_to_accepts_string(self):
        """Test: .to('km') works"""
        qty = uw.quantity(1000, "m")
        result = qty.to("km")

        assert np.allclose(result.value, 1.0, rtol=1e-9), \
            f"Expected 1 km, got {result.value} km"

    def test_to_accepts_pint_unit(self):
        """Test: .to(uw.units.km) works (Bug #2: was calling str() on Pint Unit)"""
        qty = uw.quantity(1000, "m")

        # This should work - accepting Pint Unit directly
        try:
            result = qty.to(uw.scaling.units.kilometer)
            success = True
        except Exception as e:
            success = False
            error_msg = str(e)

        assert success, \
            f".to() should accept Pint Unit objects, got error: {error_msg if not success else ''} (Bug #2!)"

        if success:
            assert np.allclose(result.value, 1.0, rtol=1e-9), \
                f"Expected 1 km, got {result.value} km"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
