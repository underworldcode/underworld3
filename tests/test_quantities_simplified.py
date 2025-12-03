"""
Test the simplified UWQuantity implementation.

Tests:
1. Basic creation and properties
2. Arithmetic via Pint (the multiplication order bug)
3. Unit conversions
4. .data property (non-dimensional values)
5. SymPy compatibility
"""

import pytest
import numpy as np


class TestSimplifiedQuantityBasics:
    """Test basic UWQuantity creation and properties."""

    def test_create_with_units(self):
        """Create quantity with units."""
        from underworld3.function.quantities_simplified import quantity

        viscosity = quantity(1e21, "Pa*s")

        assert viscosity.value == 1e21
        assert viscosity.has_units
        assert "pascal" in str(viscosity.units) or "Pa" in str(viscosity.units)

    def test_create_dimensionless(self):
        """Create dimensionless quantity."""
        from underworld3.function.quantities_simplified import quantity

        factor = quantity(0.5)

        assert factor.value == 0.5
        assert not factor.has_units
        assert factor.units is None

    def test_magnitude_alias(self):
        """Test .magnitude is alias for .value."""
        from underworld3.function.quantities_simplified import quantity

        T = quantity(1000, "kelvin")

        assert T.magnitude == T.value == 1000

    def test_dimensionality(self):
        """Test dimensionality extraction."""
        from underworld3.function.quantities_simplified import quantity

        velocity = quantity(5, "m/s")

        dim = velocity.dimensionality
        assert "[length]" in dim
        assert "[time]" in dim
        assert dim["[length]"] == 1
        assert dim["[time]"] == -1


class TestSimplifiedQuantityArithmetic:
    """Test arithmetic operations - the core of the fix."""

    def test_addition_same_units(self):
        """Add quantities with same units."""
        from underworld3.function.quantities_simplified import quantity

        T1 = quantity(1000, "kelvin")
        T2 = quantity(273, "kelvin")

        result = T1 + T2
        assert result.value == 1273
        assert "kelvin" in str(result.units).lower()

    def test_subtraction_same_units(self):
        """Subtract quantities with same units."""
        from underworld3.function.quantities_simplified import quantity

        T1 = quantity(1000, "kelvin")
        T2 = quantity(273, "kelvin")

        result = T1 - T2
        assert result.value == 727
        assert "kelvin" in str(result.units).lower()

    def test_multiplication_with_units(self):
        """Multiply quantities with units."""
        from underworld3.function.quantities_simplified import quantity

        length = quantity(100, "m")
        width = quantity(50, "m")

        area = length * width
        assert area.value == 5000
        # Should have m**2
        assert "meter ** 2" in str(area.units) or "m²" in str(area.units) or "meter**2" in str(area.units)

    def test_division_with_units(self):
        """Divide quantities with units."""
        from underworld3.function.quantities_simplified import quantity

        distance = quantity(100, "km")
        time = quantity(2, "hour")

        speed = distance / time
        assert speed.value == 50
        # Should have km/hour or equivalent

    def test_multiplication_order_bug_case1(self):
        """
        Case 1: factor * (T_right - T_left) + T_left

        This is the original multiplication order bug test.
        """
        from underworld3.function.quantities_simplified import quantity

        T_left = quantity(1000, "kelvin")
        T_right = quantity(1273, "kelvin")
        factor = quantity(0.1, "dimensionless")

        result = factor * (T_right - T_left) + T_left

        # Expected: 0.1 * 273 + 1000 = 27.3 + 1000 = 1027.3
        assert abs(result.value - 1027.3) < 0.01

    def test_multiplication_order_bug_case2(self):
        """
        Case 2: (T_right - T_left) * factor + T_left

        Multiplication in different order.
        """
        from underworld3.function.quantities_simplified import quantity

        T_left = quantity(1000, "kelvin")
        T_right = quantity(1273, "kelvin")
        factor = quantity(0.1, "dimensionless")

        result = (T_right - T_left) * factor + T_left

        # Expected: 273 * 0.1 + 1000 = 27.3 + 1000 = 1027.3
        assert abs(result.value - 1027.3) < 0.01

    def test_multiplication_order_bug_case3(self):
        """
        Case 3: T_left + factor * (T_right - T_left)

        Addition order swapped.
        """
        from underworld3.function.quantities_simplified import quantity

        T_left = quantity(1000, "kelvin")
        T_right = quantity(1273, "kelvin")
        factor = quantity(0.1, "dimensionless")

        result = T_left + factor * (T_right - T_left)

        # Expected: 1000 + 0.1 * 273 = 1000 + 27.3 = 1027.3
        assert abs(result.value - 1027.3) < 0.01

    def test_multiplication_order_bug_case4(self):
        """
        Case 4: T_left + (T_right - T_left) * factor

        All orderings should give same result.
        """
        from underworld3.function.quantities_simplified import quantity

        T_left = quantity(1000, "kelvin")
        T_right = quantity(1273, "kelvin")
        factor = quantity(0.1, "dimensionless")

        result = T_left + (T_right - T_left) * factor

        # Expected: 1000 + 273 * 0.1 = 1000 + 27.3 = 1027.3
        assert abs(result.value - 1027.3) < 0.01

    def test_all_orderings_equal(self):
        """All four orderings should give identical results."""
        from underworld3.function.quantities_simplified import quantity

        T_left = quantity(1000, "kelvin")
        T_right = quantity(1273, "kelvin")
        factor = quantity(0.1, "dimensionless")

        case1 = factor * (T_right - T_left) + T_left
        case2 = (T_right - T_left) * factor + T_left
        case3 = T_left + factor * (T_right - T_left)
        case4 = T_left + (T_right - T_left) * factor

        # All should be equal
        assert abs(case1.value - case2.value) < 1e-10
        assert abs(case2.value - case3.value) < 1e-10
        assert abs(case3.value - case4.value) < 1e-10

        # And all should be 1027.3
        assert abs(case1.value - 1027.3) < 0.01

    def test_scalar_multiplication(self):
        """Multiply quantity by scalar."""
        from underworld3.function.quantities_simplified import quantity

        T = quantity(1000, "kelvin")

        result = T * 2
        assert result.value == 2000
        assert "kelvin" in str(result.units).lower()

        result2 = 2 * T
        assert result2.value == 2000

    def test_power(self):
        """Test exponentiation."""
        from underworld3.function.quantities_simplified import quantity

        length = quantity(10, "m")

        area = length ** 2
        assert area.value == 100

        volume = length ** 3
        assert volume.value == 1000


class TestSimplifiedQuantityConversion:
    """Test unit conversion methods."""

    def test_to_conversion(self):
        """Convert between compatible units."""
        from underworld3.function.quantities_simplified import quantity

        velocity = quantity(36, "km/hour")
        velocity_ms = velocity.to("m/s")

        assert abs(velocity_ms.value - 10) < 0.01

    def test_to_base_units(self):
        """Convert to SI base units."""
        from underworld3.function.quantities_simplified import quantity

        pressure = quantity(1, "GPa")
        base = pressure.to_base_units()

        assert base.value == 1e9

    def test_to_reduced_units(self):
        """Simplify complex unit expressions."""
        from underworld3.function.quantities_simplified import quantity

        # Create a dimensionless ratio
        ratio = quantity(1000, "m") / quantity(1, "km")
        reduced = ratio.to_reduced_units()

        assert abs(reduced.value - 1.0) < 0.01


class TestSimplifiedQuantitySympyCompat:
    """Test SymPy compatibility."""

    def test_sympy_protocol(self):
        """Test _sympy_() protocol."""
        from underworld3.function.quantities_simplified import quantity
        import sympy

        T = quantity(1000, "kelvin")
        sympy_val = T._sympy_()

        assert isinstance(sympy_val, sympy.Basic)
        assert float(sympy_val) == 1000

    def test_float_conversion(self):
        """Test float() conversion."""
        from underworld3.function.quantities_simplified import quantity

        T = quantity(1000, "kelvin")
        assert float(T) == 1000

    def test_diff_returns_zero(self):
        """Derivative of constant is zero."""
        from underworld3.function.quantities_simplified import quantity
        import sympy

        x = sympy.Symbol('x')
        T = quantity(1000, "kelvin")

        assert T.diff(x) == 0


class TestSimplifiedQuantityDisplay:
    """Test string representations."""

    def test_str_with_units(self):
        """String representation includes units."""
        from underworld3.function.quantities_simplified import quantity

        T = quantity(1000, "kelvin")
        s = str(T)

        assert "1000" in s
        assert "kelvin" in s.lower()

    def test_repr(self):
        """Repr shows constructor form."""
        from underworld3.function.quantities_simplified import quantity

        T = quantity(1000, "kelvin")
        r = repr(T)

        assert "UWQuantity" in r
        assert "1000" in r

    def test_format(self):
        """Test format specification."""
        from underworld3.function.quantities_simplified import quantity

        T = quantity(1000.123456, "kelvin")
        formatted = f"{T:.2f}"

        assert "1000.12" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
