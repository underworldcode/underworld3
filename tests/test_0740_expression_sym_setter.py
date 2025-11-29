"""
Test that UWexpression.sym setter properly synchronizes internal state.

This validates the fix for the issue where setting .sym updated ._sym
but not ._pint_qty, causing evaluate() to see stale values.
"""
import pytest
import numpy as np
import underworld3 as uw


@pytest.mark.tier_a  # Production-ready - critical for lazy evaluation patterns
@pytest.mark.level_2  # Intermediate - requires units and evaluation
class TestExpressionSymSetter:
    """Test that .sym setter maintains consistency between ._sym and ._pint_qty."""

    def setup_method(self):
        """Set up model with units for each test."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K")
        )

    def test_sym_setter_updates_both_representations(self):
        """Test that setting .sym updates both ._sym and ._pint_qty."""
        t_now = uw.expression("t", 1.0, "time", units="Myr")

        # Initial state
        assert float(t_now._sym) == 1.0
        assert t_now._pint_qty.magnitude == 1.0

        # Update via .sym
        t_now.sym = 10.0

        # Both should be updated
        assert float(t_now._sym) == 10.0, "._sym should be updated"
        assert t_now._pint_qty.magnitude == 10.0, "._pint_qty should be updated"

    def test_evaluate_sees_updated_value(self):
        """Test that evaluate() sees the value set via .sym setter."""
        t_now = uw.expression("t", 1.0, "time", units="Myr")

        # Initial evaluation
        result1 = uw.function.evaluate(t_now, np.array([[0, 0]]))
        assert np.allclose(result1.to('Myr').value, 1.0)

        # Update via .sym
        t_now.sym = 10000

        # Evaluation should see new value
        result2 = uw.function.evaluate(t_now, np.array([[0, 0]]))
        assert np.allclose(result2.to('Myr').value, 10000), \
            "evaluate() should see updated value"

    def test_plain_number_preserves_units(self):
        """Test that setting .sym with a plain number preserves existing units."""
        t_now = uw.expression("t", 1.0, "time", units="Myr")

        # Update with plain number
        t_now.sym = 5000

        # Units should be preserved
        assert str(t_now._pint_qty.units) == "megayear", \
            "Plain number should preserve existing units"
        assert t_now._pint_qty.magnitude == 5000

    def test_uwquantity_updates_magnitude_and_units(self):
        """Test that setting .sym with UWQuantity can change units."""
        t_now = uw.expression("t", 1.0, "time", units="Myr")

        # Update with UWQuantity in different (compatible) units
        t_now.sym = uw.quantity(1e9, "year")

        # Should convert to existing units (Myr)
        assert np.allclose(t_now._pint_qty.to("Myr").magnitude, 1000), \
            "Should convert to existing units when compatible"

    def test_lazy_evaluation_pattern(self):
        """Test lazy evaluation pattern for time-stepping loops."""
        # Define once
        t_step = uw.expression("t_step", 0.0, "Step time", units="Myr")

        # Update many times in a loop
        times = [0, 1, 5, 10, 20, 50]
        for t in times:
            t_step.sym = t
            result = uw.function.evaluate(t_step, np.array([[0, 0]]))
            result_myr = result.to('Myr').value[0,0,0]

            assert np.allclose(result_myr, t), \
                f"At t={t}, evaluate() should return {t}, got {result_myr}"

    def test_internal_consistency(self):
        """Test that ._sym and ._pint_qty remain consistent."""
        t_expr = uw.expression("t", 100, "time", units="Myr")

        # Update several times
        for value in [1, 10, 100, 1000, 5000]:
            t_expr.sym = value

            # Check consistency
            assert float(t_expr._sym) == t_expr._pint_qty.magnitude, \
                f"After setting .sym={value}, ._sym and ._pint_qty should match"

    def test_dimensionless_expression(self):
        """Test that .sym setter works for dimensionless expressions."""
        coeff = uw.expression("coeff", 1.0, "coefficient")

        # Should not have _pint_qty
        assert not (hasattr(coeff, '_pint_qty') and coeff._pint_qty is not None), \
            "Dimensionless expression shouldn't have _pint_qty"

        # Setting .sym should still work
        coeff.sym = 5.0
        assert float(coeff._sym) == 5.0

    def test_symbolic_expression_setting(self):
        """Test that setting .sym with symbolic expression works."""
        import sympy

        param = uw.expression("param", 1.0, "parameter", units="km")
        x = sympy.Symbol('x')

        # Set to symbolic expression
        param.sym = x**2

        # Should update ._sym but not ._pint_qty (can't convert symbol to float)
        assert param._sym == x**2
        # _pint_qty should remain as it was (or be unchanged)


    def test_uwquantity_copy_synchronizes_both_representations(self):
        """Test that UWQuantity.copy() updates both ._sym and ._pint_qty."""
        q1 = uw.quantity(100, "km")
        q2 = uw.quantity(1, "km")

        # Copy from q1 to q2
        q2.copy(q1)

        # Both representations should be updated
        assert float(q2._sym) == 100, "._sym should be updated"
        assert q2._pint_qty.magnitude == 100, "._pint_qty should be updated"

    def test_uwquantity_copy_with_unit_conversion(self):
        """Test that UWQuantity.copy() handles unit conversion."""
        q_km = uw.quantity(1, "km")
        q_m = uw.quantity(1000, "m")

        # Copy 1000 m to km quantity - should convert
        q_km.copy(q_m)

        # Should convert to km (1000 m = 1 km)
        assert np.allclose(q_km._pint_qty.magnitude, 1), \
            "Should convert 1000 m to 1 km"
        assert str(q_km._pint_qty.units) == "kilometer", \
            "Units should remain as kilometer"

    def test_uwexpression_copy_accepts_uwquantity(self):
        """Test that UWexpression.copy() accepts UWQuantity objects."""
        expr = uw.expression("test", 1, "test expr", units="km")
        new_val = uw.quantity(200, "km")

        # Should accept UWQuantity
        expr.copy(new_val)

        # Evaluate should see new value
        result = uw.function.evaluate(expr, np.array([[0, 0]]))
        assert np.allclose(result.to('km').value, 200), \
            "evaluate() should see copied value"

    def test_uwexpression_copy_error_message(self):
        """Test that UWexpression.copy() has helpful error message."""
        expr = uw.expression("test", 1, "test", units="km")

        # Should raise TypeError with helpful message
        with pytest.raises(TypeError, match="UWQuantity or UWexpression"):
            expr.copy(42)

        with pytest.raises(TypeError, match="Consider using: expr.sym"):
            expr.copy("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
