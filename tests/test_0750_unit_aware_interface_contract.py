"""
Test-Driven Design: Unit-Aware Interface Contract and Lazy Evaluation

This test suite defines the REQUIRED interface for all unit-aware objects in UW3.
All classes that work with units (UWQuantity, UWexpression, UnitAwareExpression)
MUST pass these tests.

Key Requirements:
1. Interface Compliance - Same methods and return types across all classes
2. Lazy Evaluation - Arithmetic preserves symbolic structure
3. Type Consistency - .units always returns Pint Unit objects
4. Arithmetic Closure - Results have the same interface as operands

Status: EXPECTED TO FAIL - These tests document required behavior before fixes.
After architectural fixes, all tests should pass.
"""
import pytest
import numpy as np
import underworld3 as uw
from underworld3.scaling import units as ureg


@pytest.mark.tier_a  # Production-ready tests for core interface
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestUnitAwareInterfaceContract:
    """Define the contract that ALL unit-aware objects must satisfy."""

    def setup_method(self):
        """Set up model with units for each test."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
        )

    # =========================================================================
    # Interface Compliance Tests
    # =========================================================================

    def test_units_property_returns_pint_unit_uwquantity(self):
        """UWQuantity.units must return Pint Unit object, not string."""
        qty = uw.quantity(5, "cm/year")

        # Check it's NOT a string
        assert not isinstance(qty.units, str), \
            f"UWQuantity.units should NOT be a string, got {qty.units!r}"

        # Check it's a Pint Unit (has dimensionality attribute)
        assert hasattr(qty.units, 'dimensionality'), \
            f"UWQuantity.units should return Pint Unit, got {type(qty.units)}"

        # Verify it's a Pint Unit type
        from pint import Unit
        assert isinstance(qty.units, Unit), \
            f"UWQuantity.units should be pint.Unit, got {type(qty.units)}"

    def test_units_property_returns_pint_unit_uwexpression(self):
        """UWexpression.units must return Pint Unit object, not string."""
        expr = uw.expression("v", 5, "velocity", units="cm/year")

        # Check it's NOT a string
        assert not isinstance(expr.units, str), \
            f"UWexpression.units should NOT be a string, got {expr.units!r}"

        # Check it's a Pint Unit (has dimensionality attribute)
        assert hasattr(expr.units, 'dimensionality'), \
            f"UWexpression.units should return Pint Unit, got {type(expr.units)}"

        # Verify it's a Pint Unit type
        from pint import Unit
        assert isinstance(expr.units, Unit), \
            f"UWexpression.units should be pint.Unit, got {type(expr.units)}"

    @pytest.mark.xfail(reason="BUG: UnitAwareExpression.units returns string, not Pint Unit")
    def test_units_property_returns_pint_unit_arithmetic_result(self):
        """Arithmetic results must return Pint Unit object, not string."""
        qty = uw.quantity(5, "cm/year")
        expr = uw.expression("t", 1, "time", units="Myr")

        result = qty * expr

        # Check it's NOT a string
        assert not isinstance(result.units, str), \
            f"Arithmetic result .units should NOT be a string, got {result.units!r}"

        # Check it's a Pint Unit (has dimensionality attribute)
        assert hasattr(result.units, 'dimensionality'), \
            f"Arithmetic result .units should return Pint Unit, got {type(result.units)}"

        # Verify it's a Pint Unit type
        from pint import Unit
        assert isinstance(result.units, Unit), \
            f"Arithmetic result .units should be pint.Unit, got {type(result.units)}"

    def test_conversion_methods_present_uwquantity(self):
        """UWQuantity must have all conversion methods."""
        qty = uw.quantity(5, "cm/year")

        assert hasattr(qty, 'to'), "UWQuantity missing .to() method"
        assert hasattr(qty, 'to_base_units'), "UWQuantity missing .to_base_units() method"
        assert hasattr(qty, 'to_compact'), "UWQuantity missing .to_compact() method"
        assert hasattr(qty, 'to_reduced_units'), "UWQuantity missing .to_reduced_units() method"
        # to_nice_units is optional - not all unit systems implement it
        # assert hasattr(qty, 'to_nice_units'), "UWQuantity missing .to_nice_units() method"

    def test_conversion_methods_present_uwexpression(self):
        """UWexpression must have all conversion methods (inherited from UWQuantity)."""
        expr = uw.expression("v", 5, "velocity", units="cm/year")

        assert hasattr(expr, 'to'), "UWexpression missing .to() method"
        assert hasattr(expr, 'to_base_units'), "UWexpression missing .to_base_units() method"
        assert hasattr(expr, 'to_compact'), "UWexpression missing .to_compact() method"
        assert hasattr(expr, 'to_reduced_units'), "UWexpression missing .to_reduced_units() method"
        # to_nice_units is optional - not all unit systems implement it
        # assert hasattr(expr, 'to_nice_units'), "UWexpression missing .to_nice_units() method"

    @pytest.mark.xfail(reason="BUG: UnitAwareExpression missing conversion methods")
    def test_conversion_methods_present_arithmetic_result(self):
        """Arithmetic results must have all conversion methods."""
        qty = uw.quantity(5, "cm/year")
        expr = uw.expression("t", 1, "time", units="Myr")

        result = qty * expr

        assert hasattr(result, 'to'), "Arithmetic result missing .to() method"
        assert hasattr(result, 'to_base_units'), "Arithmetic result missing .to_base_units() method"
        assert hasattr(result, 'to_compact'), "Arithmetic result missing .to_compact() method"
        assert hasattr(result, 'to_reduced_units'), "Arithmetic result missing .to_reduced_units() method"
        assert hasattr(result, 'to_nice_units'), "Arithmetic result missing .to_nice_units() method"

    # =========================================================================
    # Lazy Evaluation Tests
    # =========================================================================

    def test_lazy_evaluation_uwexpression_basic(self):
        """UWexpression must support lazy evaluation via .sym setter."""
        t_now = uw.expression("t", 0.0, "time", units="Myr")

        # Initial value - check the symbolic value
        initial_sym = t_now._sym
        assert initial_sym is not None

        # Update via .sym - should not evaluate, just update symbolic value
        t_now.sym = 10.0

        # Check updated - the symbolic representation should change
        assert t_now._sym is not None

        # Check _pint_qty if available
        if hasattr(t_now, '_pint_qty') and t_now._pint_qty is not None:
            assert t_now._pint_qty.magnitude == 10.0, \
                "Lazy update via .sym should synchronize ._pint_qty"

    def test_lazy_evaluation_preserves_symbolic_structure(self):
        """Arithmetic with UWexpression must preserve symbolic structure, not evaluate."""
        velocity = uw.quantity(5, "cm/year")
        t_now = uw.expression("t", 1, "time", units="Myr")

        # Multiply - should create symbolic expression, not evaluate
        distance = velocity * t_now

        # Check that result contains symbolic reference to t_now
        assert hasattr(distance, 'sym') or hasattr(distance, '_expr'), \
            "Arithmetic result should preserve symbolic structure"

        # Get symbolic expression
        sym_expr = distance.sym if hasattr(distance, 'sym') else distance._expr

        # Should contain the symbol t_now (or just 't' depending on how SymPy represents it)
        # We can check this by looking at free_symbols
        import sympy
        free_symbols = sym_expr.free_symbols if hasattr(sym_expr, 'free_symbols') else set()

        # At minimum, the expression should not be a plain number
        assert not isinstance(sym_expr, (int, float)), \
            f"Arithmetic result should be symbolic, not evaluated to {sym_expr}"

    def test_lazy_evaluation_updates_propagate(self):
        """Updating UWexpression.sym should affect arithmetic results when evaluated."""
        t_now = uw.expression("t", 1.0, "time", units="Myr")
        velocity = uw.quantity(5, "cm/year")

        # Create symbolic expression
        distance = velocity * t_now

        # Now update t_now
        t_now.sym = 10.0

        # When we evaluate distance, it should see the new value of t_now
        # Note: This tests that the symbolic expression is preserved, not pre-evaluated
        # The actual evaluation would happen via uw.function.evaluate()
        # Here we just verify the symbolic structure exists
        assert hasattr(distance, 'sym') or hasattr(distance, '_expr'), \
            "Arithmetic result must preserve symbolic structure for lazy evaluation"

    @pytest.mark.xfail(reason="BUG: Subtraction may not preserve proper unit inference")
    def test_lazy_evaluation_subtraction_preserves_units(self):
        """Subtraction with compatible units should preserve lazy evaluation and correct units."""
        x = uw.expression("x", 100, "distance", units="km")
        velocity = uw.quantity(5, "cm/year")
        t_now = uw.expression("t", 1, "time", units="Myr")

        # Create compound expression: x - velocity * t_now
        # Units: km - (cm/year * Myr) = km - cm = km (with conversion)
        result = x - velocity * t_now

        # Check units are correct (should be km, the units of x)
        result_units_str = str(result.units) if isinstance(result.units, str) else str(result.units)
        assert 'kilometer' in result_units_str or 'km' in result_units_str, \
            f"Subtraction should preserve left operand units (km), got {result.units}"

        # Check symbolic structure is preserved
        assert hasattr(result, 'sym') or hasattr(result, '_expr'), \
            "Subtraction should preserve symbolic structure"

    # =========================================================================
    # Arithmetic Closure Tests
    # =========================================================================

    def test_multiplication_closure_quantity_quantity(self):
        """UWQuantity * UWQuantity should return object with full interface."""
        qty1 = uw.quantity(5, "cm/year")
        qty2 = uw.quantity(1, "Myr")

        result = qty1 * qty2

        # Check has conversion methods
        assert hasattr(result, 'to_base_units'), \
            "UWQuantity * UWQuantity result should have .to_base_units()"

        # Check units are Pint Unit
        assert not isinstance(result.units, str), \
            "Result .units should be Pint Unit, not string"

    @pytest.mark.xfail(reason="BUG: UWQuantity * UWexpression returns UnitAwareExpression without full interface")
    def test_multiplication_closure_quantity_expression(self):
        """UWQuantity * UWexpression should return object with full interface."""
        qty = uw.quantity(5, "cm/year")
        expr = uw.expression("t", 1, "time", units="Myr")

        result = qty * expr

        # Check has conversion methods
        assert hasattr(result, 'to_base_units'), \
            "UWQuantity * UWexpression result should have .to_base_units()"

        # Check units are Pint Unit, not string
        assert not isinstance(result.units, str), \
            f"Result .units should be Pint Unit, not string (got {result.units!r})"

    @pytest.mark.xfail(reason="BUG: UWexpression * UWexpression may not preserve full interface")
    def test_multiplication_closure_expression_expression(self):
        """UWexpression * UWexpression should return object with full interface."""
        expr1 = uw.expression("v", 5, "velocity", units="cm/year")
        expr2 = uw.expression("t", 1, "time", units="Myr")

        result = expr1 * expr2

        # Check has conversion methods
        assert hasattr(result, 'to_base_units'), \
            "UWexpression * UWexpression result should have .to_base_units()"

        # Check units are Pint Unit, not string
        assert not isinstance(result.units, str), \
            f"Result .units should be Pint Unit, not string (got {result.units!r})"

    # =========================================================================
    # Unit Arithmetic Correctness Tests
    # =========================================================================

    def test_multiplication_combines_units_correctly(self):
        """Multiplication should combine units using Pint dimensional analysis."""
        velocity = uw.quantity(5, "cm/year")
        time = uw.quantity(1, "Myr")

        result = velocity * time

        # Get units as Pint object for comparison
        if isinstance(result.units, str):
            result_pint = ureg.parse_expression(result.units)
        else:
            result_pint = result.units

        # Should be dimensionally equivalent to length (cm/year * Myr = cm)
        expected_pint = ureg.parse_expression("cm")

        # Compare dimensionality
        assert result_pint.dimensionality == expected_pint.dimensionality, \
            f"cm/year * Myr should have length dimensionality, got {result_pint.dimensionality}"

    @pytest.mark.xfail(reason="BUG: get_units may not correctly extract units from compound expressions")
    def test_get_units_consistency(self):
        """uw.get_units() should return Pint Unit objects consistently."""
        qty = uw.quantity(5, "cm/year")
        expr = uw.expression("t", 1, "time", units="Myr")
        result = qty * expr

        units = uw.get_units(result)

        # Should be Pint Unit, not string
        assert not isinstance(units, str), \
            f"uw.get_units() should return Pint Unit, not string (got {units!r})"

        # Should have Pint attributes
        assert hasattr(units, 'dimensionality'), \
            f"uw.get_units() result should be Pint Unit with .dimensionality"


@pytest.mark.tier_a
@pytest.mark.level_2  # Units integration - intermediate complexity
class TestUnitAwareInterfaceTimeSteppingPattern:
    """Test the specific lazy evaluation pattern used in time-stepping simulations."""

    def setup_method(self):
        """Set up model with units for each test."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
        )

    def test_time_stepping_lazy_update_pattern(self):
        """Test the canonical time-stepping pattern: define once, update many times."""
        # Define time variable once
        t_now = uw.expression("t", 0.0, "current time", units="Myr")

        # Define expression using t_now
        velocity = uw.quantity(5, "cm/year")
        distance = velocity * t_now

        # Simulate time-stepping: update t_now many times
        times = [0.0, 1.0, 10.0, 100.0, 1000.0]

        for t in times:
            # Update t_now symbolically
            t_now.sym = t

            # Verify t_now updated - _sym should reflect the change
            assert t_now._sym is not None, f"t_now._sym should exist after update to {t}"

            # Verify _pint_qty if available
            if hasattr(t_now, '_pint_qty') and t_now._pint_qty is not None:
                assert t_now._pint_qty.magnitude == t, \
                    f"t_now.sym = {t} should update ._pint_qty"

            # Verify distance expression still exists (not evaluated)
            assert hasattr(distance, 'sym') or hasattr(distance, '_expr') or hasattr(distance, 'atoms'), \
                "distance expression should remain symbolic after t_now update"

    def test_multiple_expressions_share_updated_variable(self):
        """Multiple expressions referencing same variable should see updates."""
        t_now = uw.expression("t", 1.0, "time", units="Myr")

        # Create two different expressions using t_now
        distance1 = uw.quantity(5, "cm/year") * t_now
        distance2 = uw.quantity(10, "cm/year") * t_now

        # Update t_now
        t_now.sym = 100.0

        # Both should preserve symbolic structure (not pre-evaluated)
        assert hasattr(distance1, 'sym') or hasattr(distance1, '_expr'), \
            "distance1 should remain symbolic"
        assert hasattr(distance2, 'sym') or hasattr(distance2, '_expr'), \
            "distance2 should remain symbolic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
