"""
Test unit conversion methods on composite expressions.

These tests ensure that .to_base_units() and .to_reduced_units()
work correctly on composite expressions containing UWexpression symbols,
preventing double-application of conversion factors during evaluation.
"""

import pytest
import underworld3 as uw
import numpy as np


@pytest.mark.level_2  # Intermediate - units system with composite expressions
@pytest.mark.tier_b   # Validated - catching recently discovered bugs
class TestUnitConversionCompositeExpressions:
    """
    Test unit conversion methods on composite expressions.

    CRITICAL: These tests verify that unit conversion methods like
    .to_base_units() and .to_reduced_units() preserve evaluation results
    for composite expressions containing UWexpression symbols.

    The bug: Previously, these methods embedded conversion factors in the
    expression tree, causing double-application during nondimensional
    evaluation cycles.

    The fix: For composite expressions, only display units are changed,
    not the expression tree itself.
    """

    def setup_method(self):
        """Set up model with nondimensional scaling."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K"),
            nondimensional_scaling=True
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

    def teardown_method(self):
        """Disable scaling after each test."""
        uw.use_nondimensional_scaling(False)

    def test_to_base_units_composite_expression(self):
        """
        Test .to_base_units() on composite expression with UWexpression symbols.

        THIS IS THE BUG WE DISCOVERED:
        - sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5
        - Units: megayear^0.5 * meter / second^0.5
        - evaluate(sqrt_2_kt) = 25122.7 m ✅
        - sqrt_2kt_m = sqrt_2_kt.to_base_units()  # Convert to meters
        - evaluate(sqrt_2kt_m) was 1.41e11 m ❌ (wrong!)
        - Should be: 25122.7 m ✅ (same as original)

        The fix: .to_base_units() now only changes display units for
        composite expressions, preventing double-application of conversion factors.
        """
        # Create composite expression
        kappa_phys = uw.quantity(1e-6, "m**2/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 'Myr'), "Current time")
        sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5

        # Check original units (should be complex)
        original_units = uw.get_units(sqrt_2_kt)
        assert "megayear" in str(original_units)
        assert "meter" in str(original_units)
        assert "second" in str(original_units)

        # Evaluate original
        result_orig = uw.function.evaluate(sqrt_2_kt, self.mesh.X.coords[60:62])
        val_orig = float(result_orig.flat[0].magnitude if hasattr(result_orig.flat[0], 'magnitude') else result_orig.flat[0])

        # Convert to base units (should simplify to just "meter")
        with pytest.warns(UserWarning, match="changing display units only"):
            sqrt_2kt_m = sqrt_2_kt.to_base_units()

        # Check units simplified
        converted_units = uw.get_units(sqrt_2kt_m)
        assert str(converted_units) == "meter"

        # Evaluate converted - MUST match original value
        result_conv = uw.function.evaluate(sqrt_2kt_m, self.mesh.X.coords[60:62])
        val_conv = float(result_conv.flat[0].magnitude if hasattr(result_conv.flat[0], 'magnitude') else result_conv.flat[0])

        # Critical assertion: Values must match
        assert np.allclose(val_orig, val_conv, rtol=1e-6), \
            f".to_base_units() changed evaluation result! Original: {val_orig:.2f} m, Converted: {val_conv:.2e} m"

    def test_to_reduced_units_composite_expression(self):
        """
        Test .to_reduced_units() on composite expression with UWexpression symbols.

        Similar to .to_base_units(), but uses Pint's to_reduced_units() for
        unit simplification by canceling common factors.
        """
        # Create composite expression
        kappa_phys = uw.quantity(1e-6, "m**2/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 'Myr'), "Current time")
        sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5

        # Evaluate original
        result_orig = uw.function.evaluate(sqrt_2_kt, self.mesh.X.coords[60:62])
        val_orig = float(result_orig.flat[0].magnitude if hasattr(result_orig.flat[0], 'magnitude') else result_orig.flat[0])

        # Reduce units (should simplify)
        with pytest.warns(UserWarning, match="changing display units only"):
            sqrt_2kt_reduced = sqrt_2_kt.to_reduced_units()

        # Check units simplified
        reduced_units = uw.get_units(sqrt_2kt_reduced)
        assert "meter" in str(reduced_units)
        # Should not have complex fractional powers anymore

        # Evaluate reduced - MUST match original value
        result_reduced = uw.function.evaluate(sqrt_2kt_reduced, self.mesh.X.coords[60:62])
        val_reduced = float(result_reduced.flat[0].magnitude if hasattr(result_reduced.flat[0], 'magnitude') else result_reduced.flat[0])

        # Critical assertion: Values must match
        assert np.allclose(val_orig, val_reduced, rtol=1e-6), \
            f".to_reduced_units() changed evaluation result! Original: {val_orig:.2f} m, Reduced: {val_reduced:.2e} m"

    def test_to_compact_still_works(self):
        """
        Test that .to_compact() still works correctly.

        .to_compact() was already working - this test ensures it stays working.
        """
        # Create composite expression
        kappa_phys = uw.quantity(1e-6, "m**2/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 'Myr'), "Current time")
        sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5

        # Evaluate original
        result_orig = uw.function.evaluate(sqrt_2_kt, self.mesh.X.coords[60:62])
        val_orig = float(result_orig.flat[0].magnitude if hasattr(result_orig.flat[0], 'magnitude') else result_orig.flat[0])

        # Compact units (automatic readable selection)
        sqrt_2kt_compact = sqrt_2_kt.to_compact()

        # Evaluate compact - MUST match original value
        result_compact = uw.function.evaluate(sqrt_2kt_compact, self.mesh.X.coords[60:62])
        val_compact = float(result_compact.flat[0].magnitude if hasattr(result_compact.flat[0], 'magnitude') else result_compact.flat[0])

        # Critical assertion: Values must match
        assert np.allclose(val_orig, val_compact, rtol=1e-6), \
            f".to_compact() changed evaluation result! Original: {val_orig:.2f} m, Compact: {val_compact:.2e} m"

    def test_simple_expression_still_converts(self):
        """
        Test that simple expressions (no UWexpression symbols) still get converted.

        For simple expressions without symbols, unit conversion should actually
        apply the conversion factor, not just change display units.
        """
        # Simple expression - just a number with units
        velocity = uw.quantity(5, "km/hour")

        # Convert to base units (should apply factor)
        velocity_ms = velocity.to_base_units()

        # Value should change (km/hour → m/s involves conversion factor)
        assert hasattr(velocity_ms, 'value')
        assert velocity_ms.value != velocity.value

        # But physical quantity should be same
        # 5 km/hour = 1.38889 m/s
        expected_ms = 5 * 1000 / 3600
        assert np.allclose(velocity_ms.value, expected_ms, rtol=1e-4)
