"""
Test unit conversion methods on composite expressions.

These tests ensure that uw.to_base_units() and uw.to_reduced_units()
work correctly on composite expressions containing UWexpression symbols,
preventing double-application of conversion factors during evaluation.

Design Note (2025-12):
- Unit conversion functions are in units.py as the SINGLE SOURCE OF TRUTH
- Use uw.to_base_units(expr) not expr.to_base_units() for composite expressions
- Composite expressions (SymPy Pow, Mul, etc.) don't have methods - use uw.* functions
"""

import pytest
import underworld3 as uw
import numpy as np


@pytest.mark.level_2  # Intermediate - units system with composite expressions
@pytest.mark.tier_b   # Validated - catching recently discovered bugs
class TestUnitConversionCompositeExpressions:
    """
    Test unit conversion methods on composite expressions.

    CRITICAL: These tests verify that uw.to_base_units() and uw.to_reduced_units()
    preserve evaluation results for composite expressions containing UWexpression symbols.

    The bug: Previously, these methods embedded conversion factors in the
    expression tree, causing double-application during nondimensional
    evaluation cycles.

    The fix: For composite expressions (raw SymPy Pow, Mul, etc.), the base functions
    return the expression unchanged with a warning - the expression tree preserves
    units through evaluation. Use uw.get_units() to check units.
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
        Test uw.to_base_units() on composite expression with UWexpression symbols.

        THIS IS THE BUG WE DISCOVERED:
        - sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5
        - Units: megayear^0.5 * meter / second^0.5
        - evaluate(sqrt_2_kt) = 25122.7 m ✅
        - sqrt_2kt_m = uw.to_base_units(sqrt_2_kt)  # Attempt to convert
        - For composite SymPy expressions, returns unchanged with warning
        - evaluate() still works correctly because units are preserved in tree

        Design (2025-12): For composite SymPy expressions (Pow, Mul, etc.),
        uw.to_base_units() returns unchanged - the expression tree preserves
        units through evaluation. Use uw.get_units() to check units.
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

        # Attempt to convert to base units - should warn and return unchanged
        # (composite SymPy expressions preserve units through evaluation)
        with pytest.warns(UserWarning, match="Unit conversion on SymPy"):
            sqrt_2kt_m = uw.to_base_units(sqrt_2_kt)

        # Expression is returned unchanged - same object
        assert sqrt_2kt_m is sqrt_2_kt, "Composite expression should be returned unchanged"

        # Evaluate again - MUST match original value (expression unchanged)
        result_conv = uw.function.evaluate(sqrt_2kt_m, self.mesh.X.coords[60:62])
        val_conv = float(result_conv.flat[0].magnitude if hasattr(result_conv.flat[0], 'magnitude') else result_conv.flat[0])

        # Critical assertion: Values must match (expression was unchanged)
        assert np.allclose(val_orig, val_conv, rtol=1e-6), \
            f"uw.to_base_units() should not change evaluation! Original: {val_orig:.2f} m, After: {val_conv:.2e} m"

    def test_to_reduced_units_composite_expression(self):
        """
        Test uw.to_reduced_units() on composite expression with UWexpression symbols.

        Similar to uw.to_base_units() - for composite SymPy expressions,
        returns unchanged with a warning.
        """
        # Create composite expression
        kappa_phys = uw.quantity(1e-6, "m**2/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 'Myr'), "Current time")
        sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5

        # Evaluate original
        result_orig = uw.function.evaluate(sqrt_2_kt, self.mesh.X.coords[60:62])
        val_orig = float(result_orig.flat[0].magnitude if hasattr(result_orig.flat[0], 'magnitude') else result_orig.flat[0])

        # Attempt to reduce units - should warn and return unchanged
        with pytest.warns(UserWarning, match="SymPy"):
            sqrt_2kt_reduced = uw.to_reduced_units(sqrt_2_kt)

        # Expression is returned unchanged - same object
        assert sqrt_2kt_reduced is sqrt_2_kt, "Composite expression should be returned unchanged"

        # Units are still the original complex units (no change)
        reduced_units = uw.get_units(sqrt_2kt_reduced)
        assert "megayear" in str(reduced_units)  # Still has original units

        # Evaluate reduced - MUST match original value
        result_reduced = uw.function.evaluate(sqrt_2kt_reduced, self.mesh.X.coords[60:62])
        val_reduced = float(result_reduced.flat[0].magnitude if hasattr(result_reduced.flat[0], 'magnitude') else result_reduced.flat[0])

        # Critical assertion: Values must match
        assert np.allclose(val_orig, val_reduced, rtol=1e-6), \
            f"uw.to_reduced_units() should not change evaluation! Original: {val_orig:.2f} m, Reduced: {val_reduced:.2e} m"

    def test_to_compact_still_works(self):
        """
        Test that uw.to_compact() works correctly on composite expressions.

        For composite SymPy expressions, uw.to_compact() returns unchanged with warning.
        """
        # Create composite expression
        kappa_phys = uw.quantity(1e-6, "m**2/s")
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 'Myr'), "Current time")
        sqrt_2_kt = ((2 * kappa_phys * t_now))**0.5

        # Evaluate original
        result_orig = uw.function.evaluate(sqrt_2_kt, self.mesh.X.coords[60:62])
        val_orig = float(result_orig.flat[0].magnitude if hasattr(result_orig.flat[0], 'magnitude') else result_orig.flat[0])

        # Attempt to compact units - should warn and return unchanged
        with pytest.warns(UserWarning, match="SymPy"):
            sqrt_2kt_compact = uw.to_compact(sqrt_2_kt)

        # Expression is returned unchanged - same object
        assert sqrt_2kt_compact is sqrt_2_kt, "Composite expression should be returned unchanged"

        # Evaluate compact - MUST match original value
        result_compact = uw.function.evaluate(sqrt_2kt_compact, self.mesh.X.coords[60:62])
        val_compact = float(result_compact.flat[0].magnitude if hasattr(result_compact.flat[0], 'magnitude') else result_compact.flat[0])

        # Critical assertion: Values must match
        assert np.allclose(val_orig, val_compact, rtol=1e-6), \
            f"uw.to_compact() should not change evaluation! Original: {val_orig:.2f} m, Compact: {val_compact:.2e} m"

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
