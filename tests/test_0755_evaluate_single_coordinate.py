"""
Test evaluate() with single coordinate points.

CRITICAL BUG FIX: evaluate() was failing with IndexError when evaluating
compound expressions (like UWQuantity * UWexpression) at single coordinate
points due to shape mismatch.

The Cython layer expects coordinates in shape (N, ndim), but single coordinates
from coords[i] have shape (ndim,). The fix ensures coordinates are always 2D
before passing to Cython.

TEST COVERAGE:
- Simple expressions with single coordinates (was working)
- Compound expressions with single coordinates (was failing, now fixed)
- Array coordinates still work correctly
"""

import pytest
import numpy as np
import underworld3 as uw


@pytest.mark.tier_a  # Production-ready
@pytest.mark.level_1  # Quick test
class TestEvaluateSingleCoordinate:
    """
    Test that evaluate() works with both single coordinates and coordinate arrays.

    This ensures proper shape handling: coords[i] has shape (ndim,), but the
    Cython layer needs shape (N, ndim).
    """

    @pytest.fixture(autouse=True)
    def setup_mesh(self):
        """Create simple mesh for testing."""
        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0),
            maxCoords=(1, 1),
            cellSize=0.1,
            qdegree=2,
        )
        yield

    def test_simple_expression_single_coordinate(self):
        """Test: Simple expression evaluates at single coordinate."""
        # This was already working, but ensure it still works
        import sympy
        x = sympy.Symbol('x')
        expr = x**2

        result = uw.function.evaluate(expr, self.mesh.X.coords[60])

        # Should return an array, even for single coordinate
        assert isinstance(result, (np.ndarray, uw.utilities.UnitAwareArray))

    def test_compound_expression_single_coordinate(self):
        """
        Test: Compound expression (UWQuantity * UWexpression) at single coordinate.

        CRITICAL: This was failing with IndexError before the fix!
        """
        t_now = uw.expression("t_now", uw.quantity(1, 's'), "Current time")
        velocity = uw.quantity(5, "cm/year")

        # This should NOT raise IndexError
        result = uw.function.evaluate(velocity * t_now, self.mesh.X.coords[60])

        assert isinstance(result, (np.ndarray, uw.utilities.UnitAwareArray))

    def test_compound_expression_array_coordinates(self):
        """Test: Compound expression at multiple coordinates (sanity check)."""
        t_now = uw.expression("t_now", uw.quantity(1, 's'), "Current time")
        velocity = uw.quantity(5, "cm/year")

        # This should work (and was working before the fix)
        result = uw.function.evaluate(velocity * t_now, self.mesh.X.coords[60:70])

        assert isinstance(result, (np.ndarray, uw.utilities.UnitAwareArray))
        # Should have 10 results for 10 coordinates
        assert result.shape[0] == 10

    def test_uwquantity_single_coordinate(self):
        """Test: UWQuantity evaluation at single coordinate."""
        qty = uw.quantity(100, "m")

        result = uw.function.evaluate(qty, self.mesh.X.coords[60])

        assert isinstance(result, (np.ndarray, uw.utilities.UnitAwareArray))

    def test_coordinate_shape_consistency(self):
        """
        Test: Coordinate shape handling is consistent.

        Single coordinate coords[i] has shape (ndim,)
        Multiple coords[i:j] has shape (N, ndim)
        Both should work identically after internal reshaping.
        """
        import sympy
        x = sympy.Symbol('x')
        expr = x**2

        # Get the same coordinate two ways
        single_coord = self.mesh.X.coords[60]
        array_coord = self.mesh.X.coords[60:61]  # Single element array

        result_single = uw.function.evaluate(expr, single_coord)
        result_array = uw.function.evaluate(expr, array_coord)

        # Both should give the same numerical result
        # (shapes might differ due to output formatting, but values should match)
        assert np.allclose(np.asarray(result_single).flat[0],
                          np.asarray(result_array).flat[0])


@pytest.mark.tier_a
@pytest.mark.level_2
class TestEvaluateSingleCoordinateNondimensional:
    """
    Test single coordinate evaluation with nondimensional scaling active.

    This is the exact scenario from the user's bug report.
    """

    @pytest.fixture(autouse=True)
    def setup_nondimensional_model(self):
        """Set up model with nondimensional scaling."""
        model = uw.Model()

        L_scale = uw.quantity(2900, "km")
        t_scale = uw.quantity(1, "Myr")
        M_scale = uw.quantity(1e24, "kg")
        T_scale = uw.quantity(1000, "K")

        model.set_reference_quantities(
            length=L_scale,
            time=t_scale,
            mass=M_scale,
            temperature=T_scale,
            nondimensional_scaling=True,
        )

        uw.use_nondimensional_scaling(True)

        L_domain = uw.quantity(2900, "km")
        H_domain = uw.quantity(1000, "km")
        cellSize_phys = L_domain / 12

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0 * uw.units.km, 0 * uw.units.km),
            maxCoords=(L_domain, H_domain),
            cellSize=cellSize_phys,
            regular=False,
            qdegree=3,
        )

        yield

        uw.use_nondimensional_scaling(False)

    def test_user_bug_report_case(self):
        """
        Test: Exact case from user bug report.

        User's code:
        t_now = uw.expression(r"t_\\textrm{now}", uw.quantity(1, 's'), "Current time")
        velocity_phys = uw.quantity(5, "cm/year")
        uw.function.evaluate((velocity_phys*t_now), mesh.X.coords[60])

        This was raising IndexError: tuple index out of range
        """
        # Use 1 year for a reasonable displacement
        t_now = uw.expression(r"t_\textrm{now}", uw.quantity(1, 'year'), "Current time")
        velocity_phys = uw.quantity(5, "cm/year")

        # This should NOT raise IndexError
        result = uw.function.evaluate((velocity_phys * t_now), self.mesh.X.coords[60])

        # Should return physical units (meters), not nondimensional
        assert hasattr(result, 'units')
        # Value should be 5 cm = 0.05 m (velocity × time = displacement)
        result_m = result.to('m') if hasattr(result, 'to') else result
        value = float(result_m.magnitude if hasattr(result_m, 'magnitude') else result_m)
        assert np.allclose(value, 0.05, rtol=1e-6), \
            f"5 cm/year × 1 year should equal 5 cm = 0.05 m, got {value} m"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
