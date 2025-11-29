"""
Test evaluation of coordinate symbols (BaseScalar objects) with nondimensional scaling.

CRITICAL BUG FIX: Evaluating coordinate symbols (x, y, z) was applying dimensional
scaling TWICE, resulting in values like 8.41×10¹² meters instead of 2900 km.

ROOT CAUSE: non_dimensionalise() was failing silently when UnitAwareArray._units
was a Pint Unit object instead of a string, causing coordinates to not be
non-dimensionalized before evaluation.

THE CHAIN OF FAILURE:
1. Coordinates stored as [0, 2.9e6] meters (dimensional)
2. non_dimensionalise(coords) failed → returned [0, 2.9e6] (unchanged)
3. evaluate(x, coords) used these as [0-1] coords → returned [0, 2.9e6]
4. dimensionalise(result) scaled again → [0, 2.9e6] × 2900 km = 8.41×10¹² m ❌

THE FIX: Use Pint directly in non_dimensionalise():
- Check units_obj.dimensionality using Pint (not string comparison)
- Extract dict(units_obj.dimensionality) only when needed
- No string conversions, pure Pint operations

TEST COVERAGE:
- Detects double-scaling bug (values > 1e10 when expecting ~1e6)
- Tests non_dimensionalise() with Pint Unit objects
- Verifies coordinate evaluation returns physical values
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw


@pytest.mark.tier_a  # Production-ready - critical for correctness
@pytest.mark.level_2  # Intermediate - involves nondimensional scaling
class TestCoordinateSymbolEvaluation:
    """
    Test that coordinate symbols (x, y, z as BaseScalar) evaluate correctly
    with nondimensional scaling active.

    This prevents the double-scaling bug where coordinates were scaled by
    the length scale twice.
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

        self.L_scale = L_scale
        self.H_domain = H_domain

        yield

        uw.use_nondimensional_scaling(False)

    def test_coordinate_x_max_value(self):
        """
        Test: max(x) should equal domain width, not domain_width².

        CRITICAL: This was returning 8.41×10¹² meters (2900 km × 2900 km)
        instead of 2.9×10⁶ meters (2900 km).
        """
        x, y = self.mesh.CoordinateSystem.X

        result = uw.function.evaluate(x, self.mesh.X.coords).max()

        # Convert to meters for comparison
        if hasattr(result, 'to'):
            result_m = result.to('m')
            value = float(result_m.value) if hasattr(result_m, 'value') else float(result_m.magnitude)
        else:
            value = float(result)

        expected_m = 2900000.0  # 2900 km in meters

        # Should NOT be 8.41×10¹² (the double-scaling bug value)
        assert value < 1e10, \
            f"max(x) is {value} m - looks like double scaling! Expected ~{expected_m} m"

        # Should be correct domain width
        assert np.allclose(value, expected_m, rtol=1e-6), \
            f"max(x) should be {expected_m} m (2900 km), got {value} m"

    def test_coordinate_no_double_scaling(self):
        """
        Test: Coordinate evaluation doesn't apply double scaling.

        The critical bug was returning 8.41×10¹² m instead of ~2.9×10⁶ m.
        This test checks that values are in the right order of magnitude.
        """
        x, y = self.mesh.CoordinateSystem.X

        result_x = uw.function.evaluate(x, self.mesh.X.coords).max()

        # Extract value
        if hasattr(result_x, 'to'):
            result_m = result_x.to('m')
            value = float(result_m.value) if hasattr(result_m, 'value') else float(result_m.magnitude)
        else:
            value = float(result_x)

        # Should be order of magnitude ~1e6 (millions of meters)
        # NOT ~1e12 (trillions - the double-scaling bug)
        assert value < 1e10, \
            f"Coordinate value {value:.2e} m suggests double-scaling bug! Should be < 1e10 m"

        assert value > 1e5, \
            f"Coordinate value {value:.2e} m is too small, expected ~1e6 m"

    def test_unitawareexpression_coordinate_to_meters(self):
        """
        Test: UnitAwareExpression(x, units.m).to('m') gives correct value.

        CRITICAL: This was also double-scaling to 8.41×10¹² meters.
        """
        x, y = self.mesh.CoordinateSystem.X
        xx = uw.expression_types.UnitAwareExpression(x, uw.units.m)

        result = uw.function.evaluate(xx.to('m'), self.mesh.X.coords).max()

        # Extract value
        if hasattr(result, 'magnitude'):
            value = float(result.magnitude)
        elif hasattr(result, 'value'):
            value = float(result.value)
        else:
            value = float(result)

        expected_m = 2900000.0  # 2900 km in meters

        # Should NOT be 8.41×10¹² (double scaling)
        assert value < 1e10, \
            f"max(xx.to('m')) is {value} m - looks like double scaling! Expected ~{expected_m} m"

        # Should be correct
        assert np.allclose(value, expected_m, rtol=1e-6), \
            f"max(xx.to('m')) should be {expected_m} m, got {value} m"

    def test_non_dimensionalise_unit_aware_array(self):
        """
        Test: non_dimensionalise() works correctly on UnitAwareArray with Pint Units.

        This is the core fix - ensure UnitAwareArray._units (Pint Unit object)
        is handled correctly, not treated as a string.
        """
        coords = self.mesh.X.coords

        # Verify coordinates have Pint Unit objects (not strings)
        assert hasattr(coords, '_units'), "Coordinates should be UnitAwareArray"
        assert hasattr(coords._units, 'dimensionality'), \
            f"Coordinates._units should be Pint Unit, got {type(coords._units)}"

        # Non-dimensionalize - this should NOT fail with string conversion error
        coords_nd = uw.non_dimensionalise(coords)

        # Should be in [0-1] range (nondimensional)
        # Extract actual values
        if hasattr(coords_nd, '_array'):
            arr = coords_nd._array
        else:
            arr = np.asarray(coords_nd)

        max_val = float(np.max(arr))
        min_val = float(np.min(arr))

        assert max_val <= 1.05, \
            f"Non-dimensionalized coords max should be ~1.0, got {max_val}"
        assert min_val >= -0.05, \
            f"Non-dimensionalized coords min should be ~0.0, got {min_val}"

        # The critical check: Should be ~[0-1], NOT millions (dimensional)
        assert max_val < 10, \
            f"Non-dim coords still dimensional! Max={max_val}, expected < 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
