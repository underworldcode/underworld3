"""
Test evaluation of coordinate symbols (BaseScalar objects) with nondimensional scaling.

BUG FIXED (2026-01): Double scaling bug in coordinate evaluation.

ORIGINAL PROBLEM:
Evaluating coordinate symbols (x, y, z) was applying dimensional scaling TWICE,
resulting in values like 8.41×10¹² meters instead of 2900 km.

ROOT CAUSE (NOW FIXED):
1. model.set_reference_quantities() stored scales in model._fundamental_scales
2. BUT the global COEFFICIENTS dict used by non_dimensionalise() was NOT updated
3. So non_dimensionalise() used default scales (1.0 meter) instead of model scales

THE FIX:
- model.set_reference_quantities() now updates global COEFFICIENTS from _fundamental_scales
- non_dimensionalise() in _scaling.py properly handles UnitAwareArray with Pint Units
- Uses Pint's dimensionality directly (no string comparisons per project policy)

TEST COVERAGE:
- Regression tests for double-scaling bug (values > 1e10 when expecting ~1e6)
- Tests non_dimensionalise() with Pint Unit objects
- Verifies coordinate evaluation returns correct physical values
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw


def extract_scalar(result):
    """
    Extract a scalar value from evaluate() result.

    evaluate() returns UnitAwareArray with shape (1,1,1) for single point evaluation.
    This helper extracts the scalar value properly.
    """
    if hasattr(result, 'flat'):
        # UnitAwareArray or numpy array - extract first element
        scalar = result.flat[0]
        if hasattr(scalar, 'item'):
            return scalar.item()
        return float(scalar)
    elif hasattr(result, 'item'):
        return result.item()
    else:
        return float(result)


def convert_and_extract(result, target_units):
    """
    Convert result to target units and extract scalar.

    Returns the scalar value in the specified units.
    """
    if hasattr(result, 'to'):
        converted = result.to(target_units)
        return extract_scalar(converted)
    else:
        # Assume result is already in base units
        return extract_scalar(result)


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

    @pytest.mark.tier_a  # Production-ready - critical regression test
    def test_coordinate_x_max_value(self):
        """
        Test: max(x) should equal domain width, not domain_width².

        BUG FIXED (2026-01): Previously returned 8.41×10¹² meters (double scaling)
        instead of 2.9×10⁶ meters (2900 km). Fixed by connecting model._fundamental_scales
        to global COEFFICIENTS in model.set_reference_quantities().
        """
        x, y = self.mesh.CoordinateSystem.X

        result = uw.function.evaluate(x, self.mesh.X.coords).max()

        # Convert to meters for comparison - handle UnitAwareArray properly
        value = convert_and_extract(result, 'm') if hasattr(result, 'to') else extract_scalar(result)

        expected_m = 2900000.0  # 2900 km in meters

        # Should NOT be 8.41×10¹² (the double-scaling bug value)
        assert value < 1e10, \
            f"max(x) is {value} m - looks like double scaling! Expected ~{expected_m} m"

        # Should be correct domain width
        assert np.allclose(value, expected_m, rtol=1e-6), \
            f"max(x) should be {expected_m} m (2900 km), got {value} m"

    @pytest.mark.tier_a  # Production-ready - critical regression test
    def test_coordinate_no_double_scaling(self):
        """
        Test: Coordinate evaluation doesn't apply double scaling.

        BUG FIXED (2026-01): Previously returned 8.41×10¹² m instead of ~2.9×10⁶ m.
        This test verifies values are in the correct order of magnitude.
        """
        x, y = self.mesh.CoordinateSystem.X

        result_x = uw.function.evaluate(x, self.mesh.X.coords).max()

        # Extract value - handle UnitAwareArray properly
        value = convert_and_extract(result_x, 'm') if hasattr(result_x, 'to') else extract_scalar(result_x)

        # Should be order of magnitude ~1e6 (millions of meters)
        # NOT ~1e12 (trillions - the double-scaling bug)
        assert value < 1e10, \
            f"Coordinate value {value:.2e} m suggests double-scaling bug! Should be < 1e10 m"

        assert value > 1e5, \
            f"Coordinate value {value:.2e} m is too small, expected ~1e6 m"

    @pytest.mark.skip(reason="UnitAwareExpression class not implemented - feature replaced by simplified units architecture")
    def test_unitawareexpression_coordinate_to_meters(self):
        """
        Test: UnitAwareExpression(x, units.m).to('m') gives correct value.

        NOTE: UnitAwareExpression was planned but not implemented.
        The units architecture was simplified (see UNITS_SIMPLIFIED_DESIGN_2025-11.md).
        This test is kept for documentation of the double-scaling bug that was fixed.

        CRITICAL BUG (fixed): This was double-scaling to 8.41×10¹² meters.
        """
        pass

    @pytest.mark.tier_a  # Production-ready - critical regression test
    def test_non_dimensionalise_unit_aware_array(self):
        """
        Test: non_dimensionalise() works correctly on UnitAwareArray with Pint Units.

        BUG FIXED (2026-01): UnitAwareArray._units (Pint Unit object) now handled
        correctly using Pint's dimensionality directly (no string comparisons).
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
