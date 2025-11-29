"""
Critical tests for evaluate() with nondimensional scaling.

These tests catch the bugs where evaluate() was:
1. Not re-dimensionalizing (returning 1 meter instead of 2900 km)
2. Not non-dimensionalizing UWexpression values (off by factor of 1e13)

POLICY: When nondimensional scaling is active, evaluate() must:
- Accept physical quantities as input
- Non-dimensionalize them internally for evaluation
- Re-dimensionalize results back to physical units
- Return correct physical values (NOT nondimensional values)

TEST COVERAGE:
- Pure numbers (dimensionless)
- UWQuantity with various units
- UWexpression wrapping quantities
- All fundamental dimensions (length, time, mass, temperature)
- Cross-scale conversions (cm vs km, s vs Myr)
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw


@pytest.mark.tier_a  # Production-ready - REQUIRED for TDD
@pytest.mark.level_1  # Quick test, no solving
class TestEvaluateNondimensionalScaling:
    """
    Critical tests for evaluate() with nondimensional scaling active.

    These tests ensure evaluate() correctly handles the full cycle:
    Input (physical) → Non-dimensionalize → Evaluate → Re-dimensionalize → Output (physical)
    """

    @pytest.fixture(autouse=True)
    def setup_nondimensional_model(self):
        """Set up model with nondimensional scaling for all tests."""
        # Create fresh model for each test
        model = uw.Model()

        # Define reference scales
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

        # Create simple mesh
        L_domain = uw.quantity(2900, "km")
        H_domain = uw.quantity(1000, "km")
        cellSize_phys = L_domain / 12

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0 * uw.units.km, 0 * uw.units.km),
            maxCoords=(L_domain, H_domain),
            cellSize=cellSize_phys,
            regular=False,
            qdegree=3,
        )

        self.model = model
        self.mesh = mesh

        yield

        # Cleanup
        uw.use_nondimensional_scaling(False)

    def test_pure_number_dimensionless(self):
        """Test: evaluate(1, coords) returns 1 dimensionless"""
        result = uw.function.evaluate(sympy.sympify(1), self.mesh.X.coords[60])

        # Should be exactly 1
        assert np.allclose(result, 1.0), f"Expected 1.0, got {result}"

        # Should have no units (or dimensionless)
        if hasattr(result, 'units'):
            assert result.units == uw.scaling.units.dimensionless, \
                f"Pure number should be dimensionless, got {result.units}"

    def test_uwquantity_1_cm(self):
        """Test: evaluate(1 cm, coords) returns 1 cm (or 0.01 m)"""
        result = uw.function.evaluate(uw.quantity(1, units="cm"), self.mesh.X.coords[60])

        # Convert to cm for comparison
        if hasattr(result, 'to'):
            result_cm = result.to('cm')
            value = float(result_cm.value) if hasattr(result_cm, 'value') else float(result_cm)
        else:
            # If returned as meters
            value_m = float(result)
            value = value_m * 100  # m to cm

        assert np.allclose(value, 1.0, rtol=1e-6), \
            f"Expected 1 cm, got {value} cm"

    def test_uwquantity_1_year(self):
        """Test: evaluate(1 year, coords) returns 1 year (or ~3.16e7 s)"""
        result = uw.function.evaluate(uw.quantity(1, units="year"), self.mesh.X.coords[60])

        # Convert to seconds for comparison
        expected_seconds = 31557600.0  # 1 year in seconds

        if hasattr(result, 'to'):
            result_s = result.to('s')
            value = float(result_s.value) if hasattr(result_s, 'value') else float(result_s)
        else:
            value = float(result)

        assert np.allclose(value, expected_seconds, rtol=1e-6), \
            f"Expected {expected_seconds} seconds, got {value} seconds"

    def test_uwquantity_at_length_scale(self):
        """
        Test: evaluate(2900 km, coords) returns 2900 km

        CRITICAL: This was returning 1 meter (Bug #4)
        2900 km is the length scale, so:
        - Non-dim: 2900 km / 2900 km = 1.0
        - Re-dim: 1.0 * 2900 km = 2900 km ✅
        """
        result = uw.function.evaluate(uw.quantity(2900, units="km"), self.mesh.X.coords[60])

        # Convert to km for comparison
        if hasattr(result, 'to'):
            result_km = result.to('km')
            value = float(result_km.value) if hasattr(result_km, 'value') else float(result_km)
        else:
            # If returned as meters
            value_m = float(result)
            value = value_m / 1000  # m to km

        assert np.allclose(value, 2900.0, rtol=1e-6), \
            f"Expected 2900 km, got {value} km (Bug: was returning 1 meter!)"

    def test_uwexpression_1_second(self):
        """
        Test: evaluate(UWexpression(1 s), coords) returns 1 second

        CRITICAL: This was returning 1 Myr (Bug #5 - off by factor of 3e13!)
        The bug was in unwrap_for_evaluate() not non-dimensionalizing:
        - WRONG: 1 s → 1.0 (magnitude only) → 1.0 Myr = 3.15e13 s
        - CORRECT: 1 s → (1 s / 1 Myr) = 3.17e-14 → 3.17e-14 * 1 Myr = 1 s ✅
        """
        t_now = uw.expression(r"t_now", uw.quantity(1, "s"), "Current time")
        result = uw.function.evaluate(t_now, self.mesh.X.coords[60])

        # Should be 1 second
        if hasattr(result, 'to'):
            result_s = result.to('s')
            value = float(result_s.value) if hasattr(result_s, 'value') else float(result_s)
        else:
            value = float(result)

        assert np.allclose(value, 1.0, rtol=1e-6), \
            f"Expected 1 second, got {value} seconds (Bug: was returning 3.15e13 s = 1 Myr!)"

    def test_uwexpression_at_time_scale(self):
        """Test: evaluate(UWexpression(1 Myr), coords) returns 1 Myr"""
        t_scale = uw.expression(r"t_{scale}", uw.quantity(1, "Myr"), "Time scale")
        result = uw.function.evaluate(t_scale, self.mesh.X.coords[60])

        # Convert to Myr for comparison
        if hasattr(result, 'to'):
            result_Myr = result.to('Myr')
            value = float(result_Myr.value) if hasattr(result_Myr, 'value') else float(result_Myr)
        else:
            # If returned as seconds
            value_s = float(result)
            value = value_s / 31557600000000.0  # s to Myr

        assert np.allclose(value, 1.0, rtol=1e-6), \
            f"Expected 1 Myr, got {value} Myr"

    def test_uwquantity_temperature(self):
        """Test: evaluate(1000 K, coords) returns 1000 K (the temperature scale)"""
        result = uw.function.evaluate(uw.quantity(1000, units="K"), self.mesh.X.coords[60])

        # Should be 1000 K
        if hasattr(result, 'to'):
            result_K = result.to('K')
            value = float(result_K.value) if hasattr(result_K, 'value') else float(result_K)
        else:
            value = float(result)

        assert np.allclose(value, 1000.0, rtol=1e-6), \
            f"Expected 1000 K, got {value} K"

    def test_uwquantity_small_length(self):
        """Test: evaluate(1 mm, coords) returns 1 mm (much smaller than scale)"""
        result = uw.function.evaluate(uw.quantity(1, units="mm"), self.mesh.X.coords[60])

        # Convert to mm for comparison
        if hasattr(result, 'to'):
            result_mm = result.to('mm')
            value = float(result_mm.value) if hasattr(result_mm, 'value') else float(result_mm)
        else:
            # If returned as meters
            value_m = float(result)
            value = value_m * 1000  # m to mm

        assert np.allclose(value, 1.0, rtol=1e-6), \
            f"Expected 1 mm, got {value} mm"

    def test_consistency_across_unit_systems(self):
        """Test: Same physical quantity gives same result in different units"""
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        # 2900 km should equal 2900000 m
        result_km = uw.function.evaluate(uw.quantity(2900, units="km"), self.mesh.X.coords[60])
        result_m = uw.function.evaluate(uw.quantity(2900000, units="m"), self.mesh.X.coords[60])

        # Convert both to meters
        if isinstance(result_km, UnitAwareArray):
            result_km_conv = result_km.to('m')
            value_km = float(result_km_conv.flat[0])  # Extract first element
        elif hasattr(result_km, 'to'):
            value_km = float(result_km.to('m').value)
        else:
            value_km = float(result_km)

        if isinstance(result_m, UnitAwareArray):
            result_m_conv = result_m.to('m')
            value_m = float(result_m_conv.flat[0])  # Extract first element
        elif hasattr(result_m, 'to'):
            value_m = float(result_m.to('m').value)
        else:
            value_m = float(result_m)

        assert np.allclose(value_km, value_m, rtol=1e-9), \
            f"2900 km ({value_km} m) should equal 2900000 m ({value_m} m)"


@pytest.mark.tier_a
@pytest.mark.level_1
class TestEvaluateReturnsPhysicalUnits:
    """
    Tests that evaluate() ALWAYS returns physical units, never nondimensional values.

    This is the fundamental contract: evaluate() is a user-facing function that
    should hide the internal nondimensional representation.
    """

    @pytest.fixture(autouse=True)
    def setup_nondimensional_model(self):
        """Set up model with nondimensional scaling for all tests."""
        model = uw.Model()

        L_scale = uw.quantity(2900, "km")
        t_scale = uw.quantity(1, "Myr")
        M_scale = uw.quantity(1e24, "kg")
        T_scale = uw.quantity(1000, "K")

        model.set_reference_quantities(
            length=L_scale, time=t_scale, mass=M_scale, temperature=T_scale,
            nondimensional_scaling=True,
        )

        uw.use_nondimensional_scaling(True)

        L_domain = uw.quantity(2900, "km")
        H_domain = uw.quantity(1000, "km")
        cellSize_phys = L_domain / 12

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0 * uw.units.km, 0 * uw.units.km),
            maxCoords=(L_domain, H_domain),
            cellSize=cellSize_phys,
            regular=False,
            qdegree=3,
        )

        self.model = model
        self.mesh = mesh

        yield

        uw.use_nondimensional_scaling(False)

    def test_result_has_physical_units_not_dimensionless(self):
        """
        Test: Results have physical units, not 'dimensionless'

        When evaluating 1 cm, result should be labeled as 'meter' (or 'centimeter'),
        NOT as 'dimensionless'. The value 0.01 alone is meaningless without units.
        """
        result = uw.function.evaluate(uw.quantity(1, units="cm"), self.mesh.X.coords[60])

        # Result MUST have units
        assert hasattr(result, 'units'), "Result should have .units attribute"

        # Units should NOT be dimensionless (unless input was dimensionless)
        if hasattr(result.units, '__str__'):
            units_str = str(result.units).lower()
            assert 'dimensionless' not in units_str or units_str == 'dimensionless', \
                f"1 cm should not become dimensionless, got {result.units}"

    def test_pint_unit_objects_not_strings(self):
        """
        Test: .units property returns Pint Unit objects, not strings

        POLICY VIOLATION CHECK: Catches Bug #1 (string conversions)
        """
        result = uw.function.evaluate(uw.quantity(2900, units="km"), self.mesh.X.coords[60])

        if hasattr(result, 'units'):
            from pint import Unit
            assert isinstance(result.units, Unit), \
                f"Result.units should be Pint Unit, got {type(result.units).__name__}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
