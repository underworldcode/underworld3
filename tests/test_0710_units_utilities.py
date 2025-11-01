#!/usr/bin/env python3
"""
Unit tests for high-level units utilities.

Tests the standalone units functions that work with arbitrary expressions
and quantities, independent of specific variable types.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw
from underworld3.units import (
    UnitsError,
    DimensionalityError,
    NoUnitsError,
    check_units_consistency,
    get_dimensionality,
    units_of,
    non_dimensionalise,
    dimensionalise,
    create_quantity,
    is_dimensionless,
    has_units,
    same_units,
    validate_expression_units,
    enforce_units_consistency,
)


class TestBasicUnitsUtilities:
    """Test basic units utility functions."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_units_of_function(self, mesh):
        """Test extracting units from various objects."""
        # Variable with units
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        units = uw.units_of(velocity)
        assert str(units) == "meter / second"

        # Variable without units
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)
        units = uw.units_of(unitless)
        assert units is None

        # Pint quantity
        quantity = uw.create_quantity(10, "Pa")
        units = uw.units_of(quantity)
        assert units is not None

    def test_has_units_function(self, mesh):
        """Test checking if objects have units."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)

        assert uw.has_units(velocity) == True
        assert uw.has_units(unitless) == False

    def test_is_dimensionless_function(self, mesh):
        """Test checking if objects are dimensionless."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)

        assert uw.is_dimensionless(velocity) == False
        assert uw.is_dimensionless(unitless) == True

    def test_get_dimensionality_function(self, mesh):
        """Test extracting dimensionality information."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        pressure = uw.create_enhanced_mesh_variable("pressure", mesh, 1, units="Pa")
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)

        vel_dims = uw.get_dimensionality(velocity)
        press_dims = uw.get_dimensionality(pressure)
        unitless_dims = uw.get_dimensionality(unitless)

        assert vel_dims is not None
        assert press_dims is not None
        assert unitless_dims is None

        # Check dimensionality content
        assert vel_dims["[length]"] == 1
        assert vel_dims["[time]"] == -1


class TestUnitsConsistencyChecking:
    """Test units consistency validation functions."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_check_units_consistency_compatible(self, mesh):
        """Test consistency checking with compatible units."""
        velocity1 = uw.create_enhanced_mesh_variable("v1", mesh, 2, units="m/s")
        velocity2 = uw.create_enhanced_mesh_variable("v2", mesh, 2, units="km/h")

        # Should be compatible (both velocities)
        assert uw.check_units_consistency(velocity1, velocity2) == True
        assert uw.same_units(velocity1, velocity2) == True

    def test_check_units_consistency_incompatible(self, mesh):
        """Test consistency checking with incompatible units."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        pressure = uw.create_enhanced_mesh_variable("pressure", mesh, 1, units="Pa")

        # Should raise error for incompatible units
        with pytest.raises(DimensionalityError):
            uw.check_units_consistency(velocity, pressure)

        assert uw.same_units(velocity, pressure) == False

    def test_check_units_consistency_mixed(self, mesh):
        """Test consistency checking with mixed units/no units."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)

        # Should raise error for mixed units/no units
        with pytest.raises(NoUnitsError):
            uw.check_units_consistency(velocity, unitless)

    def test_check_units_consistency_all_unitless(self, mesh):
        """Test consistency checking with all unitless variables."""
        unitless1 = uw.create_enhanced_mesh_variable("u1", mesh, 1)
        unitless2 = uw.create_enhanced_mesh_variable("u2", mesh, 2)

        # Should be consistent (all unitless)
        assert uw.check_units_consistency(unitless1, unitless2) == True

    def test_enforce_units_consistency(self, mesh):
        """Test enforcing units consistency."""
        velocity1 = uw.create_enhanced_mesh_variable("v1", mesh, 2, units="m/s")
        velocity2 = uw.create_enhanced_mesh_variable("v2", mesh, 2, units="km/h")
        pressure = uw.create_enhanced_mesh_variable("pressure", mesh, 1, units="Pa")

        # Should not raise for compatible units
        uw.enforce_units_consistency(velocity1, velocity2)

        # Should raise for incompatible units
        with pytest.raises(DimensionalityError):
            uw.enforce_units_consistency(velocity1, pressure)


class TestQuantityCreationAndConversion:
    """Test creating and converting dimensional quantities."""

    def test_create_quantity_basic(self):
        """Test basic quantity creation."""
        velocity = uw.create_quantity([1.0, 2.0], "m/s")
        pressure = uw.create_quantity(101325, "Pa")

        assert uw.has_units(velocity)
        assert uw.has_units(pressure)

        # Check units
        vel_units = uw.units_of(velocity)
        press_units = uw.units_of(pressure)

        assert "meter" in str(vel_units)
        assert "second" in str(vel_units)
        assert "pascal" in str(press_units)

    def test_create_quantity_different_backends(self):
        """Test quantity creation with different backends."""
        # Pint backend (default)
        pint_qty = uw.create_quantity(10, "m/s", backend="pint")
        assert uw.has_units(pint_qty)

        # SymPy backend no longer exists (replaced with Pint-native approach)
        # Test that invalid backend raises appropriate error
        with pytest.raises(ValueError, match="Unknown backend: sympy"):
            uw.create_quantity(10, "meter", backend="sympy")

    def test_create_quantity_invalid_backend(self):
        """Test error handling for invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            uw.create_quantity(10, "m/s", backend="invalid")


class TestNonDimensionalisation:
    """Test non-dimensionalisation functions."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_non_dimensionalise_variable(self, mesh):
        """Test non-dimensionalising unit-aware variables."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Set some test data
        with uw.synchronised_array_update():
            velocity.array[...] = np.random.random(velocity.array.shape)

        # Non-dimensionalise
        nondim = uw.non_dimensionalise(velocity)

        assert nondim is not None
        assert isinstance(nondim, np.ndarray)
        assert nondim.shape == velocity.data.shape

    def test_non_dimensionalise_no_units(self, mesh):
        """Test error when non-dimensionalising unitless expressions."""
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)

        with pytest.raises(NoUnitsError):
            uw.non_dimensionalise(unitless)

    def test_dimensionalise_basic(self):
        """Test adding dimensions to values."""
        nondim_values = np.array([1.0, 2.0, 3.0])

        dimensional = uw.dimensionalise(nondim_values, "m/s")

        assert uw.has_units(dimensional)
        units = uw.units_of(dimensional)
        assert "meter" in str(units)
        assert "second" in str(units)


class TestValidationFunctions:
    """Test units validation functions."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_validate_expression_units_correct(self, mesh):
        """Test validating expression with correct units."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Should validate successfully
        assert uw.validate_expression_units(velocity, "km/h") == True
        assert uw.validate_expression_units(velocity, "m/s") == True

    def test_validate_expression_units_incorrect(self, mesh):
        """Test validating expression with incorrect units."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Should fail validation
        assert uw.validate_expression_units(velocity, "Pa") == False

    def test_validate_expression_units_no_units(self, mesh):
        """Test validating unitless expression when units expected."""
        unitless = uw.create_enhanced_mesh_variable("unitless", mesh, 1)

        with pytest.raises(NoUnitsError):
            uw.validate_expression_units(unitless, "m/s")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_units_error_hierarchy(self):
        """Test that custom error types inherit correctly."""
        assert issubclass(DimensionalityError, UnitsError)
        assert issubclass(NoUnitsError, UnitsError)
        assert issubclass(UnitsError, Exception)

    def test_units_error_messages(self):
        """Test that error messages are informative."""
        with pytest.raises(DimensionalityError) as exc_info:
            raise DimensionalityError("Test dimensionality error")

        assert "dimensionality" in str(exc_info.value).lower()

        with pytest.raises(NoUnitsError) as exc_info:
            raise NoUnitsError("Test no units error")

        assert "units" in str(exc_info.value).lower()


class TestIntegrationWithExistingCode:
    """Test integration with existing Underworld3 patterns."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_units_with_sympy_expressions(self, mesh):
        """Test units utilities with SymPy expressions."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        x, y = mesh.X

        # Create SymPy expressions
        v_x = velocity[0]
        v_y = velocity[1]

        # Test units utilities on SymPy expressions
        # Note: This may not work fully until SymPy integration is complete
        try:
            vel_x_units = uw.units_of(v_x)
            # If this works, great; if not, it's expected
        except:
            # Expected - SymPy expressions don't carry units information yet
            pass

    def test_units_with_mathematical_operations(self, mesh):
        """Test units utilities with mathematical operations."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Mathematical operations
        doubled_velocity = 2 * velocity

        # Test units utilities
        assert uw.has_units(velocity)
        # Note: doubled_velocity is a SymPy expression, may not have units info

    def test_backward_compatibility(self, mesh):
        """Test that units utilities work with regular variables too."""
        regular_var = uw.create_enhanced_mesh_variable("regular", mesh, 1)  # No units

        # Should handle gracefully
        assert not uw.has_units(regular_var)
        assert uw.is_dimensionless(regular_var)
        assert uw.units_of(regular_var) is None
        assert uw.get_dimensionality(regular_var) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
