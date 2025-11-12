#!/usr/bin/env python3
"""
Comprehensive unit tests for the Underworld3 units system.

This test suite validates:
- UnitAwareMixin functionality
- Backend implementations (Pint/SymPy) 
- Enhanced variable classes
- Mathematical operations with units
- Dimensional analysis and compatibility
- Integration with existing UW3 functionality
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
# REMOVED: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw
from underworld3.utilities import UnitAwareMixin, PintBackend, make_units_aware


class TestUnitsBackends:
    """Test units backend implementations."""

    def test_pint_backend_basic(self):
        """Test basic Pint backend functionality."""
        backend = PintBackend()

        # Test quantity creation
        velocity = backend.create_quantity(1.0, "m/s")
        assert backend.get_magnitude(velocity) == 1.0
        assert str(backend.get_units(velocity)) == "meter / second"

        # Test dimensionality
        dimensionality = backend.get_dimensionality(velocity)
        assert dimensionality["[length]"] == 1
        assert dimensionality["[time]"] == -1

    def test_pint_backend_compatibility(self):
        """Test Pint backend compatibility checking."""
        backend = PintBackend()

        velocity1 = backend.create_quantity(1.0, "m/s")
        velocity2 = backend.create_quantity(2.0, "km/h")  # Different but compatible
        pressure = backend.create_quantity(100000, "Pa")

        # Compatible units
        assert backend.check_dimensionality(velocity1, velocity2)

        # Incompatible units
        assert not backend.check_dimensionality(velocity1, pressure)


class TestUnitAwareMixin:
    """Test the core UnitAwareMixin functionality."""

    class MockVariable(UnitAwareMixin):
        """Mock variable for testing."""

        def __init__(self, name, data=None, **kwargs):
            self.name = name
            self.data = data if data is not None else [1.0, 2.0, 3.0]
            super().__init__(**kwargs)

    def test_unitaware_mixin_without_units(self):
        """Test UnitAwareMixin without units specified."""
        var = self.MockVariable("test")

        assert not var.has_units
        assert var.units is None
        assert var.dimensionality is None
        assert var.non_dimensional_value() == var.data

    def test_unitaware_mixin_with_pint_units(self):
        """Test UnitAwareMixin with Pint units."""
        var = self.MockVariable("test", units="m/s")

        assert var.has_units
        assert str(var.units) == "meter / second"
        assert var.dimensionality["[length]"] == 1
        assert var.dimensionality["[time]"] == -1

    def test_units_compatibility_checking(self):
        """Test units compatibility checking."""
        var1 = self.MockVariable("var1", units="m/s")
        var2 = self.MockVariable("var2", units="km/h")  # Compatible
        var3 = self.MockVariable("var3", units="Pa")  # Incompatible
        var4 = self.MockVariable("var4")  # No units

        # Same units
        assert var1.check_units_compatibility(var1)

        # Compatible units (different but same dimensionality)
        assert var1.check_units_compatibility(var2)

        # Incompatible units
        assert not var1.check_units_compatibility(var3)

        # One with units, one without
        assert not var1.check_units_compatibility(var4)

        # Both without units
        assert var4.check_units_compatibility(self.MockVariable("var5"))

    def test_create_quantity(self):
        """Test quantity creation from values."""
        var = self.MockVariable("test", units="m/s")

        quantity = var.create_quantity([1.0, 2.0])
        assert var._units_backend.get_magnitude(quantity)[0] == 1.0
        assert var._units_backend.get_magnitude(quantity)[1] == 2.0
        assert str(var._units_backend.get_units(quantity)) == "meter / second"

    def test_units_repr(self):
        """Test units representation."""
        var_with_units = self.MockVariable("test", units="m/s")
        var_without_units = self.MockVariable("test")

        assert "units: meter / second" in var_with_units.units_repr()
        assert "no units" in var_without_units.units_repr()


class TestEnhancedMeshVariable:
    """Test EnhancedMeshVariable functionality."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_enhanced_mesh_variable_creation(self, mesh):
        """Test creation of enhanced mesh variables."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        assert velocity.has_units
        assert str(velocity.units) == "meter / second"
        assert velocity.name == "velocity"
        assert velocity.num_components == 2

    def test_enhanced_mesh_variable_without_units(self, mesh):
        """Test enhanced mesh variable without units."""
        var = uw.create_enhanced_mesh_variable("test", mesh, 1)

        assert not var.has_units
        assert var.units is None

        # Should still work mathematically
        result = 2 * var
        assert result is not None

    def test_mathematical_operations_with_units(self, mesh):
        """Test mathematical operations preserve units information."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Initialize with some data (required for PETSc vector operations)
        with uw.synchronised_array_update():
            velocity.array[...] = 1.0

        # Component access (symbolic operations - work without data)
        v_x = velocity[0]
        assert v_x is not None

        # Scalar multiplication (symbolic)
        momentum_expr = 1000 * velocity
        assert momentum_expr is not None

        # Vector operations (symbolic)
        speed_squared = velocity.dot(velocity)
        assert speed_squared is not None

        # Norm - requires actual data and explicit norm_type (PETSc norm, not SymPy norm)
        from petsc4py.PETSc import NormType

        speed_norm = velocity.norm(NormType.NORM_2)  # L2 norm
        assert speed_norm is not None
        assert isinstance(speed_norm, tuple)  # Returns tuple for multi-component

    def test_units_arithmetic_validation(self, mesh):
        """Test that arithmetic operations validate units compatibility."""
        velocity1 = uw.create_enhanced_mesh_variable("v1", mesh, 2, units="m/s")
        velocity2 = uw.create_enhanced_mesh_variable("v2", mesh, 2, units="m/s")
        pressure = uw.create_enhanced_mesh_variable("p", mesh, 1, units="Pa")

        # Compatible addition should work
        velocity_sum = velocity1 + velocity2
        assert velocity_sum is not None

        # Incompatible addition should raise error
        with pytest.raises(ValueError, match="Cannot add incompatible units"):
            velocity1 + pressure

    def test_representation_with_units(self, mesh):
        """Test variable representation includes units."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        repr_str = repr(velocity)
        assert "units=meter / second" in repr_str

        units_repr = velocity.units_repr()
        assert "units: meter / second" in units_repr


class TestFactoryFunctions:
    """Test factory functions for creating enhanced variables."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    @pytest.fixture
    def swarm(self, mesh):
        """Create a test swarm."""
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)
        return swarm

    def test_create_enhanced_mesh_variable(self, mesh):
        """Test mesh variable factory function."""
        density = uw.create_enhanced_mesh_variable("density", mesh, 1, units="kg/m^3")

        assert isinstance(density, uw.discretisation.MeshVariable)
        assert density.has_units
        assert str(density.units) == "kilogram / meter ** 3"
        assert density.name == "density"

    def test_create_enhanced_swarm_variable(self, mesh):
        """Test swarm variable factory function."""
        # Create fresh swarm to avoid PETSc field registration issues
        fresh_swarm = uw.swarm.Swarm(mesh)

        temperature = uw.create_enhanced_swarm_variable("temperature", fresh_swarm, 1, units="K")

        assert isinstance(temperature, uw.swarm.SwarmVariable)
        assert temperature.has_units
        assert str(temperature.units) == "kelvin"
        assert temperature.name == "temperature"

        # Now populate to complete the test
        fresh_swarm.populate(fill_param=2)


class TestMakeUnitsAware:
    """Test the make_units_aware factory function."""

    def test_make_units_aware_function(self):
        """Test making any class units-aware."""

        class SimpleVariable:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        # Make it units-aware
        UnitAwareSimpleVariable = make_units_aware(SimpleVariable)

        # Test creation with units
        var = UnitAwareSimpleVariable("test", 42, units="m")

        assert hasattr(var, "has_units")
        assert var.has_units
        assert str(var.units) == "meter"
        assert var.name == "test"
        assert var.value == 42


class TestIntegrationWithExistingCode:
    """Test integration with existing Underworld3 functionality."""

    @pytest.fixture
    def mesh(self):
        """Create a test mesh."""
        return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    def test_sympy_integration(self, mesh):
        """Test integration with SymPy mathematical operations."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")
        x, y = mesh.X

        # Component access in expressions
        v_x = velocity[0]
        v_y = velocity[1]

        # SymPy operations
        expr = v_x * x + v_y * y
        assert expr is not None

        # Differentiation
        div_expr = v_x.diff(x) + v_y.diff(y)
        assert div_expr is not None

    def test_array_access_with_units(self, mesh):
        """Test that array access works with units variables."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Array should be accessible
        assert velocity.array is not None
        assert velocity.array.shape[1:] == (1, 2)  # Vector field

        # Data property should work
        assert velocity.data is not None
        assert velocity.data.shape[1] == 2  # 2 components

    def test_non_dimensional_value_access(self, mesh):
        """Test non-dimensional value extraction."""
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/s")

        # Set some test data
        with uw.synchronised_array_update():
            velocity.array[...] = np.random.random(velocity.array.shape)

        # Get non-dimensional values
        nondim_values = velocity.non_dimensional_value()

        assert nondim_values is not None
        assert nondim_values.shape == velocity.data.shape


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_backend(self):
        """Test error handling for invalid backend."""

        class MockVar(UnitAwareMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        with pytest.raises(ValueError, match="Unknown backend"):
            MockVar(units="m", units_backend="invalid_backend")

    def test_units_without_backend_dependencies(self):
        """Test behavior when backend dependencies are missing."""
        # This would test what happens if pint or sympy are not available
        # For now, we assume they are available in the test environment
        pass


# Integration test that exercises the full system
class TestFullSystemIntegration:
    """Integration test for the complete units system."""

    def test_geophysical_modeling_example(self):
        """Test a realistic geophysical modeling scenario."""
        # Create mesh
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

        # Create variables with appropriate units
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, units="m/year")
        pressure = uw.create_enhanced_mesh_variable("pressure", mesh, 1, units="Pa")
        temperature = uw.create_enhanced_mesh_variable("temperature", mesh, 1, units="K")
        density = uw.create_enhanced_mesh_variable("density", mesh, 1, units="kg/m^3")

        # Check all variables have correct units
        assert str(velocity.units) == "meter / year"
        assert str(pressure.units) == "pascal"
        assert str(temperature.units) == "kelvin"
        assert str(density.units) == "kilogram / meter ** 3"

        # Mathematical operations that make physical sense
        x, y = mesh.X

        # Velocity divergence (1/time units)
        div_velocity = velocity[0].diff(x) + velocity[1].diff(y)
        assert div_velocity is not None

        # Momentum equation terms
        momentum_density = density * velocity  # kg/(m²⋅year)
        assert momentum_density is not None

        # Compatible arithmetic
        velocity_sum = velocity + velocity  # Should work
        assert velocity_sum is not None

        # Incompatible arithmetic should fail
        with pytest.raises(ValueError):
            velocity + pressure  # m/year + Pa should fail

        # Non-dimensional values for solvers
        velocity_nondim = velocity.non_dimensional_value()
        pressure_nondim = pressure.non_dimensional_value()

        assert velocity_nondim is not None
        assert pressure_nondim is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
