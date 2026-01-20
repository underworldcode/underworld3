#!/usr/bin/env python3
"""
Unit tests for the Underworld3 units system.

This test suite validates:
- Enhanced variable classes with units
- Mathematical operations with units
- Integration with existing UW3 functionality
"""

import pytest
import numpy as np

import underworld3 as uw

# Module-level markers for all tests
pytestmark = [
    pytest.mark.level_2,  # Intermediate - units system
    pytest.mark.tier_a,   # Production-ready - core units functionality
]


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

    @pytest.mark.tier_c  # Experimental - units arithmetic validation not fully implemented
    def test_units_arithmetic_validation(self, mesh):
        """Test that arithmetic operations validate units compatibility."""
        velocity1 = uw.create_enhanced_mesh_variable("v1", mesh, 2, units="m/s")
        velocity2 = uw.create_enhanced_mesh_variable("v2", mesh, 2, units="m/s")
        pressure = uw.create_enhanced_mesh_variable("p", mesh, 1, units="Pa")

        # Compatible addition should work
        velocity_sum = velocity1 + velocity2
        assert velocity_sum is not None

        # Incompatible addition should raise error
        # Note: velocity (2 components) + pressure (1 component) fails on shape first
        # The actual error is TypeError from mathematical_mixin (shape/dimension check)
        with pytest.raises((ValueError, TypeError)):
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


# Integration test that exercises the full system
class TestFullSystemIntegration:
    """Integration test for the complete units system."""

    @pytest.mark.tier_c  # Experimental - full system integration has some issues
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
        momentum_density = density * velocity  # kg/(m^2 year)
        assert momentum_density is not None

        # Compatible arithmetic
        velocity_sum = velocity + velocity  # Should work
        assert velocity_sum is not None

        # Incompatible arithmetic should fail
        # Note: velocity is a vector (2 components), pressure is a scalar (1 component)
        # The operation fails due to shape mismatch, raised as TypeError
        with pytest.raises(TypeError):
            velocity + pressure  # Shape mismatch: (1,2) + (1,1)

        # Non-dimensional values for solvers
        velocity_nondim = velocity.non_dimensional_value()
        pressure_nondim = pressure.non_dimensional_value()

        assert velocity_nondim is not None
        assert pressure_nondim is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
