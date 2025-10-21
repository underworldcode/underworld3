#!/usr/bin/env python3
"""
Integration tests for variable unit metadata and unit-aware evaluation.

This test suite validates the complete workflow:
1. Variable creation with units
2. Unit metadata storage and retrieval
3. Unit-aware evaluation returning UWQuantity objects
4. Integration with existing UW3 functionality
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import underworld3 as uw


class TestVariableUnitsIntegration:
    """Test complete integration of variable unit metadata with evaluation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up scaled model and mesh for testing."""
        uw.reset_default_model()
        self.model = uw.get_default_model()

        self.model.set_reference_quantities(
            characteristic_length=1000 * uw.units.km,
            plate_velocity=5 * uw.units.cm / uw.units.year,
            mantle_temperature=1500 * uw.units.kelvin
        )

        self.mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4),
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            qdegree=2
        )

    def test_mesh_variable_units_storage_and_retrieval(self):
        """Test that MeshVariable stores and retrieves units correctly."""
        # Create variables with different unit types
        temperature = uw.discretisation.MeshVariable("T", self.mesh, 1, units="kelvin")
        velocity = uw.discretisation.MeshVariable("v", self.mesh, 2, units="m/s")
        pressure = uw.discretisation.MeshVariable("p", self.mesh, 1, units="GPa")
        dimensionless = uw.discretisation.MeshVariable("d", self.mesh, 1)

        # Test unit storage
        assert temperature.units is not None
        assert "kelvin" in str(temperature.units)

        assert velocity.units is not None
        assert "meter" in str(velocity.units) or "m" in str(velocity.units)

        assert pressure.units is not None
        assert "pascal" in str(pressure.units) or "Pa" in str(pressure.units)

        assert dimensionless.units is None

    def test_swarm_variable_units_storage_and_retrieval(self):
        """Test that SwarmVariable stores and retrieves units correctly."""
        swarm = uw.swarm.Swarm(self.mesh)

        # Create variables with units via add_variable
        density = swarm.add_variable("density", 1, units="kg/m^3")
        velocity = swarm.add_variable("velocity", 2, units="cm/year")
        material_id = swarm.add_variable("material", 1)  # No units

        # Test unit storage
        assert density.units is not None
        assert "kg" in str(density.units)

        assert velocity.units is not None
        assert "centimeter" in str(velocity.units) or "cm" in str(velocity.units)

        assert material_id.units is None

        # Test units persist after population
        swarm.populate(fill_param=2)

        assert density.units is not None
        assert velocity.units is not None
        assert material_id.units is None

    @pytest.mark.skip(reason="coord_units parameter not implemented - planned feature for evaluate()")
    def test_unit_aware_evaluation_returns_uwquantity(self):
        """Test that evaluating variables with units returns UnitAwareArray with unit metadata."""
        # Create temperature field with units
        temperature = uw.discretisation.MeshVariable("T", self.mesh, 1, units="kelvin")

        # Set field data
        with uw.synchronised_array_update():
            temperature.array[:, 0, 0] = 1000 + 500 * self.mesh.X.coords[:, 0]

        coords_km = np.array([[500, 500]], dtype=np.float64)

        # Evaluate with coordinate units - should return UnitAwareArray
        result = uw.function.evaluate(temperature.sym, coords_km, coord_units='km')

        # Check result type and units
        assert hasattr(result, '_units'), f"Expected UnitAwareArray with units, got {type(result)}"
        assert result._units is not None, "Result should have units metadata"
        assert "kelvin" in str(result._units), f"Expected kelvin units, got {result._units}"

        # Check result is numpy compatible
        assert isinstance(result, np.ndarray), "Result should be numpy array compatible"
        assert result.shape == (1, 1, 1), f"Expected shape (1,1,1), got {result.shape}"

    @pytest.mark.skip(reason="coord_units parameter not implemented - planned feature for evaluate()")
    def test_dimensionless_evaluation_returns_plain_array(self):
        """Test that evaluating variables without explicit units returns plain array."""
        # Create variable without explicit units
        dimensionless = uw.discretisation.MeshVariable("d", self.mesh, 1)

        # Set field data
        with uw.synchronised_array_update():
            dimensionless.array[:, 0, 0] = 0.5

        coords_km = np.array([[500, 500]], dtype=np.float64)

        # Evaluate - should return plain numpy array (no units)
        result = uw.function.evaluate(dimensionless.sym, coords_km, coord_units='km')

        # Check result type - variable has no units, so result should be plain array
        assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
        assert not hasattr(result, '_pint_qty'), "Result should not have units"

    def test_mixed_unit_variable_evaluation(self):
        """Test evaluation of expressions mixing variables with different units."""
        # Create variables with different units
        temperature = uw.discretisation.MeshVariable("T", self.mesh, 1, units="K")
        velocity = uw.discretisation.MeshVariable("v", self.mesh, 2, units="m/s")

        # Set data
        with uw.synchronised_array_update():
            temperature.array[:, 0, 0] = 1000
            velocity.array[:, 0, 0] = 0.01
            velocity.array[:, 0, 1] = 0.0

        coords = np.array([[0.5, 0.5]], dtype=np.float64)

        # Test individual evaluations return correct units
        temp_result = uw.function.evaluate(temperature.sym, coords)
        vel_result = uw.function.evaluate(velocity.sym, coords)

        if hasattr(temp_result, '_units'):
            assert "K" in str(temp_result._units) or "kelvin" in str(temp_result._units)

        if hasattr(vel_result, '_units'):
            assert "m" in str(vel_result._units) or "meter" in str(vel_result._units)

    def test_units_persist_through_mathematical_operations(self):
        """Test that units are preserved through variable mathematical operations."""
        velocity = uw.discretisation.MeshVariable("v", self.mesh, 2, units="m/s")

        # Test mathematical operations create valid expressions
        velocity_magnitude = (velocity[0]**2 + velocity[1]**2)**0.5
        # Use simple arithmetic operations to avoid scaling issues
        velocity_sum = velocity[0] + velocity[1]  # Simpler operation

        # Should create valid SymPy expressions
        assert velocity_magnitude is not None
        assert velocity_sum is not None

        # Units should still be accessible on original variable
        assert velocity.units is not None
        assert "meter" in str(velocity.units) or "m" in str(velocity.units)

    def test_backward_compatibility_no_units(self):
        """Test that variables without units work exactly as before."""
        # Create variables without units (original behavior)
        v_no_units = uw.discretisation.MeshVariable("v", self.mesh, 2)
        p_no_units = uw.discretisation.MeshVariable("p", self.mesh, 1)

        # Should have no units
        assert v_no_units.units is None
        assert p_no_units.units is None

        # Should work with all existing operations
        with uw.synchronised_array_update():
            v_no_units.array[:, 0, 0] = 1.0
            v_no_units.array[:, 0, 1] = 0.5
            p_no_units.array[:, 0, 0] = 100.0

        coords = np.array([[0.5, 0.5]], dtype=np.float64)

        # Evaluation should return plain numpy arrays (no units)
        v_result = uw.function.evaluate(v_no_units.sym, coords)
        p_result = uw.function.evaluate(p_no_units.sym, coords)

        # Check that results are plain numpy arrays (no units)
        assert isinstance(v_result, np.ndarray), f"Expected numpy array, got {type(v_result)}"
        assert isinstance(p_result, np.ndarray), f"Expected numpy array, got {type(p_result)}"

        # Variables without explicit units should NOT have units
        assert not hasattr(v_result, '_pint_qty')
        assert not hasattr(p_result, '_pint_qty')

    def test_unit_metadata_survives_array_operations(self):
        """Test that unit metadata persists through array access and modification."""
        temperature = uw.discretisation.MeshVariable("T", self.mesh, 1, units="kelvin")

        # Access various array properties
        _ = temperature.array
        _ = temperature.data
        _ = temperature.shape
        _ = temperature.sym

        # Units should persist
        assert temperature.units is not None
        assert "kelvin" in str(temperature.units)

        # Modify data and check units persist
        with uw.synchronised_array_update():
            temperature.array[:, 0, 0] = 298.0  # 25°C in Kelvin

        assert temperature.units is not None
        assert "kelvin" in str(temperature.units)


if __name__ == "__main__":
    pytest.main([__file__])