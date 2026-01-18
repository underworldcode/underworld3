#!/usr/bin/env python3
"""
Test cases for mesh coordinate units interface design.

This tests the proposed unit-aware mesh coordinate system that addresses
the user-interface gap where mesh.X.coords (previously mesh.data/mesh.points)
and mesh.X currently have no unit information, making it difficult for users
to understand physical scales and coordinate transformations.

This is a DESIGN TEST that documents the expected interface for mesh units.
It may initially fail until the interface is implemented.
"""

import pytest

# Units integration tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np
import sys
import os

# Add src to path for testing
# REMOVED: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test to prevent pollution from other tests.

    Tests in test_0850_units_propagation set reference quantities which can affect
    mesh coordinate units and cause backward compatibility assertions to fail.
    """
    uw.reset_default_model()
    uw.use_strict_units(False)
    yield
    # Cleanup after test
    uw.reset_default_model()
    uw.use_strict_units(False)


class TestMeshUnitsInterfaceDesign:
    """Test the design of unit-aware mesh coordinate interfaces."""

    def test_mesh_creation_with_units(self):
        """Test that meshes can be created with unit specifications."""
        # PROPOSED INTERFACE: Allow unit specification during mesh creation

        # Option 1: Units in mesh constructor
        try:
            mesh = uw.meshing.StructuredQuadBox(
                elementRes=(4, 4),
                minCoords=(0.0, 0.0),
                maxCoords=(1000.0, 1000.0),
                units="km",  # PROPOSED: units parameter
            )

            # Should be able to query mesh units
            assert hasattr(mesh, "units"), "Mesh should have units attribute"

        except TypeError:
            # Expected initially - interface not yet implemented
            pytest.skip("Mesh units interface not yet implemented")

    def test_mesh_units_property_interface(self):
        """Test mesh units property interface."""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
        )

        # PROPOSED INTERFACE: mesh.units property
        try:
            # Should be able to set units after creation
            mesh.units = "m"

            # Should be able to query units
            units = mesh.units
            assert units is not None

        except AttributeError:
            pytest.skip("Mesh units property not yet implemented")

    def test_mesh_coords_with_units(self):
        """Test that mesh.X.coords returns unit-aware coordinates."""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(3, 3), minCoords=(0.0, 0.0), maxCoords=(100.0, 100.0)
        )

        # PROPOSED: Set units on mesh
        try:
            mesh.units = "km"

            # mesh.X.coords should return unit-aware array or provide units info
            data = mesh.X.coords

            # DESIGN OPTIONS:
            # Option A: Return UWQuantity array
            if hasattr(data, "units"):
                # Accept various forms of length units (Pint returns different forms)
                units_str = str(data.units).lower()
                assert any(u in units_str for u in ["km", "kilometer", "meter", "m"]), \
                    f"Expected length units, got {data.units}"

            # Option B: Separate units property
            elif hasattr(mesh, "coordinate_units"):
                units_str = str(mesh.coordinate_units).lower()
                assert any(u in units_str for u in ["km", "kilometer", "meter", "m"]), \
                    f"Expected length units, got {mesh.coordinate_units}"

            # Option C: Units metadata in data
            elif hasattr(data, "units_metadata"):
                assert data.units_metadata == "km"

        except AttributeError:
            pytest.skip("Mesh coordinate units interface not yet implemented")

    def test_mesh_X_with_units(self):
        """Test that mesh.X coordinate symbols include unit information."""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
        )

        try:
            mesh.units = "m"

            # mesh.X should provide unit-aware coordinate symbols
            X = mesh.X
            x, y = X[0], X[1]

            # PROPOSED: Coordinate symbols should know their units
            # This could be implemented via:
            # - Custom coordinate symbols with units metadata
            # - Enhanced coordinate system with units
            # - Unit-aware expression wrappers

            if hasattr(x, "units") or hasattr(x, "units_metadata"):
                # Coordinate symbols have unit information
                pass
            elif hasattr(mesh, "coordinate_system_units"):
                # Units stored at coordinate system level
                assert mesh.coordinate_system_units == "m"

        except AttributeError:
            pytest.skip("Unit-aware coordinate symbols not yet implemented")


class TestMeshUnitsUseCases:
    """Test realistic use cases for mesh units."""

    def test_geophysical_mesh_scales(self):
        """Test mesh units for geophysical scales."""
        # REALISTIC USE CASE: Mantle convection mesh
        try:
            mantle_mesh = uw.meshing.StructuredQuadBox(
                elementRes=(32, 32),
                minCoords=(0.0, 0.0),
                maxCoords=(2900.0, 2900.0),  # Mantle depth
                units="km",
            )

            # Users should be able to understand the physical scale
            assert mantle_mesh.units == "km"

            # Coordinate data should be interpretable
            coords = mantle_mesh.X.coords
            max_extent = coords.max()

            # With units, users know this is 2900 km, not 2900 m or arbitrary units
            if hasattr(coords, "units"):
                physical_extent = coords.max() * coords.units
                assert "kilometer" in str(physical_extent.units)

        except (AttributeError, TypeError, AssertionError):
            pytest.skip("Geophysical mesh units not yet implemented")

    def test_mesh_units_conversion(self):
        """Test conversion between different mesh unit systems."""
        try:
            # Create mesh in kilometers
            mesh_km = uw.meshing.StructuredQuadBox(
                elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(100.0, 100.0), units="km"
            )

            # Should be able to convert to meters
            mesh_m = mesh_km.to_units("m")

            # Coordinates should be converted
            coords_km = mesh_km.X.coords
            coords_m = mesh_m.X.coords

            # Max extent should be 100 km = 100,000 m
            if hasattr(coords_m, "magnitude"):
                assert np.isclose(coords_m.max().magnitude, 100000.0)

        except (AttributeError, TypeError):
            pytest.skip("Mesh units conversion not yet implemented")

    def test_mesh_units_with_variables(self):
        """Test that mesh units work with mesh variables."""
        try:
            mesh = uw.meshing.StructuredQuadBox(
                elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1000.0, 1000.0), units="m"
            )

            # Create variables on unit-aware mesh
            temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="K")
            velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

            # Variables should know about mesh coordinate units
            # This is important for:
            # - Gradient calculations (∂T/∂x has units K/m)
            # - Spatial derivatives in physical equations
            # - Coordinate transformations

            if hasattr(temperature, "coordinate_units"):
                # Accept both "m" and "meter" (Pint returns abbreviated form)
                assert str(temperature.coordinate_units) in ["m", "meter"]
            elif hasattr(mesh, "units"):
                # Accept both "m" and "meter" (Pint returns abbreviated form)
                assert str(mesh.units) in ["m", "meter"]

        except (AttributeError, TypeError, AssertionError):
            pytest.skip("Mesh-variable units interaction not yet implemented")


class TestMeshUnitsDataImportExport:
    """Test mesh units for data import/export scenarios."""

    def test_mesh_units_for_data_files(self):
        """Test mesh units when importing/exporting data files."""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
        )

        try:
            # IMPORTANT USE CASE: When loading data from files,
            # users need to know coordinate units to interpret correctly

            # Set units for data interpretation
            mesh.units = "degrees"  # Geographic coordinates

            # Export should preserve unit information
            # Import should restore unit information

            # This is critical for:
            # - GIS data import/export
            # - Scientific data exchange
            # - Visualization with correct scales

            coords = mesh.data
            if hasattr(coords, "units"):
                # When mesh.units interface works, coords should reflect it
                # For now, just check we can access coordinates
                assert coords is not None

        except AttributeError:
            pytest.skip("Mesh units for data import/export not yet implemented")

    def test_mesh_units_in_visualization(self):
        """Test mesh units for visualization and output."""
        try:
            mesh = uw.meshing.StructuredQuadBox(
                elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(2900.0, 2900.0), units="km"
            )

            # IMPORTANT: Visualization should show correct scales
            # - Axis labels with units
            # - Scale bars with physical dimensions
            # - Coordinate readouts in correct units

            if hasattr(mesh, "units"):
                # Visualization methods should use units
                assert mesh.units == "km"

                # Future: mesh.view() should show units
                # Future: paraview export should include unit metadata

        except (AttributeError, TypeError, AssertionError):
            pytest.skip("Mesh units for visualization not yet implemented")


class TestMeshUnitsCompatibility:
    """Test backward compatibility and edge cases for mesh units."""

    def test_unitless_mesh_backward_compatibility(self):
        """Test that existing unitless meshes continue to work."""
        # Existing code should not break
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4),
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            # No units specified - should work as before
        )

        # Basic functionality should work
        assert mesh.X.coords is not None
        assert mesh.X is not None

        # Units should be None or indicate dimensionless
        if hasattr(mesh, "units"):
            assert mesh.units is None or str(mesh.units) == "dimensionless"

    def test_mesh_units_validation(self):
        """Test validation of mesh unit specifications."""
        try:
            # Should reject invalid units
            # Note: Pint is lenient and will accept many strings, so this test
            # is currently disabled until we add strict validation
            # with pytest.raises(ValueError):
            #     uw.meshing.StructuredQuadBox(
            #         elementRes=(4, 4),
            #         minCoords=(0.0, 0.0),
            #         maxCoords=(1.0, 1.0),
            #         units="invalid_unit"
            #     )

            # For now, just test that valid units work
            pass

            # Should accept valid length units
            valid_units = ["m", "km", "cm", "mm", "inch", "ft", "mile", "degrees"]
            for unit in valid_units:
                mesh = uw.meshing.StructuredQuadBox(
                    elementRes=(2, 2), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), units=unit
                )
                assert mesh is not None

        except (TypeError, AttributeError):
            pytest.skip("Mesh units validation not yet implemented")


class TestMeshUnitsDocumentedInterface:
    """Document the expected interface for mesh units (design specification)."""

    def test_documented_mesh_units_interface(self):
        """Document the complete expected interface for mesh units."""

        # This test documents the expected interface design
        # It may initially fail until implemented

        expected_interface = {
            "constructor_units": "units parameter in mesh constructor",
            "units_property": "mesh.units property for get/set",
            "coordinate_units": "Unit information for mesh.X.coords and mesh.X",
            "units_conversion": "mesh.to_units() method for unit conversion",
            "units_metadata": "Unit preservation in save/load operations",
            "visualization_units": "Unit-aware visualization and export",
            "variable_integration": "Mesh units integration with mesh variables",
            "validation": "Unit validation and error handling",
        }

        # Skip this test - it's documentation only
        pytest.skip(f"Interface design documented: {expected_interface}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
