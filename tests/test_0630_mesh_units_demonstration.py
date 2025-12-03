#!/usr/bin/env python3
"""
Demonstration of proposed mesh units interface with realistic examples.

This shows how the mesh units interface would work in practice for common
geophysical and scientific modeling scenarios. These examples document the
expected workflow and user experience.
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import numpy as np
import sys
import os

# Add src to path for testing
# REMOVED: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw


class TestMeshUnitsRealisticUseCases:
    """Realistic use cases showing the value of mesh units."""

    @pytest.mark.skip(reason="Demonstrates proposed interface - not yet implemented")
    def test_mantle_convection_workflow(self):
        """Complete mantle convection modeling workflow with units."""

        # REALISTIC WORKFLOW: Mantle convection model
        print("=== Mantle Convection Model with Units ===")

        # 1. Create physical mesh with known scale
        mantle_mesh = uw.meshing.StructuredQuadBox(
            elementRes=(64, 64),
            minCoords=(0.0, 0.0),
            maxCoords=(2900.0, 2900.0),  # Mantle depth/width
            units="km",  # PROPOSED: Explicit units
        )

        print(f"Mesh scale: {mantle_mesh.X.coords.max()} {mantle_mesh.units}")
        print(f"Cell size: ~{2900.0/64:.1f} km")

        # 2. Create model with reference quantities
        model = uw.Model("mantle_convection")
        model.set_reference_quantities(
            mantle_depth=2900 * uw.units.km,  # Matches mesh
            mantle_viscosity=1e21 * uw.units.Pa * uw.units.s,
            plate_velocity=5 * uw.units.cm / uw.units.year,
        )

        # 3. Convert mesh to model units for numerics
        mesh_model = model.to_model_units(mantle_mesh.X.coords)
        print(f"Mesh in model units: max = {mesh_model.max():.2f}")

        # 4. Create variables with physical units
        temperature = uw.discretisation.MeshVariable("T", mantle_mesh, 1, units="K")
        velocity = uw.discretisation.MeshVariable("v", mantle_mesh, 2, units="mm/year")

        # 5. Physical parameters with correct units
        gravity = 9.81 * uw.units.m / uw.units.s**2
        thermal_expansion = 3e-5 / uw.units.K

        # 6. Gradient calculations with automatic unit derivation
        temperature_gradient = temperature.gradient()
        # Units should be K/km automatically derived from T units and mesh units

        print(f"Temperature gradient units: {temperature_gradient.units}")
        # Expected: kelvin/kilometer

        # 7. Dimensionless numbers with clear physical meaning
        rayleigh_number = (
            thermal_expansion
            * gravity
            * model.to_model_units(mantle_mesh.units) ** 3
            * (1600 - 300)
            * uw.units.K
            / (1e21 * uw.units.Pa * uw.units.s * 1e-6 * uw.units.m**2 / uw.units.s)
        )

        print(f"Rayleigh number: {rayleigh_number:.2e}")

        # Benefits demonstrated:
        # - Clear physical scale understanding
        # - Automatic unit derivation for derivatives
        # - Integration with model scaling
        # - Physical interpretation of results

    @pytest.mark.skip(reason="Demonstrates proposed interface - not yet implemented")
    def test_geological_survey_data_workflow(self):
        """Geological survey data import with proper units."""

        print("=== Geological Survey Data Workflow ===")

        # REALISTIC WORKFLOW: Processing field survey data

        # 1. Survey data comes with coordinates in various units
        survey_points_utm = np.array(
            [
                [500000.0, 4000000.0],  # UTM coordinates in meters
                [500100.0, 4000100.0],
                [500200.0, 4000200.0],
            ]
        )

        # 2. Create mesh from survey area with explicit units
        survey_mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(499900.0, 3999900.0),
            maxCoords=(500300.0, 4000300.0),
            cellSize=50.0,  # 50 meter cells
            units="m",  # UTM coordinates in meters
        )

        print(
            f"Survey mesh extent: {survey_mesh.X.coords.min():.0f} to {survey_mesh.X.coords.max():.0f} {survey_mesh.units}"
        )

        # 3. Convert to different coordinate systems as needed
        survey_mesh_km = survey_mesh.to("km")
        print(
            f"Same extent in km: {survey_mesh_km.X.coords.min():.3f} to {survey_mesh_km.X.coords.max():.3f} km"
        )

        # 4. Create geological variables with appropriate units
        elevation = uw.discretisation.MeshVariable("elevation", survey_mesh, 1, units="m")
        rock_density = uw.discretisation.MeshVariable("density", survey_mesh, 1, units="kg/m^3")

        # 5. Calculate gradients with correct units
        slope = elevation.gradient()  # Units: m/m = dimensionless
        density_gradient = rock_density.gradient()  # Units: (kg/m^3)/m = kg/m^4

        print(f"Slope units: {slope.units}")  # dimensionless
        print(f"Density gradient units: {density_gradient.units}")  # kg/m^4

        # 6. Export data with preserved units for GIS software
        # survey_mesh.export_to_shapefile("survey_results.shp")  # Units preserved

        # Benefits demonstrated:
        # - Import real-world coordinate data with proper units
        # - Convert between coordinate systems
        # - Calculate physical gradients with correct units
        # - Export results with unit metadata

    @pytest.mark.skip(reason="Demonstrates proposed interface - not yet implemented")
    def test_multi_scale_modeling_workflow(self):
        """Multi-scale modeling with consistent units."""

        print("=== Multi-Scale Modeling Workflow ===")

        # REALISTIC WORKFLOW: Regional to local scale modeling

        # 1. Regional model: tectonic scale
        regional_mesh = uw.meshing.StructuredQuadBox(
            elementRes=(32, 32),
            minCoords=(0.0, 0.0),
            maxCoords=(1000.0, 1000.0),  # 1000 km region
            units="km",
        )

        # 2. Local model: outcrop scale
        local_mesh = uw.meshing.StructuredQuadBox(
            elementRes=(64, 64),
            minCoords=(450.0, 450.0),  # Subset of regional
            maxCoords=(550.0, 550.0),
            units="km",
        )

        print(f"Regional scale: {regional_mesh.X.coords.max():.0f} {regional_mesh.units}")
        print(
            f"Local scale: {local_mesh.X.coords.max() - local_mesh.X.coords.min():.0f} {local_mesh.units}"
        )

        # 3. Detail model: laboratory scale
        detail_mesh = uw.meshing.StructuredQuadBox(
            elementRes=(32, 32),
            minCoords=(0.0, 0.0),
            maxCoords=(10.0, 10.0),  # 10 cm sample
            units="cm",
        )

        # 4. Coordinate transformations between scales
        # Map local region to regional coordinates
        local_in_regional = regional_mesh.to("km")
        detail_in_meters = detail_mesh.to("m")

        print(f"Detail model in meters: {detail_in_meters.X.coords.max():.2f} m")

        # 5. Variables with scale-appropriate units
        regional_velocity = uw.discretisation.MeshVariable(
            "v_regional", regional_mesh, 2, units="cm/year"
        )
        local_stress = uw.discretisation.MeshVariable("stress_local", local_mesh, 1, units="MPa")
        detail_strain = uw.discretisation.MeshVariable("strain_detail", detail_mesh, 1, units="1")

        # 6. Cross-scale calculations
        # Convert regional velocity to local mesh resolution
        regional_vel_local = regional_velocity.interpolate_to_mesh(local_mesh)
        # Units automatically converted: cm/year remains cm/year

        # Benefits demonstrated:
        # - Consistent units across multiple scales
        # - Automatic coordinate transformations
        # - Scale-appropriate variable units
        # - Cross-scale interpolation with unit preservation

    @pytest.mark.skip(reason="Demonstrates proposed interface - not yet implemented")
    def test_data_import_export_with_units(self):
        """Data import/export preserving unit information."""

        print("=== Data Import/Export with Units ===")

        # REALISTIC WORKFLOW: Scientific data exchange

        # 1. Create mesh with specific units
        experiment_mesh = uw.meshing.StructuredQuadBox(
            elementRes=(16, 16),
            minCoords=(0.0, 0.0),
            maxCoords=(50.0, 50.0),  # 50 cm experimental setup
            units="cm",
        )

        # 2. Create experimental data
        temperature = uw.discretisation.MeshVariable("T", experiment_mesh, 1, units="°C")
        pressure = uw.discretisation.MeshVariable("P", experiment_mesh, 1, units="bar")

        # 3. Export with unit metadata
        # This should preserve both coordinate units and variable units
        export_data = {
            "coordinates": experiment_mesh.X.coords,  # UWQuantity with cm units
            "coordinate_units": str(experiment_mesh.units),
            "temperature": temperature.data,
            "temperature_units": str(temperature.units),
            "pressure": pressure.data,
            "pressure_units": str(pressure.units),
        }

        print(f"Export data coordinate units: {export_data['coordinate_units']}")
        print(f"Export data temperature units: {export_data['temperature_units']}")

        # 4. Import to different mesh with unit conversion
        analysis_mesh = uw.meshing.StructuredQuadBox(
            elementRes=(32, 32),
            minCoords=(0.0, 0.0),
            maxCoords=(0.5, 0.5),  # Same physical size in meters
            units="m",
        )

        # Coordinates should be automatically converted: 50 cm = 0.5 m
        assert np.isclose(analysis_mesh.X.coords.max(), 0.5)

        # Benefits demonstrated:
        # - Unit metadata preservation in data files
        # - Automatic unit conversion during import/export
        # - Scientific data exchange with proper units
        # - Consistent interpretation across different software

    def test_current_state_documentation(self):
        """Document current state and limitations."""

        print("=== Current State: Unit Information Gap ===")

        # Current workflow - units are implicit and error-prone
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4),
            minCoords=(0.0, 0.0),
            maxCoords=(2900.0, 2900.0),  # What units? Unknown!
        )

        # Users must track units manually
        mesh_units = "km"  # Manual tracking - error prone
        coords = mesh.X.coords  # Raw numpy array, no unit info

        print(f"Current mesh coordinates: {coords.min():.0f} to {coords.max():.0f}")
        print(f"Units (manual tracking): {mesh_units}")
        print(f"Physical extent: {coords.max():.0f} {mesh_units}")

        # Problems with current approach:
        problems = [
            "No unit validation - easy to mix km and m",
            "Units lost in data processing chains",
            "Difficult to interpret coordinate scales",
            "Manual unit tracking is error-prone",
            "No automatic unit conversion",
            "Gradients have unclear units",
            "Data export loses unit information",
            "Visualization has ambiguous scales",
        ]

        print("\nProblems with current approach:")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")

        # This test always passes - it's documentation
        assert True


if __name__ == "__main__":
    # Run the documentation test to show current limitations
    test = TestMeshUnitsRealisticUseCases()
    test.test_current_state_documentation()

    print("\n" + "=" * 60)
    print("NOTE: Workflow tests are skipped - they demonstrate")
    print("the proposed interface that is not yet implemented.")
    print("Run with pytest to see all test cases.")
