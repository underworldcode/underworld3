"""
Regression tests for unit conversion at API boundaries.

These tests verify that user inputs with units are correctly converted to model units,
and that the arithmetic produces physically correct results. This is critical for:
- swarm.advection(delta_t) - time values must match velocity scaling
- coordinate queries - spatial values must match mesh scaling
- Any other direct arithmetic mixing user inputs with model quantities

Test numbering: 0814 (units system advanced tests)
"""

import pytest
import numpy as np
import underworld3 as uw


class TestSwarmAdvectionUnits:
    """Test that swarm advection handles time units correctly."""

    def test_advection_time_conversion_constant_velocity(self):
        """
        Verify delta_t with units gives correct displacement for constant velocity.

        Critical test: velocity is evaluated in model units, so delta_t MUST also
        be in model units for the arithmetic v*dt to be correct.
        """
        # Setup model with clear scaling
        # IMPORTANT: Use get_default_model() not Model() so mesh/swarm register with it
        uw.reset_default_model()  # Start fresh for this test
        model = uw.get_default_model()
        model.set_reference_quantities(
            length_scale=1000 * uw.units.km,
            time_scale=1 * uw.units.Myr,
        )

        # Create simple 2D mesh
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        # Constant velocity field: 1 cm/year in x-direction
        # In model units: need to convert
        v_physical = 1 * uw.units.cm / uw.units.year
        v_model_magnitude = model.to_model_magnitude(v_physical)

        # Create velocity field as constant in x-direction
        v_field = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        with uw.synchronised_array_update():
            v_field.array[:, 0, 0] = v_model_magnitude  # x-component
            v_field.array[:, 0, 1] = 0.0  # y-component

        # Create swarm with one particle at origin
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=1)

        # Get initial position of first particle (in physical units)
        initial_pos_phys = swarm.coords
        if hasattr(initial_pos_phys, "_pint_qty"):
            initial_pos = initial_pos_phys._pint_qty.to("km").magnitude[0]
        else:
            initial_pos = initial_pos_phys[0]

        # Advect for 100 years
        # Expected displacement: 1 cm/year * 100 years = 100 cm = 1 m = 0.001 km
        dt_years = 100 * uw.units.year

        # THIS IS THE CRITICAL TEST: does advection convert dt correctly?
        # If dt is not converted, displacement will be wrong by orders of magnitude
        swarm.advection(v_field.sym, delta_t=dt_years, order=1)

        # Get final position (in physical units)
        final_pos_phys = swarm.coords
        if hasattr(final_pos_phys, "_pint_qty"):
            final_pos = final_pos_phys._pint_qty.to("km").magnitude[0]
        else:
            final_pos = final_pos_phys[0]

        # Calculate displacement (in km, since we extracted km above)
        displacement = final_pos - initial_pos

        # Expected displacement in physical units
        # 1 cm/year * 100 years = 100 cm = 1 m = 0.001 km
        expected_displacement_km = 0.001  # km

        # Check x-displacement is correct (within 5%)
        assert np.isclose(displacement[0], expected_displacement_km, rtol=0.05), (
            f"Advection time conversion failed!\n"
            f"Expected displacement: {expected_displacement_km} km (1 m)\n"
            f"Actual displacement: {displacement[0]} km\n"
            f"This suggests delta_t or velocity is not being converted correctly."
        )

    def test_advection_different_time_units_equivalent(self):
        """
        Verify that different time unit specifications give the same result.

        100 years = 0.0001 Myr = 3.15576e9 seconds should all produce identical displacements.
        """
        # Setup
        uw.reset_default_model()  # Start fresh
        model = uw.get_default_model()
        model.set_reference_quantities(
            length_scale=1000 * uw.units.km,
            time_scale=1 * uw.units.Myr,
        )

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
        )

        # Constant velocity
        v_field = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        v_model = model.to_model_magnitude(1 * uw.units.cm / uw.units.year)
        with uw.synchronised_array_update():
            v_field.array[:, 0, 0] = v_model
            v_field.array[:, 0, 1] = 0.0

        # Test three equivalent time specifications
        time_specs = [
            100 * uw.units.year,
            0.0001 * uw.units.Myr,  # 100 years = 0.0001 million years
            3.15576e9 * uw.units.second,  # 100 years in seconds
        ]

        displacements = []
        for dt_spec in time_specs:
            # Create fresh swarm for each test
            swarm = uw.swarm.Swarm(mesh)
            swarm.populate(fill_param=1)
            # Use internal model-unit coordinates for comparison
            initial_pos = swarm._particle_coordinates.data[0].copy()

            # Advect
            swarm.advection(v_field.sym, delta_t=dt_spec, order=1)

            final_pos = swarm._particle_coordinates.data[0].copy()
            displacement = final_pos[0] - initial_pos[0]
            displacements.append(displacement)

        # All three should give the same result (within 1%)
        assert np.allclose(displacements, displacements[0], rtol=0.01), (
            f"Different time unit specifications gave different results!\n"
            f"Years: {displacements[0]:.2e}\n"
            f"Myr: {displacements[1]:.2e}\n"
            f"Seconds: {displacements[2]:.2e}\n"
            f"This suggests inconsistent time unit conversion."
        )


class TestCoordinateQueryUnits:
    """Test that mesh coordinate queries handle units correctly."""

    @pytest.mark.skip(reason="get_closest_point_on_local_boundary_faces doesn't exist")
    def test_closest_point_with_units(self):
        """Verify coordinate queries work with unit-aware inputs."""
        uw.reset_default_model()
        model = uw.get_default_model()
        model.set_reference_quantities(length_scale=1000 * uw.units.km)

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1
        )

        # Query with units: 500 km = 0.5 model units
        # Convert to model units then create numpy array
        query_x = model.to_model_magnitude(500 * uw.units.km)
        query_y = model.to_model_magnitude(500 * uw.units.km)
        query_point = np.array([[query_x, query_y]])

        # Should work without error
        try:
            closest_point = mesh.get_closest_point_on_local_boundary_faces(query_point)
            # If we get here, query works with converted coordinates
            assert closest_point is not None
        except Exception as e:
            pytest.fail(f"Coordinate query failed: {e}")

    def test_coordinate_query_units_vs_no_units_equivalent(self):
        """Verify queries with units match equivalent model-unit queries."""
        uw.reset_default_model()
        model = uw.get_default_model()
        model.set_reference_quantities(length_scale=1000 * uw.units.km)

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1
        )

        # Same query two ways - convert units before creating array
        query_x = model.to_model_magnitude(500 * uw.units.km)
        query_y = model.to_model_magnitude(500 * uw.units.km)
        query_with_units = np.array([[query_x, query_y]])
        query_model_units = np.array([[0.5, 0.5]])  # 500 km / 1000 km = 0.5

        result1 = mesh.get_closest_local_cells(query_with_units)
        result2 = mesh.get_closest_local_cells(query_model_units)

        # Should give same cell IDs
        assert np.array_equal(result1, result2), (
            f"Query with units gave different result than model units!\n"
            f"With units: {result1}\n"
            f"Model units: {result2}"
        )


class TestSwarmPointsUnits:
    """Test swarm.points getter/setter with units (deprecated but still used)."""

    @pytest.mark.skip(reason="swarm.points is deprecated; use swarm.coords instead")
    def test_points_setter_getter_roundtrip(self):
        """Verify setting points with units and getting them back is consistent."""
        uw.reset_default_model()
        model = uw.get_default_model()
        model.set_reference_quantities(length_scale=1000 * uw.units.km)

        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
        )

        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=3)

        # Get initial positions
        initial_positions = swarm.points.copy()

        # Modify and set back (should be no-op for roundtrip test)
        swarm.points = initial_positions

        # Get back
        recovered_positions = swarm.points.copy()

        # Should be identical
        assert np.allclose(initial_positions, recovered_positions, rtol=1e-10), (
            f"Roundtrip through swarm.points changed values!\n"
            f"Max difference: {np.max(np.abs(initial_positions - recovered_positions))}"
        )


class TestNoUnitsBackwardCompatibility:
    """Ensure code without units continues to work (critical regression test)."""

    def test_advection_without_units_still_works(self):
        """Verify advection without units works as before (backward compatibility)."""
        # No model, no units - old style code
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
        )

        v_field = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
        with uw.synchronised_array_update():
            v_field.array[:, 0, 0] = 0.01  # Some velocity
            v_field.array[:, 0, 1] = 0.0

        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)

        # Advect with plain number - should still work
        try:
            swarm.advection(v_field.sym, delta_t=0.1, order=1)
            # Success - backward compatibility maintained
        except Exception as e:
            pytest.fail(f"Advection without units failed: {e}")

    def test_mesh_queries_without_units_still_work(self):
        """Verify mesh queries without units work as before."""
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1
        )

        # Query with plain coordinates
        query = np.array([[0.5, 0.5]])

        try:
            cells = mesh.get_closest_local_cells(query)
            assert cells is not None
        except Exception as e:
            pytest.fail(f"Mesh query without units failed: {e}")


class TestMixedReferenceScales:
    """Test that conversions work correctly with different model scaling choices."""

    def test_different_length_scales_give_correct_results(self):
        """Verify physical results are independent of chosen length scale."""
        # Test with two different length scales
        scales = [1000 * uw.units.km, 6371 * uw.units.km]  # Arbitrary vs. Earth radius

        displacements_physical = []

        for length_scale in scales:
            uw.reset_default_model()  # Reset for each iteration
            model = uw.get_default_model()
            model.set_reference_quantities(
                length_scale=length_scale,
                time_scale=1 * uw.units.Myr,
            )

            mesh = uw.meshing.UnstructuredSimplexBox(
                minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
            )

            # Same physical velocity: 1 cm/year
            v_physical = 1 * uw.units.cm / uw.units.year
            v_model = model.to_model_magnitude(v_physical)

            v_field = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
            with uw.synchronised_array_update():
                v_field.array[:, 0, 0] = v_model
                v_field.array[:, 0, 1] = 0.0

            swarm = uw.swarm.Swarm(mesh)
            swarm.populate(fill_param=1)
            # Use internal model-unit coordinates
            initial_pos = swarm._particle_coordinates.data[0].copy()

            # Same physical time: 100 years
            swarm.advection(v_field.sym, delta_t=100 * uw.units.year, order=1)

            final_pos = swarm._particle_coordinates.data[0].copy()

            # Displacement in model units
            displacement_model = final_pos[0] - initial_pos[0]

            # Convert to physical units (meters) for comparison
            displacement_physical_qty = model.from_model_magnitude(displacement_model, "[length]")
            if hasattr(displacement_physical_qty, "_pint_qty"):
                displacement_physical = displacement_physical_qty._pint_qty.to("m").magnitude
            else:
                displacement_physical = displacement_physical_qty

            displacements_physical.append(displacement_physical)

        # Both should give same physical displacement: ~1 m (within 5%)
        expected_displacement_m = 1.0  # 1 cm/year * 100 years

        for i, disp in enumerate(displacements_physical):
            assert np.isclose(disp, expected_displacement_m, rtol=0.05), (
                f"Length scale {scales[i]} gave wrong physical displacement!\n"
                f"Expected: {expected_displacement_m} m\n"
                f"Got: {disp} m\n"
                f"Model scaling should not affect physical results."
            )

        # And they should match each other (within 1%)
        assert np.isclose(displacements_physical[0], displacements_physical[1], rtol=0.01), (
            f"Different length scales gave different physical results!\n"
            f"Scale 1: {displacements_physical[0]} m\n"
            f"Scale 2: {displacements_physical[1]} m"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
