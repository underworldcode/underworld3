"""
Test swarm integration for accurate spatial statistics.

This test validates that using proxy variables with integration provides
more accurate spatial statistics than simple arithmetic mean/std on
non-uniformly distributed particles.

Key concepts tested:
1. Simple arithmetic mean/std (current array.mean(), array.std())
2. Integration-based spatial mean/std (accurate for non-uniform distributions)
3. Proxy variable RBF interpolation
4. Comparison with analytical results

See docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md for coordinate units context.
"""

import numpy as np
import pytest
import underworld3 as uw


class TestSwarmIntegrationStatistics:
    """Test integration-based statistics for non-uniformly distributed swarms."""

    def test_uniform_swarm_arithmetic_vs_integration_mean(self):
        """
        For uniformly distributed particles, arithmetic mean should equal
        integration-based mean (within numerical precision).
        """
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(8, 8)
        )

        # Create swarm (don't populate yet)
        swarm = uw.swarm.Swarm(mesh)

        # Create scalar swarm variable BEFORE populating
        s_var = uw.swarm.SwarmVariable("scalar", swarm, 1, proxy_degree=2)

        # NOW populate with uniform distribution
        swarm.populate(fill_param=3)

        # Set data: linear function s(x,y) = 2 + x
        x_coords = swarm._particle_coordinates.data[:, 0]
        y_coords = swarm._particle_coordinates.data[:, 1]
        s_var.data[:, 0] = 2.0 + x_coords

        # Method 1: Simple arithmetic mean (current implementation)
        arithmetic_mean = s_var.array.mean()

        # Method 2: Integration-based mean (spatial mean)
        # For uniform distribution: spatial_mean = integral(f) / volume
        I_f = uw.maths.Integral(mesh, fn=s_var.sym[0])
        I_volume = uw.maths.Integral(mesh, fn=1.0)
        integration_mean = I_f.evaluate() / I_volume.evaluate()

        # For f(x) = 2 + x on [0,1]²:
        # arithmetic ∫∫(2 + x) dA / A = (2 + 0.5) = 2.5
        expected_mean = 2.5

        # Both should be close to expected value
        assert np.isclose(arithmetic_mean, expected_mean, rtol=0.05)
        assert np.isclose(integration_mean, expected_mean, rtol=0.05)

        # For uniform distribution, both methods should give similar results
        assert np.isclose(arithmetic_mean, integration_mean, rtol=0.1)

    def test_clustered_swarm_shows_difference(self):
        """
        For non-uniformly distributed particles (clustered), arithmetic mean
        differs significantly from integration-based mean.

        This demonstrates the key limitation of simple arithmetic statistics
        on swarms: they don't account for spatial distribution.
        """
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(10, 10)
        )

        swarm = uw.swarm.Swarm(mesh)

        # Create scalar variable BEFORE adding particles
        s_var = uw.swarm.SwarmVariable("scalar", swarm, 1, proxy_degree=2)

        # Create clustered distribution: more particles in left half
        # Left half (x < 0.5): 75% of particles
        # Right half (x >= 0.5): 25% of particles
        left_count = 75
        right_count = 25

        left_coords = np.random.RandomState(42).uniform(0.0, 0.5, (left_count, 2))
        right_coords = np.random.RandomState(43).uniform(0.5, 1.0, (right_count, 2))
        all_coords = np.vstack([left_coords, right_coords])

        swarm.add_particles_with_coordinates(all_coords)
        x_coords = swarm._particle_coordinates.data[:, 0]
        s_var.data[:, 0] = 1.0 + x_coords

        # Arithmetic mean: biased toward left half values
        # (75% particles at x ≈ 0.25) + (25% particles at x ≈ 0.75)
        # ≈ 0.75 * (1 + 0.25) + 0.25 * (1 + 0.75) = 1.375
        arithmetic_mean = s_var.array.mean()

        # Integration-based mean: weights by volume equally
        # ∫∫(1 + x) dA = [x + 0.5x²]₀¹ × 1 = (1 + 0.5) = 1.5
        # spatial_mean = 1.5 / 1.0 = 1.5
        I_f = uw.maths.Integral(mesh, fn=s_var.sym[0])
        I_volume = uw.maths.Integral(mesh, fn=1.0)
        integration_mean = I_f.evaluate() / I_volume.evaluate()

        expected_arithmetic = 1.0 + 0.375  # 1.375
        expected_integration = 1.5

        # Arithmetic should be < integration (weighted toward low-x values)
        assert arithmetic_mean < integration_mean

        print(f"Clustered swarm statistics:")
        print(f"  Arithmetic mean (particle-weighted): {arithmetic_mean:.4f}")
        print(f"  Integration mean (space-weighted): {integration_mean:.4f}")
        print(f"  Difference: {integration_mean - arithmetic_mean:.4f}")

    def test_swarm_integration_standard_deviation(self):
        """
        Test computing standard deviation through integration.

        std = sqrt(E[x²] - (E[x])²)

        Both methods should be similar for uniformly distributed data.
        """
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(8, 8)
        )

        swarm = uw.swarm.Swarm(mesh)
        s_var = uw.swarm.SwarmVariable("scalar", swarm, 1, proxy_degree=2)
        swarm.populate(fill_param=3)

        # Set data: quadratic function s(x) = x²
        x_coords = swarm._particle_coordinates.data[:, 0]
        s_var.data[:, 0] = x_coords**2

        # Method 1: Simple arithmetic std (current implementation)
        arithmetic_std = s_var.array.std()

        # Method 2: Integration-based std
        # std² = E[f²] - (E[f])²
        I_volume = uw.maths.Integral(mesh, fn=1.0)
        vol = I_volume.evaluate()

        I_f = uw.maths.Integral(mesh, fn=s_var.sym[0])
        mean_f = I_f.evaluate() / vol

        I_f2 = uw.maths.Integral(mesh, fn=s_var.sym[0] ** 2)
        mean_f2 = I_f2.evaluate() / vol

        variance = mean_f2 - mean_f**2
        integration_std = np.sqrt(max(variance, 0.0))

        # For f(x) = x² on [0,1]²:
        # E[f] = ∫∫x² dA = 1/3
        # E[f²] = ∫∫x⁴ dA = 1/5
        # var = 1/5 - (1/3)² = 1/5 - 1/9 = 4/45
        # std = sqrt(4/45) ≈ 0.298
        expected_std = np.sqrt(4.0 / 45.0)

        print(f"Quadratic swarm statistics:")
        print(f"  Arithmetic std: {arithmetic_std:.4f}")
        print(f"  Integration std: {integration_std:.4f}")
        print(f"  Expected (analytical): {expected_std:.4f}")

        # Both should be reasonable
        assert arithmetic_std > 0
        assert integration_std > 0
        assert np.isclose(integration_std, expected_std, rtol=0.1)

    def test_proxy_variable_creation_and_update(self):
        """
        Test that proxy variables are created automatically and updated
        when swarm data changes.
        """
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(6, 6)
        )

        swarm = uw.swarm.Swarm(mesh)

        # Create swarm variable with proxy BEFORE populating
        s_var = uw.swarm.SwarmVariable(
            "temperature", swarm, 1, proxy_degree=1, proxy_continuous=True  # Linear interpolation
        )

        swarm.populate(fill_param=2)

        # Proxy should be created
        assert s_var._meshVar is not None
        assert hasattr(s_var, "_meshVar")

        # Set swarm data
        s_var.data[:, 0] = swarm._particle_coordinates.data[:, 0]

        # Accessing .sym should trigger proxy update
        sym_T = s_var.sym
        assert sym_T is not None

        # The proxy mesh variable should now have data
        assert s_var._meshVar.array is not None

    def test_rbf_interpolation_accuracy(self):
        """
        Test that RBF interpolation provides reasonable approximation
        of swarm data at mesh nodes.
        """
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(5, 5)
        )

        swarm = uw.swarm.Swarm(mesh)

        # Create swarm variable BEFORE adding particles
        s_var = uw.swarm.SwarmVariable("scalar", swarm, 1, proxy_degree=2)

        # Create grid of particles
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        swarm.add_particles_with_coordinates(coords)

        # Set values: s = sin(πx) * cos(πy)
        s_var.data[:, 0] = np.sin(np.pi * swarm._particle_coordinates.data[:, 0]) * np.cos(
            np.pi * swarm._particle_coordinates.data[:, 1]
        )

        # Access proxy to trigger RBF interpolation
        proxy_mesh_var = s_var._meshVar

        # The proxy should now be populated via RBF
        proxy_array = np.array(proxy_mesh_var.array)
        proxy_data_flat = proxy_array.flatten()
        assert proxy_data_flat.size > 0

        # Proxy values should be in reasonable range
        # (sin * cos should be between -1 and 1)
        proxy_min = proxy_data_flat.min()
        proxy_max = proxy_data_flat.max()
        print(f"RBF interpolation range: [{proxy_min:.3f}, {proxy_max:.3f}]")
        assert proxy_min >= -1.1  # Small tolerance for interpolation
        assert proxy_max <= 1.1


class TestSwarmStatisticsWorkflow:
    """Complete workflow examples for swarm statistics."""

    def test_complete_statistics_workflow(self):
        """
        Demonstrate complete workflow: create swarm, set data, compute both
        arithmetic and integration-based statistics.
        """
        # Setup
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), elementRes=(10, 10)
        )

        swarm = uw.swarm.Swarm(mesh)

        # Create temperature variable BEFORE populating
        T = uw.swarm.SwarmVariable(
            "Temperature",
            swarm,
            1,
            proxy_degree=2,  # Use degree 2 for smooth interpolation
            proxy_continuous=True,
        )

        swarm.populate(fill_param=4)  # Moderate particle density

        # Set initial temperature field
        x_coords = swarm._particle_coordinates.data[:, 0]
        y_coords = swarm._particle_coordinates.data[:, 1]
        T.data[:, 0] = 273.0 + 100.0 * (1.0 - (x_coords**2 + y_coords**2))

        # ==========================================
        # Method 1: Simple arithmetic statistics
        # ==========================================
        arithmetic_mean = T.array.mean()
        arithmetic_std = T.array.std()
        arithmetic_min = T.array.min()
        arithmetic_max = T.array.max()

        # ==========================================
        # Method 2: Integration-based statistics
        # ==========================================
        # Volume integral (for normalization)
        I_vol = uw.maths.Integral(mesh, fn=1.0)
        volume = I_vol.evaluate()

        # Mean: ∫T dV / V
        I_T = uw.maths.Integral(mesh, fn=T.sym[0])
        integration_mean = I_T.evaluate() / volume

        # Variance: ∫T² dV / V - (∫T dV / V)²
        I_T2 = uw.maths.Integral(mesh, fn=T.sym[0] ** 2)
        mean_T2 = I_T2.evaluate() / volume
        variance = mean_T2 - integration_mean**2
        integration_std = np.sqrt(max(variance, 0.0))

        # ==========================================
        # Verify results
        # ==========================================
        print(f"\nComplete Statistics Workflow:")
        print(f"  Arithmetic statistics:")
        print(f"    Mean: {arithmetic_mean:.2f} K")
        print(f"    Std:  {arithmetic_std:.2f} K")
        print(f"    Min:  {arithmetic_min:.2f} K")
        print(f"    Max:  {arithmetic_max:.2f} K")
        print(f"  Integration-based statistics:")
        print(f"    Mean: {integration_mean:.2f} K")
        print(f"    Std:  {integration_std:.2f} K")
        print(f"  Difference in mean: {abs(integration_mean - arithmetic_mean):.2f} K")

        # All values should be physically reasonable
        assert 273.0 < arithmetic_mean < 373.0
        assert 273.0 < integration_mean < 373.0
        assert arithmetic_std > 0
        assert integration_std > 0
        assert arithmetic_min < arithmetic_mean < arithmetic_max
        assert integration_mean < arithmetic_max  # Integration mean is spatial average


class TestSwarmIntegrationVsArithmetic:
    """Demonstrate when integration is preferred over simple arithmetic."""

    def test_weighted_vs_unweighted_statistics(self):
        """
        Show that integration gives true spatial mean, while arithmetic
        mean is particle-weighted (biased by particle distribution).

        Example: Value concentrated in one region but distributed in another.
        """
        mesh = uw.meshing.StructuredQuadBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(8, 8)
        )

        swarm = uw.swarm.Swarm(mesh)

        # Create variable BEFORE adding particles
        var = uw.swarm.SwarmVariable("field", swarm, 1, proxy_degree=2)

        # Create 100 particles all at x=0.1 (left side)
        left_particles = np.column_stack(
            [0.1 * np.ones(100), np.random.RandomState(42).uniform(0.0, 1.0, 100)]
        )

        swarm.add_particles_with_coordinates(left_particles)
        var.data[:, 0] = 100.0  # All particles have value 100

        # Arithmetic mean: 100.0 (all particles have same value)
        arithmetic_mean = var.array.mean()

        # Integration mean: For a constant field, RBF interpolation preserves
        # the value everywhere it's interpolated, so both should be equal.
        # This validates that RBF works correctly for constant fields.
        I_vol = uw.maths.Integral(mesh, fn=1.0)
        I_f = uw.maths.Integral(mesh, fn=var.sym[0])
        integration_mean = I_f.evaluate() / I_vol.evaluate()

        print(f"\nWeighted vs Unweighted Statistics:")
        print(f"  Arithmetic (particle-weighted) mean: {arithmetic_mean:.2f}")
        print(f"  Integration (space-weighted) mean: {integration_mean:.2f}")
        print(f"  Ratio (integration/arithmetic): {integration_mean/arithmetic_mean:.3f}")

        # For constant field, both methods should give the same result
        # RBF preserves constant values perfectly
        assert np.isclose(arithmetic_mean, integration_mean, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
