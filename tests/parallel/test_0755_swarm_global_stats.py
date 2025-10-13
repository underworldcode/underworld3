"""
Parallel tests for SwarmVariable global statistics methods.

These tests require MPI and should be run with pytest-mpi:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0755_swarm_global_stats.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0755_swarm_global_stats.py

Tests verify:
- MPI consistency for SwarmVariable.global_max(), global_min(), etc.
- Units preservation through swarm global operations
- Correct handling of particle distribution across ranks
- Edge cases (empty swarms on some ranks, vector variables)

Note: SwarmVariable does NOT provide global_mean(), global_rms(), global_var(), or global_std()
because these are meaningless for non-uniformly distributed particles.
"""

import pytest
import numpy as np
import underworld3 as uw

# Set default timeout for all parallel tests (prevent hangs)
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(60)]


@pytest.mark.mpi(min_size=2)
def test_swarm_global_max():
    """Test SwarmVariable.global_max() across MPI ranks."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    temperature = swarm.add_variable("T", 1, units="K")
    swarm.populate(fill_param=3)

    # Initialize with rank-specific values (higher rank = higher temperature)
    with uw.synchronised_array_update():
        temperature.data[:, 0] = 300.0 + uw.mpi.rank * 100.0

    # Global max should be from highest rank
    global_max = temperature.global_max()
    expected_max = 300.0 + (uw.mpi.size - 1) * 100.0

    # Verify calculation
    assert (
        abs(float(global_max) - expected_max) < 1e-6
    ), f"Expected max {expected_max}, got {float(global_max)}"

    # Verify units preserved
    assert hasattr(global_max, "units"), "global_max should preserve units"
    assert (
        str(global_max.units) == "kelvin"
    ), f"Expected 'kelvin', got {global_max.units}"

    # Verify consistency across ranks
    all_max = uw.mpi.comm.gather(float(global_max), root=0)
    if uw.mpi.rank == 0:
        assert (
            len(set(all_max)) == 1
        ), f"All ranks must have same global_max, got {all_max}"


@pytest.mark.mpi(min_size=2)
def test_swarm_global_min():
    """Test SwarmVariable.global_min() across MPI ranks."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    pressure = swarm.add_variable("P", 1, units="Pa")
    swarm.populate(fill_param=3)

    # Initialize with rank-specific values (lower rank = lower pressure)
    with uw.synchronised_array_update():
        pressure.data[:, 0] = 1000.0 + uw.mpi.rank * 500.0

    # Global min should be from lowest rank (rank 0)
    global_min = pressure.global_min()
    expected_min = 1000.0  # From rank 0

    # Verify calculation
    assert (
        abs(float(global_min) - expected_min) < 1e-6
    ), f"Expected min {expected_min}, got {float(global_min)}"

    # Verify units preserved
    assert hasattr(global_min, "units"), "global_min should preserve units"
    assert (
        str(global_min.units) == "pascal"
    ), f"Expected 'pascal', got {global_min.units}"

    # Verify consistency across ranks
    all_min = uw.mpi.comm.gather(float(global_min), root=0)
    if uw.mpi.rank == 0:
        assert (
            len(set(all_min)) == 1
        ), f"All ranks must have same global_min, got {all_min}"


@pytest.mark.mpi(min_size=2)
def test_swarm_global_size():
    """Test SwarmVariable.global_size() returns total particle count."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable("V", 1)
    swarm.populate(fill_param=3)

    # Get local and global sizes
    local_size = swarm.local_size
    global_size = var.global_size()

    # Gather all local sizes to verify
    all_local_sizes = uw.mpi.comm.gather(local_size, root=0)
    if uw.mpi.rank == 0:
        expected_total = sum(all_local_sizes)
        assert (
            global_size == expected_total
        ), f"Expected total {expected_total}, got {global_size}"

    # Verify consistency across ranks
    all_global_sizes = uw.mpi.comm.gather(global_size, root=0)
    if uw.mpi.rank == 0:
        assert (
            len(set(all_global_sizes)) == 1
        ), f"All ranks must have same global_size, got {all_global_sizes}"


@pytest.mark.mpi(min_size=2)
def test_swarm_global_sum():
    """Test SwarmVariable.global_sum() accumulates across ranks."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    mass = swarm.add_variable("mass", 1, units="kg")
    swarm.populate(fill_param=3)

    # Give each particle mass = 1.0 kg
    with uw.synchronised_array_update():
        mass.data[:, 0] = 1.0

    # Global sum should equal total particle count
    global_sum = mass.global_sum()
    expected_sum = float(mass.global_size())  # Each particle has mass 1.0

    assert (
        abs(float(global_sum) - expected_sum) < 1e-6
    ), f"Expected sum {expected_sum}, got {float(global_sum)}"

    # Verify units preserved
    assert hasattr(global_sum, "units"), "global_sum should preserve units"
    assert (
        str(global_sum.units) == "kilogram"
    ), f"Expected 'kilogram', got {global_sum.units}"

    # Verify consistency
    all_sum = uw.mpi.comm.gather(float(global_sum), root=0)
    if uw.mpi.rank == 0:
        assert (
            len(set(all_sum)) == 1
        ), f"All ranks must have same global_sum, got {all_sum}"


@pytest.mark.mpi(min_size=2)
def test_swarm_global_norm():
    """Test SwarmVariable.global_norm() L2 norm across ranks."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    field = swarm.add_variable("field", 1, units="m")
    swarm.populate(fill_param=3)

    # Set all values to 1.0
    with uw.synchronised_array_update():
        field.data[:, 0] = 1.0

    # Global norm should be sqrt(total_particles)
    global_norm = field.global_norm()
    expected_norm = np.sqrt(float(field.global_size()))

    assert (
        abs(float(global_norm) - expected_norm) < 1e-6
    ), f"Expected norm {expected_norm}, got {float(global_norm)}"

    # Verify units preserved
    assert hasattr(global_norm, "units"), "global_norm should preserve units"

    # Verify consistency
    all_norm = uw.mpi.comm.gather(float(global_norm), root=0)
    if uw.mpi.rank == 0:
        assert (
            len(set(all_norm)) == 1
        ), f"All ranks must have same global_norm, got {all_norm}"


@pytest.mark.mpi(min_size=2)
def test_swarm_global_operations_no_mean_rms():
    """Verify that SwarmVariable does NOT have global_mean() or global_rms()."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable("V", 1)
    swarm.populate(fill_param=3)

    # These methods should NOT exist
    assert not hasattr(
        var, "global_mean"
    ), "SwarmVariable should NOT have global_mean() - particles are non-uniformly distributed"
    assert not hasattr(
        var, "global_rms"
    ), "SwarmVariable should NOT have global_rms() - particles are non-uniformly distributed"
    assert not hasattr(
        var, "global_var"
    ), "SwarmVariable should NOT have global_var() - particles are non-uniformly distributed"
    assert not hasattr(
        var, "global_std"
    ), "SwarmVariable should NOT have global_std() - particles are non-uniformly distributed"


@pytest.mark.mpi(min_size=2)
def test_swarm_vector_variable_global_max():
    """Test global operations on vector swarm variables."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    velocity = swarm.add_variable("vel", mesh.dim, units="m/s")
    swarm.populate(fill_param=3)

    # Initialize velocity components with rank-dependent values
    with uw.synchronised_array_update():
        velocity.data[:, 0] = 10.0 + uw.mpi.rank * 5.0  # x-velocity
        velocity.data[:, 1] = 20.0 + uw.mpi.rank * 5.0  # y-velocity

    # Global max should return scalar (max of ALL components across ALL ranks)
    global_max = velocity.global_max()

    # Expected: max y-velocity from highest rank
    expected_max = 20.0 + (uw.mpi.size - 1) * 5.0

    assert (
        abs(float(global_max) - expected_max) < 1e-6
    ), f"Expected {expected_max}, got {float(global_max)}"

    # Should be scalar, not array
    assert (
        not hasattr(global_max, "shape")
        or np.isscalar(global_max)
        or (hasattr(global_max, "shape") and global_max.shape == ())
    ), f"Vector global_max should return scalar"

    # Verify consistency
    all_max = uw.mpi.comm.gather(float(global_max), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_max)) == 1, f"All ranks must have same result, got {all_max}"


@pytest.mark.mpi(min_size=2)
def test_swarm_units_without_units():
    """Test global operations on swarm variables without units."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    index = swarm.add_variable("index", 1)  # No units
    swarm.populate(fill_param=3)

    # Initialize with rank-based index
    with uw.synchronised_array_update():
        index.data[:, 0] = uw.mpi.rank

    # Global max should work even without units
    global_max = index.global_max()
    expected_max = float(uw.mpi.size - 1)

    assert (
        abs(float(global_max) - expected_max) < 1e-10
    ), f"Expected {expected_max}, got {float(global_max)}"

    # Result should be dimensionless (no units attribute or units=None)
    if hasattr(global_max, "units"):
        assert (
            global_max.units is None or str(global_max.units) == "dimensionless"
        ), f"Expected dimensionless, got {global_max.units}"


@pytest.mark.mpi(min_size=2)
def test_swarm_migration_preserves_global_values():
    """Test that particle migration doesn't break global statistics."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    value = swarm.add_variable("val", 1, units="K")
    swarm.populate(fill_param=3)

    # Set unique values
    with uw.synchronised_array_update():
        value.data[:, 0] = (
            np.arange(swarm.local_size, dtype=float) + uw.mpi.rank * 1000.0
        )

    # Get global stats before migration
    max_before = value.global_max()
    min_before = value.global_min()
    size_before = value.global_size()

    # Perturb particle positions slightly (may trigger migration)
    with swarm.migration_disabled():
        # Small perturbation that shouldn't move particles between domains
        # Note: Use explicit assignment instead of += for parallel safety
        swarm.points[:, 0] = swarm.points[:, 0] + 0.001

    # Explicit migration
    swarm.migrate()

    # Get global stats after migration
    max_after = value.global_max()
    min_after = value.global_min()
    size_after = value.global_size()

    # Global stats should be unchanged (particles just redistributed)
    assert (
        abs(float(max_before) - float(max_after)) < 1e-10
    ), "Global max should be unchanged after migration"
    assert (
        abs(float(min_before) - float(min_after)) < 1e-10
    ), "Global min should be unchanged after migration"
    assert (
        size_before == size_after
    ), f"Total particle count should be unchanged: {size_before} vs {size_after}"


@pytest.mark.mpi(min_size=2)
def test_swarm_empty_on_some_ranks():
    """Test global operations when some ranks have no particles.

    This test verifies that global statistics work correctly even when some
    MPI ranks have zero particles, which can cause shape mismatches in MPI
    collective operations if not handled correctly.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh)
    value = swarm.add_variable("val", 1, units="m")

    # Strategy: Get mesh bounds for each rank, add particles only to specific ranks
    # This ensures we can control which ranks have particles

    # Get local mesh bounds for this rank
    local_coords = mesh.data  # Local vertex coordinates
    if local_coords.size > 0:
        x_min, x_max = local_coords[:, 0].min(), local_coords[:, 0].max()
        y_min, y_max = local_coords[:, 1].min(), local_coords[:, 1].max()
        # Create particles in the CENTER of this rank's domain
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
    else:
        x_center, y_center = 0.5, 0.5  # Fallback (shouldn't happen)

    # Only rank 0 adds particles (all other ranks stay empty)
    if uw.mpi.rank == 0:
        # Add 5 particles near this rank's center
        coords = np.array([
            [x_center + 0.01 * i, y_center + 0.01 * i] for i in range(5)
        ])
    else:
        # Empty array for other ranks
        coords = np.empty((0, 2))

    # Call on ALL ranks (collective operation)
    swarm.add_particles_with_coordinates(coords)

    # Initialize values based on local particle count
    # Assign sequential values so we can verify max/min work correctly
    if swarm.local_size > 0:
        value.data[:, 0] = np.arange(1.0, swarm.local_size + 1.0)

    # Gather statistics
    from mpi4py import MPI
    local_count = swarm.local_size
    all_counts = uw.mpi.comm.gather(local_count, root=0)
    total_particles = uw.mpi.comm.allreduce(local_count, op=MPI.SUM)

    # Key test: Global operations must work even if some ranks have NO particles
    # These operations should NOT hang due to shape mismatches
    global_max = value.global_max()
    global_min = value.global_min()
    global_size = value.global_size()

    # Verify results using uw.utilities.gather_data instead of raw MPI
    all_max = uw.utilities.gather_data(float(global_max), bcast=False)
    all_min = uw.utilities.gather_data(float(global_min), bcast=False)
    all_size = uw.utilities.gather_data(global_size, bcast=False)

    # Only rank 0 checks the gathered results
    if uw.mpi.rank == 0:
        print(f"Particle distribution: {all_counts}")
        print(f"Total particles: {total_particles}")

        # Verify we got at least some particles total
        assert total_particles > 0, f"Expected some particles, got {total_particles}"

        # For multi-rank runs, we expect at least one rank to be empty
        # (on single rank, this test is vacuous but shouldn't fail)
        if uw.mpi.size > 1:
            assert 0 in all_counts, f"Expected at least one empty rank, got counts {all_counts}"

        # If we have particles, verify the statistics are correct
        if total_particles > 0:
            # Max should equal the total number of particles (since we used arange starting at 1)
            assert (
                abs(float(global_max) - float(total_particles)) < 1e-10
            ), f"Expected max {total_particles}, got {float(global_max)}"

            # Min should be 1.0 (first value from arange)
            assert (
                abs(float(global_min) - 1.0) < 1e-10
            ), f"Expected min 1.0, got {float(global_min)}"

            # Total size should match
            assert global_size == total_particles, f"Expected total {total_particles}, got {global_size}"

        # All ranks should have same global values (MPI consistency check)
        assert (
            len(set(all_max)) == 1
        ), f"All ranks must have same global_max, got {all_max}"
        assert (
            len(set(all_min)) == 1
        ), f"All ranks must have same global_min, got {all_min}"
        assert (
            len(set(all_size)) == 1
        ), f"All ranks must have same global_size, got {all_size}"


@pytest.mark.mpi(min_size=2)
def test_that_tests_end():
    assert True


if __name__ == "__main__":
    # Allow running standalone for debugging
    import sys

    sys.exit(pytest.main([__file__, "-v", "--with-mpi"]))
