"""
Parallel tests for UnitAwareArray global statistics methods.

These tests require MPI and should be run with pytest-mpi:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0750_global_statistics.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0750_global_statistics.py

Tests verify:
- MPI consistency (all ranks get same global value)
- Units preservation through global operations
- Correct calculation of global statistics
- Edge cases (empty arrays, vectors, tensors)
"""

import pytest
import numpy as np
import underworld3 as uw

# Set default timeout for all parallel tests (prevent hangs)
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(60)]


@pytest.mark.mpi(min_size=2)
def test_global_max_consistency():
    """Test that global_max returns same value on all ranks."""
    uw.pprint(0, "→ Starting test_global_max_consistency")
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
        units="km",
    )

    coords = mesh.X.coords
    y_coord = coords[:, 1]

    # Local max will differ between ranks (due to domain decomposition)
    local_max = y_coord.max()

    # Global max should be same on all ranks
    global_max = y_coord.global_max()

    # Verify units preserved
    assert hasattr(global_max, "units"), "global_max should preserve units"
    assert str(global_max.units) == "kilometer", f"Expected 'kilometer', got {global_max.units}"

    # Verify consistency across ranks
    all_global = uw.mpi.comm.gather(float(global_max), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_global)) == 1, f"All ranks must have same global_max, got {all_global}"
        # Should be 1.0 km (max y-coordinate)
        assert abs(float(global_max) - 1.0) < 1e-10, f"Expected 1.0, got {float(global_max)}"


@pytest.mark.mpi(min_size=2)
def test_global_min_consistency():
    """Test that global_min returns same value on all ranks."""
    uw.pprint(0, "→ Starting test_global_min_consistency")
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
        units="m",
    )

    coords = mesh.X.coords
    x_coord = coords[:, 0]

    # Global min should be same on all ranks
    global_min = x_coord.global_min()

    # Verify units preserved
    assert hasattr(global_min, "units"), "global_min should preserve units"
    assert str(global_min.units) == "meter", f"Expected 'meter', got {global_min.units}"

    # Verify consistency across ranks
    all_global = uw.mpi.comm.gather(float(global_min), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_global)) == 1, f"All ranks must have same global_min, got {all_global}"
        # Should be 0.0 m (min x-coordinate)
        assert abs(float(global_min) - 0.0) < 1e-10, f"Expected 0.0, got {float(global_min)}"


@pytest.mark.mpi(min_size=2)
def test_global_mean_calculation():
    """Test that global_mean is calculated correctly across ranks."""
    uw.pprint(0, "→ Starting test_global_mean_calculation")
    # Create simple array with known global mean
    local_data = np.ones(10, dtype=float) * (uw.mpi.rank + 1)
    arr = uw.utilities.UnitAwareArray(local_data, units="K")

    # Global mean should be: (1*10 + 2*10 + ... + N*10) / (N*10) = (N+1)/2
    global_mean = arr.global_mean()
    expected_mean = (uw.mpi.size + 1) / 2.0

    # Verify calculation
    assert (
        abs(float(global_mean) - expected_mean) < 1e-10
    ), f"Expected mean {expected_mean}, got {float(global_mean)}"

    # Verify units preserved
    assert hasattr(global_mean, "units"), "global_mean should preserve units"
    assert str(global_mean.units) == "kelvin", f"Expected 'kelvin', got {global_mean.units}"

    # Verify consistency
    all_global = uw.mpi.comm.gather(float(global_mean), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_global)) == 1, f"All ranks must have same global_mean, got {all_global}"


@pytest.mark.mpi(min_size=2)
def test_global_size():
    """Test that global_size returns total element count."""
    uw.pprint(0, "→ Starting test_global_size")
    # Create arrays of different sizes on each rank
    local_size = 10 + uw.mpi.rank * 5
    local_data = np.random.randn(local_size)
    arr = uw.utilities.UnitAwareArray(local_data, units="m")

    # Global size should be sum of all local sizes
    global_size = arr.global_size()

    # Calculate expected size
    all_sizes = uw.mpi.comm.gather(local_size, root=0)
    if uw.mpi.rank == 0:
        expected_size = sum(all_sizes)
        assert (
            global_size == expected_size
        ), f"Expected total size {expected_size}, got {global_size}"

    # Verify consistency across ranks
    all_global_sizes = uw.mpi.comm.gather(global_size, root=0)
    if uw.mpi.rank == 0:
        assert (
            len(set(all_global_sizes)) == 1
        ), f"All ranks must have same global_size, got {all_global_sizes}"


@pytest.mark.mpi(min_size=2)
def test_global_rms_calculation():
    """Test RMS = norm / sqrt(size) relationship."""
    uw.pprint(0, "→ Starting test_global_rms_calculation")
    # Create simple array
    local_data = np.random.randn(10) + uw.mpi.rank * 100
    arr = uw.utilities.UnitAwareArray(local_data, units="m")

    # Calculate components
    norm = arr.global_norm()
    size = arr.global_size()
    rms = arr.global_rms()

    # Verify formula: RMS = norm / sqrt(size)
    expected_rms = float(norm) / np.sqrt(size)
    assert (
        abs(float(rms) - expected_rms) < 1e-6
    ), f"RMS formula mismatch: {float(rms)} vs {expected_rms}"

    # Verify units preserved
    assert hasattr(rms, "units"), "global_rms should preserve units"
    assert str(rms.units) == "meter", f"Expected 'meter', got {rms.units}"

    # Verify consistency
    all_rms = uw.mpi.comm.gather(float(rms), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_rms)) == 1, f"All ranks must have same global_rms, got {all_rms}"


@pytest.mark.mpi(min_size=2)
def test_global_variance_calculation():
    """Test that global_var uses parallel algorithm correctly."""
    uw.pprint(0, "→ Starting test_global_variance_calculation")
    # Create array with different values on each rank
    local_data = np.ones(20, dtype=float) * (uw.mpi.rank + 1)
    arr = uw.utilities.UnitAwareArray(local_data, units="K")

    # Calculate global variance
    global_var = arr.global_var()
    global_std = arr.global_std()

    # Verify std = sqrt(var)
    assert abs(float(global_std) - np.sqrt(float(global_var))) < 1e-6, "std should equal sqrt(var)"

    # Verify units (variance has squared units)
    assert hasattr(global_var, "units"), "global_var should have units"
    # Variance has squared units
    assert "kelvin ** 2" in str(global_var.units) or "kelvin²" in str(
        global_var.units
    ), f"Expected squared kelvin units, got {global_var.units}"

    # Verify consistency
    all_var = uw.mpi.comm.gather(float(global_var), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_var)) == 1, f"All ranks must have same global_var, got {all_var}"


@pytest.mark.mpi(min_size=2)
def test_global_sum_accumulation():
    """Test that global_sum correctly accumulates across ranks."""
    uw.pprint(0, "→ Starting test_global_sum_accumulation")
    # Create array where we know the global sum
    local_data = np.ones(10, dtype=float) * uw.mpi.rank
    arr = uw.utilities.UnitAwareArray(local_data, units="Pa")

    # Global sum should be: (0*10 + 1*10 + ... + (N-1)*10) = 10 * (N-1)*N/2
    global_sum = arr.global_sum()
    expected_sum = 10.0 * (uw.mpi.size - 1) * uw.mpi.size / 2

    assert (
        abs(float(global_sum) - expected_sum) < 1e-6
    ), f"Expected sum {expected_sum}, got {float(global_sum)}"

    # Verify units preserved
    assert hasattr(global_sum, "units"), "global_sum should preserve units"
    assert str(global_sum.units) == "pascal", f"Expected 'pascal', got {global_sum.units}"

    # Verify consistency
    all_sum = uw.mpi.comm.gather(float(global_sum), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_sum)) == 1, f"All ranks must have same global_sum, got {all_sum}"


@pytest.mark.mpi(min_size=2)
def test_global_norm_calculation():
    """Test that global_norm is calculated correctly (L2 norm)."""
    uw.pprint(0, "→ Starting test_global_norm_calculation")
    # Simple test case: array of ones
    local_data = np.ones(10, dtype=float)
    arr = uw.utilities.UnitAwareArray(local_data, units="m")

    # Global norm should be sqrt(total_size)
    global_norm = arr.global_norm()
    total_size = arr.global_size()
    expected_norm = np.sqrt(total_size)

    assert (
        abs(float(global_norm) - expected_norm) < 1e-6
    ), f"Expected norm {expected_norm}, got {float(global_norm)}"

    # Verify units preserved
    assert hasattr(global_norm, "units"), "global_norm should preserve units"

    # Verify consistency
    all_norm = uw.mpi.comm.gather(float(global_norm), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_norm)) == 1, f"All ranks must have same global_norm, got {all_norm}"


@pytest.mark.mpi(min_size=2)
def test_global_operations_on_mesh_coords():
    """Test global operations on actual mesh coordinates."""
    uw.pprint(0, "→ Starting test_global_operations_on_mesh_coords")
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1000.0, 2900.0),
        cellSize=250.0,
        units="km",
    )

    coords = mesh.X.coords
    y_coord = coords[:, 1]

    # Get all global statistics
    g_max = y_coord.global_max()
    g_min = y_coord.global_min()
    g_mean = y_coord.global_mean()
    g_sum = y_coord.global_sum()
    g_norm = y_coord.global_norm()
    g_rms = y_coord.global_rms()
    g_size = y_coord.global_size()

    # All should have units
    for stat in [g_max, g_min, g_mean, g_sum, g_norm, g_rms]:
        assert hasattr(stat, "units"), f"Statistic should have units"
        assert (
            "kilometer" in str(stat.units) or "km" in str(stat.units).lower()
        ), f"Expected km units, got {stat.units}"

    # Size should be dimensionless
    assert isinstance(g_size, int), f"global_size should return int, got {type(g_size)}"

    # Physical constraints for this mesh
    if uw.mpi.rank == 0:
        assert abs(float(g_max) - 2900.0) < 1.0, f"Max y should be ~2900, got {float(g_max)}"
        assert abs(float(g_min) - 0.0) < 1.0, f"Min y should be ~0, got {float(g_min)}"
        assert g_size > 0, f"Should have particles, got {g_size}"


@pytest.mark.mpi(min_size=2)
def test_vector_global_operations():
    """Test global operations on vector (multi-component) arrays."""
    uw.pprint(0, "→ Starting test_vector_global_operations")
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
        units="m",
    )

    # Full coordinate vector (shape: N, 2)
    coords = mesh.X.coords

    # Global max should return scalar (max of ALL elements)
    global_max = coords.global_max()
    assert (
        not hasattr(global_max, "shape")
        or np.isscalar(global_max)
        or (hasattr(global_max, "shape") and global_max.shape == ())
    ), f"Vector global_max should return scalar, got shape {getattr(global_max, 'shape', 'no shape')}"

    # Should be 1.0 (max coordinate value)
    assert abs(float(global_max) - 1.0) < 1e-10, f"Expected 1.0, got {float(global_max)}"

    # Verify consistency
    all_max = uw.mpi.comm.gather(float(global_max), root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_max)) == 1, f"All ranks must have same result, got {all_max}"


@pytest.mark.mpi(min_size=2)
def test_tensor_not_implemented():
    """Test that global operations on tensors raise NotImplementedError."""
    uw.pprint(0, "→ Starting test_tensor_not_implemented")
    # Create 3D tensor array
    tensor_data = np.random.randn(5, 3, 3)
    tensor_arr = uw.utilities.UnitAwareArray(tensor_data, units="Pa")

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="not implemented for tensors"):
        tensor_arr.global_max()

    with pytest.raises(NotImplementedError, match="not implemented for tensors"):
        tensor_arr.global_min()


@pytest.mark.mpi(min_size=2)
def test_empty_array_handling():
    """Test global operations when some ranks have empty arrays."""
    uw.pprint(0, "→ Starting test_empty_array_handling")
    # Rank 0 gets data, others get empty
    if uw.mpi.rank == 0:
        local_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    else:
        local_data = np.array([])

    arr = uw.utilities.UnitAwareArray(local_data, units="K")

    # Should still work - global operations handle empty local arrays
    global_max = arr.global_max()
    global_min = arr.global_min()
    global_size = arr.global_size()

    # Max should be 5.0, min should be 1.0
    assert abs(float(global_max) - 5.0) < 1e-10, f"Expected 5.0, got {float(global_max)}"
    assert abs(float(global_min) - 1.0) < 1e-10, f"Expected 1.0, got {float(global_min)}"

    # Total size should be 5
    if uw.mpi.rank == 0:
        assert global_size == 5, f"Expected size 5, got {global_size}"


if __name__ == "__main__":
    # Allow running standalone for debugging
    import sys

    sys.exit(pytest.main([__file__, "-v", "--with-mpi"]))
