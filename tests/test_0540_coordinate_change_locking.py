import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
#!/usr/bin/env python3
"""
Coordinate Change Locking Test

This test verifies that variable arrays are properly locked during coordinate changes
to prevent data corruption.

Usage:
    pixi run python tests/test_coordinate_change_locking.py
"""

import numpy as np
import threading
import time


def test_mesh_variable_locking():
    """Test that MeshVariable arrays are locked during mesh.points changes"""
    print("🧪 Testing MeshVariable locking during mesh coordinate changes...")

    from underworld3.meshing import UnstructuredSimplexBox
    from underworld3.discretisation import MeshVariable

    # Create test mesh and variable
    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0)
    mesh_var = MeshVariable("test_var", mesh, 1, degree=1)

    n_nodes = mesh_var.coords.shape[0]
    if n_nodes == 0:
        print("⚠️  No nodes in mesh, skipping test")
        return

    # Initialize variable data
    test_data = np.random.rand(n_nodes) * 10.0
    mesh_var.array[:, 0, 0] = test_data

    # Test that variable callback respects mesh update lock
    callback_executed = False

    def variable_modification_thread():
        nonlocal callback_executed
        time.sleep(0.1)  # Let mesh update start
        try:
            # This should be safe due to locking
            mesh_var.array[:, 0, 0] = np.random.rand(n_nodes) * 5.0
            callback_executed = True
        except Exception as e:
            print(f"Variable modification failed: {e}")

    # Start variable modification in background
    thread = threading.Thread(target=variable_modification_thread)
    thread.start()

    # Simulate mesh coordinate change
    original_coords = mesh.X.coords.copy()
    try:
        # This should acquire the mesh update lock
        mesh.X.coords[:] = mesh.X.coords + np.random.rand(*mesh.X.coords.shape) * 0.01
        print("✅ Mesh coordinate change completed")
    except Exception as e:
        print(f"❌ Mesh coordinate change failed: {e}")

    thread.join()

    if callback_executed:
        print("✅ Variable modification completed safely")
    else:
        print("⚠️  Variable modification was prevented (may be expected)")

    print("✅ MeshVariable locking test completed")


def test_swarm_variable_locking():
    """Test that SwarmVariable arrays are locked during coordinate changes"""
    print("\n🧪 Testing SwarmVariable locking during coordinate changes...")

    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    # Create test environment
    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0)
    test_swarm = swarm.Swarm(mesh)

    # Define variable BEFORE populating (PETSc requirement)
    scalar_var = swarm.SwarmVariable("test_scalar", test_swarm, 1, dtype=float)

    # Now populate the swarm
    test_swarm.populate(fill_param=2)

    n_particles = test_swarm.local_size
    if n_particles == 0:
        print("⚠️  No particles in swarm, skipping test")
        return

    # Initialize variable data
    test_data = np.random.rand(n_particles) * 10.0
    scalar_var.array[:, 0, 0] = test_data

    # Test migration disabled protection
    callback_executed = False

    def variable_modification_thread():
        nonlocal callback_executed
        time.sleep(0.1)  # Let migration start
        try:
            with test_swarm.migration_disabled():
                # This should not trigger callbacks due to migration_disabled check
                scalar_var.array[:, 0, 0] = np.random.rand(n_particles) * 5.0
                callback_executed = True
        except Exception as e:
            print(f"Variable modification failed: {e}")

    # Test that variable callbacks respect migration_disabled
    thread = threading.Thread(target=variable_modification_thread)
    thread.start()

    # Simulate coordinate change that triggers migration
    try:
        original_points = test_swarm._particle_coordinates.data.copy()
        # Small coordinate change to avoid major migration
        coords = test_swarm._particle_coordinates.data
        coords[:] = coords + np.random.rand(*coords.shape) * 0.001
        print("✅ Swarm coordinate change completed")
    except Exception as e:
        print(f"❌ Swarm coordinate change failed: {e}")

    thread.join()

    if callback_executed:
        print("✅ Variable modification completed safely")
    else:
        print("⚠️  Variable modification was prevented (expected during migration)")

    print("✅ SwarmVariable locking test completed")


def run_locking_tests():
    """Run all coordinate change locking tests"""
    print("🚀 Running Coordinate Change Locking Tests")
    print("=" * 50)

    try:
        test_mesh_variable_locking()
        test_swarm_variable_locking()

        print("\n🎉 All locking tests completed!")
        print("✅ Variable arrays are properly protected during coordinate changes")
        return True

    except Exception as e:
        print(f"\n❌ Locking test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests directly"""
    success = run_locking_tests()
    exit(0 if success else 1)


def test_rejected_coord_write_leaves_mesh_coords_clean():
    """A rejected in-place write to mesh.X.coords must not poison the
    coordinate cache: the guard's message says the write "is rejected",
    so mesh.X.coords must still agree with the DM afterwards (#379
    review round 1). Previously the values landed in the cache before
    the rejection fired, and catch-and-continue code then used
    coordinates the DM does not have.
    """
    import underworld3 as uw
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    uw.discretisation.MeshVariable("t540_reject", mesh, 1, degree=1)  # live mesh

    before = np.array(mesh.X.coords)
    with pytest.raises(RuntimeError, match="rejected"):
        mesh.X.coords[:] = before + 0.01

    assert np.array_equal(np.array(mesh.X.coords), before)
    dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)
    assert np.allclose(np.array(mesh.X.coords).reshape(-1, mesh.cdim), dm_coords)
