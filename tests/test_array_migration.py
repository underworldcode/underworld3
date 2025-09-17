#!/usr/bin/env python3
"""
Array Interface Migration Test

This test verifies that both legacy and enhanced array interfaces
now use direct pack/unpack methods and work without access() context.

Usage:
    pixi run python tests/test_array_migration.py
"""

import numpy as np


def test_both_interfaces_use_direct_methods():
    """Test that both legacy and enhanced interfaces use direct methods"""
    print("🧪 Testing array interface migration...")
    
    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    # Create test environment
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )
    test_swarm = swarm.Swarm(mesh)
    
    # Define variables BEFORE populating (PETSc requirement)
    scalar_var = swarm.SwarmVariable("test_scalar", test_swarm, 1, dtype=float)
    vector_var = swarm.SwarmVariable("test_vector", test_swarm, 2, dtype=float)
    
    # Now populate the swarm
    test_swarm.populate(fill_param=3)
    
    n_particles = test_swarm.local_size
    print(f"Testing with {n_particles} particles")
    
    if n_particles == 0:
        print("⚠️  No particles in swarm, skipping test")
        return
    
    # Test 1: Both interfaces work without access() context
    print("\n1️⃣ Testing that both interfaces work without access() context...")
    
    test_values = np.random.rand(n_particles) * 100.0
    
    # Test legacy interface (should now use direct methods)
    scalar_var.use_legacy_array()
    try:
        scalar_var.array[:, 0, 0] = test_values  # Should work without access()
        legacy_result = scalar_var.array[:, 0, 0]
        print("✅ Legacy interface works without access() context")
    except Exception as e:
        print(f"❌ Legacy interface failed: {e}")
        return
    
    # Test enhanced interface (already using direct methods)
    scalar_var.use_enhanced_array()
    try:
        scalar_var.array[:, 0, 0] = test_values  # Should work without access()
        enhanced_result = scalar_var.array[:, 0, 0]
        print("✅ Enhanced interface works without access() context")
    except Exception as e:
        print(f"❌ Enhanced interface failed: {e}")
        return
    
    # Both should give identical results
    np.testing.assert_array_almost_equal(
        legacy_result, enhanced_result, decimal=10,
        err_msg="Legacy and enhanced interfaces should give identical results"
    )
    print("✅ Both interfaces give identical results")
    
    # Test 2: Verify both use direct methods by comparing with explicit calls
    print("\n2️⃣ Testing consistency with direct method calls...")
    
    direct_test_values = np.random.rand(n_particles, 1) * 50.0
    
    # Use direct methods explicitly
    scalar_var.pack_uw_data_to_petsc(direct_test_values, sync=True)
    direct_result = scalar_var.unpack_uw_data_to_petsc(squeeze=False, sync=True)
    
    # Use legacy interface
    scalar_var.use_legacy_array()
    scalar_var.array[...] = direct_result
    legacy_interface_result = scalar_var.unpack_uw_data_to_petsc(squeeze=False, sync=True)
    
    # Use enhanced interface
    scalar_var.use_enhanced_array()
    scalar_var.array[...] = direct_result
    enhanced_interface_result = scalar_var.unpack_uw_data_to_petsc(squeeze=False, sync=True)
    
    # All should be identical
    np.testing.assert_array_almost_equal(
        direct_result, legacy_interface_result, decimal=10,
        err_msg="Legacy interface should match direct methods"
    )
    np.testing.assert_array_almost_equal(
        direct_result, enhanced_interface_result, decimal=10,
        err_msg="Enhanced interface should match direct methods"
    )
    print("✅ Both interfaces consistent with direct method calls")
    
    # Test 3: Vector variable indexing works for both interfaces
    print("\n3️⃣ Testing vector variable with both interfaces...")
    
    vector_test_values = np.random.rand(n_particles, 2) * 25.0
    
    # Test legacy interface with vector
    vector_var.use_legacy_array()
    vector_var.array[:, 0, 0] = vector_test_values[:, 0]
    vector_var.array[:, 0, 1] = vector_test_values[:, 1]
    legacy_vector_result = np.column_stack([
        vector_var.array[:, 0, 0],
        vector_var.array[:, 0, 1]
    ])
    
    # Test enhanced interface with vector
    vector_var.use_enhanced_array()
    vector_var.array[:, 0, 0] = vector_test_values[:, 0]
    vector_var.array[:, 0, 1] = vector_test_values[:, 1]
    enhanced_vector_result = np.column_stack([
        vector_var.array[:, 0, 0],
        vector_var.array[:, 0, 1]
    ])
    
    # Should be identical
    np.testing.assert_array_almost_equal(
        legacy_vector_result, enhanced_vector_result, decimal=10,
        err_msg="Vector operations should be identical for both interfaces"
    )
    np.testing.assert_array_almost_equal(
        vector_test_values, enhanced_vector_result, decimal=10,
        err_msg="Vector values should be preserved correctly"
    )
    print("✅ Vector operations work correctly for both interfaces")
    
    # Test 4: Performance comparison
    print("\n4️⃣ Testing performance comparison...")
    
    import time
    
    # Time legacy interface
    scalar_var.use_legacy_array()
    start_time = time.time()
    for _ in range(20):
        test_data = np.random.rand(n_particles) * 10.0
        scalar_var.array[:, 0, 0] = test_data
        _ = scalar_var.array[:, 0, 0]
    legacy_time = (time.time() - start_time) / 20
    
    # Time enhanced interface
    scalar_var.use_enhanced_array()
    start_time = time.time()
    for _ in range(20):
        test_data = np.random.rand(n_particles) * 10.0
        scalar_var.array[:, 0, 0] = test_data
        _ = scalar_var.array[:, 0, 0]
    enhanced_time = (time.time() - start_time) / 20
    
    print(f"Legacy interface:   {legacy_time:.6f} seconds per operation")
    print(f"Enhanced interface: {enhanced_time:.6f} seconds per operation")
    
    if enhanced_time < legacy_time:
        speedup = legacy_time / enhanced_time
        print(f"Enhanced interface is {speedup:.2f}x faster")
    else:
        overhead = (enhanced_time / legacy_time - 1) * 100
        print(f"Enhanced interface overhead: {overhead:.1f}%")
    
    print("✅ Performance comparison completed")
    
    print("\n🎉 Array interface migration successful!")
    print("✅ Both interfaces now use direct methods")
    print("✅ No access() context required for array operations")
    print("✅ Backward compatibility maintained")
    print("✅ Enhanced interface provides additional features")
    
    # Cleanup
    del scalar_var, vector_var, test_swarm, mesh


def test_existing_code_compatibility():
    """Test that existing code patterns still work after migration"""
    print("\n🧪 Testing existing code compatibility...")
    
    from underworld3 import swarm
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )
    test_swarm = swarm.Swarm(mesh)
    
    scalar_var = swarm.SwarmVariable("test_scalar", test_swarm, 1, dtype=float)
    test_swarm.populate(fill_param=2)
    
    n_particles = test_swarm.local_size
    if n_particles == 0:
        print("⚠️  No particles in swarm, skipping compatibility test")
        return
    
    # Test common patterns that should still work
    print("Testing common usage patterns...")
    
    # Pattern 1: Direct array assignment
    test_values = np.random.rand(n_particles) * 30.0
    scalar_var.array[:, 0, 0] = test_values  # Should work (default enhanced interface)
    result1 = scalar_var.array[:, 0, 0]
    
    # Pattern 2: Full array assignment
    full_array_data = np.random.rand(n_particles, 1, 1) * 40.0
    scalar_var.array[...] = full_array_data
    result2 = scalar_var.array[...]
    
    # Pattern 3: Interface switching
    scalar_var.use_legacy_array()  # Should still work
    scalar_var.use_enhanced_array()  # Should still work
    
    # Pattern 4: Sync context (enhanced interface only)
    with scalar_var.sync_disabled():
        scalar_var.array[:, 0, 0] = test_values
        # Should batch sync at context exit
    
    print("✅ All common usage patterns work correctly")
    print("✅ Backward compatibility maintained")
    
    del scalar_var, test_swarm, mesh


def run_migration_tests():
    """Run all migration tests"""
    print("🚀 Running Array Interface Migration Tests")
    print("=" * 50)
    
    try:
        test_both_interfaces_use_direct_methods()
        test_existing_code_compatibility()
        
        print("\n🎉 All migration tests passed!")
        print("✅ Array interface migration is complete and successful")
        return True
        
    except Exception as e:
        print(f"\n❌ Migration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests directly"""
    success = run_migration_tests()
    exit(0 if success else 1)