#!/usr/bin/env python3
"""
Test symmetric tensor data property fix
"""

import numpy as np

def test_symmetric_tensor_data_property():
    """Test that symmetric tensor data property works correctly"""
    print("🧪 Testing symmetric tensor data property...")
    
    try:
        from underworld3.meshing import UnstructuredSimplexBox
        from underworld3.discretisation import MeshVariable
        import underworld3 as uw
        
        # Create test mesh
        mesh = UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), 
            maxCoords=(1.0, 1.0, 1.0), 
            cellSize=0.5,
            qdegree=2
        )
        
        # Create symmetric tensor variable
        sym_tensor_var = MeshVariable(
            "sym_tensor", 
            mesh, 
            num_components=6,  # 3D symmetric tensor has 6 unique components
            vtype=uw.VarType.SYM_TENSOR,
            degree=1
        )
        
        n_nodes = sym_tensor_var.coords.shape[0]
        print(f"Testing with {n_nodes} mesh nodes")
        print(f"Variable type: {sym_tensor_var.vtype}")
        print(f"Num components: {sym_tensor_var.num_components}")
        print(f"Array shape: {sym_tensor_var.array.shape}")
        
        # Test data property shape - should be (n_nodes, 6) for symmetric tensor
        data_shape = sym_tensor_var.data.shape
        expected_shape = (n_nodes, 6)
        
        print(f"Data shape: {data_shape}")
        print(f"Expected: {expected_shape}")
        
        if data_shape == expected_shape:
            print("✅ Data property has correct shape for symmetric tensor")
        else:
            print(f"❌ Data property shape mismatch: got {data_shape}, expected {expected_shape}")
            return False
            
        # Test that we can write to data property
        test_values = np.random.rand(n_nodes, 6) * 10.0
        sym_tensor_var.data[...] = test_values
        
        # Read back and verify
        result_values = sym_tensor_var.data[...].copy()
        np.testing.assert_array_almost_equal(
            test_values,
            result_values,
            decimal=10,
            err_msg="Symmetric tensor data assignment should preserve values"
        )
        print("✅ Symmetric tensor data assignment works")
        
        # Test that array property still works and has different shape
        array_shape = sym_tensor_var.array.shape  
        expected_array_shape = (n_nodes, 3, 3)
        print(f"Array shape: {array_shape}")
        print(f"Expected array shape: {expected_array_shape}")
        
        if array_shape == expected_array_shape:
            print("✅ Array property has correct shape for symmetric tensor")
        else:
            print(f"❌ Array property shape mismatch: got {array_shape}, expected {expected_array_shape}")
            return False
            
        print("🎉 Symmetric tensor test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Symmetric tensor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_symmetric_tensor_data_property()
    exit(0 if success else 1)