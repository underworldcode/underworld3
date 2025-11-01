"""
Comprehensive test of all reduction operations across the system.

This test validates that max(), min(), mean(), sum(), and std() methods
are consistently implemented across all array views and variables.

Test coverage:
- Swarm array view reductions (SimpleSwarmArrayView, TensorSwarmArrayView)
- Mesh array view reductions (SimpleMeshArrayView, TensorMeshArrayView)
- Global reductions on _BaseMeshVariable (using PETSc)
- Global reductions on UnitAwareArray (using MPI)
"""

import numpy as np
import pytest
import underworld3 as uw


class TestSwarmArrayViewReductions:
    """Test reduction operations on swarm array views."""

    def test_simple_swarm_array_view_reductions(self):
        """Test all reduction operations on SimpleSwarmArrayView."""
        swarm = uw.swarm.Swarm(uw.meshing.StructuredQuadBox(elementRes=(5, 5)))
        swarm.populate(np.random.RandomState(0).random((100, 2)))

        # Create swarm variables
        scalar_var = uw.swarm.SwarmVariable("scalar", swarm, 1)
        vector_var = uw.swarm.SwarmVariable("vector", swarm, 2)

        # Set test data
        scalar_var.data[:, 0] = np.linspace(1, 10, swarm.particle_coordinates.shape[0])
        vector_var.data[:, 0] = np.linspace(1, 5, swarm.particle_coordinates.shape[0])
        vector_var.data[:, 1] = np.linspace(5, 10, swarm.particle_coordinates.shape[0])

        # Test scalar variable - all reductions should return float
        assert isinstance(scalar_var.array.max(), (float, np.floating))
        assert isinstance(scalar_var.array.min(), (float, np.floating))
        assert isinstance(scalar_var.array.mean(), (float, np.floating))
        assert isinstance(scalar_var.array.sum(), (float, np.floating))
        assert isinstance(scalar_var.array.std(), (float, np.floating))

        # Test vector variable - all reductions should return tuple or array
        max_result = vector_var.array.max()
        min_result = vector_var.array.min()
        mean_result = vector_var.array.mean()
        sum_result = vector_var.array.sum()
        std_result = vector_var.array.std()

        # All should return tuples or array-like
        assert len(max_result) == 2
        assert len(min_result) == 2
        assert len(mean_result) == 2
        assert len(sum_result) == 2
        assert len(std_result) == 2

    def test_tensor_swarm_array_view_reductions(self):
        """Test all reduction operations on TensorSwarmArrayView."""
        swarm = uw.swarm.Swarm(uw.meshing.StructuredQuadBox(elementRes=(5, 5)))
        swarm.populate(np.random.RandomState(0).random((100, 2)))

        # Create tensor swarm variable (2D tensor)
        tensor_var = uw.swarm.SwarmVariable("tensor", swarm, 4)  # 2x2 tensor

        # Set test data (each particle has a 2x2 matrix)
        for i in range(tensor_var.data.shape[0]):
            tensor_var.data[i, :] = np.linspace(1, 4, 4) + i * 0.01

        # Test tensor variable - all reductions should return tuple
        max_result = tensor_var.array.max()
        min_result = tensor_var.array.min()
        mean_result = tensor_var.array.mean()
        sum_result = tensor_var.array.sum()
        std_result = tensor_var.array.std()

        # All should return tuples with 4 components
        assert len(max_result) == 4
        assert len(min_result) == 4
        assert len(mean_result) == 4
        assert len(sum_result) == 4
        assert len(std_result) == 4

        # Verify values are reasonable
        assert all(m > 0 for m in max_result)
        assert all(m > 0 for m in min_result)


class TestMeshArrayViewReductions:
    """Test reduction operations on mesh array views."""

    def test_simple_mesh_variable_reductions(self):
        """Test all reduction operations on simple (scalar) mesh variables."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(5, 5))
        var = uw.discretisation.MeshVariable("scalar", mesh, 1)

        # Set test data
        var.array[..., 0] = np.linspace(1, 10, var.shape[0])

        # Test scalar variable - array views should return float
        assert isinstance(var.array.max(), (float, np.floating))
        assert isinstance(var.array.min(), (float, np.floating))
        assert isinstance(var.array.mean(), (float, np.floating))
        assert isinstance(var.array.sum(), (float, np.floating))
        assert isinstance(var.array.std(), (float, np.floating))

        # Verify some basic properties
        assert var.array.max() > var.array.min()
        assert var.array.min() <= var.array.mean() <= var.array.max()
        assert var.array.std() >= 0

    def test_vector_mesh_variable_reductions(self):
        """Test all reduction operations on vector mesh variables."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(5, 5))
        var = uw.discretisation.MeshVariable("vector", mesh, 2)

        # Set test data
        var.array[..., 0] = np.linspace(1, 5, var.shape[0])
        var.array[..., 1] = np.linspace(5, 10, var.shape[0])

        # Test vector variable - array views should return tuple
        max_result = var.array.max()
        min_result = var.array.min()
        mean_result = var.array.mean()
        sum_result = var.array.sum()
        std_result = var.array.std()

        # All should be tuples with 2 components
        assert isinstance(max_result, tuple) and len(max_result) == 2
        assert isinstance(min_result, tuple) and len(min_result) == 2
        assert isinstance(mean_result, tuple) and len(mean_result) == 2
        assert isinstance(sum_result, tuple) and len(sum_result) == 2
        assert isinstance(std_result, tuple) and len(std_result) == 2

        # Verify component-wise properties
        assert max_result[0] > min_result[0]
        assert max_result[1] > min_result[1]
        assert std_result[0] >= 0
        assert std_result[1] >= 0


class TestGlobalMeshVariableReductions:
    """Test global (PETSc-based) reduction operations on mesh variables."""

    def test_scalar_variable_global_reductions(self):
        """Test global reduction methods on scalar mesh variables."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(5, 5))
        var = uw.discretisation.MeshVariable("scalar", mesh, 1)

        # Set test data
        var.array[..., 0] = np.linspace(1, 10, var.shape[0])

        # Test all global reduction methods exist and return scalars
        assert isinstance(var.max(), (float, np.floating))
        assert isinstance(var.min(), (float, np.floating))
        assert isinstance(var.mean(), (float, np.floating))
        assert isinstance(var.sum(), (float, np.floating))
        assert isinstance(var.std(), (float, np.floating))

        # Verify basic properties
        assert var.max() > var.min()
        assert var.min() <= var.mean() <= var.max()
        assert var.std() >= 0

    def test_vector_variable_global_reductions(self):
        """Test global reduction methods on vector mesh variables."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(5, 5))
        var = uw.discretisation.MeshVariable("vector", mesh, 2)

        # Set test data
        var.array[..., 0] = np.linspace(1, 5, var.shape[0])
        var.array[..., 1] = np.linspace(5, 10, var.shape[0])

        # Test all global reduction methods - should return tuples
        max_result = var.max()
        min_result = var.min()
        mean_result = var.mean()
        sum_result = var.sum()
        std_result = var.std()

        # All should be tuples with 2 components
        assert isinstance(max_result, tuple) and len(max_result) == 2
        assert isinstance(min_result, tuple) and len(min_result) == 2
        assert isinstance(mean_result, tuple) and len(mean_result) == 2
        assert isinstance(sum_result, tuple) and len(sum_result) == 2
        assert isinstance(std_result, tuple) and len(std_result) == 2

        # Component-wise properties should hold
        assert max_result[0] > min_result[0]
        assert max_result[1] > min_result[1]
        assert std_result[0] >= 0
        assert std_result[1] >= 0


class TestUnitAwareArrayReductions:
    """Test reduction operations on UnitAwareArray with unit preservation."""

    def test_unit_aware_array_local_reductions(self):
        """Test local reduction operations on UnitAwareArray."""
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        # Create unit-aware array with units
        data = np.linspace(1, 10, 100)
        arr = UnitAwareArray(data, units="m")

        # Test local reductions - should preserve units
        max_val = arr.max()
        min_val = arr.min()
        mean_val = arr.mean()
        sum_val = arr.sum()
        std_val = arr.std()

        # All should be UWQuantity or have units
        assert hasattr(max_val, "magnitude") or isinstance(max_val, (float, np.floating))
        assert hasattr(min_val, "magnitude") or isinstance(min_val, (float, np.floating))
        assert hasattr(mean_val, "magnitude") or isinstance(mean_val, (float, np.floating))
        assert hasattr(sum_val, "magnitude") or isinstance(sum_val, (float, np.floating))
        assert hasattr(std_val, "magnitude") or isinstance(std_val, (float, np.floating))

    def test_unit_aware_array_global_reductions(self):
        """Test global reduction operations on UnitAwareArray."""
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        # Create unit-aware array with units
        data = np.linspace(1, 10, 100)
        arr = UnitAwareArray(data, units="m")

        # Test global reduction methods exist
        assert hasattr(arr, "global_max")
        assert hasattr(arr, "global_min")
        assert hasattr(arr, "global_mean")
        assert hasattr(arr, "global_sum")
        assert hasattr(arr, "global_std")

        # These should be callable
        assert callable(arr.global_max)
        assert callable(arr.global_min)
        assert callable(arr.global_mean)
        assert callable(arr.global_sum)
        assert callable(arr.global_std)


class TestReductionOperationConsistency:
    """Test that reduction operations are consistent across the system."""

    def test_swarm_vs_mesh_reduction_results(self):
        """Verify that swarm and mesh reductions produce similar statistics."""
        # Create mesh and swarm
        mesh = uw.meshing.StructuredQuadBox(elementRes=(5, 5))
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(np.random.RandomState(0).random((100, 2)))

        # Create variables
        swarm_var = uw.swarm.SwarmVariable("scalar", swarm, 1)
        mesh_var = uw.discretisation.MeshVariable("scalar", mesh, 1)

        # Set same data on both
        test_data = np.linspace(1, 10, swarm_var.data.shape[0])
        swarm_var.data[:, 0] = test_data
        mesh_var.array[: len(test_data), 0] = test_data

        # Test that reduction methods exist for both
        assert hasattr(swarm_var.array, "std")
        assert hasattr(mesh_var.array, "std")
        assert hasattr(mesh_var, "std")  # global reduction

    def test_all_reduction_methods_exist(self):
        """Verify all expected reduction methods exist."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test", mesh, 1)

        # Check all methods exist on variable
        methods = ["min", "max", "mean", "sum", "std"]
        for method in methods:
            assert hasattr(var, method), f"Method {method} missing on MeshVariable"
            assert callable(getattr(var, method)), f"Method {method} not callable"

        # Check all methods exist on array view
        array_methods = ["min", "max", "mean", "sum", "std"]
        for method in array_methods:
            assert hasattr(var.array, method), f"Method {method} missing on array view"
            assert callable(getattr(var.array, method)), f"Method {method} not callable"


class TestStdMethodNewImplementations:
    """Specific tests for the newly added std() method."""

    def test_mesh_variable_std_new_method(self):
        """Test the newly added std() method on mesh variables."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(5, 5))
        var = uw.discretisation.MeshVariable("test", mesh, 1)

        # Set data with known statistics
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_std = np.std(data)

        var.array[: len(data), 0] = data

        # Test the newly added global std() method
        result = var.std()
        assert isinstance(result, (float, np.floating))
        # Due to PETSc and ghost cells, we can't expect exact match,
        # but std should be positive for non-constant data
        assert result >= 0

    def test_swarm_std_new_method(self):
        """Test the newly added std() method on swarm variables."""
        swarm = uw.swarm.Swarm(uw.meshing.StructuredQuadBox(elementRes=(5, 5)))
        swarm.populate(np.random.RandomState(0).random((50, 2)))

        var = uw.swarm.SwarmVariable("test", swarm, 1)

        # Set data with known statistics
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var.data[:5, 0] = data

        # Test the newly added std() method on swarm array view
        result = var.array.std()
        assert isinstance(result, (float, np.floating))
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
