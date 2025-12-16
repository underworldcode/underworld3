"""
Test the newly added std() reduction method across array views.

This test validates that std() has been added to:
1. SimpleSwarmArrayView and TensorSwarmArrayView
2. _BaseMeshVariable (global std with PETSc)
3. SimpleMeshArrayView and TensorMeshArrayView (array views)
4. UnitAwareArray (already existed, testing for consistency)

STATUS (2025-11-15):
- SWARM TESTS FIXED: Corrected variable ordering (create before adding particles)
- All swarm tests now use proper initialization sequence
- Mesh tests validated separately
- Marked as Tier B - tests should pass, testing std() reduction method
"""

import numpy as np
import pytest
import underworld3 as uw


@pytest.mark.level_2  # Intermediate - array reductions
@pytest.mark.tier_b   # Validated - some tests have bugs, mesh tests may be OK
class TestStdMethodOnArrayViews:
    """Test std() method on array views."""

    def test_mesh_simple_array_view_std(self):
        """Test std() on SimpleMeshArrayView."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test_scalar", mesh, 1)

        # Set test data
        var.array[..., 0] = np.linspace(1, 10, var.shape[0])

        # Test that std() method exists and works
        std_result = var.array.std()
        assert isinstance(std_result, (float, np.floating, int))
        assert std_result >= 0.0

    def test_mesh_vector_array_view_std(self):
        """Test std() on TensorMeshArrayView (vector data)."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test_vector", mesh, 2)

        # Set test data
        var.array[..., 0] = np.linspace(1, 5, var.shape[0])
        var.array[..., 1] = np.linspace(5, 10, var.shape[0])

        # Test that std() method exists and returns tuple
        std_result = var.array.std()
        assert isinstance(std_result, tuple)
        assert len(std_result) == 2
        assert all(isinstance(s, (float, np.floating, int)) for s in std_result)
        assert all(s >= 0.0 for s in std_result)

    def test_swarm_simple_array_view_std(self):
        """Test std() on SimpleSwarmArrayView."""
        swarm = uw.swarm.Swarm(uw.meshing.StructuredQuadBox(elementRes=(3, 3)))
        # Use simple grid points instead of random
        pts = np.mgrid[0:1:0.2, 0:1:0.2].reshape(2, -1).T
        if len(pts) > 0:
            # Create variable BEFORE adding particles (CRITICAL!)
            var = uw.swarm.SwarmVariable("test_scalar", swarm, 1)

            # NOW add particles
            swarm.add_particles_with_coordinates(pts)
            # Use actual swarm particle count (may differ from pts if some rejected)
            n_particles = swarm._particle_coordinates.data.shape[0]
            var.data[:, 0] = np.linspace(1, 10, n_particles)

            # Test that std() method exists and works
            std_result = var.array.std()
            assert isinstance(std_result, (float, np.floating, int))
            assert std_result >= 0.0

    def test_swarm_vector_array_view_std(self):
        """Test std() on TensorSwarmArrayView (vector data).

        NOTE: Swarm array views use numpy's default std() which returns a single
        scalar (overall standard deviation), unlike mesh array views which return
        per-component tuples. This is a known inconsistency documented here.
        """
        swarm = uw.swarm.Swarm(uw.meshing.StructuredQuadBox(elementRes=(3, 3)))
        pts = np.mgrid[0:1:0.2, 0:1:0.2].reshape(2, -1).T
        if len(pts) > 0:
            # Create variable BEFORE adding particles (CRITICAL!)
            var = uw.swarm.SwarmVariable("test_vector", swarm, 2)

            # NOW add particles
            swarm.add_particles_with_coordinates(pts)
            # Use actual swarm particle count (may differ from pts if some rejected)
            n_particles = swarm._particle_coordinates.data.shape[0]
            var.data[:, 0] = np.linspace(1, 5, n_particles)
            var.data[:, 1] = np.linspace(5, 10, n_particles)

            # Test that std() method exists and returns a value
            # (Swarm arrays return scalar, unlike mesh arrays which return tuples)
            std_result = var.array.std()
            assert isinstance(std_result, (float, np.floating, int, tuple))
            if isinstance(std_result, tuple):
                assert len(std_result) == 2
                assert all(s >= 0.0 for s in std_result)
            else:
                assert std_result >= 0.0


class TestMeshVariableGlobalStd:
    """Test global std() method on mesh variables via _BaseMeshVariable."""

    def test_mesh_scalar_has_std_method(self):
        """Test that scalar mesh variables have std() method."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test_scalar", mesh, 1)

        # Set test data
        var.array[..., 0] = np.linspace(1, 10, var.shape[0])

        # Test that the base variable has std() method
        # Access via _base_var if it's wrapped
        base_var = var._base_var if hasattr(var, "_base_var") else var
        assert hasattr(base_var, "std"), "std() method missing on mesh variable"
        assert callable(getattr(base_var, "std")), "std() is not callable"

    def test_mesh_vector_has_std_method(self):
        """Test that vector mesh variables have std() method."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test_vector", mesh, 2)

        # Set test data
        var.array[..., 0] = np.linspace(1, 5, var.shape[0])
        var.array[..., 1] = np.linspace(5, 10, var.shape[0])

        # Test that the base variable has std() method
        base_var = var._base_var if hasattr(var, "_base_var") else var
        assert hasattr(base_var, "std"), "std() method missing on mesh vector"
        assert callable(getattr(base_var, "std")), "std() is not callable"


class TestReductionConsistency:
    """Test that all reduction methods (max, min, mean, sum, std) are consistently available."""

    def test_all_reductions_on_array_view(self):
        """Verify all 5 reduction methods exist on array views."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test", mesh, 1)

        reduction_methods = ["min", "max", "mean", "sum", "std"]
        for method in reduction_methods:
            assert hasattr(var.array, method), f"Missing {method} on array view"
            assert callable(getattr(var.array, method)), f"{method} is not callable"

    def test_all_reductions_executable(self):
        """Verify all 5 reduction methods can be executed without error."""
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        var = uw.discretisation.MeshVariable("test", mesh, 1)
        var.array[..., 0] = np.linspace(1, 10, var.shape[0])

        # Execute each reduction method
        min_result = var.array.min()
        max_result = var.array.max()
        mean_result = var.array.mean()
        sum_result = var.array.sum()
        std_result = var.array.std()

        # All should return valid numbers
        assert isinstance(min_result, (float, np.floating, int))
        assert isinstance(max_result, (float, np.floating, int))
        assert isinstance(mean_result, (float, np.floating, int))
        assert isinstance(sum_result, (float, np.floating, int))
        assert isinstance(std_result, (float, np.floating, int))

        # std should be non-negative
        assert std_result >= 0.0

        # Basic sanity checks
        assert min_result <= mean_result <= max_result
        assert std_result <= (max_result - min_result)  # std can't exceed range


class TestUnitAwareArrayGlobalReductions:
    """Test that UnitAwareArray has all global reduction methods."""

    def test_unit_aware_has_global_methods(self):
        """Verify UnitAwareArray has all global_* methods."""
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        data = np.linspace(1, 10, 100)
        arr = UnitAwareArray(data, units="m")

        global_methods = ["global_max", "global_min", "global_mean", "global_sum", "global_std"]
        for method in global_methods:
            assert hasattr(arr, method), f"Missing {method} on UnitAwareArray"
            assert callable(getattr(arr, method)), f"{method} is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
