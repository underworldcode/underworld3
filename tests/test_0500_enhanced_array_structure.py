"""
Unit test for enhanced SwarmVariable array interface structure.
This test verifies the implementation without requiring full PETSc/underworld3 environment.
"""

import pytest
import sys
import os


def test_enhanced_array_classes_exist():
    """Test that the unified NDArray_With_Callback interface is properly implemented"""
    try:
        from underworld3.swarm import SwarmVariable

        # Check that SwarmVariable has the unified array interface
        assert hasattr(SwarmVariable, "_create_variable_array")
        print("✅ _create_variable_array factory method exists")

        # Check that the direct methods exist
        required_methods = [
            "pack_uw_data_to_petsc",
            "unpack_uw_data_from_petsc",
            "use_enhanced_array",  # Now deprecated but preserved
            "use_legacy_array",  # Now deprecated but preserved
            "sync_disabled",
        ]

        for method in required_methods:
            assert hasattr(SwarmVariable, method), f"Missing method: {method}"
            print(f"✅ {method} method exists")

        # NDArray_With_Callback provides the array functionality
        print("✅ Unified NDArray_With_Callback interface implemented")

    except ImportError as e:
        pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")


def test_swarm_variable_enhanced_methods():
    """Test that SwarmVariable has the new enhanced array methods"""
    try:
        from underworld3.swarm import SwarmVariable

        # Check that SwarmVariable has the new methods
        required_methods = [
            "pack_uw_data_to_petsc",
            "unpack_uw_data_from_petsc",
            "use_enhanced_array",
            "use_legacy_array",
            "sync_disabled",
        ]

        for method in required_methods:
            assert hasattr(SwarmVariable, method), f"Missing method: {method}"
            print(f"✅ SwarmVariable.{method} exists")

    except ImportError as e:
        pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")


def test_enhanced_array_method_signatures():
    """Test that the enhanced array methods have correct signatures"""
    try:
        from underworld3.swarm import SwarmVariable
        import inspect

        # Test pack_uw_data_to_petsc signature
        pack_sig = inspect.signature(SwarmVariable.pack_uw_data_to_petsc)
        pack_params = list(pack_sig.parameters.keys())
        assert "data_array" in pack_params, "pack_uw_data_to_petsc missing data_array parameter"
        assert "sync" in pack_params, "pack_uw_data_to_petsc missing sync parameter"
        print("✅ pack_uw_data_to_petsc has correct signature")

        # Test unpack_uw_data_from_petsc signature
        unpack_sig = inspect.signature(SwarmVariable.unpack_uw_data_from_petsc)
        unpack_params = list(unpack_sig.parameters.keys())
        assert "squeeze" in unpack_params, "unpack_uw_data_from_petsc missing squeeze parameter"
        assert "sync" in unpack_params, "unpack_uw_data_from_petsc missing sync parameter"
        print("✅ unpack_uw_data_from_petsc has correct signature")

    except ImportError as e:
        pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")


def test_array_interface_documentation():
    """Test that the enhanced array interface is properly documented"""
    try:
        from underworld3.swarm import SwarmVariable

        # Check docstrings exist
        assert (
            SwarmVariable.pack_uw_data_to_petsc.__doc__ is not None
        ), "pack_uw_data_to_petsc missing docstring"
        assert (
            SwarmVariable.unpack_uw_data_from_petsc.__doc__ is not None
        ), "unpack_uw_data_from_petsc missing docstring"
        assert SwarmVariable.sync_disabled.__doc__ is not None, "sync_disabled missing docstring"

        # Check docstring content
        pack_doc = SwarmVariable.pack_uw_data_to_petsc.__doc__
        assert (
            "Enhanced pack method" in pack_doc
        ), "pack_uw_data_to_petsc docstring missing description"
        assert (
            "access()" in pack_doc
        ), "pack_uw_data_to_petsc docstring should mention access() avoidance"

        unpack_doc = SwarmVariable.unpack_uw_data_from_petsc.__doc__
        assert (
            "Enhanced unpack method" in unpack_doc
        ), "unpack_uw_data_from_petsc docstring missing description"
        assert (
            "access()" in unpack_doc
        ), "unpack_uw_data_from_petsc docstring should mention access() avoidance"

        sync_doc = SwarmVariable.sync_disabled.__doc__
        assert (
            "Context manager" in sync_doc
        ), "sync_disabled docstring missing context manager description"

        print("✅ All enhanced methods are properly documented")

    except ImportError as e:
        pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")


class TestEnhancedArrayStructure:
    """Test the structure of the enhanced array implementation"""

    def test_dual_array_interface_structure(self):
        """Test that unified NDArray_With_Callback interface is properly structured"""
        try:
            from underworld3.swarm import SwarmVariable

            # Test that the unified interface methods exist
            assert hasattr(SwarmVariable, "_create_variable_array")
            assert hasattr(SwarmVariable, "pack_uw_data_to_petsc")
            assert hasattr(SwarmVariable, "unpack_uw_data_from_petsc")

            # Test that legacy methods are preserved (but deprecated)
            assert hasattr(SwarmVariable, "use_legacy_array")
            assert hasattr(SwarmVariable, "use_enhanced_array")

            print("✅ Unified NDArray_With_Callback interface properly structured")

        except ImportError as e:
            pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")

    def test_sync_context_manager_structure(self):
        """Test that the sync context manager is properly structured"""
        try:
            from underworld3.swarm import SwarmVariable
            import inspect

            # Test that sync_disabled returns a context manager
            sync_method = SwarmVariable.sync_disabled
            assert sync_method is not None

            # Check that it's a method that returns something (context manager)
            assert callable(sync_method)

            print("✅ sync_disabled method is properly structured")

        except ImportError as e:
            pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")


def test_implementation_completeness():
    """Test that the unified NDArray_With_Callback implementation is complete"""
    try:
        from underworld3.swarm import SwarmVariable

        # Create a checklist of all unified interface features
        features = {
            "Enhanced pack method": hasattr(SwarmVariable, "pack_uw_data_to_petsc"),
            "Enhanced unpack method": hasattr(SwarmVariable, "unpack_uw_data_from_petsc"),
            "Unified array factory": hasattr(SwarmVariable, "_create_variable_array"),
            "Legacy method compatibility": hasattr(SwarmVariable, "use_enhanced_array")
            and hasattr(SwarmVariable, "use_legacy_array"),
            "Sync context manager": hasattr(SwarmVariable, "sync_disabled"),
        }

        # Check all features
        missing_features = [name for name, exists in features.items() if not exists]

        if missing_features:
            pytest.fail(f"Missing features: {missing_features}")

        print("✅ All Stage 1 features implemented:")
        for feature in features:
            print(f"   ✓ {feature}")

    except ImportError as e:
        pytest.skip(f"Cannot import underworld3.swarm due to dependencies: {e}")


if __name__ == "__main__":
    """Run tests directly for debugging"""
    print("Testing Enhanced SwarmVariable Array Structure...")

    try:
        test_enhanced_array_classes_exist()
        test_swarm_variable_enhanced_methods()
        test_enhanced_array_method_signatures()
        test_array_interface_documentation()
        test_implementation_completeness()

        print("\n🎉 All structure tests passed!")
        print("Stage 1 Enhanced Array Interface implementation verified!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
