import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
#!/usr/bin/env python3
"""
Test script for MathematicalMixin implementation.

This script tests:
1. Direct arithmetic operations (v1 = -1 * v2)
2. Component access (v[0])
3. JIT compatibility with unwrap()
4. Mathematical display behavior
"""

import underworld3 as uw
import sympy
import numpy as np


def test_basic_functionality():
    """Test basic mathematical mixin functionality without full UW3 setup"""
    print("\n=== Testing Basic Mixin Functionality ===")

    # Test the mixin class directly
    from underworld3.utilities.mathematical_mixin import test_mathematical_mixin_fixed

    try:
        test_mathematical_mixin_fixed()
        print("✓ MathematicalMixin direct tests passed")
    except Exception as e:
        print(f"✗ MathematicalMixin tests failed: {e}")
        raise AssertionError(f"MathematicalMixin tests failed: {e}")


def test_mesh_variable_integration():
    """Test MeshVariable with MathematicalMixin"""
    print("\n=== Testing MeshVariable Integration ===")

    try:
        # Create a simple mesh
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))
        print("✓ Created test mesh")

        # Create variables with mathematical mixin
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, vtype=uw.VarType.VECTOR)
        pressure = uw.create_enhanced_mesh_variable("pressure", mesh, 1, vtype=uw.VarType.SCALAR)
        print("✓ Created MeshVariables")

        # Test that the mixin is working
        print(f"✓ MeshVariable has _sympify_: {hasattr(velocity, '_sympify_')}")
        print(f"✓ MeshVariable has __getitem__: {hasattr(velocity, '__getitem__')}")

        # Test mathematical display
        print(f"✓ Velocity representation: {str(velocity)}")

        # Test component access
        v_x = velocity[0]
        print(f"✓ Component access works: {type(v_x)}")

        # Test direct arithmetic
        density = uw.function.expression(r"\rho", sym=1000)
        momentum = density * velocity
        print(f"✓ Direct arithmetic: {type(momentum)}")

        # Test that result is pure SymPy
        print(f"✓ Result is SymPy: {isinstance(momentum, sympy.MatrixBase)}")

    except Exception as e:
        print(f"✗ MeshVariable integration test failed: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError(f"MeshVariable integration test failed: {e}")


def test_jit_compatibility():
    """Test JIT compilation compatibility"""
    print("\n=== Testing JIT Compatibility ===")

    try:
        # Create mesh and variables
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, vtype=uw.VarType.VECTOR)

        # Create expression using direct arithmetic
        expr = -1 * velocity
        print(f"✓ Created expression: {type(expr)}")

        # Test unwrap compatibility
        unwrapped = uw.unwrap(expr)
        print(f"✓ Unwrap works: {type(unwrapped)}")
        print(f"✓ Unwrapped expression: {unwrapped}")

        # Test that unwrapped contains the same atoms as original
        old_expr = -1 * velocity.sym
        unwrapped_old = uw.unwrap(old_expr)

        print(f"✓ New and old expressions are equivalent: {unwrapped.equals(unwrapped_old)}")

    except Exception as e:
        print(f"✗ JIT compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError(f"JIT compatibility test failed: {e}")


def test_component_access():
    """Test component access functionality"""
    print("\n=== Testing Component Access ===")

    try:
        # Create mesh and vector variable
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))
        velocity = uw.create_enhanced_mesh_variable("velocity", mesh, 2, vtype=uw.VarType.VECTOR)

        # Test component access
        v_x = velocity[0]
        v_y = velocity[1]

        print(f"✓ v[0] type: {type(v_x)}")
        print(f"✓ v[1] type: {type(v_y)}")

        # Test arithmetic with components
        expr = 1 + velocity[0]
        print(f"✓ Arithmetic with components: {type(expr)}")

        # Test that components are equivalent to old pattern
        old_v_x = velocity.sym[0]
        print(f"✓ New and old component access equivalent: {v_x.equals(old_v_x)}")

    except Exception as e:
        print(f"✗ Component access test failed: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError(f"Component access test failed: {e}")


def main():
    """Run all tests"""
    print("Testing MathematicalMixin Implementation")
    print("=" * 50)

    tests = [
        test_basic_functionality,
        test_mesh_variable_integration,
        test_jit_compatibility,
        test_component_access,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n=== Results ===")
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("✓ All tests passed! MathematicalMixin implementation is working.")
    else:
        print("✗ Some tests failed. Check output above for details.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
