#!/usr/bin/env python3
"""
Test mathematical mixin __len__ method for SymPy compatibility

This test ensures that MeshVariable objects work correctly with SymPy operations
that require len() functionality, specifically the dot product operation.
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import sympy
import underworld3 as uw


def test_mathematical_mixin_len_method():
    """Test that MeshVariable.__len__() works correctly for SymPy compatibility."""

    # Create mesh and variable
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)

    # Test len() works on vector variable
    assert len(velocity) == 2  # 2-component vector
    assert len(velocity.sym) == 2  # Should be same as sym

    # Test scalar variable
    pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1)
    assert len(pressure) == 1  # 1-component scalar
    assert len(pressure.sym) == 1


def test_sympy_dot_product_compatibility():
    """Test that SymPy dot product works with MeshVariable directly."""

    # Create mesh and variable
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)

    # Create a SymPy vector for testing
    unit_vector = sympy.Matrix([1, 0])

    # Test dot product with velocity directly (should work now)
    result1 = unit_vector.dot(velocity)

    # Test dot product with velocity.sym (always worked)
    result2 = unit_vector.dot(velocity.sym)

    # Results should be equivalent
    assert result1.equals(result2)

    # Result should be the first component of velocity
    assert str(result1) == str(velocity[0])


def test_mathematical_mixin_sequence_behavior():
    """Test that MathematicalMixin behaves correctly as a sequence for SymPy."""

    # Create mesh and variables
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)
    temperature = uw.discretisation.MeshVariable("temperature", mesh, 1, degree=1)

    # Test vector variable sequence behavior
    assert len(velocity) == 2
    assert velocity[0] == velocity.sym[0]
    assert velocity[1] == velocity.sym[1]

    # Test scalar variable sequence behavior
    assert len(temperature) == 1
    assert temperature[0] == temperature.sym[0]

    # Test that indexing beyond bounds fails appropriately
    with pytest.raises(IndexError):
        _ = velocity[2]

    with pytest.raises(IndexError):
        _ = temperature[1]


def test_mathematical_operations_equivalence():
    """Test that mathematical operations work the same with and without .sym."""

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)

    # Test various SymPy operations
    x, y = mesh.CoordinateSystem.X

    # Arithmetic operations
    assert (2 * velocity).equals(2 * velocity.sym)
    assert (velocity + velocity).equals(velocity.sym + velocity.sym)
    assert (velocity - velocity).equals(velocity.sym - velocity.sym)

    # Method calls that work on symbolic form
    assert velocity.T.equals(velocity.sym.T)

    # Component access
    assert velocity[0].equals(velocity.sym[0])
    assert velocity[1].equals(velocity.sym[1])


if __name__ == "__main__":
    test_mathematical_mixin_len_method()
    test_sympy_dot_product_compatibility()
    test_mathematical_mixin_sequence_behavior()
    test_mathematical_operations_equivalence()
    print("✅ All mathematical mixin len() tests passed!")
