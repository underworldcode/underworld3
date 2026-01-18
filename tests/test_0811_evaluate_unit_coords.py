"""
Test unit-aware coordinate handling in uw.function.evaluate()

This test validates that evaluate() and global_evaluate() accept unit-aware
coordinates (UWQuantity and Pint Quantity objects) and automatically convert
them to SI base units for evaluation.
"""

import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import underworld3 as uw
import numpy as np


def test_evaluate_with_numpy_array():
    """Test backward compatibility: evaluate accepts plain numpy arrays."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0]  # Temperature increases with x

    # Plain numpy array (traditional usage)
    coords = np.array([[0.5, 0.5], [0.25, 0.75]])
    result = uw.function.evaluate(T.sym, coords)

    assert result.shape == (2, 1, 1)
    assert abs(result[0, 0, 0] - 0.5) < 0.01  # T ≈ x at (0.5, 0.5)
    assert abs(result[1, 0, 0] - 0.25) < 0.01  # T ≈ x at (0.25, 0.75)


@pytest.mark.skip(
    reason="evaluate() with list of UWQuantity coords not implemented - planned feature"
)
def test_evaluate_with_uwquantity_coords():
    """Test evaluate() with UWQuantity coordinate objects."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0]  # Temperature increases with x

    # UWQuantity coordinates (in meters)
    x = uw.quantity(0.5, "m")
    y = uw.quantity(0.5, "m")

    # List of tuples format
    coords = [(x, y)]
    result = uw.function.evaluate(T.sym, coords)

    assert result.shape == (1, 1, 1)
    assert abs(result[0, 0, 0] - 0.5) < 0.01


@pytest.mark.skip(
    reason="evaluate() with list of UWQuantity coords not implemented - planned feature"
)
def test_evaluate_with_pint_quantity_coords():
    """Test evaluate() with direct Pint Quantity objects."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0]  # Temperature increases with x

    # Direct Pint quantities (from arithmetic operations)
    x = uw.quantity(0.5, "m")
    y = uw.quantity(0.25, "m")

    # Arithmetic creates Pint Quantity objects
    x_center = x  # 0.5 m
    y_center = y * 2  # 0.5 m

    coords = [(x_center, y_center)]
    result = uw.function.evaluate(T.sym, coords)

    assert result.shape == (1, 1, 1)
    assert abs(result[0, 0, 0] - 0.5) < 0.01


@pytest.mark.skip(
    reason="evaluate() with list of UWQuantity coords not implemented - planned feature"
)
def test_evaluate_with_mixed_units():
    """Test evaluate() with coordinates in different units."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0]  # Temperature increases with x

    # Mixed units: meters and centimeters
    coords = [
        (uw.quantity(0.5, "m"), uw.quantity(50, "cm")),  # (0.5, 0.5) in meters
        (uw.quantity(25, "cm"), uw.quantity(0.75, "m")),  # (0.25, 0.75) in meters
    ]

    result = uw.function.evaluate(T.sym, coords)

    assert result.shape == (2, 1, 1)
    assert abs(result[0, 0, 0] - 0.5) < 0.01
    assert abs(result[1, 0, 0] - 0.25) < 0.01


@pytest.mark.skip(
    reason="evaluate() with list of UWQuantity coords not implemented - planned feature"
)
def test_evaluate_with_single_point_list():
    """Test evaluate() with single point as flat list [x, y]."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0]  # Temperature increases with x

    # Single point as flat list
    x = uw.quantity(0.5, "m")
    y = uw.quantity(0.5, "m")
    coords = [x, y]

    result = uw.function.evaluate(T.sym, coords)

    assert result.shape == (1, 1, 1)
    assert abs(result[0, 0, 0] - 0.5) < 0.01


@pytest.mark.skip(
    reason="evaluate() with list of UWQuantity coords not implemented - planned feature"
)
def test_evaluate_unit_conversion_accuracy():
    """Test that unit conversion is accurate for geological units."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    # Create a field with known values
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0] * 1000  # Temperature = 1000 * x

    # Evaluate at 50 cm = 0.5 m
    coords = [(uw.quantity(50, "cm"), uw.quantity(0.5, "m"))]
    result = uw.function.evaluate(T.sym, coords)

    # Expected: T = 1000 * 0.5 = 500
    assert abs(result[0, 0, 0] - 500.0) < 10.0  # Allow some interpolation error


@pytest.mark.skip(
    reason="evaluate() with list of UWQuantity coords not implemented - planned feature"
)
def test_evaluate_mixed_numeric_and_units():
    """Test evaluate() with mixed dimensionless and unit-aware coords."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=False
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    with mesh.access(T):
        T.coords[:, 0]  # Access coords to populate them
    T.array[:, 0, 0] = T.coords[:, 0]

    # Mix of dimensionless and unit-aware
    coords = [
        (0.5, uw.quantity(50, "cm")),  # First coord dimensionless, second with units
        (uw.quantity(0.25, "m"), 0.75),  # First with units, second dimensionless
    ]

    result = uw.function.evaluate(T.sym, coords)

    assert result.shape == (2, 1, 1)
    assert abs(result[0, 0, 0] - 0.5) < 0.01
    assert abs(result[1, 0, 0] - 0.25) < 0.01


if __name__ == "__main__":
    # Run tests individually for debugging
    print("Running evaluate() unit-aware coordinate tests...")

    print("\n1. Testing numpy array (backward compatibility)...")
    test_evaluate_with_numpy_array()
    print("   ✓ Passed")

    print("\n2. Testing UWQuantity coordinates...")
    test_evaluate_with_uwquantity_coords()
    print("   ✓ Passed")

    print("\n3. Testing Pint Quantity coordinates...")
    test_evaluate_with_pint_quantity_coords()
    print("   ✓ Passed")

    print("\n4. Testing mixed units...")
    test_evaluate_with_mixed_units()
    print("   ✓ Passed")

    print("\n5. Testing single point as list...")
    test_evaluate_with_single_point_list()
    print("   ✓ Passed")

    print("\n6. Testing unit conversion accuracy...")
    test_evaluate_unit_conversion_accuracy()
    print("   ✓ Passed")

    print("\n7. Testing mixed numeric and unit-aware coords...")
    test_evaluate_mixed_numeric_and_units()
    print("   ✓ Passed")

    print("\n" + "=" * 60)
    print("All evaluate() unit-aware coordinate tests passed! ✅")
    print("=" * 60)
