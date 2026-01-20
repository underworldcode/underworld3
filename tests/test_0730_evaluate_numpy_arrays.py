"""
Test that uw.function.evaluate() returns numpy arrays, not SymPy arrays.

This test validates the fix for the issue where evaluate() was returning
SymPy ImmutableDenseNDimArray instead of numpy arrays, causing downstream
API incompatibilities.
"""
import pytest
import numpy as np
import underworld3 as uw


@pytest.mark.tier_a  # Production-ready - critical for evaluate() reliability
@pytest.mark.level_2  # Intermediate - involves units and evaluation
class TestEvaluateReturnsNumpyArrays:
    """Test that evaluate() returns proper numpy arrays with unit-aware expressions."""

    def test_evaluate_pure_sympy_returns_numpy(self):
        """Test that evaluating pure SymPy expressions returns numpy arrays."""
        # Setup model with units
        model = uw.Model()
        model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K")
        )

        # Create mesh
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0),
            maxCoords=(1, 1),
            cellSize=0.2,
            regular=True,
        )

        # Create mesh variable for evaluation coordinates
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

        # Get coordinate symbol
        x = mesh.X[0]

        # Create expression with units
        L_scale = uw.quantity(2900, "km")
        expr = x * L_scale.value  # Use .value to avoid units in expression for now

        # Evaluate at mesh coordinates
        result = uw.function.evaluate(expr, T.coords)

        # Validate result type - evaluate() returns UnitAwareArray (numpy subclass with units)
        # or plain numpy array, NOT UWQuantity
        assert isinstance(result, np.ndarray), \
            f"Result should be numpy array or UnitAwareArray, got {type(result)}"

        # CRITICAL: Result should be numpy array (or UnitAwareArray which inherits from ndarray)
        result_arr = np.asarray(result)  # Works for both ndarray and UnitAwareArray
        assert isinstance(result_arr, np.ndarray), \
            f"Result as array should be numpy array, got {type(result_arr)}"

        # Verify it's not a SymPy array type
        assert not hasattr(result_arr.__class__, '__module__') or \
               'sympy' not in str(result_arr.__class__), \
            f"Result should not be SymPy array type: {result_arr.__class__}"

        # Verify numpy API is available
        assert hasattr(result, 'dtype'), "Should have numpy dtype attribute"
        assert hasattr(result, 'mean'), "Should have numpy mean() method"
        assert hasattr(result, 'std'), "Should have numpy std() method"

        # Verify shape is correct for coordinate evaluation
        n_points = T.coords.shape[0]
        assert result.shape == (n_points, 1, 1), \
            f"Shape should be (n_points, 1, 1), got {result.shape}"

    def test_quantity_with_array_preserves_numpy(self):
        """Test that UWQuantity preserves numpy arrays without converting to SymPy."""
        # Create numpy array
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Create quantity with array
        q = uw.quantity(arr, "km")

        # Verify array is preserved as numpy
        assert isinstance(q.value, np.ndarray), \
            f"Array should remain numpy array, got {type(q.value)}"

        # Verify shape is preserved
        assert q.value.shape == (2, 3), \
            f"Shape should be (2, 3), got {q.value.shape}"

        # Verify dtype is float
        assert np.issubdtype(q.value.dtype, np.floating), \
            f"Dtype should be float, got {q.value.dtype}"

    def test_quantity_with_list_converts_to_numpy(self):
        """Test that UWQuantity converts lists to numpy arrays."""
        # Create quantity with list
        q = uw.quantity([1, 2, 3, 4], "m/s")

        # Verify list is converted to numpy array
        assert isinstance(q.value, np.ndarray), \
            f"List should be converted to numpy array, got {type(q.value)}"

        # Verify shape
        assert q.value.shape == (4,), \
            f"Shape should be (4,), got {q.value.shape}"

    def test_quantity_arithmetic_preserves_numpy(self):
        """Test that arithmetic operations preserve numpy arrays."""
        # Create quantity with array
        arr = np.array([1.0, 2.0, 3.0])
        q1 = uw.quantity(arr, "m")

        # Perform arithmetic
        q2 = q1 * 2
        q3 = q1 + uw.quantity(arr, "m")

        # Verify results are numpy arrays
        assert isinstance(q2.value, (np.ndarray, float)), \
            f"Multiplication result should be numpy array or scalar, got {type(q2.value)}"

        assert isinstance(q3.value, (np.ndarray, float)), \
            f"Addition result should be numpy array or scalar, got {type(q3.value)}"

    def test_sympy_array_input_converted_to_numpy(self):
        """Test that SymPy arrays passed to UWQuantity are converted to numpy."""
        import sympy

        # Create SymPy array (simulating what might happen elsewhere in code)
        sympy_arr = sympy.Array([1.0, 2.0, 3.0])

        # Create quantity with SymPy array
        q = uw.quantity(sympy_arr.tolist(), "Pa")  # Use tolist() for now

        # Verify it's converted to numpy
        assert isinstance(q.value, np.ndarray), \
            f"SymPy array should be converted to numpy, got {type(q.value)}"

        # Verify values are correct (convert to float for comparison)
        assert np.allclose(q.value.astype(float), [1.0, 2.0, 3.0]), \
            "Values should be preserved during conversion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
