#!/usr/bin/env python3
"""
Comprehensive unit tests for MathematicalMixin.

This test suite provides full coverage of MathematicalMixin functionality
and checks for inconsistent patterns in the implementation.
"""

import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import sympy
import sys
import os

# Add src to path for testing
# REMOVED: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from underworld3.utilities.mathematical_mixin import MathematicalMixin


class TestMathematicalMixinCore:
    """Test core MathematicalMixin functionality."""

    class MockVariable(MathematicalMixin):
        """Mock variable for testing MathematicalMixin."""

        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def scalar_var(self):
        """Create a scalar variable."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        return self.MockVariable("temperature", scalar_sym)

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    @pytest.fixture
    def matrix_var(self):
        """Create a matrix/tensor variable."""
        x, y = sympy.symbols("x y")
        matrix_sym = sympy.Matrix(
            [
                [sympy.Function("S_00")(x, y), sympy.Function("S_01")(x, y)],
                [sympy.Function("S_10")(x, y), sympy.Function("S_11")(x, y)],
            ]
        )
        return self.MockVariable("stress", matrix_sym)

    def test_sympify_protocol(self, scalar_var, vector_var):
        """Test the _sympify_ protocol implementation."""
        # _sympify_ should return the .sym property
        assert scalar_var._sympify_() == scalar_var.sym
        assert vector_var._sympify_() == vector_var.sym

        # Should be identical objects, not just equal
        assert scalar_var._sympify_() is scalar_var.sym
        assert vector_var._sympify_() is vector_var.sym

    def test_sympify_enables_sympy_operations(self, scalar_var):
        """Test that _sympify_ enables SymPy to work with variables."""
        x = sympy.Symbol("x")

        # SymPy should automatically call _sympify_ when needed
        expr1 = x * scalar_var  # SymPy-initiated operation
        expr2 = x * scalar_var.sym  # Direct operation

        assert expr1.equals(expr2)

        # Test with SymPy functions
        expanded1 = sympy.expand(x * scalar_var)
        expanded2 = sympy.expand(x * scalar_var.sym)
        assert expanded1.equals(expanded2)


class TestComponentAccess:
    """Test component access functionality."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    @pytest.fixture
    def scalar_var(self):
        """Create a scalar variable."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        return self.MockVariable("temperature", scalar_sym)

    def test_vector_component_access(self, vector_var):
        """Test accessing components of vector variables."""
        # Test valid indices
        v0 = vector_var[0]
        v1 = vector_var[1]

        assert v0.equals(vector_var.sym[0])
        assert v1.equals(vector_var.sym[1])

        # Test that results are SymPy expressions
        assert isinstance(v0, sympy.Basic)
        assert isinstance(v1, sympy.Basic)

    def test_scalar_component_access_error(self, scalar_var):
        """Test that scalar variables raise appropriate errors for component access."""
        # Scalar variables should raise errors for component access
        with pytest.raises((IndexError, TypeError)):
            scalar_var[0]

    def test_vector_out_of_bounds_access(self, vector_var):
        """Test out-of-bounds component access."""
        # Should raise IndexError for out-of-bounds access
        with pytest.raises(IndexError):
            vector_var[2]  # Vector only has indices 0, 1

        with pytest.raises(IndexError):
            vector_var[-1]  # Negative indices may not be supported

    def test_component_in_expressions(self, vector_var):
        """Test that components work correctly in expressions."""
        x = sympy.Symbol("x")

        # Components should work in arithmetic
        expr1 = x * vector_var[0]
        expr2 = x * vector_var.sym[0]
        assert expr1.equals(expr2)

        # Components should work with other components
        expr3 = vector_var[0] + vector_var[1]
        expr4 = vector_var.sym[0] + vector_var.sym[1]
        assert expr3.equals(expr4)


class TestArithmeticOperations:
    """Test arithmetic operations implementation."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def scalar_var(self):
        """Create a scalar variable."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        return self.MockVariable("temperature", scalar_sym)

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    def test_multiplication_consistency(self, scalar_var, vector_var):
        """Test multiplication operations for consistency."""
        # Left multiplication
        result1 = scalar_var * 2
        expected1 = scalar_var.sym * 2
        assert result1.equals(expected1)

        result2 = vector_var * 3
        expected2 = vector_var.sym * 3
        assert result2.equals(expected2)

        # Right multiplication
        result3 = 2 * scalar_var
        expected3 = 2 * scalar_var.sym
        assert result3.equals(expected3)

        result4 = 3 * vector_var
        expected4 = 3 * vector_var.sym
        assert result4.equals(expected4)

        # Commutativity check
        assert (scalar_var * 2).equals(2 * scalar_var)
        assert (vector_var * 3).equals(3 * vector_var)

    def test_division_consistency(self, scalar_var, vector_var):
        """Test division operations for consistency."""
        # Left division (always supported)
        result1 = scalar_var / 2
        expected1 = scalar_var.sym / 2
        assert result1.equals(expected1)

        result2 = vector_var / 3
        expected2 = vector_var.sym / 3
        assert result2.equals(expected2)

        # Right division - scalar should work
        result3 = 2 / scalar_var
        expected3 = 2 / scalar_var.sym
        assert result3.equals(expected3)

        # Right division by vector should raise error (unsupported)
        with pytest.raises(TypeError, match="not supported"):
            result4 = 3 / vector_var

    def test_power_operations(self, scalar_var, vector_var):
        """Test power operations."""
        # Left power (raising variable to scalar power)
        result1 = scalar_var**2
        expected1 = scalar_var.sym**2
        assert result1.equals(expected1)

        # Vector power should raise error (non-square matrices can't be raised to powers)
        with pytest.raises(TypeError, match="Cannot raise"):
            result2 = vector_var**2

        # Right power with scalar - should work
        result3 = 2**scalar_var
        expected3 = 2**scalar_var.sym
        assert result3.equals(expected3)

        # Right power with vector exponent should raise error (mathematically undefined)
        with pytest.raises(TypeError, match="not supported"):
            result4 = 2**vector_var

    def test_unary_operations(self, scalar_var, vector_var):
        """Test unary operations."""
        # Negation
        neg_scalar = -scalar_var
        expected_neg_scalar = -scalar_var.sym
        assert neg_scalar.equals(expected_neg_scalar)

        neg_vector = -vector_var
        expected_neg_vector = -vector_var.sym
        assert neg_vector.equals(expected_neg_vector)

        # Positive (identity) - works for scalar
        pos_scalar = +scalar_var
        expected_pos_scalar = +scalar_var.sym
        assert pos_scalar.equals(expected_pos_scalar)

        # Positive for vector should return the vector itself (our fallback behavior)
        pos_vector = +vector_var
        # Our implementation returns the sym itself when unary + fails
        assert pos_vector.equals(vector_var.sym)


class TestScalarBroadcasting:
    """Test scalar broadcasting functionality."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    @pytest.fixture
    def scalar_var(self):
        """Create a scalar variable (no broadcasting needed)."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        return self.MockVariable("temperature", scalar_sym)

    def test_addition_scalar_broadcasting(self, vector_var):
        """Test scalar broadcasting for addition."""
        # Adding scalar to vector should broadcast
        result = vector_var + 5

        # Should broadcast 5 to vector shape
        expected = vector_var.sym + 5 * sympy.ones(*vector_var.sym.shape)
        assert result.equals(expected)

        # Right addition should also work
        result_right = 5 + vector_var
        assert result_right.equals(expected)

    def test_subtraction_scalar_broadcasting(self, vector_var):
        """Test scalar broadcasting for subtraction."""
        # Subtracting scalar from vector
        result = vector_var - 3
        expected = vector_var.sym - 3 * sympy.ones(*vector_var.sym.shape)
        assert result.equals(expected)

        # Right subtraction
        result_right = 3 - vector_var
        expected_right = 3 * sympy.ones(*vector_var.sym.shape) - vector_var.sym
        assert result_right.equals(expected_right)

    def test_no_broadcasting_for_scalars(self, scalar_var):
        """Test that scalar variables don't trigger broadcasting."""
        # Scalar + scalar should use normal SymPy addition
        result = scalar_var + 5
        expected = scalar_var.sym + 5
        assert result.equals(expected)

        # Should not create a matrix
        assert not hasattr(result, "shape") or getattr(result, "shape", None) is None

    def test_broadcasting_error_conditions(self, vector_var):
        """Test error conditions in broadcasting."""
        # Test what happens with incompatible operations
        # This tests the fallback behavior in the except blocks

        # These should still work via normal SymPy operations
        other_vector = sympy.Matrix([1, 2])
        result = vector_var + other_vector  # Should work via SymPy
        expected = vector_var.sym + other_vector
        assert result.equals(expected)


class TestMethodDelegation:
    """Test __getattr__ method delegation functionality."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    @pytest.fixture
    def matrix_var(self):
        """Create a matrix variable."""
        x, y = sympy.symbols("x y")
        matrix_sym = sympy.Matrix(
            [
                [sympy.Function("S_00")(x, y), sympy.Function("S_01")(x, y)],
                [sympy.Function("S_10")(x, y), sympy.Function("S_11")(x, y)],
            ]
        )
        return self.MockVariable("stress", matrix_sym)

    def test_property_delegation(self, vector_var, matrix_var):
        """Test delegation of properties."""
        # Test shape property
        assert vector_var.shape == vector_var.sym.shape
        assert matrix_var.shape == matrix_var.sym.shape

        # Test transpose property
        result_T = vector_var.T
        expected_T = vector_var.sym.T
        assert result_T.equals(expected_T)

        result_T_matrix = matrix_var.T
        expected_T_matrix = matrix_var.sym.T
        assert result_T_matrix.equals(expected_T_matrix)

    def test_method_delegation_simple(self, vector_var):
        """Test delegation of methods without arguments."""
        # Test norm method - our implementation provides default argument (2-norm)
        result_norm = vector_var.norm()
        expected_norm = vector_var.sym.norm(2)  # Our default
        assert result_norm.equals(expected_norm)

        # Test methods that return new expressions
        result_simplify = vector_var.simplify()
        # Since simplify() modifies in-place and returns None for the direct call,
        # our delegation should return the modified sym object
        assert result_simplify is not None
        assert hasattr(result_simplify, "equals")  # Should be a SymPy object
        # The result should be the same as the original since these expressions are already simple
        assert result_simplify.equals(vector_var.sym)

    def test_method_delegation_with_args(self, vector_var):
        """Test delegation of methods with arguments."""
        x = sympy.Symbol("x")

        # Test diff method (should work via delegation too)
        result_diff = vector_var.diff(x)
        expected_diff = vector_var.sym.diff(x)
        assert result_diff.equals(expected_diff)

        # Test subs method
        result_subs = vector_var.subs(x, 1)
        expected_subs = vector_var.sym.subs(x, 1)
        assert result_subs.equals(expected_subs)

    def test_method_delegation_with_mathematical_mixin_args(self, vector_var):
        """Test delegation with MathematicalMixin objects as arguments."""
        # Create another vector for testing
        x, y = sympy.symbols("x y")
        other_sym = sympy.Matrix([sympy.Function("U_0")(x, y), sympy.Function("U_1")(x, y)])
        other_var = self.MockVariable("other", other_sym)

        # Test dot product - should convert arguments automatically
        result_dot = vector_var.dot(other_var)
        expected_dot = vector_var.sym.dot(other_var.sym)
        assert result_dot.equals(expected_dot)

        # Test cross product requires 3D vectors, so create 3D versions
        vector_3d_sym = sympy.Matrix(
            [sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y), sympy.Function("V_2")(x, y)]
        )
        other_3d_sym = sympy.Matrix(
            [sympy.Function("U_0")(x, y), sympy.Function("U_1")(x, y), sympy.Function("U_2")(x, y)]
        )
        vector_3d = self.MockVariable("vector_3d", vector_3d_sym)
        other_3d = self.MockVariable("other_3d", other_3d_sym)

        # Test cross product with 3D vectors
        if hasattr(vector_3d.sym, "cross"):
            result_cross = vector_3d.cross(other_3d)
            expected_cross = vector_3d.sym.cross(other_3d.sym)
            assert result_cross.equals(expected_cross)

    def test_method_delegation_mixed_args(self, vector_var):
        """Test delegation with mixed argument types."""
        x = sympy.Symbol("x")
        other_matrix = sympy.Matrix([1, 2])

        # Method with both regular and MathematicalMixin arguments
        # This tests the argument conversion logic
        result = vector_var.dot(other_matrix)
        expected = vector_var.sym.dot(other_matrix)
        assert result.equals(expected)

    def test_attribute_error_for_nonexistent(self, vector_var):
        """Test that nonexistent attributes raise AttributeError."""
        with pytest.raises(AttributeError):
            vector_var.nonexistent_method()

        with pytest.raises(AttributeError):
            vector_var.nonexistent_property


class TestDisplayAndRepresentation:
    """Test display and representation methods."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def scalar_var(self):
        """Create a scalar variable."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        return self.MockVariable("temperature", scalar_sym)

    def test_repr_method(self, scalar_var):
        """Test __repr__ method."""
        var_repr = repr(scalar_var)
        sym_repr = repr(scalar_var.sym)

        # Should return the symbolic representation
        assert var_repr == sym_repr

    def test_latex_representation(self, scalar_var):
        """Test _repr_latex_ method."""
        latex_repr = scalar_var._repr_latex_()

        # Should be wrapped in $$ for LaTeX
        assert latex_repr.startswith("$$")
        assert latex_repr.endswith("$$")

        # Should contain LaTeX representation of the symbol
        from sympy import latex

        expected_latex = latex(scalar_var.sym)
        assert expected_latex in latex_repr

    def test_ipython_display(self, scalar_var):
        """Test _ipython_display_ method."""
        # This method calls IPython display functions
        # We can't easily test the actual display, but we can test it doesn't crash
        try:
            # This might fail if IPython is not available, which is OK
            scalar_var._ipython_display_()
        except ImportError:
            # Expected if IPython is not available
            pass
        except Exception as e:
            # Should not raise other exceptions
            pytest.fail(f"_ipython_display_ raised unexpected exception: {e}")

    def test_sym_repr_method(self, scalar_var):
        """Test sym_repr method."""
        sym_repr = scalar_var.sym_repr()
        expected = str(scalar_var.sym)

        assert sym_repr == expected


class TestDifferentiation:
    """Test differentiation functionality."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def scalar_var(self):
        """Create a scalar variable."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        return self.MockVariable("temperature", scalar_sym)

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    def test_direct_diff_method(self, scalar_var, vector_var):
        """Test the direct diff method."""
        x, y = sympy.symbols("x y")

        # Test scalar differentiation
        result_scalar = scalar_var.diff(x)
        expected_scalar = scalar_var.sym.diff(x)
        assert result_scalar.equals(expected_scalar)

        # Test vector differentiation
        result_vector = vector_var.diff(x)
        expected_vector = vector_var.sym.diff(x)
        assert result_vector.equals(expected_vector)

        # Test partial differentiation
        result_partial = scalar_var.diff(x, y)
        expected_partial = scalar_var.sym.diff(x, y)
        assert result_partial.equals(expected_partial)

    def test_diff_with_keyword_args(self, scalar_var):
        """Test diff method with keyword arguments."""
        x = sympy.Symbol("x")

        # Test with keyword arguments
        result = scalar_var.diff(x, evaluate=False)
        expected = scalar_var.sym.diff(x, evaluate=False)
        assert result.equals(expected)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    class BrokenSymVariable(MathematicalMixin):
        """Variable with broken sym property for testing."""

        def __init__(self):
            pass

        @property
        def sym(self):
            raise RuntimeError("Broken sym property")

    def test_operations_with_broken_sym(self):
        """Test behavior when sym property is broken."""
        broken_var = self.BrokenSymVariable()

        # Operations should fail gracefully
        with pytest.raises(RuntimeError):
            broken_var + 1

        with pytest.raises(RuntimeError):
            2 * broken_var

        with pytest.raises(RuntimeError):
            broken_var[0]

    def test_operations_with_none_sym(self):
        """Test behavior when sym is None."""
        none_var = self.MockVariable("none", None)

        # Should raise ValueError when sym is None (our validation catches this)
        with pytest.raises(ValueError, match="variable not properly initialized"):
            none_var + 1

        with pytest.raises(ValueError, match="variable not properly initialized"):
            none_var[0]

    def test_complex_nested_operations(self):
        """Test complex nested operations."""
        x, y = sympy.symbols("x y")

        # Create multiple variables
        scalar1 = self.MockVariable("T1", sympy.Function("T1")(x, y))
        scalar2 = self.MockVariable("T2", sympy.Function("T2")(x, y))
        vector1 = self.MockVariable(
            "V1", sympy.Matrix([sympy.Function("V1_0")(x, y), sympy.Function("V1_1")(x, y)])
        )

        # Complex nested expression with proper dimensionality
        # (scalar1 + scalar2) gives scalar, * vector1 gives vector
        vector_expr = (scalar1 + scalar2) * vector1
        # Adding another vector of same shape
        vector2 = 2 * scalar1 * sympy.ones(*vector1.sym.shape)  # Make it vector-shaped
        expr = vector_expr + vector2

        # Should be a valid SymPy expression (matrices inherit from MatrixBase, not Basic directly)
        assert isinstance(expr, (sympy.Basic, sympy.MatrixBase))

        # Should be differentiable
        diff_expr = sympy.diff(expr, x)
        assert isinstance(diff_expr, (sympy.Basic, sympy.MatrixBase))


class TestConsistencyPatterns:
    """Test for consistency patterns and potential issues."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    @pytest.fixture
    def vector_var(self):
        """Create a vector variable."""
        x, y = sympy.symbols("x y")
        vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
        return self.MockVariable("velocity", vector_sym)

    def test_arithmetic_vs_delegation_consistency(self, vector_var):
        """Test consistency between explicit arithmetic and delegation."""
        # Multiplication should be consistent between explicit and delegated
        explicit_mul = vector_var.__mul__(2)
        delegated_mul = vector_var * 2
        assert explicit_mul.equals(delegated_mul)

        # Addition should be consistent
        explicit_add = vector_var.__add__(1)
        delegated_add = vector_var + 1
        assert explicit_add.equals(delegated_add)

    def test_left_vs_right_operations_consistency(self, vector_var):
        """Test consistency between left and right operations."""
        # For commutative operations, left and right should give same result
        left_mul = vector_var * 3
        right_mul = 3 * vector_var
        assert left_mul.equals(right_mul)

        # Addition should be commutative
        left_add = vector_var + 5
        right_add = 5 + vector_var
        assert left_add.equals(right_add)

        # Division should NOT be commutative (this tests correct implementation)
        left_div = vector_var / 2
        # Right division by matrix should raise error (division by matrix not supported)
        with pytest.raises(TypeError, match="not supported"):
            right_div = 2 / vector_var

    def test_broadcasting_vs_normal_operations(self, vector_var):
        """Test that broadcasting and normal operations are consistent where applicable."""
        # When broadcasting is not needed, should get same result as normal operation
        other_vector = sympy.Matrix([1, 1])  # Compatible shape

        # This should use normal SymPy addition, not broadcasting
        result = vector_var + other_vector
        expected = vector_var.sym + other_vector
        assert result.equals(expected)

    def test_method_delegation_vs_direct_calls(self, vector_var):
        """Test that delegated methods give same results as direct calls."""
        # Test that delegation gives same result as direct call
        delegated_norm = vector_var.norm()
        direct_norm = vector_var.sym.norm()
        assert delegated_norm.equals(direct_norm)

        # Test with arguments
        x = sympy.Symbol("x")
        delegated_diff = vector_var.diff(x)
        direct_diff = vector_var.sym.diff(x)
        assert delegated_diff.equals(direct_diff)

    def test_sympify_vs_explicit_sym_access(self, vector_var):
        """Test that _sympify_ and .sym give same results."""
        # These should be identical
        sympify_result = vector_var._sympify_()
        sym_result = vector_var.sym

        assert sympify_result is sym_result  # Should be same object
        assert sympify_result.equals(sym_result)


class TestPerformanceAndMemory:
    """Test performance-related patterns."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value
            self._call_count = 0

        @property
        def sym(self):
            self._call_count += 1
            return self._sym_value

        @sym.setter
        def sym(self, value):
            self._sym_value = value

    def test_sympify_caching(self):
        """Test that _sympify_ doesn't create unnecessary objects."""
        x, y = sympy.symbols("x y")
        scalar_sym = sympy.Function("T")(x, y)
        var = self.MockVariable("test", scalar_sym)

        # Multiple calls to _sympify_ should return same object
        result1 = var._sympify_()
        result2 = var._sympify_()

        assert result1 is result2  # Same object reference
        assert var._call_count >= 1  # sym was accessed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
