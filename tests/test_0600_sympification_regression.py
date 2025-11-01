#!/usr/bin/env python3
"""
Regression tests for SymPy sympification protocol issues.

These tests prevent regressions of critical sympification failures that
occurred with UWexpression objects in coordinate systems and mathematical
operations. All tests are designed to fail fast if the sympification
protocol is broken.

Issues covered:
1. UWexpression sympification in coordinate transformations
2. SymPy automatic sympification in mathematical contexts
3. Boolean evaluation preventing __len__ calls
4. String representation for SymPy internal operations
"""

import pytest
import sympy
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw


class TestUWExpressionSympificationProtocol:
    """Test UWexpression sympification protocol compliance."""

    def test_sympify_protocol_basic(self):
        """Test that UWexpression objects can be sympified by SymPy."""
        # Create a basic UWexpression
        expr = uw.function.expression(r"\alpha", sym=1.5)

        # Direct sympification should work
        sympified = sympy.sympify(expr)
        assert sympified is not None
        assert isinstance(sympified, (sympy.Basic, float, int))

        # The key test: _sympify_() should return internal representation, not self
        internal_sympified = expr._sympify_()
        assert internal_sympified is not expr
        assert isinstance(internal_sympified, (sympy.Basic, float, int))

    def test_sympify_in_coordinate_systems(self):
        """Test UWexpression sympification in coordinate system contexts."""
        # This was the original failing case: spherical coordinates
        # Create expressions like those in coordinate systems
        r_expr = uw.function.expression(r"r", sym=2.0)
        z_expr = uw.function.expression(r"z", sym=1.0)

        # Test automatic sympification in mathematical operations
        # This should not raise SympifyError
        try:
            result = z_expr / r_expr  # SymPy handles this internally
            assert result is not None
        except Exception as e:
            pytest.fail(f"Coordinate expression division failed: {e}")

    def test_sympify_strict_mode(self):
        """Test UWexpression works with SymPy's strict sympify mode."""
        expr = uw.function.expression(r"\beta", sym=3.14)

        # Test strict sympify (this was failing before the fix)
        try:
            strict_result = sympy.sympify(expr, strict=True)
            assert strict_result is not None
        except sympy.SympifyError as e:
            pytest.fail(f"Strict sympify failed: {e}")

    def test_boolean_evaluation_no_len_calls(self):
        """Test that UWexpression boolean evaluation doesn't call __len__."""
        expr = uw.function.expression(r"\gamma", sym=0.0)

        # Boolean evaluation should work without __len__ calls
        # This was causing issues in SymPy internal operations
        bool_result = bool(expr)
        assert isinstance(bool_result, bool)
        assert bool_result == True  # UWexpression should always be truthy

    def test_repr_for_sympify_operations(self):
        """Test that __repr__ returns string suitable for SymPy operations."""
        expr = uw.function.expression(r"\delta", sym=42.0)

        # __repr__ should return the symbol name for SymPy compatibility
        repr_result = repr(expr)
        assert isinstance(repr_result, str)
        assert len(repr_result) > 0

        # Should be the symbol name, not a complex description
        assert r"\delta" in repr_result or "delta" in repr_result


class TestAutomaticSympificationInMathOperations:
    """Test automatic sympification in various mathematical contexts."""

    def test_binary_operations_with_sympify(self):
        """Test that binary operations trigger correct sympification."""
        expr1 = uw.function.expression(r"x", sym=2.0)
        expr2 = uw.function.expression(r"y", sym=3.0)

        # These should all work via SymPy's automatic sympification
        operations = [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
            lambda a, b: a / b,
            lambda a, b: a**2,  # Powers
        ]

        for op in operations:
            try:
                result = op(expr1, expr2)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Binary operation {op} failed: {e}")

    def test_sympify_with_mixed_types(self):
        """Test sympification when mixing UWexpression with other types."""
        expr = uw.function.expression(r"mixed", sym=5.0)

        # Mix with regular numbers
        result1 = expr + 10
        assert result1 is not None

        # Mix with SymPy symbols
        x = sympy.Symbol("x")
        result2 = expr * x
        assert result2 is not None

        # Mix with SymPy expressions
        result3 = expr + sympy.sin(x)
        assert result3 is not None

    def test_function_calls_with_sympification(self):
        """Test SymPy function calls that trigger sympification."""
        expr = uw.function.expression(r"func_arg", sym=1.5)

        # SymPy functions should be able to sympify UWexpression arguments
        try:
            sin_result = sympy.sin(expr)
            cos_result = sympy.cos(expr)
            exp_result = sympy.exp(expr)

            assert all(r is not None for r in [sin_result, cos_result, exp_result])
        except Exception as e:
            pytest.fail(f"SymPy function sympification failed: {e}")


class TestCoordinateSystemRegression:
    """Specific regression tests for coordinate system sympification issues."""

    def test_spherical_coordinate_expressions(self):
        """Test the exact pattern that was failing in spherical coordinates."""
        # Recreate the failing spherical coordinate pattern
        r = uw.function.expression(
            r"r", sym=sympy.sqrt(sympy.Symbol("x") ** 2 + sympy.Symbol("y") ** 2)
        )
        z = uw.function.expression(r"z", sym=sympy.Symbol("z"))

        # This division was causing SympifyError before the fix
        try:
            theta_expr = z / r
            assert theta_expr is not None

            # Should be able to use in further operations
            advanced_expr = sympy.acos(theta_expr)
            assert advanced_expr is not None

        except sympy.SympifyError as e:
            pytest.fail(f"Spherical coordinate expression failed: {e}")

    def test_coordinate_system_creation(self):
        """Test that coordinate systems can be created without sympify errors."""
        # This should not raise any sympification errors
        try:
            # Create a mesh to test coordinate systems
            mesh = uw.meshing.StructuredQuadBox(
                elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1)
            )

            # Access coordinate expressions (this uses sympification internally)
            x, y = mesh.X
            assert x is not None
            assert y is not None

            # Create derived coordinate expressions
            r = sympy.sqrt(x**2 + y**2)
            theta = sympy.atan2(y, x)

            assert r is not None
            assert theta is not None

        except Exception as e:
            pytest.fail(f"Coordinate system creation failed: {e}")


class TestRecursionPreventionRegression:
    """Test that recursion issues in sympification are prevented."""

    def test_no_infinite_recursion_in_atoms(self):
        """Test that atoms() method doesn't cause infinite recursion."""
        expr = uw.function.expression(r"recursive_test", sym=42.0)

        # Set a low recursion limit to catch infinite recursion quickly
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)

        try:
            # This was causing infinite recursion before the fix
            atoms_result = expr.atoms(sympy.Symbol)
            assert isinstance(atoms_result, set)

            # Should be able to call multiple times
            atoms_result2 = expr.atoms(sympy.Symbol)
            assert isinstance(atoms_result2, set)

        except RecursionError:
            pytest.fail("atoms() method caused infinite recursion")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_sympify_does_not_return_self(self):
        """Test that _sympify_ returns internal representation, not self."""
        expr = uw.function.expression(r"self_test", sym=7.5)

        sympified = expr._sympify_()

        # Should NOT return self (this was causing recursion)
        assert sympified is not expr

        # Should return the internal symbolic representation
        assert sympified == expr._sym or sympified == expr.sym


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
