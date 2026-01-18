"""
Unit tests for automatic lambdification optimization paths.

This test suite documents the different evaluation paths available in UW3:
1. Pure sympy expressions (lambdified - fast)
2. SymPy functions like erf, sin, cos (lambdified - fast)
3. UW3 MeshVariable symbols (RBF interpolation - correct for data)
4. UWexpression parameters (automatic substitution + lambdification)
5. Mesh coordinates (BaseScalar - lambdified)
6. rbf flag behavior (should not affect pure sympy optimization)

Performance expectations:
- Lambdified path: ~0.001-0.01s for 100 points (after compilation)
- RBF path: ~0.01-0.1s for 100 points (interpolation overhead)
- Old path (bypassed): ~1-10s for 100 points (SLOW - should never happen)

Created: 2025-11-17
Purpose: Document optimization paths and prevent regressions
"""

import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np
import sympy
import underworld3 as uw
from scipy import special


@pytest.fixture
def setup_mesh():
    """Create a simple mesh for testing."""
    uw.reset_default_model()
    uw.use_nondimensional_scaling(False)
    uw.use_strict_units(False)  # Allow units without reference quantities for testing
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
        regular=False
    )
    return mesh


@pytest.fixture
def sample_points():
    """Create sample evaluation points."""
    return np.array([
        [0.0, 0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ])


class TestPureSympyExpressions:
    """Test pure sympy expressions use lambdification."""

    def test_simple_polynomial(self, sample_points):
        """Simple polynomial should be lambdified."""
        x = sympy.Symbol('x')
        expr = x**2 + 2*x + 1

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        expected = sample_points[:, 0]**2 + 2*sample_points[:, 0] + 1
        assert np.allclose(result.flatten(), expected)

    def test_multiple_variables(self, sample_points):
        """Expression with multiple symbols should be lambdified."""
        x, y = sympy.symbols('x y')
        expr = x**2 + y**2

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        expected = sample_points[:, 0]**2 + sample_points[:, 1]**2
        assert np.allclose(result.flatten(), expected)

    def test_constant_expression(self, sample_points):
        """Constant expression should be optimized."""
        expr = sympy.sympify(3.14)

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify all values are the constant
        assert np.allclose(result.flatten(), 3.14)


class TestSympyFunctions:
    """Test SymPy built-in functions use lambdification."""

    def test_erf_function(self, setup_mesh, sample_points):
        """erf() should be lambdified (not rejected as UW3 Function)."""
        x = setup_mesh.X[0]
        expr = sympy.erf(5 * x - 2) / 2

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness against scipy
        expected = special.erf(5 * sample_points[:, 0] - 2) / 2
        assert np.allclose(result.flatten(), expected)

    def test_trigonometric_functions(self, setup_mesh, sample_points):
        """sin() and cos() should be lambdified."""
        x, y = setup_mesh.X
        expr = sympy.sin(2*sympy.pi*x) * sympy.cos(2*sympy.pi*y)

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        expected = (np.sin(2*np.pi*sample_points[:, 0]) *
                   np.cos(2*np.pi*sample_points[:, 1]))
        assert np.allclose(result.flatten(), expected)

    def test_exponential_function(self, setup_mesh, sample_points):
        """exp() should be lambdified."""
        x, y = setup_mesh.X
        expr = sympy.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01)

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        expected = np.exp(-((sample_points[:, 0] - 0.5)**2 +
                           (sample_points[:, 1] - 0.5)**2) / 0.01)
        assert np.allclose(result.flatten(), expected)


class TestMeshCoordinates:
    """Test mesh coordinates (BaseScalar) use lambdification."""

    def test_mesh_coordinates_simple(self, setup_mesh, sample_points):
        """Simple mesh coordinate expression should be lambdified."""
        x, y = setup_mesh.X
        expr = x**2 + y**2

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        expected = sample_points[:, 0]**2 + sample_points[:, 1]**2
        assert np.allclose(result.flatten(), expected)

    @pytest.mark.xfail(reason="UWCoordinate hash collision causes SymPy subs() issues when multiple meshes exist - architectural limitation")
    def test_mesh_coordinates_complex(self, setup_mesh, sample_points):
        """Complex mesh coordinate expression should be lambdified.

        NOTE: This test can fail when run after other tests that create meshes.
        The issue is that UWCoordinate objects from different meshes have the same
        hash (based on underlying BaseScalar name like "N.x"), causing SymPy's
        internal caching to substitute wrong coordinate objects.
        """
        x, y = setup_mesh.X
        # Use a simpler expression that's easier to verify
        expr = sympy.sqrt(x**2 + y**2) + sympy.sin(x)

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        r = np.sqrt(sample_points[:, 0]**2 + sample_points[:, 1]**2)
        expected = r + np.sin(sample_points[:, 0])
        assert np.allclose(result.flatten(), expected)


class TestUW3MeshVariables:
    """Test UW3 MeshVariable symbols use RBF interpolation (not lambdified)."""

    def test_mesh_variable_symbol(self, setup_mesh, sample_points):
        """UW3 MeshVariable should use RBF path, not lambdification."""
        # Create a MeshVariable with data
        T = uw.discretisation.MeshVariable("T", setup_mesh, 1, degree=2)
        T.array[...] = 300.0

        # Expression using MeshVariable symbol
        expr = T.sym[0]

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Should get the interpolated value (approximately 300)
        assert np.allclose(result.flatten(), 300.0, atol=1.0)

    def test_mesh_variable_in_expression(self, setup_mesh, sample_points):
        """Expression mixing MeshVariable and coordinates should use RBF."""
        T = uw.discretisation.MeshVariable("T", setup_mesh, 1, degree=2)
        T.array[...] = 100.0

        x = setup_mesh.X[0]
        # Mixed expression: UW3 variable + coordinate
        expr = T.sym[0] + x**2

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Should get T value (~100) plus x**2
        expected_approx = 100.0 + sample_points[:, 0]**2
        assert np.allclose(result.flatten(), expected_approx, atol=1.0)


class TestUWexpressionParameters:
    """Test UWexpression symbols are automatically substituted."""

    def test_uwexpression_numeric(self, setup_mesh, sample_points):
        """UWexpression with numeric value should be substituted and lambdified."""
        alpha = uw.function.expression(r'\alpha', sym=0.1, description="coefficient")
        x = setup_mesh.X[0]
        expr = alpha * x**2

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify alpha was substituted
        expected = 0.1 * sample_points[:, 0]**2
        assert np.allclose(result.flatten(), expected)

    def test_uwexpression_in_sympy_function(self, setup_mesh, sample_points):
        """UWexpression in SymPy function should work correctly."""
        beta = uw.function.expression(r'\beta', sym=2.5, description="scaling")
        x = setup_mesh.X[0]
        expr = beta * sympy.sin(sympy.pi * x)

        result = uw.function.evaluate(expr, sample_points, rbf=True)

        # Verify correctness
        expected = 2.5 * np.sin(np.pi * sample_points[:, 0])
        assert np.allclose(result.flatten(), expected)


class TestRBFFlagBehavior:
    """Test that rbf flag doesn't affect pure sympy optimization."""

    def test_rbf_false_pure_sympy(self, setup_mesh, sample_points):
        """Pure sympy should be lambdified even with rbf=False."""
        x = setup_mesh.X[0]
        expr = sympy.erf(5 * x - 2) / 2

        # Both should use lambdification and give same results
        result_rbf_false = uw.function.evaluate(expr, sample_points, rbf=False)
        result_rbf_true = uw.function.evaluate(expr, sample_points, rbf=True)

        # Results should be identical
        assert np.allclose(result_rbf_false.flatten(), result_rbf_true.flatten())

        # Verify correctness
        expected = special.erf(5 * sample_points[:, 0] - 2) / 2
        assert np.allclose(result_rbf_false.flatten(), expected)

    @pytest.mark.xfail(reason="DMInterpolationSetUp_UW error 98 when run after other mesh tests - state pollution issue")
    def test_rbf_false_mesh_variable(self, setup_mesh, sample_points):
        """MeshVariable with rbf=False should still work (use RBF).

        NOTE: This test can fail when run after other tests that create meshes.
        The PETSc DMInterpolation setup fails with error 98, likely due to
        state pollution from previous mesh instances.
        """
        T = uw.discretisation.MeshVariable("T", setup_mesh, 1, degree=2)
        T.array[...] = 200.0

        expr = T.sym[0]

        # Both should use RBF interpolation
        result_rbf_false = uw.function.evaluate(expr, sample_points, rbf=False)
        result_rbf_true = uw.function.evaluate(expr, sample_points, rbf=True)

        # Results should be similar (both using RBF)
        assert np.allclose(result_rbf_false.flatten(), result_rbf_true.flatten(), atol=0.1)
        assert np.allclose(result_rbf_false.flatten(), 200.0, atol=1.0)


class TestDetectionMechanism:
    """Test the is_pure_sympy_expression detection logic."""

    def test_detection_pure_sympy(self):
        """Pure sympy expression should be detected correctly."""
        from underworld3.function.pure_sympy_evaluator import is_pure_sympy_expression

        x = sympy.Symbol('x')
        expr = x**2 + 1

        is_pure, symbols, sym_type = is_pure_sympy_expression(expr)

        assert is_pure is True
        assert sym_type == 'symbol'

    def test_detection_mesh_coordinates(self, setup_mesh):
        """Mesh coordinates should be detected as pure (BaseScalar)."""
        from underworld3.function.pure_sympy_evaluator import is_pure_sympy_expression

        x = setup_mesh.X[0]
        expr = x**2

        is_pure, symbols, sym_type = is_pure_sympy_expression(expr)

        assert is_pure is True
        assert sym_type == 'coordinate'

    def test_detection_sympy_function(self, setup_mesh):
        """SymPy function (erf) should be detected as pure."""
        from underworld3.function.pure_sympy_evaluator import is_pure_sympy_expression

        x = setup_mesh.X[0]
        expr = sympy.erf(x)

        is_pure, symbols, sym_type = is_pure_sympy_expression(expr)

        assert is_pure is True
        # erf is a SymPy function, should not be rejected

    def test_detection_uw3_variable(self, setup_mesh):
        """UW3 MeshVariable should be detected as NOT pure."""
        from underworld3.function.pure_sympy_evaluator import is_pure_sympy_expression

        T = uw.discretisation.MeshVariable("T", setup_mesh, 1, degree=2)
        expr = T.sym[0]

        is_pure, symbols, sym_type = is_pure_sympy_expression(expr)

        assert is_pure is False
        # Should be None because it needs RBF interpolation
        assert symbols is None


class TestPerformanceExpectations:
    """Test that performance is as expected (not exact timing, just sanity checks)."""

    def test_lambdify_caching(self, setup_mesh, sample_points):
        """Cached lambdified evaluations should be fast."""
        import time

        x = setup_mesh.X[0]
        expr = sympy.erf(5 * x - 2) / 2

        # First call (compilation)
        start = time.time()
        result1 = uw.function.evaluate(expr, sample_points, rbf=True)
        time1 = time.time() - start

        # Cached call
        start = time.time()
        result2 = uw.function.evaluate(expr, sample_points, rbf=True)
        time2 = time.time() - start

        # Cached should be faster or at least not slower
        # (For small evaluations, overhead dominates so speedup may be small)
        assert time2 <= time1 * 2, f"Cached call slower than first: {time2} vs {time1}"

        # Both should be reasonably fast (< 10ms for 3 points)
        assert time2 < 0.01, f"Cached call too slow: {time2}s"

        # Results should be identical
        assert np.allclose(result1.flatten(), result2.flatten())

    def test_rbf_false_not_slow(self, setup_mesh, sample_points):
        """rbf=False should not be dramatically slower for pure sympy."""
        import time

        x = setup_mesh.X[0]
        expr = sympy.erf(5 * x - 2) / 2

        # Warm up cache
        _ = uw.function.evaluate(expr, sample_points, rbf=False)

        # Cached call with rbf=False should be fast (< 10ms for 3 points)
        start = time.time()
        result = uw.function.evaluate(expr, sample_points, rbf=False)
        elapsed = time.time() - start

        assert elapsed < 0.01, f"rbf=False too slow: {elapsed}s (should be < 10ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
