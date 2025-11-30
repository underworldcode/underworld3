#!/usr/bin/env python3
"""
Regression tests for recursion depth issues in mathematical object chains.

These tests prevent regressions of the infinite recursion issue that occurred
in the advection-diffusion solver when UWQuantity.atoms() method called
_sympify_() which returned self, creating an infinite loop.

Issues covered:
1. UWQuantity.atoms() infinite recursion via _sympify_()
2. Mathematical object chains with circular references
3. SymPy integration recursion issues
4. Function evaluation recursion in solvers
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import sympy
import sys
import os

# Add src to path for testing
# REMOVED: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import underworld3 as uw


class TestRecursionPreventionInMathematicalObjects:
    """Test recursion prevention in mathematical object chains."""

    def test_uwquantity_atoms_no_recursion(self):
        """Test that UWQuantity.atoms() doesn't cause infinite recursion."""

        # Create a UWQuantity that had recursion issues
        quantity = uw.function.expression(r"\eta", sym=1.5)

        # Set recursion limit to catch infinite recursion quickly
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)  # Low limit to catch recursion fast

        try:
            # This was causing infinite recursion before the fix
            atoms_result = quantity.atoms(sympy.Symbol)

            # Should return a set of symbols
            assert isinstance(atoms_result, set)

            # Should be able to call multiple times without issue
            atoms_result2 = quantity.atoms(sympy.Symbol)
            assert isinstance(atoms_result2, set)

            # Should be able to call with different types
            atoms_all = quantity.atoms()
            assert isinstance(atoms_all, set)

        except RecursionError:
            pytest.fail("UWQuantity.atoms() caused infinite recursion")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_sympify_returns_internal_representation(self):
        """Test that _sympify_() returns internal representation, not self."""

        quantity = uw.function.expression(r"test", sym=42.0)

        # _sympify_() should NOT return self (this was causing recursion)
        sympified = quantity._sympify_()

        assert sympified is not quantity, "_sympify_() should not return self"

        # Should return the internal symbolic representation
        assert sympified == quantity.sym or sympified == quantity._sym

        # Sympified result should be a pure SymPy object
        assert isinstance(sympified, (sympy.Basic, float, int))

    def test_mathematical_object_chain_safety(self):
        """Test that mathematical object chains don't cause recursion."""

        # Create a chain of mathematical objects
        expr1 = uw.function.expression(r"a", sym=1.0)
        expr2 = uw.function.expression(r"b", sym=2.0)
        expr3 = uw.function.expression(r"c", sym=3.0)

        # Create compound expressions (these should not cause recursion)
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50)

        try:
            # Mathematical operations should not trigger recursion
            compound1 = expr1 + expr2
            compound2 = expr2 * expr3
            compound3 = expr1 / (expr2 + expr3)

            # Should be able to call atoms on compound expressions
            atoms1 = compound1.atoms(sympy.Symbol) if hasattr(compound1, "atoms") else set()
            atoms2 = compound2.atoms(sympy.Symbol) if hasattr(compound2, "atoms") else set()
            atoms3 = compound3.atoms(sympy.Symbol) if hasattr(compound3, "atoms") else set()

            # All should return sets
            assert isinstance(atoms1, set)
            assert isinstance(atoms2, set)
            assert isinstance(atoms3, set)

        except RecursionError:
            pytest.fail("Mathematical object chains caused recursion")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_advection_diffusion_parameter_evaluation(self):
        """Test that advection-diffusion parameter evaluation doesn't recurse."""

        # Recreate the exact scenario that was causing recursion
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

        # Create variables
        velocity = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
        temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

        # Create advection-diffusion solver
        adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=temperature, V_fn=velocity)

        # Set constitutive model with UWexpression diffusivity (this was failing)
        adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
        diffusivity = uw.function.expression(r"\kappa", sym=1.0)
        adv_diff.constitutive_model.Parameters.diffusivity = diffusivity

        # Set recursion limit to catch the issue
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50)

        try:
            # This function evaluation was causing recursion in estimate_dt()
            max_diffusivity = uw.function.evaluate(
                adv_diff.constitutive_model.Parameters.diffusivity,
                temperature.coords[:5],  # Small subset
            )

            assert max_diffusivity is not None
            assert isinstance(max_diffusivity, (float, int, complex)) or hasattr(
                max_diffusivity, "shape"
            )

        except RecursionError:
            pytest.fail("Advection-diffusion parameter evaluation caused recursion")
        finally:
            sys.setrecursionlimit(old_limit)


class TestSymPyIntegrationRecursionPrevention:
    """Test recursion prevention in SymPy integration scenarios."""

    def test_sympy_function_calls_with_uwexpressions(self):
        """Test SymPy function calls don't cause recursion with UWexpressions."""

        expr = uw.function.expression(r"func_test", sym=0.5)

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)

        try:
            # SymPy functions should not cause recursion when applied to UWexpressions
            sin_result = sympy.sin(expr)
            cos_result = sympy.cos(expr)
            exp_result = sympy.exp(expr)
            log_result = sympy.log(expr + 1)  # Avoid log(0)

            # All should work without recursion
            results = [sin_result, cos_result, exp_result, log_result]
            for result in results:
                assert result is not None

        except RecursionError:
            pytest.fail("SymPy function calls caused recursion with UWexpressions")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_sympy_substitution_no_recursion(self):
        """Test that SymPy substitution doesn't cause recursion."""

        expr = uw.function.expression(r"sub_test", sym=sympy.Symbol("x"))

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)

        try:
            # Substitution operations should not cause recursion
            x = sympy.Symbol("x")
            substituted = expr.subs(x, 2.0)

            assert substituted is not None

        except RecursionError:
            pytest.fail("SymPy substitution caused recursion")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_sympy_differentiation_no_recursion(self):
        """Test that SymPy differentiation doesn't cause recursion."""

        x = sympy.Symbol("x")
        expr = uw.function.expression(r"diff_test", sym=x**2 + 2 * x + 1)

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)

        try:
            # Differentiation should not cause recursion
            diff_result = sympy.diff(expr, x)

            assert diff_result is not None

        except RecursionError:
            pytest.fail("SymPy differentiation caused recursion")
        finally:
            sys.setrecursionlimit(old_limit)


class TestRecursionPreventionInSolvers:
    """Test recursion prevention in solver contexts."""

    def test_estimate_dt_no_recursion(self):
        """Test that estimate_dt() doesn't cause infinite recursion.

        Note: This tests for INFINITE recursion bugs (like the original
        UWQuantity._sympify_() returning self), not deep but finite recursion
        from SymPy expression tree traversal. The limit is set high enough
        to allow normal SymPy operations but low enough to catch infinite loops.
        """

        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        velocity = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
        temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

        # Create solver
        adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=temperature, V_fn=velocity)
        adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
        adv_diff.constitutive_model.Parameters.diffusivity = uw.function.expression(
            r"\kappa", sym=1e-6
        )

        old_limit = sys.getrecursionlimit()
        # Set limit high enough for SymPy tree traversal but low enough to catch infinite loops
        # Original bug (UWQuantity._sympify_() returning self) would hit even high limits
        sys.setrecursionlimit(300)

        try:
            # This was the specific call that failed with the original recursion bug
            dt_estimate = adv_diff.estimate_dt()

            assert dt_estimate is not None
            # Can be float, int, or numpy scalar
            import numpy as np

            if hasattr(dt_estimate, "item"):
                dt_val = dt_estimate.item()
            else:
                dt_val = float(dt_estimate)
            assert dt_val > 0

        except RecursionError:
            pytest.fail("estimate_dt() caused infinite recursion")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_constitutive_model_parameter_access_no_recursion(self):
        """Test constitutive model parameter access doesn't cause recursion."""

        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        u = uw.discretisation.MeshVariable("U_recurs", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable("P_recurs", mesh, 1, degree=1)

        # CORRECT API: Let Stokes solver create the constitutive model
        stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

        viscosity = uw.function.expression(r"\eta", sym=1e21)
        stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity

        # Note: Use 'constitutive_model' not 'model' to avoid confusion with uw.model
        constitutive_model = stokes.constitutive_model

        old_limit = sys.getrecursionlimit()
        # Set reasonable limit to catch infinite recursion but allow normal operations
        sys.setrecursionlimit(300)

        try:
            # Accessing parameters should not cause recursion
            visc_param = constitutive_model.Parameters.shear_viscosity_0

            # Should be able to evaluate it
            if hasattr(visc_param, "atoms"):
                atoms = visc_param.atoms(sympy.Symbol)
                assert isinstance(atoms, set)

        except RecursionError:
            pytest.fail("Constitutive model parameter access caused recursion")
        finally:
            sys.setrecursionlimit(old_limit)


class TestRecursionDetectionUtilities:
    """Utility tests for detecting and preventing recursion."""

    def test_recursion_limit_detection(self):
        """Test that recursion limit detection works correctly."""

        def recursive_function(n):
            if n <= 0:
                return 0
            return recursive_function(n - 1)

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50)

        try:
            # This should hit recursion limit
            with pytest.raises(RecursionError):
                recursive_function(100)

        finally:
            sys.setrecursionlimit(old_limit)

    def test_safe_recursion_patterns(self):
        """Test patterns that safely prevent recursion."""

        # Pattern 1: Check for self-reference
        class SafeObject:
            def __init__(self, value):
                self.value = value

            def _sympify_(self):
                # SAFE: return internal value, not self
                return self.value

            def atoms(self, *types):
                # SAFE: sympify first, then call atoms on result
                sympified = self._sympify_()
                if sympified is not self and hasattr(sympified, "atoms"):
                    return sympified.atoms(*types)
                return set()

        # Test safe pattern
        safe_obj = SafeObject(sympy.Symbol("x"))

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50)

        try:
            atoms = safe_obj.atoms(sympy.Symbol)
            assert isinstance(atoms, set)

        except RecursionError:
            pytest.fail("Safe recursion pattern still caused recursion")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_recursion_debugging_helpers(self):
        """Test utilities for debugging recursion issues."""

        # Helper function to detect potential recursion
        def check_for_recursion_risk(obj):
            """Check if an object might cause recursion in atoms()."""
            if not hasattr(obj, "_sympify_") or not hasattr(obj, "atoms"):
                return False

            # Check if _sympify_() returns self (recursion risk)
            try:
                sympified = obj._sympify_()
                return sympified is obj  # Risk if returns self
            except:
                return True  # Risk if _sympify_() fails

        # Test with known safe and unsafe patterns
        safe_expr = uw.function.expression(r"safe", sym=1.0)

        # After fix, should not have recursion risk
        has_risk = check_for_recursion_risk(safe_expr)
        assert not has_risk, "Fixed UWexpression still shows recursion risk"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
