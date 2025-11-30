#!/usr/bin/env python3
"""
Regression tests for expression caching and hspace-based uniqueness.

This test suite validates that:
1. Expressions differing only by hspace are correctly distinguished
2. Identical derivatives are cached and reused (performance optimization)
3. The _unique_name_generation flag creates unique expressions
4. Derivative caching doesn't produce warnings (intentional behavior)
5. Hspace size is sufficiently small (nearly invisible in LaTeX)
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import re
import underworld3 as uw
from underworld3.function.expressions import UWexpression


class TestExpressionUniqueness:
    """Test that hspace-based uniqueness correctly distinguishes expressions."""

    def test_different_mesh_variables_have_unique_derivatives(self):
        """
        Derivatives from variables on different meshes should be unique.

        This validates that the hspace mechanism correctly creates unique
        derivative expressions even when variable names are identical.
        """
        # Create two different meshes (different instance numbers)
        mesh1 = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        mesh2 = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        # Verify meshes have different instance numbers
        assert mesh1.instance_number != mesh2.instance_number

        # Create variables with same name on different meshes
        T1 = uw.discretisation.MeshVariable("T", mesh1, 1, degree=2)
        T2 = uw.discretisation.MeshVariable("T", mesh2, 1, degree=2)

        # Variables should have different symbols (due to hspace)
        assert str(T1.symbol) != str(T2.symbol)

        # Create derivatives
        y1 = mesh1.N.y
        y2 = mesh2.N.y

        deriv1 = T1.diff(y1)[0]
        deriv2 = T2.diff(y2)[0]

        # Derivatives should be DIFFERENT objects
        assert deriv1 is not deriv2, "Derivatives from different meshes should be unique objects"

        # Derivative names should be different strings
        assert str(deriv1) != str(deriv2), "Derivative names should differ by hspace"


class TestDerivativeCaching:
    """Test that identical derivatives are properly cached."""

    def test_same_derivative_is_cached(self):
        """
        Creating the same derivative twice should return the cached object.

        This validates the performance optimization where identical derivatives
        are reused rather than recreated.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        y = mesh.N.y

        # Create the same derivative twice
        deriv1 = T.diff(y)[0]
        deriv2 = T.diff(y)[0]

        # Should be the SAME object (cached)
        assert deriv1 is deriv2, "Identical derivatives should be cached and reused"

    def test_derivative_caching_no_warning(self, capfd):
        """
        Derivative caching should not produce warnings.

        Since derivative caching is intentional and beneficial, it should
        not trigger warnings.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        y = mesh.N.y

        # Create derivative twice - should not warn
        deriv1 = T.diff(y)[0]
        deriv2 = T.diff(y)[0]

        # Capture stderr to check for warnings
        captured = capfd.readouterr()

        # Should NOT contain any expression warnings
        assert (
            "Each expression should have a unique name" not in captured.err
        ), "Derivative caching should not produce warnings"


class TestExpressionSilentUpdate:
    """Test that expressions silently update when recreated with same name."""

    def test_expression_updates_silently(self, capfd):
        """
        Recreating an expression with the same name should silently update it.

        This is natural Python behavior - preserving object identity while
        updating internal state. No warnings should be produced.
        """
        # Create expression with initial value
        alpha = UWexpression(r"\alpha", sym=1.0, description="First value")

        # Recreate with same name but different value
        alpha2 = UWexpression(r"\alpha", sym=2.0, description="Second value")

        # Should be the SAME object (identity preserved)
        assert alpha is alpha2, "Recreating expression should preserve object identity"

        # Should have updated sym value
        assert alpha2.sym == 2.0, "Expression sym should be updated to new value"

        # Should have updated description
        assert alpha2.description == "Second value", "Expression description should be updated"

        # Should NOT produce warnings
        captured = capfd.readouterr()
        assert (
            "Each expression should have a unique name" not in captured.err
        ), "Expression update should be silent (no warnings)"

    def test_expression_update_in_loop(self):
        """
        Updating expressions in loops should work naturally.

        This is a common pattern where expressions are recreated in each
        iteration with updated values.
        """
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, val in enumerate(values):
            eta = UWexpression(r"\eta", sym=val)

            # All iterations should return the same object
            if i == 0:
                first_eta = eta
            else:
                assert eta is first_eta, "Loop should reuse same expression object"

            # Value should be updated each iteration
            assert eta.sym == val, f"Iteration {i}: expected sym={val}, got {eta.sym}"

    def test_unique_flag_creates_new_objects(self):
        """
        Using _unique_name_generation=True should create new objects.

        This is the opt-in mechanism for when you genuinely need
        multiple distinct expressions with the same symbol.
        """
        eta1 = UWexpression(r"\eta", sym=1.0, _unique_name_generation=True)
        eta2 = UWexpression(r"\eta", sym=2.0, _unique_name_generation=True)

        # Should be DIFFERENT objects (unique flag respected)
        assert eta1 is not eta2, "Unique flag should create distinct objects"

        # Each should maintain its own value
        assert eta1.sym == 1.0, "First expression should keep its value"
        assert eta2.sym == 2.0, "Second expression should keep its value"


class TestUniqueNameGeneration:
    """Test the _unique_name_generation flag for creating distinct expressions."""

    def test_unique_name_flag_creates_distinct_expressions(self):
        """
        The _unique_name_generation flag should create unique expressions
        even with identical visible symbols.

        This enables hierarchical namespaces (mesh.solver.model.viscosity)
        to coexist with SymPy's global namespace.
        """
        # Create two expressions with same symbol but unique flag
        eta1 = UWexpression(r"\eta", sym=1.0, _unique_name_generation=True)
        eta2 = UWexpression(r"\eta", sym=2.0, _unique_name_generation=True)

        # Should be DIFFERENT objects
        assert eta1 is not eta2, "Expressions with _unique_name_generation should be unique objects"

        # Should have different internal names (hspace differs)
        assert str(eta1.name) != str(
            eta2.name
        ), "Unique expressions should have different internal names"

        # But should have the same given name (visible symbol)
        assert (
            eta1._given_name == eta2._given_name == r"\eta"
        ), "Unique expressions should share the same visible symbol"

    def test_unique_expressions_have_different_values(self):
        """
        Unique expressions with same symbol can have different values.

        This is critical for having multiple constitutive models with
        the same parameter names but different values.
        """
        eta1 = UWexpression(r"\eta", sym=1.0e18, _unique_name_generation=True)
        eta2 = UWexpression(r"\eta", sym=1.0e22, _unique_name_generation=True)

        # Should have different values
        assert eta1.sym != eta2.sym, "Unique expressions should maintain independent values"

        # Verify they are truly independent (changing one doesn't affect other)
        eta1.sym = 5.0
        assert eta2.sym != 5.0, "Unique expressions should be independent"


class TestHspaceSize:
    """Test that hspace size is sufficiently small (nearly invisible)."""

    def test_mesh_variable_hspace_is_tiny(self):
        """
        Mesh variable symbols should use tiny hspace (100x smaller than old 0.01pt).

        The hspace should be nearly invisible in LaTeX rendering while
        still providing uniqueness for SymPy.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        # Skip test if this is the first mesh (no hspace needed)
        if mesh.instance_number <= 1:
            pytest.skip("First mesh instance doesn't use hspace")

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

        # Extract hspace from symbol
        match = re.search(r"\\hspace\{ ([\d.e-]+)pt \}", T.symbol)

        if match:
            hspace = float(match.group(1))

            # Should be much smaller than old 0.01pt minimum
            # Old formula: instance_number / 100 (minimum 0.01pt for instance 1)
            # New formula: instance_number / 10000 (100x smaller)
            old_hspace = mesh.instance_number / 100
            assert (
                hspace < old_hspace
            ), f"New hspace ({hspace}pt) should be < old hspace ({old_hspace}pt)"

            # Verify formula is correct (instance_number / 10000)
            expected_hspace = mesh.instance_number / 10000
            assert (
                abs(hspace - expected_hspace) < 1e-10
            ), f"Hspace should be {expected_hspace}pt, got {hspace}pt"

    def test_derivative_hspace_is_tiny(self):
        """
        Derivative expressions should inherit tiny hspace from parent variable.

        This ensures derivatives also have nearly invisible spacing in LaTeX.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        # Skip test if this is the first mesh
        if mesh.instance_number <= 1:
            pytest.skip("First mesh instance doesn't use hspace")

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        y = mesh.N.y

        deriv = T.diff(y)[0]

        # Extract hspace from derivative name
        match = re.search(r"\\hspace\{ ([\d.e-]+)pt \}", str(deriv))

        if match:
            hspace = float(match.group(1))

            # Should be 100x smaller than old formula
            # Derivatives inherit hspace from parent mesh variable
            old_hspace = mesh.instance_number / 100
            assert (
                hspace < old_hspace
            ), f"Derivative hspace ({hspace}pt) should be < old hspace ({old_hspace}pt)"

            # Verify it matches the mesh variable's hspace
            expected_hspace = mesh.instance_number / 10000
            assert (
                abs(hspace - expected_hspace) < 1e-10
            ), f"Derivative hspace should match variable hspace: {expected_hspace}pt, got {hspace}pt"


class TestExpressionRegistryConsistency:
    """Test that expression registry remains consistent."""

    def test_cached_derivatives_not_duplicated(self):
        """
        Cached derivatives should only appear once in the expression registry.

        This validates that caching doesn't create duplicate entries.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        y = mesh.N.y

        # Count derivatives before
        initial_count = len([name for name in UWexpression._expr_names.keys() if "_{," in name])

        # Create same derivative multiple times
        deriv1 = T.diff(y)[0]
        deriv2 = T.diff(y)[0]
        deriv3 = T.diff(y)[0]

        # Count derivatives after
        final_count = len([name for name in UWexpression._expr_names.keys() if "_{," in name])

        # Should have added exactly ONE derivative
        assert (
            final_count == initial_count + 1
        ), f"Expected 1 new derivative, got {final_count - initial_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
