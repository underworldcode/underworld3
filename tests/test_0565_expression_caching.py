#!/usr/bin/env python3
"""
Regression tests for expression caching and symbol disambiguation.

This test suite validates that:
1. Symbols from different meshes are correctly distinguished (via _uw_id)
2. Identical derivatives are cached and reused (performance optimization)
3. The _unique_name_generation flag creates unique expressions
4. Derivative caching doesn't produce warnings (intentional behavior)

NOTE: As of 2025-12-15, symbol disambiguation uses SymPy's native _uw_id mechanism
instead of invisible \\hspace{} whitespace. Symbols now have clean display names
but are distinguished by internal ID. See SYMBOL_DISAMBIGUATION_2025-12.md.
"""

import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
import re
import underworld3 as uw
from underworld3.function.expressions import UWexpression


class TestExpressionUniqueness:
    """Test that _uw_id-based uniqueness correctly distinguishes expressions."""

    def test_different_mesh_variables_have_unique_symbols(self):
        """
        Variables from different meshes should have distinct SymPy symbols.

        This validates that the _uw_id mechanism correctly creates unique
        symbols even when variable names are identical. The symbols have
        the same display name but are distinguished by their function class's _uw_id.

        NOTE: SymPy applied functions have the same hash if they have the same
        name and arguments. The distinction is in the function class's _uw_id,
        which is checked in __eq__, not __hash__. This is consistent with
        SymPy's design where hash collisions are allowed but equality is strict.
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

        # Variables should have different symbols (distinguished by _uw_id)
        # The str() representation is now the SAME (clean names), but symbols are distinct
        assert T1.sym != T2.sym, "Variables from different meshes should have different symbols"
        assert T1.sym is not T2.sym, "Variables should be different objects"

        # The function CLASSES should be different (with different _uw_id)
        assert T1.sym[0, 0].func is not T2.sym[0, 0].func, "Function classes should be distinct"
        assert T1.sym[0, 0].func._uw_id != T2.sym[0, 0].func._uw_id, "Function _uw_id should differ"

        # Applied functions should NOT be equal (even if hash is same)
        assert T1.sym[0, 0] != T2.sym[0, 0], "Applied functions should not be equal"

    def test_different_mesh_variables_have_unique_derivatives(self):
        """
        Derivatives from variables on different meshes should be unique.

        This validates that the _uw_id mechanism correctly creates unique
        derivative expressions even when variable names are identical.

        NOTE: The .diff()[0] method returns a UWexpression wrapper that is
        cached by display name. To check symbol uniqueness, we need to look
        at the raw SymPy derivative matrix, not the wrapped result.
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

        # Create derivatives using raw SymPy (not the wrapped interface)
        y1 = mesh1.N.y
        y2 = mesh2.N.y

        # Use raw SymPy diff on the symbolic form
        raw_deriv1 = T1.sym.diff(y1)[0]
        raw_deriv2 = T2.sym.diff(y2)[0]

        # Raw derivatives should be DIFFERENT objects (via _uw_id)
        assert raw_deriv1 is not raw_deriv2, "Raw derivatives from different meshes should be unique objects"

        # Derivatives should be symbolically different (not equal via SymPy __eq__)
        assert raw_deriv1 != raw_deriv2, "Raw derivatives from different meshes should be distinct"


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

        NOTE: With the _uw_id mechanism, unique expressions have the same
        display name but different _uw_id values (and thus different hashes).
        """
        # Create two expressions with same symbol but unique flag
        eta1 = UWexpression(r"\eta", sym=1.0, _unique_name_generation=True)
        eta2 = UWexpression(r"\eta", sym=2.0, _unique_name_generation=True)

        # Should be DIFFERENT objects
        assert eta1 is not eta2, "Expressions with _unique_name_generation should be unique objects"

        # Should have different _uw_id values (this is how they're distinguished now)
        assert eta1._uw_id != eta2._uw_id, "Unique expressions should have different _uw_id values"

        # Should have different hashes (due to _uw_id in _hashable_content)
        assert hash(eta1) != hash(eta2), "Unique expressions should have different hashes"

        # They should NOT be equal via SymPy's __eq__
        assert eta1 != eta2, "Unique expressions should not be equal"

        # But display name is the same (clean output - this is the new behavior!)
        assert str(eta1.name) == str(eta2.name), "Display names are now identical (clean output)"

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


class TestCleanSymbolNames:
    """Test that symbol names are now clean (no hspace hack)."""

    def test_mesh_variable_has_clean_symbol_name(self):
        """
        Mesh variable symbols should have clean display names without hspace.

        The new _uw_id mechanism uses SymPy's native identity system instead
        of invisible whitespace in symbol names.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

        # Symbol name should NOT contain hspace (old mechanism removed)
        assert "\\hspace" not in T.symbol, "Symbol should have clean name without hspace"

        # Symbol name should be simple
        assert "{T}" in T.symbol or "T" == T.symbol, f"Symbol should be clean: {T.symbol}"

    def test_different_meshes_have_clean_but_distinct_symbols(self):
        """
        Variables on different meshes have clean names but distinct symbols.

        This demonstrates the new mechanism: same display name, different identity.
        The _uw_id mechanism ensures distinction via __eq__, not __hash__.
        """
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

        T1 = uw.discretisation.MeshVariable("T", mesh1, 1, degree=2)
        T2 = uw.discretisation.MeshVariable("T", mesh2, 1, degree=2)

        # Both should have clean symbol names (no hspace)
        assert "\\hspace" not in T1.symbol, "T1 symbol should be clean"
        assert "\\hspace" not in T2.symbol, "T2 symbol should be clean"

        # But they should be distinct symbols (via _uw_id in __eq__)
        assert T1.sym != T2.sym, "Symbols should be distinct despite same display name"
        assert T1.sym[0, 0] != T2.sym[0, 0], "Applied functions should not be equal"

        # Function classes should have different _uw_id
        assert T1.sym[0, 0].func._uw_id != T2.sym[0, 0].func._uw_id, "Function _uw_id should differ"


class TestExpressionRegistryConsistency:
    """Test that expression registry remains consistent."""

    def test_cached_derivatives_are_same_object(self):
        """
        Cached derivatives should return the same object when accessed multiple times.

        This validates that derivative caching works correctly.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        y = mesh.N.y

        # Create same derivative multiple times
        deriv1 = T.diff(y)[0]
        deriv2 = T.diff(y)[0]
        deriv3 = T.diff(y)[0]

        # All should be the SAME object (cached)
        assert deriv1 is deriv2, "First two derivatives should be same object"
        assert deriv2 is deriv3, "Second and third derivatives should be same object"

        # They should all have the same hash
        assert hash(deriv1) == hash(deriv2) == hash(deriv3), "All derivatives should have same hash"

    def test_different_derivatives_are_distinct(self):
        """
        Derivatives with respect to different variables should be distinct.
        """
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=0.2,
        )

        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        x = mesh.N.x
        y = mesh.N.y

        # Create derivatives w.r.t. different variables
        dT_dx = T.diff(x)[0]
        dT_dy = T.diff(y)[0]

        # Should be different objects
        assert dT_dx is not dT_dy, "Derivatives w.r.t. different variables should be distinct"
        assert dT_dx != dT_dy, "Derivatives should not be equal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
