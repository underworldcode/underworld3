"""
Test addition and subtraction operations preserve symbols regardless of operand order.

These tests ensure that UWexpression symbols are preserved during arithmetic
operations, preventing sympification that converts symbols to numeric values.
"""

import pytest
import underworld3 as uw
import numpy as np


@pytest.mark.level_2  # Intermediate - arithmetic with units
@pytest.mark.tier_a   # Production-ready - critical arithmetic bug
class TestSubtractionOrder:
    """
    Test subtraction operations with UWexpression preserve symbols.

    CRITICAL BUG FOUND (2025-11-25):
    - Expression: 0.000001 * xx - xx0
    - Was sympifying xx0 to 0.4 (losing symbol)
    - Result: Wrong evaluation (-1159997 m instead of -397 m)

    The fix: UWexpression IS a SymPy Symbol, use it directly.
    """

    def setup_method(self):
        """Set up model with nondimensional scaling."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K"),
            nondimensional_scaling=True
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        # Set up coordinate symbols
        x, y = self.mesh.CoordinateSystem.X
        self.xx = uw.expression_types.UnitAwareExpression(x, uw.units.m)
        self.yy = uw.expression_types.UnitAwareExpression(y, uw.units.m)

    def teardown_method(self):
        """Disable scaling after each test."""
        uw.use_nondimensional_scaling(False)

    def test_subtraction_left_uwexpression_minus_right_uwexpression(self):
        """
        Test: (coord expression) - (UWexpression) preserves both symbols.

        This is the CRITICAL BUG CASE:
        - 0.000001 * xx - xx0
        - Was converting xx0 to 0.4 (numeric)
        - Should preserve xx0 as symbol
        """
        xx0 = uw.expression(r"x_0", uw.quantity(0.4, 'km'), "Initial position")

        # Create expression
        expr = 0.000001 * self.xx - xx0

        # Check expression tree contains the symbol
        import sympy
        if hasattr(expr, '_expr'):
            symbols = list(expr._expr.atoms(sympy.Symbol))
            assert len(symbols) > 0, f"Expression lost all symbols! _expr: {expr._expr}"

            # Should contain x_0 symbol - use .name attribute, not str()
            symbol_names = [s.name for s in symbols]
            assert 'x_0' in symbol_names or any('x_0' in name for name in symbol_names), \
                f"Expression lost x_0 symbol! Found: {symbol_names}, _expr: {expr._expr}"

        # Evaluate and check result
        result = uw.function.evaluate(expr, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: max(0.000001 * 2900000 m - 400 m) = max(2.9 - 400) = -397.1 m
        expected = -397.1
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Subtraction gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"

    def test_subtraction_right_uwexpression_minus_left_expression(self):
        """Test: (UWexpression) - (coord expression) preserves both symbols."""
        xx0 = uw.expression(r"x_0", uw.quantity(0.4, 'km'), "Initial position")

        # Create expression (reversed order)
        expr = xx0 - 0.000001 * self.xx

        # Check expression tree
        import sympy
        if hasattr(expr, '_expr'):
            symbols = list(expr._expr.atoms(sympy.Symbol))
            assert len(symbols) > 0, f"Expression lost all symbols! _expr: {expr._expr}"

            symbol_names = [str(s) for s in symbols]
            assert 'x_0' in symbol_names or any('x_0' in name for name in symbol_names), \
                f"Expression lost x_0 symbol! Found: {symbol_names}"

        # Evaluate and check result
        result = uw.function.evaluate(expr, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: max(400 m - 0.000001 * 2900000 m) = max(400 - 2.9) = 397.1 m
        expected = 397.1
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Subtraction gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"

    def test_subtraction_uwexpression_minus_uwexpression(self):
        """Test: (UWexpression) - (UWexpression) preserves both symbols."""
        x0 = uw.expression(r"x_0", uw.quantity(0.5, 'km'), "Position 1")
        x1 = uw.expression(r"x_1", uw.quantity(0.3, 'km'), "Position 2")

        # Create expression
        diff = x0 - x1

        # Check expression tree
        import sympy
        if hasattr(diff, '_expr'):
            symbols = list(diff._expr.atoms(sympy.Symbol))
            symbol_names = [str(s) for s in symbols]

            # Should contain both symbols
            assert 'x_0' in symbol_names or any('x_0' in name for name in symbol_names), \
                f"Expression lost x_0 symbol! Found: {symbol_names}"
            assert 'x_1' in symbol_names or any('x_1' in name for name in symbol_names), \
                f"Expression lost x_1 symbol! Found: {symbol_names}"

        # Evaluate and check result
        result = uw.function.evaluate(diff, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: 500 m - 300 m = 200 m
        expected = 200.0
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Subtraction gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"


@pytest.mark.level_2  # Intermediate - arithmetic with units
@pytest.mark.tier_a   # Production-ready - critical arithmetic bug
class TestAdditionOrder:
    """
    Test addition operations with UWexpression preserve symbols.

    Same issue as subtraction - must preserve symbolic nature.
    """

    def setup_method(self):
        """Set up model with nondimensional scaling."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            mass=uw.quantity(1e24, "kg"),
            temperature=uw.quantity(1000, "K"),
            nondimensional_scaling=True
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        x, y = self.mesh.CoordinateSystem.X
        self.xx = uw.expression_types.UnitAwareExpression(x, uw.units.m)

    def teardown_method(self):
        """Disable scaling after each test."""
        uw.use_nondimensional_scaling(False)

    def test_addition_left_expression_plus_right_uwexpression(self):
        """Test: (coord expression) + (UWexpression) preserves both symbols."""
        xx0 = uw.expression(r"x_0", uw.quantity(0.4, 'km'), "Initial position")

        # Create expression
        expr = 0.000001 * self.xx + xx0

        # Check expression tree
        import sympy
        if hasattr(expr, '_expr'):
            symbols = list(expr._expr.atoms(sympy.Symbol))
            assert len(symbols) > 0, f"Expression lost all symbols! _expr: {expr._expr}"

            symbol_names = [str(s) for s in symbols]
            assert 'x_0' in symbol_names or any('x_0' in name for name in symbol_names), \
                f"Expression lost x_0 symbol! Found: {symbol_names}"

        # Evaluate and check result
        result = uw.function.evaluate(expr, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: max(0.000001 * 2900000 m + 400 m) = max(2.9 + 400) = 402.9 m
        expected = 402.9
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Addition gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"

    def test_addition_right_uwexpression_plus_left_expression(self):
        """Test: (UWexpression) + (coord expression) preserves both symbols."""
        xx0 = uw.expression(r"x_0", uw.quantity(0.4, 'km'), "Initial position")

        # Create expression (reversed order)
        expr = xx0 + 0.000001 * self.xx

        # Check expression tree
        import sympy
        if hasattr(expr, '_expr'):
            symbols = list(expr._expr.atoms(sympy.Symbol))
            assert len(symbols) > 0, f"Expression lost all symbols! _expr: {expr._expr}"

            symbol_names = [str(s) for s in symbols]
            assert 'x_0' in symbol_names or any('x_0' in name for name in symbol_names), \
                f"Expression lost x_0 symbol! Found: {symbol_names}"

        # Evaluate and check result
        result = uw.function.evaluate(expr, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: max(400 m + 0.000001 * 2900000 m) = max(400 + 2.9) = 402.9 m
        expected = 402.9
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Addition gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"

    def test_addition_uwexpression_plus_uwexpression(self):
        """Test: (UWexpression) + (UWexpression) preserves both symbols."""
        x0 = uw.expression(r"x_0", uw.quantity(0.3, 'km'), "Position 1")
        x1 = uw.expression(r"x_1", uw.quantity(0.2, 'km'), "Position 2")

        # Create expression
        summed = x0 + x1

        # Check expression tree
        import sympy
        if hasattr(summed, '_expr'):
            symbols = list(summed._expr.atoms(sympy.Symbol))
            symbol_names = [str(s) for s in symbols]

            # Should contain both symbols
            assert 'x_0' in symbol_names or any('x_0' in name for name in symbol_names), \
                f"Expression lost x_0 symbol! Found: {symbol_names}"
            assert 'x_1' in symbol_names or any('x_1' in name for name in symbol_names), \
                f"Expression lost x_1 symbol! Found: {symbol_names}"

        # Evaluate and check result
        result = uw.function.evaluate(summed, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: 300 m + 200 m = 500 m
        expected = 500.0
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Addition gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"


@pytest.mark.level_2  # Intermediate - arithmetic with units
@pytest.mark.tier_a   # Production-ready - critical arithmetic bug
class TestMixedArithmeticOrder:
    """Test complex expressions with multiple operations preserve all symbols."""

    def setup_method(self):
        """Set up model with nondimensional scaling."""
        self.model = uw.Model()
        self.model.set_reference_quantities(
            length=uw.quantity(2900, "km"),
            time=uw.quantity(1, "Myr"),
            nondimensional_scaling=True
        )

        uw.use_nondimensional_scaling(True)

        self.mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0, 0), maxCoords=(1, 1),
            cellSize=0.1, qdegree=2
        )

        x, y = self.mesh.CoordinateSystem.X
        self.xx = uw.expression_types.UnitAwareExpression(x, uw.units.m)

    def teardown_method(self):
        """Disable scaling after each test."""
        uw.use_nondimensional_scaling(False)

    def test_complex_expression_preserves_all_symbols(self):
        """Test: (expr1 + expr2) - (expr3 * expr4) preserves all symbols."""
        x0 = uw.expression(r"x_0", uw.quantity(0.5, 'km'), "Position")
        x1 = uw.expression(r"x_1", uw.quantity(0.2, 'km'), "Offset 1")
        x2 = uw.expression(r"x_2", uw.quantity(0.1, 'km'), "Offset 2")

        # Complex expression
        expr = (x0 + x1) - (x2 + 0.000001 * self.xx)

        # Check expression tree contains all symbols
        import sympy
        if hasattr(expr, '_expr'):
            symbols = list(expr._expr.atoms(sympy.Symbol))
            symbol_names = [str(s) for s in symbols]

            # Should contain all three UWexpression symbols
            for name in ['x_0', 'x_1', 'x_2']:
                assert name in symbol_names or any(name in sname for sname in symbol_names), \
                    f"Expression lost {name} symbol! Found: {symbol_names}, _expr: {expr._expr}"

        # Evaluate and check result
        result = uw.function.evaluate(expr, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: (500 + 200) - (100 + 2.9) = 700 - 102.9 = 597.1 m
        expected = 597.1
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Complex expression gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"

    def test_subtraction_chain_preserves_symbols(self):
        """Test: a - b - c preserves all symbols."""
        x0 = uw.expression(r"x_0", uw.quantity(1.0, 'km'), "Start")
        x1 = uw.expression(r"x_1", uw.quantity(0.3, 'km'), "Offset 1")
        x2 = uw.expression(r"x_2", uw.quantity(0.2, 'km'), "Offset 2")

        # Chained subtraction
        expr = x0 - x1 - x2

        # Check expression tree
        import sympy
        if hasattr(expr, '_expr'):
            symbols = list(expr._expr.atoms(sympy.Symbol))
            symbol_names = [str(s) for s in symbols]

            # All three symbols should be preserved
            for name in ['x_0', 'x_1', 'x_2']:
                assert name in symbol_names or any(name in sname for sname in symbol_names), \
                    f"Chained subtraction lost {name} symbol! Found: {symbol_names}"

        # Evaluate
        result = uw.function.evaluate(expr, self.mesh.X.coords)
        val_max = float(result.max().magnitude if hasattr(result.max(), 'magnitude') else result.max())

        # Expected: 1000 - 300 - 200 = 500 m
        expected = 500.0
        assert np.allclose(val_max, expected, rtol=0.01), \
            f"Chained subtraction gave wrong result: {val_max:.2f} m, expected {expected:.2f} m"
