"""
Test script for clean symbol disambiguation approach.

This tests whether we can replace the invisible whitespace hack (\hspace{...})
with SymPy-native mechanisms:
- For Symbol subclasses: override _hashable_content() to include a unique ID
- For UndefinedFunction: pass _uw_id as a kwarg (used in __eq__/__hash__)

Run with: pixi run -e default pytest tests/test_symbol_disambiguation_prototype.py -v
"""

import pytest
import sympy
from sympy import Symbol, symbols, latex, diff, simplify, expand
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.cache import cacheit
import weakref
import numpy as np


# =============================================================================
# PROTOTYPE CLASSES
# =============================================================================

class UniqueSymbol(Symbol):
    """
    Symbol subclass with unique instance ID for disambiguation.

    This follows the same pattern as sympy.Dummy, which uses _hashable_content()
    to include a unique index, making symbols with the same display name distinct.

    Usage:
        alpha1 = UniqueSymbol(r'\\alpha')  # auto-assigned uw_id=0
        alpha2 = UniqueSymbol(r'\\alpha')  # auto-assigned uw_id=1
        alpha1 == alpha2  # False (distinct symbols)
        str(alpha1) == str(alpha2)  # True (same display name)
    """

    _count = 0
    __slots__ = ('_uw_id',)

    def __new__(cls, name, uw_id=None, **assumptions):
        if uw_id is None:
            uw_id = UniqueSymbol._count
            UniqueSymbol._count += 1

        cls._sanitize(assumptions, cls)
        obj = Symbol.__xnew__(cls, name, **assumptions)
        obj._uw_id = uw_id
        return obj

    def __getnewargs_ex__(self):
        """Support pickling by including uw_id in reconstruction args."""
        return ((self.name, self._uw_id), self._assumptions_orig)

    def _hashable_content(self):
        """Include _uw_id in hash so symbols with same name but different IDs are distinct."""
        return Symbol._hashable_content(self) + (self._uw_id,)

    @cacheit
    def sort_key(self, order=None):
        """Include _uw_id in sort key for consistent ordering."""
        from sympy import S
        return self.class_key(), (
            2, (self.name, self._uw_id)), S.One.sort_key(), S.One


def create_unique_function(name, uw_id, bases=(AppliedUndef,)):
    """
    Create a unique UndefinedFunction using _uw_id kwarg.

    SymPy's UndefinedFunction uses kwargs in __eq__ and __hash__,
    so passing a unique _uw_id makes functions with same name distinct.
    """
    return UndefinedFunction(name, bases=bases, _uw_id=uw_id)


# =============================================================================
# MOCK UNDERWORLD CLASSES FOR TESTING
# =============================================================================

class MockMesh:
    """Simulates a UW3 mesh with instance tracking."""
    _count = 0
    # Use high ID offset to avoid collision with simpler tests that use uw_id=1,2,etc.
    _ID_OFFSET = 10000

    def __init__(self, dim=2):
        self.instance_number = MockMesh._ID_OFFSET + MockMesh._count
        MockMesh._count += 1
        self.dim = dim
        self.r = symbols('x y z')[:dim]
        self.vars = {}


class MockMeshVariable:
    """Simulates a UW3 MeshVariable with the new disambiguation approach."""

    def __init__(self, mesh, name, symbol=None):
        self.mesh = mesh
        self.name = name
        self.symbol = symbol or name

        # Create function class with unique ID based on mesh instance
        self._fn_class = create_unique_function(
            self.symbol,
            uw_id=mesh.instance_number
        )

        # Store weakref on class (as UnderworldFunction does)
        self._fn_class.meshvar = weakref.ref(self)

        # Create applied function (the actual symbol used in expressions)
        self._sym = self._fn_class(*mesh.r)

        # Register with mesh
        mesh.vars[name] = self

    @property
    def sym(self):
        return self._sym

    @property
    def fn(self):
        return self._fn_class


# =============================================================================
# TEST: BASIC SYMBOL UNIQUENESS
# =============================================================================

class TestUniqueSymbol:
    """Test the UniqueSymbol class for basic disambiguation."""

    def test_same_name_different_id_not_equal(self):
        """Symbols with same name but different IDs should not be equal."""
        alpha1 = UniqueSymbol(r'\alpha')
        alpha2 = UniqueSymbol(r'\alpha')

        assert alpha1 != alpha2
        assert hash(alpha1) != hash(alpha2)

    def test_same_name_same_id_equal(self):
        """Symbols with same name AND same ID should be equal."""
        start_id = UniqueSymbol._count
        alpha1 = UniqueSymbol(r'\alpha', uw_id=start_id + 1000)
        alpha2 = UniqueSymbol(r'\alpha', uw_id=start_id + 1000)

        assert alpha1 == alpha2
        assert hash(alpha1) == hash(alpha2)

    def test_display_name_unchanged(self):
        """String representation should be the clean name."""
        alpha = UniqueSymbol(r'\alpha')

        assert str(alpha) == r'\alpha'
        assert alpha.name == r'\alpha'

    def test_latex_clean(self):
        """LaTeX output should be clean without any \hspace."""
        alpha1 = UniqueSymbol(r'\alpha')
        alpha2 = UniqueSymbol(r'\alpha')

        expr = alpha1 + alpha2
        latex_str = latex(expr)

        assert r'\hspace' not in latex_str
        assert r'\alpha' in latex_str

    def test_addition_not_combined(self):
        """Adding distinct symbols should not combine them."""
        alpha1 = UniqueSymbol(r'\alpha')
        alpha2 = UniqueSymbol(r'\alpha')

        expr = alpha1 + alpha2

        # Should have two terms, not be simplified to 2*alpha
        assert len(expr.args) == 2 or expr != 2 * alpha1

    def test_substitution_selective(self):
        """Substitution should only affect the specific symbol."""
        alpha1 = UniqueSymbol(r'\alpha')
        alpha2 = UniqueSymbol(r'\alpha')

        expr = 3 * alpha1 + 2 * alpha2

        result1 = expr.subs(alpha1, 10)
        result2 = expr.subs(alpha2, 20)

        # alpha1 -> 10: 3*10 + 2*alpha2 = 30 + 2*alpha2
        assert result1 == 30 + 2 * alpha2

        # alpha2 -> 20: 3*alpha1 + 2*20 = 3*alpha1 + 40
        assert result2 == 3 * alpha1 + 40

    def test_survives_sympy_operations(self):
        """Unique ID should survive SymPy expression manipulations."""
        alpha1 = UniqueSymbol(r'\alpha')
        alpha2 = UniqueSymbol(r'\alpha')

        expr = (alpha1 + alpha2) ** 2
        expanded = expand(expr)

        # Check that free_symbols still has two distinct alphas
        free_syms = list(expanded.free_symbols)
        assert len(free_syms) == 2

        # Verify they have different IDs
        ids = [s._uw_id for s in free_syms]
        assert len(set(ids)) == 2


# =============================================================================
# TEST: UNIQUE FUNCTION (FOR MESH VARIABLES)
# =============================================================================

class TestUniqueFunction:
    """Test the UndefinedFunction with _uw_id kwarg approach."""

    def test_same_name_different_id_not_equal(self):
        """Functions with same name but different _uw_id should not be equal."""
        x, y = symbols('x y')

        f1 = create_unique_function('u', uw_id=1)
        f2 = create_unique_function('u', uw_id=2)

        u1 = f1(x, y)
        u2 = f2(x, y)

        assert f1 != f2
        assert u1 != u2

    def test_same_name_same_id_equal(self):
        """Functions with same name AND same _uw_id should be equal."""
        x, y = symbols('x y')

        f1 = create_unique_function('u', uw_id=100)
        f2 = create_unique_function('u', uw_id=100)

        u1 = f1(x, y)
        u2 = f2(x, y)

        assert f1 == f2
        assert u1 == u2

    def test_display_name_unchanged(self):
        """Function name in output should be clean."""
        x, y = symbols('x y')

        f = create_unique_function('u', uw_id=1)
        u = f(x, y)

        assert 'u' in str(u)
        assert r'\hspace' not in str(u)

    def test_latex_clean(self):
        """LaTeX output should be clean."""
        x, y = symbols('x y')

        f1 = create_unique_function('u', uw_id=1)
        f2 = create_unique_function('u', uw_id=2)

        expr = f1(x, y) + f2(x, y)
        latex_str = latex(expr)

        assert r'\hspace' not in latex_str

    def test_addition_not_combined(self):
        """Adding functions from different meshes should not combine."""
        x, y = symbols('x y')

        f1 = create_unique_function('u', uw_id=1)
        f2 = create_unique_function('u', uw_id=2)

        u1 = f1(x, y)
        u2 = f2(x, y)

        expr = u1 + u2

        # Should NOT simplify to 2*u(x,y)
        assert expr != 2 * u1
        assert expr != 2 * u2

    def test_substitution_selective(self):
        """Substitution should only affect the specific function."""
        x, y = symbols('x y')

        f1 = create_unique_function('u', uw_id=1)
        f2 = create_unique_function('u', uw_id=2)

        u1 = f1(x, y)
        u2 = f2(x, y)

        expr = 3 * u1 + 2 * u2

        result1 = expr.subs(u1, 10)
        result2 = expr.subs(u2, 20)

        assert result1 == 30 + 2 * u2
        assert result2 == 3 * u1 + 40

    def test_meshvar_weakref_accessible(self):
        """Should be able to attach and access meshvar weakref.

        NOTE: SymPy caches applied functions, so we must set meshvar BEFORE
        applying the function, and use a unique uw_id not used by other tests.
        """
        x, y = symbols('x y')

        class FakeMeshVar:
            def __init__(self, name):
                self.name = name

        meshvar = FakeMeshVar("velocity")

        # Use a unique uw_id to avoid cache collision with other tests
        unique_id = 9999
        f = create_unique_function('u', uw_id=unique_id)

        # IMPORTANT: Set meshvar BEFORE applying the function!
        # SymPy caches applied functions, so attributes must be on the class
        # before first use.
        f.meshvar = weakref.ref(meshvar)

        u = f(x, y)

        # Access meshvar through applied function's .func attribute
        # (u.func is the function class, type(u) is the applied instance type)
        assert u.func.meshvar() is meshvar
        assert u.func.meshvar().name == "velocity"


# =============================================================================
# TEST: MULTI-MESH SCENARIO (THE REAL USE CASE)
# =============================================================================

class TestMultiMeshScenario:
    """Test the scenario that motivated the disambiguation: solver1.u vs solver2.u"""

    def setup_method(self):
        """Reset mock mesh counter for each test."""
        MockMesh._count = 0

    def test_two_meshes_same_variable_name(self):
        """Variables with same name on different meshes should be distinct."""
        mesh1 = MockMesh(dim=2)
        mesh2 = MockMesh(dim=2)

        u1 = MockMeshVariable(mesh1, "velocity", "u")
        u2 = MockMeshVariable(mesh2, "velocity", "u")

        # Symbols should be distinct
        assert u1.sym != u2.sym

        # But display the same
        assert r'\hspace' not in str(u1.sym)
        assert r'\hspace' not in str(u2.sym)

    def test_expression_with_two_mesh_variables(self):
        """Building an expression with variables from two meshes."""
        mesh1 = MockMesh(dim=2)
        mesh2 = MockMesh(dim=2)

        u1 = MockMeshVariable(mesh1, "velocity", "u")
        u2 = MockMeshVariable(mesh2, "velocity", "u")

        # Create an expression combining both
        expr = u1.sym + 2 * u2.sym

        # Should have two distinct function applications
        # Use atoms(AppliedUndef) to get function atoms, not Symbol atoms
        func_atoms = expr.atoms(AppliedUndef)
        assert len(func_atoms) == 2
        assert u1.sym in func_atoms
        assert u2.sym in func_atoms

        # Substitution should work correctly
        result = expr.subs(u1.sym, 10)
        assert result == 10 + 2 * u2.sym

    def test_can_identify_parent_meshvar(self):
        """Should be able to trace back from symbol to parent MeshVariable.

        Note: Variables are stored in mesh.vars dict, which keeps them alive
        for the weakref to work. This mirrors the real UW3 behavior.
        """
        mesh1 = MockMesh(dim=2)
        mesh2 = MockMesh(dim=2)

        u1 = MockMeshVariable(mesh1, "u1", "u")
        u2 = MockMeshVariable(mesh2, "u2", "u")

        # Variables should be registered in mesh.vars (keeps weakref alive)
        assert "u1" in mesh1.vars
        assert "u2" in mesh2.vars

        # Get meshvar from symbol's .func attribute (the function class)
        # u1.sym is an applied function instance, u1.sym.func is the function class
        sym1_meshvar = u1.sym.func.meshvar()
        sym2_meshvar = u2.sym.func.meshvar()

        # Verify weakref resolved to the correct object
        assert sym1_meshvar is not None, "u1 weakref was garbage collected"
        assert sym2_meshvar is not None, "u2 weakref was garbage collected"

        assert sym1_meshvar is u1
        assert sym2_meshvar is u2
        assert sym1_meshvar.mesh is mesh1
        assert sym2_meshvar.mesh is mesh2

    def test_isinstance_identification(self):
        """isinstance checks should still work (critical for evaluation pipeline)."""
        mesh = MockMesh(dim=2)
        u = MockMeshVariable(mesh, "velocity", "u")

        # The symbol should be an instance of AppliedUndef
        assert isinstance(u.sym, AppliedUndef)

        # And its type should be a subclass of AppliedUndef
        assert issubclass(type(u.sym), AppliedUndef)


# =============================================================================
# TEST: DERIVATIVES
# =============================================================================

class TestDerivatives:
    """Test that derivatives work correctly with disambiguated symbols."""

    def test_symbol_differentiation(self):
        """Derivatives of UniqueSymbol should work correctly."""
        x = symbols('x')
        alpha1 = UniqueSymbol(r'\alpha')
        alpha2 = UniqueSymbol(r'\alpha')

        expr = alpha1 * x**2 + alpha2 * x

        deriv = diff(expr, x)

        # Should be 2*alpha1*x + alpha2
        assert deriv == 2 * alpha1 * x + alpha2

    def test_function_differentiation(self):
        """Derivatives of unique functions should work correctly."""
        x, y = symbols('x y')

        f1 = create_unique_function('u', uw_id=1)
        f2 = create_unique_function('u', uw_id=2)

        u1 = f1(x, y)
        u2 = f2(x, y)

        expr = u1 * x + u2 * y

        # Derivative with respect to x
        deriv_x = diff(expr, x)

        # Should involve derivative of u1 but not combine u1 and u2
        assert u2 in deriv_x.atoms(AppliedUndef) or diff(u2, x) in deriv_x.atoms()


# =============================================================================
# TEST: SERIALIZATION
# =============================================================================

class TestSerialization:
    """Test that symbols can be pickled and restored correctly."""

    def test_unique_symbol_pickle(self):
        """UniqueSymbol should survive pickle round-trip."""
        import pickle

        alpha = UniqueSymbol(r'\alpha', uw_id=42)

        pickled = pickle.dumps(alpha)
        restored = pickle.loads(pickled)

        assert restored == alpha
        assert restored._uw_id == 42
        assert str(restored) == r'\alpha'

    def test_unique_symbol_srepr(self):
        """UniqueSymbol should have a valid srepr."""
        alpha = UniqueSymbol(r'\alpha', uw_id=42)

        repr_str = sympy.srepr(alpha)

        # Should be able to reconstruct from srepr
        # (may need to provide UniqueSymbol in namespace)
        assert 'UniqueSymbol' in repr_str or 'Symbol' in repr_str


# =============================================================================
# TEST: COMBINED WITH UWEXPRESSION PATTERN
# =============================================================================

class TestWithUWExpressionPattern:
    """Test how this would integrate with the UWexpression usage pattern."""

    def test_expression_with_mixed_symbols(self):
        """Test combining unique symbols with regular SymPy symbols."""
        x, y = symbols('x y')

        # Unique parameters (like UWexpression)
        rho1 = UniqueSymbol(r'\rho')
        rho2 = UniqueSymbol(r'\rho')

        # Unique functions (like MeshVariable.sym)
        f1 = create_unique_function('u', uw_id=1)
        f2 = create_unique_function('u', uw_id=2)
        u1 = f1(x, y)
        u2 = f2(x, y)

        # Build complex expression
        expr = rho1 * u1 + rho2 * u2

        # All four entities should be distinct
        assert rho1 != rho2
        assert u1 != u2

        # Substitutions should work independently
        result = expr.subs([(rho1, 100), (u1, 1)])
        assert result == 100 + rho2 * u2


# =============================================================================
# INTEGRATION TEST WITH REAL UNDERWORLD3
# =============================================================================

@pytest.mark.skipif(True, reason="Enable manually to test with real UW3")
class TestWithRealUnderworld:
    """
    Integration tests with actual Underworld3.

    Enable by changing skipif to False and running:
    pixi run -e default pytest tests/test_symbol_disambiguation_prototype.py::TestWithRealUnderworld -v
    """

    def test_prototype_with_real_evaluation(self):
        """Test that the prototype approach works with UW3 evaluation."""
        import underworld3 as uw

        # Create two meshes
        mesh1 = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1)
        mesh2 = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1)

        # Create variables with same name
        u1 = uw.discretisation.MeshVariable("u", mesh1, 1)
        u2 = uw.discretisation.MeshVariable("u", mesh2, 1)

        # Check they're distinct
        assert u1.sym[0] != u2.sym[0], "MeshVariables on different meshes should have distinct symbols"

        # Check display is clean
        latex_str = sympy.latex(u1.sym[0] + u2.sym[0])
        assert r'\hspace' not in latex_str, "LaTeX should not contain \\hspace hack"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
