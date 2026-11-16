"""A re-declared mesh variable refuses to change; ``uw.unwrap`` expands completely (#1118, #824).

#1118: a ``MeshVariable`` is identified by its name on the mesh. Declaring the
name again returned the existing variable and dropped every new argument, with
only a print, so a caller asking for two components, degree 1 and the symbol
``\\bar{f}`` got one component, degree 2 and ``{f}``. A re-declaration that
would change the number of components, the type, the degree, the continuity or
the symbol now raises. One that asks for what the variable already is returns
it (the rule #780 set for expressions).

#824: ``uw.unwrap`` passed ``keep_constants`` and ``return_self`` positionally
into ``expressions.unwrap(fn, depth, keep_constants, return_self)``, so
``keep_constants`` arrived as ``depth`` (one level of a nested expression was
expanded, or none) and ``return_self`` as ``keep_constants``. Each test pins a
two-level expression to its complete expansion, ``6 x``.
"""

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


@pytest.fixture
def mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5)


@pytest.mark.tier_b
@pytest.mark.parametrize("change, kwargs", [
    ("components", dict(num_components=2)),
    ("vtype", dict(num_components=2, vtype=uw.VarType.VECTOR)),
    ("degree", dict(num_components=1, degree=1)),
    ("continuity", dict(num_components=1, degree=2, continuous=False)),
    ("symbol", dict(num_components=1, degree=2, varsymbol=r"\bar{f}")),
])
def test_a_redeclaration_that_changes_the_variable_refuses(mesh, change, kwargs):
    first = uw.discretisation.MeshVariable("f", mesh, 1, degree=2)
    kwargs.setdefault("degree", 2)
    n = kwargs.pop("num_components")
    with pytest.raises(ValueError, match="already exists"):
        uw.discretisation.MeshVariable("f", mesh, n, **kwargs)
    # the existing variable is untouched
    held = mesh.vars["f"]
    assert (held.num_components, held.degree, held.continuous, held.symbol) == (1, 2, True, "{f}")
    assert first.sym[0] == held.sym[0]


@pytest.mark.tier_b
def test_a_redeclaration_of_the_same_variable_returns_it(mesh):
    first = uw.discretisation.MeshVariable("g", mesh, 2, degree=2, varsymbol=r"\mathbf{g}")
    again = uw.discretisation.MeshVariable("g", mesh, 2, degree=2, varsymbol=r"\mathbf{g}")
    assert again._base_var is first._base_var


def _nested():
    x = sympy.Symbol("x")
    inner = uw.expression(r"a_{in824}", 2 * x, "inner")
    outer = uw.expression(r"a_{out824}", 3 * inner, "outer")
    return x, outer


@pytest.mark.tier_b
def test_unwrap_expands_a_nested_expression_completely():
    x, outer = _nested()
    assert sympy.simplify(uw.unwrap(outer) - 6 * x) == 0


@pytest.mark.tier_b
def test_unwrap_without_constants_expands_a_nested_expression_completely():
    x, outer = _nested()
    assert sympy.simplify(uw.unwrap(outer, keep_constants=False) - 6 * x) == 0
