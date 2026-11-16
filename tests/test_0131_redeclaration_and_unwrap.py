"""A re-declared mesh variable refuses to change; ``uw.unwrap`` expands completely (#1118, #824).

#1118: a ``MeshVariable`` is identified by its name on the mesh. Declaring the
name again returned the existing variable and dropped every new argument, with
only a print, so a caller asking for two components, degree 1 and the symbol
``\\bar{f}`` got one component, degree 2 and ``{f}``. A re-declaration that
would change the kind (mesh or integration-point variable), the type, the
number of components, the degree, the continuity or the symbol now raises. One
that asks for what the variable already is returns it (the rule #780 set for
expressions); the type and components are compared as the variable resolves
them when it is built, so a type that overrides the count is the same request.

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
    ("type", dict(num_components=2)),
    ("degree", dict(num_components=1, degree=1)),
    ("continuity", dict(num_components=1, degree=2, continuous=False)),
    ("symbol", dict(num_components=1, degree=2, varsymbol=r"\bar{f}")),
])
def test_a_redeclaration_that_changes_the_variable_refuses(mesh, change, kwargs):
    first = uw.discretisation.MeshVariable("f", mesh, 1, degree=2)
    kwargs.setdefault("degree", 2)
    n = kwargs.pop("num_components")
    with pytest.raises(ValueError, match=rf"would change its {change}\. "):
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


@pytest.mark.tier_b
def test_a_change_of_components_alone_refuses(mesh):
    uw.discretisation.MeshVariable("m", mesh, (2, 3), vtype=uw.VarType.MATRIX)
    with pytest.raises(ValueError, match=r"would change its number of components\. "):
        uw.discretisation.MeshVariable("m", mesh, (3, 2), vtype=uw.VarType.MATRIX)


@pytest.mark.tier_b
def test_a_type_only_change_refuses(mesh):
    # (2, 2) components either way; the type alone differs.
    uw.discretisation.MeshVariable("t", mesh, (2, 2), vtype=uw.VarType.TENSOR)
    with pytest.raises(ValueError, match="would change its type"):
        uw.discretisation.MeshVariable("t", mesh, (2, 2), vtype=uw.VarType.SYM_TENSOR)


@pytest.mark.tier_b
@pytest.mark.parametrize("declare", [
    # the type overrides the component count, as it does when the variable is built
    lambda m: uw.discretisation.MeshVariable("u", m, m.dim, vtype=uw.VarType.SCALAR),
    lambda m: uw.discretisation.IntegrationPointVariable("s", m, vtype=uw.VarType.VECTOR),
])
def test_the_same_declaration_twice_returns_the_variable(mesh, declare):
    first = declare(mesh)
    assert declare(mesh)._base_var is first._base_var


@pytest.mark.tier_b
def test_a_manifold_vector_redeclared_with_dim_components_returns_it():
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.5)
    first = uw.discretisation.MeshVariable("w", sphere, sphere.cdim, degree=1)
    assert uw.discretisation.MeshVariable("w", sphere, sphere.dim, degree=1)._base_var is first._base_var


@pytest.mark.tier_b
def test_an_integration_point_variable_cannot_be_redeclared_as_a_mesh_variable(mesh):
    uw.discretisation.IntegrationPointVariable("q", mesh)
    with pytest.raises(ValueError, match="would change its kind"):
        uw.discretisation.MeshVariable("q", mesh, 1, degree=0, continuous=False)


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
