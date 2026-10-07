"""The JIT's graph lowering (#823, tier 2): nodes, their derivatives, and the C emitted
from them.

Each non-constant ``UWexpression`` becomes a node, an applied function of the leaves its
value depends on, and SymPy's chain rule differentiates through it by ``fdiff``. Each
test closes one risk of the design note
(``docs/developer/design/jit-shared-graph-codegen.md``), against the expanded tree as
the reference.
"""
import numpy as np
import pytest
import sympy
from sympy.core.function import AppliedUndef
from sympy.vector.scalar import BaseScalar

import underworld3 as uw
import underworld3.function.expressions as ex
from underworld3.function import diff_wrt_field
from underworld3.utilities import _jit_graph as jg

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture(scope="module")
def box():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    T = uw.discretisation.MeshVariable("T0024", mesh, 1, degree=1)
    v = uw.discretisation.MeshVariable("V0024", mesh, 2, degree=2)
    return mesh, T, v


def _numbers(exprs, seed=0, rest=()):
    """Every leaf of the expanded ``exprs`` (field values and gradients, coordinates)
    as a number, the same number wherever it occurs; leaves in ``rest`` are zero."""
    rng = np.random.default_rng(seed)
    leaves = set()
    for e in exprs:
        leaves |= e.atoms(AppliedUndef)
        leaves |= {s for s in e.free_symbols if isinstance(s, BaseScalar)}
    leaves = sorted(leaves, key=sympy.default_sort_key)
    return {s: sympy.Float(0.0 if s in rest else rng.uniform(0.2, 1.5)) for s in leaves}


def _value(e, numbers):
    e = ex.unwrap_expression(e, mode="nondimensional")
    return complex(e.xreplace(numbers).evalf())


def test_a_derivative_through_nodes_is_the_derivative_of_the_tree(box):
    """By field value, by field gradient and by coordinate, through
    ``diff_wrt_field`` and through ``sympy.diff`` (which swaps the variable for a
    ``Dummy`` before ``fdiff`` sees it: a derivative keyed by the leaf OBJECT would be
    zero). The coordinate case holds a field and a coordinate in one node: a slot
    derivative taken as a total derivative would count the field's gradient twice."""
    mesh, T, v = box
    x, y = mesh.N.x, mesh.N.y
    u, ux = T.sym[0], T.sym[0].diff(x)
    k = uw.expression(r"k_{0024a}", 0.7, "constant")
    # by field value and gradient
    a = uw.expression(r"a_{0024a}", u ** 2 + u * ux, "inner")
    b = uw.expression(r"b_{0024a}", sympy.exp(k * a) + a * u + sympy.sqrt(a + ux ** 2),
                      "outer")
    # by coordinate: a field and a coordinate in one node (a gradient leaf would need
    # a second derivative of the field, which Underworld refuses on either route)
    c = uw.expression(r"c_{0024a}", u ** 2 + x * u + y, "inner, with coordinates")
    d = uw.expression(r"d_{0024a}", sympy.exp(k * c) + c * u * x, "outer, with coordinates")
    for law, wrt in ((b * u + a, (u, ux)), (d * u + c, (u, x, y))):
        graph = jg.KernelGraph()
        lowered = graph.lower(law)
        tree = ex.unwrap_expression(law, mode="symbolic_keep_constants")
        assert lowered.atoms(jg._KernelNode), "nothing was lowered to a node"
        for w in wrt:
            d_tree = diff_wrt_field(tree, w)
            n = _numbers([d_tree, tree])
            reference = _value(d_tree, n)
            assert abs(reference) > 1.0e-3, w
            for dg in (diff_wrt_field(lowered, w), sympy.diff(lowered, w)):
                assert abs(_value(dg, n) - reference) <= 1.0e-12 * abs(reference), (w, dg)


def test_second_derivatives_and_a_constant_sensitivity_pass_through_nodes(box):
    mesh, T, v = box
    u = T.sym[0]
    c = uw.expression(r"c_{0024b}", 1.3, "constant")
    a = uw.expression(r"a_{0024b}", c * u ** 3 + sympy.log(u + c), "inner")
    law = a ** 2
    lowered = jg.KernelGraph().lower(law)
    tree = ex.unwrap_expression(law, mode="symbolic_keep_constants")
    for d_graph, d_tree in (
            (diff_wrt_field(diff_wrt_field(lowered, u), u),
             diff_wrt_field(diff_wrt_field(tree, u), u)),
            (sympy.diff(lowered, c), sympy.diff(tree, c))):
        n = _numbers([d_tree])
        reference = _value(d_tree, n)
        assert abs(reference) > 1.0e-3
        assert abs(_value(d_graph, n) - reference) <= 1.0e-12 * abs(reference)


def test_two_constants_with_one_name_stay_two(box):
    """Two viscosities are both called eta (one per material). Nodes are identified by
    body, and SymPy's equality tells the two apart, so each keeps its own slot."""
    mesh, T, v = box
    from underworld3.utilities._jitextension import _manifest_from

    vv = uw.discretisation.MeshVariable("V0024c", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0024c", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=vv, pressureField=p)
    lower = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="lo")
    lower.Parameters.shear_viscosity_0 = 1.0
    upper = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="up")
    upper.Parameters.shear_viscosity_0 = 1000.0
    a = lower.Parameters.shear_viscosity_0
    b = upper.Parameters.shear_viscosity_0
    assert a != b and a.name == b.name
    u = T.sym[0]
    ea = uw.expression(r"e_{0024c}", a * u ** 2, "first material")
    eb = uw.expression(r"e_{0024c}", b * u ** 2, "second material",
                       _unique_name_generation=True)
    lowered = jg.lower_callbacks([ea + 2 * eb], mesh)
    manifest, subs = _manifest_from(jg.constant_leaves(lowered))
    assert len(manifest) == 2
    nodes = lowered[0].atoms(jg._KernelNode)
    assert len({n.func for n in nodes}) == 2, nodes


def test_a_changed_body_is_lowered_afresh_without_clearing_the_cache(box):
    """Each lowering's node classes carry its serial in their SymPy identity, so
    SymPy's cache cannot hand back a node whose body is an earlier version. The hard
    case: two constants share a display name and the re-declared atom swaps their
    roles, so the new node has the old one's name AND arguments."""
    mesh, T, v = box
    u = T.sym[0]
    c1 = uw.expression(r"c_{0024d}", 2.0, "first")
    c2 = uw.expression(r"c_{0024d}", 5.0, "second", _unique_name_generation=True)
    assert c1 != c2 and c1.name == c2.name
    a = uw.expression(r"a_{0024d}", c1 * u ** 3 + c2 * u, "re-declared")
    first = jg.KernelGraph().lower(a)
    a.sym = c2 * u ** 3 + c1 * u
    second = jg.KernelGraph().lower(a)
    assert first.func.__name__ == second.func.__name__   # the hard case holds
    assert first.args == second.args
    assert jg.expand_nodes(second) == c2 * u ** 3 + c1 * u
    assert jg.expand_nodes(diff_wrt_field(second, u)) == 3 * c2 * u ** 2 + c1


def test_the_guarded_lowering_is_the_guarded_tree(box, monkeypatch):
    """The Newton source on the graph (the sqrt guard in each node body) evaluates to
    the guarded tree, and so does its tangent, including at a state of rest where the
    unguarded tangent is 0/0."""
    from underworld3.cython.generic_solvers import _jacobian_unwrap

    mesh, T, v = box
    vv = uw.discretisation.MeshVariable("V0024e", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0024e", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=vv, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = uw.expression(r"C_{0024e}", 0.5) + 0.3 * p.sym[0]
    cm.Parameters.yield_stress_min = uw.expression(r"\tau_{0024e}", 0.01)
    flux = sympy.Matrix(stokes.F1.sym)

    monkeypatch.setenv("UW_JIT_GRAPH", "0")
    tree = _jacobian_unwrap(flux)
    monkeypatch.setenv("UW_JIT_GRAPH", "1")
    graph = _jacobian_unwrap(flux)
    assert graph.atoms(jg._KernelNode), "the Newton source was not lowered"

    L = stokes.Unknowns.L
    rest = {L[i, j] for i in range(2) for j in range(2)}
    for state in (dict(seed=1), dict(seed=2, rest=rest)):
        for i in range(2):
            for j in range(2):
                d_tree = diff_wrt_field(tree[i, j], L[0, 1])
                d_graph = diff_wrt_field(graph[i, j], L[0, 1])
                n = _numbers([tree[i, j], d_tree], **state)
                for t, g in ((tree[i, j], graph[i, j]), (d_tree, d_graph)):
                    vt, vg = _value(t, n), _value(g, n)
                    assert np.isfinite(vt) and np.isfinite(vg), (state, t)
                    assert abs(vg - vt) <= 1.0e-12 * max(abs(vt), 1.0), (state, i, j)


def _header(solver, monkeypatch, route):
    """The generated header of ``solver`` on ``route``, with the module name and the
    symbol prefix canonicalised as ``getext`` canonicalises them."""
    import underworld3.utilities._jitextension as jx

    seen = {}
    generate = jx.generate_c_source

    def keep(*a, **k):
        modname, codeguys, diag = generate(*a, **k)
        seen["h"] = dict(codeguys)["cy_ext.h"].replace(modname, "M").replace(
            diag["randstr"], "R")
        return modname, codeguys, diag

    monkeypatch.setattr(jx, "generate_c_source", keep)
    monkeypatch.setenv("UW_JIT_GRAPH", route)
    solver.is_setup = False
    solver._setup_pointwise_functions()
    return seen["h"]


def _poisson(name, mesh):
    u = uw.discretisation.MeshVariable("U" + name, mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    k0 = uw.expression(r"k_{0024f}", 2.0, "conductivity")
    g = uw.expression(r"g_{0024f}", sympy.sqrt(1 + u.sym[0].diff(mesh.N.x) ** 2), "slope")
    pois.constitutive_model.Parameters.diffusivity = k0 * g + u.sym[0] ** 2
    pois.f = 1.0
    pois.consistent_jacobian = True
    return pois


def test_the_emitted_source_does_not_depend_on_what_came_before(monkeypatch):
    """The C is a function of the mathematics and the data layout: the same law,
    declared after a preamble of unrelated objects (every creation counter shifted)
    and declared twice, emits the same header."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    first = _header(_poisson("0024f", mesh), monkeypatch, "1")
    for k in range(7):
        uw.expression(rf"junk_{{0024f,{k}}}", float(k), "preamble")
    other = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    uw.discretisation.MeshVariable("J0024f", other, 1, degree=1)
    mesh2 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    second = _header(_poisson("0024f", mesh2), monkeypatch, "1")
    assert first == second
    assert "uwt_0" in first, "the kernel has no temporaries: nothing was lowered"


def test_a_law_with_no_named_quantity_emits_the_tree_route_source(monkeypatch):
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    u = uw.discretisation.MeshVariable("U0024g", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = 1.0
    pois.f = 2.0
    assert _header(pois, monkeypatch, "0") == _header(pois, monkeypatch, "1")


def test_the_manifest_from_the_leaves_is_the_scanned_manifest(box):
    """The constants the lowered kernels read are the constants the expressions hold,
    in the same slots."""
    from underworld3.utilities._jitextension import _extract_constants, _manifest_from

    mesh, T, v = box
    u = T.sym[0]
    c1 = uw.expression(r"c_{0024h,1}", 0.5, "constant")
    c2 = uw.expression(r"c_{0024h,2}", 2.0, "constant")
    c3 = uw.expression(r"c_{0024h,3}", 3.0, "constant in a constant")
    c4 = uw.expression(r"c_{0024h,4}", c3 * 2, "constant of a constant")
    a = uw.expression(r"a_{0024h}", c1 * u + c4 * u ** 2, "inner")
    b = uw.expression(r"b_{0024h}", a / (c2 + a ** 2), "outer")
    fns = (b, sympy.Matrix([[b * u, a]]))
    scanned, _ = _extract_constants(fns, mesh)
    leaves, _ = _manifest_from(jg.constant_leaves(jg.lower_callbacks(fns, mesh)))
    assert [e for _, e in leaves] == [e for _, e in scanned]
    assert len(scanned) == 3     # c4 is a slot of its own; c3 is folded into it
