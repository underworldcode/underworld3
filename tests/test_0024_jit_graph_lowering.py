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
    """A node's derivative is a node, so it differentiates again; and a derivative
    with respect to a constant atom (a sensitivity) reaches every body that reads it."""
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


def _guarded_tree(e):
    """The expanded-tree Newton source the graph replaced, with the guard it had then:
    every non-constant atom expanded, then 1e-36 added to the base of every
    half-integer power with free symbols, by SymPy's own replace. (Both routes have
    guarded every fractional power since #841.)"""
    eps2 = sympy.Float(1.0e-36)
    tree = ex.unwrap_expression(e, mode="symbolic_keep_constants")
    return tree.replace(
        lambda n: (n.is_Pow and n.exp.is_Rational and n.exp.q == 2
                   and n.args[0].free_symbols),
        lambda n: sympy.Pow(n.args[0] + eps2, n.exp))


def test_the_guarded_lowering_is_the_guarded_tree(box):
    """The Newton source on the graph (the sqrt guard in each node body) evaluates to
    the guarded tree, and so does its tangent, including at a state of rest where the
    unguarded tangent is 0/0.

    Except where the tree escapes its own guard: in a power law on a named invariant,
    eta = edot**(1/n - 1) with edot = sqrt(g), the tree merges the powers into
    g**((1/n - 1)/2), not a half-integer power, so its Newton flux is NaN at rest. The
    graph keeps edot a node with a guarded body and stays finite there."""
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
    viscoplastic = sympy.Matrix(stokes.F1.sym)

    # a power law on a named strain-rate invariant, singular at rest unless guarded
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    edot = uw.expression(r"\dot\varepsilon_{0024e}", stokes.Unknowns.Einv2, "invariant")
    n = uw.expression(r"n_{0024e}", 3, "stress exponent")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = edot ** (1 / n - 1)
    power_law = sympy.Matrix(stokes.F1.sym)

    L = stokes.Unknowns.L
    rest = {L[i, j] for i in range(2) for j in range(2)}
    for flux, tree_escapes_at_rest in ((viscoplastic, False), (power_law, True)):
        tree = flux.applyfunc(_guarded_tree)
        graph = _jacobian_unwrap(flux, route="graph")
        assert graph.atoms(jg._KernelNode), "the Newton source was not lowered"
        escaped = _same_guarded_values(tree, graph, L, rest)
        assert escaped == tree_escapes_at_rest, escaped


def _same_guarded_values(tree, graph, L, rest):
    """Graph against tree at a random state and at rest; returns whether the tree was
    non-finite anywhere (only ever at rest), where the graph must be finite."""
    escaped = False
    for state in (dict(seed=1), dict(seed=2, rest=rest)):
        for i in range(2):
            for j in range(2):
                d_tree = diff_wrt_field(tree[i, j], L[0, 1])
                d_graph = diff_wrt_field(graph[i, j], L[0, 1])
                n = _numbers([tree[i, j], d_tree], **state)
                for t, g in ((tree[i, j], graph[i, j]), (d_tree, d_graph)):
                    vt, vg = _value(t, n), _value(g, n)
                    assert np.isfinite(vg), (state, g)
                    if not np.isfinite(vt):
                        assert "rest" in state, (state, t)
                        escaped = True
                        continue
                    assert abs(vg - vt) <= 1.0e-12 * max(abs(vt), 1.0), (state, i, j)
    return escaped


def _header(solver, monkeypatch):
    """The generated header of ``solver``, with the module name and the symbol prefix
    canonicalised as ``getext`` canonicalises them."""
    import underworld3.utilities._jitextension as jx

    seen = {}
    generate = jx.generate_c_source

    def keep(*a, **k):
        modname, codeguys, diag = generate(*a, **k)
        seen["h"] = dict(codeguys)["cy_ext.h"].replace(modname, "M").replace(
            diag["randstr"], "R")
        return modname, codeguys, diag

    monkeypatch.setattr(jx, "generate_c_source", keep)
    solver.jit_route = "graph"
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
    first = _header(_poisson("0024f", mesh), monkeypatch)
    for k in range(7):
        uw.expression(rf"junk_{{0024f,{k}}}", float(k), "preamble")
    other = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    uw.discretisation.MeshVariable("J0024f", other, 1, degree=1)
    mesh2 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    second = _header(_poisson("0024f", mesh2), monkeypatch)
    assert first == second
    assert "uwt_0" in first, "the kernel has no temporaries: nothing was lowered"
    # each temporary carries the name of the quantity it computes
    assert "/* g_{0024f} */" in first


def test_a_law_with_no_named_quantity_has_no_temporaries(monkeypatch):
    """A constant law has no node to lower: its kernels are the outputs alone, as the
    expanded tree printed them."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    u = uw.discretisation.MeshVariable("U0024g", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    # named constants: leaves, not nodes
    pois.constitutive_model.Parameters.diffusivity = uw.expression(r"k_{0024g}", 2.0)
    pois.f = uw.expression(r"f_{0024g}", 1.0)
    header = _header(pois, monkeypatch)
    assert "out[0]" in header and "uwt_" not in header


def test_the_manifest_is_the_constants_the_kernels_read(box):
    """Every constant atom a lowered kernel reads, at any depth, has a slot, ordered by
    name; a constant inside a constant is folded into its holder's slot. The expanded
    route's scan gives the same manifest."""
    from underworld3.utilities._jitextension import _extract_constants, _manifest_of

    mesh, T, v = box
    u = T.sym[0]
    c1 = uw.expression(r"c_{0024h,1}", 0.5, "constant")
    c2 = uw.expression(r"c_{0024h,2}", 2.0, "constant")
    c3 = uw.expression(r"c_{0024h,3}", 3.0, "constant in a constant")
    c4 = uw.expression(r"c_{0024h,4}", c3 * 2, "constant of a constant")
    a = uw.expression(r"a_{0024h}", c1 * u + c4 * u ** 2, "inner")
    b = uw.expression(r"b_{0024h}", a / (c2 + a ** 2), "outer")
    fns = (b, sympy.Matrix([[b * u, a]]))
    graph, placeholders = _manifest_of(jg.lower_callbacks(fns, mesh))
    assert [e for _, e in graph] == [c1, c2, c4]
    assert len(set(placeholders.values())) == 3
    # the expanded route's scan finds the same slots
    expanded, _ = _extract_constants(fns, mesh)
    assert [e for _, e in expanded] == [c1, c2, c4]


def test_a_repeated_condition_stays_a_condition(box):
    """A condition that repeats inside one body (two Piecewise on the same test) is
    shared by the body's common sub-expression split; it must stay a Boolean, not
    become a node, which Piecewise refuses as a condition (the fault-network laws,
    test_0850 and test_0851)."""
    mesh, T, v = box
    x = mesh.N.x
    u = T.sym[0]
    a = uw.expression(r"a_{0024i}",
                      sympy.Piecewise((u, x > 0.5), (u ** 2, True))
                      + sympy.Piecewise((2 * u, x > 0.5), (u ** 3, True)),
                      "two branches on one test")
    lowered = jg.KernelGraph().lower(a * u)
    tree = ex.unwrap_expression(a * u, mode="symbolic_keep_constants")
    for xv in (0.2, 0.8):
        n = _numbers([tree, diff_wrt_field(tree, u)])
        n[x] = sympy.Float(xv)
        for t, g in ((tree, lowered), (diff_wrt_field(tree, u), diff_wrt_field(lowered, u))):
            assert abs(_value(g, n) - _value(t, n)) <= 1.0e-12 * abs(_value(t, n))


def test_a_deep_law_is_compiled_as_one_temporary_per_layer(monkeypatch):
    """A law of twelve named layers, each using the one below twice: the expanded tree
    doubles with every layer (4096 copies of the bottom), the graph has one temporary
    per layer. The cost of the tree must not come back by any route that expands
    nodes."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    u = uw.discretisation.MeshVariable("U0024j", mesh, 1, degree=1)
    c = uw.expression(r"c_{0024j}", 0.5, "constant")
    layer = uw.expression(r"e_{0024j,0}", 1 + u.sym[0] ** 2, "bottom")
    for k in range(1, 13):
        layer = uw.expression(rf"e_{{0024j,{k}}}", c * layer + sympy.sqrt(layer), "layer")
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = layer
    pois.f = 1.0
    pois.consistent_jacobian = True
    header = _header(pois, monkeypatch)
    assert header.count("const double uwt_") <= 12 * 8, header.count("const double uwt_")
    assert len(header) < 60_000, len(header)


def test_a_coordinate_without_its_c_name_is_spelled_from_its_index():
    """A coordinate leaf can be a UWCoordinate that SymPy's cache returned for its
    equal base scalar, or a fresh base scalar, without the C name the mesh set: it is
    written from its index and system (test_0850 and test_0851 met it)."""
    from underworld3.utilities._jitextension import _spell_leaf

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    x = mesh.X[1]                        # the UWCoordinate wrapping mesh.N.y
    base = x._original_base_scalar
    saved = base.__dict__.pop("_ccodestr")
    try:
        assert _spell_leaf(x, {}, {}) == "petsc_x[1]"
        assert _spell_leaf(base, {}, {}) == "petsc_x[1]"
    finally:
        base._ccodestr = saved
    normal = mesh._Gamma.base_scalars()[0]
    assert _spell_leaf(normal, {}, {}) == "petsc_n[0]"


def test_a_mesh_coordinate_beside_a_field_keeps_its_partial_derivative(box):
    """``mesh.X`` coordinates are UWCoordinates, equal to the base scalars they wrap. In
    a named quantity that reads one beside a field, the per-body common sub-expression
    split rebuilt the coordinate with a cloned coordinate system, a symbol no longer
    equal to ``mesh.N.x``: the derivative by the coordinate lost its explicit term."""
    mesh, T, v = box
    X = mesh.X
    u = T.sym[0]
    q = uw.expression(r"q_{0024k}", X[0] * u + sympy.sin(X[1]) * u ** 2,
                      "coordinates beside a field")
    lowered = jg.KernelGraph().lower(q * u)
    tree = ex.unwrap_expression(q * u, mode="symbolic_keep_constants")
    for w in (mesh.N.x, mesh.N.y):
        d_tree = sympy.diff(tree, w)
        n = _numbers([d_tree, tree])
        reference = _value(d_tree, n)
        assert abs(reference) > 1.0e-3
        assert abs(_value(sympy.diff(lowered, w), n) - reference) <= 1.0e-12 * abs(reference)


def test_a_matrix_or_vector_valued_atom_is_expanded_in_place(box):
    """A node is one scalar temporary, so an atom whose value is a matrix or a vector is
    expanded in place, as the tree expanded it."""
    mesh, T, v = box
    u = T.sym[0]
    row = uw.expression(r"M_{0024l}", sympy.Matrix([[u, u ** 2]]), "a row")
    lowered = jg.lower_callbacks([row], mesh)[0]
    assert lowered.shape == (1, 2)
    assert jg.expand_nodes(lowered) == sympy.Matrix([[u, u ** 2]])
    arrow = uw.expression(r"A_{0024l}", u * mesh.N.i + u ** 2 * mesh.N.j, "a vector")
    lowered = jg.lower_callbacks([arrow], mesh)[0]
    assert lowered.shape == (2, 1)
    assert jg.expand_nodes(lowered) == sympy.Matrix([[u], [u ** 2]])


def test_constancy_is_decided_on_the_graph(box, monkeypatch):
    """Whether an atom is a constant is decided bottom-up on the graph, by structure,
    not by ``_is_truly_constant``: that unwraps an atom completely, a full expansion of
    everything under it, for every atom, which costs 2**depth on a law whose every
    layer reads the one below twice. The two agree except where a constant's current
    value collapses an expression to a number: ``(1 + T**2)**(-m) + 1`` at ``m = 0``
    is a node reading ``m``, so ``m`` ramps (test_0104)."""
    import underworld3.utilities._jitextension as jx

    mesh, T, v = box
    u = T.sym[0]
    x = mesh.X[0]
    c = uw.expression(r"c_{0024n}", 0.5, "constant")
    nested = uw.expression(r"d_{0024n}", 2 * c + 1, "constant of a constant")
    atoms = [
        c, nested,
        uw.expression(r"e_{0024n}", nested * u, "reads a field"),
        uw.expression(r"g_{0024n}", c * x, "reads a coordinate"),
        uw.expression(r"h_{0024n}", uw.quantity(3.0, "m/s"), "a quantity"),
        uw.expression(r"k_{0024n}", sympy.Matrix([[c, 2 * c]]), "a constant row"),
    ]
    expected = [jx._is_truly_constant(a, ex.UWexpression) for a in atoms]
    assert expected == [True, True, False, False, True, True], expected
    m = uw.expression(r"m_{0024n}", 0, "zero for now")
    collapsing = uw.expression(r"p_{0024n}", (1 + u ** 2) ** (-m) + 1, "2 while m is 0")
    assert jx._is_truly_constant(collapsing, ex.UWexpression)
    assert not jg.KernelGraph().is_constant(collapsing)

    calls = []
    original = jx._is_truly_constant
    monkeypatch.setattr(jx, "_is_truly_constant",
                        lambda *args: calls.append(args) or original(*args))
    graph = jg.KernelGraph()
    assert [graph.is_constant(a) for a in atoms] == expected
    layer = uw.expression(r"e_{0024n,0}", 1 + u ** 2, "bottom")
    for k in range(1, 17):
        layer = uw.expression(rf"e_{{0024n,{k}}}", c * layer + sympy.sqrt(layer), "layer")
    graph.lower(layer)
    assert not calls, f"{len(calls)} complete unwraps"


def test_a_number_symbol_in_a_temporary_is_declared_once(monkeypatch):
    """The C99 printer declares a number symbol (EulerGamma, Catalan) before the
    expression that reads it; inside a temporary that declaration was written into
    the temporary's own initialiser, which does not compile."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    u = uw.discretisation.MeshVariable("U0024m", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    k = uw.expression(r"k_{0024m}", 1 + sympy.EulerGamma * u.sym[0] ** 2 + sympy.Catalan,
                      "number symbols")
    pois.constitutive_model.Parameters.diffusivity = k
    pois.f = 1.0
    pois.consistent_jacobian = True
    header = _header(pois, monkeypatch)          # compiles
    assert "uwt_0" in header
    for body in header.split("\nvoid ")[1:]:
        assert body.count("const double EulerGamma =") <= 1


def test_a_name_cannot_close_its_comment(box):
    """Each temporary's comment is the name of its quantity; a name holding the
    comment terminator is written so that it cannot end the comment early."""
    from sympy.printing.c import c_code_printers
    from underworld3.utilities._jitextension import _print_kernel

    mesh, T, v = box
    t = jg._Temporary(0)
    printer = c_code_printers["c99"]()
    code = _print_kernel(printer, [(t, sympy.Float(2.0), "a */ b\nc")],
                         sympy.Matrix([[t]]), sympy.MatrixSymbol("out", 1, 1))
    first = code.splitlines()[0]
    assert first.endswith("/* a * / b c */") and first.count("*/") == 1, first
