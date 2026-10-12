"""The memoised unwrap, the memoised sqrt guard and the identity-walk symbol scan (#823)
give exactly what the algorithms they replace gave.

The old ``unwrap_expression`` iterated ``subs`` passes to a fixed point, and in
``symbolic_keep_constants`` mode ran ``_is_truly_constant`` (itself a full unwrap) for every
atom on every pass. On the Spiegelman notch viscosity that took 563 s; the memoised
version takes 0.11 s. These tests hold the replacement to STRUCTURAL identity (``srepr``)
with a copy of the old algorithm, on the constitutive laws the solvers compile, in every
mode. Identical expressions mean identical generated C and identical JIT cache keys.
"""
import pytest
import sympy

import underworld3 as uw
import underworld3.function.expressions as ex

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

MODES = ("nondimensional", "symbolic", "symbolic_keep_constants", "dimensional")


def _fixed_point_unwrap(expr, mode):
    """The algorithm #823 replaced, kept here as the reference."""
    if isinstance(expr, ex.UWQuantity) and not isinstance(expr, ex.UWexpression):
        return ex._unwrap_atom(expr, mode)
    result = expr
    nxt = ex._unwrap_expression_once(result, mode)
    it = 0
    while result is not nxt and it < 100:
        result = nxt
        nxt = ex._unwrap_expression_once(result, mode)
        it += 1
    return result


def _laws():
    """(name, expression) for the laws the solvers compile, built on a small box."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("V0022", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P0022", mesh, 1, degree=1)
    f = uw.discretisation.MeshVariable("F0022", mesh, 1, degree=0, continuous=False)
    out = []
    for law in ("min", "sqrt", "powermean"):
        s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
        cm = s.constitutive_model
        cm.Parameters.shear_viscosity_0 = 1.0 + f.sym[0]
        C = uw.expression(rf"C_{{{law}}}", 0.5, "cohesion")
        cm.Parameters.yield_stress = C + 0.3 * p.sym[0]
        cm.Parameters.yield_stress_min = 0.02 * C
        cm.Parameters.shear_viscosity_min = 1.0e-3 * (1.0 + x)
        if law != "min":
            cm.yield_mode = "softmin"
            cm.yield_smoother = law
            cm.yield_softness = 0.1
            cm.viscosity_min_rounding = 1.0e-4
        out.append((f"viscoplastic_{law}", cm.viscosity))
        out.append((f"viscoplastic_{law}_flux", s.F1.sym))
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=1)
    cm = s.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = sympy.Rational(1, 10)
    cm.Parameters.yield_stress = 0.3
    out.append(("viscoelastic_plastic_flux", s.F1.sym))
    u = uw.discretisation.MeshVariable("U0022", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = 1.0 + u.sym[0] ** 2
    out.append(("poisson_k_of_u_flux", pois.F1.sym))
    return out


@pytest.fixture(scope="module")
def laws():
    uw.reset_default_model()
    return _laws()


def _unwrap_each(expr, fn):
    if isinstance(expr, sympy.MatrixBase):
        return expr.applyfunc(fn)
    return fn(expr)


@pytest.mark.parametrize("mode", MODES)
def test_memoised_unwrap_is_the_fixed_point(laws, mode):
    for name, expr in laws:
        try:
            old = _unwrap_each(expr, lambda e: _fixed_point_unwrap(e, mode))
        except Exception as failure:
            # e.g. 'symbolic' mode on a parameter holding a units quantity: subs()
            # could not sympify it. The replacement must fail the same way.
            with pytest.raises(type(failure)):
                _unwrap_each(expr, lambda e: ex.unwrap_expression(e, mode=mode))
            continue
        new = _unwrap_each(expr, lambda e: ex.unwrap_expression(e, mode=mode))
        assert sympy.srepr(new) == sympy.srepr(old), (name, mode)


def test_the_jacobian_sqrt_guard_matches_replace(laws):
    """The expanded JIT route's Newton source: its guard, a memoised rebuild, against
    SymPy's own replace with the same rule (every fractional power, #841)."""
    from underworld3.cython.generic_solvers import _jacobian_unwrap

    eps2 = sympy.Float(1.0e-36)

    def old_guard(e):
        return e.replace(
            lambda n: (n.is_Pow and n.exp.is_integer is not True
                       and n.args[0].free_symbols),
            lambda n: sympy.Pow(n.args[0] + eps2, n.exp))

    for name, expr in laws:
        new = _jacobian_unwrap(expr, route="expanded")
        old = _unwrap_each(expr, lambda e: old_guard(
            _fixed_point_unwrap(e, "symbolic_keep_constants")))
        assert sympy.srepr(new) == sympy.srepr(old), name


def test_the_identity_walk_finds_the_same_symbols(laws):
    from underworld3.utilities._jitextension import _unique_symbols

    for name, expr in laws:
        for e in (expr, ex.unwrap_expression(expr, mode="nondimensional")
                  if not isinstance(expr, sympy.MatrixBase)
                  else expr.applyfunc(lambda a: ex.unwrap_expression(a, mode="nondimensional"))):
            assert _unique_symbols(e) == set(e.atoms(sympy.Symbol)), name


def test_the_shared_xreplace_matches_xreplace(laws):
    """The constants[] substitution in the JIT lowering visits each node object once;
    it must build exactly what xreplace builds, the generated C and the cache key
    both depend on it."""
    from underworld3.utilities._jitextension import _xreplace_shared, _unique_symbols

    swapped = 0
    for name, expr in laws:
        unwrapped = _unwrap_each(expr, lambda e: ex.unwrap_expression(
            e, mode="symbolic_keep_constants"))
        held = sorted((a for a in _unique_symbols(unwrapped)
                       if isinstance(a, ex.UWexpression)), key=str)
        if not held:
            continue                    # k(u) = 1 + u**2 has no parameter to swap
        rule = {a: sympy.Symbol(f"slot_{k}") for k, a in enumerate(held)}
        assert sympy.srepr(_xreplace_shared(unwrapped, rule)) == \
            sympy.srepr(unwrapped.xreplace(rule)), name
        swapped += 1
    assert swapped >= 7               # the VP (three laws, viscosity and flux) and VEP


def test_the_graph_guard_matches_replace(laws):
    """The guard every Newton node body gets (``_jit_graph.guard_fractional_powers``),
    a memoised rebuild, against SymPy's own ``replace`` with the same rule."""
    from underworld3.utilities._jit_graph import guard_fractional_powers

    eps2 = sympy.Float(1.0e-36)

    def old_guard(e):
        return e.replace(
            lambda n: (n.is_Pow and n.exp.is_integer is not True
                       and n.args[0].free_symbols),
            lambda n: sympy.Pow(n.args[0] + eps2, n.exp))

    for name, expr in laws:
        tree = _unwrap_each(expr, lambda e: _fixed_point_unwrap(e, "symbolic_keep_constants"))
        new = _unwrap_each(tree, guard_fractional_powers)
        old = _unwrap_each(tree, old_guard)
        assert sympy.srepr(new) == sympy.srepr(old), name


def test_each_atom_is_tested_for_constancy_once(monkeypatch):
    """The cost that #823 removed: the fixed-point passes asked whether each atom was
    a constant on every pass, and each answer was itself a full unwrap. On a chain of
    12 nested expressions with 25 distinct UW atoms the passes asked 103 times; the
    memoised unwrap asks once per distinct atom."""
    import underworld3.utilities._jitextension as jx

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    T = uw.discretisation.MeshVariable("T0022c", mesh, 1, degree=1)
    levels = 12
    e = uw.expression(r"e_{0022c,0}", T.sym[0], "base")
    for k in range(1, levels + 1):
        c = uw.expression(rf"c_{{0022c,{k}}}", 0.1 * k, "constant")
        e = uw.expression(rf"e_{{0022c,{k}}}", e + c * T.sym[0] + e * c, "level")
    calls = []
    is_truly_constant = jx._is_truly_constant

    def counted(atom, uw_type):
        calls.append(atom)
        return is_truly_constant(atom, uw_type)

    monkeypatch.setattr(jx, "_is_truly_constant", counted)
    ex.unwrap_expression(e, mode="symbolic_keep_constants")
    assert len(calls) == 2 * levels + 1, len(calls)


def test_a_cyclic_expression_is_refused():
    a = uw.expression(r"a_{cyc}", 1.0, "cyclic test")
    a.sym = a + 1
    with pytest.raises(ValueError, match="cyclic"):
        ex.unwrap_expression(a, mode="symbolic")
