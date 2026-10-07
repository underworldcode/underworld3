"""The memoised unwrap, the memoised sqrt guard and the identity-walk symbol scan (#823)
give exactly what the algorithms they replace gave.

The old ``unwrap_expression`` iterated ``subs`` passes to a fixed point, and in
``symbolic_keep_constants`` mode ran ``_is_truly_constant`` (itself a full unwrap) for every
atom on every pass. On the Spiegelman notch viscosity that took 563 s; the memoised
version takes 0.11 s. These tests hold the replacement to STRUCTURAL identity (``srepr``)
with a copy of the old algorithm, on the constitutive laws the solvers compile, in every
mode. Identical expressions mean identical generated C and identical JIT cache keys.
"""
import numpy as np
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
    from underworld3.cython.generic_solvers import _jacobian_unwrap

    eps2 = sympy.Float(1.0e-36)

    def old_guard(e):
        return e.replace(
            lambda n: (n.is_Pow and n.exp.is_Rational and n.exp.q == 2
                       and n.args[0].free_symbols),
            lambda n: sympy.Pow(n.args[0] + eps2, n.exp))

    for name, expr in laws:
        new = _jacobian_unwrap(expr)
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


def test_field_values_coordinates_and_constant_slots_are_declared_real():
    """SymPy treats an undefined function, a sympy.vector coordinate and a plain Symbol
    as possibly complex, so a power with a symbolic exponent made it work out real and
    imaginary parts before combining powers (#823). Field values, their gradients and
    coordinates are real and finite. A UW expression reads its realness from its
    content, so one built on them is real too."""
    from underworld3.utilities._jitextension import _JITConstant

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    v = uw.discretisation.MeshVariable("V0022r", mesh, mesh.dim, degree=2)
    x = mesh.N.x
    for atom in (v.sym[0], v.sym[0].diff(x), x, mesh.N.y, mesh.X[0]):
        assert atom.is_real is True and atom.is_finite is True, atom
        assert sympy.im(atom) == 0, atom
    # content with no number in it: before #823 neither a field value nor a
    # coordinate was known real, so this was None and im() stayed unevaluated
    e = uw.expression(r"e_{0022}", v.sym[0] ** 2 + v.sym[1].diff(x) + mesh.X[1],
                      "built on a field and a coordinate")
    assert e.is_extended_real is True
    assert sympy.im(e) == 0
    # `real` asked FIRST: with a class handler instead of a construction-time
    # assumption this read a plain Symbol's cached None (test-order dependent)
    for _ in range(20):
        assert _JITConstant(0, "c").is_real is True
    # and nothing leaks the other way: a plain SymPy symbol stays unknown
    plain = sympy.Symbol("plain_0022")
    assert plain.is_extended_real is None and plain.is_real is None


def test_derivatives_with_respect_to_a_field_keep_its_realness():
    """sympy.diff differentiates with respect to a field value or gradient through an
    assumption-free Dummy, so a real field is complex while it is differentiated: an
    Abs that realness made out of sqrt(g**2) leaves Derivative(u, u) unevaluated, and
    an explicit Abs(u) comes back in re() and im() (both unprintable). The solvers
    differentiate through _diff_wrt_field, whose stand-in is real."""
    from underworld3.function._function import (
        _diff_wrt_field, _derive_by_array_wrt_field)

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    u = uw.discretisation.MeshVariable("U0022d", mesh, 1, degree=1)
    c = sympy.Symbol("c_0022", real=True)
    s = sympy.Symbol("s_0022", positive=True)
    for w in (u.sym[0], u.sym[0].diff(mesh.N.x)):
        cases = {
            sympy.sqrt((c + w) ** 2): sympy.sign(c + w),
            sympy.Abs(w - c): sympy.sign(w - c),
            (w ** 2 + c ** 2) ** s: 2 * s * w * (w ** 2 + c ** 2) ** (s - 1),
        }
        for f, expected in cases.items():
            d = _diff_wrt_field(f, w)
            assert not d.has(sympy.Derivative, sympy.re, sympy.im), (f, d)
            assert sympy.simplify(d - expected) == 0, (f, d)
    # a coordinate goes to sympy.diff unchanged
    x = mesh.N.x
    assert _diff_wrt_field(x ** 2 * u.sym[0], x) == sympy.diff(x ** 2 * u.sym[0], x)
    # the array form has derive_by_array's layout: [i, j] = d f[j] / d dx[i]
    f = sympy.Array([u.sym[0] ** 2, sympy.Abs(u.sym[0] - c)])
    dx = sympy.Array([u.sym[0], c])
    got = _derive_by_array_wrt_field(f, dx)
    assert got.shape == (2, 2)
    assert got[0, 0] == 2 * u.sym[0] and got[0, 1] == sympy.sign(u.sym[0] - c)
    assert got[1, 0] == 0 and got[1, 1] == sympy.sign(c - u.sym[0])


def test_newton_poisson_with_an_abs_conductivity_matches_kirchhoff():
    """A Jacobian through Abs of the unknown: k(u) = 1 + |u - 0.2| on a unit square,
    u = 0.5 at the bottom and 1 at the top, no source. The flux depends on y alone, and
    for u > 0.2 the Kirchhoff transform phi(u) = 0.8 u + u**2 / 2 is linear in y, so
    u(y) = -0.8 + sqrt(0.64 + 2 (0.525 + 0.775 y)). Before #823 the Newton tangent
    (re and im of an applied function) could not be printed."""
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 16), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0))
    u = uw.discretisation.MeshVariable("U0022k", mesh, 1, degree=2)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = 1 + sympy.Abs(
        u.sym[0] - sympy.Rational(1, 5))
    pois.f = 0
    pois.add_dirichlet_bc(0.5, "Bottom")
    pois.add_dirichlet_bc(1.0, "Top")
    pois.consistent_jacobian = True
    pois.tolerance = 1.0e-10
    u.array[...] = 0.75
    pois.solve(zero_init_guess=False)
    assert pois.snes.getConvergedReason() > 0
    exact = -0.8 + np.sqrt(0.64 + 2 * (0.525 + 0.775 * u.coords[:, 1]))
    assert np.max(np.abs(u.array[:, 0, 0] - exact)) < 1.0e-4
    # Newton on a smooth (u > 0.2 everywhere) problem: a handful of iterations
    assert pois.snes.getIterationNumber() <= 6


def test_newton_drucker_prager_with_a_yield_floor_compiles():
    """The case that found it: at yield softness 0 the yield-stress floor is
    smooth_max(tau_y, tau_min, 0) = (a + b + sqrt((a - b)**2)) / 2, and the Newton
    tangent differentiates it with respect to the pressure field."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("V0022n", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0022n", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    params = stokes.constitutive_model.Parameters
    params.shear_viscosity_0 = 1.0
    params.yield_stress = (uw.expression(r"C_{0022n}", 0.5, "cohesion")
                           + uw.expression(r"\mu_{0022n}", 0.6, "friction") * p.sym[0])
    params.yield_stress_min = uw.expression(r"\tau_{0022n}", 0.01, "yield floor")
    stokes.constitutive_model.yield_softness = 0
    stokes.consistent_jacobian = True
    stokes._setup_pointwise_functions()
    for block in (stokes._up_G0, stokes._up_G1, stokes._up_G2, stokes._up_G3):
        assert not sympy.Matrix(block).has(sympy.Derivative), block


def test_a_dirac_delta_compiles_as_zero_and_says_so():
    """The coordinates are real, so sqrt((x - a)**2) is Abs(x - a) and its second
    derivative is 2*DiracDelta(x - a), which C cannot print. It compiles as 0, its value
    away from x = a: the solve is the solve without the delta, to the last bit."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    x, y = mesh.X
    kink = sympy.sqrt((x - sympy.Rational(3, 10)) ** 2)
    assert sympy.diff(kink, x, 2).has(sympy.DiracDelta)

    def solve(f, name):
        u = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
        pois = uw.systems.Poisson(mesh, u_Field=u)
        pois.constitutive_model = uw.constitutive_models.DiffusionModel
        pois.constitutive_model.Parameters.diffusivity = 1.0
        pois.f = f
        pois.add_dirichlet_bc(0.0, "Bottom")
        pois.solve()
        return np.array(u.array)

    with pytest.warns(UserWarning, match="DiracDelta"):
        with_delta = solve(1 + sympy.diff(kink, x, 2) * sympy.sin(sympy.pi * y), "U0022a")
    without = solve(sympy.Integer(1), "U0022b")
    assert np.array_equal(with_delta, without)
