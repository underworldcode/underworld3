"""Field values, coordinates and parameters are declared real, and every derivative
with respect to a field keeps that realness (#823).

SymPy treats an undefined function, a sympy.vector coordinate and a plain Symbol as
possibly complex, so it worked out real and imaginary parts all through the
constitutive laws. Declaring them real changes what SymPy builds (sqrt(g**2) is
Abs(g)), and sympy.diff differentiates with respect to a field through a stand-in
with no assumptions. These tests hold both halves: the declarations, and the
solvers' differentiation through uw.function.diff_wrt_field.
"""
import pathlib
from fractions import Fraction

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


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
    differentiate through uw.function.diff_wrt_field, whose stand-in is real."""
    from underworld3.function import diff_wrt_field, derive_by_array_wrt_field

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
            d = diff_wrt_field(f, w)
            assert not d.has(sympy.Derivative, sympy.re, sympy.im), (f, d)
            assert sympy.simplify(d - expected) == 0, (f, d)
    # a coordinate goes to sympy.diff unchanged, and so do an unevaluated Derivative
    # (the swap would hide it from its variable) and a plain number
    x = mesh.N.x
    assert diff_wrt_field(x ** 2 * u.sym[0], x) == sympy.diff(x ** 2 * u.sym[0], x)
    held = sympy.Derivative(u.sym[0] ** 2, x)
    assert diff_wrt_field(held, u.sym[0]) == sympy.diff(held, u.sym[0]) != 0
    assert diff_wrt_field(3, u.sym[0]) == 0
    # the array form has derive_by_array's layout: [i, j] = d f[j] / d dx[i]
    f = sympy.Array([u.sym[0] ** 2, sympy.Abs(u.sym[0] - c)])
    dx = sympy.Array([u.sym[0], c])
    got = derive_by_array_wrt_field(f, dx)
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
    assert np.max(np.abs(u.array[:, 0, 0] - exact)) < 1.0e-7
    # Newton on a smooth (u > 0.2 everywhere) problem: a handful of iterations
    assert pois.snes.getIterationNumber() <= 6


def test_newton_drucker_prager_with_a_yield_floor_compiles_and_solves():
    """The case that found it: at yield softness 0 the yield-stress floor is
    smooth_max(tau_y, tau_min, 0) = (a + b + sqrt((a - b)**2)) / 2 = Max, and the
    Newton tangent differentiates it with respect to the pressure field. The yield
    stress is far above the stress, so the solve must reproduce the viscous one."""

    def shear_box(name, constitutive_model):
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        v = uw.discretisation.MeshVariable("V" + name, mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("P" + name, mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = constitutive_model
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((1.0, 0.0), "Top")
        stokes.bodyforce = sympy.Matrix([0, -mesh.X[0]])
        stokes.tolerance = 1.0e-10
        return stokes, v, p

    uw.reset_default_model()
    stokes, v, p = shear_box("0023n", uw.constitutive_models.ViscoPlasticFlowModel)
    params = stokes.constitutive_model.Parameters
    params.shear_viscosity_0 = 1.0
    params.yield_stress = (uw.expression(r"C_{0023n}", 100.0, "cohesion")
                           + uw.expression(r"\mu_{0023n}", 0.6, "friction") * p.sym[0])
    params.yield_stress_min = uw.expression(r"\tau_{0023n}", 0.01, "yield floor")
    stokes.constitutive_model.yield_softness = 0
    stokes.consistent_jacobian = True
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0

    viscous, v0, _ = shear_box("0023v", uw.constitutive_models.ViscousFlowModel)
    viscous.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    viscous.solve()
    v, v0 = np.asarray(v.array), np.asarray(v0.array)
    assert np.max(np.abs(v - v0)) < 1.0e-8 * np.max(np.abs(v0))


def test_no_solver_differentiates_with_plain_sympy_diff():
    """Every Jacobian in the solvers differentiates with respect to fields, so every
    one must go through diff_wrt_field. A site written with sympy.diff would pass
    every other test here until a law put an Abs or a sign of a field in its flux."""
    source = (pathlib.Path(__file__).parents[1] / "src" / "underworld3" / "cython"
              / "petsc_generic_snes_solvers.pyx").read_text()
    for plain in ("sympy.diff(", "sympy.derive_by_array(", ".diff(self.", ".diff(U", ".diff(L"):
        assert plain not in source, plain
    assert source.count("diff_wrt_field(") >= 38


def test_newton_through_a_step_of_the_unknown_compiles_without_the_jump():
    """Differentiating a Heaviside of the unknown gives a DiracDelta, which a
    pointwise kernel cannot carry: the tangent leaves the jump out and says so.
    Before #823 the tangent could not be printed at all."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    u = uw.discretisation.MeshVariable("U0023h", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=u)
    pois.constitutive_model = uw.constitutive_models.DiffusionModel
    pois.constitutive_model.Parameters.diffusivity = 1 + sympy.Heaviside(u.sym[0] - 2)
    pois.f = 1.0
    pois.add_dirichlet_bc(0.0, "Bottom")
    pois.consistent_jacobian = True
    with pytest.warns(UserWarning, match="DiracDelta"):
        pois.solve()
    # u stays below 2, where k = 1: the Poisson solve with k = 1 exactly
    assert pois.snes.getConvergedReason() > 0
    reference = uw.discretisation.MeshVariable("U0023r", mesh, 1, degree=1)
    plain = uw.systems.Poisson(mesh, u_Field=reference)
    plain.constitutive_model = uw.constitutive_models.DiffusionModel
    plain.constitutive_model.Parameters.diffusivity = 1.0
    plain.f = 1.0
    plain.add_dirichlet_bc(0.0, "Bottom")
    plain.solve()
    assert np.max(np.abs(np.asarray(u.array) - np.asarray(reference.array))) < 1.0e-10


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
        with_delta = solve(1 + sympy.diff(kink, x, 2) * sympy.sin(sympy.pi * y), "U0023a")
    without = solve(sympy.Integer(1), "U0023b")
    assert np.array_equal(with_delta, without)

    # evaluate follows the same rule, with and without a field in the expression
    points = np.array([[0.1, 0.5], [0.3, 0.5], [0.8, 0.2]])
    T = uw.discretisation.MeshVariable("T0023e", mesh, 1, degree=1)
    T.array[...] = 2.0
    for f, expected in ((1 + sympy.diff(kink, x, 2), 1.0),
                        (T.sym[0] * (1 + sympy.diff(kink, x, 2)), 2.0)):
        with pytest.warns(UserWarning, match="DiracDelta"):
            values = uw.function.evaluate(f, points)
        assert np.asarray(values).shape == (len(points), 1, 1)
        assert np.allclose(np.asarray(values).ravel(), expected, rtol=1.0e-12)


def test_parameters_given_with_units_are_known_real():
    """A UW expression reports realness from its content, and a parameter given with
    units holds a UWQuantity, which had no realness to report. Every power over a sum
    containing one was then evaluated in the complex plane: 36 s of the 43 s Newton
    source on the Spiegelman notch (#823). A UWQuantity is real when its value is."""
    from underworld3.cython.generic_solvers import _jacobian_unwrap
    from underworld3.function.expressions import UWexpression
    from underworld3.utilities._jitextension import _unique_symbols

    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        domain_depth=uw.quantity(100, "km"),
        material_viscosity=uw.quantity(1e21, "Pa*s"),
        lithostatic_pressure=uw.quantity(1e8, "Pa"),
    )
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    v = uw.discretisation.MeshVariable("V0022u", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0022u", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    params = stokes.constitutive_model.Parameters
    params.shear_viscosity_0 = uw.quantity(1e24, "Pa*s")
    params.shear_viscosity_min = uw.quantity(1e20, "Pa*s")
    params.yield_stress = uw.quantity(1e8, "Pa")
    params.yield_stress_min = uw.quantity(0, "Pa")

    for value in (uw.quantity(1e24, "Pa*s"), uw.quantity(0, "Pa"),
                  uw.quantity(Fraction(1, 2), "Pa")):
        assert value.is_extended_real is True and value.is_finite is True
    infinite = uw.quantity(float("inf"), "Pa")
    assert infinite.is_extended_real is True and infinite.is_finite is False
    # unknown, as SymPy says for nan
    nan = uw.quantity(float("nan"), "Pa")
    assert nan.is_extended_real is None and nan.is_finite is None

    flux = _jacobian_unwrap(stokes.constitutive_model.flux)
    held = [a for a in _unique_symbols(flux) if isinstance(a, UWexpression)]
    assert held, "the Newton flux keeps its constant parameters as atoms"
    unknown = [a for a in held if a.is_extended_real is not True]
    assert not unknown, unknown


def test_the_power_mean_sharpness_follows_the_softness():
    """The power mean's sharpness s = 1/(delta + 0.001) is its own constant atom (an
    inline -1/(delta + 0.001) exponent made SymPy evaluate im() of the whole base on
    every rebuild: 40 s of the notch Newton source, #823). It must follow delta: a
    solve after changing delta equals a fresh model built at that delta."""

    def yielding_box(name, delta):
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        v = uw.discretisation.MeshVariable("V" + name, mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("P" + name, mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
        cm = stokes.constitutive_model
        cm.Parameters.shear_viscosity_0 = 1.0
        cm.Parameters.yield_stress = 0.8
        cm.yield_mode = "softmin"
        cm.yield_smoother = "powermean"
        cm.yield_softness = delta
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((1.0, 0.0), "Top")
        stokes.tolerance = 1.0e-10
        stokes.consistent_jacobian = True
        return stokes, v

    uw.reset_default_model()
    ramped, v_ramped = yielding_box("0023s", 1.0)
    from underworld3.cython.generic_solvers import _jacobian_unwrap
    from underworld3.utilities._jitextension import _unique_symbols

    sharpness = ramped.constitutive_model._get_yield_sharpness()
    assert sharpness in _unique_symbols(_jacobian_unwrap(ramped.constitutive_model.flux))
    ramped.solve()
    assert ramped.snes.getConvergedReason() > 0
    at_one = np.array(v_ramped.array)
    ramped.constitutive_model.yield_softness = 0.25
    ramped.solve(zero_init_guess=False)
    fresh, v_fresh = yielding_box("0023f", 0.25)
    fresh.solve()
    assert ramped.snes.getConvergedReason() > 0 and fresh.snes.getConvergedReason() > 0
    assert np.max(np.abs(np.asarray(v_ramped.array) - np.asarray(v_fresh.array))) < 1.0e-7
    # and the softness matters here, or the comparison could not fail
    assert np.max(np.abs(at_one - np.asarray(v_fresh.array))) > 1.0e-3
