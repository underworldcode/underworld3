"""The memoised unwrap (#823) gives exactly what the algorithm it replaces gave.

(The memoised sqrt guard, the identity-walk symbol scan and the shared xreplace were
tested here too; they went with the expanded-tree JIT route, and the graph route's
guard is held to the guarded tree by test_0024.)

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
