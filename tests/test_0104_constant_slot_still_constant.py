"""A constants[] slot that stops being constant must say so, not pack a zero.

An expression is given a ``constants[]`` slot because it resolved to a single
number when the kernel was compiled. Ramping an atom nested inside it can make
it depend on position again — the compiled kernel still reads a scalar, and
packing a zero into that slot hands the solve a zero coefficient. Silent, and
catastrophic: a zero diffusivity diverges and nothing says why.

This is the failure behind the "rampable constant in exponent position does not
ramp" report. The atom is not compiled out; the ENCLOSING expression collapses
to a number while the atom is zero, banks as one constant, and then stops being
one.
"""

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _build(initial):
    import underworld3 as uw

    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    T = uw.discretisation.MeshVariable("T_slot", mesh, 1, degree=2)
    m = uw.expression(r"m_slot", initial, "rampable atom")
    # constant while m == 0 (anything**0 is 1), field-dependent as soon as it isn't
    kappa = uw.expression(
        r"\kappa_slot", (1.0 + 0.5 * T.sym[0] ** 2) ** (-m) + 1.0, "collapsing"
    )
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = kappa
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.petsc_options.delValue("ksp_monitor")
    return uw, poisson, T, m


def test_a_slot_that_stops_being_constant_raises():
    uw, poisson, T, m = _build(0.0)
    poisson.solve(zero_init_guess=True)

    # Compiled while the whole expression was the number 2, so the diffusivity
    # banks as a single scalar slot. (It is named for the parameter wrapper,
    # not for the inner expression — the collector stops at the outermost thing
    # that is truly constant and does not recurse past it.)
    assert len(poisson.constants_manifest) == 1

    m.sym = sympy.sympify(0.5)  # now depends on T again
    with pytest.raises(RuntimeError, match="no longer.*reduces to a number"):
        poisson.solve(zero_init_guess=True)


def test_the_message_names_the_slot_and_says_how_to_recover():
    uw, poisson, T, m = _build(0.0)
    poisson.solve(zero_init_guess=True)
    m.sym = sympy.sympify(0.5)
    with pytest.raises(RuntimeError) as excinfo:
        poisson.solve(zero_init_guess=True)
    message = str(excinfo.value)
    assert "no longer" in message
    assert "_needs_function_rewire" in message
    assert "constants[] slot" in message


def test_forcing_a_rebuild_recovers_and_the_atom_then_ramps():
    """The recovery the message prescribes must actually work."""
    uw, poisson, T, m = _build(0.0)
    poisson.solve(zero_init_guess=True)

    results = []
    for value in (0.5, 1.0):
        m.sym = sympy.sympify(value)
        poisson.is_setup = False
        poisson._needs_function_rewire = True
        poisson.constitutive_model._solver_is_setup = False
        poisson.solve(zero_init_guess=True)
        results.append(float(np.asarray(T.data)[:, 0].mean()))

    assert all(np.isfinite(results))
    assert results[0] != pytest.approx(results[1]), "the atom still does not ramp"


def test_an_expression_that_stays_constant_is_unaffected():
    """The negative control: a slot that remains a number keeps working, so the
    guard is not just refusing every ramp."""
    uw, poisson, T, m = _build(0.0)
    # a plainly constant coefficient, ramped in the ordinary way
    c = uw.expression(r"c_plain", 1.0, "ordinary rampable constant")
    poisson.constitutive_model.Parameters.diffusivity = c
    poisson.solve(zero_init_guess=True)
    first = float(np.asarray(T.data)[:, 0].mean())
    c.sym = sympy.sympify(2.0)
    poisson.solve(zero_init_guess=True)
    second = float(np.asarray(T.data)[:, 0].mean())
    assert second == pytest.approx(first / 2.0, rel=1e-6)
