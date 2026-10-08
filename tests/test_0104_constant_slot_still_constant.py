"""A rampable constant in exponent position ramps; a constants[] slot that stops
being constant says so, not pack a zero.

The "rampable constant in exponent position does not ramp" report: with the atom at
zero, the ENCLOSING expression ``(1 + T**2)**(-m) + 1`` is the number 2. When the JIT
decided constancy by the atom's current value, that expression banked as one
constants[] slot, and ramping ``m`` left the kernel reading a scalar where the law
depends on position again. The JIT now decides constancy by structure (#823, tier 2):
the expression reads a field, so it is compiled as a quantity reading ``T`` and the
``m`` slot, and ``m`` ramps without a recompile.

A slot can still stop being constant: a constant atom whose content is replaced by one
that reads a field, without a rebuild. Packing a zero into it would hand the solve a
zero coefficient, silently; it must raise.
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


def _mean_after_fresh_build(value):
    uw, poisson, T, m = _build(value)
    poisson.solve(zero_init_guess=True)
    return float(np.asarray(T.data)[:, 0].mean())


def test_an_atom_in_exponent_position_ramps_without_a_rebuild():
    uw, poisson, T, m = _build(0.0)
    poisson.solve(zero_init_guess=True)
    # m has a slot of its own: the collapsing expression is not banked as one
    assert m in [expr for _, expr in poisson.constants_manifest]
    compiled = poisson._current_jit_cache_key

    ramped = {}
    for value in (0.5, 1.0):
        m.sym = sympy.sympify(value)
        poisson.solve(zero_init_guess=True)
        assert poisson._current_jit_cache_key == compiled, "ramping m recompiled"
        ramped[value] = float(np.asarray(T.data)[:, 0].mean())
    assert ramped[0.5] != pytest.approx(ramped[1.0])
    for value, mean in ramped.items():
        assert mean == pytest.approx(_mean_after_fresh_build(value), rel=1.0e-10), value


def _slot_that_stops_being_constant():
    uw, poisson, T, m = _build(0.0)
    c = uw.expression(r"c_slot", 1.0, "a constant coefficient")
    poisson.constitutive_model.Parameters.diffusivity = c
    poisson.solve(zero_init_guess=True)
    # one slot: the diffusivity parameter, whose content is c
    assert len(poisson.constants_manifest) == 1
    c.sym = 1.0 + T.sym[0] ** 2       # now reads the field; the kernel reads a slot
    return poisson


def test_a_slot_that_stops_being_constant_raises():
    poisson = _slot_that_stops_being_constant()
    with pytest.raises(RuntimeError, match="no longer.*reduces to a number"):
        poisson.solve(zero_init_guess=True)


def test_the_message_names_the_slot_and_says_how_to_recover():
    poisson = _slot_that_stops_being_constant()
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
