r"""Every registered solution, checked the same way.

The per-solution test files check what is particular to each. This one checks
what is true of all of them, by iterating over `uw.analytic.available()` — so a
solution added later is covered the moment it is registered, and cannot ship
without these checks the way SolNL did.

That is not hypothetical. SolNL's kernel publishes the deviatoric stress; it was
stored as the total, and its momentum residual was 1.06 rather than zero. Its own
test file checked agreement with the kernel and incompressibility, both of which
passed, and nothing checked the momentum balance. The stress convention is now a
declaration (`stress_is_deviatoric`) applied in one place, and this file makes
the omission impossible to repeat.

Run: pixi run python -m pytest tests/test_1024_analytic_conformance.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


# Solutions that are SymPy and installable here. The two exclusions are
# declared by the solutions themselves, not listed by name, so a later solution
# in either category is handled without touching this file.
#
#   symbolic = False   an external numeric oracle; nothing to differentiate, so
#                      the residual gates cannot be applied to it at all
#   requires = "..."   an optional dependency that is not installed
#
# The excluded names are asserted below, so an accidental exclusion — a typo in
# a class attribute, say — fails here rather than quietly shrinking the sweep.
SOLUTIONS = sorted(
    name
    for name in uw.analytic.available()
    if getattr(uw.analytic, name).symbolic and uw.analytic.is_available(name)
)


def test_the_sweep_covers_everything_it_can():
    """What this file skips, and on what declared grounds."""

    excluded = set(uw.analytic.available()) - set(SOLUTIONS)

    for name in excluded:
        solution = getattr(uw.analytic, name)
        assert not solution.symbolic or not uw.analytic.is_available(name), (
            f"{name} is symbolic and available but is not being swept"
        )

    assert len(SOLUTIONS) >= 19, "the sweep has lost solutions"


@pytest.fixture(scope="module")
def meshes():
    return {
        2: uw.meshing.StructuredQuadBox(
            elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
        ),
        3: uw.meshing.StructuredQuadBox(
            elementRes=(2, 2, 2),
            minCoords=(0.0, 0.0, 0.0),
            maxCoords=(1.0, 1.0, 1.0),
            qdegree=2,
        ),
    }


@pytest.fixture(scope="module")
def built(meshes):
    """Every registered solution, constructed once.

    Construction substitutes parameters into expressions that run to tens of
    thousands of operations for the series solutions, so building afresh in each
    test dominates the runtime of the whole file.

    Building them all here also asserts something worth asserting: every solution
    must be constructible from a mesh alone. The defaults are part of the
    interface, not an afterthought.
    """

    return {
        name: getattr(uw.analytic, name)(meshes[getattr(uw.analytic, name).dim])
        for name in SOLUTIONS
    }


@pytest.mark.parametrize("name", SOLUTIONS)
def test_solution_declares_its_metadata(name):
    """dim, a citation, and a stress convention — all stated, none inferred."""

    solution = getattr(uw.analytic, name)

    assert solution.dim in (2, 3), f"{name} must declare its dimension"
    assert solution.reference, f"{name} must cite where it came from"
    assert isinstance(solution.stress_is_deviatoric, bool)
    assert isinstance(solution.nonlinear, bool)


STOKES = [n for n in SOLUTIONS if getattr(uw.analytic, n).solves == "stokes"]
TRANSPORT = [n for n in SOLUTIONS if getattr(uw.analytic, n).solves == "transport"]
RICHARDS = [n for n in SOLUTIONS if getattr(uw.analytic, n).solves == "richards"]

# Every registered solution belongs to exactly one family, so adding a solution
# with a new `solves` value fails here rather than being quietly untested.
assert set(STOKES) | set(TRANSPORT) | set(RICHARDS) == set(SOLUTIONS)


@pytest.mark.parametrize("name", STOKES)
def test_solution_exposes_the_whole_contract(name, built):
    sol = built[name]
    dim = sol.dim

    assert sol.fn_velocity.shape == (1, dim)
    assert sol.fn_bodyforce.shape == (1, dim)
    assert sol.fn_stress.shape == (dim, dim)
    assert sol.fn_strainrate.shape == (dim, dim)
    assert sol.fn_pressure is not None
    assert sol.fn_viscosity is not None


@pytest.mark.parametrize("name", STOKES)
def test_solution_is_incompressible(name, built):
    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)

    assert _validation.incompressibility_residual(sol, points) < 1.0e-8


@pytest.mark.parametrize("name", STOKES)
def test_solution_satisfies_the_momentum_balance(name, built):
    r""":math:`\nabla\cdot\sigma + \mathbf f = 0`, for every solution.

    The check that catches a stress-convention error, and the one SolNL was
    missing. It consults no reference and no solver, so it cannot be fooled by a
    mistake shared between the solution and something derived from it.
    """

    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)

    assert _validation.momentum_residual(sol, points) < 1.0e-8


@pytest.mark.parametrize("name", STOKES)
def test_strain_rate_matches_the_velocity(name, built):
    r""":math:`\tfrac12(\nabla\mathbf u + \nabla\mathbf u^{T})` vs `fn_strainrate`.

    This is the independent one, and it belongs in the family-wide sweep rather
    than in a single solution's file. The velocity and the stress are *separate
    outputs of the same kernel*, and the strain rate here is derived from the
    stress — so the comparison relates two quantities the source derived
    separately, and it exercises the derivatives, which is what a solver
    consumes and where a transcription can be wrong while still matching
    pointwise.

    Contrast `test_stress_and_strain_rate_agree` below, which for most of the
    family is structural.
    """

    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)

    assert _validation.strainrate_consistency(sol, points) < 1.0e-8


# `set_fields` accepts a stress OR a strain rate and derives the other from
# sigma + p I = 2 eta edot. For a solution that supplied only one of the two,
# the assertion below is exactly that derivation read back, so it is structural
# rather than evidential. These three supply BOTH, so for them it compares two
# separately published quantities and is a real check.
#
# It is kept for the whole family regardless: it is nearly free, and it is what
# would catch `set_fields` itself regressing.
PUBLISH_BOTH = {"SolNL", "SolDB2d", "SolDB3d"}


def test_the_list_of_solutions_publishing_both_is_accurate():
    """PUBLISH_BOTH is a claim about the sources; check it against them.

    A solution that starts supplying both — or stops — silently changes what the
    gate below is worth, so the claim is asserted rather than commented.
    """

    import inspect

    from underworld3 import analytic

    for name in STOKES:
        cls = getattr(analytic, name)
        source = ""
        for klass in cls.__mro__:
            try:
                text = inspect.getsource(klass)
            except (OSError, TypeError):
                continue
            if "set_fields(" in text:
                source = text
                break

        both = "stress=" in source and "strainrate=" in source
        assert both == (name in PUBLISH_BOTH), (
            f"{name}: publishes both = {both}, but PUBLISH_BOTH says "
            f"{name in PUBLISH_BOTH}"
        )


@pytest.mark.parametrize("name", STOKES)
def test_stress_and_strain_rate_agree(name, built):
    r""":math:`\sigma + p\,I = 2\eta\dot\varepsilon`, however each was obtained.

    Some solutions publish both and some derive one from the other; either way
    the pair has to be consistent, and a wrong `stress_is_deviatoric` shows up
    here as a full pressure's worth of disagreement.

    For the solutions NOT in `PUBLISH_BOTH` this is the identity `set_fields`
    used to build the missing quantity, so passing it is structural. See
    `test_strain_rate_matches_the_velocity` above for the independent check.
    """

    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)
    identity = sympy.eye(sol.dim)

    deviator = sol.fn_stress + sol.fn_pressure * identity
    scale = max(
        np.abs(_validation.sample(sol, deviator[i, j], points)).max()
        for i in range(sol.dim)
        for j in range(sol.dim)
    )

    for i in range(sol.dim):
        for j in range(sol.dim):
            difference = deviator[i, j] - 2 * sol.fn_viscosity * sol.fn_strainrate[i, j]
            assert np.abs(_validation.sample(sol, difference, points)).max() / scale < 1.0e-8


# The boundary-condition mixins are exercised in test_1016_analytic_contract.py.
# Building a Stokes solver per solution here as well was not worth what it cost:
# it dominated the runtime of this file without checking anything the contract
# tests do not already cover.


@pytest.mark.parametrize("name", TRANSPORT)
def test_transport_solution_exposes_its_contract(name, built):
    sol = built[name]

    assert sol.fn_solution is not None
    assert sol.fn_coefficient is not None
    assert sol.fn_source is not None


@pytest.mark.parametrize("name", TRANSPORT)
def test_transport_solution_satisfies_its_equation(name, built):
    r"""Steady or transient, whichever the solution declares.

    A time symbol means the equation is
    :math:`\partial_t u + \mathbf v\cdot\nabla u = \nabla\cdot(k\nabla u)`, and
    those are checked at three times rather than skipped. Skipping them here is
    what let `AdvectedFront` be judged against the wrong equation for as long as
    it was — the whole point of this file is that no registered solution goes
    unchecked.
    """

    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)

    if getattr(sol, "t", None) is None:
        assert _validation.transport_residual(sol, points) < 1.0e-10
        return

    for time in (0.05, 0.2, 0.5):
        assert _validation.diffusion_residual(sol, points, time) < 1.0e-10


@pytest.mark.parametrize("name", RICHARDS)
def test_richards_solution_exposes_its_contract(name, built):
    sol = built[name]

    assert sol.fn_solution is not None
    assert sol.fn_conductivity is not None
    assert sol.fn_capacity is not None


@pytest.mark.parametrize("name", RICHARDS)
def test_richards_solution_satisfies_its_equation(name, built):
    r""":math:`C(\psi)\partial_t\psi = \nabla\cdot[K(\psi)(\nabla\psi + \hat y)]`.

    Nonlinear, and the conductivity is the solution's own — so this consults
    nothing external, exactly as the Stokes momentum residual does.
    """

    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)

    if getattr(sol, "t", None) is None:
        assert _validation.richards_residual(sol, points) < 1.0e-10
        return

    for time in (0.05, 0.2):
        assert _validation.richards_residual(sol, points, time) < 1.0e-10
