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


SOLUTIONS = sorted(uw.analytic.available())


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


@pytest.mark.parametrize("name", SOLUTIONS)
def test_solution_exposes_the_whole_contract(name, built):
    sol = built[name]
    dim = sol.dim

    assert sol.fn_velocity.shape == (1, dim)
    assert sol.fn_bodyforce.shape == (1, dim)
    assert sol.fn_stress.shape == (dim, dim)
    assert sol.fn_strainrate.shape == (dim, dim)
    assert sol.fn_pressure is not None
    assert sol.fn_viscosity is not None


@pytest.mark.parametrize("name", SOLUTIONS)
def test_solution_is_incompressible(name, built):
    from underworld3.analytic import _validation

    sol = built[name]
    points = sol.sample_points(count=8)

    assert _validation.incompressibility_residual(sol, points) < 1.0e-8


@pytest.mark.parametrize("name", SOLUTIONS)
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


@pytest.mark.parametrize("name", SOLUTIONS)
def test_stress_and_strain_rate_agree(name, built):
    r""":math:`\sigma + p\,I = 2\eta\dot\varepsilon`, however each was obtained.

    Some solutions publish both and some derive one from the other; either way
    the pair has to be consistent, and a wrong `stress_is_deviatoric` shows up
    here as a full pressure's worth of disagreement.
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
