r"""SolKx — Stokes flow with an exponentially varying viscosity.

The companion to SolCx: same box, same forcing shape, but the viscosity varies
*smoothly* as :math:`e^{2Bx}` instead of jumping. The two fail differently. A
jump tests how a discretisation copes with a discontinuity inside an element; a
gradient tests whether the operator stays well conditioned while the contrast
builds across every element in the domain.

Validated by the equations rather than against a compiled kernel. The forcing and
the boundary conditions are both known, so a field set that satisfies Stokes with
them is *the* solution by uniqueness — the momentum and incompressibility
residuals settle it without an oracle, and free slip is checked directly.

Run: pixi run python -m pytest tests/test_1021_analytic_solkx.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


# B = 2.3026 is a decade of viscosity contrast per unit length, so e^2B ~ 100
# across the box; B = 5 is four orders. Both wavenumbers, integer and not.
# The canonical case only, on every PR. SolKx declares
# `expensive_to_validate`: an exponential viscosity makes every residual
# here a symbolic differentiation of a very large expression, and the four
# cases together were 88s of the analytic suite's 1010s.
#
# The remaining cases below are not dropped — they run in
# tests/analytic_full/, which sweeps this solution over its parameter table
# with the same residual gates. Every CHECK in this file still runs on every
# PR; what is reduced is how many parameter values it runs on.
CASES = [
    (2.302585092994046, 3, 2),
]



@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


# Interior points, and points on each wall for the free-slip check.
INTERIOR = np.array([(0.2, 0.3), (0.7, 0.8), (0.5, 0.5), (0.9, 0.15), (0.05, 0.95)])
LEFT = np.array([(0.0, t) for t in (0.13, 0.47, 0.82)])
RIGHT = np.array([(1.0, t) for t in (0.13, 0.47, 0.82)])
BOTTOM = np.array([(t, 0.0) for t in (0.13, 0.47, 0.82)])
TOP = np.array([(t, 1.0) for t in (0.13, 0.47, 0.82)])


@pytest.fixture(scope="module")
def cache():
    """One construction per (B, n, m), shared across the gates below.

    Constructing SolKx substitutes parameters into an expression tens of
    thousands of operations long. Two gates run over every case, so building
    afresh in each doubled the file's cost for nothing.
    """

    return {}


def _sol(cache, mesh, B, n, m):
    key = (B, n, m)
    if key not in cache:
        cache[key] = uw.analytic.SolKx(mesh, B=B, n=n, m=m)
    return cache[key]


def _at(sol, expression, points):
    """Values of an expression of the mesh coordinates, over a whole point set.

    Goes through the validation harness, which swaps the mesh coordinates for
    plain symbols first — lambdify cannot bind mesh coordinates as arguments.

    Evaluated for all points in one call on purpose: these expressions run to
    tens of thousands of operations, and lambdifying one per point turns a
    two-minute suite into an unrunnable one.
    """

    from underworld3.analytic import _validation

    return np.abs(_validation.sample(sol, expression, points))


@pytest.mark.parametrize("B,n,m", CASES)
def test_solkx_satisfies_the_stokes_equations(mesh, B, n, m):
    r"""":math:`\nabla\cdot\sigma + \mathbf f = 0` and :math:`\nabla\cdot\mathbf v = 0`.

    The residual is normalised by the forcing, so the tolerance means what it
    says regardless of how strong the driving is.
    """

    sol = uw.analytic.SolKx(mesh, B=B, n=n, m=m)
    x, z = mesh.X

    scale = _at(sol, sol.fn_bodyforce[0, 1], INTERIOR).max()

    residual_x = _at(
        sol, sympy.diff(sol.fn_stress[0, 0], x) + sympy.diff(sol.fn_stress[0, 1], z), INTERIOR
    )
    residual_z = _at(
        sol,
        sympy.diff(sol.fn_stress[1, 0], x)
        + sympy.diff(sol.fn_stress[1, 1], z)
        + sol.fn_bodyforce[0, 1],
        INTERIOR,
    )
    divergence = _at(
        sol,
        sympy.diff(sol.fn_velocity[0, 0], x) + sympy.diff(sol.fn_velocity[0, 1], z),
        INTERIOR,
    )

    assert residual_x.max() / scale < 1.0e-10
    assert residual_z.max() / scale < 1.0e-10
    assert divergence.max() < 1.0e-10


@pytest.mark.parametrize("B,n,m", CASES)
def test_solkx_is_free_slip_on_every_wall(mesh, B, n, m):
    """The solution is posed with free slip, so the normal velocity must vanish.

    Together with the Stokes residual this pins the solution uniquely, which is
    what makes an oracle unnecessary here.
    """

    sol = uw.analytic.SolKx(mesh, B=B, n=n, m=m)
    vx, vz = sol.fn_velocity[0, 0], sol.fn_velocity[0, 1]

    assert _at(sol, vx, LEFT).max() < 1.0e-10
    assert _at(sol, vx, RIGHT).max() < 1.0e-10
    assert _at(sol, vz, BOTTOM).max() < 1.0e-10
    assert _at(sol, vz, TOP).max() < 1.0e-10


def test_solkx_viscosity_is_the_exponential_it_claims(mesh):
    r""":math:`\eta = e^{2Bx}`, and the stress is consistent with it."""

    B = 2.0
    sol = uw.analytic.SolKx(mesh, B=B, n=2, m=1)
    x, z = mesh.X

    where = np.array([(px, 0.5) for px in (0.0, 0.3, 1.0)])
    assert np.allclose(_at(sol, sol.fn_viscosity, where), np.exp(2 * B * where[:, 0]))

    # sigma = -p I + 2 eta edot, so the deviator recovered from the stress and
    # the strain rate computed from the velocity must agree.
    exz = (
        sympy.diff(sol.fn_velocity[0, 0], z) + sympy.diff(sol.fn_velocity[0, 1], x)
    ) / 2
    points = np.array([(0.25, 0.4), (0.6, 0.7), (0.85, 0.2)])
    difference = _at(sol, sol.fn_stress[0, 1] - 2 * sol.fn_viscosity * exz, points)
    scale = _at(sol, sol.fn_stress[0, 1], points).max()

    assert difference.max() / scale < 1.0e-10


def test_solkx_rejects_a_fractional_vertical_wavenumber(mesh):
    """A fractional m breaks free slip on the top wall while still solving Stokes.

    That is the dangerous kind of wrong — the residual checks would all pass and
    the benchmark would quietly be a different problem — so it is refused.
    """

    with pytest.raises(ValueError, match="sin\\(m\\*pi\\) = 0"):
        uw.analytic.SolKx(mesh, m=1.5)


def test_solkx_is_registered(mesh):
    assert "SolKx" in uw.analytic.available()
    assert uw.analytic.SolKx(mesh).nonlinear is False
