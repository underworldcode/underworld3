r"""The Dohrmann–Bochev manufactured solutions, in 2D and 3D.

Polynomial velocities with a body force chosen to make them exact. SolDB2d is
isoviscous; SolDB3d (Burstedde et al.) carries a smooth viscosity peaked in the
interior and is **the suite's only 3D solution** — several parts of a Stokes
discretisation genuinely differ between two and three dimensions (the pressure
space, the null space, the tensor assembly), and a 2D benchmark cannot see a term
that is wrong only in the third.

These are the easiest solutions here to be sure of. The fields are short enough
that the residuals reduce *symbolically* to zero rather than to something small,
so the assertions are exact rather than tolerance-based — no sampling, no
conditioning question, nothing to argue about.

Run: pixi run python -m pytest tests/test_1022_analytic_soldb.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


@pytest.fixture(scope="module")
def box2d():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


@pytest.fixture(scope="module")
def box3d():
    return uw.meshing.StructuredQuadBox(
        elementRes=(2, 2, 2),
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        qdegree=2,
    )


def _plain(sol, expression):
    """Rewrite an expression over plain symbols before simplifying.

    Mesh coordinates are not ordinary Symbols, and `simplify` will not combine
    exponentials written in them — the beta = 0 cases reduce and the others do
    not, purely because of the symbol type. Swapping first makes the reduction
    the algebraic question it is meant to be.
    """

    plain = sympy.symbols(f"_p0:{sol.dim}", real=True)
    return sympy.sympify(expression).subs(dict(zip(sol.mesh.X, plain)))


def _divergence(sol):
    return sum(
        sympy.diff(sol.fn_velocity[0, i], sol.mesh.X[i]) for i in range(sol.dim)
    )


def _momentum(sol):
    r""":math:`\nabla\cdot\sigma + \mathbf f`, component by component."""

    return [
        sol.fn_bodyforce[0, i]
        + sum(sympy.diff(sol.fn_stress[i, j], sol.mesh.X[j]) for j in range(sol.dim))
        for i in range(sol.dim)
    ]


def test_soldb2d_is_incompressible(box2d):
    sol = uw.analytic.SolDB2d(box2d)
    assert sympy.simplify(_plain(sol, _divergence(sol))) == 0


def test_soldb2d_satisfies_the_momentum_balance(box2d):
    """Exactly, not approximately — the expressions are small enough to reduce."""

    sol = uw.analytic.SolDB2d(box2d)
    for residual in _momentum(sol):
        assert sympy.simplify(_plain(sol, residual)) == 0


def test_soldb2d_is_isoviscous(box2d):
    assert uw.analytic.SolDB2d(box2d).fn_viscosity == 1


@pytest.mark.parametrize("beta", [0.0, 4.0, 10.0])
def test_soldb3d_is_incompressible(box3d, beta):
    sol = uw.analytic.SolDB3d(box3d, beta=beta)
    assert sympy.simplify(_plain(sol, _divergence(sol))) == 0


@pytest.mark.parametrize("beta", [0.0, 4.0, 10.0])
def test_soldb3d_satisfies_the_momentum_balance(box3d, beta):
    """Holds for every viscosity exponent, including the isoviscous beta = 0."""

    sol = uw.analytic.SolDB3d(box3d, beta=beta)
    for residual in _momentum(sol):
        assert sympy.simplify(_plain(sol, residual)) == 0


def test_soldb3d_viscosity_is_the_published_form(box3d):
    r""":math:`\eta = e^{1-\beta[x(1-x)+y(1-y)+z(1-z)]}`, peaked in the interior."""

    beta = 4.0
    sol = uw.analytic.SolDB3d(box3d, beta=beta)
    x, y, z = box3d.X

    # Rational, matching what the solution substitutes: an exact 4 and a float
    # 4.0 in an exponent are equal but SymPy will not cancel them.
    expected = sympy.exp(
        1 - sympy.Rational(beta) * (x * (1 - x) + y * (1 - y) + z * (1 - z))
    )
    assert sympy.simplify(_plain(sol, sol.fn_viscosity - expected)) == 0

    # Smallest where the bracket is largest, i.e. at the centre of the cube.
    eta = sympy.lambdify(tuple(box3d.X), sol.fn_viscosity, "numpy")
    assert float(eta(0.5, 0.5, 0.5)) < float(eta(0.1, 0.1, 0.1))


def test_soldb3d_stress_is_consistent_with_the_strain_rate(box3d):
    r""":math:`\sigma = -p\,I + 2\eta\dot\varepsilon`.

    The kernel publishes the deviatoric stress and the strain rate separately, so
    this checks two of its outputs against each other and pins the convention —
    reading the deviator as if it were the total would leave the momentum
    residual wrong by exactly :math:`\nabla p`.
    """

    sol = uw.analytic.SolDB3d(box3d, beta=4.0)

    deviator = sol.fn_stress + sol.fn_pressure * sympy.eye(3)
    for i in range(3):
        for j in range(3):
            assert (
                sympy.simplify(
                    _plain(sol, deviator[i, j] - 2 * sol.fn_viscosity * sol.fn_strainrate[i, j])
                )
                == 0
            )


def test_soldb_are_registered(box2d, box3d):
    assert {"SolDB2d", "SolDB3d"} <= set(uw.analytic.available())
    assert uw.analytic.SolDB2d(box2d).dim == 2
    assert uw.analytic.SolDB3d(box3d).dim == 3


def test_soldb3d_refuses_a_2d_mesh(box2d):
    """The dimension check in the contract does its job."""

    with pytest.raises(ValueError, match="3D solution"):
        uw.analytic.SolDB3d(box2d)
