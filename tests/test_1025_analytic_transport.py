r"""The scalar transport solutions.

Diffusion, advection–diffusion, Darcy and Poisson — one unknown field rather than
the velocity-and-pressure pair the Velic family solves for. These were already in
the repository, inline in the tests that used them; collected into
`uw.analytic` they gain the same oracle-free residual the Stokes solutions have.

The steady ones reduce to *exactly* zero, so those assertions are exact. The
transient ones carry a time symbol and are checked at several times against
:math:`\partial_t u + \mathbf v\cdot\nabla u = \nabla\cdot(k\nabla u)`.

Run: pixi run python -m pytest tests/test_1025_analytic_transport.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


@pytest.mark.parametrize("source", ["none", "constant", "sinusoid"])
def test_poisson_satisfies_its_equation(mesh, source):
    from underworld3.analytic import _validation

    sol = uw.analytic.Poisson1D(mesh, source=source)
    assert _validation.transport_residual(sol, sol.sample_points(count=8)) == 0.0


def test_poisson_profiles_are_what_they_claim(mesh):
    """Short enough to assert outright, which is the cheapest possible check."""

    x, z = mesh.X

    assert sympy.simplify(uw.analytic.Poisson1D(mesh, "none").fn_solution - (1 - z)) == 0
    assert (
        sympy.simplify(uw.analytic.Poisson1D(mesh, "constant").fn_solution - z * (1 - z))
        == 0
    )
    assert (
        sympy.simplify(
            uw.analytic.Poisson1D(mesh, "sinusoid").fn_solution - sympy.sin(sympy.pi * z)
        )
        == 0
    )


def test_poisson_rejects_an_unknown_source(mesh):
    with pytest.raises(ValueError, match="source must be one of"):
        uw.analytic.Poisson1D(mesh, source="quartic")


def test_two_layer_darcy_satisfies_its_equation(mesh):
    r""":math:`\nabla\cdot(k\nabla p) = 0` across a permeability jump."""

    from underworld3.analytic import _validation

    sol = uw.analytic.TwoLayerDarcy(mesh, k_lower=1.0, k_upper=0.1)
    assert _validation.transport_residual(sol, sol.sample_points(count=8)) == 0.0


def test_two_layer_darcy_conserves_flux_across_the_interface(mesh):
    r""":math:`k\,\partial p/\partial z` is continuous even though the gradient is not.

    This is the physical content of the solution and the thing a Darcy solver
    has to get right, so it is asserted directly rather than inferred from the
    residual.
    """

    from underworld3.analytic import _validation

    k1, k2, interface = 1.0, 0.1, 0.5
    sol = uw.analytic.TwoLayerDarcy(
        mesh, k_lower=k1, k_upper=k2, interface=interface
    )
    x, z = mesh.X

    gradient = sympy.diff(sol.fn_solution, z)
    below = np.array([[0.4, interface - 0.05]])
    above = np.array([[0.4, interface + 0.05]])

    flux_below = k1 * _validation.sample(sol, gradient, below)[0]
    flux_above = k2 * _validation.sample(sol, gradient, above)[0]

    assert np.isclose(flux_below, flux_above, rtol=1.0e-10)
    # ... and the gradient itself is genuinely discontinuous, so the test above
    # is not passing for the trivial reason that nothing changes at all.
    assert not np.isclose(
        _validation.sample(sol, gradient, below)[0],
        _validation.sample(sol, gradient, above)[0],
    )


@pytest.mark.parametrize("time", [0.05, 0.2, 0.5])
def test_erfc_diffusion_satisfies_the_heat_equation(mesh, time):
    from underworld3.analytic import _validation

    sol = uw.analytic.ErfcDiffusion(mesh, diffusivity=0.5)
    assert _validation.diffusion_residual(sol, sol.sample_points(count=8), time) < 1e-10


@pytest.mark.parametrize("time", [0.05, 0.2, 0.5])
def test_advected_front_satisfies_advection_diffusion(mesh, time):
    r""":math:`\partial_t c + u\,\partial_x c = \kappa\,\partial_{xx} c`.

    The advection term matters: without it this same solution reports a residual
    of order one, which looks like a broken solution rather than a check applied
    to the wrong equation.
    """

    from underworld3.analytic import _validation

    sol = uw.analytic.AdvectedFront(mesh, kappa=1.0e-2, speed=0.5)
    assert _validation.diffusion_residual(sol, sol.sample_points(count=8), time) < 1e-10


def test_advected_front_travels(mesh):
    """The pulse is where advection says it should be, and it spreads."""

    from underworld3.analytic import _validation

    speed, x0, x1 = 0.5, 0.1, 0.3
    sol = uw.analytic.AdvectedFront(mesh, kappa=1.0e-4, speed=speed, x0=x0, x1=x1)

    centre = (x0 + x1) / 2
    for time in (0.2, 0.6):
        expected = centre + speed * time
        here = np.array([[expected, 0.5]])
        elsewhere = np.array([[expected - 0.3, 0.5]])

        at_pulse = _validation.sample(sol, sol.fn_solution.subs(sol.t, time), here)[0]
        away = _validation.sample(sol, sol.fn_solution.subs(sol.t, time), elsewhere)[0]

        assert at_pulse > 0.9, "pulse is not where advection puts it"
        assert away < 0.1, "pulse has not stayed compact"


def test_transport_solutions_prescribe_their_own_field(mesh):
    """They cannot reuse the Stokes mixins, which apply a velocity."""

    sol = uw.analytic.Poisson1D(mesh)

    temperature = uw.discretisation.MeshVariable("Tb", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, temperature)
    sol.apply_boundary_conditions(poisson)

    assert {bc.boundary for bc in poisson.essential_bcs} == set(sol.boundaries)


def test_transport_solutions_are_registered():
    registered = set(uw.analytic.available())
    assert {"Poisson1D", "TwoLayerDarcy", "ErfcDiffusion", "AdvectedFront"} <= registered

    for name in ("Poisson1D", "TwoLayerDarcy", "ErfcDiffusion", "AdvectedFront"):
        assert getattr(uw.analytic, name).solves == "transport"
