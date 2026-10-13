r"""The Gardner unsaturated-flow solutions.

Richards is the one nonlinear scalar equation in the suite: the conductivity
depends on the unknown head. Gardner's :math:`K = K_s e^{\alpha\psi}` is the case
that closes, because :math:`u = e^{\alpha\psi}` linearises it *exactly*.

These solutions were already in the repository as NumPy functions in
`utilities/retention_curves.py`, where nothing checked them against the equation
they solve. Both are exact here, so the residual assertions are tight.

Run: pixi run python -m pytest tests/test_1026_analytic_richards.py -v
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


def test_steady_satisfies_richards(mesh):
    from underworld3.analytic import _validation

    sol = uw.analytic.GardnerSteady(mesh)
    assert _validation.richards_residual(sol, sol.sample_points(count=8)) < 1e-12


@pytest.mark.parametrize("time", [0.05, 0.1, 0.2])
def test_transient_satisfies_richards(mesh, time):
    from underworld3.analytic import _validation

    sol = uw.analytic.GardnerTransient(mesh)
    points = sol.sample_points(count=8)
    assert _validation.richards_residual(sol, points, time) < 1e-12


@pytest.mark.parametrize(
    "break_it, label",
    [
        (
            lambda sol: setattr(
                sol,
                "fn_conductivity",
                sol.Ks * sympy.exp(sympy.Rational(11, 10) * sol.alpha * sol.fn_solution),
            ),
            "wrong alpha in K",
        ),
        (
            lambda sol: setattr(sol, "fn_conductivity", sympy.sympify(sol.Ks)),
            "conductivity independent of head",
        ),
        (
            lambda sol: setattr(
                sol, "fn_solution", sol.fn_solution * sympy.Rational(101, 100)
            ),
            "head scaled by 1%",
        ),
    ],
)
def test_the_residual_discriminates(mesh, break_it, label):
    """Gate 5: a check that passes a broken input is measuring nothing.

    Note what is *not* used as a control here. Scaling :math:`K` by a constant
    leaves the residual at zero — correctly, because that is a genuine symmetry
    of the steady equation, not a defect the check missed. A negative control
    has to break the relationship between the conductivity and the head, which
    is the thing the solution actually asserts.
    """

    from underworld3.analytic import _validation

    sol = uw.analytic.GardnerSteady(mesh)
    points = sol.sample_points(count=8)
    assert _validation.richards_residual(sol, points) < 1e-12

    break_it(sol)
    assert _validation.richards_residual(sol, points) > 1e-3, label


def test_scaling_conductivity_is_a_symmetry_not_a_defect(mesh):
    """The counterpart to the control above, asserted rather than assumed."""

    from underworld3.analytic import _validation

    sol = uw.analytic.GardnerSteady(mesh)
    sol.fn_conductivity = 2 * sol.fn_conductivity
    assert _validation.richards_residual(sol, sol.sample_points(count=8)) < 1e-12


def test_steady_flux_is_constant_down_the_column(mesh):
    r""":math:`K(\psi)(\partial_y\psi + 1)` is the same at every height.

    This is the physical content of the steady solution and a sharper test of a
    Richards solver than the head profile: the head can look right while the
    conductivity is evaluated at the wrong place, and then the flux drifts.
    """

    from underworld3.analytic import _validation

    sol = uw.analytic.GardnerSteady(mesh)
    y = mesh.X[1]

    flux = sol.fn_conductivity * (sympy.diff(sol.fn_solution, y) + 1)
    heights = np.column_stack(
        [np.full(9, 0.5), np.linspace(0.05, 0.95, 9)]
    )
    values = _validation.sample(sol, flux, heights)

    assert np.ptp(values) / np.abs(values).max() < 1e-12
    # and it is the flux the construction reported
    assert np.allclose(values, float(sol.Ks) * float(sol.flux), rtol=1e-10)


def test_steady_meets_its_boundary_heads(mesh):
    from underworld3.analytic import _validation

    psi_0, psi_L = -1.5, -4.0
    sol = uw.analytic.GardnerSteady(mesh, psi_0=psi_0, psi_L=psi_L, L=1.0)

    ends = np.array([[0.5, 0.0], [0.5, 1.0]])
    assert np.allclose(
        _validation.sample(sol, sol.fn_solution, ends), [psi_0, psi_L], rtol=1e-10
    )


def _half_saturation_depth(sol, time, alpha, psi_dry, psi_wet):
    r"""Depth at which :math:`u = e^{\alpha\psi}` is halfway between its limits.

    Halfway in :math:`u`, not in :math:`\psi`. The two are very different: at
    :math:`\alpha=1` with :math:`\psi` running from -5 to -0.5, the midpoint
    *head* sits at :math:`H \approx 0.1`, out in the leading tail, and tracking
    it measures how far the tail has spread rather than where the front is.
    """

    from underworld3.analytic import _validation

    level = np.log((np.exp(alpha * psi_dry) + np.exp(alpha * psi_wet)) / 2) / alpha
    column = np.column_stack([np.full(2000, 0.5), np.linspace(0.0, 1.0, 2000)])
    values = _validation.sample(sol, sol.fn_solution.subs(sol.t, time), column)

    wetted = column[values > level, 1]
    return 1.0 - wetted.min() if wetted.size else np.nan


def test_transient_front_advances_at_the_advective_speed(mesh):
    r"""The front tracks :math:`V = K_s/\Delta\theta` when advection dominates.

    Checked at :math:`\mathrm{Pe} = \alpha L = 20`, where the front is sharp
    enough for "where the front is" to mean something. The default parameters
    give :math:`\mathrm{Pe} = 1` — see the test below, which is the same
    measurement and deliberately does *not* assert the advective speed.
    """

    from underworld3.analytic import _validation

    alpha, psi_dry, psi_wet = 20.0, -0.5, -0.05
    sol = uw.analytic.GardnerTransient(
        mesh, psi_dry=psi_dry, psi_wet=psi_wet, alpha=alpha
    )
    assert np.isclose(sol.speed, 1.0 / (0.45 - 0.05))
    assert np.isclose(sol.speed / sol.diffusivity, alpha)

    early = _half_saturation_depth(sol, 0.05, alpha, psi_dry, psi_wet)
    late = _half_saturation_depth(sol, 0.15, alpha, psi_dry, psi_wet)

    assert late > early, "front did not advance"
    assert np.isclose(late - early, sol.speed * 0.10, rtol=0.15)

    surface = np.array([[0.5, 1.0]])
    head = sol.fn_solution.subs(sol.t, 0.05)
    assert _validation.sample(sol, head, surface)[0] > psi_dry, "surface not wet"


def test_transient_front_is_diffusion_dominated_at_unit_peclet(mesh):
    """At Pe = 1 the front outruns advection, because it is mostly spreading.

    Recorded so the tolerance in the test above is not mistaken for a general
    claim about where the front is: at the default parameters the same
    measurement gives roughly 1.6x the advective distance, and that is the
    solution behaving correctly rather than a defect.
    """

    alpha, psi_dry, psi_wet = 1.0, -5.0, -0.5
    sol = uw.analytic.GardnerTransient(
        mesh, psi_dry=psi_dry, psi_wet=psi_wet, alpha=alpha
    )
    assert np.isclose(sol.speed / sol.diffusivity, alpha)

    early = _half_saturation_depth(sol, 0.05, alpha, psi_dry, psi_wet)
    late = _half_saturation_depth(sol, 0.15, alpha, psi_dry, psi_wet)

    assert late > early
    assert (late - early) > 1.3 * sol.speed * 0.10


def test_transient_reports_when_the_semi_infinite_form_expires(mesh):
    """The solution is only usable while the front is inside the column."""

    sol = uw.analytic.GardnerTransient(mesh, L=1.0)

    assert sol.front_depth(0.1) < 1.0
    assert sol.front_depth(1.0) > 1.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(alpha=0.0), "alpha and Ks must be positive"),
        (dict(Ks=-1.0), "alpha and Ks must be positive"),
        (dict(theta_r=0.5, theta_s=0.4), "theta_s must exceed theta_r"),
    ],
)
def test_rejects_unphysical_parameters(mesh, kwargs, message):
    with pytest.raises(ValueError, match=message):
        uw.analytic.GardnerSteady(mesh, **kwargs)


def test_transient_requires_the_surface_to_be_wetter(mesh):
    with pytest.raises(ValueError, match="psi_wet must be wetter"):
        uw.analytic.GardnerTransient(mesh, psi_dry=-0.5, psi_wet=-5.0)


def test_retention_curve_wrappers_agree_with_the_solution(mesh):
    """The NumPy functions and the mesh classes are one formula, not two.

    `retention_curves` keeps its published signatures; this asserts it did not
    keep a second copy of the arithmetic along with them.
    """

    from underworld3.analytic import _validation
    from underworld3.utilities import retention_curves as rc

    heights = np.linspace(0.05, 0.95, 11)
    points = np.column_stack([np.full(heights.size, 0.5), heights])

    steady = uw.analytic.GardnerSteady(
        mesh, psi_0=-1.0, psi_L=-5.0, L=1.0, alpha=1.0
    )
    assert np.allclose(
        _validation.sample(steady, steady.fn_solution, points),
        rc.gardner_steady_state_psi(heights, -1.0, -5.0, 1.0, 1.0),
        rtol=1e-12,
    )

    transient = uw.analytic.GardnerTransient(
        mesh, psi_dry=-5.0, psi_wet=-0.5, L=1.0, Ks=1.0, alpha=1.0,
        theta_r=0.05, theta_s=0.45,
    )
    assert np.allclose(
        _validation.sample(transient, transient.fn_solution.subs(transient.t, 0.1), points),
        rc.gardner_transient_psi(
            heights, 0.1, -5.0, -0.5, 1.0, 1.0, 1.0, 0.05, 0.45
        ),
        rtol=1e-12,
    )


def test_richards_solutions_are_registered():
    registered = set(uw.analytic.available())
    assert {"GardnerSteady", "GardnerTransient"} <= registered

    for name in ("GardnerSteady", "GardnerTransient"):
        assert getattr(uw.analytic, name).solves == "richards"
