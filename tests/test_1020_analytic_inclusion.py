r"""Schmid & Podladchikov's elliptical inclusion.

The authors publish pressure, stress and the rotation rate but not the velocity,
so the velocity here is reconstructed from the Muskhelishvili potentials. That
makes validation the whole story, and it has to come from somewhere other than
the thing being validated. Four independent sources are used:

- the **Stokes residual** :math:`\eta\nabla^2\mathbf v = \nabla p`, which pairs
  the reconstructed velocity against the *published* pressure;
- **velocity continuity** across the inclusion boundary, which pairs the exterior
  reconstruction against the interior uniform-gradient field;
- the **far field**, against the imposed shear computed independently here;
- **interior uniformity**, the Eshelby property the solution must have.

Two subtleties are pinned because each produced a plausible-looking wrong answer:

- a purely imaginary constant in :math:`\varphi'` is invisible to pressure and
  stress, so the published data cannot constrain it. Omitting it gives a flow
  with the correct strain and no spin — a simple shear that comes out as pure
  shear. `test_far_field_matches_the_imposed_shear` catches that.
- inverting the conformal map with ``sqrt(z**2 - 4)`` picks the wrong sheet left
  of the origin. `test_far_field_matches_the_imposed_shear` samples negative x
  for exactly that reason.

Run: pixi run python -m pytest tests/test_1020_analytic_inclusion.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


# Weak and strong inclusions, both shear types, both signs of orientation.
CASES = [
    (1.0e3, 2.0, -np.pi / 6, 0.0, 1.0),
    (1.0e-2, 3.0, 0.4, 1.0, 0.5),
    (1.0e6, 1.5, 0.0, 1.0, 0.0),
    (0.1, 4.0, -1.1, 0.5, 1.0),
]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(-3.0, -3.0), maxCoords=(3.0, 3.0), qdegree=3
    )


def _real(mesh, expression):
    """Lambdify a field that is real-valued but complex-typed.

    The velocity is built from complex potentials, and SymPy cannot prove the
    imaginary part vanishes even though it does to roundoff.
    """

    f = sympy.lambdify(tuple(mesh.X), expression, "numpy")
    return lambda x, y: complex(f(x, y)).real


def _solution(mesh, viscosity_ratio, aspect_ratio, alpha, pure_shear, simple_shear):
    return uw.analytic.EllipticalInclusion(
        mesh,
        viscosity_ratio=viscosity_ratio,
        aspect_ratio=aspect_ratio,
        alpha=alpha,
        pure_shear=pure_shear,
        simple_shear=simple_shear,
        semi_major=1.0,
    )


@pytest.mark.parametrize("mc,t,alpha,er,gr", CASES)
def test_stokes_residual_vanishes_in_the_matrix(mesh, mc, t, alpha, er, gr):
    r"""With no body force, :math:`\eta\nabla^2\mathbf v - \nabla p = 0`.

    The pressure is the published closed form and the velocity is the
    reconstruction, so this is a cross-check between two different sources, not a
    restatement of one.
    """

    sol = _solution(mesh, mc, t, alpha, er, gr)
    x, y = mesh.X
    vx, vy, p = sol.fn_velocity[0, 0], sol.fn_velocity[0, 1], sol.fn_pressure

    residual_x = _real(mesh, sympy.diff(vx, x, 2) + sympy.diff(vx, y, 2) - sympy.diff(p, x))
    residual_y = _real(mesh, sympy.diff(vy, x, 2) + sympy.diff(vy, y, 2) - sympy.diff(p, y))
    divergence = _real(mesh, sympy.diff(vx, x) + sympy.diff(vy, y))

    points = [(2.4, 1.9), (-2.2, 1.3), (1.5, -2.6), (-1.1, -2.9), (0.0, 2.8)]
    for px, py in points:
        assert abs(residual_x(px, py)) < 1.0e-10
        assert abs(residual_y(px, py)) < 1.0e-10
        assert abs(divergence(px, py)) < 1.0e-10


@pytest.mark.parametrize("mc,t,alpha,er,gr", CASES)
def test_velocity_is_continuous_across_the_interface(mesh, mc, t, alpha, er, gr):
    """Interior and exterior are different expressions; they must agree on the boundary.

    Nothing in the construction forces this — the interior comes from the
    published interior stress and rotation rate, the exterior from the
    potentials — so agreement is real evidence.
    """

    sol = _solution(mesh, mc, t, alpha, er, gr)
    vx, vy = _real(mesh, sol.fn_velocity[0, 0]), _real(mesh, sol.fn_velocity[0, 1])
    a, b = sol.semi_axes

    step = 1.0e-7
    for theta in np.linspace(0.1, 2 * np.pi - 0.1, 9):
        px, py = a * np.cos(theta), b * np.sin(theta)
        radius = np.hypot(px, py)
        inner = np.array(
            [vx(px * (1 - step / radius), py * (1 - step / radius)),
             vy(px * (1 - step / radius), py * (1 - step / radius))]
        )
        outer = np.array(
            [vx(px * (1 + step / radius), py * (1 + step / radius)),
             vy(px * (1 + step / radius), py * (1 + step / radius))]
        )
        # The tolerance is set by the finite step across the interface, not by
        # the solution: the velocity is continuous but its gradient is not.
        assert np.linalg.norm(inner - outer) < 1.0e-5


@pytest.mark.parametrize("mc,t,alpha,er,gr", CASES)
def test_far_field_matches_the_imposed_shear(mesh, mc, t, alpha, er, gr):
    r"""Far from the inclusion the flow is the shear that drives it.

    The comparison is computed here from :math:`\alpha`, the shear rates and the
    Muskhelishvili far field, independently of the solution object. Negative
    :math:`x` is sampled deliberately: that is where the conformal map's branch
    goes wrong if the square root is written as a single ``sqrt(z**2 - 4)``.
    """

    sol = _solution(mesh, mc, t, alpha, er, gr)
    vx, vy = _real(mesh, sol.fn_velocity[0, 0]), _real(mesh, sol.fn_velocity[0, 1])

    BC = (2 * er - 1j * gr) * np.exp(2j * alpha)

    for px, py in [(300.0, 220.0), (-400.0, 150.0), (0.0, 500.0), (-250.0, -330.0)]:
        z = px + 1j * py
        expected = np.conj(BC) * np.conj(z) / 2 + 1j * (-gr / 2) * z
        got = np.array([vx(px, py), vy(px, py)])
        want = np.array([expected.real, expected.imag])

        # The inclusion's own perturbation decays as 1/r^2, so at r ~ 500 with a
        # unit inclusion a few parts in 1e6 remain. That is physics, not error.
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1.0e-4


def test_interior_deformation_is_uniform(mesh):
    """The Eshelby property: strain rate is constant inside the inclusion."""

    sol = _solution(mesh, 1.0e3, 2.0, -np.pi / 6, 0.0, 1.0)
    x, y = mesh.X

    exx = _real(mesh, sympy.diff(sol.fn_velocity[0, 0], x))
    exy = _real(
        mesh,
        (sympy.diff(sol.fn_velocity[0, 0], y) + sympy.diff(sol.fn_velocity[0, 1], x)) / 2,
    )

    interior = [(0.1, 0.05), (-0.3, 0.1), (0.5, -0.15), (0.0, 0.0)]
    reference = (exx(*interior[0]), exy(*interior[0]))
    for px, py in interior[1:]:
        assert abs(exx(px, py) - reference[0]) < 1.0e-12
        assert abs(exy(px, py) - reference[1]) < 1.0e-12


def test_a_circular_inclusion_turns_with_the_far_field(mesh):
    """A circle rotates at the far-field vorticity whatever its viscosity.

    This is the identity that fixes the otherwise-unconstrained rotation in the
    potentials, so it is worth asserting rather than trusting.
    """

    for viscosity_ratio in (1.0e-3, 1.0, 1.0e6):
        sol = _solution(mesh, viscosity_ratio, 1.0001, 0.3, 0.7, 1.0)
        assert abs(sol.rotation_rate - (-0.5)) < 1.0e-3


def test_a_circle_exactly_is_refused(mesh):
    """Aspect ratio one is a degenerate conformal map, not a usable circle."""

    with pytest.raises(ValueError, match="aspect_ratio must exceed 1"):
        _solution(mesh, 10.0, 1.0, 0.0, 0.0, 1.0)


def test_geometry_scales_with_semi_major(mesh):
    """Lengths scale; the strain rate does not."""

    small = uw.analytic.EllipticalInclusion(mesh, aspect_ratio=2.0, semi_major=0.5)
    large = uw.analytic.EllipticalInclusion(mesh, aspect_ratio=2.0, semi_major=2.0)

    assert np.isclose(small.semi_axes[0], 0.5)
    assert np.isclose(large.semi_axes[0], 2.0)
    assert np.isclose(small.semi_axes[0] / small.semi_axes[1], 2.0)
