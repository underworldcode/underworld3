r"""SolKz — Stokes flow with a depth-dependent viscosity.

The vertical twin of SolKx: :math:`\eta = e^{2Bz}` instead of :math:`e^{2Bx}`.
Not a redundant one — a viscosity varying with *depth* stratifies the flow along
the direction the buoyancy acts, coupling pressure and vertical velocity through
the varying coefficient in a way a horizontal gradient never does. It is also the
closer analogue of a real mantle viscosity profile.

Validated by the equations: the forcing and the boundary conditions are known, so
satisfying Stokes with them identifies the solution uniquely.

The convention trap this solution carries is pinned by
`test_kernel_publishes_the_deviatoric_stress`. Its kernel writes into an array
named `total_stress`, but the contents are the deviator. Reading the name at face
value leaves the momentum residual at order |f| *and* invents a horizontal body
force in a benchmark that has none.

Run: pixi run python -m pytest tests/test_1023_analytic_solkz.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


CASES = [
    (2.302585092994046, 3, 2),
    (1.0, 2, 1),
    (4.0, 1, 3),
    (5.0, 2, 2),
]

INTERIOR = np.array([(0.2, 0.3), (0.7, 0.8), (0.5, 0.5), (0.9, 0.15), (0.05, 0.95)])
WALLS = {
    "left": np.array([(0.0, t) for t in (0.13, 0.47, 0.82)]),
    "right": np.array([(1.0, t) for t in (0.13, 0.47, 0.82)]),
    "bottom": np.array([(t, 0.0) for t in (0.13, 0.47, 0.82)]),
    "top": np.array([(t, 1.0) for t in (0.13, 0.47, 0.82)]),
}


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


def _at(sol, expression, points):
    """Magnitudes over a whole point set — lambdified once, not once per point."""

    from underworld3.analytic import _validation

    return np.abs(_validation.sample(sol, expression, points))


@pytest.mark.parametrize("B,n,m", CASES)
def test_solkz_satisfies_the_stokes_equations(mesh, B, n, m):
    sol = uw.analytic.SolKz(mesh, B=B, n=n, m=m)
    x, z = mesh.X

    scale = _at(sol, sol.fn_bodyforce[0, 1], INTERIOR).max()

    residual_x = _at(
        sol,
        sympy.diff(sol.fn_stress[0, 0], x) + sympy.diff(sol.fn_stress[0, 1], z),
        INTERIOR,
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
def test_solkz_is_free_slip_on_every_wall(mesh, B, n, m):
    sol = uw.analytic.SolKz(mesh, B=B, n=n, m=m)
    vx, vz = sol.fn_velocity[0, 0], sol.fn_velocity[0, 1]

    assert _at(sol, vx, WALLS["left"]).max() < 1.0e-10
    assert _at(sol, vx, WALLS["right"]).max() < 1.0e-10
    assert _at(sol, vz, WALLS["bottom"]).max() < 1.0e-10
    assert _at(sol, vz, WALLS["top"]).max() < 1.0e-10


def test_kernel_publishes_the_deviatoric_stress(mesh):
    r"""The kernel's ``total_stress`` array holds :math:`\tau`, not :math:`\sigma`.

    Two independent signatures, both checked here because the array's *name*
    says otherwise and following it silently breaks the momentum balance:

    - a deviator is traceless, so its xx and zz entries are exact negatives;
    - :math:`\tau = 2\eta\dot\varepsilon`, and the strain rate follows from the
      velocity, which is a different output of the same kernel.
    """

    sol = uw.analytic.SolKz(mesh, B=2.0, n=2, m=1)
    x, z = mesh.X

    deviator = sol.fn_stress + sol.fn_pressure * sympy.eye(2)

    trace = _at(sol, deviator[0, 0] + deviator[1, 1], INTERIOR)
    magnitude = _at(sol, deviator[0, 0], INTERIOR).max()
    assert trace.max() / magnitude < 1.0e-10

    from_velocity = sol.fn_viscosity * (
        sympy.diff(sol.fn_velocity[0, 0], z) + sympy.diff(sol.fn_velocity[0, 1], x)
    )
    shear = _at(sol, deviator[0, 1] - from_velocity, INTERIOR)
    shear_scale = _at(sol, deviator[0, 1], INTERIOR).max()
    assert shear.max() / shear_scale < 1.0e-10


def test_solkz_viscosity_varies_with_depth_not_width(mesh):
    """The distinction from SolKx, asserted so a copy-paste error cannot hide."""

    B = 2.0
    sol = uw.analytic.SolKz(mesh, B=B, n=2, m=1)
    eta = sympy.lambdify(tuple(mesh.X), sol.fn_viscosity, "numpy")

    assert np.isclose(float(eta(0.2, 0.7)), np.exp(2 * B * 0.7))
    assert np.isclose(float(eta(0.9, 0.7)), np.exp(2 * B * 0.7))  # x makes no difference
    assert float(eta(0.5, 0.1)) < float(eta(0.5, 0.9))


def test_solkz_rejects_fractional_wavenumbers(mesh):
    with pytest.raises(ValueError, match="n .*must be a positive integer"):
        uw.analytic.SolKz(mesh, n=1.5)
    with pytest.raises(ValueError, match="m .*must be a positive integer"):
        uw.analytic.SolKz(mesh, m=2.5)


def test_solkz_is_registered():
    assert "SolKz" in uw.analytic.available()
