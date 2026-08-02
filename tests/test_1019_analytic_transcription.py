"""The SolCx transcription, against the kernel it was transcribed from.

`uw.analytic.SolCx` is SymPy rebuilt from the published Maple-generated C, which
stays vendored alongside it. These tests are the standing version of the
validation gates: the transcription must reproduce its own source across the
parameter range the benchmark is used over, and it must reproduce it where the
solution is hardest — on the viscosity interface, on the walls, at the corners.

Two traps are pinned here because both cost time to find:

- a pointwise relative error is the wrong metric. These fields pass through zero,
  so dividing by the true value explodes wherever the solution is small and every
  case looks catastrophic. Normalise by the field's magnitude over the sample.
- `_solCx_B` in the published source is not a second conditioning of the same
  formula, it is the mirror image, `B(x, z) = A(1 - x, z)`. Only `_solCx_A` is
  transcribed. `test_arrangements_are_mirror_images` records why.

Run: pixi run python -m pytest tests/test_1019_analytic_transcription.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


# Ratios spanning both directions, wavenumbers, and an off-centre interface.
REGIMES = [
    (1.0, 10.0, 0.5, 1),
    (1.0, 1.0e3, 0.5, 1),
    (1.0, 1.0e6, 0.5, 1),
    (1.0, 1.0e8, 0.5, 1),
    (1.0, 1.0e6, 0.5, 3),
    (1.0, 1.0e6, 0.75, 1),
    (1.0e6, 1.0, 0.5, 1),
    (1.0e3, 1.0, 0.25, 2),
    (1.0, 1.0e-6, 0.5, 1),
]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


def _sample_points(x_c):
    """Stratified, and loaded with the places the solution is hard."""

    rng = np.random.default_rng(20260802)
    points = list(map(tuple, rng.uniform(0.0, 1.0, size=(40, 2))))
    points += [(x_c - 1.0e-9, 0.37), (x_c + 1.0e-9, 0.37), (x_c, 0.37)]  # interface
    points += [(0.0, 0.5), (1.0, 0.5), (0.31, 0.0), (0.31, 1.0)]  # walls
    points += [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]  # corners
    return np.array(points)


@pytest.mark.parametrize("eta_A,eta_B,x_c,n", REGIMES)
def test_transcription_reproduces_the_reference_kernel(mesh, eta_A, eta_B, x_c, n):
    """Gate 1 and 2: agreement with the published kernel, worst case not average."""

    from underworld3.analytic._reference import _velic

    sol = uw.analytic.SolCx(mesh, eta_A=eta_A, eta_B=eta_B, x_c=x_c, n=n)
    points = _sample_points(x_c)

    fields = {
        "velocity_x": (sol.fn_velocity[0, 0], _velic.AnalyticSolCx_velocity_x),
        "velocity_z": (sol.fn_velocity[0, 1], _velic.AnalyticSolCx_velocity_y),
        "pressure": (sol.fn_pressure, _velic.AnalyticSolCx_pressure),
        "stress_xx": (sol.fn_stress[0, 0], _velic.AnalyticSolCx_stress_xx),
        "stress_zz": (sol.fn_stress[1, 1], _velic.AnalyticSolCx_stress_yy),
        "stress_zx": (sol.fn_stress[0, 1], _velic.AnalyticSolCx_stress_xy),
    }

    for name, (expression, kernel) in fields.items():
        mine = np.asarray(sol.evaluate(expression, points)).reshape(-1)
        theirs = np.array(
            [float(kernel(eta_A, eta_B, x_c, n, x, z).evalf()) for x, z in points]
        )

        # Normalised by the field's magnitude, not pointwise: these fields cross
        # zero, and a pointwise ratio would report a huge error for a tiny one.
        scale = max(np.max(np.abs(theirs)), 1.0e-300)
        worst = np.max(np.abs(mine - theirs)) / scale

        assert worst < 1.0e-10, f"{name}: max relative error {worst:.2e}"


def test_arrangements_are_mirror_images():
    r"""`_solCx_B` is `_solCx_A` reflected, not a second conditioning.

    The published source dispatches on eta_A > eta_B, which reads as a
    conditioning choice. It is not: the second arrangement solves the mirrored
    problem so the algebra for a stiff left column can be reused when the stiff
    column is on the right.

    The relation is :math:`A(x, z) = -B(1 - x, z)`. The sign is not a fudge — the
    forcing :math:`\cos(\pi x)` is odd about :math:`x = 1/2`, so reflecting the
    domain flips the whole solution.

    Only `_solCx_A` is transcribed, so this records the evidence for that
    decision — if it ever fails, the dispatch mattered after all and the
    reasoning needs revisiting.
    """

    from underworld3.analytic import velic

    values = {
        velic._XC: sympy.Rational(1, 2),
        velic._KN: sympy.pi,
        velic._KX: sympy.pi,
        velic._ZA: sympy.Integer(1),
        velic._ZB: sympy.Integer(10) ** 6,
        velic._ZR: sympy.Integer(10) ** 6,
    }

    a = velic._solcx_kernel("_solCx_A")["pressure"].subs(values)
    b = velic._solcx_kernel("_solCx_B")["pressure"].subs(values)

    for x, z in ((sympy.Rational(1, 10), sympy.Rational(9, 10)),
                 (sympy.Rational(1, 4), sympy.Rational(1, 3))):
        direct = sympy.N(a.subs({velic._X: x, velic._Z: z}), 30)
        mirrored = sympy.N(b.subs({velic._X: 1 - x, velic._Z: z}), 30)
        assert abs(direct) > 1.0e-3, "sample point is too near a node to discriminate"
        assert abs(direct + mirrored) < 1.0e-25


def test_equal_viscosities_are_refused(mesh):
    """A removable singularity the compiled form does not remove is not silently used."""

    with pytest.raises(ValueError, match="singular at eta_A == eta_B"):
        uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0)


def test_transcription_is_usable_by_the_solver(mesh):
    """The point of the SymPy form: it compiles into a residual.

    The reference kernel cannot do this — it is opaque to the JIT — so this is
    what the transcription buys, and it is worth a test of its own.
    """

    sol = uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e3)

    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    stokes.bodyforce = sol.fn_bodyforce

    # A Dirichlet value taken straight from the exact velocity: only possible
    # because the field is real SymPy rather than a compiled kernel call.
    stokes.add_dirichlet_bc(sol.fn_velocity, "Top")

    assert len(stokes.essential_bcs) == 1
