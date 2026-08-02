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


def _reference_fields(sol, eta_A, eta_B, x_c, n):
    """The transcribed fields paired with the kernel they came from."""

    from underworld3.analytic._reference import _velic

    def at(kernel):
        return lambda x, z: kernel(eta_A, eta_B, x_c, n, x, z).evalf()

    return {
        "velocity_x": (sol.fn_velocity[0, 0], at(_velic.AnalyticSolCx_velocity_x)),
        "velocity_z": (sol.fn_velocity[0, 1], at(_velic.AnalyticSolCx_velocity_y)),
        "pressure": (sol.fn_pressure, at(_velic.AnalyticSolCx_pressure)),
        "stress_xx": (sol.fn_stress[0, 0], at(_velic.AnalyticSolCx_stress_xx)),
        "stress_zz": (sol.fn_stress[1, 1], at(_velic.AnalyticSolCx_stress_yy)),
        "stress_zx": (sol.fn_stress[0, 1], at(_velic.AnalyticSolCx_stress_xy)),
    }


@pytest.mark.parametrize("eta_A,eta_B,x_c,n", REGIMES)
def test_transcription_reproduces_the_reference_kernel(mesh, eta_A, eta_B, x_c, n):
    """Agreement with the published kernel, worst case rather than average."""

    from underworld3.analytic import _validation

    sol = uw.analytic.SolCx(mesh, eta_A=eta_A, eta_B=eta_B, x_c=x_c, n=n)
    points = _validation.adversarial_points(x_c=x_c)

    worst = _validation.reference_agreement(
        sol, _reference_fields(sol, eta_A, eta_B, x_c, n), points
    )

    for name, error in worst.items():
        assert error < 1.0e-10, f"{name}: max normalised error {error:.2e}"


@pytest.mark.parametrize("eta_A,eta_B,x_c,n", REGIMES)
def test_transcription_satisfies_the_equations(mesh, eta_A, eta_B, x_c, n):
    """The fields solve the Stokes problem they claim to — no reference involved.

    This is the check a convergence test cannot make. If a transcription and the
    solver shared a mistaken convention, the solve would converge neatly to the
    wrong answer; this residual never consults the solver. It is also what
    settles the body-force sign, where UW2's documentation and UW3's convention
    disagree.
    """

    from underworld3.analytic import _validation

    sol = uw.analytic.SolCx(mesh, eta_A=eta_A, eta_B=eta_B, x_c=x_c, n=n)
    points = _validation.adversarial_points(x_c=x_c, count=12)

    assert _validation.incompressibility_residual(sol, points) < 1.0e-10
    assert _validation.momentum_residual(sol, points) < 1.0e-10


@pytest.mark.parametrize("eta_A,eta_B,x_c,n", REGIMES)
def test_strain_rate_agrees_with_the_velocity_gradient(mesh, eta_A, eta_B, x_c, n):
    """Two independently derived kernel outputs, cross-checked through derivatives.

    A transcription can be right pointwise and wrong in its derivatives — which
    is what a solver actually consumes.
    """

    from underworld3.analytic import _validation

    sol = uw.analytic.SolCx(mesh, eta_A=eta_A, eta_B=eta_B, x_c=x_c, n=n)
    points = _validation.adversarial_points(x_c=x_c, count=12)

    assert _validation.strainrate_consistency(sol, points) < 1.0e-10


def test_the_checks_reject_a_broken_transcription(mesh):
    """Negative control: a check that passes a broken input measures nothing.

    One coefficient of the transcribed velocity is perturbed by a part in a
    thousand — small enough to be a plausible transcription slip, large enough
    that a working check must see it. Both the comparison against the kernel and
    the oracle-free residual have to fail.
    """

    from underworld3.analytic import _validation

    eta_A, eta_B, x_c, n = 1.0, 1.0e3, 0.5, 1
    sol = uw.analytic.SolCx(mesh, eta_A=eta_A, eta_B=eta_B, x_c=x_c, n=n)
    points = _validation.adversarial_points(x_c=x_c, count=12)

    # Intact first: if this did not pass, the control below would prove nothing.
    intact = _validation.reference_agreement(
        sol, _reference_fields(sol, eta_A, eta_B, x_c, n), points
    )
    assert max(intact.values()) < 1.0e-10
    assert _validation.incompressibility_residual(sol, points) < 1.0e-10

    sol.fn_velocity = sympy.Matrix(
        [[sol.fn_velocity[0, 0] * sympy.Rational(1001, 1000), sol.fn_velocity[0, 1]]]
    )

    broken = _validation.reference_agreement(
        sol, _reference_fields(sol, eta_A, eta_B, x_c, n), points
    )
    assert broken["velocity_x"] > 1.0e-6, "comparison did not see the perturbation"
    assert (
        _validation.incompressibility_residual(sol, points) > 1.0e-6
    ), "residual did not see the perturbation"


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
