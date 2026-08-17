r"""The residual gates, applied AWAY from the default parameters.

`test_1024_analytic_conformance.py` builds every solution from a mesh alone.
That is deliberate — the defaults are part of the interface — but it means a
coefficient that happens to be unity by default multiplies a term nothing ever
looks at.

That is not hypothetical either. SolA's published $\sigma_{zz}$ (solA.c:156) is
missing the factor of viscosity that its own $\sigma_{xx}$ carries and that
solB's matching line carries. The shortfall is $\tau_{zz}(1-Z)/Z$, identically
zero at $Z = 1$ — the default `eta`, and the only value the kernel's own
disabled driver ever exercised. Every gate in the conformance file passed. At
`eta=3` three of them fail.

So: each solution is swept over parameters that are *not* its defaults, and the
oracle-free gates are re-applied. The sweep table is explicit rather than
generated, because the point is to move the coefficients that matter for each
solution — a viscosity contrast, a wavenumber, an interface position — and
which those are is per-solution knowledge.

Run: pixi run python -m pytest tests/test_1028_analytic_parameter_sweep.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw
from underworld3.analytic import _validation


# solution -> parameter sets to try IN ADDITION to the defaults. Every entry
# moves at least one coefficient off unity or off its default position.
SWEEP = {
    "SolA": [{"eta": 3.0}, {"eta": 0.25}, {"eta": 3.0, "n": 2, "m": 1.5}],
    "SolB": [{"eta": 3.0}, {"eta": 0.25}, {"eta": 3.0, "n": 2, "m": 1.5}],
    "SolCx": [{"eta_A": 3.0, "eta_B": 0.1}, {"eta_A": 1.0e4, "eta_B": 1.0, "x_c": 0.3}],
    "SolM": [{"eta_0": 3.0}, {"eta_0": 0.5, "n": 2, "m": 1, "r": 3.0}],
    "SolNL": [{"eta_0": 2.0}, {"r": 2.5}],
    "SolDB2d": [],  # a manufactured solution with no free parameters
    "SolDB3d": [{"beta": 1.0}, {"beta": 8.0}],
    "EllipticalInclusion": [
        {"viscosity_ratio": 10.0, "aspect_ratio": 1.5},
        {"matrix_viscosity": 3.0},
        {"matrix_viscosity": 0.5, "viscosity_ratio": 100.0},
    ],
    # The tip position is this solution's equivalent of a viscosity off unity:
    # at the default it sits on the origin, where a dropped offset cancels.
    "FaultedMedium": [
        {"U0": 2.5},
        {"R0": 3.0, "eta": 0.25},
        {"U0": 1.3, "R0": 2.0, "eta": 0.7, "tip": (0.3, -0.2)},
    ],
}

ALL_STOKES = sorted(
    name
    for name in uw.analytic.available()
    if getattr(uw.analytic, name).symbolic
    and uw.analytic.is_available(name)
    and getattr(uw.analytic, name).solves == "stokes"
)

# SolC, SolDA, SolH, SolKx and SolKz declare `expensive_to_validate` and are
# swept in tests/analytic_full/ instead — same table, same gates, just not on
# every PR. See that file, and the note on the class attribute, for the cost.
STOKES = [n for n in ALL_STOKES if not getattr(uw.analytic, n).expensive_to_validate]

CASES = [(name, kw) for name in STOKES for kw in SWEEP.get(name, [])]


def test_every_stokes_solution_is_swept():
    """A solution added later must be given a sweep, or fail here.

    SolDB2d is listed with an empty sweep rather than omitted: "this one has no
    parameters" is a claim worth recording, and it is different from "nobody got
    round to it".

    A solution that declares itself expensive belongs in the full-family table
    rather than this one, so it must be absent here and present there — the
    conformance file asserts the second half.
    """

    assert set(SWEEP) == set(STOKES), (
        f"unswept: {set(STOKES) - set(SWEEP)}; stale: {set(SWEEP) - set(STOKES)}"
    )


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
def cache():
    """One construction per (solution, parameters), shared across the gates.

    Constructing a series solution substitutes parameters into expressions tens
    of thousands of operations long. Three gates run over every case here, so
    building afresh in each would treble the cost of the file for nothing —
    the same reason the conformance sweep uses a module-scoped fixture.
    """

    return {}


def _built(cache, meshes, name, kw):
    key = (name, tuple(sorted(kw.items())))
    if key not in cache:
        cls = getattr(uw.analytic, name)
        cache[key] = cls(meshes[cls.dim], **kw)
    return cache[key]


@pytest.mark.parametrize("name,kw", CASES, ids=[f"{n}-{kw}" for n, kw in CASES])
def test_momentum_balance_off_default(cache, meshes, name, kw):
    r""":math:`\nabla\cdot\sigma + \mathbf f = 0` away from the defaults."""

    sol = _built(cache, meshes, name, kw)
    assert _validation.momentum_residual(sol, sol.sample_points(count=8)) < 1.0e-8


@pytest.mark.parametrize("name,kw", CASES, ids=[f"{n}-{kw}" for n, kw in CASES])
def test_deviator_is_traceless_off_default(cache, meshes, name, kw):
    r""":math:`\mathrm{tr}(\sigma + p\mathbf I) = 0`.

    Equivalent to the pressure being :math:`-\mathrm{tr}\,\sigma/d`, i.e. to
    `fn_stress` being the total stress with pressure positive in compression.
    Independent of the body force, which is what made it the gate that localised
    the SolA erratum: it fired without any reference to the forcing.
    """

    sol = _built(cache, meshes, name, kw)
    points = sol.sample_points(count=8)
    deviator = sol.fn_stress + sol.fn_pressure * sympy.eye(sol.dim)

    trace = sum(deviator[i, i] for i in range(sol.dim))
    scale = max(
        np.abs(_validation.sample(sol, deviator[i, i], points)).max()
        for i in range(sol.dim)
    )

    assert np.abs(_validation.sample(sol, trace, points)).max() / max(scale, 1e-300) < 1.0e-8


@pytest.mark.parametrize("name,kw", CASES, ids=[f"{n}-{kw}" for n, kw in CASES])
def test_strain_rate_matches_the_velocity_off_default(cache, meshes, name, kw):
    r""":math:`\tfrac12(\nabla\mathbf u + \nabla\mathbf u^{T})` vs `fn_strainrate`.

    The genuinely independent pair: the velocity and the stress are separate
    outputs of the same kernel, so this is not the identity `set_fields` used to
    derive one from the other.
    """

    sol = _built(cache, meshes, name, kw)
    assert _validation.strainrate_consistency(sol, sol.sample_points(count=8)) < 1.0e-8


# --------------------------------------------------------------------------
# Negative controls. A gate that has never been seen to fail is a gate nobody
# has checked can fail.
# --------------------------------------------------------------------------


# Two solutions are driven entirely by their boundaries and have no body force
# to flip. That is a property of the problems, so they are named here and the
# claim is asserted below rather than skipped silently.
FORCE_FREE = {"EllipticalInclusion", "FaultedMedium"}

FORCED = [n for n in STOKES if n not in FORCE_FREE]


@pytest.mark.parametrize("name", FORCED)
def test_flipping_the_body_force_breaks_the_momentum_balance(cache, meshes, name):
    r"""The momentum gate must REJECT :math:`\nabla\cdot\sigma - \mathbf f = 0`.

    This is what pins the sign convention to UW3's own. The solver assembles
    ``F0 = -bodyforce`` against ``F1 = stress``, whose strong form is
    :math:`\nabla\cdot\sigma + \mathbf f = 0`; if the suite tolerated the other
    sign it would not be validating that.

    The solutions in FORCE_FREE are excluded because they have no body force to
    flip — they are driven entirely by their boundaries. That is a property of
    those problems, so they are excluded by name here and asserted below rather
    than skipped silently.
    """

    sol = _built(cache, meshes, name, {})
    points = sol.sample_points(count=8)

    assert _validation.momentum_residual(sol, points) < 1.0e-8

    keep = sol.fn_bodyforce
    sol.fn_bodyforce = sympy.Matrix([[-c for c in keep]])
    try:
        flipped = _validation.momentum_residual(sol, points)
    finally:
        sol.fn_bodyforce = keep

    assert flipped > 1.0e-2, (
        f"{name}: flipping the body force left the momentum residual at "
        f"{flipped:.3e} — the gate cannot see the sign it is supposed to certify"
    )


@pytest.mark.parametrize("name", sorted(FORCE_FREE))
def test_the_force_free_solutions_really_have_no_body_force(cache, meshes, name):
    """The stated grounds for excluding them from the negative control above."""

    sol = _built(cache, meshes, name, {})
    assert all(component == 0 for component in sol.fn_bodyforce)


@pytest.mark.parametrize("kernel_name,repaired", [("solA", True), ("solB", False)])
def test_only_solA_is_missing_the_viscosity_in_its_zz_stress(meshes, kernel_name, repaired):
    r"""The erratum is real, and it is confined to solA.

    Read the two kernels directly and test the deviator they publish for
    tracelessness at a viscosity that is not one. solB passes as published;
    solA does not, and passes only after the missing factor of $Z$ is restored.

    Asserting BOTH halves is the point. That solA needs the repair is one claim;
    that solB does not is the control which says the repair is a correction to a
    defect rather than a convention we imposed on the family.
    """

    from underworld3.analytic import velic

    mesh = meshes[2]
    eta = 3.0
    sol = uw.analytic.SolB(mesh, eta=eta)  # only a carrier for `sample`
    points = sol.sample_points(count=8)

    kernel = velic._solab_kernel(kernel_name)
    values = {
        velic._SIGMA: sympy.Rational(1.0),
        velic._ETA0: sympy.Rational(eta),
        velic._N: 3,
        velic._KM: sympy.Rational(2) * sympy.pi,
        velic._X: mesh.X[0],
        velic._Z: mesh.X[1],
    }
    published = {k: v.subs(values) for k, v in kernel.items()}

    tau_xx = published["stress_xx"] + published["pressure"]
    tau_zz = published["stress_zz"] + published["pressure"]

    scale = np.abs(_validation.sample(sol, tau_xx, points)).max()
    as_published = np.abs(_validation.sample(sol, tau_xx + tau_zz, points)).max() / scale

    if repaired:
        assert as_published > 1.0e-2, (
            f"{kernel_name}: expected the published deviator to be non-traceless "
            f"at eta={eta}, got {as_published:.3e} — has the vendored kernel changed?"
        )
        restored = np.abs(
            _validation.sample(sol, tau_xx + sympy.Rational(eta) * tau_zz, points)
        ).max() / scale
        assert restored < 1.0e-10
    else:
        assert as_published < 1.0e-10


def test_solNL_refuses_its_pole():
    """r = 2 makes the published pressure's denominator vanish identically."""

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )

    with pytest.raises(ValueError, match="pole"):
        uw.analytic.SolNL(mesh, r=2.0)


def test_the_inclusion_honours_its_matrix_viscosity(cache, meshes):
    r"""Rescaling $\eta$ at fixed boundary velocity scales $\sigma$ AND $p$.

    The velocity and strain rate are invariant; the stress and the pressure both
    carry the factor. Scaling only the viscous part leaves $\nabla\cdot\sigma = 0$
    broken while tracelessness and strain-rate consistency still pass, which is
    why this needs asserting directly.
    """

    one = _built(cache, meshes, "EllipticalInclusion", {})
    three = _built(cache, meshes, "EllipticalInclusion", {"matrix_viscosity": 3.0})
    points = one.sample_points(count=8)

    for i in range(2):
        a = _validation.sample(one, one.fn_velocity[0, i], points)
        b = _validation.sample(three, three.fn_velocity[0, i], points)
        assert np.abs(a - b).max() / max(np.abs(a).max(), 1e-300) < 1.0e-10

    a = _validation.sample(one, one.fn_pressure, points)
    b = _validation.sample(three, three.fn_pressure, points)
    assert np.abs(b - 3.0 * a).max() / max(np.abs(b).max(), 1e-300) < 1.0e-10
