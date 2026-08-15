r"""The whole analytic family, every gate, every parameter case — the slow tier.

**Run it with one command:**

```bash
pixi run -e amr-dev python -m pytest tests/analytic_full/ -v
```

Nothing in CI runs this file, and that is deliberate rather than an oversight.
Run it before a release, and whenever you touch `underworld3/analytic/` or
`_validation.py`.

Why it is separate
------------------

The per-PR sweep (`tests/test_1024_analytic_conformance.py` and
`tests/test_1028_analytic_parameter_sweep.py`) covers every solution that is
cheap to validate. Five are not: SolC accumulates over forty modes, SolDA and
SolH over several more, and SolKx and SolKz carry an exponential viscosity. Each
produces expressions with tens of thousands of operations, and every residual
gate differentiates them symbolically and runs common-subexpression elimination
over the result.

Measured on a quiet machine, those five cost **565s of the analytic suite's
1010s**, against about 17s for the other eight together. CI was already within
five minutes of its 60-minute cap before this suite existed, so they cannot ride
on every PR.

They declare the fact themselves — `AnalyticSolution.expensive_to_validate` —
so this file and the per-PR files partition the family from one source of truth
rather than from two lists that can drift. `test_1024` asserts that every
solution it skips is named here.

**The split is by solutions per run, never by checks per solution.** Every gate
that runs in the per-PR tier runs here too, on every solution: the momentum
residual, incompressibility, tracelessness, strain-rate consistency, and the
body-force negative control. Those are the gates that caught SolA's missing
viscosity, SolM's wrong published stress, SolC's density-not-force and the
elliptical inclusion's unscaled pressure, and none of them is weakened here.

Why file placement rather than a marker
---------------------------------------

`scripts/test.sh` batches by **file glob** (`tests/test_101*py`,
`tests/test_102*py`, ...), not by marker, so a `slow` marker alone would not keep
this out of CI — every batch line would have to remember to deselect it. Those
globs do not recurse, so a subdirectory is excluded by construction. The
`level_2` mark below additionally keeps it out of
`pytest -m "level_1 and tier_a" tests/`, which *does* recurse.
"""

import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a, pytest.mark.slow]

import numpy as np
import sympy
import underworld3 as uw
from underworld3.analytic import _validation


ALL_SYMBOLIC = sorted(
    name
    for name in uw.analytic.available()
    if getattr(uw.analytic, name).symbolic and uw.analytic.is_available(name)
)

STOKES = [n for n in ALL_SYMBOLIC if getattr(uw.analytic, n).solves == "stokes"]
TRANSPORT = [n for n in ALL_SYMBOLIC if getattr(uw.analytic, n).solves == "transport"]
RICHARDS = [n for n in ALL_SYMBOLIC if getattr(uw.analytic, n).solves == "richards"]

# The five that the per-PR sweep drops. Named explicitly so `test_1024` can
# assert they are covered here, and so that reading this file tells you what it
# is for without having to evaluate a comprehension.
EXPENSIVE = ["SolC", "SolDA", "SolH", "SolKx", "SolKz"]

# The complete parameter table. The cheap solutions' entries duplicate
# `test_1028`'s on purpose: this file is the full sweep, and it should not depend
# on which half of the family happens to be cheap this month.
SWEEP = {
    "SolA": [{"eta": 3.0}, {"eta": 0.25}, {"eta": 3.0, "n": 2, "m": 1.5}],
    "SolB": [{"eta": 3.0}, {"eta": 0.25}, {"eta": 3.0, "n": 2, "m": 1.5}],
    "SolC": [{"eta": 3.0}, {"eta": 0.25, "x_c": 0.3}],
    "SolCx": [{"eta_A": 3.0, "eta_B": 0.1}, {"eta_A": 1.0e4, "eta_B": 1.0, "x_c": 0.3}],
    "SolDA": [{"eta_A": 2.0, "eta_B": 0.5}, {"sigma": 3.0, "z_c": 0.4}],
    "SolDB2d": [],
    "SolDB3d": [{"beta": 1.0}, {"beta": 8.0}],
    "SolH": [{"eta": 3.0}, {"eta": 0.25, "dx": 0.3, "dy": 0.4}],
    # exactly the cases test_1021 dropped from its per-PR CASES
    "SolKx": [{"B": 1.0, "n": 2, "m": 1}, {"B": 4.0, "n": 1, "m": 3}, {"B": 5.0, "n": 2, "m": 2}],
    # exactly the cases test_1023 dropped from its per-PR CASES
    "SolKz": [{"B": 1.0, "n": 2, "m": 1}, {"B": 4.0, "n": 1, "m": 3}, {"B": 5.0, "n": 2, "m": 2}],
    "SolM": [{"eta_0": 3.0}, {"eta_0": 0.5, "n": 2, "m": 1, "r": 3.0}],
    "SolNL": [{"eta_0": 2.0}, {"r": 2.5}],
    "EllipticalInclusion": [
        {"viscosity_ratio": 10.0, "aspect_ratio": 1.5},
        {"matrix_viscosity": 3.0},
        {"matrix_viscosity": 0.5, "viscosity_ratio": 100.0},
    ],
}

# Defaults first, then every off-default case.
CASES = [(name, {}) for name in STOKES] + [
    (name, kw) for name in STOKES for kw in SWEEP.get(name, [])
]


def test_the_expensive_list_is_accurate():
    """`EXPENSIVE` is a hand-written list; check it against the declarations."""

    declared = sorted(
        n for n in ALL_SYMBOLIC if getattr(uw.analytic, n).expensive_to_validate
    )
    assert declared == sorted(EXPENSIVE), (
        f"declared expensive_to_validate = {declared}, but EXPENSIVE says {EXPENSIVE}"
    )


def test_this_file_sweeps_the_whole_family():
    """Nothing is dropped from BOTH tiers."""

    assert set(SWEEP) == set(STOKES), (
        f"unswept: {set(STOKES) - set(SWEEP)}; stale: {set(SWEEP) - set(STOKES)}"
    )
    assert len(ALL_SYMBOLIC) >= 19, "the sweep has lost solutions"


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

    Building a series solution substitutes parameters into expressions tens of
    thousands of operations long — SolDA takes 12s and SolKz 6s. Four gates run
    over every case here, so rebuilding per test would multiply the file's cost
    by four for nothing.
    """

    return {}


def _built(cache, meshes, name, kw):
    key = (name, tuple(sorted(kw.items())))
    if key not in cache:
        cls = getattr(uw.analytic, name)
        cache[key] = cls(meshes[cls.dim], **kw)
    return cache[key]


IDS = [f"{n}-{kw}" for n, kw in CASES]


@pytest.mark.parametrize("name,kw", CASES, ids=IDS)
def test_momentum_balance(cache, meshes, name, kw):
    r""":math:`\nabla\cdot\sigma + \mathbf f = 0`."""

    sol = _built(cache, meshes, name, kw)
    assert _validation.momentum_residual(sol, sol.sample_points(count=8)) < 1.0e-8


@pytest.mark.parametrize("name,kw", CASES, ids=IDS)
def test_incompressible(cache, meshes, name, kw):
    sol = _built(cache, meshes, name, kw)
    assert _validation.incompressibility_residual(sol, sol.sample_points(count=8)) < 1.0e-8


@pytest.mark.parametrize("name,kw", CASES, ids=IDS)
def test_deviator_is_traceless(cache, meshes, name, kw):
    r""":math:`\mathrm{tr}(\sigma + p\mathbf I) = 0`.

    Independent of the body force, which is what made it the gate that localised
    the SolA erratum.
    """

    sol = _built(cache, meshes, name, kw)
    points = sol.sample_points(count=8)
    deviator = sol.fn_stress + sol.fn_pressure * sympy.eye(sol.dim)

    sampled = [
        _validation.sample(sol, deviator[i, i], points) for i in range(sol.dim)
    ]
    scale = max(float(np.abs(v).max()) for v in sampled)

    assert float(np.abs(sum(sampled)).max()) / max(scale, 1e-300) < 1.0e-8


@pytest.mark.parametrize("name,kw", CASES, ids=IDS)
def test_strain_rate_matches_the_velocity(cache, meshes, name, kw):
    r""":math:`\tfrac12(\nabla\mathbf u + \nabla\mathbf u^{T})` vs `fn_strainrate`."""

    sol = _built(cache, meshes, name, kw)
    assert _validation.strainrate_consistency(sol, sol.sample_points(count=8)) < 1.0e-8


FORCED = [n for n in STOKES if n != "EllipticalInclusion"]


@pytest.mark.parametrize("name", FORCED)
def test_flipping_the_body_force_breaks_the_momentum_balance(cache, meshes, name):
    r"""The momentum gate must REJECT :math:`\nabla\cdot\sigma - \mathbf f = 0`.

    The negative control, run here over the whole family. Without it the momentum
    gate is only an assertion that a small number is small.

    EllipticalInclusion is excluded because it is boundary-driven and has no body
    force to flip — asserted below rather than skipped silently.
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


def test_the_inclusion_really_has_no_body_force(cache, meshes):
    """The stated grounds for excluding it from the negative control above."""

    sol = _built(cache, meshes, "EllipticalInclusion", {})
    assert all(component == 0 for component in sol.fn_bodyforce)


@pytest.mark.parametrize("name", TRANSPORT)
def test_transport_solution_satisfies_its_equation(cache, meshes, name):
    """Steady or transient, whichever the solution declares — as in `test_1024`."""

    sol = _built(cache, meshes, name, {})
    points = sol.sample_points(count=8)

    if getattr(sol, "t", None) is None:
        assert _validation.transport_residual(sol, points) < 1.0e-10
        return

    for time in (0.05, 0.2, 0.5):
        assert _validation.diffusion_residual(sol, points, time) < 1.0e-10


@pytest.mark.parametrize("name", RICHARDS)
def test_richards_solution_satisfies_its_equation(cache, meshes, name):
    r""":math:`C(\psi)\partial_t\psi = \nabla\cdot[K(\psi)(\nabla\psi + \hat y)]`."""

    sol = _built(cache, meshes, name, {})
    points = sol.sample_points(count=8)

    if getattr(sol, "t", None) is None:
        assert _validation.richards_residual(sol, points) < 1.0e-10
        return

    for time in (0.05, 0.2):
        assert _validation.richards_residual(sol, points, time) < 1.0e-10
