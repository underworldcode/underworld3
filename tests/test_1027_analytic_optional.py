r"""Solutions behind an optional dependency.

`assess` (Kramer et al. 2021) is imported by four curved-geometry benchmark
scripts under `docs/examples/` and declared in no dependency list — so those
scripts fail with a bare `ModuleNotFoundError` on a normal install. It is now an
optional extra with a wrapper that says what to do about it.

These tests cover the *missing*-dependency path, which is the path a normal
install takes and the one the previous arrangement got wrong. Where `assess` is
installed, the constructing tests run too.

Run: pixi run python -m pytest tests/test_1027_analytic_optional.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import underworld3 as uw

from underworld3.analytic.kramer import assess_available

needs_assess = pytest.mark.skipif(
    not assess_available(), reason="the optional 'assess' package is not installed"
)


def test_importing_underworld_does_not_need_the_optional_package():
    """The whole point of a lazy import: absence must cost nothing at import."""

    import underworld3.analytic.kramer  # noqa: F401

    assert uw.analytic.CylindricalStokes is not None


def test_the_solution_is_listed_even_when_it_cannot_be_built():
    """Listed, and marked — not omitted.

    Omitting it would make `available()` truthful about what constructs and
    silent about what exists, which leaves a user who wants a curved-geometry
    benchmark with no way to discover that one is a `pip install` away.
    """

    assert "CylindricalStokes" in uw.analytic.available()
    assert uw.analytic.is_available("CylindricalStokes") == assess_available()

    installed = uw.analytic.available(installed_only=True)
    assert ("CylindricalStokes" in installed) == assess_available()


def test_every_other_solution_is_unconditionally_available():
    for name in uw.analytic.available():
        if name == "CylindricalStokes":
            continue
        assert uw.analytic.is_available(name), name
        assert getattr(uw.analytic, name).requires is None


@pytest.mark.skipif(assess_available(), reason="'assess' is installed here")
def test_describe_says_what_is_missing():
    summary = uw.analytic.describe("CylindricalStokes")

    assert "unavailable" in summary
    assert "assess" in summary


@pytest.mark.skipif(assess_available(), reason="'assess' is installed here")
def test_constructing_it_explains_rather_than_tracebacks():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    with pytest.raises(ImportError) as raised:
        uw.analytic.CylindricalStokes(mesh)

    message = str(raised.value)
    assert "pip install" in message
    assert "assess" in message
    # and it cites the paper, so the user knows what they are installing
    assert "Kramer" in message


def test_it_declares_that_the_residual_gates_cannot_reach_it():
    """A numeric oracle is not a validated solution, and must not pass as one.

    Every other solution is SymPy and clears the physics residual. This one
    cannot: there is no expression to differentiate. The declaration is what
    keeps the conformance sweep from either skipping it silently or reporting a
    pass it never earned.
    """

    assert uw.analytic.CylindricalStokes.symbolic is False
    assert uw.analytic.CylindricalStokes.requires == "assess"

    for name in uw.analytic.available():
        if name != "CylindricalStokes":
            assert getattr(uw.analytic, name).symbolic, name


def test_it_rejects_unknown_cases_before_reaching_the_dependency(monkeypatch):
    """Argument validation must not be hidden behind the optional import.

    Otherwise a typo in `density=` reports a missing package on machines without
    `assess` and a wrong answer on machines with it.
    """

    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    with pytest.raises(ValueError, match="density must be"):
        uw.analytic.CylindricalStokes(mesh, density="gaussian")

    with pytest.raises(ValueError, match="boundary must be"):
        uw.analytic.CylindricalStokes(mesh, boundary="sticky")


@pytest.fixture(scope="module")
def annulus():
    return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.15)


@needs_assess
@pytest.mark.parametrize("density", ["delta", "smooth"])
@pytest.mark.parametrize("boundary", ["free", "zero"])
def test_all_four_cases_construct_and_evaluate(annulus, density, boundary):
    sol = uw.analytic.CylindricalStokes(
        annulus, n=2, k=2, density=density, boundary=boundary
    )

    coords = annulus.X.coords
    velocity = sol.evaluate("velocity", coords)
    pressure = sol.evaluate("pressure", coords)

    assert velocity.shape == (len(coords), 2)
    assert pressure.shape == (len(coords),)
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()


@needs_assess
def test_the_delta_case_is_genuinely_two_sided(annulus):
    """The branch selection is the benchmark, so it has to actually branch."""

    sol = uw.analytic.CylindricalStokes(annulus, density="delta", r_int=0.8)
    assert sol._above is not sol._below

    radii = np.array([[0.6, 0.0], [0.9, 0.0]])
    above, below = sol._split(radii)
    assert below[0] and above[1]


@needs_assess
def test_it_rejects_fields_it_does_not_have(annulus):
    sol = uw.analytic.CylindricalStokes(annulus)

    with pytest.raises(ValueError, match="velocity"):
        sol.evaluate("temperature", annulus.X.coords)

    velocity = uw.discretisation.MeshVariable("Vk", annulus, annulus.dim, degree=2)
    with pytest.raises(ValueError, match="no symbolic form"):
        sol.error("velocity", velocity, norm="integral")
