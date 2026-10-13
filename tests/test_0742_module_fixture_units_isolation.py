"""A module-scoped fixture must be built with the global model reset (#567).

``conftest.isolate_test_state`` resets the default model before every test, and
that reads like enough isolation. It is not. pytest builds higher-scoped
fixtures first, so a ``scope="module"`` fixture — the natural place to put a
mesh several tests share — is set up BEFORE the first test's function-scoped
reset. Whatever the process did earlier is still in force while that mesh is
built, including module-level code that ran during COLLECTION.

That is how a units test broke the point locator. With the units system active,
``var.coords`` returns DIMENSIONAL coordinates: for a 2900 km length scale, the
unit box reads 0..2.9e6 instead of 0..1. A fixture that fills nodal values from
``var.coords`` then samples its field at points 2.9 million times too far
apart, and every later evaluation disagrees with the closed form by O(1) — a
failure that looks like a broken locator and is nothing of the sort.

It only bit when the affected file happened to run FIRST in its process, which
under ``pytest --dist loadfile`` is decided by the worker count. Hence a suite
that was green serially and at 4 and 8 workers, and red at 16.

The check runs a real pytest in a subprocess against a copy of the live
``conftest.py``, because the thing under test IS that conftest: delete the
module-scoped reset from it and this test fails.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

pytest_plugins = ["pytester"]

_LIVE_CONFTEST = (Path(__file__).parent / "conftest.py").read_text()

# The polluter and the victim, in one file. Module-level code runs at
# collection, exactly as an import-time `set_reference_quantities` does; the
# module-scoped fixture is the victim. Deliberately NOT run in this process —
# it would be the very leak we are pinning against.
_LEAKING_MODULE = '''
import numpy as np
import pytest
import underworld3 as uw

# Import-time global state, as test_0741 used to have.
_orchestration_model = uw.Model()
_orchestration_model.set_reference_quantities(
    length=uw.quantity(2900, "km"),
    time=uw.quantity(1, "Myr"),
)


@pytest.fixture(scope="module")
def coordinate_extent():
    """A module-scoped mesh, and how far its variable coordinates reach."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=0.5, regular=False, qdegree=2)
    field = uw.discretisation.MeshVariable("u_iso", mesh, 1, degree=1)
    return float(np.asarray(field.coords).max())


def test_the_fixture_saw_the_undimensionalised_unit_box(coordinate_extent):
    assert coordinate_extent == pytest.approx(1.0), (
        f"the module-scoped fixture read coordinates reaching "
        f"{coordinate_extent:.6g} on a unit box — it was built while the "
        f"units system was still switched on by import-time code")
'''


def test_a_module_scoped_fixture_is_not_built_under_leaked_units(pytester, monkeypatch):
    # The live conftest carries the collection-time guard (#575), which would
    # correctly refuse the leaking module below and end the sub-run before it
    # reached its assertion. The leak is this test's fixture, so the guard is
    # turned off for the sub-run only.
    monkeypatch.setenv("UW_TEST_COLLECTION_GUARD", "off")

    pytester.makeconftest(_LIVE_CONFTEST)
    pytester.makepyfile(test_leaking_module=_LEAKING_MODULE)

    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=1)
