"""The collection-time global-state guard in `tests/conftest.py` (#575).

The guard's whole value is that it fires, so it is tested by running pytest on a
module that offends and on one that does not. Without the second run the first
proves only that something failed.

The distributed case has its own test because it is what CI runs and because the
guard failed it in its first form: aborting the session from
`pytest_collection_finish` left an xdist worker part-collected, and the
controller reported `INTERNALERROR ... assert not crashitem`. Reporting a
collection error against the offending module instead is carried by both
runners.

The sub-runs load the guard's hooks out of the real `tests/conftest.py` by path,
so this tests the shipped hooks rather than a copy of them.
"""

import pathlib

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.level_1

_CONFTEST = pathlib.Path(__file__).parent / "conftest.py"

# The sub-run gets its own rootdir, so it does not inherit our conftest. Load it
# by path under a name of its own: `from conftest import *` would find the
# sub-run's own half-initialised `conftest` module in sys.modules and import
# nothing, which reads as the guard staying silent.
_SUB_CONFTEST = f"""
import importlib.util

spec = importlib.util.spec_from_file_location("uw_collection_guard", {str(_CONFTEST)!r})
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

pytest_make_collect_report = guard.pytest_make_collect_report
"""

_OFFENDER = (
    "import underworld3 as uw\n"
    "mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))\n"
    "def test_x():\n    assert True\n"
)


def test_guard_reports_module_level_work_as_a_collection_error(pytester):
    """A mesh built at module level fails the run, with the module named."""

    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(test_offender=_OFFENDER)

    result = pytester.runpytest_subprocess("-q")

    assert result.ret != 0
    result.stdout.fnmatch_lines(
        ["*changed global state while being COLLECTED*", "*ERROR*test_offender.py*"]
    )


def test_guard_reports_the_same_way_under_xdist(pytester):
    """The distributed runner carries it as a collection error, not an INTERNALERROR."""

    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(test_offender=_OFFENDER)

    result = pytester.runpytest_subprocess("-q", "-n", "2")

    assert result.ret != 0
    result.stdout.fnmatch_lines(["*ERROR*test_offender.py*"])
    assert "INTERNALERROR" not in result.stdout.str()


def test_guard_can_be_turned_off_for_a_generated_offender(pytester, monkeypatch):
    """`UW_TEST_COLLECTION_GUARD=off` lets the same module through.

    `test_0742` needs this: it generates a module that leaks units at import,
    because what it pins is that the module-scoped reset survives exactly that.
    """

    monkeypatch.setenv("UW_TEST_COLLECTION_GUARD", "off")
    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(test_offender=_OFFENDER)

    result = pytester.runpytest_subprocess("-q")

    assert result.ret == 0


def test_guard_is_silent_when_nothing_is_built(pytester):
    """The control: a module that only defines a test collects and runs cleanly."""

    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(test_clean="def test_x():\n    assert True\n")

    result = pytester.runpytest_subprocess("-q")

    assert result.ret == 0
