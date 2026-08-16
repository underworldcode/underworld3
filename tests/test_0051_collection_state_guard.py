"""The collection-time global-state guard in `tests/conftest.py` (#575).

The guard's whole value is that it fires, so it is tested by running pytest on a
module that offends and on one that does not. Without the second run the first
proves only that something failed.

The sub-run loads the guard's hooks out of the real `tests/conftest.py` by path,
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

pytest_collectstart = guard.pytest_collectstart
pytest_collectreport = guard.pytest_collectreport
pytest_collection_finish = guard.pytest_collection_finish
"""


def test_guard_fails_the_run_on_module_level_work(pytester):
    """A mesh built at module level is reported, with the module named."""

    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(
        test_offender="import underworld3 as uw\n"
        "mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))\n"
        "def test_x():\n    assert True\n"
    )

    result = pytester.runpytest_subprocess("--collect-only")

    assert result.ret != 0
    result.stderr.fnmatch_lines(
        ["*changed global state while being COLLECTED*", "*test_offender.py*"]
    )


def test_guard_is_silent_when_nothing_is_built(pytester):
    """The control: a module that only defines a test collects cleanly."""

    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(test_clean="def test_x():\n    assert True\n")

    result = pytester.runpytest_subprocess("--collect-only")

    assert result.ret == 0
