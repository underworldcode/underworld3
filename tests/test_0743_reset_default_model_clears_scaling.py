"""Non-dimensional scaling does not leak from one test into the next (#1001).

``set_reference_quantities`` switches the global scaling flag on, and
``reset_default_model`` leaves it as it was (``test_0816`` pins that: the flag is
global, not the model's). ``conftest``'s isolation fixtures reset the model, so
without resetting the flag too, a test that set reference quantities hands the
next test a fresh model with no reference quantities and scaling still on.

The check runs a real pytest in a subprocess against a copy of the live
``conftest.py``, because the thing under test is that conftest.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

pytest_plugins = ["pytester"]

_LIVE_CONFTEST = (Path(__file__).parent / "conftest.py").read_text()

_TWO_TESTS = '''
import underworld3 as uw


def test_a_sets_reference_quantities():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"))
    assert uw.is_nondimensional_scaling_active()


def test_b_starts_unscaled():
    assert not uw.get_default_model().has_units()
    assert not uw.is_nondimensional_scaling_active(), (
        "scaling is still on from the previous test")
'''


def test_scaling_flag_does_not_leak_between_tests(pytester):
    pytester.makeconftest(_LIVE_CONFTEST)
    pytester.makepyfile(test_flag_leak=_TWO_TESTS)

    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=2)
