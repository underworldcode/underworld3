"""``uw.unwrap`` forwards its arguments to ``expressions.unwrap`` by keyword (#824).

``uw.unwrap(fn, keep_constants=True, return_self=True, apply_scaling=False)``
used to call ``expressions.unwrap(fn, keep_constants, return_self)``, whose
second positional parameter is ``depth``. So ``keep_constants=True`` arrived as
``depth=True`` (one level of expansion), ``keep_constants=False`` as ``depth=0``
(none), and ``return_self`` landed in ``keep_constants``.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

x = sympy.Symbol("x")


@pytest.fixture
def nested():
    inner = uw.expression(r"a_{in,824}", 2 * x, "inner")
    outer = uw.expression(r"a_{out,824}", 3 * inner, "outer")
    return outer


def test_top_level_unwrap_expands_every_level(nested):
    assert sympy.simplify(uw.unwrap(nested) - 6 * x) == 0


def test_top_level_unwrap_matches_the_expressions_module(nested):
    reference = uw.function.expressions.unwrap(nested)
    assert sympy.simplify(uw.unwrap(nested) - reference) == 0


def test_keep_constants_false_still_expands(nested):
    assert sympy.simplify(uw.unwrap(nested, keep_constants=False) - 6 * x) == 0


def test_apply_scaling_path_forwards_the_same_way(nested):
    assert sympy.simplify(uw.unwrap(nested, apply_scaling=True) - 6 * x) == 0
