#!/usr/bin/env python3
"""The re-declaration guard, and the one path that is allowed to refresh.

An expression container's identity is its NAME, so a second construction under
a name reaches the object the first one made. Construction used to write the
contents anyway, which meant a line that reads like a declaration could rebind
a name two unrelated owners both used, under every formula already written
against it. The guard refuses that.

A solver's own templates are the case that must still work: ``F0`` and ``F1``
are re-derived on every access, so their contents move as the timestep and the
Adams-Moulton coefficients do. ``template_expression`` is the path that says
so, and these tests pin both halves -- the refusal and the refresh -- because
weakening either one restores a silent overwrite.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function.expressions import (
    UWexpression,
    reset_expression_registry,
    template_expression,
)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Names are process-wide, so a test that leaves one behind fails its neighbour."""
    reset_expression_registry()
    yield
    reset_expression_registry()


def test_a_second_declaration_with_different_contents_is_refused():
    """The guard's whole point: two owners reaching one name must not pass quietly."""
    UWexpression(r"guard_0027_a", sympy.Symbol("first"), "owner one")
    with pytest.raises(ValueError, match="already exists and holds"):
        UWexpression(r"guard_0027_a", sympy.Symbol("second"), "owner two")


def test_a_second_declaration_of_the_same_contents_is_allowed():
    """A factory rebuilding an unmutated problem changes nothing, and no formula
    written against the name can tell. Refusing it would be noise."""
    first = UWexpression(r"guard_0027_b", sympy.Symbol("same"), "owner")
    again = UWexpression(r"guard_0027_b", sympy.Symbol("same"), "owner")
    assert again is first
    assert again.sym == sympy.Symbol("same")


def test_a_template_refresh_updates_the_container_in_place():
    """What the solver properties rely on: the same object, new contents, so a
    formula already written against the name sees the refreshed value."""
    first = template_expression(r"guard_0027_c", sympy.Symbol("dt_1"), "a template")
    formula = 2 * first

    second = template_expression(r"guard_0027_c", sympy.Symbol("dt_2"), "a template")

    assert second is first, "a refresh must not make a second container"
    assert first.sym == sympy.Symbol("dt_2")
    # the formula holds the container, not a copy of its contents
    assert sympy.Symbol("dt_2") in formula.subs(first, first.sym).atoms(sympy.Symbol)


def test_a_template_refresh_keeps_the_units_it_was_declared_with():
    """A refresh goes through the same wrapping construction uses.

    Writing straight to ``.sym`` does not wrap, so a refresh that skipped the
    wrapping replaced ``1.0 [metre]`` with a bare ``2.0`` and the container
    lost the units it was declared with -- silently, since nothing downstream
    asks whether a length is still a length.
    """
    declared = template_expression(r"guard_0027_d", 1.0, "a length", units="m")
    assert declared.units is not None

    refreshed = template_expression(r"guard_0027_d", 2.0, "a length", units="m")

    assert refreshed is declared
    assert refreshed.units is not None, "the refresh dropped the units"
    assert str(refreshed.units) == str(declared.units)


def test_a_template_refresh_carries_the_description_it_was_given():
    """The description is what ``describe()`` and the transcript print, so a
    refresh that kept the first one would make them quote a stale line."""
    template_expression(r"guard_0027_e", sympy.Symbol("q"), "first wording")
    refreshed = template_expression(r"guard_0027_e", sympy.Symbol("r"), "second wording")
    assert refreshed._description == "second wording"


def test_resetting_the_registry_frees_lookup_and_leaves_live_objects_alone():
    """``_expr_names`` belongs to the process, not to a model.

    A reset is about LOOKUP: the next declaration under a freed name builds a
    fresh container. Objects already holding a container keep working, and the
    creation counter keeps counting, so a reset cannot be used to make two runs
    look alike by rewinding the numbering.
    """
    held = UWexpression(r"guard_0027_f", sympy.Symbol("kept"), "held by a formula")
    formula = held + 1
    count_before = UWexpression._expr_count

    reset_expression_registry()

    assert r"guard_0027_f" not in UWexpression._expr_names
    assert held.sym == sympy.Symbol("kept"), "a live container must be untouched"
    assert formula.subs(held, held.sym) == sympy.Symbol("kept") + 1
    assert UWexpression._expr_count >= count_before, "the counter must not rewind"

    # the freed name is now available, and declaring it is not a collision
    fresh = UWexpression(r"guard_0027_f", sympy.Symbol("new"), "a different owner")
    assert fresh is not held
    assert held.sym == sympy.Symbol("kept")


def test_the_guard_does_not_fire_on_a_fetch_with_no_value():
    """Asking for a container by name is a lookup, not a declaration."""
    made = UWexpression(r"guard_0027_g", sympy.Symbol("v"), "owner")
    fetched = uw.expression(r"guard_0027_g")
    assert fetched is made
    assert fetched.sym == sympy.Symbol("v")
