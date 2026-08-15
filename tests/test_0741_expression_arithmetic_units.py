"""Units propagate through products of quantities and expressions.

``cm/year * Myr`` is a length however the two factors are spelled — as
:func:`uw.quantity` (a value with units) or as :func:`uw.expression` (a named
symbol carrying units) — and in either order. Four spellings of the same
product, all of which must come out in centimetres.

The reference quantities live in a FIXTURE, not at module level. This file used
to run its setup at import time, which meant ``set_reference_quantities`` ran
during pytest COLLECTION and switched the units system on globally before a
single test executed. Any module whose own fixture is module-scoped is then
built with units active — and ``var.coords`` returns dimensional coordinates
when they are — which is how a units file broke the point-locator suite
(#567). Global state belongs inside a fixture that takes it down again.
"""
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


@pytest.fixture
def mantle_scaling():
    """A model with a length and a time scale, torn down after the test."""
    orchestration_model = uw.Model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(2900, "km"),
        time=uw.quantity(1, "Myr"),
    )
    yield orchestration_model
    uw.reset_default_model()


def _is_length_in_cm(product):
    return "centimeter" in str(uw.get_units(product))


def test_quantity_times_quantity_is_a_length(mantle_scaling):
    velocity = uw.quantity(5, "cm/year")
    interval = uw.quantity(1, "Myr")
    assert _is_length_in_cm(velocity * interval)


def test_quantity_times_expression_is_a_length(mantle_scaling):
    velocity = uw.quantity(5, "cm/year")
    interval = uw.expression(r"t_\textrm{now}", 1, "Current time", units="Myr")
    assert _is_length_in_cm(velocity * interval)


def test_expression_times_quantity_is_a_length(mantle_scaling):
    """The reverse order goes through ``__rmul__`` and used to lose the units."""
    velocity = uw.quantity(5, "cm/year")
    interval = uw.expression(r"t_\textrm{now}", 1, "Current time", units="Myr")
    assert _is_length_in_cm(interval * velocity)


def test_expression_times_expression_is_a_length(mantle_scaling):
    velocity = uw.expression("v", 5, "velocity", units="cm/year")
    interval = uw.expression("t", 1, "time", units="Myr")
    assert _is_length_in_cm(velocity * interval)


def test_the_product_carries_its_units_onto_its_symbol(mantle_scaling):
    """The units must survive the trip to ``.sym``, which is what the solvers
    and the JIT compiler actually read."""
    product = uw.quantity(5, "cm/year") * uw.expression(
        "t", 1, "time", units="Myr")
    assert hasattr(product, "sym")
    assert _is_length_in_cm(product.sym)
