#!/usr/bin/env python3
"""Regression: UWQuantity handles offset temperature units (degC/degF).

Pint forbids scalar arithmetic on offset (non-multiplicative) units — both
``value * u.degC`` and ``qty * 2`` raise ``OffsetUnitCalculusError``. Previously
``UWQuantity.__init__`` built its Pint quantity as ``value * unit``, so
``uw.quantity(20, "degC")`` crashed at construction, and any offset temperature
that slipped through non-dimensionalised silently to the wrong value.

The fix normalises an offset temperature to absolute **kelvin at the input
boundary**, so the stored quantity is always multiplicative and no internal code
(arithmetic, non-dimensionalisation) ever meets an offset unit. The user reads it
back in degC via ``.to("degC")`` — the only place an offset unit reappears is the
output boundary. A temperature *difference* (``delta_degC``) is multiplicative and
must be left untouched.
"""
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_construct_degC_does_not_crash():
    """uw.quantity(., 'degC') constructs (previously OffsetUnitCalculusError)."""
    q = uw.quantity(20.0, "degC")
    assert q is not None


def test_offset_normalised_to_kelvin_internally():
    """An offset temperature is stored as absolute kelvin (multiplicative)."""
    q = uw.quantity(20.0, "degC")
    assert str(q.units) == "kelvin"
    assert q.value == pytest.approx(293.15)

    qf = uw.quantity(32.0, "degF")
    assert str(qf.units) == "kelvin"
    assert qf.value == pytest.approx(273.15)


def test_display_round_trips_to_degC():
    """.to('degC') reproduces the input value (output boundary keeps offset)."""
    q = uw.quantity(20.0, "degC")
    back = q.to("degC")
    assert str(back.units) == "degree_Celsius"
    assert back.value == pytest.approx(20.0)


def test_kelvin_displays_to_degC():
    """A kelvin quantity converts to degC for display."""
    assert uw.quantity(300.0, "K").to("degC").value == pytest.approx(26.85)


def test_scalar_multiply_offset_input_does_not_raise():
    """Scalar arithmetic works because the stored unit is kelvin, not degC."""
    q = uw.quantity(20.0, "degC")  # -> 293.15 K
    doubled = q * 2
    assert doubled.value == pytest.approx(586.3)
    assert str(doubled.units) == "kelvin"


def test_delta_degC_is_not_normalised():
    """A temperature DIFFERENCE is multiplicative and must be preserved."""
    q = uw.quantity(5.0, "delta_degC")
    assert "delta" in str(q.units)
    assert q.value == pytest.approx(5.0)


def test_non_dimensionalise_offset_matches_kelvin_equivalent():
    """nd(500 degC) equals nd(773.15 K) under an active model (no silent error)."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
        temperature_difference=uw.quantity(1000, "K"),
    )
    nd_degC = float(uw.quantity(500.0, "degC").data)   # 500 degC = 773.15 K
    nd_K = float(uw.quantity(773.15, "K").data)
    assert nd_degC == pytest.approx(nd_K)
    assert nd_degC == pytest.approx(0.77315)


def test_non_offset_units_unaffected():
    """A plain (non-temperature) unit is stored and non-dimensionalised as before."""
    q = uw.quantity(1000.0, "km")
    assert str(q.units) == "kilometer"
    assert q.value == pytest.approx(1000.0)
