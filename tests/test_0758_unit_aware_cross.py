#!/usr/bin/env python3
"""Regression: np.cross on a 2-D UnitAwareArray under numpy 2 (#247 follow-up).

numpy 2.0 removed the 2-D cross product (the scalar z-component). The
UnitAwareArray np.cross handler must emulate the 2-D behaviour for (...,2)
inputs (returning the z-component) rather than delegating straight to np.cross,
which raises ``ValueError: Both input arrays must be 3-dimensional vectors``.
3-D cross is unchanged.
"""
import numpy as np
import pytest

from underworld3.utilities.unit_aware_array import UnitAwareArray

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_2d_unit_aware_cross_returns_z_component():
    a = UnitAwareArray([1.0, 2.0], units="m")
    b = UnitAwareArray([3.0, 4.0], units="m")
    r = np.asarray(np.cross(a, b))  # 1*4 - 2*3 = -2
    assert r == pytest.approx(-2.0)


def test_2d_stacked_unit_aware_cross():
    a = UnitAwareArray([[1.0, 0.0], [0.0, 1.0]], units="m")
    b = UnitAwareArray([[0.0, 1.0], [1.0, 0.0]], units="m")
    r = np.asarray(np.cross(a, b))  # [1, -1]
    assert np.allclose(r, [1.0, -1.0])


def test_3d_unit_aware_cross_unchanged():
    a = UnitAwareArray([1.0, 0.0, 0.0], units="m")
    b = UnitAwareArray([0.0, 1.0, 0.0], units="m")
    r = np.asarray(np.cross(a, b))
    assert np.allclose(r, [0.0, 0.0, 1.0])


def test_cross_carries_units():
    a = UnitAwareArray([1.0, 2.0], units="m")
    b = UnitAwareArray([3.0, 4.0], units="m")
    r = np.cross(a, b)
    # result should still be unit-aware (m * m)
    assert getattr(r, "_units", None) is not None
