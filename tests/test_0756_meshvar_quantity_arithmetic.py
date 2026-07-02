#!/usr/bin/env python3
"""Regression: EnhancedMeshVariable arithmetic with a UWQuantity operand (#282).

A mesh variable's ``.sym`` lives in non-dimensional (model-unit) space, so a
unit-bearing scalar operand (a ``UWQuantity``) must be reduced to its
non-dimensional value before composing. Previously ``T - uw.quantity(500, "K")``
raised ``TypeError: Cannot subtract UWQuantity from EnhancedMeshVariable`` even
for compatible units — blocking buoyancy expressions ``T - T_ref``.

Scope: the variable on the LEFT (``meshvar ± quantity``, ``meshvar * quantity``).
``quantity - meshvar`` (quantity on the left) is handled by the UWQuantity class
and is out of scope here.
"""
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def units_mesh():
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
        temperature_difference=uw.quantity(1000, "K"),
    )
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1000.0, 1000.0), units="km"
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="K")
    return mesh, T


def _scalar(expr):
    return expr[0, 0] if hasattr(expr, "shape") else expr


def test_meshvar_minus_quantity(units_mesh):
    """T - quantity[K] composes, and equals T.sym - non_dimensional(quantity)."""
    _, T = units_mesh
    result = _scalar(T - uw.quantity(500.0, "K"))
    # 500 K / 1000 K scale -> 0.5 non-dimensional
    assert sympy.simplify(result - (T.sym[0, 0] - 0.5)) == 0


def test_meshvar_quantity_unit_prefix(units_mesh):
    """A prefixed unit (kK) is reduced consistently: 0.5 kK == 500 K -> 0.5 ND."""
    _, T = units_mesh
    r_kK = _scalar(T - uw.quantity(0.5, "kK"))
    r_K = _scalar(T - uw.quantity(500.0, "K"))
    assert sympy.simplify(r_kK - r_K) == 0


def test_meshvar_plus_and_times_quantity(units_mesh):
    """Addition and multiplication with a quantity also compose."""
    _, T = units_mesh
    # T + 500K -> T.sym + 0.5
    assert sympy.simplify(_scalar(T + uw.quantity(500.0, "K")) - (T.sym[0, 0] + 0.5)) == 0
    # T * 500K -> T.sym * 0.5
    assert sympy.simplify(_scalar(T * uw.quantity(500.0, "K")) - (T.sym[0, 0] * 0.5)) == 0


def test_non_quantity_arithmetic_unchanged(units_mesh):
    """The fix must not disturb non-quantity operands."""
    _, T = units_mesh
    # variable + variable, and scalar multiply, still work
    _ = T + T
    assert sympy.simplify(_scalar(T * 2.0) - (T.sym[0, 0] * 2.0)) == 0
