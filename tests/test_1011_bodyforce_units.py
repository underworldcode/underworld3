#!/usr/bin/env python3
"""Regression: Stokes.bodyforce accepts a UWQuantity component (units-active).

A units-active buoyancy force ``F = ρ₀ α g (T − T_ref) ẑ`` is a UWQuantity with a
*symbolic* magnitude (it carries the T field). The bodyforce setter must
non-dimensionalise it before building the (ND) body-force matrix. Previously it
called ``item._sympify_()`` — which does not exist on UWQuantity (``_sympy_``
also raises on a dimensional quantity) — so any units-active buoyancy assignment
crashed (the second blocker, after #267/#282, for the thermal-convection units
tutorial).
"""
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_bodyforce_accepts_uwquantity_buoyancy():
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        mantle_viscosity=uw.quantity(1e21, "Pa*s"),
        temperature_difference=uw.quantity(1000, "K"),
    )
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1000.0, 1000.0)
    )
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="K")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

    rho = uw.quantity(3300, "kg/m**3")
    alpha = uw.quantity(3e-5, "1/K")
    g = uw.quantity(9.8, "m/s**2")
    T_ref = uw.quantity(500, "K")
    buoyancy = rho * alpha * g * (T - T_ref)  # symbolic UWQuantity

    # Must not raise; the vertical component becomes a non-dimensional sympy expr.
    stokes.bodyforce = [0, buoyancy]
    fz = stokes.bodyforce.sym[1]
    assert isinstance(fz, sympy.Expr)
    assert fz.free_symbols, "body force should depend on the T field"


def test_bodyforce_plain_vector_unchanged():
    """A plain (non-quantity) body force is unaffected."""
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("v2", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p2", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    assert stokes.bodyforce.sym[1] == -1.0
