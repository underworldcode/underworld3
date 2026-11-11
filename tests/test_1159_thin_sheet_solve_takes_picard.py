"""A manifold thin sheet takes ``solve(picard=N)`` and an untransported history (#876, #887).

A sheet on a sphere always solves on the rotated path. ``SNES_Vector.solve``
handed that path ``picard=0``, and ``ThinSheet.solve`` took no ``picard``:
``sheet.solve(picard=2)`` was a TypeError, so the ``solve(picard=N)`` recipe of the
plasticity guides could not reach a nonlinear sheet. And a manifold sheet refused
``stress_transport="none"``, which stress-transport.md recommends for a tidal cycle.

- ``picard=3`` stages at least three Picard iterations, then Newton, and converges
  to the answer of the plain solve;
- ``stress_transport="none"`` is taken on a manifold and agrees with the grid
  history on a slow flow; the trace-back flavours are still refused.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _sphere():
    return uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3, degree=1, qdegree=5)


def _toroidal(sphere):
    x, y, z = sphere.X
    return sympy.Matrix([[2 * y * z, -2 * x * z, 0]])


def _nonlinear_sheet(sphere):
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    u = sheet.u.sym
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1 + 50 * u.dot(u)
    sheet.basal_drag = 1
    sheet.bodyforce = list(_toroidal(sphere))
    sheet.tolerance = 1.0e-8
    return sheet


def test_picard_stages_a_picard_phase_on_a_manifold_sheet():
    sphere = _sphere()
    plain = _nonlinear_sheet(sphere)
    plain.solve()
    u_plain = np.asarray(plain.u.array).copy()

    staged = _nonlinear_sheet(sphere)
    # development: TypeError, unexpected keyword argument 'picard'
    staged.solve(picard=3)
    info = staged._rotated_freeslip_info
    assert info["continuation_switched"], "no Picard phase: picard did not reach the rotated loop"
    assert info["nonlinear_iterations"] >= 3, info["nonlinear_iterations"]
    assert info["converged"]
    u_staged = np.asarray(staged.u.array)
    assert np.linalg.norm(u_staged - u_plain) < 1.0e-6 * np.linalg.norm(u_plain)


def test_a_manifold_sheet_takes_an_untransported_history():
    # stress-transport.md recommends "none" for a tidal or seismic cycle; on a
    # manifold it was refused with the trace-back flavours (#887). A slow flow
    # moves far less than a cell a step, so the untransported history agrees with
    # the grid-carried one, and both stay tangent to the sphere.
    sphere = _sphere()
    x, y, z = sphere.X
    taus = {}
    for transport in ("eulerian", "none"):
        sheet = uw.systems.ThinSheet(sphere)
        sheet.stress_transport = transport
        assert sheet.stress_transport == transport
        sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            sheet.Unknowns, order=1, integrator="etd")
        sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
        sheet.constitutive_model.Parameters.shear_modulus = 1
        sheet.basal_drag = 4
        sheet.bodyforce = list(4.0e-3 * sympy.Matrix([[2 * y * z, -2 * x * z, 0]]))
        sheet.tolerance = 1.0e-9
        for _ in range(4):
            sheet.solve(timestep=0.1, zero_init_guess=False)
        taus[transport] = np.asarray(sheet.tau.array).copy()
        c = sheet.tau.coords
        rhat = c / np.linalg.norm(c, axis=1)[:, None]
        normal = np.linalg.norm(np.einsum("nij,nj->ni", taus[transport], rhat), axis=1)
        assert normal.max() < 1.0e-12 * np.abs(taus[transport]).max()
    scale = np.abs(taus["eulerian"]).max()
    assert scale > 0.0
    # they differ (1.1e-4 of the stress, measured), so "none" did take effect
    difference = np.abs(taus["none"] - taus["eulerian"]).max()
    assert 0.0 < difference < 1.0e-3 * scale
    with pytest.raises(ValueError, match="manifold"):
        uw.systems.ThinSheet(sphere).stress_transport = "semi_lagrangian"
