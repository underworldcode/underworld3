"""The declared plastic rate strengthening eta_reg: eta_pl -> tau_y/(2 edot_II) + m eta_reg.

Why: a yielded element's consistent tangent has no modulus along its own strain-rate
direction, so a yielded layer is a mechanism the velocity multigrid cannot solve
(Spiegelman notch, 2026-10-03: Rayleigh quotient 1.8e-7). A static viscosity floor clips
only the fastest cells; eta_reg floors the plastic viscosity AND the tangent everywhere.
Its scale is the model's (no universal dimensionless value), it is laddered as a
multiplier m from the viscous limit down to 1, and the solve records it.
"""
import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _box(tag, eta_reg):
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("Vr" + tag, mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pr" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = s.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = 0.3 + 0.3 * sympy.sqrt(p.sym[0] ** 2 + 1.0e-2)
    if eta_reg:
        cm.Parameters.plastic_rate_strengthening = eta_reg
    s.add_essential_bc((1.0, 0.0), "Top")
    s.add_essential_bc((0.0, 0.0), "Bottom")
    s.petsc_options.delValue("ksp_monitor")
    s.tolerance = 1.0e-6
    s.petsc_options["snes_max_it"] = 300
    return s, v, p, cm


def test_the_term_floors_the_plastic_viscosity_and_is_recorded():
    """At a strain rate far above yield the effective viscosity sits at m*eta_reg (the
    plastic branch tau_y/2edot -> 0), and the solve config says what was stated."""
    s, v, p, cm = _box("a", 0.05)
    from underworld3.function.expressions import unwrap_expression
    eta = unwrap_expression(cm.viscosity, mode="symbolic_keep_constants")
    # substitute a huge strain rate and p = 0 by evaluating the unwrapped symbolic law
    L = s.Unknowns.L
    subs = {L[i, j]: (1.0e6 if (i, j) == (0, 1) else 0.0) for i in range(2) for j in range(2)}
    subs[p.sym[0]] = 0.0
    val = float(unwrap_expression(sympy.sympify(eta).subs(subs), mode="nondimensional"))
    assert abs(val - 0.05) < 2.0e-3, val            # eta_reg, with the soft-min's rounding
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            s.solve()
        except RuntimeWarning as w:
            assert "plastic_rate_strengthening" not in str(w), str(w)
    reg = s.solve_report.config["regularisation"]
    assert not str(reg["plastic_rate_strengthening"]).startswith("0 ") and str(reg["plastic_rate_strengthening"]) != "0"
    assert reg["rate_strengthening_scale"] == 1.0


def test_pressure_dependent_yield_without_the_term_warns():
    s, v, p, cm = _box("b", 0.0)
    with pytest.warns(RuntimeWarning, match="no stated plastic_rate_strengthening"):
        s.solve()
    assert str(s.solve_report.config["regularisation"]["plastic_rate_strengthening"]).startswith("0")


def test_the_ladder_ends_at_the_stated_problem():
    """solve(homotopy="rate_strengthening") finds the viscous limit from the fields,
    ladders m down to exactly 1, and the result is the stated problem's solution."""
    s, v, p, cm = _box("c", 0.05)
    report = s.solve(homotopy="rate_strengthening", homotopy_options=dict(verbose=False))
    assert report["reached_one"] and report["converged"], report
    assert report["rungs"][-1][0] == 1.0
    assert cm.rate_strengthening_scale == 1.0
    v_ladder = np.asarray(v.data).copy()
    s2, v2, p2, cm2 = _box("d", 0.05)
    s2.solve()
    assert s2.solve_report.converged
    dv = np.abs(v_ladder - np.asarray(v2.data)).max()
    assert dv < 1.0e-4 * np.abs(np.asarray(v2.data)).max(), dv
