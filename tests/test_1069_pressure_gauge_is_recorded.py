"""The pressure gauge a Stokes solve runs under is recorded, and a gauge-fixed pressure
feeding a pressure-dependent viscosity is warned about.

Why: with ``petsc_use_pressure_nullspace=True`` the projection imposes mean(p) = 0 on every
Krylov solve. A Drucker-Prager yield stress ``C + sin(phi) p`` then reads a gauge choice, not
a physical pressure, and the choice moves with the yielded region between iterations. On an
open domain (free surface, traction boundary) there is no nullspace and the flag must be
off; on a closed box the physical level has to be supplied separately. Either way the
transcript must say which it was, without a live process to introspect.
"""
import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _box(tag, closed, pressure_dependent, nullspace):
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("Vg" + tag, mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pg" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    if pressure_dependent:
        s.constitutive_model.Parameters.yield_stress = 0.3 + 0.3 * sympy.sqrt(p.sym[0] ** 2 + 1.0e-2)
        # a stated regularisation, so the only warning these tests can see is the gauge one
        s.constitutive_model.Parameters.plastic_rate_strengthening = 0.05
    else:
        s.constitutive_model.Parameters.yield_stress = 0.3
    s.add_essential_bc((1.0, 0.0), "Top")
    s.add_essential_bc((0.0, 0.0), "Bottom")
    if closed:
        s.add_essential_bc((0.0, sympy.oo), "Left")
        s.add_essential_bc((0.0, sympy.oo), "Right")
    s.petsc_use_pressure_nullspace = nullspace
    s.petsc_options.delValue("ksp_monitor")
    s.tolerance = 1.0e-5
    s.petsc_options["snes_max_it"] = 60
    return s


def test_gauge_facts_are_in_the_solve_config_open_domain():
    """Open sides: no nullspace attached, the rheology's pressure dependence is recorded."""
    s = _box("open", closed=False, pressure_dependent=True, nullspace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            s.solve()
        except RuntimeWarning as w:          # only the gauge warning is a failure here
            assert "GAUGE" not in str(w), str(w)
    g = s.solve_report.config["gauge"]
    assert g["pressure_nullspace_requested"] is False
    assert g["pressure_nullspace_attached"] is False
    assert g["pressure_in_rheology"] is True
    assert g["pressure_dirichlet_bcs"] == []
    assert s.solve_report.config["ksp"]["restart"] == 100


def test_closed_box_with_pressure_dependent_viscosity_and_nullspace_warns():
    """Closed box + nullspace + p-dependent yield: the projection fixes the gauge the yield
    law reads. Warned, and recorded as attached."""
    s = _box("closed", closed=True, pressure_dependent=True, nullspace=True)
    with pytest.warns(RuntimeWarning, match="GAUGE-fixed pressure"):
        s.solve()
    g = s.solve_report.config["gauge"]
    assert g["pressure_nullspace_requested"] is True
    assert g["pressure_nullspace_attached"] is True
    assert g["pressure_in_rheology"] is True


def test_closed_box_with_constant_yield_does_not_warn():
    s = _box("const", closed=True, pressure_dependent=False, nullspace=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s.solve()
    assert not [w for w in caught if "GAUGE" in str(w.message)]
    g = s.solve_report.config["gauge"]
    assert g["pressure_in_rheology"] is False
    assert g["pressure_nullspace_attached"] is True


def test_transcript_event_carries_the_gauge_and_the_tangent():
    """The step transcript's solve event records the gauge and tangent facts, and the gauge
    warning itself lands in the step as a warning."""
    s = _box("tr", closed=True, pressure_dependent=True, nullspace=True)
    model = uw.get_default_model()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with model.step(0.1, label="gauge step"):
            s.solve()
    entry = model.transcript[-1]
    solves = [e for e in entry.events if e["kind"] == "solve"]
    assert solves, [e["kind"] for e in entry.events]
    ev = solves[-1]
    assert ev["gauge"]["pressure_nullspace_attached"] is True
    assert ev["gauge"]["pressure_in_rheology"] is True
    assert "consistent_jacobian" in ev["tangent"]
    assert "newton_pressure_coupling" in ev["tangent"]
    recorded = [e for e in entry.events
                if e["kind"] == "warning" and "GAUGE" in str(e.get("message", ""))]
    assert recorded, "the gauge warning was not recorded in the step"


def test_krylov_restart_default_is_in_force_and_leaves_a_user_value_alone():
    """Restart 100 is set at construction for the outer KSP (every class) and the Stokes
    velocity split. It was first written inside the `tolerance` / `strategy` setters, so
    a Stokes solver kept FGMRES(30) unless `strategy` was assigned, and a `tolerance`
    assignment overwrote a user's own restart."""
    from petsc4py import PETSc
    opts = PETSc.Options()
    s = _box("rs", closed=False, pressure_dependent=False, nullspace=False)
    pfx = s.petsc_options_prefix
    assert opts.getInt(pfx + "ksp_gmres_restart") == 100
    assert opts.getInt(pfx + "fieldsplit_velocity_ksp_gmres_restart") == 100
    s.petsc_options["ksp_gmres_restart"] = 30
    s.tolerance = 1.0e-7
    assert opts.getInt(pfx + "ksp_gmres_restart") == 30
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    T = uw.discretisation.MeshVariable("Trs", mesh, 1, degree=1)
    pois = uw.systems.Poisson(mesh, u_Field=T)
    assert opts.getInt(pois.petsc_options_prefix + "ksp_gmres_restart") == 100
