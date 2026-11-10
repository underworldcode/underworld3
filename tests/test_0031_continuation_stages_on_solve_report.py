"""A continuation solve's stage breakdown is on its solve report (#778).

``consistent_jacobian = "continuation"`` is one SNES solve whose tangent is
blended from Picard to Newton by the residual. The question asked of a slow
or surprising one is whether the frozen stage found the basin or ran out, and
the report could not answer it: the iterations by tangent went only to the
private ``_continuation_stages``, and ``config["tangent"]["alpha_end"]`` read
0.0 because the solve puts alpha back to 0 after it ends, before the report
is taken.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    V = uw.discretisation.MeshVariable("V_cs", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_cs", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1 / (1 + 4 * stokes.Unknowns.Einv2)
    stokes.bodyforce = sympy.Matrix([0.0, -sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)])
    for b in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    for b in ("Left", "Right"):
        stokes.add_dirichlet_bc((0.0, sympy.oo), b)
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.tolerance = 1.0e-10
    return stokes


@pytest.mark.tier_a
def test_a_continuation_solve_reports_its_iterations_by_tangent_and_where_alpha_ended():
    stokes = _stokes()
    stokes.consistent_jacobian = "continuation"
    stokes.solve(zero_init_guess=True)
    report = stokes.solve_report
    assert report.converged, report
    tangent = report.config["tangent"]
    stages = tangent["stages"]
    assert stages["frozen_iterations"] >= 1
    assert (stages["frozen_iterations"] + stages["blended_iterations"]
            + stages["newton_iterations"]) >= 1
    assert stages["total_nl_iterations"] == report.nl_its
    # converged on the Newton end of the ramp, not where the solve left alpha after
    assert tangent["alpha_end"] == pytest.approx(stages["final_alpha"])
    assert tangent["alpha_end"] > 0.0
    assert float(stokes._get_newton_alpha().sym) == 0.0      # still put back for the next step


@pytest.mark.tier_a
def test_a_newton_solve_after_a_continuation_one_carries_no_stale_stages():
    stokes = _stokes()
    stokes.consistent_jacobian = "continuation"
    stokes.solve(zero_init_guess=True)
    stokes.consistent_jacobian = True
    stokes.solve(zero_init_guess=True)
    assert "stages" not in stokes.solve_report.config.get("tangent", {})
