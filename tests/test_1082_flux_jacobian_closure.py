"""A constitutive model's tangent flux (``flux_jacobian``) is composed with the
solver's own closure, never substituted for the whole momentum flux.

The model supplies the deviatoric stress; the solver turns it into its momentum
flux (Stokes: less the mechanical pressure, with the augmented-Lagrangian
penalty, plus DEVSS; the thin sheet: the plane-stress trace term times the
thickness). The Newton tangent has to see the same closure applied to the
model's tangent flux. Contract used here: on a LINEAR problem, a tangent flux
equal to the flux leaves Newton exactly as fast as no tangent flux at all —
the tangent is exact either way. When the tangent flux replaced the whole
flux, the penalty, DEVSS and pressure terms (Stokes) fell out of the tangent
and the linear solve diverged; on a vector solver (the sheet) the tangent flux
was silently ignored altogether (a flattening error swallowed by a bare
except), so the companion check is that a WRONG tangent flux (twice the flux)
does slow Newton down: the hook is live.
"""

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _newton_iterations(build, tangent_scale=None):
    solver = build()
    solver.consistent_jacobian = True
    if tangent_scale is not None:
        solver.constitutive_model.flux_jacobian = (
            tangent_scale * sympy.Matrix(solver.constitutive_model.flux))
    solver.solve()
    return solver.snes.getIterationNumber(), solver


def _stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.penalty = 10
    stokes.bodyforce = sympy.Matrix([[0, sympy.sin(sympy.pi * x)]])
    for wall in ("Left", "Right", "Top", "Bottom"):
        stokes.add_essential_bc([0, 0], wall)
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["ksp_rtol"] = 1.0e-12
    return stokes


def _sheet():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    x, y = mesh.X
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    sheet.thickness = 2
    sheet.basal_drag = 1
    sheet.bodyforce = sympy.Matrix([[sympy.sin(sympy.pi * y), x * y]])
    sheet.add_essential_bc([0, 0], "Left")
    sheet.tolerance = 1.0e-8
    sheet.petsc_options["ksp_rtol"] = 1.0e-12
    return sheet


@pytest.mark.tier_b
@pytest.mark.parametrize("build", [_stokes, _sheet], ids=["stokes", "thin_sheet"])
def test_tangent_flux_keeps_the_solver_closure(build):
    exact, _ = _newton_iterations(build)
    composed, solver = _newton_iterations(build, tangent_scale=1)
    assert solver.snes.getConvergedReason() > 0
    assert composed == exact, (
        f"Newton took {composed} iterations with a tangent flux equal to the "
        f"flux, {exact} without one: the tangent lost the solver's closure")
    wrong, _ = _newton_iterations(build, tangent_scale=2)
    assert wrong > exact, (
        f"a tangent flux of twice the flux left Newton at {wrong} iterations "
        f"({exact} exact): the tangent flux is not being used")
