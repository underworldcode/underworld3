"""Opt-in rotated KSP settings preserve exact hydrostatic balance and defaults."""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

pytestmark = pytest.mark.level_1

_BLOCK_OPTIONS = {
    "rotated_fieldsplit_velocity_ksp_type": "preonly",
    "rotated_fieldsplit_pressure_ksp_type": "preonly",
    "rotated_ksp_max_it": 500,
}


def _hydrostatic_stokes():
    """u=0, p=x-1/2 exactly balances f=(1,0) in the unit square."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3
    )
    velocity = uw.discretisation.MeshVariable("Uoptions", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("Poptions", mesh, 1, degree=1)
    load = uw.discretisation.MeshVariable("LoadOptions", mesh, 1, degree=1)
    load.array[:, 0, 0] = 1
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sp.Integer(1)
    stokes.saddle_preconditioner = sp.Integer(1)
    stokes.bodyforce = sp.Matrix([load.sym[0], 0])
    stokes.tolerance = 1e-9
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options["snes_type"] = "ksponly"
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_rotated_freeslip_bc(0, wall)
    return stokes, load


def _assert_hydrostatic_balance(stokes, amplitude: int):
    """Check against exact pressure and zero velocity, independently of KSP type."""
    mesh = stokes.mesh
    pressure = stokes.p.sym[0]
    mean = uw.maths.Integral(mesh, pressure).evaluate()
    error = pressure - mean - amplitude * (mesh.X[0] - sp.Rational(1, 2))
    p_error = np.sqrt(12 * uw.maths.Integral(mesh, error**2).evaluate()) / amplitude
    u_error = np.sqrt(uw.maths.Integral(mesh, stokes.u.sym.dot(stokes.u.sym)).evaluate())
    assert p_error < 1e-7, f"hydrostatic pressure relative L2 error: {p_error}"
    assert u_error / amplitude < 1e-8, f"nonzero velocity under exactly balanced forcing: {u_error}"
    assert stokes._rotated_freeslip_info["converged"]
    assert stokes._rotated_freeslip_info["ksp_reason"] > 0


@pytest.mark.tier_b
@pytest.mark.parametrize("overrides", [{}, {"rotated_ksp_max_it": 500}, _BLOCK_OPTIONS])
def test_rotated_solver_honours_only_explicit_overrides(overrides):
    """The opt-in keys configure the actual KSP; absent keys retain native defaults."""
    stokes, _load = _hydrostatic_stokes()
    for key, value in overrides.items():
        stokes.petsc_options[key] = value
    stokes.solve()
    _assert_hydrostatic_balance(stokes, 1)
    ksp = stokes._rotated_linear_cache["ctx"]["ksp"]
    velocity, pressure = ksp.getPC().getFieldSplitSubKSP()
    assert ksp.getType() == "fgmres"
    assert ksp.getTolerances()[0] == pytest.approx(1e-9)
    assert ksp.getTolerances()[3] == overrides.get("rotated_ksp_max_it", 300)
    assert velocity.getType() == overrides.get("rotated_fieldsplit_velocity_ksp_type", "fgmres")
    assert pressure.getType() == overrides.get("rotated_fieldsplit_pressure_ksp_type", "fgmres")
    info = stokes._rotated_freeslip_info
    assert info["velocity_ksp_type"] == velocity.getType()
    assert info["pressure_ksp_type"] == pressure.getType()
    assert info["ksp_max_it"] == ksp.getTolerances()[3]
    for inner, tolerance in ((velocity, 3.3e-11), (pressure, 1e-10)):
        if inner.getType() == "fgmres":
            assert inner.getTolerances()[0] == pytest.approx(tolerance)
            assert inner.getTolerances()[3] == 200
        else:
            # PREONLY has no iterative tolerance: its contract is one PC apply.
            assert inner.getIterationNumber() == 1


@pytest.mark.tier_b
def test_rotated_cached_solver_refreshes_when_overrides_change():
    """Removing an opt-in restores defaults; unchanged options reuse the workspace."""
    stokes, load = _hydrostatic_stokes()
    for key, value in _BLOCK_OPTIONS.items():
        stokes.petsc_options[key] = value
    stokes.solve()
    _assert_hydrostatic_balance(stokes, 1)
    initial_context = stokes._rotated_linear_cache["ctx"]
    load.array[:, 0, 0] = 2
    stokes.solve()
    _assert_hydrostatic_balance(stokes, 2)
    assert stokes._rotated_linear_cache["ctx"] is initial_context
    for key in _BLOCK_OPTIONS:
        stokes.petsc_options.delValue(key)
    load.array[:, 0, 0] = 3
    stokes.solve()
    _assert_hydrostatic_balance(stokes, 3)
    context = stokes._rotated_linear_cache["ctx"]
    assert context is not initial_context
    velocity, pressure = context["ksp"].getPC().getFieldSplitSubKSP()
    assert velocity.getType() == pressure.getType() == "fgmres"
    assert context["ksp"].getTolerances()[3] == 300
