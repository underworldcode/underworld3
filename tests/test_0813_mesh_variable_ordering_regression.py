"""
REGRESSION TEST: Creating mesh variables after solving ("Batman" pattern).

This test verifies that creating new MeshVariables on a mesh after solving
produces correct results in subsequent projections.

BUG STATUS: FIXED (2025-10-14)
- Creating variables AFTER solve: NOW WORKS
- Creating variables BEFORE solve: Works

FIX: When rebuilding the DM after adding new variables, properly invalidate
all existing variables' vectors and restore their data from the new DM.
(discretisation_mesh_variables.py, DM rebuild path)

These tests are deliberately UNITS-FREE (2026-07 audit, BF-10a): the DM
ordering/corruption behaviour they guard is independent of the units
system, and the previous unit-aware setup was blocked by a separate bug
(see the TODO(DESIGN) note on UnitAwareDerivativeMatrix in
utilities/mathematical_mixin.py). Part (b) of BF-10 re-unitizes them once
UnitAwareDerivativeMatrix arithmetic is implemented.
"""

import pytest

import underworld3 as uw
import numpy as np

# Projection-solver regression tests
pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# Domain and BCs: T = 300 at y=0, T = 1600 at y=500 on a 1000 x 500 box,
# so the exact solution is linear with dT/dy = 1300/500 = 2.6.
L_X = 1000.0
L_Y = 500.0
T_BOTTOM = 300.0
T_TOP = 1600.0
EXPECTED_GRADIENT = (T_TOP - T_BOTTOM) / L_Y  # 2.6


def _solve_poisson(mesh, T):
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(T_BOTTOM, "Bottom")
    poisson.add_dirichlet_bc(T_TOP, "Top")
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0, "Solver did not converge"


def _project_dTdy(mesh, T, gradT):
    x, y = mesh.X
    proj = uw.systems.Projection(mesh, gradT, degree=1)
    # NOTE: T.sym.diff(y), not T.diff(y) — the MathematicalMixin diff path
    # returns a UnitAwareDerivativeMatrix wrapper whose missing arithmetic
    # is exactly the BF-10 part (b) bug (see the TODO(DESIGN) note in
    # utilities/mathematical_mixin.py). The DM-ordering coverage here does
    # not depend on that wrapper.
    proj.uw_function = T.sym.diff(y)
    proj.solve()


def _dTdy_at_centre(gradT):
    centre = np.array([[L_X / 2, L_Y / 2]])
    return uw.function.evaluate(gradT.sym, centre)[0, 0, 0]


def test_kill_batman():
    """
    KILL BATMAN: Verify that variables can be created AFTER solve() without errors.

    This test explicitly checks that the Batman Pattern anti-pattern is NOT
    required. If this test fails, the DM state corruption bug has returned and
    you MUST NOT work around it by declaring variables upfront - FIX THE BUG
    instead.

    Batman Pattern = requiring all variables declared before any solve
    operations.
    """
    uw.reset_default_model()

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10), minCoords=(0.0, 0.0), maxCoords=(L_X, L_Y)
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    # Solve FIRST (this used to "finalize" the DM and prevent adding variables)
    _solve_poisson(mesh, T)

    # NOW create a derived variable AFTER solving (the test for Batman Pattern)
    gradT = uw.discretisation.MeshVariable("gradT", mesh, 1, degree=1)
    _project_dTdy(mesh, T, gradT)

    dT_dy = _dTdy_at_centre(gradT)
    assert abs(dT_dy - EXPECTED_GRADIENT) < 0.1, (
        "BATMAN ERRORS DETECTED: the DM state corruption bug has returned! "
        "Do NOT work around this by declaring variables upfront - fix the "
        "DM rebuild path in discretisation_mesh_variables.py. "
        f"Expected {EXPECTED_GRADIENT:.3f}, got {dT_dy:.3f}"
    )


def test_gradient_projection_variable_created_after_solve():
    """
    Test creating gradient variable AFTER Poisson solve.

    This previously failed due to DM state corruption
    (Expected: 2.6, Got: 6.09).
    """
    uw.reset_default_model()

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10), minCoords=(0.0, 0.0), maxCoords=(L_X, L_Y)
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    _solve_poisson(mesh, T)

    # Create gradient variable AFTER solving (THIS IS THE BUG's trigger)
    gradT = uw.discretisation.MeshVariable("gradT", mesh, 1, degree=1)
    _project_dTdy(mesh, T, gradT)

    dT_dy = _dTdy_at_centre(gradT)
    assert abs(dT_dy - EXPECTED_GRADIENT) < 0.1, (
        f"Gradient computation failed: expected {EXPECTED_GRADIENT:.3f}, got {dT_dy:.3f}"
    )


def test_gradient_projection_variable_created_before_solve():
    """
    Control case: creating the gradient variable BEFORE the Poisson solve
    has always worked and must keep working.
    """
    uw.reset_default_model()

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10), minCoords=(0.0, 0.0), maxCoords=(L_X, L_Y)
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    # Create gradient variable BEFORE solving
    gradT = uw.discretisation.MeshVariable("gradT", mesh, 1, degree=1)

    _solve_poisson(mesh, T)
    _project_dTdy(mesh, T, gradT)

    dT_dy = _dTdy_at_centre(gradT)
    assert abs(dT_dy - EXPECTED_GRADIENT) < 0.1, (
        f"Gradient computation failed: expected {EXPECTED_GRADIENT:.3f}, got {dT_dy:.3f}"
    )
