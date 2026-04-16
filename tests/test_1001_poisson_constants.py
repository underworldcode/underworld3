# Tests for the PetscDS constants array mechanism.
#
# These tests verify that UWexpression parameters routed through
# PETSc's constants[] array work correctly:
#   1. Solver produces correct results with constants
#   2. Changing a constant value and re-solving WITHOUT _force_setup
#      gives correct results for the new value
#   3. No JIT recompilation occurs when only constant values change

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities._jitextension import _ext_dict

# Solver-level tests
pytestmark = pytest.mark.level_1


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test."""
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


def test_poisson_constant_diffusivity_expression():
    """Poisson with UWexpression diffusivity solves correctly."""

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)
    x, y = mesh.X

    u = uw.discretisation.MeshVariable("u_cK", mesh, 1, degree=2)

    K = uw.expression("K_diff", 1.0)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.f = 0.0

    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    # Check linear profile u(y) = 1 - y
    sample_y = np.linspace(0.05, 0.95, 10)
    sample_x = np.full_like(sample_y, 0.5)
    sample_points = np.column_stack([sample_x, sample_y])

    u_num = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_exact = 1 - sample_y

    error = np.sqrt(np.mean((u_num - u_exact) ** 2))
    assert error < 1e-3, f"K=1 linear profile error {error:.3e} too large"

    del poisson


def test_poisson_change_constant_no_recompile():
    """Change a constant UWexpression value and re-solve without recompilation.

    This is the key test for the constants[] mechanism:
    - Solve with K=1, source f=-2 → u(y) = y^2
    - Change K to 2 (but same structural expression)
    - Re-solve WITHOUT _force_setup
    - Verify: correct result for K=2, and no new JIT module compiled
    """

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.15)
    x, y = mesh.X

    u = uw.discretisation.MeshVariable("u_recomp", mesh, 1, degree=2)

    K = uw.expression("K_recomp", 1.0)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = K
    # Source term: f = -2 with K=1 and BCs u(0)=0, u(1)=1
    # gives u(y) = y^2  (since -K * u'' = f → -1 * 2 = -2 ✓, u(0)=0, u(1)=1)
    poisson.f = -2.0

    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")

    # --- First solve with K=1 ---
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0

    sample_y = np.linspace(0.05, 0.95, 15)
    sample_x = np.full_like(sample_y, 0.5)
    sample_points = np.column_stack([sample_x, sample_y])

    u_num_1 = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_exact_1 = sample_y ** 2
    error_1 = np.sqrt(np.mean((u_num_1 - u_exact_1) ** 2))
    assert error_1 < 5e-3, f"K=1 solve error {error_1:.3e} too large"

    # Record JIT cache size
    n_modules_before = len(_ext_dict)

    # --- Change K to 2 and re-solve ---
    # With K=2 and f=-2: -K*u'' = f → -2*u'' = -2 → u'' = 1 → u(y) = y²/2 + Ay + B
    # BCs: u(0)=0 → B=0, u(1)=1 → 1/2 + A = 1 → A = 1/2
    # So u(y) = y²/2 + y/2 = y(y+1)/2
    K.sym = 2.0

    # Re-solve — should use _update_constants(), NOT recompile
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0

    n_modules_after = len(_ext_dict)

    u_num_2 = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_exact_2 = sample_y * (sample_y + 1) / 2
    error_2 = np.sqrt(np.mean((u_num_2 - u_exact_2) ** 2))
    assert error_2 < 5e-3, f"K=2 solve error {error_2:.3e} too large"

    # Verify no new JIT compilation occurred
    assert n_modules_after == n_modules_before, (
        f"JIT recompilation detected: {n_modules_before} → {n_modules_after} modules. "
        f"Constants mechanism should have avoided recompilation."
    )

    # Verify the two solutions are actually different
    diff = np.max(np.abs(u_num_2 - u_num_1))
    assert diff > 0.01, (
        f"Solutions with K=1 and K=2 are suspiciously similar (max diff={diff:.3e}). "
        f"Constants update may not be working."
    )

    del poisson


def test_poisson_constant_source_expression():
    """Poisson with UWexpression source term routed through constants[]."""

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)
    x, y = mesh.X

    u = uw.discretisation.MeshVariable("u_cS", mesh, 1, degree=2)

    S = uw.expression("S_source", 1.0)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    # -u'' = S with u(0)=0, u(1)=0 → u(y) = S/2 * y * (1-y)
    poisson.f = S

    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    sample_y = np.linspace(0.05, 0.95, 15)
    sample_x = np.full_like(sample_y, 0.5)
    sample_points = np.column_stack([sample_x, sample_y])

    u_num = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_exact = 0.5 * sample_y * (1 - sample_y)

    error = np.sqrt(np.mean((u_num - u_exact) ** 2))
    assert error < 5e-3, f"Constant source error {error:.3e} too large"

    del poisson


def test_stokes_constant_viscosity_expression():
    """Stokes with UWexpression viscosity routed through constants[]."""

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)

    v = uw.discretisation.MeshVariable("v_cV", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p_cV", mesh, 1, degree=1, continuous=True)

    eta = uw.expression("eta_const", 1.0)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.bodyforce = sympy.Matrix([0, 0])

    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")

    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0

    # Simple shear: v_x should be linear in y (from 0 at bottom to 1 at top)
    sample_y = np.linspace(0.05, 0.95, 10)
    sample_x = np.full_like(sample_y, 0.5)
    sample_points = np.column_stack([sample_x, sample_y])

    vx_num = uw.function.evaluate(v.sym[0], sample_points, rbf=False).squeeze()
    vx_exact = sample_y

    error = np.sqrt(np.mean((vx_num - vx_exact) ** 2))
    assert error < 1e-2, f"Stokes simple shear error {error:.3e} too large"

    del stokes


def test_poisson_constant_in_essential_bc():
    """Essential BC with UWexpression amplitude — change without recompile."""

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)
    x, y = mesh.X

    u = uw.discretisation.MeshVariable("u_ebc", mesh, 1, degree=2)

    A = uw.expression("A_bc", 1.0)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0

    # u = A*sin(pi*x) on top, u = 0 on bottom
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(sympy.Matrix([A * sympy.sin(sympy.pi * x)]), "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    # Check BC value at top
    pts_top = np.array([[0.5, 1.0]])
    u_top_1 = uw.function.evaluate(u.sym[0], pts_top, rbf=False).squeeze()
    assert abs(u_top_1 - 1.0) < 1e-3, f"A=1 BC at top: {u_top_1:.4f} != 1.0"

    # Change A, re-solve without recompilation
    n_before = len(_ext_dict)
    A.sym = 3.0
    poisson.solve()
    n_after = len(_ext_dict)

    assert poisson.snes.getConvergedReason() > 0
    assert n_after == n_before, (
        f"Recompiled when changing BC constant: {n_before} → {n_after}"
    )

    u_top_2 = uw.function.evaluate(u.sym[0], pts_top, rbf=False).squeeze()
    assert abs(u_top_2 - 3.0) < 1e-3, f"A=3 BC at top: {u_top_2:.4f} != 3.0"

    del poisson


def test_stokes_natural_bc_constant_no_recompile():
    """Stokes natural BC with constant traction — change without recompile.

    Also verifies the Null_Boundary bug fix: _build() must not re-add
    Null_Boundary on every call, which would reset is_setup and force
    recompilation.
    """

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)

    v = uw.discretisation.MeshVariable("v_nbc", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p_nbc", mesh, 1, degree=1, continuous=True)

    tau = uw.expression("tau_nbc", 1.0)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, 0])

    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.add_natural_bc((tau, 0.0), "Top")

    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0

    # tau should be in the constants manifest
    const_names = [expr.name for _, expr in stokes.constants_manifest]
    assert "tau_nbc" in const_names, (
        f"tau_nbc not found in constants manifest: {const_names}"
    )

    # Null_Boundary should appear exactly once
    null_count = sum(1 for bc in stokes.natural_bcs if bc.boundary == "Null_Boundary")
    assert null_count == 1, f"Null_Boundary appears {null_count} times (expected 1)"

    # Change traction and re-solve — no recompilation
    n_before = len(_ext_dict)
    tau.sym = 5.0
    stokes.solve()
    n_after = len(_ext_dict)

    assert stokes.snes.getConvergedReason() > 0
    assert n_after == n_before, (
        f"Recompiled when changing natural BC constant: {n_before} → {n_after}. "
        f"Null_Boundary count: {sum(1 for bc in stokes.natural_bcs if bc.boundary == 'Null_Boundary')}"
    )

    # Null_Boundary should still be exactly one
    null_count_2 = sum(1 for bc in stokes.natural_bcs if bc.boundary == "Null_Boundary")
    assert null_count_2 == 1, f"Null_Boundary duplicated to {null_count_2} after second solve"

    del stokes
