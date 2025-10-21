"""
Test Poisson solver with unit-aware boundary conditions.

This test replicates the exact workflow from Notebook 13 to ensure
unit-aware BCs produce correct results, not just that they are accepted.
"""

import pytest
import underworld3 as uw
import numpy as np


def test_poisson_linear_gradient_with_pint_quantities():
    """
    Test that Poisson solver with Pint Quantity BCs produces correct gradient.

    This replicates Notebook 13: solve ∇²T = 0 with linear BC,
    expecting constant gradient ∂T/∂y = ΔT/Ly.
    """
    uw.reset_default_model()
    # Physical parameters using Pint Quantities (like Notebook 13)
    L_x = 1000 * uw.units("m")
    L_y = 500 * uw.units("m")
    T_bottom = 300 * uw.units("K")
    T_top = 1600 * uw.units("K")
    Delta_T = T_top - T_bottom

    # Expected gradient
    expected_gradient = (Delta_T / L_y).magnitude  # Should be 2.6 K/m

    # Create mesh - USING STRUCTURED MESH (Unstructured has gradient projection bugs)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10),
        minCoords=(0.0, 0.0),
        maxCoords=(L_x, L_y),
        units="metre"
    )

    # Create temperature variable
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # CRITICAL: Create gradient variable BEFORE solving
    # (creating it after causes mesh state corruption)
    gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

    # Set up Poisson solver
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0

    # Apply BCs with Pint Quantities (THE KEY TEST)
    poisson.add_dirichlet_bc(T_bottom, "Bottom")
    poisson.add_dirichlet_bc(T_top, "Top")

    # Solve
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0, "Solver did not converge"

    # Compute gradient via projection (variable already created)
    x, y = mesh.X
    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T.sym)
    gradient_proj.solve()

    # Evaluate at center
    x_center = L_x / 2
    y_center = L_y / 2
    grad = uw.function.evaluate(gradT.sym, [(x_center, y_center)])

    dT_dx = grad[0, 0, 0]
    dT_dy = grad[0, 0, 1]

    print(f"\nPoisson with Pint Quantity BCs:")
    print(f"  Expected gradient: {expected_gradient:.3f} K/m")
    print(f"  Computed ∂T/∂x: {dT_dx:.6f} K/m (expected: 0)")
    print(f"  Computed ∂T/∂y: {dT_dy:.6f} K/m (expected: {expected_gradient:.3f})")

    # Check results
    assert abs(dT_dx) < 0.1, f"∂T/∂x should be ~0, got {dT_dx}"
    assert abs(dT_dy - expected_gradient) < 0.1, \
        f"∂T/∂y should be {expected_gradient:.3f}, got {dT_dy:.3f}"


def test_poisson_linear_gradient_with_uwquantity():
    """
    Same test but using uw.quantity() instead of uw.units().
    """
    uw.reset_default_model()
    # Physical parameters using UWQuantity
    L_x = uw.quantity(1000, "m")
    L_y = uw.quantity(500, "m")
    T_bottom = uw.quantity(300, "K")
    T_top = uw.quantity(1600, "K")
    Delta_T = T_top - T_bottom

    # Expected gradient
    expected_gradient = (Delta_T / L_y).value

    # Create mesh - USING STRUCTURED MESH (Unstructured has gradient projection bugs)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10),
        minCoords=(0.0, 0.0),
        maxCoords=(L_x, L_y),
        units="metre"
    )

    # Create temperature variable
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # CRITICAL: Create gradient variable BEFORE solving
    gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

    # Set up Poisson solver
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0

    # Apply BCs with UWQuantity
    poisson.add_dirichlet_bc(T_bottom, "Bottom")
    poisson.add_dirichlet_bc(T_top, "Top")

    # Solve
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0

    # Compute gradient (variable already created)
    x, y = mesh.X
    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T.sym)
    gradient_proj.solve()

    # Evaluate at center
    x_center = L_x / 2
    y_center = L_y / 2
    grad = uw.function.evaluate(gradT.sym, [(x_center, y_center)])

    dT_dx = grad[0, 0, 0]
    dT_dy = grad[0, 0, 1]

    print(f"\nPoisson with UWQuantity BCs:")
    print(f"  Expected gradient: {expected_gradient:.3f} K/m")
    print(f"  Computed ∂T/∂x: {dT_dx:.6f} K/m")
    print(f"  Computed ∂T/∂y: {dT_dy:.6f} K/m")

    # Check results
    assert abs(dT_dx) < 0.1, f"∂T/∂x should be ~0, got {dT_dx}"
    assert abs(dT_dy - expected_gradient) < 0.1, \
        f"∂T/∂y should be {expected_gradient:.3f}, got {dT_dy:.3f}"


def test_poisson_check_bc_values():
    """
    Verify that the BC values are actually being set correctly in the solution.
    """
    uw.reset_default_model()
    L_x = 1000 * uw.units("m")
    L_y = 500 * uw.units("m")
    T_bottom = 300 * uw.units("K")
    T_top = 1600 * uw.units("K")

    # Create mesh - USING STRUCTURED MESH (Unstructured has gradient projection bugs)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10),
        minCoords=(0.0, 0.0),
        maxCoords=(L_x, L_y),
        units="metre"
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0

    poisson.add_dirichlet_bc(T_bottom, "Bottom")
    poisson.add_dirichlet_bc(T_top, "Top")
    poisson.solve()

    # Check boundary values
    x, y = mesh.X

    # Evaluate at bottom boundary
    bottom_points = [(L_x / 2, uw.quantity(0, "m"))]
    T_at_bottom = uw.function.evaluate(T.sym, bottom_points)

    # Evaluate at top boundary
    top_points = [(L_x / 2, L_y)]
    T_at_top = uw.function.evaluate(T.sym, top_points)

    print(f"\nBoundary value check:")
    print(f"  T at bottom: {T_at_bottom[0,0,0]:.1f} K (expected: 300)")
    print(f"  T at top: {T_at_top[0,0,0]:.1f} K (expected: 1600)")

    assert abs(T_at_bottom[0,0,0] - 300) < 1.0, \
        f"Bottom BC not applied correctly: {T_at_bottom[0,0,0]} != 300"
    assert abs(T_at_top[0,0,0] - 1600) < 1.0, \
        f"Top BC not applied correctly: {T_at_top[0,0,0]} != 1600"


if __name__ == "__main__":
    print("="*70)
    print("Testing Poisson solver with unit-aware BCs (Notebook 13 replication)")
    print("="*70)

    print("\n1. Testing with Pint Quantities (uw.units())...")
    test_poisson_linear_gradient_with_pint_quantities()
    print("   ✓ Passed")

    print("\n2. Testing with UWQuantity (uw.quantity())...")
    test_poisson_linear_gradient_with_uwquantity()
    print("   ✓ Passed")

    print("\n3. Testing BC value application...")
    test_poisson_check_bc_values()
    print("   ✓ Passed")

    print("\n" + "="*70)
    print("All Poisson unit tests passed! ✅")
    print("="*70)
