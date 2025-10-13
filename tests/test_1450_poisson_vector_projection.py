#!/usr/bin/env python3
"""
Regression test for Vector_Projection after Poisson solve.

This test specifically validates the fix for the DM field synchronization bug where
Vector_Projection would fail after a Poisson solve with "Invalid field number" error.

Bug: When a variable was added AFTER a solver ran, the Vector_Projection solver would
clone a stale DM that didn't know about the new variable's field.

Fix: Changed SNES_Vector to use mesh.dm.getNumFields() instead of self.dm.getNumFields()
to get the current field count from the live mesh DM, not the cloned DM.
"""

import pytest
import underworld3 as uw
import numpy as np


def test_vector_projection_after_poisson():
    """
    Test that Vector_Projection works after solving Poisson equation.

    This was failing with PETSc error: "Invalid field number 1; not in [0, 1)"
    because the Vector_Projection solver was using a stale DM clone.
    """
    # Create mesh
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 0.5),
    )

    # Create temperature variable
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    # Set up and solve Poisson equation
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")

    poisson.solve()

    # Now create gradient variable (AFTER Poisson solve)
    gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

    # This should work now (was failing before the fix)
    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T.sym)
    gradient_proj.solve()

    # Verify gradient was computed
    assert gradT.data.shape == (gradT.coords.shape[0], mesh.dim)

    # Check that gradient is non-zero (temperature gradient exists)
    grad_magnitude = np.linalg.norm(gradT.data, axis=1)
    assert np.max(grad_magnitude) > 0, "Gradient should be non-zero"


def test_vector_projection_after_poisson_with_units():
    """
    Test that Vector_Projection works with unit-aware meshes after Poisson solve.

    This is the exact sequence from the tutorial notebook that was failing.
    """
    # Create mesh with units (like in tutorial)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16),
        minCoords=(0.0, 0.0),
        maxCoords=(1000.0, 500.0),
        units="meter"
    )

    # Create temperature variable with units
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # Set up and solve Poisson equation
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(300.0, "Bottom")
    poisson.add_dirichlet_bc(1600.0, "Top")

    poisson.solve()

    # Create gradient variable AFTER Poisson solve
    gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

    # This should work now
    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T.sym)
    gradient_proj.solve()

    # Verify gradient was computed
    assert gradT.data.shape == (gradT.coords.shape[0], mesh.dim)

    # Check that gradient is non-zero
    grad_magnitude = np.linalg.norm(gradT.data, axis=1)
    assert np.max(grad_magnitude) > 0, "Gradient should be non-zero"


def test_multiple_solvers_sequence():
    """
    Test that multiple solvers can be used in sequence.

    Validates: Poisson → Poisson → Vector_Projection all work correctly.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 0.5),
    )

    # First Poisson solve
    T1 = uw.discretisation.MeshVariable("T1", mesh, 1, degree=2)
    poisson1 = uw.systems.Poisson(mesh, u_Field=T1)
    poisson1.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson1.constitutive_model.Parameters.diffusivity = 1
    poisson1.f = 0.0
    poisson1.add_dirichlet_bc(0.0, "Bottom")
    poisson1.add_dirichlet_bc(1.0, "Top")
    poisson1.solve()

    # Second Poisson solve (this always worked)
    T2 = uw.discretisation.MeshVariable("T2", mesh, 1, degree=2)
    poisson2 = uw.systems.Poisson(mesh, u_Field=T2)
    poisson2.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson2.constitutive_model.Parameters.diffusivity = 1
    poisson2.f = 0.0
    poisson2.add_dirichlet_bc(0.5, "Bottom")
    poisson2.add_dirichlet_bc(1.5, "Top")
    poisson2.solve()

    # Now Vector_Projection (this was failing)
    gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)
    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T1.sym)
    gradient_proj.solve()

    # All solvers should have completed successfully
    assert T1.data is not None
    assert T2.data is not None
    assert gradT.data is not None


if __name__ == "__main__":
    test_vector_projection_after_poisson()
    print("✓ test_vector_projection_after_poisson passed")

    test_vector_projection_after_poisson_with_units()
    print("✓ test_vector_projection_after_poisson_with_units passed")

    test_multiple_solvers_sequence()
    print("✓ test_multiple_solvers_sequence passed")

    print("\n✓ All tests passed!")
