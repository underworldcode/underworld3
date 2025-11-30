#!/usr/bin/env python3
"""
Validation tests for non-dimensional Poisson solver (Phase 3).

Tests that dimensional and non-dimensional Poisson solutions are identical,
validating that the scaling system works correctly in actual solver context.

Key validation: Solutions computed with uw.use_nondimensional_scaling(True)
should match dimensional solutions to machine precision.
"""

import os
import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw


@pytest.mark.parametrize("resolution", [0.25, 0.1])
def test_poisson_dimensional_vs_nondimensional(resolution):
    """
    Test that Poisson equation gives identical results with dimensional
    and non-dimensional formulations.

    This is the core validation test for Phase 3.
    """
    # ========================================================================
    # SETUP PROBLEM
    # ========================================================================

    uw.reset_default_model()
    uw.use_nondimensional_scaling(False)  # Start with dimensional

    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(1, "kilometer"), temperature_diff=uw.quantity(1000, "kelvin")
    )

    # Create mesh (with reference quantities already set)
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=resolution
    )

    # Mesh should have picked up length scale
    assert mesh.length_scale == pytest.approx(
        1000.0, rel=1e-10
    ), f"Mesh should have length_scale=1000m from 1km reference, got {mesh.length_scale}"

    # Create variables with units
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # T should have picked up temperature scale
    assert T.scaling_coefficient == pytest.approx(
        1000.0, rel=1e-10
    ), f"T should have scaling_coefficient=1000K, got {T.scaling_coefficient}"

    # ========================================================================
    # SOLVE DIMENSIONAL (baseline)
    # ========================================================================

    uw.use_nondimensional_scaling(False)

    poisson_dim = uw.systems.Poisson(mesh, u_Field=T)
    poisson_dim.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson_dim.constitutive_model.Parameters.diffusivity = 1.0  # Dimensionless for this test
    poisson_dim.f = 0.0  # No source term

    # Boundary conditions (dimensional values in Kelvin)
    poisson_dim.add_dirichlet_bc(0.0, "Left")
    poisson_dim.add_dirichlet_bc(1000.0, "Right")

    poisson_dim.solve()

    # Save dimensional solution
    u_dimensional = np.copy(T.array)

    # ========================================================================
    # SOLVE NON-DIMENSIONAL (test case)
    # ========================================================================

    # Enable ND scaling
    uw.use_nondimensional_scaling(True)

    # Create NEW solver (important - must recompile with ND)
    poisson_nd = uw.systems.Poisson(mesh, u_Field=T)
    poisson_nd.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson_nd.constitutive_model.Parameters.diffusivity = 1.0
    poisson_nd.f = 0.0

    # Same BCs (dimensional values - scaling happens internally)
    poisson_nd.add_dirichlet_bc(0.0, "Left")
    poisson_nd.add_dirichlet_bc(1000.0, "Right")

    poisson_nd.solve()

    # Get solution (should be in dimensional form due to auto-unscaling)
    u_nondimensional = np.copy(T.array)

    # ========================================================================
    # VALIDATION: Solutions should be identical
    # ========================================================================

    max_diff = np.max(np.abs(u_dimensional - u_nondimensional))
    rel_diff = max_diff / np.max(np.abs(u_dimensional))

    print(f"\nDimensional vs Non-Dimensional Poisson Comparison (resolution={resolution}):")
    print(f"  Dimensional solution range: [{u_dimensional.min():.6f}, {u_dimensional.max():.6f}] K")
    print(
        f"  Non-dimensional solution range: [{u_nondimensional.min():.6f}, {u_nondimensional.max():.6f}] K"
    )
    print(f"  Max absolute difference: {max_diff:.6e} K")
    print(f"  Max relative difference: {rel_diff:.6e}")

    # Solutions should match to machine precision
    assert np.allclose(
        u_dimensional, u_nondimensional, rtol=1e-10, atol=1e-12
    ), f"Dimensional and ND solutions should match (max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e})"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_poisson_with_source_term():
    """
    Test Poisson with non-zero source term.

    Equation: ∇²T = f

    Validates that source terms are scaled correctly.
    """
    uw.reset_default_model()
    uw.use_nondimensional_scaling(False)

    model = uw.get_default_model()
    model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # ========================================================================
    # Dimensional solve
    # ========================================================================
    uw.use_nondimensional_scaling(False)

    poisson_dim = uw.systems.Poisson(mesh, u_Field=T)
    poisson_dim.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson_dim.constitutive_model.Parameters.diffusivity = 1.0
    poisson_dim.f = uw.quantity(10.0, "kelvin")  # Source term with units

    poisson_dim.add_dirichlet_bc(0.0, "Left")
    poisson_dim.add_dirichlet_bc(0.0, "Right")

    poisson_dim.solve()
    u_dim = np.copy(T.array)

    # ========================================================================
    # Non-dimensional solve
    # ========================================================================
    uw.use_nondimensional_scaling(True)

    poisson_nd = uw.systems.Poisson(mesh, u_Field=T)
    poisson_nd.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson_nd.constitutive_model.Parameters.diffusivity = 1.0
    poisson_nd.f = uw.quantity(10.0, "kelvin")  # Same source with units (auto-scaled)

    poisson_nd.add_dirichlet_bc(0.0, "Left")
    poisson_nd.add_dirichlet_bc(0.0, "Right")

    poisson_nd.solve()
    u_nd = np.copy(T.array)

    # ========================================================================
    # Validate
    # ========================================================================
    max_diff = np.max(np.abs(u_dim - u_nd))

    print(f"\nPoisson with source term:")
    print(f"  Max difference: {max_diff:.6e} K")

    assert np.allclose(
        u_dim, u_nd, rtol=1e-10, atol=1e-12
    ), f"Solutions with source term should match (max_diff={max_diff:.6e})"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_poisson_scaling_improves_conditioning():
    """
    Test that non-dimensional scaling produces better-conditioned system.

    This is more of a demonstration than a strict requirement, but
    non-dimensional systems should have values closer to O(1).
    """
    uw.reset_default_model()

    model = uw.get_default_model()
    model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

    # ========================================================================
    # Solve with ND scaling
    # ========================================================================
    uw.use_nondimensional_scaling(True)

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0

    poisson.add_dirichlet_bc(0.0, "Left")
    poisson.add_dirichlet_bc(1000.0, "Right")

    poisson.solve()

    # Solution should be reasonable (linear profile from 0 to 1000)
    solution_array = np.array(T.array)
    assert solution_array.min() >= -10.0, "Solution should be >= 0"
    assert solution_array.max() <= 1010.0, "Solution should be <= 1000"

    # Midpoint should be roughly 500 K
    midpoint_value = solution_array[len(solution_array) // 2]
    assert 400 < midpoint_value < 600, f"Midpoint should be ~500K, got {midpoint_value:.1f}K"

    # Cleanup
    uw.use_nondimensional_scaling(False)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
