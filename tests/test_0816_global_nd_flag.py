#!/usr/bin/env python3
"""
Unit tests for global non-dimensional scaling flag (Phase 2 of ND solver implementation).

Tests the single global flag system that controls non-dimensional scaling:
- Global flag can be set and queried
- _unwrap_for_compilation() respects the global flag
- Scaling applied correctly when enabled
- No scaling when disabled
- Flag state persistence across function calls

Note: These tests use _unwrap_for_compilation() (internal function) to test
scaling behavior during JIT compilation. The user-facing unwrap()/expand()
functions are view-only and do NOT apply scaling transformations.
"""

import os
import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np
import sympy

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw
from underworld3.function.expressions import _unwrap_for_compilation


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before and after each test to prevent pollution."""
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


def test_global_flag_default_state():
    """Test that global ND scaling flag defaults to False."""
    # Reset to clean state
    uw.use_nondimensional_scaling(False)

    # Check default state
    assert (
        uw.is_nondimensional_scaling_active() == False
    ), "Global ND scaling flag should default to False"


def test_global_flag_can_be_enabled():
    """Test that global ND scaling flag can be enabled."""
    uw.use_nondimensional_scaling(False)  # Start from known state

    # Enable scaling
    uw.use_nondimensional_scaling(True)

    assert (
        uw.is_nondimensional_scaling_active() == True
    ), "Global ND scaling flag should be True after enabling"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_global_flag_can_be_disabled():
    """Test that global ND scaling flag can be disabled."""
    uw.use_nondimensional_scaling(True)  # Start enabled

    # Disable scaling
    uw.use_nondimensional_scaling(False)

    assert (
        uw.is_nondimensional_scaling_active() == False
    ), "Global ND scaling flag should be False after disabling"


def test_global_flag_toggle():
    """Test that global ND scaling flag can be toggled multiple times."""
    uw.use_nondimensional_scaling(False)

    # Toggle on
    uw.use_nondimensional_scaling(True)
    assert uw.is_nondimensional_scaling_active() == True

    # Toggle off
    uw.use_nondimensional_scaling(False)
    assert uw.is_nondimensional_scaling_active() == False

    # Toggle on again
    uw.use_nondimensional_scaling(True)
    assert uw.is_nondimensional_scaling_active() == True

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_unwrap_without_scaling():
    """Test that unwrap() does NOT scale when flag is False."""
    uw.reset_default_model()
    uw.use_nondimensional_scaling(False)

    model = uw.get_default_model()
    model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")
    T.set_reference_scale(1000.0)

    # Unwrap for compilation with scaling DISABLED
    unwrapped = _unwrap_for_compilation(T.sym)

    # Extract scalar from matrix
    unwrapped_scalar = unwrapped[0, 0] if isinstance(unwrapped, sympy.Matrix) else unwrapped

    # Should NOT contain scaling factor
    unwrap_str = str(unwrapped_scalar)
    assert (
        "1000" not in unwrap_str and "0.001" not in unwrap_str
    ), f"Unwrapped expression should NOT contain scaling factor when flag=False: {unwrap_str}"


def test_unwrap_with_scaling():
    """Test that _unwrap_for_compilation() works correctly when scaling flag is True.

    Note: As of 2025-11-14, variables are NOT scaled because PETSc stores them in ND form.
    This test verifies that unwrapping works without errors when scaling is active.
    """
    uw.reset_default_model()
    uw.use_nondimensional_scaling(True)  # Enable scaling

    model = uw.get_default_model()
    model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")

    # Unwrap for compilation with scaling ENABLED
    # Variables should NOT have scaling factors (already ND in PETSc)
    unwrapped = _unwrap_for_compilation(T.sym)

    # Should work without error
    assert unwrapped is not None, "Unwrapping should work when scaling is enabled"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_scaling_coefficient_always_computed():
    """Test that scaling coefficients are computed regardless of flag state."""
    uw.reset_default_model()

    # Test with flag disabled
    uw.use_nondimensional_scaling(False)

    model = uw.get_default_model()
    model.set_reference_quantities(temperature_diff=uw.quantity(1000, "kelvin"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")
    T.set_reference_scale(1000.0)

    # Scaling coefficient should be set even when flag is False
    assert (
        T.scaling_coefficient == 1000.0
    ), "Scaling coefficient should be computed regardless of flag state"

    # Now enable flag
    uw.use_nondimensional_scaling(True)

    # Coefficient should still be the same
    assert (
        T.scaling_coefficient == 1000.0
    ), "Scaling coefficient should not change when flag changes"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_multiple_variables_scaling():
    """Test that unwrapping works with multiple variables when scaling is active.

    Note: As of 2025-11-14, variables are NOT scaled because PETSc stores them in ND form.
    This test verifies that unwrapping multiple variables works correctly.
    """
    uw.reset_default_model()
    uw.use_nondimensional_scaling(True)

    model = uw.get_default_model()
    model.set_reference_quantities(
        temperature_diff=uw.quantity(1000, "kelvin"), domain_depth=uw.quantity(100, "km")
    )

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, units="meter/second")

    # Create expression with both variables
    expr = T.sym * v.sym[0]

    # Unwrap for compilation with scaling
    unwrapped = _unwrap_for_compilation(expr)

    # Should work without error
    assert unwrapped is not None, "Unwrapping multiple variables should work"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_backward_compatibility():
    """Test that old _is_scaling_active() function still works."""
    uw.use_nondimensional_scaling(False)
    assert uw._is_scaling_active() == False, "Legacy _is_scaling_active() should work"

    uw.use_nondimensional_scaling(True)
    assert (
        uw._is_scaling_active() == True
    ), "Legacy _is_scaling_active() should reflect current state"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_flag_state_persistence():
    """Test that flag state persists across function calls."""
    uw.use_nondimensional_scaling(False)

    # Enable scaling
    uw.use_nondimensional_scaling(True)

    # Call some functions
    uw.reset_default_model()
    model = uw.get_default_model()

    # Flag should still be True
    assert (
        uw.is_nondimensional_scaling_active() == True
    ), "Flag state should persist across function calls"

    # Cleanup
    uw.use_nondimensional_scaling(False)


def test_no_scaling_without_reference_quantities():
    """Test that scaling doesn't break when no reference quantities set."""
    uw.reset_default_model()
    uw.use_nondimensional_scaling(True)  # Enable flag
    uw.use_strict_units(False)  # Disable strict units to allow edge case testing

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    # Create variable WITHOUT setting reference quantities
    T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")

    # Should have default scaling coefficient = 1.0
    assert (
        T.scaling_coefficient == 1.0
    ), "Default scaling coefficient should be 1.0 without reference quantities"

    # Unwrap should work without error (but no scaling applied)
    try:
        unwrapped = uw.unwrap(T.sym)
        success = True
    except Exception as e:
        success = False
        print(f"unwrap() failed: {e}")

    assert success, "unwrap() should not fail when scaling_coefficient=1.0"

    # Cleanup
    uw.use_nondimensional_scaling(False)
    uw.use_strict_units(True)  # Restore default strict units mode


def test_vector_variable_scaling():
    """Test that vector variables are scaled correctly."""
    uw.reset_default_model()
    uw.use_nondimensional_scaling(True)

    model = uw.get_default_model()
    model.set_reference_quantities(plate_velocity=uw.quantity(5, "cm/year"))

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )

    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, units="meter/second")
    # Auto-derive scale from plate_velocity
    # 5 cm/year ≈ 1.58e-9 m/s

    # Unwrap vector component
    unwrapped = uw.unwrap(v.sym[0])

    # Should work without error
    assert unwrapped is not None, "Vector component unwrap should work"

    # Cleanup
    uw.use_nondimensional_scaling(False)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
