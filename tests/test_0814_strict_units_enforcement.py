"""
Test strict units mode enforcement.

This tests the Phase 1 implementation of strict units mode, which enforces
the principle: "Variables with units require reference quantities"

When strict mode is enabled (opt-in), creating a variable with units but
no reference quantities raises a clear error. When disabled (default),
a warning is issued but the variable is created (backward compatible).
"""

import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import underworld3 as uw
import warnings


def test_strict_units_default_state():
    """Default is strict mode ON (enforces best practices from the start)."""
    # Reset to ensure clean state
    uw.reset_default_model()

    # Default is True (strict mode ON)
    # Units haven't been rolled out yet, so we enforce from the start
    assert uw.is_strict_units_active() == True


def test_strict_units_requires_reference_quantities():
    """Strict mode: variables with units require reference quantities."""
    # Start fresh (strict mode ON by default)
    uw.reset_default_model()

    # Verify strict mode is enabled by default
    assert uw.is_strict_units_active() == True

    # Create mesh WITHOUT reference quantities
    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Should raise - no reference quantities
    with pytest.raises(ValueError, match="Strict units mode.*reference quantities"):
        v = uw.discretisation.MeshVariable("v", mesh1, 2, units="m/s")

    # IMPORTANT: Set reference quantities BEFORE creating mesh
    # (model locks units after mesh creation)
    uw.reset_default_model()
    # Strict mode still ON by default after reset
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year")
    )

    # Create new mesh with reference quantities set
    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Now variable creation should succeed
    v = uw.discretisation.MeshVariable("v", mesh2, 2, units="m/s")

    # Verify proper scaling (not identity scaling)
    assert v.scaling_coefficient != 1.0


def test_non_strict_allows_units_without_reference():
    """Non-strict mode: variables with units allowed (with warning)."""
    # Start fresh and ensure strict mode is OFF
    uw.reset_default_model()
    uw.use_strict_units(False)

    # Verify non-strict mode
    assert uw.is_strict_units_active() == False

    # Create mesh
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Should work but warn
    with pytest.warns(UserWarning, match="no reference quantities"):
        v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

    # Variable should exist with identity scaling (half-way zone)
    assert v.scaling_coefficient == 1.0


def test_strict_units_toggle():
    """Can toggle strict units mode on and off."""
    uw.reset_default_model()

    # Start with strict ON (default)
    assert uw.is_strict_units_active() == True

    # Disable strict mode (expert/debugging use)
    uw.use_strict_units(False)
    assert uw.is_strict_units_active() == False

    # Re-enable strict mode
    uw.use_strict_units(True)
    assert uw.is_strict_units_active() == True


def test_strict_units_clear_error_message():
    """Error message provides clear, actionable guidance."""
    uw.reset_default_model()
    # Strict mode ON by default

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Capture the error
    with pytest.raises(ValueError) as exc_info:
        v = uw.discretisation.MeshVariable("test_var", mesh, 2, units="m/s")

    error_msg = str(exc_info.value)

    # Check that error message contains all required elements
    assert "Strict units mode" in error_msg
    assert "test_var" in error_msg
    assert "m/s" in error_msg
    assert "reference quantities" in error_msg

    # Check that error provides solutions
    assert "Options:" in error_msg
    assert "model.set_reference_quantities" in error_msg
    assert "uw.quantity" in error_msg
    assert "uw.use_strict_units(False)" in error_msg


def test_strict_units_variables_without_units_always_allowed():
    """Variables without units always work, regardless of strict mode."""
    uw.reset_default_model()

    # Test with strict mode ON (default)
    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v1 = uw.discretisation.MeshVariable("v1", mesh1, 2)  # No units
    assert v1 is not None

    # Test with strict mode OFF
    uw.reset_default_model()
    uw.use_strict_units(False)
    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v2 = uw.discretisation.MeshVariable("v2", mesh2, 2)  # No units
    assert v2 is not None


def test_strict_units_with_reference_quantities_works():
    """With reference quantities set, variables with units work in both modes."""
    # Test strict mode ON (default)
    uw.reset_default_model()

    model1 = uw.get_default_model()
    model1.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year")
    )

    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v1 = uw.discretisation.MeshVariable("v1", mesh1, 2, units="m/s")
    assert v1.scaling_coefficient != 1.0

    # Test strict mode OFF
    uw.reset_default_model()
    uw.use_strict_units(False)

    model2 = uw.get_default_model()
    model2.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year")
    )

    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v2 = uw.discretisation.MeshVariable("v2", mesh2, 2, units="m/s")
    assert v2.scaling_coefficient != 1.0


def test_strict_units_multiple_variables():
    """Strict mode applies to all variables consistently."""
    uw.reset_default_model()
    # Strict mode ON by default

    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # First variable should fail
    with pytest.raises(ValueError, match="Strict units mode"):
        v1 = uw.discretisation.MeshVariable("v1", mesh1, 2, units="m/s")

    # Second variable should also fail
    with pytest.raises(ValueError, match="Strict units mode"):
        T1 = uw.discretisation.MeshVariable("T1", mesh1, 1, units="K")

    # Set reference quantities BEFORE creating new mesh
    # Include temperature reference quantity for temperature variable
    uw.reset_default_model()
    # Strict mode still ON by default after reset
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
        temperature_diff=uw.quantity(1000, "K")  # Add temperature reference
    )

    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Now both types should work
    v2 = uw.discretisation.MeshVariable("v2", mesh2, 2, units="m/s")
    T2 = uw.discretisation.MeshVariable("T2", mesh2, 1, units="K")

    assert v2.scaling_coefficient != 1.0
    assert T2.scaling_coefficient != 1.0


def test_strict_units_preserves_backward_compatibility():
    """Existing code without strict mode continues to work."""
    uw.reset_default_model()
    # Explicitly disable strict mode to test backward compatibility
    uw.use_strict_units(False)

    # This is how existing code works - should still work with warning
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    with pytest.warns(UserWarning):
        v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

    # Variable exists but has identity scaling (half-way zone)
    # This is the backward-compatible behavior
    assert v.scaling_coefficient == 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
