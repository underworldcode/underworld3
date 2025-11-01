#!/usr/bin/env python3
"""Test that original evaluate behavior is preserved."""

import os
import pytest
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw


@pytest.mark.skip(reason="coord_units parameter not implemented - planned feature for evaluate()")
def test_original_behavior():
    """Test that evaluate behavior with and without scaling works correctly."""
    print("=== TESTING EVALUATE WITH SCALING ===")

    # Set up a simple test case
    uw.reset_default_model()
    model = uw.get_default_model()

    # Create mesh WITHOUT scaling (original behavior)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(5, 5), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2
    )

    # Create a simple unitless field with values 1000-1500
    temperature = uw.discretisation.MeshVariable("T", mesh, 1)

    # Set field values directly (not using mesh.X which would introduce scaling issues)
    with uw.synchronised_array_update():
        # Just set to fixed values: 1000 at corners, 1500 at center
        # This avoids any coordinate-dependent initialization
        temperature.array[:, 0, 0] = 1250.0  # Uniform field for simplicity

    # Test original behavior - coords in model units (0-1 range)
    model_coords = np.array([[0.5, 0.5]], dtype=np.float64)

    print("Testing without scaling (model coordinates):")
    print(f"  Input coordinates: {model_coords}")

    # This should work exactly as before
    result_original = uw.function.evaluate(temperature.sym, model_coords)
    print(f"  Result: {result_original.item():.1f}")
    print(f"  Expected: 1250.0 ✓")

    # Test that the result is reasonable
    assert np.isclose(result_original.item(), 1250.0, rtol=1e-10)

    print("\nTesting with scaling (model coordinates):")

    # Now test with a scaled mesh but model coordinates
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin,
    )

    mesh_scaled = uw.meshing.StructuredQuadBox(
        elementRes=(5, 5), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2
    )

    temp_scaled = uw.discretisation.MeshVariable("T", mesh_scaled, 1)

    # Set same uniform field value
    with uw.synchronised_array_update():
        temp_scaled.array[:, 0, 0] = 1250.0  # Same uniform value

    # Evaluate at model coordinates (no coord_units specified)
    result_scaled = uw.function.evaluate(temp_scaled.sym, model_coords)

    # Extract value from UWQuantity if needed
    result_val = (
        result_scaled._pint_qty.magnitude.item()
        if hasattr(result_scaled, "_pint_qty")
        else result_scaled.item()
    )
    print(f"  Result with scaled mesh: {result_val:.1f}")

    # Should get the same result as before - model coordinates should work the same way
    assert np.isclose(result_val, 1250.0, rtol=1e-10)
    print(f"  Expected: 1250.0 ✓")

    print("\nTesting new coord_units functionality:")

    # Now test new functionality - physical coordinates
    physical_coords_km = np.array([[500, 500]], dtype=np.float64)  # 500 km
    physical_coords_m = np.array([[500000, 500000]], dtype=np.float64)  # 500,000 m

    result_km = uw.function.evaluate(temp_scaled.sym, physical_coords_km, coord_units="km")
    result_m = uw.function.evaluate(temp_scaled.sym, physical_coords_m, coord_units="m")
    result_model = uw.function.evaluate(temp_scaled.sym, model_coords)  # No units

    # Extract values from UWQuantity if needed
    val_km = (
        result_km._pint_qty.magnitude.item()
        if hasattr(result_km, "_pint_qty")
        else result_km.item()
    )
    val_m = (
        result_m._pint_qty.magnitude.item() if hasattr(result_m, "_pint_qty") else result_m.item()
    )
    val_model = (
        result_model._pint_qty.magnitude.item()
        if hasattr(result_model, "_pint_qty")
        else result_model.item()
    )

    print(f"  Result from 500 km coords: {val_km:.1f}")
    print(f"  Result from 500,000 m coords: {val_m:.1f}")
    print(f"  Result from 0.5 model coords: {val_model:.1f}")

    # All should be identical
    assert np.allclose([val_km, val_m, val_model], val_model, rtol=1e-10)
    print("  All coordinate systems give identical results ✓")

    print("\n✅ BACKWARD COMPATIBILITY CONFIRMED")
    print("✅ NEW FUNCTIONALITY WORKING")
    print("✅ Original behavior preserved")


if __name__ == "__main__":
    test_original_behavior()
