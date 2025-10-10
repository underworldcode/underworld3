#!/usr/bin/env python3
"""Test that original evaluate behavior is preserved."""

import os
import numpy as np

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw

def test_original_behavior():
    """Test that original evaluate function behavior is preserved."""
    print("=== TESTING BACKWARD COMPATIBILITY ===")

    # Set up a simple test case
    uw.reset_default_model()
    model = uw.get_default_model()

    # Create mesh WITHOUT scaling (original behavior)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(5, 5),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        qdegree=2
    )

    # Create a simple field
    temperature = uw.discretisation.MeshVariable("T", mesh, 1)

    with uw.synchronised_array_update():
        temperature.array[:, 0, 0] = 1000 + 500 * mesh.data[:, 0]  # Linear in x

    # Test original behavior - coords in model units (0-1 range)
    model_coords = np.array([[0.5, 0.5]], dtype=np.float64)

    print("Testing original behavior (model coordinates, no units):")
    print(f"  Input coordinates: {model_coords}")

    # This should work exactly as before
    result_original = uw.function.evaluate(temperature.sym, model_coords)
    print(f"  Result: {result_original.item():.1f}")

    # Test that the result is reasonable (should be 1000 + 500*0.5 = 1250)
    expected = 1000 + 500 * 0.5
    assert np.isclose(result_original.item(), expected, rtol=1e-10)
    print(f"  Expected: {expected:.1f} ✓")

    print("\nTesting backward compatibility with scaled mesh:")

    # Now test with a scaled mesh but model coordinates
    model.set_reference_quantities(
        characteristic_length=1000 * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    mesh_scaled = uw.meshing.StructuredQuadBox(
        elementRes=(5, 5),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        qdegree=2
    )

    temp_scaled = uw.discretisation.MeshVariable("T", mesh_scaled, 1)

    with uw.synchronised_array_update():
        temp_scaled.array[:, 0, 0] = 1000 + 500 * mesh_scaled.data[:, 0]

    # This should still work - model coordinates with no coord_units
    result_scaled = uw.function.evaluate(temp_scaled.sym, model_coords)

    # Extract value from UWQuantity if needed
    result_val = result_scaled._pint_qty.magnitude.item() if hasattr(result_scaled, '_pint_qty') else result_scaled.item()
    print(f"  Result with scaled mesh: {result_val:.1f}")

    # Should get the same result as before
    assert np.isclose(result_val, expected, rtol=1e-10)
    print(f"  Expected: {expected:.1f} ✓")

    print("\nTesting new coord_units functionality:")

    # Now test new functionality - physical coordinates
    physical_coords_km = np.array([[500, 500]], dtype=np.float64)  # 500 km
    physical_coords_m = np.array([[500000, 500000]], dtype=np.float64)  # 500,000 m

    result_km = uw.function.evaluate(temp_scaled.sym, physical_coords_km, coord_units='km')
    result_m = uw.function.evaluate(temp_scaled.sym, physical_coords_m, coord_units='m')
    result_model = uw.function.evaluate(temp_scaled.sym, model_coords)  # No units

    # Extract values from UWQuantity if needed
    val_km = result_km._pint_qty.magnitude.item() if hasattr(result_km, '_pint_qty') else result_km.item()
    val_m = result_m._pint_qty.magnitude.item() if hasattr(result_m, '_pint_qty') else result_m.item()
    val_model = result_model._pint_qty.magnitude.item() if hasattr(result_model, '_pint_qty') else result_model.item()

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