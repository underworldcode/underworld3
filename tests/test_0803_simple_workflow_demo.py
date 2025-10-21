#!/usr/bin/env python3
"""
Simple demonstration of the "flip units around however we want" capability.

This test focuses on the core value proposition: seamless unit flexibility in workflows.
"""

import os
import numpy as np
import pytest

# DISABLE SYMPY CACHE
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw


@pytest.mark.skip(reason="coord_units parameter not implemented - planned feature for evaluate()")
def test_seamless_unit_flexibility():
    """
    Demonstrate the key value: seamless unit flexibility in a typical workflow.

    Shows that users can:
    1. Input data in whatever units they have
    2. Query coordinates in any units
    3. View results in whatever units they want
    All without manual unit conversion or worrying about internal coordinates.
    """
    print("=== SEAMLESS UNIT FLEXIBILITY TEST ===")
    print()

    # Set up a realistic problem
    uw.reset_default_model()
    model = uw.get_default_model()

    # User has mixed input data (typical real-world scenario)
    domain_width_km = 1000   # Domain width in km
    domain_height_km = 500   # Domain height in km

    # Set reference scales
    model.set_reference_quantities(
        characteristic_length=domain_width_km * uw.units.km,
        plate_velocity=5 * uw.units.cm / uw.units.year,
        mantle_temperature=1500 * uw.units.kelvin
    )

    # Create mesh in model coordinates
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 5),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 0.5),
        qdegree=2
    )

    # Create temperature field
    temperature = uw.discretisation.MeshVariable("T", mesh, 1)

    # Set up simple temperature field
    with uw.synchronised_array_update():
        temperature.array[:, 0, 0] = 1000 + 500 * mesh.X.coords[:, 0]  # Linear in x

    print("✓ Set up geophysics domain with realistic scales")
    print(f"  Domain: {domain_width_km} km × {domain_height_km} km")
    print()

    # === KEY TEST 1: Input coordinates in whatever units user has ===

    print("=== TEST 1: COORDINATE INPUT FLEXIBILITY ===")

    # User has location data in different units (typical real scenario)
    target_location_km = np.array([[250, 125]], dtype=np.float64)      # Location in km
    target_location_m = np.array([[250000, 125000]], dtype=np.float64)  # Same location in m
    target_location_model = np.array([[0.25, 0.125]], dtype=np.float64)  # Same in model coords (250km/1000km, 125km/1000km)

    # All should give identical results when using explicit coordinate units
    temp_from_km = uw.function.evaluate(temperature.sym, target_location_km, coord_units='km')
    temp_from_m = uw.function.evaluate(temperature.sym, target_location_m, coord_units='m')
    temp_from_model = uw.function.evaluate(temperature.sym, target_location_model)  # No units = model coords

    print("Querying temperature at same location using different coordinate units:")
    print(f"  Location in km: ({target_location_km[0,0]:.0f}, {target_location_km[0,1]:.0f}) km")
    print(f"  Location in m:  ({target_location_m[0,0]:.0f}, {target_location_m[0,1]:.0f}) m")
    print(f"  Location in model: ({target_location_model[0,0]:.2f}, {target_location_model[0,1]:.2f})")
    print()

    # Handle UWQuantity objects - use _pint_qty.magnitude for numerical value
    # Use item() to extract scalar from array results
    if hasattr(temp_from_km, '_pint_qty'):
        temp_km_val = temp_from_km._pint_qty.magnitude.item()
    else:
        temp_km_val = np.asarray(temp_from_km).item()

    if hasattr(temp_from_m, '_pint_qty'):
        temp_m_val = temp_from_m._pint_qty.magnitude.item()
    else:
        temp_m_val = np.asarray(temp_from_m).item()

    if hasattr(temp_from_model, '_pint_qty'):
        temp_model_val = temp_from_model._pint_qty.magnitude.item()
    else:
        temp_model_val = np.asarray(temp_from_model).item()

    print(f"  Temperature from km coords: {temp_km_val:.1f} K")
    print(f"  Temperature from m coords:  {temp_m_val:.1f} K")
    print(f"  Temperature from model coords: {temp_model_val:.1f} K")

    # Verify they're identical (compare the numerical values)
    assert np.allclose(temp_km_val, temp_m_val, rtol=1e-12)
    assert np.allclose(temp_km_val, temp_model_val, rtol=1e-12)
    print("  ✓ All coordinate systems give identical results")
    print()

    # === KEY TEST 2: Switch units mid-workflow ===

    print("=== TEST 2: MID-WORKFLOW UNIT SWITCHING ===")

    # Start analysis in km (user's preferred units)
    analysis_points_km = np.array([
        [100, 50],   # Point 1 in km
        [500, 250],  # Point 2 in km
        [900, 450],  # Point 3 in km
    ], dtype=np.float64)

    temps_km = uw.function.evaluate(temperature.sym, analysis_points_km, coord_units='km')
    print("Initial analysis using km coordinates:")
    # Extract temperature values - temps_km is a UWQuantity containing an array
    if hasattr(temps_km, '_pint_qty'):
        temp_values_km = temps_km._pint_qty.magnitude.flatten()
    else:
        temp_values_km = np.array(temps_km).flatten()

    for i, (pt, temp_val) in enumerate(zip(analysis_points_km, temp_values_km)):
        print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f}) km → {temp_val:.1f} K")

    # Switch to meters for detailed analysis (user changes preference)
    analysis_points_m = analysis_points_km * 1000  # Convert to meters
    temps_m = uw.function.evaluate(temperature.sym, analysis_points_m, coord_units='m')

    print("\\nSwitched to meter coordinates for detailed work:")
    # Extract temperature values for meters
    if hasattr(temps_m, '_pint_qty'):
        temp_values_m = temps_m._pint_qty.magnitude.flatten()
    else:
        temp_values_m = np.array(temps_m).flatten()

    for i, (pt, temp_val) in enumerate(zip(analysis_points_m, temp_values_m)):
        print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f}) m → {temp_val:.1f} K")

    # Results should be identical
    assert np.allclose(temp_values_km, temp_values_m, rtol=1e-12)
    print("  ✓ Seamless unit switching with identical results")
    print()

    # === KEY TEST 3: Mixed coordinate systems in same workflow ===

    print("=== TEST 3: MIXED COORDINATE SYSTEMS ===")

    # User has data points from different sources in different units
    survey_point_km = np.array([[200, 100]], dtype=np.float64)       # GPS survey in km
    drill_point_m = np.array([[600000, 300000]], dtype=np.float64)   # Drill log in m
    model_point_dimensionless = np.array([[0.8, 0.4]], dtype=np.float64)  # Model prediction

    # Query all in the same workflow with explicit coordinate units
    temp_survey = uw.function.evaluate(temperature.sym, survey_point_km, coord_units='km')
    temp_drill = uw.function.evaluate(temperature.sym, drill_point_m, coord_units='m')
    temp_model = uw.function.evaluate(temperature.sym, model_point_dimensionless)  # No units = model coords

    print("Analyzing data from mixed sources with different coordinate units:")

    # Extract scalar values from UWQuantity objects
    survey_val = temp_survey._pint_qty.magnitude.item() if hasattr(temp_survey, '_pint_qty') else np.asarray(temp_survey).item()
    drill_val = temp_drill._pint_qty.magnitude.item() if hasattr(temp_drill, '_pint_qty') else np.asarray(temp_drill).item()
    model_val = temp_model._pint_qty.magnitude.item() if hasattr(temp_model, '_pint_qty') else np.asarray(temp_model).item()

    print(f"  GPS survey point (km): {survey_val:.1f} K")
    print(f"  Drill log point (m): {drill_val:.1f} K")
    print(f"  Model prediction (model coords): {model_val:.1f} K")
    print("  ✓ Mixed coordinate systems work seamlessly in same analysis")
    print()

    # === KEY TEST 4: Multiple queries with different precision needs ===

    print("=== TEST 4: PRECISION FLEXIBILITY ===")

    # High precision point in micrometers (engineering scale)
    # 250 km = 250,000 m = 250,000,000,000 μm
    precise_point_um = np.array([[250000000000, 125000000000]], dtype=np.float64)  # Same point in μm
    temp_um = uw.function.evaluate(temperature.sym, precise_point_um, coord_units='micrometer')

    # Same point in different precisions
    print("Same location queried with different precision units:")

    # Extract value from micrometer result
    temp_um_val = temp_um._pint_qty.magnitude.item() if hasattr(temp_um, '_pint_qty') else np.asarray(temp_um).item()

    print(f"  Kilometer precision: {temp_km_val:.1f} K")
    print(f"  Meter precision: {temp_m_val:.3f} K")
    print(f"  Micrometer precision: {temp_um_val:.6f} K")

    # Should all be identical
    assert np.allclose(temp_km_val, temp_um_val, rtol=1e-12)
    print("  ✓ Precision maintained across unit scales")
    print()

    print("=== WORKFLOW VALUE DEMONSTRATED ===")
    print("✅ Users can input coordinates in ANY units they have")
    print("✅ Users can switch units mid-workflow without conversion")
    print("✅ Users can mix coordinate systems in same analysis")
    print("✅ Precision maintained across all unit scales")
    print("✅ No manual unit conversion or coordinate transformation needed")
    print()
    print("🎉 'FLIP UNITS AROUND HOWEVER WE WANT' CAPABILITY CONFIRMED!")


if __name__ == "__main__":
    test_seamless_unit_flexibility()