#!/usr/bin/env python3
"""Simple demonstration of mesh coordinate units patterns."""

import underworld3 as uw

# Setup
uw.reset_default_model()
model = uw.get_default_model()

# Define physical dimensions
L_x = uw.quantity(1000, "m")  # 1 km
L_y = uw.quantity(500, "m")   # 0.5 km

# Set reference quantities - domain_depth sets the length scale
model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=L_y,  # Length scale = 500m = 0.5 km
    mantle_temperature=uw.quantity(1000, "K"),
)

print("REFERENCE SCALES:")
scales = model.get_fundamental_scales()
print(f"  Length scale: {scales['length']}")
print()

print("PHYSICAL DIMENSIONS:")
print(f"  L_x = {L_x} = 1 km")
print(f"  L_y = {L_y} = 0.5 km")
print()

print("IN MODEL UNITS:")
L_x_model = model.to_model_units(L_x)
L_y_model = model.to_model_units(L_y)
print(f"  L_x = {L_x_model} model units  (1000m / 500m)")
print(f"  L_y = {L_y_model} model units  (500m / 500m)")
print()

print("="*70)
print("CORRECT PATTERN 1: Specify units, pass physical values")
print("="*70)
print("mesh = uw.meshing.StructuredQuadBox(")
print("    elementRes=(4, 4),")
print("    minCoords=(0.0, 0.0),")
print("    maxCoords=(1.0, 0.5),  # Values in km")
print("    units='km',             # Interpret as km")
print(")")
print()
print("Result: 1.0 km × 0.5 km domain → 2.0 × 1.0 in model units ✓")
print()

print("="*70)
print("CORRECT PATTERN 2: No units, pass model units")
print("="*70)
print("mesh = uw.meshing.StructuredQuadBox(")
print("    elementRes=(4, 4),")
print("    minCoords=(0.0, 0.0),")
print(f"    maxCoords=({L_x_model}, {L_y_model}),  # Model units (dimensionless)")
print("    # No units parameter")
print(")")
print()
print("Result: 2.0 × 1.0 model units → 1.0 km × 0.5 km physical ✓")
print()

print("="*70)
print("WRONG PATTERN: Your code (double conversion)")
print("="*70)
print("mesh = uw.meshing.StructuredQuadBox(")
print("    elementRes=(4, 4),")
print("    minCoords=(0.0, 0.0),")
print(f"    maxCoords=({L_x_model}, {L_y_model}),  # These are model units!")
print("    units='km',  # But mesh thinks they're km values!")
print(")")
print()
print("What happens:")
print(f"  1. You pass maxCoords=({L_x_model}, {L_y_model}) - already dimensionless")
print(f"  2. Mesh sees units='km', interprets 2.0 as '2.0 km'")
print(f"  3. Mesh converts: 2.0 km / 0.5 km = 4.0 model units")
print(f"  4. Result: 4.0 × 2.0 model units (WRONG - should be 2.0 × 1.0)")
print()

print("="*70)
print("THE FIX:")
print("="*70)
print("Choose ONE of these:")
print()
print("Option A - Use physical units:")
print("  mesh = uw.meshing.StructuredQuadBox(")
print("      maxCoords=(1.0, 0.5),  # km values")
print("      units='km'")
print("  )")
print()
print("Option B - Use model units (recommended):")
print("  mesh = uw.meshing.StructuredQuadBox(")
print(f"      maxCoords=({L_x_model}, {L_y_model}),  # dimensionless")
print("      # No units parameter")
print("  )")
print()
print("DON'T mix converted model values with units parameter!")
