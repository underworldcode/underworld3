#!/usr/bin/env python3
"""
Examples showing correct mesh coordinate patterns with units system.
Demonstrates the three ways to specify mesh coordinates.
"""

import underworld3 as uw

# Setup
uw.reset_default_model()
model = uw.get_default_model()

# Define physical dimensions
L_x = uw.quantity(1000, "m")  # 1 km
L_y = uw.quantity(500, "m")   # 0.5 km

# Set reference quantities
model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=L_y,  # Model length scale = 0.5 km
    mantle_temperature=uw.quantity(1000, "K"),
)

print("=" * 70)
print("Model length scale:", model.get_fundamental_scales()['length'])
print("L_x in model units:", model.to_model_units(L_x))
print("L_y in model units:", model.to_model_units(L_y))
print("=" * 70)

# ============================================================================
# PATTERN 1: Specify mesh units, pass physical values
# ============================================================================
print("\n### PATTERN 1: Mesh with units='km', pass km values ###")

mesh1 = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.1,  # 0.1 km (100 m)
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 0.5),  # 1.0 km × 0.5 km IN KILOMETERS
    units="km",  # Coordinates are interpreted as km
)

print(f"Mesh 1 coordinates (first 3 points):")
print(f"  X.coords: {mesh1.X.coords[0:3]}")
print(f"  DM coords: {mesh1.dm.getCoordinates().array[0:6]}")
print(f"  Expected: Domain is 1km × 0.5km = 2.0 × 1.0 in model units")

# ============================================================================
# PATTERN 2: No units specified, pass model units (dimensionless)
# ============================================================================
print("\n### PATTERN 2: No units, pass model coordinate values ###")

L_x_model = model.to_model_units(L_x)  # 2.0
L_y_model = model.to_model_units(L_y)  # 1.0

mesh2 = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.2,  # 0.2 model units
    minCoords=(0.0, 0.0),
    maxCoords=(L_x_model, L_y_model),  # 2.0 × 1.0 in model units
    # NO units parameter - coordinates are model units
)

print(f"Mesh 2 coordinates (first 3 points):")
print(f"  X.coords: {mesh2.X.coords[0:3]}")
print(f"  DM coords: {mesh2.dm.getCoordinates().array[0:6]}")
print(f"  Expected: Domain is 2.0 × 1.0 in model units")

# ============================================================================
# PATTERN 3: Pass UWQuantity directly (automatic conversion)
# ============================================================================
print("\n### PATTERN 3: Pass UWQuantity directly, no units parameter ###")

mesh3 = uw.meshing.UnstructuredSimplexBox(
    cellSize=model.to_model_units(uw.quantity(100, "m")),  # Convert to model units
    minCoords=(0.0, 0.0),
    maxCoords=(L_x_model, L_y_model),  # Already in model units
    # No units - values are model units
)

print(f"Mesh 3 coordinates (first 3 points):")
print(f"  X.coords: {mesh3.X.coords[0:3]}")
print(f"  DM coords: {mesh3.dm.getCoordinates().array[0:6]}")
print(f"  Expected: Domain is 2.0 × 1.0 in model units")

# ============================================================================
# ANTI-PATTERN: What you were doing (WRONG)
# ============================================================================
print("\n### ANTI-PATTERN: Double conversion (WRONG) ###")

mesh_wrong = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.2,
    minCoords=(0.0, 0.0),
    maxCoords=(L_x_model, L_y_model),  # These are ALREADY model units (2.0, 1.0)
    units="km",  # But mesh treats them as km values!
)

print(f"Mesh WRONG coordinates (first 3 points):")
print(f"  X.coords: {mesh_wrong.X.coords[0:3]}")
print(f"  DM coords: {mesh_wrong.dm.getCoordinates().array[0:6]}")
print(f"  Problem: 2.0 interpreted as '2.0 km', converted to 4.0 model units!")
print(f"  Domain became 4.0 × 2.0 in model units instead of 2.0 × 1.0")

# ============================================================================
# VERIFICATION: Check actual mesh sizes
# ============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Mesh bounding boxes in model units")
print("=" * 70)

for i, mesh in enumerate([mesh1, mesh2, mesh3, mesh_wrong], 1):
    coords = mesh.dm.getCoordinates().array
    x_coords = coords[0::mesh.dim]
    y_coords = coords[1::mesh.dim]
    name = "WRONG" if i == 4 else f"{i}"
    print(f"Mesh {name}: X ∈ [{x_coords.min():.1f}, {x_coords.max():.1f}], "
          f"Y ∈ [{y_coords.min():.1f}, {y_coords.max():.1f}]")

print("\nExpected for correct meshes: X ∈ [0.0, 2.0], Y ∈ [0.0, 1.0]")
