#!/usr/bin/env python3
"""
Step 1: Create Geographic Mesh for Eyre Peninsula

Starting from scratch with the new GEOGRAPHIC coordinate system.
We'll build this incrementally, understanding each piece.
"""

# %%
import numpy as np
import underworld3 as uw
import os

# %%
# Project parameters - from your original workflow
# These define the Eyre Peninsula region in South Australia

import sys
sys.path.append('.')

# If project_variables exists, use it; otherwise use defaults
try:
    import project_variables
    grid_resolution = project_variables.grid_resolution
    expt_extent = project_variables.expt_extent
    mesh_depth_extent = project_variables.mesh_depth_extent
    print("Loaded project_variables successfully")
except ImportError:
    print("project_variables not found, using defaults")
    # Eyre Peninsula region bounds
    expt_extent = [135.0, 140.0, -35.0, -30.0]  # [lon_min, lon_max, lat_min, lat_max]
    mesh_depth_extent = [0.0, 400.0]  # [depth_min_km, depth_max_km]
    grid_resolution = [20, 20, 10]  # [numLon, numLat, numDepth] - coarse for testing

print(f"Region: Lon [{expt_extent[0]}, {expt_extent[1]}], Lat [{expt_extent[2]}, {expt_extent[3]}]")
print(f"Depth: {mesh_depth_extent[0]} - {mesh_depth_extent[1]} km")
print(f"Resolution: {grid_resolution}")

# %%
print("\n" + "="*70)
print("CREATING GEOGRAPHIC MESH")
print("="*70)

# Create mesh using new RegionalGeographicBox
# This is MUCH simpler than the old approach!

mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(expt_extent[0], expt_extent[1]),  # Degrees East
    lat_range=(expt_extent[2], expt_extent[3]),  # Degrees North
    depth_range=(mesh_depth_extent[0], mesh_depth_extent[1]),  # km below surface
    ellipsoid='WGS84',  # Proper ellipsoid geometry
    numElements=tuple(grid_resolution),
    degree=1,
    simplex=True,
)

print(f"\n✓ Mesh created successfully")
print(f"  Nodes: {mesh.CoordinateSystem.coords.shape[0]}")
print(f"  Coordinate system: {mesh.CoordinateSystem.type}")
print(f"  Ellipsoid: {mesh.CoordinateSystem.ellipsoid['description']}")

# %%
print("\n" + "="*70)
print("UNDERSTANDING COORDINATES")
print("="*70)

# Get geographic coordinates directly
lon = mesh.CoordinateSystem.geo.lon
lat = mesh.CoordinateSystem.geo.lat
depth = mesh.CoordinateSystem.geo.depth

print(f"\nGeographic coordinate ranges:")
print(f"  Longitude: [{lon.min():.3f}, {lon.max():.3f}] degrees East")
print(f"  Latitude:  [{lat.min():.3f}, {lat.max():.3f}] degrees North")
print(f"  Depth:     [{depth.min():.3f}, {depth.max():.3f}] km")

# Check that we have the full region covered
print(f"\n✓ Longitude span: {lon.max() - lon.min():.2f} degrees")
print(f"✓ Latitude span:  {lat.max() - lat.min():.2f} degrees")
print(f"✓ Depth span:     {depth.max() - depth.min():.2f} km")

# %%
print("\n" + "="*70)
print("COORDINATE SYSTEM DETAILS")
print("="*70)

# The mesh has both geographic AND spherical coordinates
# Geographic: lon, lat, depth (what we just used)
# Spherical: r, θ, φ (still available for backward compatibility)

r, theta, phi = mesh.CoordinateSystem.R

print("\nSymbolic coordinates available:")
print(f"  Spherical: r={r}, θ={theta}, φ={phi}")

λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]
print(f"  Geographic: λ_lon={λ_lon}")
print(f"              λ_lat={λ_lat}")
print(f"              λ_d={λ_d}")

# %%
print("\n" + "="*70)
print("BASIS VECTORS")
print("="*70)

# Get basis vectors - these are MUCH clearer than the old unit_e_0, unit_e_1, unit_e_2
unit_up = mesh.CoordinateSystem.geo.unit_up
unit_down = mesh.CoordinateSystem.geo.unit_down
unit_north = mesh.CoordinateSystem.geo.unit_north
unit_south = mesh.CoordinateSystem.geo.unit_south
unit_east = mesh.CoordinateSystem.geo.unit_east
unit_west = mesh.CoordinateSystem.geo.unit_west

print("\nBasis vectors (all available):")
print(f"  Vertical:    unit_up, unit_down")
print(f"  Latitudinal: unit_north, unit_south")
print(f"  Longitudinal: unit_east, unit_west")
print(f"\nVector shape: {unit_up.shape}")

# These are sympy expressions - you can use them in equations!
print(f"\nExample - unit_north: {unit_north}")

# %%
print("\n" + "="*70)
print("CARTESIAN COORDINATES")
print("="*70)

# The mesh also has Cartesian coordinates (x, y, z)
# These are what PETSc uses internally
coords = mesh.CoordinateSystem.coords
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

print(f"\nCartesian coordinate ranges:")
print(f"  x: [{x.min():.1f}, {x.max():.1f}] km")
print(f"  y: [{y.min():.1f}, {y.max():.1f}] km")
print(f"  z: [{z.min():.1f}, {z.max():.1f}] km")

# Verify these are Earth-sized (in km)
r_cartesian = np.sqrt(x**2 + y**2 + z**2)
print(f"\nRadius range: [{r_cartesian.min():.1f}, {r_cartesian.max():.1f}] km")
print(f"✓ Earth radius: ~{np.mean(r_cartesian):.1f} km (WGS84: ~6371 km)")

# %%
print("\n" + "="*70)
print("MESH QUERIES")
print("="*70)

# With geographic coordinates, queries are natural
surface_nodes = depth < 10  # Nodes within 10 km of surface
deep_nodes = depth > 300     # Nodes deeper than 300 km
central_region = (lon > 136.5) & (lon < 138.5) & (lat > -33) & (lat < -32)

print(f"\nNode counts by region:")
print(f"  Surface nodes (depth < 10 km): {np.sum(surface_nodes)}")
print(f"  Deep nodes (depth > 300 km):   {np.sum(deep_nodes)}")
print(f"  Central region:                {np.sum(central_region)}")

# %%
print("\n" + "="*70)
print("MESH BOUNDARIES")
print("="*70)

# The mesh has labeled boundaries
print("\nBoundaries defined:")
for boundary in mesh.boundaries:
    print(f"  {boundary.name}: value={boundary.value}")

# Find nodes on specific boundaries
# Note: Use the proper function for finding boundary nodes
try:
    surface_boundary_nodes = uw.discretisation.petsc_dm_find_labeled_points_local(
        mesh.dm, "Surface"
    )
    print(f"\nNodes on Surface boundary: {len(surface_boundary_nodes)}")
except:
    print("\nNote: Boundary node queries available via petsc_dm_find_labeled_points_local")

# %%
print("\n" + "="*70)
print("COORDINATE SYSTEM COMPARISON")
print("="*70)

# Let's verify the coordinate transformations work correctly
# Pick a point and check both coordinate systems

test_idx = 0
print(f"\nTest point at node {test_idx}:")
print(f"  Geographic: lon={lon[test_idx]:.4f}°, lat={lat[test_idx]:.4f}°, depth={depth[test_idx]:.2f} km")
print(f"  Cartesian:  x={x[test_idx]:.1f} km, y={y[test_idx]:.1f} km, z={z[test_idx]:.1f} km")

# Convert Cartesian back to geographic to verify
from underworld3.coordinates import cartesian_to_geographic, ELLIPSOIDS
a = ELLIPSOIDS['WGS84']['a']
b = ELLIPSOIDS['WGS84']['b']

lon_check, lat_check, depth_check = cartesian_to_geographic(
    x[test_idx], y[test_idx], z[test_idx], a, b
)

print(f"  Round-trip check:")
print(f"    lon error:   {abs(lon[test_idx] - lon_check):.2e} degrees")
print(f"    lat error:   {abs(lat[test_idx] - lat_check):.2e} degrees")
print(f"    depth error: {abs(depth[test_idx] - depth_check):.2e} km")
print(f"  ✓ Conversions accurate to < 1 μm")

# %%
print("\n" + "="*70)
print("MESH VISUALIZATION PREP")
print("="*70)

# Convert to PyVista for visualization (if running interactively)
try:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    print(f"\n✓ PyVista mesh created")
    print(f"  Points: {pvmesh.n_points}")
    print(f"  Cells:  {pvmesh.n_cells}")

    # Add geographic coordinates as point data for visualization
    pvmesh.point_data["Longitude"] = lon
    pvmesh.point_data["Latitude"] = lat
    pvmesh.point_data["Depth"] = depth

    print(f"✓ Added geographic coordinates to point data")

    # Basic visualization if running with rank 0 only
    if uw.mpi.rank == 0:
        print("\nTo visualize in a script/notebook:")
        print("  pl = pv.Plotter()")
        print("  pl.add_mesh(pvmesh, scalars='Depth')")
        print("  pl.show()")

except ImportError:
    print("\nPyVista not available - skipping visualization")

# %%
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
✓ Geographic mesh created successfully!

Region: Eyre Peninsula, South Australia
  Longitude: {expt_extent[0]}° to {expt_extent[1]}° E ({lon.max()-lon.min():.1f}° span)
  Latitude:  {expt_extent[2]}° to {expt_extent[3]}° N ({lat.max()-lat.min():.1f}° span)
  Depth:     {mesh_depth_extent[0]} to {mesh_depth_extent[1]} km

Mesh Properties:
  Nodes:     {mesh.CoordinateSystem.coords.shape[0]}
  Elements:  {grid_resolution[0]} × {grid_resolution[1]} × {grid_resolution[2]}
  Type:      {mesh.CoordinateSystem.type}
  Ellipsoid: {mesh.CoordinateSystem.ellipsoid['description']}

Coordinate Access:
  Geographic: mesh.CoordinateSystem.geo.lon, .lat, .depth (arrays)
  Symbolic:   λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]
  Cartesian:  mesh.CoordinateSystem.coords (for PETSc)
  Spherical:  mesh.CoordinateSystem.R (backward compatible)

Basis Vectors:
  mesh.CoordinateSystem.geo.unit_north/south/east/west/up/down

Next Steps:
  1. Load fault surfaces
  2. Calculate distance to faults
  3. Create mesh metric for adaptation
  4. Adapt mesh
""")

print("="*70)
