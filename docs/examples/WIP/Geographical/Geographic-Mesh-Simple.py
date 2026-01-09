#!/usr/bin/env python3
"""
Simplified geographic mesh workflow using the new GEOGRAPHIC coordinate system.

This demonstrates how the new API makes working with geographic meshes much simpler.
"""

# %%
import numpy as np
import underworld3 as uw

# %%
# Original workflow from project_variables (simplified)
# Eyre Peninsula region, South Australia
expt_extent = [135.0, 140.0, -35.0, -30.0]  # [lon_min, lon_max, lat_min, lat_max]
mesh_depth_extent = [0.0, 400.0]  # [depth_min_km, depth_max_km]
grid_resolution = [10, 10, 10]  # [numLon, numLat, numDepth]

# %%
# BEFORE: Old workflow with RegionalSphericalBox
# Had to manually calculate radius from depth:
# radius_outer = 1.0
# radius_inner = 1.0 - (mesh_depth_extent[1] / 6370)
#
# cs_mesh = uw.meshing.RegionalSphericalBox(
#     SWcorner=[expt_extent[0], expt_extent[2]],
#     NEcorner=[expt_extent[1], expt_extent[3]],
#     radiusOuter=radius_outer,
#     radiusInner=radius_inner,
#     numElementsLon=grid_resolution[0],
#     numElementsLat=grid_resolution[1],
#     numElementsDepth=grid_resolution[2],
#     simplex=True,
# )

# %%
# AFTER: New workflow with RegionalGeographicBox
# Much more intuitive - specify geographic coordinates directly!

mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(expt_extent[0], expt_extent[1]),  # Degrees East
    lat_range=(expt_extent[2], expt_extent[3]),  # Degrees North
    depth_range=(mesh_depth_extent[0], mesh_depth_extent[1]),  # km below surface
    ellipsoid='WGS84',  # Proper ellipsoid geometry
    numElements=tuple(grid_resolution),
    degree=1,
    simplex=True,
)

print(f"Created mesh with {mesh.CoordinateSystem.coords.shape[0]} nodes")
print(f"Coordinate system type: {mesh.CoordinateSystem.type}")

# %%
# BEFORE: Extracting basis vectors was confusing
# unit_vertical = cs_mesh.CoordinateSystem.unit_e_0
# unit_SN = -cs_mesh.CoordinateSystem.unit_e_1  # Why negative??
# unit_EW = cs_mesh.CoordinateSystem.unit_e_2

# AFTER: Clear and intuitive basis vectors
unit_up = mesh.CoordinateSystem.geo.unit_up
unit_down = mesh.CoordinateSystem.geo.unit_down
unit_north = mesh.CoordinateSystem.geo.unit_north
unit_south = mesh.CoordinateSystem.geo.unit_south
unit_east = mesh.CoordinateSystem.geo.unit_east
unit_west = mesh.CoordinateSystem.geo.unit_west

print("\nBasis vectors accessed successfully")
print(f"unit_up shape: {unit_up.shape}")

# %%
# BEFORE: Manual coordinate conversions everywhere
# R = uw.function.evalf(cs_mesh.CoordinateSystem.R, cs_mesh.X.coords)
# for node in range(cs_mesh.X.coords.shape[0]):
#     ph1 = R[node, 2]
#     th1 = R[node, 1]
#     longitude = 360 * ph1 / (2 * np.pi)
#     latitude = 90 - 360 * th1 / (2 * np.pi)  # Sign confusion!

# AFTER: Direct access to geographic coordinates
lon = mesh.CoordinateSystem.geo.lon
lat = mesh.CoordinateSystem.geo.lat
depth = mesh.CoordinateSystem.geo.depth

print("\nGeographic coordinates:")
print(f"Longitude range: [{lon.min():.2f}, {lon.max():.2f}] degrees East")
print(f"Latitude range: [{lat.min():.2f}, {lat.max():.2f}] degrees North")
print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] km")

# %%
# Create mesh variables - same as before
topo = uw.discretisation.MeshVariable(
    "h", mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"\mathcal{H}"
)

print(f"\nCreated topography variable: {topo.name}")

# %%
# BEFORE: Complicated topography mapping with manual conversions
# with cs_mesh.access(topo):
#     R = uw.function.evalf(cs_mesh.CoordinateSystem.R, cs_mesh.X.coords)
#     for node in range(cs_mesh.X.coords.shape[0]):
#         ph1 = R[node, 2]
#         th1 = R[node, 1]
#         topo.data[node, 0] = ep_topo_value(
#             360 * ph1 / (2 * np.pi),
#             90 - 360 * th1 / (2 * np.pi)
#         )

# AFTER: Direct geographic coordinates - much simpler!
# Example: Set topography to a simple function for demonstration
topo.array[:] = (np.sin(np.radians(lon)) * np.cos(np.radians(lat)) * 1000).reshape(-1, 1, 1)  # meters

print(f"Topography range: [{np.asarray(topo.array).min():.1f}, {np.asarray(topo.array).max():.1f}] meters")

# %%
# Access symbolic coordinates for equations
λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]

print("\nSymbolic coordinates:")
print(f"λ_lon: {λ_lon}")
print(f"λ_lat: {λ_lat}")
print(f"λ_d: {λ_d}")

# %%
# Example: Create a temperature field that varies with depth
T = uw.discretisation.MeshVariable(
    "T", mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"T"
)

# Temperature decreases with depth: T = 1600 - 0.5 * depth
import sympy
T_expr = 1600 - 0.5 * λ_d

# Evaluate expression at mesh coordinates
import underworld3.function
T_values = underworld3.function.evaluate(T_expr, T.coords)
T.array[:, 0] = T_values.reshape(-1, 1)

print(f"\nTemperature range: [{np.asarray(T.array).min():.1f}, {np.asarray(T.array).max():.1f}] K")
print(f"Temperature at surface (depth=0): ~{np.asarray(T.array).max():.1f} K")
print(f"Temperature at bottom (depth=400km): ~{np.asarray(T.array).min():.1f} K")

# %%
# BEFORE: Depth calculation was manual
# depth = (1 - uw.function.evalf(r, coords)) * 6370

# AFTER: Depth is directly available
print("\nDirect depth access:")
print(f"Depth array shape: {depth.shape}")
print(f"Shallow nodes (depth < 10 km): {np.sum(depth < 10)}")
print(f"Deep nodes (depth > 300 km): {np.sum(depth > 300)}")

# %%
# Backward compatibility: spherical coordinates still work
r, theta, phi = mesh.CoordinateSystem.R

print("\nBackward compatibility check:")
print(f"Spherical coordinate r: {r}")
print(f"Spherical coordinate θ: {theta}")
print(f"Spherical coordinate φ: {phi}")

# Evaluate at one point
r_val = underworld3.function.evaluate(r, mesh.CoordinateSystem.coords[0:1])
print(f"Radius at first node: {float(r_val[0]):.6f}")

# %%
# Summary: What the new API provides
print("\n" + "="*60)
print("SUMMARY: Benefits of the new GEOGRAPHIC coordinate system")
print("="*60)
print("✓ Natural coordinate specification (lon, lat, depth)")
print("✓ No manual radius calculations")
print("✓ No manual coordinate conversions")
print("✓ Clear basis vector names (no sign confusion)")
print("✓ Proper ellipsoid geometry (WGS84)")
print("✓ Direct access to geographic arrays")
print("✓ Symbolic coordinates for equations")
print("✓ Works for any planet (Earth, Mars, Moon, Venus)")
print("✓ Backward compatible with spherical coordinates")
print("="*60)

# %%
# Example use cases that are now simpler:

# 1. Query points by geographic coordinates
surface_nodes = depth < 1.0  # Nodes within 1 km of surface
print(f"\nSurface nodes: {np.sum(surface_nodes)}")

# 2. Select region by lat/lon
australia_southeast = (lon > 137) & (lon < 139) & (lat > -33) & (lat < -31)
print(f"SE Australia region nodes: {np.sum(australia_southeast)}")

# 3. Create boundary conditions using natural directions
# Example: No vertical flow at surface
# v_surface = 0 * mesh.CoordinateSystem.geo.unit_up

# 4. Create fields that vary with geographic location
# Example: Temperature anomaly in a specific region
temp_anomaly = np.where(
    (lon > 136) & (lon < 138) & (lat > -34) & (lat < -32) & (depth < 100),
    100.0,  # 100K hotter
    0.0
)
print(f"Nodes with temperature anomaly: {np.sum(temp_anomaly > 0)}")

print("\n✓ Workflow demonstration complete!")
