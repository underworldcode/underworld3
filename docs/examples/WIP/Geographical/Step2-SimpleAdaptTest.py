#!/usr/bin/env python3
"""
Step 2: Simple Mesh Adaptation Test

Quick test to verify mesh adaptation works with geographic meshes.
"""

import numpy as np
import underworld3 as uw

# Very coarse mesh for fast testing
expt_extent = [135.0, 140.0, -35.0, -30.0]
mesh_depth_extent = [0.0, 400.0]
grid_resolution = [5, 5, 5]  # Very coarse!

print("Creating base mesh...")
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(expt_extent[0], expt_extent[1]),
    lat_range=(expt_extent[2], expt_extent[3]),
    depth_range=(mesh_depth_extent[0], mesh_depth_extent[1]),
    ellipsoid='WGS84',
    numElements=tuple(grid_resolution),
    degree=1,
    simplex=True,
)

print(f"Base mesh: {mesh.CoordinateSystem.coords.shape[0]} nodes")

# Test uniform metric
H = uw.discretisation.MeshVariable("H", mesh, 1)
Metric = uw.discretisation.MeshVariable("M", mesh, 1, degree=1)

print("\nTesting uniform metric H = 100...")
H.array[:] = 100.0

print("Adapting...")
try:
    icoord, adapted_mesh = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)
    print(f"✓ Success! Adapted mesh: {adapted_mesh.CoordinateSystem.coords.shape[0]} nodes")
    print(f"  Ratio: {adapted_mesh.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")
except Exception as e:
    print(f"✗ Failed: {e}")
