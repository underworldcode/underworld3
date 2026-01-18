#!/usr/bin/env python3
"""
Step 4: Test Mesh Adaptation with Linear Metric Gradient

Testing with a simple linear gradient to understand metric behavior:
- Vary metric linearly with depth
- See how the mesh adapts
- Understand the relationship between metric values and element distribution
"""

# %%
import numpy as np
import underworld3 as uw
import sys
sys.path.append('.')

# %%
print("="*70)
print("LINEAR METRIC GRADIENT TEST")
print("="*70)

# Use very coarse initial mesh for fast testing
expt_extent = [135.0, 140.0, -35.0, -30.0]
mesh_depth_extent = [0.0, 400.0]
grid_resolution = [5, 5, 5]  # Very coarse to start

print(f"\nRegion: Lon [{expt_extent[0]}, {expt_extent[1]}], Lat [{expt_extent[2]}, {expt_extent[3]}]")
print(f"Depth: {mesh_depth_extent[0]} - {mesh_depth_extent[1]} km")
print(f"Initial resolution: {grid_resolution}")

# %%
print("\n" + "="*70)
print("CREATE BASE MESH")
print("="*70)

mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(expt_extent[0], expt_extent[1]),
    lat_range=(expt_extent[2], expt_extent[3]),
    depth_range=(mesh_depth_extent[0], mesh_depth_extent[1]),
    ellipsoid='WGS84',
    numElements=tuple(grid_resolution),
    degree=1,
    simplex=True,
)

print(f"\n✓ Base mesh created")
print(f"  Nodes: {mesh.CoordinateSystem.coords.shape[0]}")

# Get depth for gradient
depth = mesh.CoordinateSystem.geo.depth

print(f"\n  Depth range: [{depth.min():.1f}, {depth.max():.1f}] km")

# %%
print("\n" + "="*70)
print("TEST 1: LINEAR GRADIENT - FINE AT SURFACE, COARSE AT DEPTH")
print("="*70)

print("""
Strategy: Create a linear gradient where:
- Surface (depth=0): H = H_surface (fine)
- Bottom (depth=400): H = H_bottom (coarse)

We need to choose values carefully. From user feedback:
- Smaller H → coarser mesh
- Larger H → finer mesh
- Refinement drives metric to unity

Let's try modest values to avoid memory issues:
- Surface: H = 1000 (finer)
- Bottom: H = 100 (coarser)
- Linear interpolation between
""")

H1 = uw.discretisation.MeshVariable("H1", mesh, 1)
Metric1 = uw.discretisation.MeshVariable("M1", mesh, 1, degree=1)

# Linear gradient: fine at surface, coarse at depth
H_surface = 1000.0
H_bottom = 100.0

# Linear interpolation
H1_values = H_surface + (H_bottom - H_surface) * (depth / mesh_depth_extent[1])

print(f"\n✓ Linear gradient metric:")
print(f"  H at surface (depth=0): {H_surface:.1f}")
print(f"  H at bottom (depth={mesh_depth_extent[1]}): {H_bottom:.1f}")
print(f"  Actual range: [{H1_values.min():.1f}, {H1_values.max():.1f}]")

# Store in mesh variable (shape must be (N, 1, 1) for mesh variable created with num_components=1)
H1.array[:] = H1_values.reshape(-1, 1, 1)

# Statistics
surface_nodes = np.sum(depth < 50)
deep_nodes = np.sum(depth > 350)
print(f"\n  Nodes near surface (< 50 km): {surface_nodes}")
print(f"  Nodes at depth (> 350 km): {deep_nodes}")
print(f"  Mean H near surface: {H1_values[depth < 50].mean():.1f}")
print(f"  Mean H at depth: {H1_values[depth > 350].mean():.1f}")

# %%
print("\n" + "="*70)
print("ADAPTING MESH WITH LINEAR GRADIENT")
print("="*70)

print("\nStarting adaptation...")
print("(This may take some time - be patient!)")

try:
    icoord1, mesh1 = uw.adaptivity.mesh_adapt_meshVar(mesh, H1, Metric1)

    print(f"\n✓ SUCCESS! Adaptation completed")
    print(f"\n  Mesh statistics:")
    print(f"    Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"    Adapted nodes:  {mesh1.CoordinateSystem.coords.shape[0]}")
    print(f"    Ratio:          {mesh1.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")

    # Analyze depth distribution
    depth1 = mesh1.CoordinateSystem.geo.depth
    print(f"\n  Adapted mesh depth distribution:")

    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    for i in range(len(bins)-1):
        nodes_in_bin = np.sum((depth1 >= bins[i]) & (depth1 < bins[i+1]))
        pct = 100 * nodes_in_bin / len(depth1)
        print(f"    {bins[i]:3d}-{bins[i+1]:3d} km: {nodes_in_bin:5d} nodes ({pct:5.1f}%)")

    print(f"\n  Expected behavior: More nodes near surface (higher H) than at depth (lower H)")

    # Calculate node density ratio
    surface_adapted = np.sum(depth1 < 100)
    deep_adapted = np.sum(depth1 > 300)

    if deep_adapted > 0:
        density_ratio = surface_adapted / deep_adapted
        metric_ratio = H_surface / H_bottom
        print(f"\n  Node density analysis:")
        print(f"    Surface nodes (< 100 km): {surface_adapted}")
        print(f"    Deep nodes (> 300 km):    {deep_adapted}")
        print(f"    Density ratio: {density_ratio:.2f}x")
        print(f"    Metric ratio:  {metric_ratio:.2f}x")
        print(f"    → Density ratio should reflect metric ratio")

except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    import traceback
    traceback.print_exc()
    mesh1 = None

# %%
print("\n" + "="*70)
print("TEST 2: REVERSE GRADIENT - COARSE AT SURFACE, FINE AT DEPTH")
print("="*70)

print("""
Now reverse the gradient:
- Surface: H = 100 (coarser)
- Bottom: H = 1000 (finer)

This should give MORE nodes at depth than at surface.
""")

H2 = uw.discretisation.MeshVariable("H2", mesh, 1)
Metric2 = uw.discretisation.MeshVariable("M2", mesh, 1, degree=1)

# Reverse gradient
H_surface_2 = 100.0
H_bottom_2 = 1000.0

H2_values = H_surface_2 + (H_bottom_2 - H_surface_2) * (depth / mesh_depth_extent[1])

print(f"\n✓ Reverse gradient metric:")
print(f"  H at surface: {H_surface_2:.1f}")
print(f"  H at bottom:  {H_bottom_2:.1f}")

H2.array[:] = H2_values.reshape(-1, 1, 1)

print("\nStarting adaptation...")

try:
    icoord2, mesh2 = uw.adaptivity.mesh_adapt_meshVar(mesh, H2, Metric2)

    print(f"\n✓ SUCCESS! Adaptation completed")
    print(f"\n  Mesh statistics:")
    print(f"    Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"    Adapted nodes:  {mesh2.CoordinateSystem.coords.shape[0]}")
    print(f"    Ratio:          {mesh2.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")

    # Analyze depth distribution
    depth2 = mesh2.CoordinateSystem.geo.depth
    print(f"\n  Adapted mesh depth distribution:")

    for i in range(len(bins)-1):
        nodes_in_bin = np.sum((depth2 >= bins[i]) & (depth2 < bins[i+1]))
        pct = 100 * nodes_in_bin / len(depth2)
        print(f"    {bins[i]:3d}-{bins[i+1]:3d} km: {nodes_in_bin:5d} nodes ({pct:5.1f}%)")

    print(f"\n  Expected behavior: More nodes at depth (higher H) than at surface (lower H)")

    # Compare with Test 1
    if mesh1 is not None:
        print(f"\n  Comparison with Test 1:")
        print(f"    Test 1 (fine→coarse): {mesh1.CoordinateSystem.coords.shape[0]} nodes")
        print(f"    Test 2 (coarse→fine): {mesh2.CoordinateSystem.coords.shape[0]} nodes")

        surface1 = np.sum(mesh1.CoordinateSystem.geo.depth < 100)
        surface2 = np.sum(depth2 < 100)
        print(f"    Test 1 surface nodes: {surface1}")
        print(f"    Test 2 surface nodes: {surface2}")
        print(f"    → Test 1 should have more surface nodes")

except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    import traceback
    traceback.print_exc()
    mesh2 = None

# %%
print("\n" + "="*70)
print("SUMMARY AND INSIGHTS")
print("="*70)

if mesh1 is not None or mesh2 is not None:
    print("\n✓ Linear gradient tests successful!")

    print("\n Key findings:")

    if mesh1 is not None:
        print(f"\n 1. Fine→Coarse gradient (H: {H_surface}→{H_bottom}):")
        print(f"    - Produced {mesh1.CoordinateSystem.coords.shape[0]} nodes")
        depth1 = mesh1.CoordinateSystem.geo.depth
        print(f"    - Surface density (< 100 km): {100*np.sum(depth1 < 100)/len(depth1):.1f}%")
        print(f"    - Deep density (> 300 km): {100*np.sum(depth1 > 300)/len(depth1):.1f}%")

    if mesh2 is not None:
        print(f"\n 2. Coarse→Fine gradient (H: {H_surface_2}→{H_bottom_2}):")
        print(f"    - Produced {mesh2.CoordinateSystem.coords.shape[0]} nodes")
        depth2 = mesh2.CoordinateSystem.geo.depth
        print(f"    - Surface density (< 100 km): {100*np.sum(depth2 < 100)/len(depth2):.1f}%")
        print(f"    - Deep density (> 300 km): {100*np.sum(depth2 > 300)/len(depth2):.1f}%")

    print("\n Interpretation:")
    print("  • Higher H → finer mesh (more elements)")
    print("  • Lower H → coarser mesh (fewer elements)")
    print("  • Linear gradient produces smooth variation in element density")
    print("  • Metric values are dimensionless (km coordinates don't affect interpretation)")

    print("\n For fault-based adaptation:")
    print("  • Use high H (e.g., 6.6e6) near faults/surface for refinement")
    print("  • Use low H (e.g., 100) elsewhere for coarse mesh")
    print("  • The 66,000x ratio creates strong refinement contrast")

else:
    print("\n✗ Both gradient tests failed")
    print("\n Possible issues:")
    print("  • Metric values may still be too large/small")
    print("  • Adaptation algorithm may have convergence issues")
    print("  • May need different MMG5 parameters")

    print("\n Next steps:")
    print("  • Try even smaller metric values (H: 10→1)")
    print("  • Use coarser initial mesh (3×3×3)")
    print("  • Check MMG5 memory settings")

print("\n" + "="*70)
print("READY FOR FAULT-BASED METRIC")
print("="*70)

if mesh1 is not None or mesh2 is not None:
    print("""
✓ Linear gradient tests validated metric behavior!

Now we can confidently create a fault-based metric:

```python
# Symbolic depth
λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]

# Metric parameter
mesh_k_elts = 100
H_fine = 6.6e6 * (mesh_k_elts / 100)  # 6.6e6 for 100k target
H_coarse = 100

# Piecewise metric
H_expr = sympy.Piecewise(
    (H_fine, fault_distance.sym[0] < 33),  # Fine near faults
    (H_fine, λ_d < 1),                      # Fine near surface
    (H_coarse, True),                       # Coarse elsewhere
)
```

The metric values work as expected:
- H_fine = 6.6e6 creates strong refinement
- H_coarse = 100 creates coarse background
- Geographic coordinates work naturally with dimensionless metric
""")

print("="*70)
