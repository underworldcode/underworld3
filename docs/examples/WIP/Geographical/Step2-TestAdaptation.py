#!/usr/bin/env python3
"""
Step 2: Test Mesh Adaptation with Constant Metric

Following the user's suggestion: "One way to work out what adaptation does is to use it
with a constant value in the metric to see if it makes uniform meshes of a different density."

This will help us understand how the metric values relate to element count with geographic meshes.
"""

# %%
import numpy as np
import underworld3 as uw
import sys
sys.path.append('.')

# %%
# Load project parameters if available
try:
    import project_variables
    expt_extent = project_variables.expt_extent
    mesh_depth_extent = project_variables.mesh_depth_extent
    print("Loaded project_variables successfully")
except ImportError:
    print("project_variables not found, using defaults")
    expt_extent = [135.0, 140.0, -35.0, -30.0]
    mesh_depth_extent = [0.0, 400.0]

# Start with coarse mesh for testing
grid_resolution = [10, 10, 10]

print(f"\nRegion: Lon [{expt_extent[0]}, {expt_extent[1]}], Lat [{expt_extent[2]}, {expt_extent[3]}]")
print(f"Depth: {mesh_depth_extent[0]} - {mesh_depth_extent[1]} km")

# %%
print("\n" + "="*70)
print("CREATING BASE MESH")
print("="*70)

# Create base mesh - coarse for fast testing
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
print(f"  Resolution: {grid_resolution}")

# %%
print("\n" + "="*70)
print("UNDERSTANDING MESH METRIC")
print("="*70)

print("""
The mesh metric H specifies the target element size at each point.
The mesh adaptation algorithm creates elements with sizes close to H.

Key question: What are the units of H with a geographic mesh?

Testing strategy:
1. Test uniform constant values (H = 1e6, 1e7, etc.)
2. Compare resulting element counts
3. Determine the relationship between H and element count

From the original workflow:
- mesh_adaptation_parameter = 6.6e6 * (mesh_k_elts/100)
- With mesh_k_elts=100: H_fine = 6.6e6, H_coarse = 100
- This is likely DIMENSIONLESS despite geographic coordinates being in km
""")

# %%
print("\n" + "="*70)
print("TEST 1: UNIFORM COARSE MESH (H = 100)")
print("="*70)

# Create mesh variables for adaptation
H1 = uw.discretisation.MeshVariable("H1", mesh, 1)
Metric1 = uw.discretisation.MeshVariable("M1", mesh, 1, degree=1)

# Set uniform coarse metric - start with small value to avoid memory issues
H1.array[:] = 100.0

print(f"✓ Metric H1 set to uniform value: {np.asarray(H1.array).mean():.2e}")
print(f"  Min: {np.asarray(H1.array).min():.2e}")
print(f"  Max: {np.asarray(H1.array).max():.2e}")

# Adapt mesh
print("\nAdapting mesh...")
try:
    icoord1, mesh1 = uw.adaptivity.mesh_adapt_meshVar(mesh, H1, Metric1)

    print(f"\n✓ Adaptation successful!")
    print(f"  Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"  Adapted nodes:  {mesh1.CoordinateSystem.coords.shape[0]}")
    print(f"  Node ratio:     {mesh1.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")
except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    mesh1 = None

# %%
print("\n" + "="*70)
print("TEST 2: UNIFORM MEDIUM MESH (H = 1000)")
print("="*70)

H2 = uw.discretisation.MeshVariable("H2", mesh, 1)
Metric2 = uw.discretisation.MeshVariable("M2", mesh, 1, degree=1)

# Set uniform medium metric (10x larger than Test 1)
H2.array[:] = 1000.0

print(f"✓ Metric H2 set to uniform value: {np.asarray(H2.array).mean():.2e}")

print("\nAdapting mesh...")
try:
    icoord2, mesh2 = uw.adaptivity.mesh_adapt_meshVar(mesh, H2, Metric2)

    print(f"\n✓ Adaptation successful!")
    print(f"  Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"  Adapted nodes:  {mesh2.CoordinateSystem.coords.shape[0]}")
    print(f"  Node ratio:     {mesh2.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")

    if mesh1 is not None:
        print(f"\n  Comparison with Test 1:")
        print(f"    Test 1 (H=1e6): {mesh1.CoordinateSystem.coords.shape[0]} nodes")
        print(f"    Test 2 (H=1e7): {mesh2.CoordinateSystem.coords.shape[0]} nodes")
        print(f"    Ratio: {mesh2.CoordinateSystem.coords.shape[0] / mesh1.CoordinateSystem.coords.shape[0]:.2f}x")
except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    mesh2 = None

# %%
print("\n" + "="*70)
print("TEST 3: UNIFORM FINE MESH (H = 10000)")
print("="*70)

H3 = uw.discretisation.MeshVariable("H3", mesh, 1)
Metric3 = uw.discretisation.MeshVariable("M3", mesh, 1, degree=1)

# Set uniform fine metric
H3.array[:] = 10000.0

print(f"✓ Metric H3 set to uniform value: {np.asarray(H3.array).mean():.2e}")

print("\nAdapting mesh...")
try:
    icoord3, mesh3 = uw.adaptivity.mesh_adapt_meshVar(mesh, H3, Metric3)

    print(f"\n✓ Adaptation successful!")
    print(f"  Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"  Adapted nodes:  {mesh3.CoordinateSystem.coords.shape[0]}")
    print(f"  Node ratio:     {mesh3.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")
except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    mesh3 = None

# %%
print("\n" + "="*70)
print("TEST 4: PHYSICAL INTERPRETATION (H = 10 km)")
print("="*70)

print("""
What if we interpret H as physical element size in km?
Domain: ~5° × 5° × 400 km ≈ (550 km)² × 400 km
Let's try H = 10 km (target element size)
""")

H4 = uw.discretisation.MeshVariable("H4", mesh, 1)
Metric4 = uw.discretisation.MeshVariable("M4", mesh, 1, degree=1)

H4.array[:] = 10.0  # 10 km elements

print(f"✓ Metric H4 set to: {float(H4.array[0, 0])} km")

print("\nAdapting mesh...")
try:
    icoord4, mesh4 = uw.adaptivity.mesh_adapt_meshVar(mesh, H4, Metric4)

    print(f"\n✓ Adaptation successful!")
    print(f"  Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"  Adapted nodes:  {mesh4.CoordinateSystem.coords.shape[0]}")
    print(f"  Node ratio:     {mesh4.CoordinateSystem.coords.shape[0] / mesh.CoordinateSystem.coords.shape[0]:.2f}x")
except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    mesh4 = None

# %%
print("\n" + "="*70)
print("TEST 5: SPATIALLY VARYING METRIC")
print("="*70)

print("""
Test with spatially varying metric: fine in upper half, coarse in lower half.
This verifies that adaptation can create non-uniform meshes.
""")

H5 = uw.discretisation.MeshVariable("H5", mesh, 1)
Metric5 = uw.discretisation.MeshVariable("M5", mesh, 1, degree=1)

# Get depth array
depth = mesh.CoordinateSystem.geo.depth

# Fine mesh in upper 200 km, coarse in lower 200 km
H_fine = 5000.0
H_coarse = 500.0

H5_values = np.where(depth < 200, H_fine, H_coarse)
H5.array[:, 0] = H5_values

print(f"✓ Spatially varying metric set:")
print(f"  Upper half (depth < 200 km): H = {H_fine:.2e}")
print(f"  Lower half (depth ≥ 200 km): H = {H_coarse:.2e}")
print(f"  Actual min: {np.asarray(H5.array).min():.2e}")
print(f"  Actual max: {np.asarray(H5.array).max():.2e}")

print("\nAdapting mesh...")
try:
    icoord5, mesh5 = uw.adaptivity.mesh_adapt_meshVar(mesh, H5, Metric5)

    print(f"\n✓ Adaptation successful!")
    print(f"  Original nodes: {mesh.CoordinateSystem.coords.shape[0]}")
    print(f"  Adapted nodes:  {mesh5.CoordinateSystem.coords.shape[0]}")

    # Check depth distribution
    depth5 = mesh5.CoordinateSystem.geo.depth
    upper_nodes = np.sum(depth5 < 200)
    lower_nodes = np.sum(depth5 >= 200)

    print(f"\n  Depth distribution:")
    print(f"    Upper half: {upper_nodes} nodes ({100*upper_nodes/len(depth5):.1f}%)")
    print(f"    Lower half: {lower_nodes} nodes ({100*lower_nodes/len(depth5):.1f}%)")
    print(f"    Ratio:      {upper_nodes/lower_nodes:.2f}x (should be ~{H_fine/H_coarse:.1f}x if linear)")
except Exception as e:
    print(f"\n✗ Adaptation failed: {e}")
    mesh5 = None

# %%
print("\n" + "="*70)
print("SUMMARY OF ADAPTATION TESTS")
print("="*70)

results = []
if mesh1: results.append(("100 (coarse)", mesh1.CoordinateSystem.coords.shape[0]))
if mesh2: results.append(("1000 (medium)", mesh2.CoordinateSystem.coords.shape[0]))
if mesh3: results.append(("10000 (fine)", mesh3.CoordinateSystem.coords.shape[0]))
if mesh4: results.append(("10 km (physical)", mesh4.CoordinateSystem.coords.shape[0]))

if results:
    print("\nElement count vs metric value:")
    print(f"{'Metric H':<20} {'Nodes':<10} {'vs 100':<10}")
    print("-" * 40)

    baseline = results[0][1] if results else 1
    for name, count in results:
        ratio = count / baseline
        print(f"{name:<20} {count:<10} {ratio:>8.2f}x")

    print("\nKey observations:")
    if len(results) >= 2:
        # Check relationship between H and element count
        h_ratio = 1000.0 / 100.0  # 10x
        count_ratio = results[1][1] / results[0][1]
        print(f"  • H increased {h_ratio}x → nodes increased {count_ratio:.2f}x")

        if count_ratio > 5:
            print(f"  • Relationship appears roughly linear or superlinear")
            print(f"  • Higher H → more elements (confirms H is not element size)")
        else:
            print(f"  • Relationship is sublinear")

    print(f"\n  • The metric H is likely DIMENSIONLESS")
    print(f"  • Geographic coordinates (km) don't change the metric interpretation")
    print(f"  • H controls element density: larger H → more elements")
    print(f"  • Original values (6.6e6 fine, 100 coarse) make sense as dimensionless")
else:
    print("\n✗ No successful adaptations to compare")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
Now that we understand how the metric works with geographic meshes:
1. Load fault surface data
2. Calculate distance to faults
3. Set metric based on fault distance (fine near faults, coarse elsewhere)
4. Adapt mesh to resolve fault structures

The metric will use dimensionless values as before:
- Near faults: H = 6.6e6 * (mesh_k_elts/100)
- Away from faults: H = 100
- Conditions use physical coordinates (fault_distance < 33 km, depth < 1 km)
""")

print("="*70)
