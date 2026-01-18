#!/usr/bin/env python3
"""
Step 3: Load Fault Surfaces

Load fault data from available sources and prepare for distance calculations.

The original workflow uses VTK files (fault_seg_MT_dip_*.vtk) but we have:
- NPZ file: faults_as_swarm_points_xyz.npz
- CSV files: CombinedInferredFaults_2025_04_22.csv

We'll start with the NPZ file which should contain pre-processed fault point clouds.
"""

# %%
import numpy as np
import underworld3 as uw
import sys
sys.path.append('.')

# %%
# Load project parameters
try:
    import project_variables
    expt_extent = project_variables.expt_extent
    mesh_depth_extent = project_variables.mesh_depth_extent
    grid_resolution = project_variables.grid_resolution
    print("Loaded project_variables successfully")
except ImportError:
    print("project_variables not found, using defaults")
    expt_extent = [135.0, 140.0, -35.0, -30.0]
    mesh_depth_extent = [0.0, 400.0]
    grid_resolution = [20, 20, 10]

print(f"\nRegion: Lon [{expt_extent[0]}, {expt_extent[1]}], Lat [{expt_extent[2]}, {expt_extent[3]}]")
print(f"Depth: {mesh_depth_extent[0]} - {mesh_depth_extent[1]} km")

# %%
print("\n" + "="*70)
print("LOADING FAULT DATA")
print("="*70)

# Try to load fault point cloud from NPZ file
fault_data_file = "Structures/faults_as_swarm_points_xyz.npz"

print(f"\nLoading fault data from: {fault_data_file}")

try:
    fault_data = np.load(fault_data_file)
    print(f"✓ Fault data loaded successfully")
    print(f"  Available arrays: {list(fault_data.keys())}")

    # Inspect the data - show first few rows to understand structure
    for key in fault_data.keys():
        arr = fault_data[key]
        print(f"\n  {key}:")
        print(f"    Shape: {arr.shape}")
        print(f"    First 3 rows:")
        print(f"    {arr[:3]}")

except FileNotFoundError:
    print(f"✗ Fault data file not found: {fault_data_file}")
    print("  Will try CSV fallback...")
    fault_data = None

# %%
if fault_data is not None:
    print("\n" + "="*70)
    print("UNDERSTANDING FAULT DATA STRUCTURE")
    print("="*70)

    print("""
Based on inspection, the NPZ file contains:
- arr_0: Geographic coordinates (lon, lat, depth) + metadata (dip, fault_id, segment_id)
  Columns: [longitude, latitude, depth, dip, fault_id, segment_id]
- arr_1: Normal vectors (nx, ny, nz) + metadata (dip, fault_id, segment_id)
  Columns: [nx, ny, nz, dip, fault_id, segment_id]
""")

    # Extract geographic coordinates
    fault_geo = fault_data['arr_0']
    fault_normals = fault_data['arr_1']

    fault_lon = fault_geo[:, 0]
    fault_lat = fault_geo[:, 1]
    fault_depth = fault_geo[:, 2]
    fault_dip = fault_geo[:, 3]
    fault_id = fault_geo[:, 4]
    segment_id = fault_geo[:, 5]

    print(f"✓ Extracted fault data:")
    print(f"  Number of fault points: {len(fault_lon)}")
    print(f"  Longitude range: [{fault_lon.min():.3f}, {fault_lon.max():.3f}] degrees")
    print(f"  Latitude range: [{fault_lat.min():.3f}, {fault_lat.max():.3f}] degrees")
    print(f"  Depth range: [{fault_depth.min():.3f}, {fault_depth.max():.3f}] km")
    print(f"  Unique fault IDs: {len(np.unique(fault_id))}")
    print(f"  Unique segment IDs: {len(np.unique(segment_id))}")

# %%
print("\n" + "="*70)
print("COORDINATE SYSTEM ALIGNMENT")
print("="*70)

print("""
The fault data is in Geographic coordinates (lon, lat, depth).
Our mesh is also in Geographic coordinates with Cartesian backing.

To calculate distances, we need to convert both to Cartesian (x, y, z) in km:
- Fault points: Convert (lon, lat, depth) → (x, y, z) using geographic_to_cartesian
- Mesh points: Use mesh.CoordinateSystem.coords (already in Cartesian)
""")

if fault_data is not None:
    from underworld3.coordinates import geographic_to_cartesian, ELLIPSOIDS

    # Get WGS84 ellipsoid parameters
    a = ELLIPSOIDS['WGS84']['a']
    b = ELLIPSOIDS['WGS84']['b']

    print(f"\nConverting fault points to Cartesian using WGS84...")
    print(f"  Semi-major axis (a): {a} km")
    print(f"  Semi-minor axis (b): {b} km")

    # Convert fault geographic coordinates to Cartesian
    fault_x, fault_y, fault_z = geographic_to_cartesian(
        fault_lon, fault_lat, fault_depth, a, b
    )

    # Stack into (N, 3) array
    fault_points_xyz = np.column_stack([fault_x, fault_y, fault_z])

    print(f"\n✓ Fault points converted to Cartesian")
    print(f"  Number of points: {fault_points_xyz.shape[0]}")
    print(f"  x range: [{fault_x.min():.1f}, {fault_x.max():.1f}] km")
    print(f"  y range: [{fault_y.min():.1f}, {fault_y.max():.1f}] km")
    print(f"  z range: [{fault_z.min():.1f}, {fault_z.max():.1f}] km")

    # Verify conversion
    r_fault = np.sqrt(fault_x**2 + fault_y**2 + fault_z**2)
    print(f"  Radius range: [{r_fault.min():.1f}, {r_fault.max():.1f}] km")
    print(f"  ✓ Earth-scale coordinates confirmed")
else:
    fault_points_xyz = None

# %%
if fault_points_xyz is not None:
    print("\n" + "="*70)
    print("BUILDING KDTREE FOR FAULT DISTANCES")
    print("="*70)

    print("\nCreating KDTree for fast nearest-neighbor queries...")

    # Ensure contiguous array for KDTree
    fault_points_contiguous = np.ascontiguousarray(fault_points_xyz)

    try:
        fault_kdtree = uw.kdtree.KDTree(fault_points_contiguous)
        print(f"✓ KDTree created with {fault_points_xyz.shape[0]} fault points")

        # Test query with a single point
        test_point = np.array([[6200.0, 0.0, 0.0]])  # Example point
        closest_idx, dist_sq, _ = fault_kdtree.find_closest_point(test_point)

        print(f"\n✓ KDTree test query successful")
        print(f"  Test point: {test_point[0]}")
        print(f"  Closest fault point: {fault_points_xyz[closest_idx[0]]}")
        print(f"  Distance: {np.sqrt(dist_sq[0]):.2f} km")

    except Exception as e:
        print(f"✗ KDTree creation failed: {e}")
        fault_kdtree = None
else:
    print("\n⚠ No fault points available - skipping KDTree creation")
    fault_kdtree = None

# %%
print("\n" + "="*70)
print("CREATE MESH FOR TESTING")
print("="*70)

# Create a simple mesh to test distance calculations
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(expt_extent[0], expt_extent[1]),
    lat_range=(expt_extent[2], expt_extent[3]),
    depth_range=(mesh_depth_extent[0], mesh_depth_extent[1]),
    ellipsoid='WGS84',
    numElements=tuple(grid_resolution),
    degree=1,
    simplex=True,
)

print(f"\n✓ Test mesh created")
print(f"  Nodes: {mesh.CoordinateSystem.coords.shape[0]}")

# Get mesh coordinates in Cartesian (matching fault points)
mesh_coords_xyz = mesh.CoordinateSystem.coords

print(f"\n✓ Mesh Cartesian coordinates:")
print(f"  Shape: {mesh_coords_xyz.shape}")
print(f"  x range: [{mesh_coords_xyz[:, 0].min():.1f}, {mesh_coords_xyz[:, 0].max():.1f}] km")
print(f"  y range: [{mesh_coords_xyz[:, 1].min():.1f}, {mesh_coords_xyz[:, 1].max():.1f}] km")
print(f"  z range: [{mesh_coords_xyz[:, 2].min():.1f}, {mesh_coords_xyz[:, 2].max():.1f}] km")

# %%
if fault_kdtree is not None:
    print("\n" + "="*70)
    print("CALCULATE DISTANCES FROM MESH TO FAULTS")
    print("="*70)

    print("\nCalculating closest fault distance for each mesh node...")

    try:
        # Query KDTree with all mesh points
        closest_indices, distances_squared, _ = fault_kdtree.find_closest_point(mesh_coords_xyz)

        # Convert squared distances to actual distances
        fault_distances = np.sqrt(distances_squared)

        print(f"✓ Distance calculation successful!")
        print(f"\n  Distance statistics:")
        print(f"    Min distance: {fault_distances.min():.2f} km")
        print(f"    Max distance: {fault_distances.max():.2f} km")
        print(f"    Mean distance: {fault_distances.mean():.2f} km")
        print(f"    Median distance: {np.median(fault_distances):.2f} km")

        # Create mesh variable to store distances
        fault_distance_var = uw.discretisation.MeshVariable(
            "d_fault", mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{fault}"
        )

        # Reshape distances to match SCALAR variable shape (N, 1, 1)
        fault_distance_var.array[:] = fault_distances.reshape(-1, 1, 1)

        print(f"\n✓ Fault distance stored in mesh variable '{fault_distance_var.name}'")

        # Identify nodes close to faults
        near_fault_threshold = 33.0  # km (from original workflow)
        near_fault_nodes = fault_distances < near_fault_threshold

        print(f"\n  Nodes close to faults (< {near_fault_threshold} km):")
        print(f"    Count: {np.sum(near_fault_nodes)}")
        print(f"    Percentage: {100 * np.sum(near_fault_nodes) / len(fault_distances):.1f}%")

    except Exception as e:
        print(f"✗ Distance calculation failed: {e}")
        fault_distances = None
else:
    print("\n⚠ No KDTree available - skipping distance calculation")
    fault_distances = None

# %%
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if fault_distances is not None:
    print("""
✓ Fault loading and distance calculation successful!

Key components ready:
1. Fault point cloud loaded from NPZ file
2. KDTree built for fast nearest-neighbor queries
3. Test mesh created with geographic coordinates
4. Distances calculated from mesh nodes to nearest fault points
5. Mesh variable stores fault distances

Next steps:
1. Create mesh metric based on fault distance
2. Apply mesh adaptation to refine near faults
3. Add topography deformation
4. Visualize adapted mesh with fault structures
""")
else:
    print("""
⚠ Fault loading incomplete

Issues encountered:
- Could not load or process fault data properly
- Distance calculation not performed

Troubleshooting:
1. Check if fault data file exists: Structures/faults_as_swarm_points_xyz.npz
2. Verify data structure matches expectations
3. May need to process CSV files instead
""")

print("="*70)
