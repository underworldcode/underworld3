# Geographic Mesh Workflow - Progress Summary

## What We've Accomplished

### ✅ Step 1: Create Geographic Mesh (COMPLETE)

**File**: `Step1-CreateMesh.py`

Successfully created a geographic mesh using the new `RegionalGeographicBox` function with:
- Natural coordinate specification: longitude (°E), latitude (°N), depth (km)
- WGS84 ellipsoid geometry (proper Earth shape)
- Multiple coordinate access methods:
  - Geographic: `mesh.CoordinateSystem.geo.lon/lat/depth` (arrays)
  - Symbolic: `λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]` (for equations)
  - Cartesian: `mesh.CoordinateSystem.coords` (for computations)
  - Spherical: `mesh.CoordinateSystem.R` (backward compatible)

**Key achievements**:
- Clear basis vector names: `unit_north`, `unit_east`, `unit_up` (no sign confusion!)
- Direct depth access in km (no manual `(1-r)*6370` conversions)
- Coordinate conversion accuracy verified to < 1 μm
- Full documentation of mesh properties and coordinate systems

### ✅ Step 2: Understand Mesh Metric (COMPLETE)

**Files**:
- `MESH_METRIC_WITH_UNITS.md` - Comprehensive analysis
- `ADAPTATION_TIMING_NOTE.md` - Practical insights
- `Step2-TestAdaptation.py` - Test scripts (adaptation timing issues encountered)

**Key insights from user feedback**:

> "Yes, the idea of the metric is that mesh refinement drives it to near unity. If the metric is 100, say, the points need to be 100 times closer together than places where it is one."

This clarifies the metric behavior:
- **Metric H drives to unity** during refinement
- **Smaller H → coarser mesh** (fewer elements)
- **Larger H → finer mesh** (more elements)
- Relationship: **H ∝ 1 / desired_spacing²** (or similar)

**From original workflow**:
- Fine mesh (near faults/surface): `H = 6.6e6 * (mesh_k_elts/100)` with `mesh_k_elts=100` → `H = 6.6e6`
- Coarse mesh (elsewhere): `H = 100`
- Ratio: `6.6e6 / 100 = 66,000x` difference

**Conclusion**:
- The metric H is **dimensionless** - not in km or physical units
- Geographic coordinates in km **don't change** the metric interpretation
- Use **original empirical values** that are calibrated to work with the adaptation algorithm
- Conditions use **physical coordinates** (fault_distance < 33 km, depth < 1 km)

### ✅ Step 3: Load Fault Surfaces and Calculate Distances (COMPLETE)

**File**: `Step3-LoadFaults.py`

Successfully loaded fault data and calculated distances to mesh nodes:

**Data source**: `Structures/faults_as_swarm_points_xyz.npz`
- Contains 874 fault points in geographic coordinates
- Two arrays:
  - `arr_0`: Geographic data (lon, lat, depth, dip, fault_id, segment_id)
  - `arr_1`: Normal vectors (nx, ny, nz, dip, fault_id, segment_id)
- 34 unique fault IDs, 10 unique segment IDs

**Processing steps**:
1. ✅ Loaded fault geographic coordinates (lon, lat, depth)
2. ✅ Converted to Cartesian (x, y, z) using `geographic_to_cartesian()` with WGS84
3. ✅ Built KDTree for fast nearest-neighbor queries
4. ✅ Created test mesh with 2338 nodes
5. ✅ Calculated distance from each mesh node to nearest fault point
6. ✅ Stored distances in mesh variable

**Results**:
- Min distance to fault: **8.35 km**
- Max distance to fault: **583.52 km**
- Mean distance: **309.01 km**
- Nodes within 33 km of faults: **43 nodes (1.8%)**

**Technical notes**:
- Fault depth values ranged from -30000 to 0 km (likely encoding issue - most at surface depth=0)
- Cartesian conversion produces Earth-scale coordinates (radius ~6371 km)
- Distance calculation uses Cartesian coordinates for both mesh and faults
- Geographic coordinates provide natural queries and conditions

---

## What Remains

### ⏳ Step 4: Create Fault-Based Mesh Metric (PENDING)

**Goal**: Create a mesh metric variable that refines the mesh near faults and surface.

**Implementation** (from original workflow):
```python
# Get symbolic depth coordinate
λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]

# Set up metric parameter
mesh_k_elts = 100  # Target ~100k elements
mesh_adaptation_parameter = 6.6e6 * (mesh_k_elts / 100)

# Create piecewise metric
import sympy
H_expr = sympy.Piecewise(
    (mesh_adaptation_parameter, fault_distance.sym[0] < 33),  # Fine near faults (< 33 km)
    (mesh_adaptation_parameter, λ_d < 1),                      # Fine near surface (< 1 km depth)
    (100, True),                                                # Coarse elsewhere
)

# Evaluate and store
import underworld3.function
H_values = underworld3.function.evaluate(H_expr, H.coords)
H.array[:, 0] = H_values.reshape(-1, 1)
```

**Key points**:
- Use dimensionless metric values (6.6e6 fine, 100 coarse)
- Conditions use physical coordinates (33 km distance, 1 km depth)
- Symbolic variables work with new geographic system

### ⏳ Step 5: Apply Mesh Adaptation (PENDING)

**Goal**: Use the fault-based metric to adapt the mesh.

```python
# Adapt mesh
icoord, adapted_mesh = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)
```

**Unknowns**:
- Will adaptation converge with fault-based metric?
- How long will it take? (uniform metrics timed out in testing)
- Will the adapted mesh have the expected refinement pattern?

**User's suggestion** was tested but timed out - may need to:
- Use smaller initial mesh
- Accept that adaptation takes time
- Trust that it works based on original workflow success

### ⏳ Step 6: Transfer Data to Adapted Mesh (PENDING)

Once adapted mesh is created, need to:
1. Create new mesh variables on adapted mesh
2. Recalculate fault distances for new node positions
3. Transfer any other data (topography, MT data, etc.)
4. Apply topography deformation to mesh

### ⏳ Step 7: Visualization (PENDING)

Create visualizations showing:
- Original vs adapted mesh
- Fault distance field
- Mesh refinement near faults
- Cross-sections through the model

---

## Key Technical Insights

### 1. Geographic vs Spherical Coordinates

**Old (Spherical)**:
```python
radius_inner = 1.0 - (mesh_depth_extent[1] / 6370)
cs_mesh = uw.meshing.RegionalSphericalBox(
    radiusOuter=1.0,
    radiusInner=radius_inner,
    ...
)
depth = (1 - uw.function.evalf(r, coords)) * 6370
```

**New (Geographic)**:
```python
mesh = uw.meshing.RegionalGeographicBox(
    depth_range=(0, 400),  # Direct km!
    ...
)
depth = mesh.CoordinateSystem.geo.depth  # Already in km!
```

**Benefits**: Natural units, no conversions, clear physical meaning

### 2. Basis Vectors

**Old (Confusing)**:
```python
unit_vertical = cs_mesh.CoordinateSystem.unit_e_0
unit_SN = -cs_mesh.CoordinateSystem.unit_e_1  # Why negative??
unit_EW = cs_mesh.CoordinateSystem.unit_e_2
```

**New (Clear)**:
```python
unit_up = mesh.CoordinateSystem.geo.unit_up
unit_north = mesh.CoordinateSystem.geo.unit_north
unit_east = mesh.CoordinateSystem.geo.unit_east
```

**Benefits**: No sign confusion, multiple aliases, intuitive names

### 3. Coordinate Transformations

**Old (Manual, error-prone)**:
```python
R = uw.function.evalf(meshA.CoordinateSystem.R, meshA.data)
for node in range(meshA.data.shape[0]):
    ph1 = R[node, 2]
    th1 = R[node, 1]
    lon = 360 * ph1 / (2 * np.pi)
    lat = 90 - 360 * th1 / (2 * np.pi)  # Sign confusion!
```

**New (Direct, vectorized)**:
```python
lon = mesh.CoordinateSystem.geo.lon
lat = mesh.CoordinateSystem.geo.lat
depth = mesh.CoordinateSystem.geo.depth
```

**Benefits**: No loops, no conversions, no sign errors

### 4. Mesh Metric Interpretation

**Critical understanding**:
- Metric H is **dimensionless**
- Refinement **drives H to unity**
- Smaller H = coarser, larger H = finer
- Values are **empirically calibrated** (H=6.6e6 fine, H=100 coarse)
- Geographic coordinates **don't change this** - metric stays dimensionless
- Physical conditions work naturally (distance < 33 km, depth < 1 km)

### 5. Data Access Patterns

**Old**:
```python
with mesh.access(var):
    var.data[:, 0] = values
```

**New**:
```python
var.array[:] = values.reshape(-1, 1, 1)  # For SCALAR
var.array[:] = values.reshape(-1, 1, 3)  # For VECTOR
```

**Note**: SCALAR variables have shape `(N, 1, 1)`, not `(N,)` or `(N, 1)`

---

## Files Created

### Working Scripts
1. `Step1-CreateMesh.py` - ✅ Creates geographic mesh, explores coordinate systems
2. `Step2-TestAdaptation.py` - ⚠️ Tests uniform metrics (timing issues)
3. `Step2-SimpleAdaptTest.py` - ⚠️ Minimal adaptation test (timing issues)
4. `Step3-LoadFaults.py` - ✅ Loads faults, calculates distances

### Documentation
1. `MESH_METRIC_WITH_UNITS.md` - Comprehensive analysis of metric interpretation
2. `MIGRATION_NOTES.md` - API migration guide with before/after patterns
3. `ADAPTATION_TIMING_NOTE.md` - Notes on adaptation performance issues
4. `PROGRESS_SUMMARY.md` - This file

### Example Code
1. `Geographic-Mesh-Simple.py` - ✅ Working demonstration of new API

---

## Next Actions

### Immediate (Can do now)
1. ✅ Document current progress (this file)
2. Create Step 4: Build fault-based metric
3. Test metric creation (without adaptation)
4. Verify metric values and spatial distribution

### Short-term (Need testing)
1. Attempt mesh adaptation with fault-based metric
2. Monitor performance and convergence
3. If successful, proceed with workflow
4. If timing out, investigate:
   - Coarser initial mesh
   - Different metric values
   - MMG5 parameters

### Long-term (Full workflow)
1. Complete adaptation pipeline
2. Add topography deformation
3. Load and interpolate MT data
4. Create visualization tools
5. Document complete workflow

---

## Questions for User

1. **Adaptation timing**: The uniform metric tests timed out (>2 minutes for 5×5×5 mesh). Is this expected? Should we:
   - Use coarser initial meshes?
   - Accept long adaptation times?
   - Investigate MMG5 parameters?

2. **Fault depth values**: The fault data shows depths from -30000 to 0 km. Most points are at depth=0. Is this:
   - An encoding issue?
   - Actual surface-only faults?
   - Should we use fault segments from VTK files instead?

3. **Metric calibration**: The original values (H=6.6e6 fine, H=100 coarse) are empirically calibrated. Should we:
   - Use these exact values?
   - Test with different values?
   - Document the relationship between H and element count?

---

## Summary

We have successfully implemented the first three steps of the geographic mesh workflow:
1. ✅ Created geographic mesh with natural coordinates
2. ✅ Understood mesh metric (dimensionless, drives to unity)
3. ✅ Loaded faults and calculated distances

The foundation is solid. The new GEOGRAPHIC coordinate system works beautifully - natural units, clear naming, no sign confusion.

Next step is to create the fault-based metric and test adaptation. The metric interpretation is now clear thanks to user feedback, and we have all the necessary components (mesh, faults, distances) to proceed.

The main unknown is adaptation performance - our tests with uniform metrics timed out, suggesting the algorithm may be very sensitive to metric values or mesh size. We'll need to test carefully with the actual fault-based metric to see if it behaves better than uniform values.
