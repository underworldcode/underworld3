# Migration Notes for Mesh-Adapted-2-Faults.py

## Summary of Changes Needed

Your original workflow in `Mesh-Adapted-2-Faults.py` needs several updates to work with the current UW3 API:

1. **New GEOGRAPHIC coordinate system** (just implemented!)
2. **Data access API changes** (from `with mesh.access()` to direct array access)
3. **Function evaluation changes** (`uw.function.evalf` → `underworld3.function.evaluate`)
4. **Coordinate access simplification** (direct lon/lat/depth instead of manual conversions)

---

## Key Updates Needed

### 1. Mesh Creation (Lines 100-114)

**Current (OLD)**:
```python
radius_outer = 1.0
radius_inner = 1.0 - (mesh_depth_extent[1] / 6370)

cs_mesh = uw.meshing.RegionalSphericalBox(
    SWcorner=[expt_extent[0], expt_extent[2]],
    NEcorner=[expt_extent[1], expt_extent[3]],
    radiusOuter=radius_outer,
    radiusInner=radius_inner,
    numElementsLon=grid_resolution[0],
    numElementsLat=grid_resolution[1],
    numElementsDepth=grid_resolution[2],
    simplex=True,
)
```

**Recommended (NEW)**:
```python
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(expt_extent[0], expt_extent[1]),  # Direct degrees East
    lat_range=(expt_extent[2], expt_extent[3]),  # Direct degrees North
    depth_range=(mesh_depth_extent[0], mesh_depth_extent[1]),  # Direct km
    ellipsoid='WGS84',  # Proper ellipsoid geometry
    numElements=tuple(grid_resolution),
    degree=1,
    simplex=True,
)
```

**Benefits**:
- Natural coordinate specification
- No manual radius calculations
- Proper WGS84 ellipsoid geometry
- Works anywhere on Earth

---

### 2. Basis Vectors (Lines 121-124)

**Current (OLD)**:
```python
unit_vertical = cs_mesh.CoordinateSystem.unit_e_0
unit_SN = -cs_mesh.CoordinateSystem.unit_e_1  # Sign confusion!
unit_EW = cs_mesh.CoordinateSystem.unit_e_2
```

**Recommended (NEW)**:
```python
# Clear, intuitive names - no sign confusion!
unit_up = mesh.CoordinateSystem.geo.unit_up
unit_down = mesh.CoordinateSystem.geo.unit_down
unit_north = mesh.CoordinateSystem.geo.unit_north
unit_south = mesh.CoordinateSystem.geo.unit_south
unit_east = mesh.CoordinateSystem.geo.unit_east
unit_west = mesh.CoordinateSystem.geo.unit_west
```

**Benefits**:
- Clear naming (no more wondering about signs!)
- Multiple aliases available (unit_WE, unit_SN, unit_down as primaries)
- Matches natural geographic directions

---

### 3. Coordinate Access for Topography (Lines 236-244)

**Current (OLD)** - Manual conversion with loops:
```python
with meshA.access(topoA):
    R = uw.function.evalf(meshA.CoordinateSystem.R, meshA.data)

    for node in range(meshA.data.shape[0]):
        ph1 = R[node, 2]
        th1 = R[node, 1]
        topoA.data[node, 0] = ep_topo_value(
            360 * ph1 / (2 * np.pi), 90 - 360 * th1 / (2 * np.pi)
        )
```

**Recommended (NEW)** - Direct geographic coordinates:
```python
# Get geographic coordinates directly - no conversions needed!
lon = meshA.CoordinateSystem.geo.lon
lat = meshA.CoordinateSystem.geo.lat

# Vectorized topography lookup
topo_values = ep_topo_value(lon, lat)  # ep_topo_value needs to handle arrays
topoA.array[:] = topo_values.reshape(-1, 1, 1)
```

**If ep_topo_value doesn't handle arrays**, use:
```python
lon = meshA.CoordinateSystem.geo.lon
lat = meshA.CoordinateSystem.geo.lat

# Loop is simpler now
for i in range(len(lon)):
    topoA.array[i, 0, 0] = ep_topo_value(lon[i], lat[i])
```

**Benefits**:
- No manual coordinate conversions
- No sign confusion (latitude is already North positive)
- Cleaner, more readable code
- Can vectorize if ep_topo_value supports it

---

### 4. Data Access Pattern (Lines 174-183, 196-207, 236-244, 266-283, 361-367)

**Current (OLD)** - Using `with mesh.access()`:
```python
with cs_mesh.access(fault_distance):
    fault_distance.data[:, 0] = 1e10
    # ... operations ...
```

**Recommended (NEW)** - Direct array access:
```python
# Single variable - direct access
fault_distance.array[:, 0] = 1e10
# ... operations ...
```

**For multiple variables** - Use synchronised update:
```python
with uw.synchronised_array_update():
    fault_distanceA.array[:, 0] = 1e10
    fault_normalsA.array[...] = segment_cells_array[closest_points, 3:6]
```

**Benefits**:
- Simpler syntax
- No need to list variables in `access()`
- Automatic PETSc synchronization
- Works with MPI

---

### 5. Function Evaluation (Lines 186, 197, 246, 319-336)

**Current (OLD)**:
```python
depth = (1 - uw.function.evalf(r, fault_distance.coords)) * 6370

H.data[:, 0] = uw.function.evalf(
    sympy.Piecewise(...),
    H.coords,
)

normal = uw.function.evalf(unit_EW, pvmeshA.center_of_mass().reshape(1, 3))
```

**Recommended (NEW)**:
```python
import underworld3.function

# For depth - use direct geographic coordinates instead
depth = mesh.CoordinateSystem.geo.depth  # Already in km!

# For expressions
H_values = underworld3.function.evaluate(
    sympy.Piecewise(...),
    H.coords,
)
H.array[:, 0] = H_values.reshape(-1, 1)

# For basis vectors
normal = underworld3.function.evaluate(unit_EW, pvmeshA.center_of_mass().reshape(1, 3))
```

**Benefits**:
- Depth is directly available in km
- Consistent function evaluation API
- No hardcoded Earth radius (6370 km)

---

### 6. Depth Calculations (Line 186, 200)

**Current (OLD)**:
```python
depth = (1 - uw.function.evalf(r, fault_distance.coords)) * 6370

# In piecewise expression:
((1 - r) * 6370) < 1  # depth < 1 km
```

**Recommended (NEW)**:
```python
# Direct depth access
depth = mesh.CoordinateSystem.geo.depth  # Already in km below surface!

# In piecewise expression - use symbolic depth
λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]
# Then use λ_d in your expressions
sympy.Piecewise(
    (mesh_adapation_parameter, λ_d < 1),  # depth < 1 km - much clearer!
    (100, True),
)
```

**Benefits**:
- No hardcoded Earth radius
- Works with ellipsoid (not just sphere)
- Clearer intent
- Natural units (km, not normalized radius)

---

### 7. MT Data Coordinate Transformation (Lines 177-190)

**Current (OLD)** - Manual spherical conversion:
```python
mt_arr_rtp[:, 0] = 1 + mt_arr[:, 2] / 6370000  # radius
mt_arr_rtp[:, 1] = np.radians(lats)  # colatitude
mt_arr_rtp[:, 2] = np.radians(lons)  # azimuth

# Then convert to Cartesian manually
mt_arr_xyz[:, 0] = (
    mt_arr_rtp[:, 0] * np.cos(mt_arr_rtp[:, 1]) * np.cos(mt_arr_rtp[:, 2])
)
mt_arr_xyz[:, 1] = (
    mt_arr_rtp[:, 0] * np.cos(mt_arr_rtp[:, 1]) * np.sin(mt_arr_rtp[:, 2])
)
mt_arr_xyz[:, 2] = mt_arr_rtp[:, 0] * np.sin(mt_arr_rtp[:, 1])
```

**Recommended (NEW)** - Use geographic conversion:
```python
from underworld3.coordinates import geographic_to_cartesian, ELLIPSOIDS

# MT data in geographic coordinates
mt_lons = lons  # degrees East
mt_lats = lats  # degrees North (already geodetic from pyproj)
mt_depths = -1.0e-3 * mt_arr[:, 2]  # Convert to km below surface

# Convert to Cartesian using proper ellipsoid
a = ELLIPSOIDS['WGS84']['a']
b = ELLIPSOIDS['WGS84']['b']
mt_arr_xyz[:, 0], mt_arr_xyz[:, 1], mt_arr_xyz[:, 2] = \
    geographic_to_cartesian(mt_lons, mt_lats, mt_depths, a, b)
```

**Benefits**:
- Proper geodetic latitude (matches pyproj output)
- Consistent with mesh geometry
- No hardcoded Earth radius
- Works with ellipsoid

---

## Migration Strategy

### Option 1: Gradual Migration (Safer)
Keep `RegionalSphericalBox` for now, but update:
1. ✅ Data access patterns (`mesh.access()` → direct array access)
2. ✅ Function evaluation (`evalf` → `evaluate`)
3. ✅ Use `mesh.data` → `mesh.CoordinateSystem.coords`

### Option 2: Full Migration (Recommended for new work)
Switch to `RegionalGeographicBox` and update everything:
1. ✅ Mesh creation
2. ✅ Basis vectors
3. ✅ Coordinate access
4. ✅ Data access patterns
5. ✅ Function evaluation
6. ✅ MT data transformation

---

## API Changes Summary

| Old Pattern | New Pattern | Notes |
|-------------|-------------|-------|
| `with mesh.access(var):` | `var.array[...]` or `with uw.synchronised_array_update():` | Direct access for single var, context for multiple |
| `var.data` | `var.array` | Preferred interface |
| `mesh.data` | `mesh.CoordinateSystem.coords` | mesh.data is deprecated |
| `uw.function.evalf()` | `underworld3.function.evaluate()` | Module restructure |
| Manual lon/lat conversion | `mesh.CoordinateSystem.geo.lon/lat/depth` | GEOGRAPHIC system only |
| `unit_e_0`, `unit_e_1`, `unit_e_2` | `unit_up/down`, `unit_north/south`, `unit_east/west` | GEOGRAPHIC system only |
| `RegionalSphericalBox` | `RegionalGeographicBox` | New function for natural coords |

---

## What Still Works (Backward Compatible)

✅ **Spherical coordinates** still available via `mesh.CoordinateSystem.R`:
```python
r, theta, phi = mesh.CoordinateSystem.R  # Still works!
```

✅ **Old mesh function** still available:
```python
cs_mesh = uw.meshing.RegionalSphericalBox(...)  # Still works!
```

✅ **Basis vector access** via `unit_e_*`:
```python
unit_e_0 = mesh.CoordinateSystem.unit_e_0  # Still works!
```

The new GEOGRAPHIC system **adds** functionality, doesn't break existing code!

---

## Next Steps

1. **Test with simple case** - Use `Geographic-Mesh-Simple.py` as reference
2. **Update mesh creation** - Switch to `RegionalGeographicBox`
3. **Update coordinate access** - Use direct lon/lat/depth
4. **Update data access** - Remove `with mesh.access()` contexts
5. **Update function calls** - Use `underworld3.function.evaluate()`
6. **Test incrementally** - Don't change everything at once!

The example `Geographic-Mesh-Simple.py` demonstrates all the new patterns working together.
