# Geographic Coordinate System - Final API

**Status**: Ready for Implementation
**Date**: 2025-10-13

## System Name: GEOGRAPHIC (not "planetary")

**Rationale**: More intuitive and guessable for Earth scientists, even though it works for any planet.

---

## Quick Reference

### Coordinate Access

```python
# Data arrays (numpy)
lon = mesh.geo.lon      # Longitude (degrees East)
lat = mesh.geo.lat      # Latitude (degrees North)
depth = mesh.geo.depth  # Depth below surface (km)

# Symbolic (for equations)
λ_lon, λ_lat, λ_d = mesh.geo[:]
```

### Basis Vectors - All Available Names

#### Longitude Direction (East/West)
```python
mesh.geo.unit_WE      # Primary (West→East)
mesh.geo.unit_east    # Directional alias
mesh.geo.unit_lon     # Coordinate alias
mesh.geo.unit_west    # Opposite direction (-unit_WE)
```

#### Latitude Direction (North/South)
```python
mesh.geo.unit_SN      # Primary (South→North)
mesh.geo.unit_north   # Directional alias
mesh.geo.unit_lat     # Coordinate alias
mesh.geo.unit_south   # Opposite direction (-unit_SN)
```

#### Depth Direction (Up/Down)
```python
mesh.geo.unit_down    # Primary (into planet)
mesh.geo.unit_depth   # Coordinate alias
mesh.geo.unit_up      # Opposite direction (-unit_down)
```

**Right-handed system**: WE × SN = down ✓

---

## Mesh Creation

```python
# Regional ellipsoidal mesh
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(135, 140),      # Degrees East
    lat_range=(-35, -30),      # Degrees North
    depth_range=(0, 400),      # km below surface
    ellipsoid='WGS84',         # or True, or 'Mars', or (a, b)
    numElements=(10, 10, 10),
    degree=1,
    simplex=True,
)

# Backward compatible (spherical)
mesh = uw.meshing.RegionalSphericalBox(
    SWcorner=[135, -35],
    NEcorner=[140, -30],
    radiusOuter=1.0,
    radiusInner=0.94,
    numElementsLon=10,
    numElementsLat=10,
    numElementsDepth=10,
)
```

---

## Usage Examples

### Basic Coordinate Access

```python
# Your current workflow (BEFORE):
R = uw.function.evalf(mesh.CoordinateSystem.R, mesh.data)
for node in range(mesh.data.shape[0]):
    ph1 = R[node, 2]
    th1 = R[node, 1]
    longitude = 360 * ph1 / (2 * np.pi)
    latitude = 90 - 360 * th1 / (2 * np.pi)  # Sign confusion!

# New API (AFTER):
lon = mesh.geo.lon
lat = mesh.geo.lat
depth = mesh.geo.depth
# Done! No conversions needed
```

### Symbolic Equations

```python
# Temperature decreasing with depth
λ_lon, λ_lat, λ_d = mesh.geo[:]
T = 1600 - 0.5 * λ_d

# Lateral temperature gradient
dT_dlon = T.diff(λ_lon)
grad_T_lon = dT_dlon * mesh.geo.unit_lon
```

### Boundary Conditions

```python
# Surface: no penetration (perpendicular to surface)
v_surface = 0 * mesh.geo.unit_up

# Bottom: fixed downward velocity
v_bottom = 10 * mesh.geo.unit_down  # or: -10 * mesh.geo.unit_up

# Side: horizontal flow
v_side = v_east * mesh.geo.unit_east + v_north * mesh.geo.unit_north
```

### Forces and Fields

```python
# Gravitational acceleration (downward)
g_vec = 9.81 * mesh.geo.unit_down

# Buoyancy force (upward)
F_buoyancy = rho * g * mesh.geo.unit_up

# Horizontal flow field
v_horizontal = (
    v_east * mesh.geo.unit_WE +
    v_north * mesh.geo.unit_SN
)
```

---

## Implementation Status

### ✅ Completed
- Enum: `CoordinateSystemType.GEOGRAPHIC`
- Ellipsoid parameters: `ELLIPSOIDS` dictionary
- Design documents with all naming conventions
- API specification

### 🔨 To Implement
1. GEOGRAPHIC case in `CoordinateSystem.__init__()`
2. Coordinate conversion functions
3. `GeographicCoordinateAccessor` class
4. `RegionalGeographicBox()` mesh function
5. Testing and validation

---

## Key Design Decisions

1. **Name**: "Geographic" (not "planetary") - more guessable
2. **Basis vectors**: `unit_WE`, `unit_SN`, `unit_down` with multiple aliases
3. **Right-handed**: WE × SN = down ✓
4. **Depth reference**: Reference ellipsoid surface (depth=0 at surface)
5. **Latitude**: Geodetic (matches GPS, not geocentric)
6. **Ellipsoid**: `ellipsoid=True` defaults to WGS84
7. **Works for any planet**: Despite name, supports Mars/Moon/Venus via ellipsoid parameter

---

## Naming Philosophy

**Primary names** (`unit_WE`, `unit_SN`, `unit_down`):
- Explicit about direction and positivity
- Unambiguous in formal documentation
- Right-handed system verification

**Directional aliases** (`unit_east`, `unit_north`, `unit_depth`):
- Natural language in physics code
- Clearer intent in equations
- Recommended for most users

**Coordinate aliases** (`unit_lon`, `unit_lat`):
- For gradient/derivative contexts
- Matches mathematical notation
- Parallel with spherical coordinates

**Opposite directions** (`unit_west`, `unit_south`, `unit_up`):
- Convenience (avoid explicit negation)
- Clearer physical meaning
- Matches physics conventions (upward buoyancy)

**Use what makes your code clearest!** All are equivalent.

---

## Comparison: Old vs New

### Your Current Workflow (from Mesh-Adapted-2-Faults.py)

**BEFORE**:
```python
# Manual conversions everywhere
R = uw.function.evalf(meshA.CoordinateSystem.R, meshA.data)
for node in range(meshA.data.shape[0]):
    ph1 = R[node, 2]
    th1 = R[node, 1]
    topoA.data[node, 0] = ep_topo_value(
        360 * ph1 / (2 * np.pi),
        90 - 360 * th1 / (2 * np.pi)
    )

# Confusing basis vectors
unit_vertical = cs_mesh.CoordinateSystem.unit_e_0
unit_SN = -cs_mesh.CoordinateSystem.unit_e_1  # Why negative??
unit_EW = cs_mesh.CoordinateSystem.unit_e_2

# Hardcoded Earth radius
radius_inner = 1.0 - (mesh_depth_extent[1] / 6370)
depth = (1 - uw.function.evalf(r, coords)) * 6370
```

**AFTER**:
```python
# Direct coordinate access
lon, lat, depth = mesh.geo.lon, mesh.geo.lat, mesh.geo.depth
topo_values = ep_topo_value(lon, lat)
topoA.data[:, 0] = topo_values

# Clear basis vectors (no sign confusion!)
unit_up = mesh.geo.unit_up       # or unit_down
unit_north = mesh.geo.unit_north
unit_east = mesh.geo.unit_east

# Natural depth specification in mesh creation
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(135, 140),
    lat_range=(-35, -30),
    depth_range=(0, 400),  # km below surface!
    ellipsoid='WGS84',
)
```

**Benefits**:
- ✅ No manual conversions
- ✅ No sign confusion
- ✅ Natural units (degrees, km)
- ✅ Proper ellipsoid geometry
- ✅ Clear basis vectors
- ✅ Cleaner code

---

## Ready to Code!

The design is finalized. Let's implement the GEOGRAPHIC coordinate system.
