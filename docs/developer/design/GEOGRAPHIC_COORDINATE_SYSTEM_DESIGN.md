# Geographic Coordinate System Design for Underworld3

**Status**: Draft Design for Discussion
**Date**: 2025-10-13
**Purpose**: Support ellipsoidal meshes with geographic coordinates for Earth and planetary science

---

## Executive Summary

This design adds **geographic coordinate system support** to Underworld3, enabling natural coordinate specification for Earth and planetary science applications. Users will be able to work in **(longitude, latitude, depth)** coordinates while the solver continues to use Cartesian (x,y,z) internally.

**Key Features**:
- Three coordinate views: Cartesian (solver), Spherical (math), Geographic (user)
- Ellipsoidal geometry support (WGS84, custom ellipsoids for planets)
- Automatic coordinate conversions with proper geodetic calculations
- Geographic basis vectors (unit_east, unit_north, unit_up)
- Depth handling with topography/bathymetry support

---

## Current Problems

### Problem 1: Manual Coordinate Conversions

**Current workflow** (from `Mesh-Adapted-2-Faults.py`):
```python
# Get spherical coords, then manually convert to geographic
R = uw.function.evalf(meshA.CoordinateSystem.R, meshA.data)
for node in range(meshA.data.shape[0]):
    ph1 = R[node, 2]  # phi
    th1 = R[node, 1]  # theta (colatitude)
    # Manual conversion with sign corrections
    longitude = 360 * ph1 / (2 * np.pi)
    latitude = 90 - 360 * th1 / (2 * np.pi)  # Note: 90 - theta!
    topo_value = get_topo(longitude, latitude)

# Reverse conversion for data mapping
mt_arr_rtp[:, 1] = np.radians(lats)  # lat -> theta
mt_arr_rtp[:, 2] = np.radians(lons)  # lon -> phi
# Then manual spherical -> Cartesian
mt_arr_xyz[:, 0] = r * np.cos(theta) * np.cos(phi)
mt_arr_xyz[:, 1] = r * np.cos(theta) * np.sin(phi)
mt_arr_xyz[:, 2] = r * np.sin(theta)
```

**Issues**:
- Error-prone sign conventions (colatitude vs latitude)
- Repeated boilerplate conversions
- No ellipsoid support (sphere only)
- Hardcoded Earth radius (6370 km)

### Problem 2: Depth Calculations

**Current approach**:
```python
radius_inner = 1.0 - (mesh_depth_extent[1] / 6370)
depth = (1 - uw.function.evalf(r, coords)) * 6370
```

**Issues**:
- Hardcoded Earth radius
- Assumes spherical geometry
- Unclear if depth is below surface or below reference ellipsoid

### Problem 3: Spherical Mesh for Ellipsoidal Earth

**Current**: `RegionalSphericalBox` creates perfect sphere
**Reality**: Earth is ellipsoidal (21 km difference equator to pole)
**Impact**: Map data (GPS, satellite imagery) doesn't align properly

---

## Design Overview

### Three Coordinate Systems

```
┌─────────────────────────────────────────────────────────────┐
│                     Underworld3 Mesh                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  mesh.X          Cartesian (x, y, z)                         │
│                  - Native solver coordinates                 │
│                  - What PETSc uses internally                │
│                  - Symbolic: sympy expressions               │
│                                                               │
│  mesh.R          Spherical (r, θ, φ)                         │
│                  - Mathematical spherical coords             │
│                  - r = radius from origin                    │
│                  - θ = colatitude (0 to π)                   │
│                  - φ = azimuth (-π to π)                     │
│                  - Useful for spherical harmonics, etc.      │
│                                                               │
│  mesh.geo        Geographic (lon, lat, depth)                │
│                  - User-facing geographic coords             │
│                  - lon = longitude East (degrees)            │
│                  - lat = latitude North (degrees)            │
│                  - depth = depth below surface (km)          │
│                  - Handles ellipsoid geometry                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Coordinate System Hierarchy

```python
class CoordinateSystem:
    """Base coordinate system class"""

class SphericalCoordinateSystem(CoordinateSystem):
    """Perfect sphere - current implementation"""
    # mesh.R = (r, θ, φ)

class GeographicCoordinateSystem(SphericalCoordinateSystem):
    """Ellipsoidal coordinates - new implementation"""
    # mesh.geo = (lon, lat, depth)
    # Inherits spherical but adds ellipsoid support
```

---

## API Design

### 1. Mesh Creation with Ellipsoid

#### Regional Ellipsoidal Mesh

```python
mesh = uw.meshing.RegionalEllipsoidalBox(
    # Geographic extent (degrees)
    lon_range=(135, 140),     # Or SWcorner/NEcorner for backward compat
    lat_range=(-35, -30),
    depth_range=(0, 400),     # km below surface

    # Ellipsoid specification
    ellipsoid=True,           # Shorthand for 'WGS84'
    # OR
    ellipsoid='WGS84',        # Named ellipsoid
    # OR
    ellipsoid=(6378.137, 6356.752),  # Custom (a, b) in km
    # OR
    ellipsoid='sphere',       # Perfect sphere (backward compat)

    # Mesh resolution
    numElements=(10, 10, 10),  # Or numElementsLon/Lat/Depth separately

    # Standard options
    degree=1,
    simplex=True,
)
```

**Backward compatibility option**:
```python
# Old API still works (creates spherical mesh)
mesh = uw.meshing.RegionalSphericalBox(
    SWcorner=[-45, -45],
    NEcorner=[45, 45],
    radiusOuter=1.0,
    radiusInner=0.547,
    numElementsLon=5,
    numElementsLat=5,
    numElementsDepth=5,
)
```

#### Global Ellipsoidal Mesh

```python
mesh = uw.meshing.GlobalEllipsoidalShell(
    depth_range=(0, 2890),    # km - crust to CMB
    ellipsoid='WGS84',
    numElements=(128, 64, 32),  # lon, lat, depth
    degree=1,
)
```

### 2. Coordinate Access

#### Array Access (Data)

```python
# Cartesian coordinates (always available)
xyz = mesh.X.coords           # (N, 3) array - physical units
x, y, z = mesh.X.coords.T     # Unpack

# Geographic coordinates (NEW)
geo_coords = mesh.geo.coords  # (N, 3) array of (lon, lat, depth)
lon = mesh.geo.lon            # (N,) array - degrees East
lat = mesh.geo.lat            # (N,) array - degrees North
depth = mesh.geo.depth        # (N,) array - km below surface

# Alternative access
lon, lat, depth = mesh.geo[:] # Unpack all three

# Spherical coordinates (mathematical)
rtp = mesh.R.coords           # (N, 3) array of (r, θ, φ)
r, theta, phi = mesh.R[:]     # Unpack
```

#### Symbolic Access (Equations)

```python
# For use in sympy expressions
x, y, z = mesh.X              # Symbolic Cartesian coords
r, theta, phi = mesh.R        # Symbolic spherical coords
lon, lat, depth = mesh.geo    # Symbolic geographic coords (NEW)

# Example: Define temperature field
import sympy
T = 1600 - 0.5 * depth        # Temperature decreasing with depth
                              # (depth is symbolic sympy expression)

# Example: Radial velocity component
v_r = sympy.exp(-depth / 100) * sympy.sin(lat * sympy.pi / 180)
```

### 3. Coordinate Conversions

#### Automatic Conversions

```python
# Convert external data to mesh coordinates
external_coords = np.array([
    [138.0, -33.0, 50],    # lon, lat, depth in degrees and km
    [138.5, -33.5, 100],
])

# Convert to Cartesian for mesh operations
xyz = mesh.geo.to_cartesian(external_coords)
# OR
xyz = mesh.geo.from_geographic(lon, lat, depth)

# Convert mesh coordinates to geographic
lon, lat, depth = mesh.geo.from_cartesian(mesh.X.coords)

# Convert geographic to spherical (if needed)
r, theta, phi = mesh.geo.to_spherical(lon, lat, depth)
```

#### Projection Support (with pyproj if available)

```python
# UTM to geographic
easting = np.array([500000, 550000])
northing = np.array([6200000, 6250000])
lon, lat = mesh.geo.from_utm(easting, northing, zone=53, southern=True)

# Geographic to UTM
easting, northing = mesh.geo.to_utm(lon, lat, zone=53)

# Arbitrary CRS (requires pyproj)
lon, lat = mesh.geo.from_crs(x, y, from_crs='EPSG:28353')  # MGA Zone 53
```

### 4. Basis Vectors and Directions

```python
# Planetary basis vectors - PRIMARY NAMES (canonical):
mesh.planetary.unit_WE       # West to East (longitude direction)
mesh.planetary.unit_SN       # South to North (latitude direction)
mesh.planetary.unit_down     # Into planet (depth direction, positive downward)

# DIRECTIONAL ALIASES (clearer names):
mesh.planetary.unit_east     # = unit_WE
mesh.planetary.unit_north    # = unit_SN
mesh.planetary.unit_depth    # = unit_down

# COORDINATE ALIASES (for derivatives/gradients):
mesh.planetary.unit_lon      # = unit_WE (longitude direction)
mesh.planetary.unit_lat      # = unit_SN (latitude direction)

# OPPOSITE DIRECTIONS:
mesh.planetary.unit_west     # = -unit_WE
mesh.planetary.unit_south    # = -unit_SN
mesh.planetary.unit_up       # = -unit_down (radial outward)

# Original spherical basis vectors (still available for backward compat)
mesh.CoordinateSystem.unit_e_0  # Radial
mesh.CoordinateSystem.unit_e_1  # Meridional
mesh.CoordinateSystem.unit_e_2  # Azimuthal

# Example: Horizontal flow (multiple naming options)
v_h = v_east * mesh.planetary.unit_WE + v_north * mesh.planetary.unit_SN  # Primary
v_h = v_east * mesh.planetary.unit_east + v_north * mesh.planetary.unit_north  # Clearer
v_h = v_lon * mesh.planetary.unit_lon + v_lat * mesh.planetary.unit_lat  # Coordinate-based

# Example: Vertical boundary condition
v_bc = v_magnitude * mesh.planetary.unit_down
# or
v_bc = -v_magnitude * mesh.planetary.unit_up  # Upward velocity
```

**Right-handed system**: unit_WE × unit_SN = unit_down ✓

### 5. Ellipsoid Properties

```python
# Query ellipsoid parameters
mesh.geo.ellipsoid_name   # 'WGS84' or 'sphere' or 'custom'
mesh.geo.semi_major_axis  # Equatorial radius (km)
mesh.geo.semi_minor_axis  # Polar radius (km)
mesh.geo.flattening       # (a - b) / a
mesh.geo.eccentricity     # sqrt(1 - (b/a)^2)

# For custom ellipsoids
mesh.geo.is_spherical     # True if a == b
mesh.geo.planet_name      # 'Earth', 'Mars', etc. (if set)
```

### 6. Topography and Bathymetry

```python
# Sample raster data (if gdal/rasterio available)
topo = mesh.geo.sample_raster(
    'SRTM_elevation.tif',
    method='bilinear',    # or 'nearest', 'cubic'
    output_units='km',    # Convert to km
)
# Returns MeshVariable with elevation at each node

# Deform mesh to follow topography
mesh.geo.deform_to_topography(
    topo_var,
    reference='surface',  # Deform relative to outer surface
    scale=1.0,           # Scaling factor
)

# Manual deformation (current method still works)
new_coords = mesh.X.coords.copy()
# ... apply deformation ...
mesh.deform_mesh(new_coords)
```

### 7. Distance and Area Calculations

```python
# Great circle distance (on ellipsoid)
dist = mesh.geo.distance(
    lon1, lat1,
    lon2, lat2,
    method='vincenty'  # or 'haversine' for sphere
)

# Area calculations (accounting for ellipsoid)
area = mesh.geo.surface_area()         # Total surface area
cell_areas = mesh.geo.cell_areas()     # Area of each cell

# Arc length along meridian/parallel
arc_north = mesh.geo.arc_length_meridional(lat1, lat2, lon)
arc_east = mesh.geo.arc_length_parallel(lon1, lon2, lat)
```

---

## Code Examples: Current Workflow → New API

### Example 1: Mesh Creation

**Before** (current):
```python
# Create spherical mesh with hardcoded radius
radius_outer = 1.0
radius_inner = 1.0 - (mesh_depth_extent[1] / 6370)  # Hardcoded Earth radius

mesh = uw.meshing.RegionalSphericalBox(
    SWcorner=[135, -35],
    NEcorner=[140, -30],
    radiusOuter=radius_outer,
    radiusInner=radius_inner,
    numElementsLon=10,
    numElementsLat=10,
    numElementsDepth=10,
    simplex=True,
)
```

**After** (new API):
```python
# Create ellipsoidal mesh with natural depth units
mesh = uw.meshing.RegionalEllipsoidalBox(
    lon_range=(135, 140),    # Degrees East
    lat_range=(-35, -30),    # Degrees North
    depth_range=(0, 400),    # km below surface
    ellipsoid='WGS84',       # Proper ellipsoid
    numElements=(10, 10, 10),
    simplex=True,
)
```

### Example 2: Coordinate Conversions for Data Mapping

**Before** (current):
```python
# Manual conversion from mesh to geographic
R = uw.function.evalf(mesh.CoordinateSystem.R, mesh.data)
for node in range(mesh.data.shape[0]):
    ph1 = R[node, 2]
    th1 = R[node, 1]
    longitude = 360 * ph1 / (2 * np.pi)
    latitude = 90 - 360 * th1 / (2 * np.pi)  # Watch the sign!
    topo_value = get_topo(longitude, latitude)
    topo.data[node, 0] = topo_value

# Manual conversion from external data to mesh
mt_arr_rtp[:, 0] = 1 + mt_arr[:, 2] / 6370000
mt_arr_rtp[:, 1] = np.radians(lats)
mt_arr_rtp[:, 2] = np.radians(lons)
mt_arr_xyz[:, 0] = mt_arr_rtp[:, 0] * np.cos(mt_arr_rtp[:, 1]) * np.cos(mt_arr_rtp[:, 2])
mt_arr_xyz[:, 1] = mt_arr_rtp[:, 0] * np.cos(mt_arr_rtp[:, 1]) * np.sin(mt_arr_rtp[:, 2])
mt_arr_xyz[:, 2] = mt_arr_rtp[:, 0] * np.sin(mt_arr_rtp[:, 1])
```

**After** (new API):
```python
# Direct access to geographic coordinates
lon = mesh.geo.lon
lat = mesh.geo.lat
depth = mesh.geo.depth

# Vectorized sampling (if you have a function that takes lon/lat)
topo_values = get_topo(lon, lat)
topo.data[:, 0] = topo_values

# Or use built-in raster sampling
topo = mesh.geo.sample_raster("SRTM_elevation.tif")

# Convert external data (lon, lat, depth) to Cartesian
external_data = np.column_stack([lons, lats, depths])
xyz = mesh.geo.to_cartesian(external_data)

# Or if you have separate arrays
xyz = mesh.geo.from_geographic(lon=lons, lat=lats, depth=depths)
```

### Example 3: Topography Deformation

**Before** (current):
```python
# Manual deformation with confusing scaling
delta_r = uw.function.evalf(
    topo.sym[0] * (r - radius_inner) / (radius_outer - radius_inner),
    mesh.data
)
new_coords = mesh.data.copy()
new_coords *= (radius_outer + delta_r.reshape(-1, 1) / 6370000) / radius_outer
mesh.deform_mesh(new_coords)
```

**After** (new API):
```python
# Simple topography deformation
mesh.geo.deform_to_topography(topo, reference='surface')

# Or for custom deformation with clear units
delta_depth = topo.data[:, 0]  # In km
new_depth = mesh.geo.depth - delta_depth  # Subtract because topo is positive up
new_coords = mesh.geo.from_geographic(mesh.geo.lon, mesh.geo.lat, new_depth)
mesh.deform_mesh(new_coords)
```

### Example 4: Basis Vectors

**Before** (current):
```python
# Confusing sign conventions
unit_vertical = mesh.CoordinateSystem.unit_e_0
unit_SN = -mesh.CoordinateSystem.unit_e_1  # Why negative?
unit_EW = mesh.CoordinateSystem.unit_e_2
```

**After** (new API):
```python
# Clear geographic directions
unit_up = mesh.geo.unit_up        # Radial outward
unit_north = mesh.geo.unit_north  # Meridional (no sign confusion!)
unit_east = mesh.geo.unit_east    # Azimuthal
```

### Example 5: PyProj Integration

**Before** (current):
```python
# Manual pyproj setup
from pyproj import CRS, Transformer
from_crs = CRS.from_proj4("+proj=utm +zone=53 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
to_crs = CRS.from_epsg(4326)
proj = Transformer.from_crs(from_crs, to_crs, always_xy=True)
lons, lats = proj.transform(mt_arr[:, 0], mt_arr[:, 1])
```

**After** (new API):
```python
# Integrated conversion (if pyproj available)
lons, lats = mesh.geo.from_utm(mt_arr[:, 0], mt_arr[:, 1], zone=53, southern=True)

# Or general CRS
lons, lats = mesh.geo.from_crs(x, y, from_crs='EPSG:28353')
```

---

## Implementation Details

### Ellipsoid Parameters

```python
ELLIPSOIDS = {
    'WGS84': {
        'a': 6378.137,      # km
        'b': 6356.752,      # km
        'f': 1/298.257223563,
        'planet': 'Earth',
    },
    'GRS80': {
        'a': 6378.137,
        'b': 6356.752,
        'f': 1/298.257222101,
        'planet': 'Earth',
    },
    'sphere': {
        'a': 6371.0,
        'b': 6371.0,
        'f': 0.0,
        'planet': 'Earth',
    },
    # Planetary ellipsoids
    'Mars': {
        'a': 3396.2,
        'b': 3376.2,
        'f': 1/169.8,
        'planet': 'Mars',
    },
    'Moon': {
        'a': 1738.1,
        'b': 1736.0,
        'f': 1/824.7,
        'planet': 'Moon',
    },
    'Venus': {
        'a': 6051.8,
        'b': 6051.8,
        'f': 0.0,  # Nearly perfect sphere
        'planet': 'Venus',
    },
}
```

### Coordinate Conversion Formulas

#### Geographic (lon, lat, depth) → Cartesian (x, y, z)

For ellipsoid with semi-major axis `a` and semi-minor axis `b`:

```python
def geographic_to_cartesian(lon, lat, depth, a, b):
    """
    Convert geographic coordinates to Cartesian.

    lon, lat: degrees
    depth: km below surface (positive downward)
    a, b: ellipsoid semi-axes in km

    Returns: x, y, z in km
    """
    # Convert to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # Eccentricity squared
    e2 = 1 - (b/a)**2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Height above ellipsoid (negative of depth)
    h = -depth

    # Cartesian coordinates
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + h) * np.sin(lat_rad)

    return x, y, z
```

#### Cartesian (x, y, z) → Geographic (lon, lat, depth)

```python
def cartesian_to_geographic(x, y, z, a, b, max_iterations=10, tolerance=1e-12):
    """
    Convert Cartesian coordinates to geographic.

    Uses iterative algorithm for geodetic latitude.
    """
    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Latitude requires iteration
    e2 = 1 - (b/a)**2
    p = np.sqrt(x**2 + y**2)

    # Initial guess
    lat = np.arctan2(z, p * (1 - e2))

    # Iterate to refine
    for _ in range(max_iterations):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), p)

        if np.abs(lat_new - lat).max() < tolerance:
            break
        lat = lat_new

    # Height above ellipsoid
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N

    # Depth is negative height
    depth = -h

    # Convert to degrees
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)

    return lon_deg, lat_deg, depth
```

### Basis Vector Calculations

For ellipsoidal coordinates, basis vectors are:

```python
# Unit vector in longitude direction (East)
unit_east = [-sin(lon), cos(lon), 0]

# Unit vector in latitude direction (North)
# (normalized tangent to meridian)
unit_north = [
    -sin(lat) * cos(lon),
    -sin(lat) * sin(lon),
    cos(lat)
] / (scaling factor from ellipsoid)

# Unit vector in radial direction (Up)
# (perpendicular to ellipsoid surface at this point)
unit_up = [
    cos(lat) * cos(lon),
    cos(lat) * sin(lon),
    sin(lat)
] / (scaling factor from ellipsoid)
```

---

## Decision Points

### Decision 1: Latitude Type

**Question**: Should `mesh.geo.lat` be **geodetic** (perpendicular to ellipsoid) or **geocentric** (angle from center)?

**Geodetic** (recommended):
- ✅ Matches GPS, maps, satellite data
- ✅ What Earth scientists expect
- ✅ Standard in GIS systems
- ❌ More complex math (iterative conversion)

**Geocentric**:
- ✅ Simpler math (direct from spherical)
- ❌ Doesn't match real-world data
- ❌ Confusing for users

**Proposal**: Use **geodetic latitude** as default, but provide both:
```python
mesh.geo.lat           # Geodetic (default)
mesh.geo.lat_geodetic  # Explicit
mesh.geo.lat_geocentric  # For advanced users
```

**Your input needed**: Is geodetic latitude acceptable even though it's more complex?

---

### Decision 2: Depth Reference

**Question**: What does `depth = 0` mean?

**Option A: Surface of reference ellipsoid** (recommended)
- ✅ Clear mathematical definition
- ✅ Works before topography applied
- ❌ Doesn't match actual surface after topography

**Option B: Actual deformed surface**
- ✅ Intuitive "depth below ground"
- ❌ Changes meaning when topography applied
- ❌ Undefined before topography

**Proposal**: Provide multiple depth coordinates:
```python
mesh.geo.depth         # Below reference ellipsoid (primary)
mesh.geo.depth_surface  # Below actual surface (if topography applied)
mesh.geo.radius        # From Earth center (geometric)
mesh.geo.altitude      # Above reference ellipsoid (negative of depth)
```

**Your input needed**: Is this too many options? What's most useful?

---

### Decision 3: Vertical Direction

**Question**: Should `unit_vertical` point radially outward or perpendicular to ellipsoid surface?

**Radial** (from Earth center):
- ✅ Simpler math
- ✅ Physically meaningful (gravity roughly radial)
- ❌ Not perpendicular to ellipsoid surface

**Surface normal**:
- ✅ Perpendicular to ellipsoid (more "vertical" feeling)
- ❌ Not exactly radial (except at poles/equator)
- ❌ Gravity not exactly along this direction

**Proposal**: Use **radial** but provide both:
```python
mesh.geo.unit_up       # Radial outward (default)
mesh.geo.unit_normal   # Perpendicular to ellipsoid
```

For a sphere they're identical. For Earth ellipsoid, difference is <0.2° at most.

**Your input needed**: Is radial acceptable for "vertical"?

---

### Decision 4: Ellipsoid in Mesh Creation

**Question**: Should ellipsoid be part of mesh creation or applied as deformation?

**Option A: Built into mesh creation** (recommended)
```python
mesh = uw.meshing.RegionalEllipsoidalBox(..., ellipsoid='WGS84')
# Mesh is ellipsoidal from the start
```
- ✅ Cleaner API
- ✅ Gmsh can create ellipsoidal geometry directly
- ❌ Need new mesh function (or modify existing)

**Option B: Applied as deformation**
```python
mesh = uw.meshing.RegionalSphericalBox(...)  # Sphere
mesh.deform_to_ellipsoid('WGS84')  # Then deform
```
- ✅ Backward compatible
- ✅ Flexible (can deform later)
- ❌ Extra step for users
- ❌ Need to track ellipsoid state

**Proposal**: Option A (built-in) with backward compatibility:
```python
# New way (recommended)
mesh = uw.meshing.RegionalEllipsoidalBox(..., ellipsoid='WGS84')

# Old way (still works, spherical)
mesh = uw.meshing.RegionalSphericalBox(...)

# Deformation option (for complex cases)
mesh.geo.deform_to_ellipsoid('WGS84')
```

**Your input needed**: Should we deprecate `RegionalSphericalBox` or keep both?

---

### Decision 5: PyProj Dependency

**Question**: How tightly should we integrate with pyproj?

**Option A: Optional with fallback** (recommended)
```python
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Core conversions work without pyproj
mesh.geo.to_cartesian(lon, lat, depth)  # Always works

# Enhanced features require pyproj
mesh.geo.from_utm(...)  # Raises informative error if no pyproj
```

**Option B: Required dependency**
- ✅ Full geodetic support always available
- ❌ Extra dependency for all users

**Option C: Completely independent**
- ✅ No external dependencies
- ❌ Have to reimplement all CRS transformations

**Proposal**: Option A - core functions independent, enhanced features with pyproj

**Your input needed**: Is pyproj acceptable as optional dependency?

---

### Decision 6: Coordinate System Type

**Question**: Should Geographic be a subtype of Spherical or independent?

**Option A: Inherit from Spherical** (recommended)
```python
class GeographicCoordinateSystem(SphericalCoordinateSystem):
    # Has all spherical properties plus geographic
```
- ✅ mesh.R still works (spherical coords)
- ✅ mesh.geo adds geographic coords
- ✅ Code reuse

**Option B: Independent system**
```python
class GeographicCoordinateSystem(CoordinateSystem):
    # Standalone, no spherical inheritance
```
- ✅ Cleaner separation
- ❌ Lose spherical coordinate access
- ❌ Code duplication

**Proposal**: Option A with clear hierarchy:
```
CoordinateSystem (base)
├─ SphericalCoordinateSystem
│  └─ GeographicCoordinateSystem (adds ellipsoid + geo coords)
├─ CylindricalCoordinateSystem
└─ CartesianCoordinateSystem
```

**Your input needed**: Should geographic meshes still expose `mesh.R` (spherical)?

---

### Decision 7: Planetary Ellipsoids

**Question**: How should we handle different planets?

**Option A: Named ellipsoids** (recommended)
```python
mesh = uw.meshing.GlobalEllipsoidalShell(
    ellipsoid='Mars',  # Predefined
    depth_range=(0, 1000),
)
```

**Option B: Explicit planet parameter**
```python
mesh = uw.meshing.GlobalEllipsoidalShell(
    planet='Mars',
    depth_range=(0, 1000),
)
```

**Option C: Custom ellipsoid only**
```python
mesh = uw.meshing.GlobalEllipsoidalShell(
    ellipsoid=(3396.2, 3376.2),  # Mars radii
    depth_range=(0, 1000),
)
```

**Proposal**: Support all three for flexibility:
```python
# Named planet
ellipsoid='Mars'

# Custom ellipsoid
ellipsoid=(a, b)

# Planet attribute for metadata
mesh.geo.planet_name  # 'Mars' if recognized
```

**Your input needed**: Which planets should we include predefined?
- Earth (WGS84, GRS80)
- Mars
- Moon
- Venus
- Others?

---

## Implementation Plan

### Phase 1: Core Geographic Coordinate System
1. Create `GeographicCoordinateSystem` class
2. Implement coordinate conversions (geo ↔ Cartesian)
3. Add geodetic latitude calculations
4. Implement basis vectors (unit_east, unit_north, unit_up)

### Phase 2: Ellipsoid Support
1. Add ellipsoid parameter handling
2. Implement ellipsoid geometry in gmsh mesh generation
3. Add ellipsoid properties (a, b, f, e)
4. Support named ellipsoids (WGS84, Mars, etc.)

### Phase 3: Mesh Creation Functions
1. Create `RegionalEllipsoidalBox` function
2. Create `GlobalEllipsoidalShell` function
3. Maintain backward compatibility with `RegionalSphericalBox`
4. Add `deform_to_ellipsoid` method

### Phase 4: Geographic Utilities
1. Add raster sampling (with gdal/rasterio)
2. Implement topography deformation
3. Add distance/area calculations
4. Integrate pyproj for CRS conversions (optional)

### Phase 5: Testing and Documentation
1. Unit tests for coordinate conversions
2. Validation against pyproj/geopy
3. Example notebooks
4. Update existing examples to use new API

---

## Testing Strategy

### Unit Tests
```python
def test_geographic_cartesian_roundtrip():
    """Test conversion accuracy"""
    lon, lat, depth = 138.0, -33.0, 50.0
    x, y, z = geo_to_cart(lon, lat, depth, 'WGS84')
    lon2, lat2, depth2 = cart_to_geo(x, y, z, 'WGS84')
    assert np.allclose([lon, lat, depth], [lon2, lat2, depth2])

def test_ellipsoid_vs_sphere():
    """Verify ellipsoid differs from sphere"""
    # At equator, WGS84 radius is 6378.137 km
    # At pole, WGS84 radius is 6356.752 km
    # Difference: 21.385 km
    pass

def test_geodetic_vs_geocentric_latitude():
    """Verify latitude type difference"""
    # Maximum difference at 45° latitude: ~11 arcminutes
    pass

def test_basis_vectors_orthonormal():
    """Verify basis vectors are orthonormal"""
    assert np.allclose(np.dot(unit_east, unit_north), 0)
    assert np.allclose(np.dot(unit_east, unit_up), 0)
    assert np.allclose(np.dot(unit_north, unit_up), 0)
    assert np.allclose(np.linalg.norm(unit_east), 1)
```

### Integration Tests
```python
def test_regional_mesh_creation():
    """Test creating regional ellipsoidal mesh"""
    mesh = uw.meshing.RegionalEllipsoidalBox(
        lon_range=(135, 140),
        lat_range=(-35, -30),
        depth_range=(0, 400),
        ellipsoid='WGS84',
        numElements=(5, 5, 5),
    )
    assert mesh.geo.ellipsoid_name == 'WGS84'
    assert mesh.geo.lon.min() >= 135
    assert mesh.geo.lat.max() <= -30

def test_topography_deformation():
    """Test mesh deformation with topography"""
    # Create mesh, add topography, verify deformation
    pass
```

### Validation Tests
```python
def test_against_pyproj():
    """Validate against pyproj (if available)"""
    if not HAS_PYPROJ:
        pytest.skip("pyproj not available")

    # Compare our conversions with pyproj
    pass

def test_against_geopy():
    """Validate distance calculations"""
    if not HAS_GEOPY:
        pytest.skip("geopy not available")

    # Compare great circle distances
    pass
```

---

## Open Questions

1. **Depth handling with topography**: Should `mesh.geo.depth` update when topography is applied, or stay relative to reference ellipsoid?

2. **Units**: Should depth be in km (current practice) or meters (SI standard)? Or configurable?

3. **Coordinate bounds**: Should longitude be (-180, 180) or (0, 360)? Or support both?

4. **Symbolic expressions**: How should symbolic `lon`, `lat`, `depth` work in SymPy equations? Need careful handling of units and ranges.

5. **Performance**: Are the iterative geodetic conversions fast enough for large meshes? May need Cython/numba acceleration.

6. **Mesh refinement**: How does adaptive refinement work with ellipsoidal geometry? Need callback to maintain ellipsoid shape.

7. **Parallel decomposition**: How does domain decomposition work with geographic coordinates? Need to ensure proper halo exchange.

---

## Next Steps

1. **Review this design document** - your feedback on decisions
2. **Prioritize features** - what's essential vs nice-to-have?
3. **Prototype core conversions** - validate math and performance
4. **Implement Phase 1** - basic geographic coordinate system
5. **Test with your workflow** - ensure it solves your actual problems

---

## References

- WGS84 Definition: NIMA Technical Report TR8350.2
- Vincenty's Formulae for geodetic calculations
- pyproj documentation: https://pyproj4.github.io/pyproj/
- GDAL/rasterio for raster data handling
- Geographic coordinate systems: Snyder, J.P. "Map Projections--A Working Manual"
