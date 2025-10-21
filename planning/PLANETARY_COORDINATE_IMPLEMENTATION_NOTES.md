# Planetary Coordinate System Implementation Notes

## Implementation Progress

### Completed:
1. ✅ Added `CoordinateSystemType.PLANETARY` enum
2. ✅ Added `ELLIPSOIDS` dictionary with WGS84, Mars, Moon, Venus

### Next Steps:
1. Add PLANETARY case to `Coordinate System.__init__()`
2. Implement coordinate conversion functions
3. Add `mesh.planetary` accessor (like `mesh.geo`)
4. Add basis vectors (unit_WE, unit_SN, unit_down)
5. Create `RegionalPlanetaryBox` mesh function

## Key Design Decisions (from user feedback):

**Basis Vectors** (right-handed):
- `unit_WE`: West→East (positive East, longitude direction)
- `unit_SN`: South→North (positive North, latitude direction)
- `unit_down`: pointing down into planet (positive downward)
- **Right-handed check**: WE × SN = down ✓

**Symbolic Coordinates**:
- `λ_lon` (lambda_lon) - longitude in degrees
- `λ_lat` (lambda_lat) - latitude in degrees (geodetic)
- `λ_d` (lambda_d) - depth below reference ellipsoid in km

**Depth Reference**:
- `depth = 0` at reference ellipsoid surface
- `depth > 0` below surface (into planet)
- Also provide `radius` from planet center
- Relationship: `depth = surface_radius_at_lat - radius`

**Naming**: **Planetary** not "Geographic" (works for any planet!)

**Ellipsoid Parameter**: `ellipsoid=True` defaults to WGS84

## Coordinate Conversions Required

### (lon, lat, depth) → (x, y, z)

For ellipsoid with semi-major axis `a`, semi-minor axis `b`:

```python
def planetary_to_cartesian(lon_deg, lat_deg, depth_km, a, b):
    """
    Convert planetary coordinates to Cartesian.

    Uses geodetic latitude (perpendicular to ellipsoid surface).
    """
    # Convert to radians
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)

    # Eccentricity squared
    e2 = 1 - (b/a)**2

    # Prime vertical radius of curvature at this latitude
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    # Height above ellipsoid (negative of depth)
    h = -depth_km

    # Cartesian coordinates
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)

    return x, y, z
```

### (x, y, z) → (lon, lat, depth)

```python
def cartesian_to_planetary(x, y, z, a, b, max_iter=10, tol=1e-12):
    """
    Convert Cartesian to planetary coordinates.

    Uses iterative algorithm for geodetic latitude.
    """
    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Latitude requires iteration (Bowring's method)
    e2 = 1 - (b/a)**2
    p = np.sqrt(x**2 + y**2)

    # Initial guess for latitude
    lat = np.arctan2(z, p * (1 - e2))

    # Iterate to converge on geodetic latitude
    for _ in range(max_iter):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), p)

        if np.abs(lat_new - lat).max() < tol:
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

## Basis Vectors (Symbolic)

For ellipsoidal coordinates at (lon, lat):

### unit_WE (East direction)
```python
# Tangent to line of constant latitude, pointing East
unit_WE = sympy.Matrix([
    -sympy.sin(lon_rad),
    sympy.cos(lon_rad),
    0
])
# Already unit length
```
**Aliases**: `unit_east`, `unit_lon` (longitude direction)

### unit_SN (North direction)
```python
# Tangent to meridian, pointing North
# (perpendicular to ellipsoid, in meridional plane)
unit_SN = sympy.Matrix([
    -sympy.sin(lat_rad) * sympy.cos(lon_rad),
    -sympy.sin(lat_rad) * sympy.sin(lon_rad),
    sympy.cos(lat_rad)
]) / (1 + ellipsoid_correction_factor)
# Needs normalization for ellipsoid
```
**Aliases**: `unit_north`, `unit_lat` (latitude direction)

### unit_down (Radial inward)
```python
# Pointing toward planet center (approximately)
# For simplicity, use geocentric radial (exact normal to ellipsoid is complex)
unit_down = -sympy.Matrix([
    sympy.cos(lat_rad) * sympy.cos(lon_rad),
    sympy.cos(lat_rad) * sympy.sin(lon_rad),
    sympy.sin(lat_rad)
])
# Negative because "down" points inward
```
**Aliases**: `unit_depth` (depth direction)

### Opposite directions
- `unit_up` = -`unit_down` (radial outward)
- `unit_west` = -`unit_WE`
- `unit_south` = -`unit_SN`

**Right-handed check**:
- WE × SN should give down (inward radial)
- East × North = Down ✓

## Implementation in CoordinateSystem.__init__()

```python
elif system == CoordinateSystemType.PLANETARY and self.mesh.dim == 3:
    """
    Planetary coordinate system with ellipsoid support.
    Coordinates: (longitude, latitude, depth)
    - longitude: degrees East (-180 to 180 or 0 to 360)
    - latitude: degrees North (-90 to 90), geodetic
    - depth: km below reference ellipsoid surface (positive downward)
    """
    self.type = "Planetary"

    # Store ellipsoid parameters (must be set externally)
    if not hasattr(self.mesh, '_ellipsoid_params'):
        # Default to sphere if not specified
        self.mesh._ellipsoid_params = ELLIPSOIDS['sphere']

    self._ellipsoid = self.mesh._ellipsoid_params
    self._a = self._ellipsoid['a']  # semi-major axis (km)
    self._b = self._ellipsoid['b']  # semi-minor axis (km)
    self._f = self._ellipsoid['f']  # flattening
    self._planet = self._ellipsoid.get('planet', 'Unknown')

    # Cartesian coordinates (native)
    self._X = self._N.copy()
    self._x = self._X

    # Planetary symbolic coordinates (pure symbols)
    # Using lambda notation as requested
    self._lambda = sympy.Matrix([
        sympy.symbols(r"\lambda_{lon}", real=True),  # longitude
        sympy.symbols(r"\lambda_{lat}", real=True),  # latitude
        sympy.symbols(r"\lambda_{d}", real=True, positive=True),  # depth
    ])

    # Expressions for planetary coords in terms of Cartesian
    x, y, z = self.X

    # These are the conversion formulas (approximations for symbolic use)
    # Exact conversions require numerical iteration for geodetic latitude

    # Longitude (straightforward)
    lon = expression(
        R"\lambda_{lon}",
        sympy.atan2(y, x) * 180 / sympy.pi,  # degrees
        "Longitude (degrees East)",
    )

    # Latitude (geocentric approximation for symbolic)
    # Note: True geodetic latitude requires iteration
    r_xy = sympy.sqrt(x**2 + y**2)
    lat = expression(
        R"\lambda_{lat}",
        sympy.atan2(z, r_xy) * 180 / sympy.pi,  # degrees
        "Latitude (degrees North, approximate)",
    )

    # Radius and depth
    r = expression(
        R"r",
        sympy.sqrt(x**2 + y**2 + z**2),
        "Radius from center",
    )

    # Depth below surface (approximate - assumes spherical)
    # TODO: Improve this for ellipsoid
    depth = expression(
        R"\lambda_d",
        self._a - r,  # depth = surface_radius - current_radius
        "Depth below surface (km)",
    )

    # Store planetary coordinate expressions
    self._P = sympy.Matrix([[lon, lat, depth]])

    # Basis vectors (symbolic)
    lon_rad = lon * sympy.pi / 180
    lat_rad = lat * sympy.pi / 180

    # unit_WE: West to East (longitude direction, azimuthal)
    self._unit_WE = sympy.Matrix([
        -sympy.sin(lon_rad),
        sympy.cos(lon_rad),
        self.independent_of_N,  # No z-component
    ])

    # unit_SN: South to North (latitude direction, meridional)
    self._unit_SN = sympy.Matrix([
        -sympy.sin(lat_rad) * sympy.cos(lon_rad),
        -sympy.sin(lat_rad) * sympy.sin(lon_rad),
        sympy.cos(lat_rad),
    ])

    # unit_down: pointing into planet (negative radial)
    self._unit_down = -sympy.Matrix([
        x / r,
        y / r,
        z / r,
    ])

    # Also maintain spherical coords (inherits from SPHERICAL)
    # This gives us mesh.R as well as mesh.planetary
    self._r = sympy.Matrix([sympy.symbols(R"r, \theta, \phi")])

    th = expression(
        R"\theta",
        sympy.acos(z / r),
        "co-latitude",
    )

    ph = expression(
        R"\phi",
        sympy.atan2(y, x),
        "longitude (radians)",
    )

    self._R = sympy.Matrix([[r, th, ph]])

    # Rotation matrices (for compatibility)
    self._xRotN = sympy.eye(self.mesh.dim)

    # rRotN maps from Cartesian to spherical basis
    # (keep spherical for backward compatibility)
    rz = sympy.sqrt(x**2 + y**2)
    r_x_rz = sympy.sqrt((x**2 + y**2 + z**2) * (x**2 + y**2))

    self._rRotN = sympy.Matrix([
        [x / r, y / r, z / r],  # radial
        [(x * z) / r_x_rz, (y * z) / r_x_rz, -(x**2 + y**2) / r_x_rz],  # meridional
        [-y / rz, +x / rz, self.independent_of_N],  # azimuthal
    ])
```

## PlanetaryCoordinateAccessor Class

Add a property accessor for cleaner API:

```python
class PlanetaryCoordinateAccessor:
    """
    Accessor for planetary coordinates on ellipsoidal meshes.

    Provides:
    - mesh.planetary.lon, .lat, .depth (data arrays)
    - mesh.planetary[0], [1], [2] (symbolic coords)
    - mesh.planetary.unit_WE, .unit_SN, .unit_down (basis vectors)
    - mesh.planetary.to_cartesian(), .from_cartesian() (conversions)
    """

    def __init__(self, coordinate_system):
        self.cs = coordinate_system
        self.mesh = coordinate_system.mesh

    @property
    def lon(self):
        """Longitude coordinate data (degrees East)"""
        # Convert mesh Cartesian coords to planetary
        return self._cart_to_planetary()[0]

    @property
    def lat(self):
        """Latitude coordinate data (degrees North)"""
        return self._cart_to_planetary()[1]

    @property
    def depth(self):
        """Depth below reference ellipsoid (km)"""
        return self._cart_to_planetary()[2]

    def __getitem__(self, idx):
        """Access symbolic coordinates: lon, lat, depth = mesh.planetary[:]"""
        return self.cs._P[idx]

    # Primary basis vectors (canonical names)
    @property
    def unit_WE(self):
        """Unit vector pointing East (West to East)"""
        return self.cs._unit_WE

    @property
    def unit_SN(self):
        """Unit vector pointing North (South to North)"""
        return self.cs._unit_SN

    @property
    def unit_down(self):
        """Unit vector pointing down (into planet)"""
        return self.cs._unit_down

    # Aliases for clarity
    @property
    def unit_east(self):
        """Alias for unit_WE (East direction, longitude)"""
        return self.cs._unit_WE

    @property
    def unit_lon(self):
        """Alias for unit_WE (longitude direction)"""
        return self.cs._unit_WE

    @property
    def unit_north(self):
        """Alias for unit_SN (North direction, latitude)"""
        return self.cs._unit_SN

    @property
    def unit_lat(self):
        """Alias for unit_SN (latitude direction)"""
        return self.cs._unit_SN

    @property
    def unit_depth(self):
        """Alias for unit_down (depth direction, into planet)"""
        return self.cs._unit_down

    # Opposite directions
    @property
    def unit_up(self):
        """Unit vector pointing up (out of planet), opposite of unit_down"""
        return -self.cs._unit_down

    @property
    def unit_west(self):
        """Unit vector pointing West, opposite of unit_WE"""
        return -self.cs._unit_WE

    @property
    def unit_south(self):
        """Unit vector pointing South, opposite of unit_SN"""
        return -self.cs._unit_SN

    def _cart_to_planetary(self):
        """Convert mesh Cartesian coordinates to planetary"""
        xyz = self.mesh.X.coords
        lon, lat, depth = cartesian_to_planetary(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            self.cs._a, self.cs._b
        )
        return lon, lat, depth

    def to_cartesian(self, lon, lat, depth):
        """Convert planetary coords to Cartesian"""
        return planetary_to_cartesian(
            lon, lat, depth,
            self.cs._a, self.cs._b
        )

    def from_cartesian(self, x, y, z):
        """Convert Cartesian coords to planetary"""
        return cartesian_to_planetary(
            x, y, z,
            self.cs._a, self.cs._b
        )
```

Then in `CoordinateSystem.__init__()`, add:
```python
if system == CoordinateSystemType.PLANETARY:
    self.planetary = PlanetaryCoordinateAccessor(self)
```

## Usage Examples

```python
# Create mesh with WGS84 ellipsoid
mesh = uw.meshing.RegionalPlanetaryBox(
    lon_range=(135, 140),
    lat_range=(-35, -30),
    depth_range=(0, 400),
    ellipsoid='WGS84',  # or True, or 'Mars'
    numElements=(10, 10, 10),
)

# Access coordinate data
lon = mesh.planetary.lon      # (N,) array, degrees East
lat = mesh.planetary.lat      # (N,) array, degrees North
depth = mesh.planetary.depth  # (N,) array, km below surface

# Symbolic coordinates for equations
λ_lon, λ_lat, λ_d = mesh.planetary[:]

# Temperature decreasing with depth
T = 1600 - 0.5 * λ_d

# Basis vectors - multiple names available:

# Primary names (canonical):
v_bc = v_magnitude * mesh.planetary.unit_down
v_h = v_east * mesh.planetary.unit_WE + v_north * mesh.planetary.unit_SN

# Directional aliases (clearer):
v_bc = v_magnitude * mesh.planetary.unit_depth
v_h = v_east * mesh.planetary.unit_east + v_north * mesh.planetary.unit_north

# Coordinate-based aliases:
v_h = v_lon * mesh.planetary.unit_lon + v_lat * mesh.planetary.unit_lat

# Opposite directions:
buoyancy_force = rho * g * mesh.planetary.unit_up  # Upward
surface_flow = v * mesh.planetary.unit_west        # Westward

# Check right-handed system
cross_product = mesh.planetary.unit_WE.cross(mesh.planetary.unit_SN)
# Should equal mesh.planetary.unit_down ✓
```

## Complete Basis Vector API

**Primary (canonical) names:**
- `mesh.planetary.unit_WE` - West to East (lon direction)
- `mesh.planetary.unit_SN` - South to North (lat direction)
- `mesh.planetary.unit_down` - into planet (depth direction)

**Directional aliases:**
- `mesh.planetary.unit_east` = `unit_WE`
- `mesh.planetary.unit_north` = `unit_SN`
- `mesh.planetary.unit_depth` = `unit_down`

**Coordinate-based aliases:**
- `mesh.planetary.unit_lon` = `unit_WE`
- `mesh.planetary.unit_lat` = `unit_SN`

**Opposite directions:**
- `mesh.planetary.unit_west` = -`unit_WE`
- `mesh.planetary.unit_south` = -`unit_SN`
- `mesh.planetary.unit_up` = -`unit_down`

**Right-handed system**: WE × SN = down ✓

## Testing Plan

1. **Test coordinate conversions**:
   - Round-trip: (lon,lat,depth) → (x,y,z) → (lon,lat,depth)
   - Test at equator, poles, mid-latitudes
   - Compare WGS84 vs sphere

2. **Test basis vectors**:
   - Verify orthonormality
   - Verify right-handed system
   - Test at various locations

3. **Test with existing workflow**:
   - Port `Mesh-Adapted-2-Faults.py` to use new API
   - Verify coordinate conversions match old manual method
   - Check topography deformation works

4. **Performance**:
   - Benchmark conversion functions
   - Consider Cython/numba if too slow

## Next Implementation Steps

1. ✅ Add PLANETARY enum and ELLIPSOIDS dict
2. → Add PLANETARY case to CoordinateSystem.__init__()
3. → Implement conversion functions
4. → Add PlanetaryCoordinateAccessor class
5. → Create RegionalPlanetaryBox mesh function
6. → Test with simple example
7. → Port existing workflow example
8. → Documentation and refinement
