from typing import Optional, Tuple
from enum import Enum

import tempfile
import numpy as np
from petsc4py import PETSc

import underworld3
from underworld3 import VarType
import sympy

expression = lambda *x, **X: underworld3.function.expressions.UWexpression(
    *x, _unique_name_generation=True, **X
)


class CoordinateSystemType(Enum):
    """
    Meshes can have natural coordinate systems which lie on top of the Cartesian
    coordinate system that we use for constructing the solvers (usually)
    """

    CARTESIAN = 0
    CYLINDRICAL2D = 10  # Cyl2D and Polar are equivalent here
    POLAR = 10  #
    CYLINDRICAL3D = 100  # (Not really used for anything)
    SPHERICAL = 200
    GEOGRAPHIC = 300  # Ellipsoidal coordinates (lon, lat, depth) for Earth and planets


# Ellipsoid parameters for geographic coordinate systems
# Semi-major axis (a), semi-minor axis (b), flattening (f), planet name
ELLIPSOIDS = {
    'WGS84': {
        'a': 6378.137,           # km
        'b': 6356.752,           # km
        'f': 1/298.257223563,
        'planet': 'Earth',
        'description': 'World Geodetic System 1984',
    },
    'GRS80': {
        'a': 6378.137,
        'b': 6356.752,
        'f': 1/298.257222101,
        'planet': 'Earth',
        'description': 'Geodetic Reference System 1980',
    },
    'sphere': {
        'a': 6371.0,             # Mean Earth radius
        'b': 6371.0,
        'f': 0.0,
        'planet': 'Earth',
        'description': 'Perfect sphere (mean Earth radius)',
    },
    'Mars': {
        'a': 3396.2,
        'b': 3376.2,
        'f': 1/169.8,
        'planet': 'Mars',
        'description': 'Mars ellipsoid',
    },
    'Moon': {
        'a': 1738.1,
        'b': 1736.0,
        'f': 1/824.7,
        'planet': 'Moon',
        'description': 'Moon ellipsoid',
    },
    'Venus': {
        'a': 6051.8,
        'b': 6051.8,
        'f': 0.0,                # Nearly perfect sphere
        'planet': 'Venus',
        'description': 'Venus ellipsoid',
    },
}


def geographic_to_cartesian(lon_deg, lat_deg, depth_km, a, b):
    """
    Convert geographic coordinates to Cartesian coordinates.

    Uses geodetic latitude (perpendicular to ellipsoid surface).

    Parameters
    ----------
    lon_deg : float or array
        Longitude in degrees East (-180 to 180 or 0 to 360)
    lat_deg : float or array
        Latitude in degrees North (-90 to 90), geodetic
    depth_km : float or array
        Depth below ellipsoid surface in km (positive downward)
    a : float
        Semi-major axis (equatorial radius) in km
    b : float
        Semi-minor axis (polar radius) in km

    Returns
    -------
    x, y, z : float or array
        Cartesian coordinates in km
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


def cartesian_to_geographic(x, y, z, a, b, max_iterations=10, tolerance=1e-12):
    """
    Convert Cartesian coordinates to geographic coordinates.

    Uses iterative algorithm (Bowring's method) for geodetic latitude.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates in km
    a : float
        Semi-major axis (equatorial radius) in km
    b : float
        Semi-minor axis (polar radius) in km
    max_iterations : int, optional
        Maximum iterations for latitude convergence (default: 10)
    tolerance : float, optional
        Convergence tolerance in radians (default: 1e-12)

    Returns
    -------
    lon_deg, lat_deg, depth_km : float or array
        Geographic coordinates:
        - lon_deg: Longitude in degrees East
        - lat_deg: Latitude in degrees North (geodetic)
        - depth_km: Depth below ellipsoid surface in km
    """
    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Latitude requires iteration (Bowring's method for geodetic latitude)
    e2 = 1 - (b/a)**2
    p = np.sqrt(x**2 + y**2)

    # Initial guess for latitude (geocentric)
    lat = np.arctan2(z, p * (1 - e2))

    # Iterate to converge on geodetic latitude
    for i in range(max_iterations):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), p)

        # Check convergence
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


class GeographicCoordinateAccessor:
    """
    Accessor for geographic coordinates on ellipsoidal meshes.

    Provides intuitive access to longitude, latitude, and depth coordinates,
    plus basis vectors with multiple naming conventions.

    Accessed via: mesh.geo

    Examples:
        # Data arrays (numpy)
        lon = mesh.geo.lon         # Longitude (degrees East)
        lat = mesh.geo.lat         # Latitude (degrees North)
        depth = mesh.geo.depth     # Depth below surface (km)

        # Symbolic (for equations)
        λ_lon, λ_lat, λ_d = mesh.geo[:]

        # Basis vectors (multiple naming options)
        mesh.geo.unit_WE           # Primary: West to East
        mesh.geo.unit_east         # Directional alias
        mesh.geo.unit_lon          # Coordinate alias
        mesh.geo.unit_down         # Primary: into planet
        mesh.geo.unit_depth        # Coordinate alias
    """

    def __init__(self, coordinate_system):
        """
        Initialize geographic coordinate accessor.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The coordinate system object with GEOGRAPHIC type
        """
        self.cs = coordinate_system
        self.mesh = coordinate_system.mesh

        # Cache for coordinate arrays
        self._lon_cache = None
        self._lat_cache = None
        self._depth_cache = None
        self._cache_valid = False

    def _invalidate_cache(self):
        """Mark coordinate cache as invalid (call when mesh coordinates change)."""
        self._cache_valid = False

    def _compute_coordinates(self):
        """Compute geographic coordinates from Cartesian mesh coordinates."""
        if self._cache_valid:
            return

        # Get Cartesian coordinates
        coords = self.mesh.CoordinateSystem.coords  # Shape: (N, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # Get ellipsoid parameters
        a = self.cs.ellipsoid['a']
        b = self.cs.ellipsoid['b']

        # Convert to geographic using our utility function
        self._lon_cache, self._lat_cache, self._depth_cache = \
            cartesian_to_geographic(x, y, z, a, b)

        self._cache_valid = True

    @property
    def lon(self):
        """Longitude in degrees East (-180 to 180)."""
        self._compute_coordinates()
        return self._lon_cache

    @property
    def lat(self):
        """Geodetic latitude in degrees North (-90 to 90)."""
        self._compute_coordinates()
        return self._lat_cache

    @property
    def depth(self):
        """Depth below ellipsoid surface in km (positive downward)."""
        self._compute_coordinates()
        return self._depth_cache

    def __getitem__(self, idx):
        """
        Access symbolic geographic coordinates.

        Returns:
            λ_lon, λ_lat, λ_d: Symbolic coordinates for use in equations
        """
        if idx == slice(None, None, None):  # mesh.geo[:]
            return self.cs._geo_coords[0, 0], self.cs._geo_coords[0, 1], self.cs._geo_coords[0, 2]
        else:
            return self.cs._geo_coords[0, idx]

    # === Basis Vectors - Primary Names ===

    @property
    def unit_WE(self):
        """West to East unit vector (primary name, positive East)."""
        return self.cs._unit_WE

    @property
    def unit_SN(self):
        """South to North unit vector (primary name, positive North)."""
        return self.cs._unit_SN

    @property
    def unit_down(self):
        """Downward unit vector (primary name, positive into planet)."""
        return self.cs._unit_down

    # === Directional Aliases ===

    @property
    def unit_east(self):
        """Eastward unit vector (directional alias for unit_WE)."""
        return self.cs._unit_WE

    @property
    def unit_west(self):
        """Westward unit vector (opposite of unit_WE)."""
        return -self.cs._unit_WE

    @property
    def unit_north(self):
        """Northward unit vector (directional alias for unit_SN)."""
        return self.cs._unit_SN

    @property
    def unit_south(self):
        """Southward unit vector (opposite of unit_SN)."""
        return -self.cs._unit_SN

    @property
    def unit_up(self):
        """Upward unit vector (opposite of unit_down)."""
        return -self.cs._unit_down

    # === Coordinate Aliases ===

    @property
    def unit_lon(self):
        """Longitude direction unit vector (coordinate alias for unit_WE)."""
        return self.cs._unit_WE

    @property
    def unit_lat(self):
        """Latitude direction unit vector (coordinate alias for unit_SN)."""
        return self.cs._unit_SN

    @property
    def unit_depth(self):
        """Depth direction unit vector (coordinate alias for unit_down)."""
        return self.cs._unit_down

    def __repr__(self):
        """String representation showing available coordinates and basis vectors."""
        return (
            f"GeographicCoordinateAccessor(\n"
            f"  ellipsoid='{self.cs.ellipsoid.get('description', 'Unknown')}',\n"
            f"  a={self.cs.ellipsoid['a']} km, b={self.cs.ellipsoid['b']} km,\n"
            f"  Coordinates: lon, lat, depth (data arrays)\n"
            f"              λ_lon, λ_lat, λ_d (symbolic)\n"
            f"  Basis vectors: unit_WE, unit_SN, unit_down (primary)\n"
            f"                unit_east, unit_north, unit_depth (directional)\n"
            f"                unit_lon, unit_lat (coordinate)\n"
            f"                unit_west, unit_south, unit_up (opposites)\n"
            f")"
        )


# Maybe break this out into it's own file - this needs to cover, basis vectors,
# coordinate arrays in the natural coordinate system (and setting them once the other
# setup is complete), direction of the vertical, implementation of
# rotated boundary conditions, null spaces ...


class CoordinateSystem:
    R"""This class is attached to a mesh to provide programmatic access to coordinate system operations.

    `CoordinateSystem.R` - the coordinates in the natural reference (sympy symbols)
    `CoordinateSystem.xR` - the coordinates in the natural reference (sympy symbols in terms of mesh.X)
    `CoordinateSystem._Rot` - the matrix to rotate from X to R, written symbolically
    `CoordinateSystem._xRot` - the matrix to rotate from X to R, written in terms of mesh.X

    Need these:

      Coordinates N -> Native, xi_1, xi_2 by default, but over-written with meaningful symbols if possible
      Coordinates R -> Natural XYZ or R Theta Phi or R Theta z
      Coordinates X -> Cartesian XYZ

      nRotr - Matrix to map N to R
      rRotn - nRotr.T
      nRotx - Matrix to map N to X
      xRotn - nRotx.T
      xRotr = xRotn * nRotr
      rRotx = rRotn * nRotx

      Either R or X will be an alias for N depending on whether the DM is Cartesian  (Alias has the same vectors but different names)


    """

    def __init__(
        self,
        mesh,
        system: Optional[CoordinateSystemType] = CoordinateSystemType.CARTESIAN,
    ):
        # Guard against SymPy trying to construct a CoordinateSystem from sympified elements
        # SymPy may iterate over the object and try to recreate it from elements
        if isinstance(mesh, (list, tuple)) or not hasattr(mesh, 'r'):
            raise TypeError(
                f"CoordinateSystem requires a mesh object, got {type(mesh).__name__}. "
                "This object is not meant to be reconstructed by SymPy."
            )

        self.mesh = mesh
        self.coordinate_type = system

        # are the mesh coordinates XYZ or have they been replaced by
        # "Natural" coordinates like r, theta, z ?
        self.CartesianDM = True

        # This is the default: Cartesian
        self._N = sympy.Matrix(self.mesh.r).T
        self._N[0]._latex_form = r"\mathrm{x}"
        self._N[1]._latex_form = r"\mathrm{y}"
        if self.mesh.cdim == 3:
            self._N[2]._latex_form = r"\mathrm{z}"

        # This is a how we can find our way back to the
        # originating coordinate system if we only know the base-scalars

        self._N[0].CS = self
        self._N[1].CS = self
        if self.mesh.cdim == 3:
            self._N[2].CS = self

        self._R = self._N.copy()

        # We need this to define zeros in the coordinate transforms
        # (since they need to indicate they are coordinate functions even
        # if they are independent of all coordinates)

        if self.mesh.cdim == 3:
            self.independent_of_N = expression(
                r"\vec{0}",
                underworld3.maths.functions.vanishing
                * self._N[0]
                * self._N[1]
                * self._N[2],
                "independent of N0, N1, N2",
            )
        else:
            self.independent_of_N = expression(
                r"\vec{0}",
                underworld3.maths.functions.vanishing * self._N[0] * self._N[1],
                "independent of N0, N1",
            )

        ## Change specific coordinates systems as required

        if system == CoordinateSystemType.CYLINDRICAL2D and self.mesh.dim == 2:
            """
            This describes the situation for an annulus mesh with base coordinates
            in Cartesian (x,y,z).
            """

            self.type = "Cylindrical 2D"

            self._X = self._N.copy()
            self._x = self._X

            x, y = self.N
            r = expression(R"r", sympy.sqrt(x**2 + y**2), "Radial Coordinate")

            t = expression(
                R"\theta",
                sympy.Piecewise((0, x == 0), (sympy.atan2(y, x), True)),
                "Angular coordinate",
            )

            self._R = sympy.Matrix([[r, t]])
            self._r = sympy.Matrix([sympy.symbols(R"r, \theta")], real=True)

            th = self._r[1]
            self._rRotN_sym = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th)],
                    [-sympy.sin(th), sympy.cos(th)],
                ]
            )
            self._rRotN = self._rRotN_sym.subs(th, sympy.atan2(y, x))
            self._xRotN = sympy.eye(self.mesh.dim)


        elif system == CoordinateSystemType.CYLINDRICAL3D and self.mesh.dim == 3:
            self.type = "Cylindrical 3D"

            self._X = self._N.copy()
            self._x = self._X

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, z")], real=True)

            x, y, z = self.X
            r = sympy.sqrt(x**2 + y**2)
            t = sympy.Piecewise((0, x == 0), (sympy.atan2(y, x), True))

            self._R = sympy.Matrix([[r, t, z]])

            th = self._r[1]
            self._rRotN_sym = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th), 0],
                    [-sympy.sin(th), sympy.cos(th), 0],
                    [0, 0, 1],
                ]
            )

            self._rRotN = self._rRotN_sym.subs(th, t)
            self._xRotN = sympy.eye(self.mesh.dim)


        elif system == CoordinateSystemType.SPHERICAL and self.mesh.dim == 3:
            self.type = "Spherical"

            self._X = self._N.copy()
            self._x = self._X

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, \phi")])

            x, y, z = self.X

            r = expression(
                R"r",
                sympy.sqrt(x**2 + y**2 + z**2),
                "Radial coordinate",
            )

            th = expression(
                R"\theta",
                sympy.acos(z / r),
                "co-latitude",
            )

            ph = expression(
                R"\phi",
                sympy.atan2(y, x),
                "longitude",
            )

            self._R = sympy.Matrix([[r, th, ph]])

            r1 = self._r[1]
            r2 = self._r[2]
            rRotN_sym = sympy.Matrix(
                [
                    [
                        sympy.sin(r1) * sympy.cos(r2),
                        sympy.sin(r1) * sympy.sin(r2),
                        sympy.cos(r1),
                    ],
                    [
                        sympy.cos(r1) * sympy.cos(r2),
                        sympy.cos(r1) * sympy.sin(r2),
                        -sympy.sin(r1),
                    ],
                    [
                        -sympy.sin(r2),
                        +sympy.cos(r2),
                        self.independent_of_N,
                    ],
                ]
            )

            rz = sympy.sqrt(x**2 + y**2)
            r_x_rz = sympy.sqrt((x**2 + y**2 + z**2) * (x**2 + y**2))

            rRotN = sympy.Matrix(
                [
                    [
                        x / r,
                        y / r,
                        z / r,
                    ],
                    [
                        (x * z) / r_x_rz,
                        (y * z) / r_x_rz,
                        -(x**2 + y**2) / r_x_rz,
                    ],
                    [
                        -y / rz,
                        +x / rz,
                        self.independent_of_N,
                    ],
                ]
            )

            self._rRotN_sym = rRotN_sym
            self._rRotN = rRotN

            self._xRotN = sympy.eye(self.mesh.dim)


        elif system == CoordinateSystemType.GEOGRAPHIC and self.mesh.dim == 3:
            """
            Geographic coordinate system for ellipsoidal meshes.

            Coordinates: (lon, lat, depth)
            - lon: Longitude in degrees East
            - lat: Geodetic latitude in degrees North (perpendicular to ellipsoid)
            - depth: Depth below ellipsoid surface in km (positive downward)

            Ellipsoid parameters stored in self.ellipsoid dict.
            mesh.R remains as spherical coordinates (r, θ, φ) for backward compatibility.
            """

            self.type = "Geographic"

            # Store ellipsoid parameters (will be set by mesh creation function)
            # Default to WGS84 if not specified
            if not hasattr(self, 'ellipsoid'):
                self.ellipsoid = ELLIPSOIDS['WGS84'].copy()

            # Cartesian coordinates remain primary
            self._X = self._N.copy()
            self._x = self._X

            # Define symbolic geographic coordinates (λ notation as per user request)
            self._geo = sympy.Matrix([sympy.symbols(R"\lambda_{lon}, \lambda_{lat}, \lambda_d")])

            # Get Cartesian symbols
            x, y, z = self.X

            # Spherical coordinates (r, θ, φ) - kept for backward compatibility
            r = expression(
                R"r",
                sympy.sqrt(x**2 + y**2 + z**2),
                "Radial coordinate (from center)",
            )

            th = expression(
                R"\theta",
                sympy.acos(z / r),
                "co-latitude (spherical)",
            )

            ph = expression(
                R"\phi",
                sympy.atan2(y, x),
                "longitude (radians)",
            )

            # Spherical coordinate matrix (mesh.R - backward compatible)
            self._R = sympy.Matrix([[r, th, ph]])
            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, \phi")])

            # Geographic coordinates in terms of Cartesian (symbolic)
            # Longitude (degrees East)
            lon_rad = sympy.atan2(y, x)
            lon_deg = lon_rad * 180 / sympy.pi

            # For latitude, we use a simplified symbolic form
            # (actual geodetic latitude requires iterative solution, handled in accessor)
            rxy = sympy.sqrt(x**2 + y**2)
            lat_approx = sympy.atan2(z, rxy) * 180 / sympy.pi

            # Depth (simplified - proper depth calculated in accessor)
            depth_approx = self.ellipsoid['a'] - r

            # Geographic coordinate expressions (approximations for symbolic work)
            lambda_lon = expression(R"\lambda_{lon}", lon_deg, "Longitude (degrees East)")
            lambda_lat = expression(R"\lambda_{lat}", lat_approx, "Latitude (degrees North)")
            lambda_d = expression(R"\lambda_d", depth_approx, "Depth (km below surface)")

            self._geo_coords = sympy.Matrix([[lambda_lon, lambda_lat, lambda_d]])

            # Basis vectors for geographic system
            # Following user's naming: unit_WE (West to East), unit_SN (South to North), unit_down

            # Spherical basis vectors (for reference)
            r1 = self._r[1]  # theta
            r2 = self._r[2]  # phi

            rz = sympy.sqrt(x**2 + y**2)
            r_x_rz = sympy.sqrt((x**2 + y**2 + z**2) * (x**2 + y**2))

            # Spherical rotation matrix (same as SPHERICAL case)
            rRotN = sympy.Matrix(
                [
                    [
                        x / r,
                        y / r,
                        z / r,
                    ],
                    [
                        (x * z) / r_x_rz,
                        (y * z) / r_x_rz,
                        -(x**2 + y**2) / r_x_rz,
                    ],
                    [
                        -y / rz,
                        +x / rz,
                        self.independent_of_N,
                    ],
                ]
            )

            # Geographic basis vectors (stored separately)
            # unit_down: radial direction (into planet, positive downward)
            self._unit_down = -rRotN[0, :]  # Negative of radial outward

            # unit_SN: South to North (meridional direction, positive northward)
            self._unit_SN = -rRotN[1, :]  # Negative of colatitude direction

            # unit_WE: West to East (azimuthal direction, positive eastward)
            self._unit_WE = rRotN[2, :]  # Azimuthal direction

            # Use spherical rotation matrix for mesh.R coordinate system
            self._rRotN = rRotN
            self._xRotN = sympy.eye(self.mesh.dim)


        else:  # Cartesian by default
            self.type = f"Cartesian {self.mesh.dim}D"

            self._X = self._N  # .copy()
            self._x = self._X

            self._xRotN = sympy.eye(self.mesh.dim)
            self._rRotN = sympy.eye(self.mesh.dim)

        # For all meshes: Mark as scaled if the mesh has a model with units
        # Note: Coordinates are already in model units, but we set _scaled flag
        # to indicate the mesh is unit-aware
        self._apply_units_scaling()

        # Create geographic accessor if this is a GEOGRAPHIC coordinate system
        if system == CoordinateSystemType.GEOGRAPHIC:
            self._geo_accessor = GeographicCoordinateAccessor(self)
        else:
            self._geo_accessor = None

        return

    def _apply_units_scaling(self):
        """Mark coordinate system as scaled if model has units."""
        try:
            # Get the model from the mesh
            if hasattr(self.mesh, '_model') and self.mesh._model is not None:
                model = self.mesh._model
            else:
                # Fall back to default model if mesh doesn't have one
                import underworld3 as uw
                model = uw.get_default_model()

            # Check if the model has units scaling enabled
            if not model.has_units():
                self._scaled = False
                return  # No scaling to apply

            # Get fundamental scales from the model
            scales = model.get_fundamental_scales()

            # Set scaling information (but don't transform coordinates)
            if 'length' in scales:
                length_scale = scales['length']

                # Get scale factor as dimensionless number in base SI units
                if hasattr(length_scale, 'to_base_units'):
                    # Convert to base SI units first, then get magnitude
                    scale_factor = length_scale.to_base_units().magnitude
                elif hasattr(length_scale, 'magnitude'):
                    scale_factor = length_scale.magnitude
                else:
                    scale_factor = float(length_scale)

                # Mark coordinate system as scaled
                # Note: We don't transform mesh.X because coordinates are already in model units
                self._scaled = True
                self._length_scale = scale_factor

                # Store scale factors for potential debugging
                self._scale_factors = {
                    'length': scale_factor,
                    'source': f"model '{model.name}' length scale"
                }
            else:
                self._scaled = False

        except Exception as e:
            # If scaling fails, just continue without scaling
            # This ensures backward compatibility
            self._scaled = False
            # Optionally log the error for debugging
            # print(f"Units scaling not applied: {e}")
            pass

    # === Coordinate Data Access (mesh.X interface) ===

    def __getitem__(self, idx):
        """Support mesh.X[0] for x-coordinate access."""
        return self._X[idx]

    def __iter__(self):
        """Support x, y = mesh.X unpacking."""
        return iter(self._X)

    def __len__(self):
        """Support len(mesh.X)."""
        return len(self._X)

    @property
    def coords(self):
        """
        Coordinate data array in physical units.

        Returns the mesh node coordinates, applying scaling if the mesh has
        reference quantities set. When mesh.units is specified, returns a
        UnitAwareArray.

        Returns:
            numpy.ndarray or UnitAwareArray: Node coordinates
        """
        model_coords = self.mesh._coords

        # Apply scaling to convert model coordinates to physical coordinates
        if hasattr(self, '_scaled') and self._scaled:
            scale_factor = self._length_scale
            coords = model_coords * scale_factor
        else:
            coords = model_coords

        # Wrap with unit-aware array if units are specified
        if self.mesh.units is not None:
            from underworld3.utilities.unit_aware_array import UnitAwareArray
            # Extract just the unit part if it's a Quantity (has .units attribute)
            units = self.mesh.units.units if hasattr(self.mesh.units, 'units') else self.mesh.units
            # Convert Unit to string for UnitAwareArray
            units = str(units) if units is not None else None
            return UnitAwareArray(coords, units=units)

        return coords

    @property
    def units(self):
        """
        Coordinate units.

        Returns the units for the coordinate system. This is the same as mesh.units
        and indicates what physical units the coordinates are expressed in.

        Returns:
            str or None: Coordinate units (e.g., 'km', 'm', 'degrees')
        """
        return self.mesh.units

    @property
    def shape(self):
        """Shape of the symbolic coordinate matrix."""
        return self._X.shape

    # === SymPy Integration (for mathematical operations) ===

    def _sympify_(self):
        """
        Tell SymPy how to convert this object to a SymPy expression.

        This enables CoordinateSystem to work seamlessly with SymPy operations
        like diff, jacobian, and arithmetic operations.
        """
        return self._X

    def __sympy__(self):
        """Alternative SymPy conversion protocol."""
        return self._X

    # === SymPy Type Checking Properties ===
    # These allow SymPy to treat CoordinateSystem correctly in expressions

    @property
    def is_symbol(self):
        """SymPy type check - CoordinateSystem contains symbols but is not itself a symbol."""
        return False

    @property
    def is_Matrix(self):
        """SymPy type check - CoordinateSystem behaves like a Matrix."""
        return True

    @property
    def is_scalar(self):
        """SymPy type check - CoordinateSystem is a Matrix, not a scalar."""
        return False

    @property
    def is_number(self):
        """SymPy type check - CoordinateSystem is not a number."""
        return False

    @property
    def is_commutative(self):
        """SymPy type check - delegate to underlying matrix."""
        return self._X.is_commutative if hasattr(self._X, 'is_commutative') else True

    def __getattr__(self, name):
        """
        Delegate SymPy-specific attributes to the underlying symbolic matrix.

        This allows CoordinateSystem to be used transparently in SymPy operations
        by forwarding attribute access to _X when the attribute doesn't exist
        on CoordinateSystem itself.
        """
        # Prevent infinite recursion for _X access
        if name == '_X':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_X'")

        # Try to get the attribute from the underlying symbolic matrix
        try:
            return getattr(self._X, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # === Arithmetic Operations (delegate to symbolic matrix) ===

    def __add__(self, other):
        """Support mesh.X + other."""
        return self._X + other

    def __radd__(self, other):
        """Support other + mesh.X."""
        return other + self._X

    def __sub__(self, other):
        """Support mesh.X - other."""
        return self._X - other

    def __rsub__(self, other):
        """Support other - mesh.X."""
        return other - self._X

    def __mul__(self, other):
        """Support mesh.X * other."""
        return self._X * other

    def __rmul__(self, other):
        """Support other * mesh.X."""
        return other * self._X

    def __truediv__(self, other):
        """Support mesh.X / other."""
        return self._X / other

    def __rtruediv__(self, other):
        """Support other / mesh.X."""
        return other / self._X

    def __pow__(self, other):
        """Support mesh.X ** other."""
        return self._X ** other

    def __neg__(self):
        """Support -mesh.X."""
        return -self._X

    # === Original Properties (for internal use) ===

    @property
    def X(self) -> sympy.Matrix:
        return self._X

    @property
    def x(self) -> sympy.Matrix:
        return self._x

    @property
    def N(self) -> sympy.Matrix:
        return self._N

    @property
    def R(self) -> sympy.Matrix:
        return self._R

    @property
    def r(self) -> sympy.Matrix:
        return self._r

    @property  # alias for backward compat
    def xR(self) -> sympy.Matrix:
        return self._R

    @property
    def geo(self):
        """
        Geographic coordinate accessor for GEOGRAPHIC coordinate systems.

        Returns GeographicCoordinateAccessor for longitude, latitude, depth coordinates
        and geographic basis vectors.

        Raises AttributeError if coordinate system is not GEOGRAPHIC.

        Examples:
            lon = mesh.geo.lon         # Longitude data array
            lat = mesh.geo.lat         # Latitude data array
            depth = mesh.geo.depth     # Depth data array
            λ_lon, λ_lat, λ_d = mesh.geo[:]  # Symbolic coordinates
            unit_east = mesh.geo.unit_east    # East basis vector
        """
        if self._geo_accessor is None:
            raise AttributeError(
                f"Geographic coordinate accessor is only available for GEOGRAPHIC coordinate systems. "
                f"Current coordinate system type: {self.coordinate_type}"
            )
        return self._geo_accessor

    @property
    def rRotN(self) -> sympy.Matrix:
        return self._rRotN

    @property
    def xRotN(self) -> sympy.Matrix:
        return self._xRotN

    @property
    def unit_e_0(self) -> sympy.Matrix:
        return self._rRotN[0, :]

    @property
    def unit_e_1(self) -> sympy.Matrix:
        return self._rRotN[1, :]

    @property
    def unit_e_2(self) -> sympy.Matrix:
        if self.mesh.dim == 3:
            return self._rRotN[2, :]
        else:
            return None

    @property
    def unit_i(self) -> sympy.Matrix:
        return self._xRotN[0, :]

    @property
    def unit_j(self) -> sympy.Matrix:
        return self._xRotN[1, :]

    @property
    def unit_k(self) -> sympy.Matrix:
        if self.mesh.dim == 3:
            return self._xRotN[2, :]
        else:
            return None

    # Should validate on dim
    def unit_ijk(self, dirn) -> sympy.Matrix:
        if dirn <= self.mesh.dim:
            return self._xRotN[dirn, :]
        else:
            return None

    # Geometric direction properties for different coordinate systems
    @property
    def unit_vertical(self) -> sympy.Matrix:
        """Primary vertical direction for this coordinate system"""
        if self.coordinate_type in [CoordinateSystemType.CARTESIAN]:
            # In Cartesian, vertical is the last coordinate direction
            if self.mesh.dim == 2:
                return self.unit_e_1  # y-direction in 2D
            else:
                return self.unit_e_2  # z-direction in 3D
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            # In cylindrical 2D, "vertical" is ambiguous but typically means Cartesian y
            return sympy.Matrix([0, 1])
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            # In spherical, "vertical" typically means radial outward
            return self.unit_e_0
        else:
            raise NotImplementedError(f"unit_vertical not defined for coordinate system {self.coordinate_type}")

    @property 
    def unit_horizontal(self) -> sympy.Matrix:
        """Primary horizontal direction for this coordinate system"""
        if self.coordinate_type in [CoordinateSystemType.CARTESIAN]:
            return self.unit_e_0  # x-direction
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            # In cylindrical, horizontal could be radial or tangential - choose radial as primary
            return self.unit_e_0  # radial direction
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            # In spherical, horizontal is typically tangential (theta direction)
            return self.unit_e_1  # meridional direction
        else:
            raise NotImplementedError(f"unit_horizontal not defined for coordinate system {self.coordinate_type}")

    @property
    def unit_horizontal_0(self) -> sympy.Matrix:
        """First horizontal direction (alias for unit_horizontal)"""
        return self.unit_horizontal

    @property
    def unit_horizontal_1(self) -> sympy.Matrix:
        """Second horizontal direction (for 3D systems)"""
        if self.coordinate_type in [CoordinateSystemType.CARTESIAN]:
            if self.mesh.dim >= 2:
                return self.unit_e_1  # y-direction in 3D Cartesian
            else:
                raise ValueError("unit_horizontal_1 not available in 1D")
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            return self.unit_e_1  # tangential direction
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_2  # azimuthal direction
        else:
            raise NotImplementedError(f"unit_horizontal_1 not defined for coordinate system {self.coordinate_type}")

    @property
    def unit_radial(self) -> sympy.Matrix:
        """Radial direction (for cylindrical/spherical coordinate systems)"""
        if self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D,
                                     CoordinateSystemType.CYLINDRICAL3D]:
            return self.unit_e_0
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_0
        else:
            raise NotImplementedError(f"unit_radial not defined for coordinate system {self.coordinate_type}")

    @property
    def unit_tangential(self) -> sympy.Matrix:
        """Tangential direction (for cylindrical coordinate systems)"""
        if self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D,
                                     CoordinateSystemType.CYLINDRICAL3D]:
            return self.unit_e_1
        else:
            raise NotImplementedError(f"unit_tangential not defined for coordinate system {self.coordinate_type}")

    @property
    def unit_meridional(self) -> sympy.Matrix:
        """Meridional direction (for spherical coordinate systems)"""
        if self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_1
        else:
            raise NotImplementedError(f"unit_meridional not defined for coordinate system {self.coordinate_type}")

    @property
    def unit_azimuthal(self) -> sympy.Matrix:
        """Azimuthal direction (for spherical coordinate systems)"""
        if self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_2
        else:
            raise NotImplementedError(f"unit_azimuthal not defined for coordinate system {self.coordinate_type}")

    @property
    def geometric_dimension_names(self) -> list:
        """Names of geometric dimensions for this coordinate system"""
        if self.coordinate_type in [CoordinateSystemType.CARTESIAN]:
            if self.mesh.dim == 2:
                return ['horizontal', 'vertical']
            else:
                return ['horizontal_x', 'horizontal_y', 'vertical']
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            return ['radial', 'tangential']
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL3D]:
            return ['radial', 'tangential', 'vertical']
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return ['radial', 'meridional', 'azimuthal']
        else:
            return [f'dimension_{i}' for i in range(self.mesh.dim)]

    @property
    def primary_directions(self) -> dict:
        """Dictionary of all available geometric directions for this mesh type"""
        directions = {
            'unit_e_0': self.unit_e_0,
            'unit_e_1': self.unit_e_1,
        }
        if self.mesh.dim >= 3:
            directions['unit_e_2'] = self.unit_e_2

        # Add coordinate-system-specific directions
        try:
            directions['unit_horizontal'] = self.unit_horizontal
            directions['unit_horizontal_0'] = self.unit_horizontal_0
        except NotImplementedError:
            pass

        try:
            directions['unit_horizontal_1'] = self.unit_horizontal_1  
        except (NotImplementedError, ValueError):
            pass

        try:
            directions['unit_vertical'] = self.unit_vertical
        except NotImplementedError:
            pass

        try:
            directions['unit_radial'] = self.unit_radial
        except NotImplementedError:
            pass

        try:
            directions['unit_tangential'] = self.unit_tangential
        except NotImplementedError:
            pass

        try:
            directions['unit_meridional'] = self.unit_meridional
        except NotImplementedError:
            pass

        try:
            directions['unit_azimuthal'] = self.unit_azimuthal
        except NotImplementedError:
            pass

        return directions

    def create_line_sample(self, start_point, direction_vector, length, num_points=50):
        """
        Create sample points along a line defined by sympy expressions.
        
        Parameters
        ----------
        start_point : list or numpy.ndarray
            Starting point coordinates in Cartesian space
        direction_vector : sympy.Matrix  
            Direction vector (should be unit vector for accurate length)
        length : float
            Length of the line to sample
        num_points : int, optional
            Number of sample points to generate
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'cartesian_coords': numpy array of Cartesian coordinates for global_evaluate()
            - 'natural_coords': numpy array of natural coordinates for plotting
            - 'parameters': numpy array of parameter values along the line (0 to length)
        """
        import numpy as np
        
        # Create parameter values along the line
        t_values = np.linspace(0, length, num_points)
        
        # Convert start point to numpy array
        start_point = np.array(start_point)
        if len(start_point) != self.mesh.dim:
            raise ValueError(f"Start point must have {self.mesh.dim} coordinates for {self.mesh.dim}D mesh")
        
        # Generate Cartesian coordinates by evaluating the direction vector
        cartesian_coords = np.zeros((num_points, self.mesh.dim))
        
        # Get coordinate symbols
        coord_symbols = list(self.mesh.X)
        
        for i, t in enumerate(t_values):
            # Current point = start + t * direction
            current_cartesian = start_point.copy()
            
            # Evaluate direction vector at start point to get Cartesian direction
            direction_at_start = direction_vector
            for j, symbol in enumerate(coord_symbols):
                direction_at_start = direction_at_start.subs(symbol, start_point[j])
            
            # Convert to numpy for arithmetic
            direction_vals = np.array([float(val) for val in direction_at_start])
            current_cartesian = current_cartesian + t * direction_vals
            
            cartesian_coords[i] = current_cartesian
        
        # Convert Cartesian coordinates to natural coordinates
        natural_coords = self._cartesian_to_natural_coords(cartesian_coords)
        
        return {
            'cartesian_coords': cartesian_coords,
            'natural_coords': natural_coords,
            'parameters': t_values
        }
    
    def _cartesian_to_natural_coords(self, cartesian_coords):
        """
        Convert Cartesian coordinates to natural coordinate system.
        
        Parameters
        ----------
        cartesian_coords : numpy.ndarray
            Array of Cartesian coordinates (N_points, dim)
            
        Returns
        -------
        numpy.ndarray
            Array of natural coordinates (N_points, dim)
        """
        import numpy as np
        
        if self.coordinate_type == CoordinateSystemType.CARTESIAN:
            # For Cartesian, natural coordinates are the same as Cartesian
            return cartesian_coords.copy()
        
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            # Convert (x, y) to (r, theta)
            x = cartesian_coords[:, 0]
            y = cartesian_coords[:, 1] 
            
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            natural_coords = np.column_stack([r, theta])
            return natural_coords
        
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL3D]:
            # Convert (x, y, z) to (r, theta, z)
            x = cartesian_coords[:, 0]
            y = cartesian_coords[:, 1]
            z = cartesian_coords[:, 2]
            
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            natural_coords = np.column_stack([r, theta, z])
            return natural_coords
        
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            # Convert (x, y, z) to (r, theta, phi)
            x = cartesian_coords[:, 0]
            y = cartesian_coords[:, 1]
            z = cartesian_coords[:, 2]
            
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / (r + 1e-16))  # colatitude (0 to pi)
            phi = np.arctan2(y, x)  # azimuth (-pi to pi)
            
            natural_coords = np.column_stack([r, theta, phi])
            return natural_coords
        
        else:
            # For unknown coordinate systems, return Cartesian coordinates
            return cartesian_coords.copy()
    
    def create_profile_sample(self, profile_type, **params):
        """
        Create sample points for common profile types in this coordinate system.
        
        Parameters
        ----------
        profile_type : str
            Type of profile to create. Options depend on coordinate system:
            - Cartesian: 'horizontal', 'vertical', 'diagonal'
            - Cylindrical: 'radial', 'tangential', 'vertical'  
            - Spherical: 'radial', 'meridional', 'azimuthal'
        **params
            Profile-specific parameters (see individual profile documentation)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'cartesian_coords': numpy array of Cartesian coordinates for global_evaluate()
            - 'natural_coords': numpy array of natural coordinates for plotting
            - 'parameters': numpy array of parameter values along the profile
        """
        
        if self.coordinate_type == CoordinateSystemType.CARTESIAN:
            return self._create_cartesian_profile(profile_type, **params)
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            return self._create_cylindrical_profile(profile_type, **params)
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self._create_spherical_profile(profile_type, **params)
        else:
            raise NotImplementedError(f"Profile sampling not implemented for coordinate system {self.coordinate_type}")
    
    def _create_cartesian_profile(self, profile_type, **params):
        """Create profiles for Cartesian coordinate systems"""
        import numpy as np
        
        num_points = params.get('num_points', 50)
        
        if profile_type == 'horizontal':
            # Horizontal line at specified y-position
            y_position = params.get('y_position', 0.5)
            x_range = params.get('x_range', (0.0, 1.0))
            
            x_values = np.linspace(x_range[0], x_range[1], num_points)
            if self.mesh.dim == 2:
                cartesian_coords = np.column_stack([x_values, np.full(num_points, y_position)])
            else:  # 3D
                z_position = params.get('z_position', 0.5)
                cartesian_coords = np.column_stack([x_values, np.full(num_points, y_position), np.full(num_points, z_position)])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': cartesian_coords.copy(),  # Same for Cartesian
                'parameters': x_values
            }
        
        elif profile_type == 'vertical':
            # Vertical line at specified x-position
            x_position = params.get('x_position', 0.5)
            if self.mesh.dim == 2:
                y_range = params.get('y_range', (0.0, 1.0))
                y_values = np.linspace(y_range[0], y_range[1], num_points)
                cartesian_coords = np.column_stack([np.full(num_points, x_position), y_values])
                return {
                    'cartesian_coords': cartesian_coords,
                    'natural_coords': cartesian_coords.copy(),
                    'parameters': y_values
                }
            else:  # 3D
                y_position = params.get('y_position', 0.5)
                z_range = params.get('z_range', (0.0, 1.0))
                z_values = np.linspace(z_range[0], z_range[1], num_points)
                cartesian_coords = np.column_stack([np.full(num_points, x_position), np.full(num_points, y_position), z_values])
                return {
                    'cartesian_coords': cartesian_coords,
                    'natural_coords': cartesian_coords.copy(),
                    'parameters': z_values
                }
        
        elif profile_type == 'diagonal':
            # Diagonal line from start to end point
            start_point = params.get('start_point', [0.0] * self.mesh.dim)
            end_point = params.get('end_point', [1.0] * self.mesh.dim)
            
            start_point = np.array(start_point)
            end_point = np.array(end_point)
            
            t_values = np.linspace(0, 1, num_points)
            cartesian_coords = np.array([start_point + t * (end_point - start_point) for t in t_values])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': cartesian_coords.copy(),
                'parameters': t_values
            }
        
        else:
            raise ValueError(f"Unknown Cartesian profile type: {profile_type}")
    
    def _create_cylindrical_profile(self, profile_type, **params):
        """Create profiles for cylindrical coordinate systems"""
        import numpy as np
        
        num_points = params.get('num_points', 50)
        
        if profile_type == 'radial':
            # Radial line at specified angle
            theta = params.get('theta', 0.0)  # Angle in radians
            r_range = params.get('r_range', (0.5, 1.0))
            
            r_values = np.linspace(r_range[0], r_range[1], num_points)
            
            # Convert to Cartesian coordinates
            x_values = r_values * np.cos(theta)
            y_values = r_values * np.sin(theta)
            cartesian_coords = np.column_stack([x_values, y_values])
            
            # Natural coordinates
            natural_coords = np.column_stack([r_values, np.full(num_points, theta)])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': natural_coords,
                'parameters': r_values
            }
        
        elif profile_type == 'tangential':
            # Tangential (circular arc) at specified radius
            radius = params.get('radius', 0.75)
            theta_range = params.get('theta_range', (0.0, 2*np.pi))
            
            theta_values = np.linspace(theta_range[0], theta_range[1], num_points)
            
            # Convert to Cartesian coordinates
            x_values = radius * np.cos(theta_values)
            y_values = radius * np.sin(theta_values)
            cartesian_coords = np.column_stack([x_values, y_values])
            
            # Natural coordinates
            natural_coords = np.column_stack([np.full(num_points, radius), theta_values])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': natural_coords,
                'parameters': theta_values
            }
        
        elif profile_type == 'vertical':
            # Vertical line in Cartesian y-direction
            x_position = params.get('x_position', 0.0)
            y_range = params.get('y_range', (0.0, 1.0))
            
            y_values = np.linspace(y_range[0], y_range[1], num_points)
            cartesian_coords = np.column_stack([np.full(num_points, x_position), y_values])
            
            # Convert to natural coordinates
            natural_coords = self._cartesian_to_natural_coords(cartesian_coords)
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': natural_coords,
                'parameters': y_values
            }
        
        else:
            raise ValueError(f"Unknown cylindrical profile type: {profile_type}")
    
    def _create_spherical_profile(self, profile_type, **params):
        """Create profiles for spherical coordinate systems"""
        import numpy as np
        
        num_points = params.get('num_points', 50)
        
        if profile_type == 'radial':
            # Radial line at specified theta, phi
            theta = params.get('theta', np.pi/2)  # Colatitude (0 to pi)
            phi = params.get('phi', 0.0)  # Azimuth (-pi to pi)
            r_range = params.get('r_range', (0.5, 1.0))
            
            r_values = np.linspace(r_range[0], r_range[1], num_points)
            
            # Convert to Cartesian coordinates
            x_values = r_values * np.sin(theta) * np.cos(phi)
            y_values = r_values * np.sin(theta) * np.sin(phi)
            z_values = r_values * np.cos(theta)
            cartesian_coords = np.column_stack([x_values, y_values, z_values])
            
            # Natural coordinates
            natural_coords = np.column_stack([r_values, np.full(num_points, theta), np.full(num_points, phi)])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': natural_coords,
                'parameters': r_values
            }
        
        elif profile_type == 'meridional':
            # Meridional line (constant phi, varying theta) at specified radius
            radius = params.get('radius', 0.75)
            phi = params.get('phi', 0.0)
            theta_range = params.get('theta_range', (0.0, np.pi))
            
            theta_values = np.linspace(theta_range[0], theta_range[1], num_points)
            
            # Convert to Cartesian coordinates
            x_values = radius * np.sin(theta_values) * np.cos(phi)
            y_values = radius * np.sin(theta_values) * np.sin(phi)
            z_values = radius * np.cos(theta_values)
            cartesian_coords = np.column_stack([x_values, y_values, z_values])
            
            # Natural coordinates
            natural_coords = np.column_stack([np.full(num_points, radius), theta_values, np.full(num_points, phi)])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': natural_coords,
                'parameters': theta_values
            }
        
        elif profile_type == 'azimuthal':
            # Azimuthal line (constant theta, varying phi) at specified radius
            radius = params.get('radius', 0.75)
            theta = params.get('theta', np.pi/2)
            phi_range = params.get('phi_range', (0.0, 2*np.pi))
            
            phi_values = np.linspace(phi_range[0], phi_range[1], num_points)
            
            # Convert to Cartesian coordinates
            x_values = radius * np.sin(theta) * np.cos(phi_values)
            y_values = radius * np.sin(theta) * np.sin(phi_values)
            z_values = radius * np.full(num_points, np.cos(theta))
            cartesian_coords = np.column_stack([x_values, y_values, z_values])
            
            # Natural coordinates
            natural_coords = np.column_stack([np.full(num_points, radius), np.full(num_points, theta), phi_values])
            
            return {
                'cartesian_coords': cartesian_coords,
                'natural_coords': natural_coords,  
                'parameters': phi_values
            }
        
        else:
            raise ValueError(f"Unknown spherical profile type: {profile_type}")

    def zero_matrix(self, shape):
        """Matrix of spatial coordinates equivalent to zeros (but still dependent on X) -
        Add this when you have a matrix with a mix of constants and functions - sympy / numpy
        can become upset if the constants are not specific functions too.
        """

        # Direct construction to avoid SymPy Matrix scalar multiplication issues
        Z = sympy.Matrix.ones(*shape)
        Z = sympy.Matrix(*shape, lambda i, j: Z[i, j] * self.independent_of_N)

        return Z

    ## Here we can add an ipython_display method to add the class documentation and a description of the
    ## entities that are defined (use sympy printing to make that work automatically)
