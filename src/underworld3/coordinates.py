r"""
Coordinate system definitions and coordinate variable types.

This module provides coordinate system infrastructure for Underworld3,
enabling both Cartesian and curvilinear (cylindrical, spherical) coordinate
systems with proper differential geometry support.

Key Components
--------------
CoordinateSystemType : enum
    Available coordinate system types (Cartesian, Cylindrical, Spherical).
UWCoordinate : class
    A SymPy-compatible coordinate variable (x, y, z) that integrates
    with mesh evaluation and differentiation.
CoordinateSystem : class
    Factory for creating coordinate systems with proper vector calculus
    operators (gradient, divergence, curl) in the chosen geometry.

The coordinate infrastructure ensures that symbolic expressions using
mesh coordinates integrate seamlessly with SymPy's differentiation
machinery while providing efficient numerical evaluation.

See Also
--------
underworld3.maths.vector_calculus : Vector calculus in curvilinear systems.
underworld3.discretisation : Mesh classes using coordinate systems.
"""
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


# =============================================================================
# UWCoordinate - Cartesian coordinate variable (x, y, z)
# =============================================================================

from sympy.vector.scalar import BaseScalar


class UWCoordinate(BaseScalar):
    """
    A Cartesian coordinate variable (x, y, or z).

    This class represents a mesh coordinate that:
    - Subclasses BaseScalar for full SymPy differentiation compatibility
    - Is recognizable by type (isinstance check) for unwrap/evaluate logic
    - Parallels MeshVariable pattern: .sym for symbolic, .data for numeric
    - Works transparently with sympy.diff() for expressions containing N.x, N.y, N.z

    The key insight is that by subclassing BaseScalar and implementing __eq__/__hash__
    to match the original N.x/N.y/N.z objects, SymPy's differentiation machinery
    recognizes UWCoordinate as equivalent to the original BaseScalar.

    Parameters
    ----------
    index : int
        Index in the coordinate system (0, 1, or 2)
    system : CoordSys3D
        The SymPy coordinate system (mesh.N)
    mesh : Mesh, optional
        Parent mesh object
    axis_index : int, optional
        Axis index (0=x, 1=y, 2=z) - same as index but kept for API consistency

    Examples
    --------
    >>> x, y = mesh.X  # UWCoordinate objects
    >>> x.sym          # Returns self (UWCoordinate IS a BaseScalar)
    >>> x.data         # numpy array of x-coordinates
    >>> r = sympy.sqrt(x**2 + y**2)  # Works in expressions
    >>> sympy.diff(v.sym[0], y)      # Works correctly! (key improvement)
    """

    # Track instances for debugging
    _coordinate_count = 0

    def __new__(cls, index, system, pretty_str=None, latex_str=None, mesh=None, axis_index=None):
        # Create as a BaseScalar with the same index and system
        obj = BaseScalar.__new__(cls, index, system, pretty_str, latex_str)
        return obj

    def __init__(self, index, system, pretty_str=None, latex_str=None, mesh=None, axis_index=None):
        # Store UW3-specific attributes
        self._mesh = mesh
        self._axis_index = axis_index if axis_index is not None else index
        self._coord_name = pretty_str or f"x_{index}"

        # Cache the original BaseScalar for equality comparison
        # This is what makes sympy.diff() work!
        self._original_base_scalar = system.base_scalars()[index]

        # Track for debugging
        UWCoordinate._coordinate_count += 1
        self._instance_id = UWCoordinate._coordinate_count

    def __eq__(self, other):
        """
        Equal to the original BaseScalar from the SAME coordinate system.

        This is the key to making sympy.diff() work - when SymPy checks if
        the differentiation variable matches symbols in the expression,
        this makes UWCoordinate match the original BaseScalar.

        IMPORTANT: We only match BaseScalars from the SAME mesh's coordinate
        system using object identity (`is`), not name comparison. This prevents
        cross-mesh coordinate pollution where coordinates from different meshes
        would be treated as equal due to having the same name "N.x".
        """
        if other is self._original_base_scalar:
            return True
        if isinstance(other, UWCoordinate) and hasattr(other, '_original_base_scalar'):
            if other._original_base_scalar is self._original_base_scalar:
                return True
        # DON'T fall back to BaseScalar.__eq__ which compares by name!
        # This caused cross-mesh coordinate pollution (issue discovered 2025-12-15)
        return False

    def __hash__(self):
        """
        Hash must match _original_base_scalar's hash for SymPy compatibility.

        Since __eq__ returns True when comparing to the original BaseScalar,
        we MUST return the same hash (Python requirement: a == b → hash(a) == hash(b)).

        This is critical for SymPy's differentiation to work correctly -
        sympy.diff() uses hash-based lookup to find matching symbols.

        Note: Cross-mesh coordinate collision is prevented by __eq__ checking
        object identity of _original_base_scalar, not by different hashes.
        """
        return hash(self._original_base_scalar)

    def _numpycode(self, printer):
        """
        NumPy code generation for lambdify().

        Returns a unique dummy name that lambdify will map to array columns.
        Uses the coordinate's internal index to generate a consistent name.
        """
        # Use the short coordinate name (x, y, z) based on axis index
        coord_names = ['_uw_x', '_uw_y', '_uw_z']
        return coord_names[self._axis_index]

    def _lambdacode(self, printer):
        """Lambda code generation."""
        coord_names = ['_uw_x', '_uw_y', '_uw_z']
        return coord_names[self._axis_index]

    def _pythoncode(self, printer):
        """Python code generation."""
        coord_names = ['_uw_x', '_uw_y', '_uw_z']
        return coord_names[self._axis_index]

    @property
    def sym(self):
        """
        Symbolic representation for JIT/symbolic operations.

        For UWCoordinate (which IS a BaseScalar), this returns self.
        This maintains API compatibility with code that accesses .sym
        """
        return self._original_base_scalar

    @property
    def data(self):
        """
        Coordinate values from mesh.

        Returns dimensional values if mesh has units, ND otherwise.
        Mirrors MeshVariable.data pattern.

        Returns
        -------
        numpy.ndarray
            Coordinate values for this axis at all mesh nodes
        """
        if self._mesh is None:
            raise ValueError("UWCoordinate not attached to a mesh")
        return self._mesh.X.coords[:, self._axis_index]

    @property
    def mesh(self):
        """Parent mesh."""
        return self._mesh

    @property
    def axis(self):
        """Axis index (0=x, 1=y, 2=z)."""
        return self._axis_index

    @property
    def _base_scalar(self):
        """Backward compatibility: return the original BaseScalar."""
        return self._original_base_scalar

    @property
    def units(self):
        """
        Units of this coordinate, delegated from the original BaseScalar.

        The mesh's patch_coordinate_units() sets ._units on mesh.N.x, mesh.N.y, etc.
        UWCoordinate wraps these, so we delegate to get the units.
        """
        return getattr(self._original_base_scalar, '_units', None)

    # NOTE: We intentionally do NOT define _units as a property here.
    # SymPy's internal machinery (derive_by_array, etc.) does hasattr checks
    # that can be confused by properties. Instead, get_units() should check
    # the .units property explicitly, or access _original_base_scalar._units.
    # The .units property above is sufficient for the public API.

    @property
    def _ccodestr(self):
        """
        Delegate C code string to the original BaseScalar.

        The mesh sets _ccodestr on the original N.x, N.y, N.z objects
        for JIT code generation (e.g., "petsc_x[0]", "petsc_x[1]").
        UWCoordinate must expose the same attribute for JIT to work.
        """
        return self._original_base_scalar._ccodestr

    @_ccodestr.setter
    def _ccodestr(self, value):
        """Allow setting _ccodestr (propagates to original BaseScalar)."""
        self._original_base_scalar._ccodestr = value

    def _ccode(self, printer, **kwargs):
        """
        C code representation for JIT compilation.

        The SymPy CCodePrinter looks for this method on symbols.
        We delegate to the original BaseScalar's _ccodestr.
        """
        return self._ccodestr

    def __repr__(self):
        return f"{self._coord_name}"

    def _latex(self, printer):
        """LaTeX representation for SymPy printing."""
        return r"\mathrm{" + self._coord_name + "}"


# =============================================================================
# Helper function for differentiation with UWCoordinates
# =============================================================================

def uwdiff(expr, *symbols):
    """
    Differentiate an expression with respect to coordinates.

    .. deprecated:: December 2025
        Since UWCoordinate now subclasses BaseScalar with proper __eq__/__hash__,
        you can use ``sympy.diff(expr, y)`` directly. This function is kept for
        backward compatibility but simply delegates to sympy.diff().

    Parameters
    ----------
    expr : sympy.Expr
        The expression to differentiate
    symbols : UWCoordinate or sympy.Symbol
        The variables to differentiate with respect to

    Returns
    -------
    sympy.Expr
        The derivative

    Examples
    --------
    >>> x, y = mesh.X  # UWCoordinates
    >>> v = mesh_variable  # MeshVariable
    >>> # Both now work identically:
    >>> dv_dy = sympy.diff(v.sym[0], y)  # Preferred
    >>> dv_dy = uw.uwdiff(v.sym[0], y)   # Backward compatible
    """
    import warnings
    warnings.warn(
        "uwdiff() is deprecated. Use sympy.diff() directly - "
        "UWCoordinates now work transparently with SymPy differentiation.",
        DeprecationWarning,
        stacklevel=2
    )
    return sympy.diff(expr, *symbols)


class CoordinateSystemType(Enum):
    """
    Coordinate system types for mesh geometry.

    Meshes can have natural coordinate systems that overlay the Cartesian
    coordinate system used internally for solver assembly. The coordinate
    system type determines how vector calculus operators (gradient, divergence,
    curl) are computed.

    Attributes
    ----------
    CARTESIAN : int
        Standard Cartesian coordinates (x, y, z).
    CYLINDRICAL2D : int
        2D cylindrical/polar coordinates (r, theta).
    POLAR : int
        Alias for CYLINDRICAL2D.
    CYLINDRICAL3D : int
        3D cylindrical coordinates (r, theta, z).
    SPHERICAL : int
        Spherical coordinates (r, theta, phi).
    GEOGRAPHIC : int
        Ellipsoidal geographic coordinates (lon, lat, depth) for
        Earth and planetary modeling.

    See Also
    --------
    underworld3.maths.vector_calculus : Operators for each coordinate system.
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
    "WGS84": {
        "a": 6378.137,  # km
        "b": 6356.752,  # km
        "f": 1 / 298.257223563,
        "planet": "Earth",
        "description": "World Geodetic System 1984",
    },
    "GRS80": {
        "a": 6378.137,
        "b": 6356.752,
        "f": 1 / 298.257222101,
        "planet": "Earth",
        "description": "Geodetic Reference System 1980",
    },
    "sphere": {
        "a": 6371.0,  # Mean Earth radius
        "b": 6371.0,
        "f": 0.0,
        "planet": "Earth",
        "description": "Perfect sphere (mean Earth radius)",
    },
    "Mars": {
        "a": 3396.2,
        "b": 3376.2,
        "f": 1 / 169.8,
        "planet": "Mars",
        "description": "Mars ellipsoid",
    },
    "Moon": {
        "a": 1738.1,
        "b": 1736.0,
        "f": 1 / 824.7,
        "planet": "Moon",
        "description": "Moon ellipsoid",
    },
    "Venus": {
        "a": 6051.8,
        "b": 6051.8,
        "f": 0.0,  # Nearly perfect sphere
        "planet": "Venus",
        "description": "Venus ellipsoid",
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
    e2 = 1 - (b / a) ** 2

    # Prime vertical radius of curvature at this latitude
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

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
    e2 = 1 - (b / a) ** 2
    p = np.sqrt(x**2 + y**2)

    # Initial guess for latitude (geocentric)
    lat = np.arctan2(z, p * (1 - e2))

    # Iterate to converge on geodetic latitude
    for i in range(max_iterations):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), p)

        # Check convergence
        if np.abs(lat_new - lat).max() < tolerance:
            break
        lat = lat_new

    # Height above ellipsoid
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N

    # Depth is negative height
    depth = -h

    # Convert to degrees
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)

    return lon_deg, lat_deg, depth


class GeographicCoordinateAccessor:
    r"""
    Geographic coordinates on ellipsoidal (WGS84) meshes.

    This class provides natural coordinate access for geographic meshes,
    including longitude, latitude, and depth data arrays, symbolic coordinates
    for equations (:math:`\lambda_{lon}`, :math:`\lambda_{lat}`, :math:`\lambda_d`),
    and ellipsoidal basis vectors.

    Access via ``mesh.X.geo`` on GEOGRAPHIC meshes. Use ``.view()`` for a
    complete summary of available properties and methods.

    Attributes
    ----------
    lon : ndarray
        Longitude in degrees East (-180 to 180)
    lat : ndarray
        Geodetic latitude in degrees North (-90 to 90)
    depth : ndarray
        Depth below ellipsoid surface in km (positive downward)
    coords : ndarray
        Combined (N, 3) array: [lon, lat, depth]

    Examples
    --------
    >>> geo = mesh.X.geo
    >>> geo.view()               # Show all available properties
    >>> geo.coords               # (N, 3) array of [lon, lat, depth]
    >>> lon, lat, d = geo[:]     # Symbolic coordinates for equations
    >>> geo.unit_down            # Geodetic normal (into planet)

    Notes
    -----
    The 'up' direction is the geodetic normal—perpendicular to the ellipsoid
    surface—NOT the radial direction. At mid-latitudes, these differ by
    approximately 10-11 arcminutes, which matters for regional models.
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
        self._nondimensional = False  # Set by _compute_coordinates

    def _invalidate_cache(self):
        """Mark coordinate cache as invalid (call when mesh coordinates change)."""
        self._cache_valid = False

    def _compute_coordinates(self):
        """Compute geographic coordinates from Cartesian mesh coordinates."""
        if self._cache_valid:
            return

        # Get raw Cartesian coordinates from DM (avoids unit wrapping)
        # The mesh stores nondimensional coords when units are active
        dm_coords = self.mesh.dm.getCoordinates().array.reshape(-1, 3)
        x, y, z = dm_coords[:, 0], dm_coords[:, 1], dm_coords[:, 2]

        # Get ellipsoid parameters
        # Use nondimensional if available (when mesh created with units)
        # Otherwise use raw km values
        ellipsoid = self.cs.ellipsoid
        if "a_nd" in ellipsoid and "b_nd" in ellipsoid:
            # Units were active - use nondimensional ellipsoid
            a = ellipsoid["a_nd"]
            b = ellipsoid["b_nd"]
            self._nondimensional = True
        else:
            # No units - use km values
            a = ellipsoid["a"]
            b = ellipsoid["b"]
            self._nondimensional = False

        # Convert to geographic using our utility function
        # lon and lat are in degrees (angles - no scaling needed)
        # depth is in same units as input (nondimensional or km)
        self._lon_cache, self._lat_cache, self._depth_cache = cartesian_to_geographic(x, y, z, a, b)

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
        """
        Depth below ellipsoid surface (positive downward).

        Returns nondimensional values when units are active, km otherwise.
        Use with units system for proper dimensional output.
        """
        self._compute_coordinates()

        # Return with units if available
        if hasattr(self, "_nondimensional") and self._nondimensional:
            # Check if units system is active for wrapping
            import underworld3 as uw

            model = uw.get_default_model()
            if model is not None and model.has_units():
                from underworld3.utilities.unit_aware_array import UnitAwareArray

                # Get reference length to dimensionalize
                L_ref = self.cs.ellipsoid.get("L_ref_km", 1000)  # km
                # Depth in km = nondimensional * L_ref
                depth_km = self._depth_cache * L_ref
                return UnitAwareArray(depth_km, units="km")

        return self._depth_cache

    @property
    def coords(self):
        """
        Geographic coordinates as (N, 3) array: [lon, lat, depth].

        Returns mesh node coordinates in geographic form, matching the
        layout of `mesh.X.coords` but in (longitude, latitude, depth) format.

        Returns
        -------
        numpy.ndarray
            Shape (N, 3) array where columns are:
            - Column 0: Longitude (degrees East, -180 to 180)
            - Column 1: Latitude (degrees North, -90 to 90)
            - Column 2: Depth (km below ellipsoid surface, positive down)

        Examples
        --------
        >>> geo_coords = mesh.CoordinateSystem.geo.coords
        >>> print(f"Node 0: lon={geo_coords[0,0]:.2f}°, lat={geo_coords[0,1]:.2f}°")

        See Also
        --------
        mesh.X.coords : Cartesian coordinates (x, y, z)
        lon, lat, depth : Individual coordinate arrays
        """
        import numpy as np
        self._compute_coordinates()
        return np.column_stack([self._lon_cache, self._lat_cache, self._depth_cache])

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

    # === Coordinate Conversion Methods ===

    def to_cartesian(self, lon, lat, depth):
        """
        Convert geographic coordinates to Cartesian (x, y, z).

        Uses the mesh's ellipsoid parameters automatically.
        When units are active, depth should be nondimensional and returns
        nondimensional Cartesian coordinates.

        Parameters
        ----------
        lon : float or array_like
            Longitude in degrees East (-180 to 180)
        lat : float or array_like
            Geodetic latitude in degrees North (-90 to 90)
        depth : float or array_like
            Depth below ellipsoid surface (nondimensional if units active, km otherwise)

        Returns
        -------
        tuple
            (x, y, z) coordinates (nondimensional if units active, km otherwise)

        Examples
        --------
        >>> # Import external data in geographic coordinates
        >>> tomo_lon = np.array([136.0, 136.5, 137.0])
        >>> tomo_lat = np.array([-34.0, -33.5, -33.0])
        >>> tomo_depth = np.array([10.0, 20.0, 30.0])  # km or nondimensional
        >>> x, y, z = mesh.geo.to_cartesian(tomo_lon, tomo_lat, tomo_depth)
        """
        # Use nondimensional ellipsoid if available
        ellipsoid = self.cs.ellipsoid
        if "a_nd" in ellipsoid and "b_nd" in ellipsoid:
            a = ellipsoid["a_nd"]
            b = ellipsoid["b_nd"]
        else:
            a = ellipsoid["a"]
            b = ellipsoid["b"]
        return geographic_to_cartesian(lon, lat, depth, a, b)

    def from_cartesian(self, x, y, z):
        """
        Convert Cartesian coordinates to geographic (lon, lat, depth).

        Uses the mesh's ellipsoid parameters automatically.
        When units are active, coordinates should be nondimensional.

        Parameters
        ----------
        x : float or array_like
            X coordinate (nondimensional if units active, km otherwise)
        y : float or array_like
            Y coordinate
        z : float or array_like
            Z coordinate

        Returns
        -------
        tuple
            (lon, lat, depth) where:
            - lon: Longitude in degrees East (-180 to 180)
            - lat: Geodetic latitude in degrees North (-90 to 90)
            - depth: Depth (nondimensional if units active, km otherwise)

        Examples
        --------
        >>> # Convert mesh points to geographic for comparison with data
        >>> x, y, z = mesh.data[:, 0], mesh.data[:, 1], mesh.data[:, 2]
        >>> lon, lat, depth = mesh.geo.from_cartesian(x, y, z)
        """
        # Use nondimensional ellipsoid if available
        ellipsoid = self.cs.ellipsoid
        if "a_nd" in ellipsoid and "b_nd" in ellipsoid:
            a = ellipsoid["a_nd"]
            b = ellipsoid["b_nd"]
        else:
            a = ellipsoid["a"]
            b = ellipsoid["b"]
        return cartesian_to_geographic(x, y, z, a, b)

    def points_to_cartesian(self, points_geo):
        """
        Convert array of geographic points to Cartesian coordinates.

        Convenience method for importing external data.

        Parameters
        ----------
        points_geo : array_like
            Array of shape (N, 3) with columns [lon, lat, depth]
            - lon: Longitude in degrees East
            - lat: Geodetic latitude in degrees North
            - depth: Depth in km below surface

        Returns
        -------
        ndarray
            Array of shape (N, 3) with columns [x, y, z] in km

        Examples
        --------
        >>> # Import seismicity catalog
        >>> eq_llz = np.loadtxt("earthquakes.csv", delimiter=",")  # [lon, lat, depth]
        >>> eq_xyz = mesh.geo.points_to_cartesian(eq_llz)
        >>> # Now use with KDTree or swarm.add_particles_with_coordinates()
        """
        import numpy as np
        points_geo = np.asarray(points_geo)
        lon, lat, depth = points_geo[:, 0], points_geo[:, 1], points_geo[:, 2]
        x, y, z = self.to_cartesian(lon, lat, depth)
        return np.column_stack([x, y, z])

    def points_from_cartesian(self, points_xyz):
        """
        Convert array of Cartesian points to geographic coordinates.

        Parameters
        ----------
        points_xyz : array_like
            Array of shape (N, 3) with columns [x, y, z] in km

        Returns
        -------
        ndarray
            Array of shape (N, 3) with columns [lon, lat, depth]

        Examples
        --------
        >>> # Export mesh coordinates to geographic
        >>> mesh_xyz = mesh.data  # or mesh.CoordinateSystem.coords
        >>> mesh_llz = mesh.geo.points_from_cartesian(mesh_xyz)
        """
        import numpy as np
        points_xyz = np.asarray(points_xyz)
        x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
        lon, lat, depth = self.from_cartesian(x, y, z)
        return np.column_stack([lon, lat, depth])

    def __repr__(self):
        """String representation showing available coordinates and basis vectors."""
        return (
            f"GeographicCoordinates(\n"
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

    def view(self):
        r"""
        Display a formatted summary of available properties and methods.

        This method prints a helpful guide to the geographic coordinate system,
        showing all available data arrays, symbolic coordinates, unit vectors,
        and conversion methods.
        """
        ellipsoid = self.cs.ellipsoid
        info = f"""
Geographic Coordinates (Ellipsoidal)
====================================
Ellipsoid: {ellipsoid.get('description', 'Unknown')}
  Semi-major axis (a): {ellipsoid['a']:.3f} km
  Semi-minor axis (b): {ellipsoid['b']:.3f} km
  Flattening: {ellipsoid['f']:.6f}

Data Arrays (numpy):
  .coords        → (N, 3) array of [lon, lat, depth]
  .lon           → Longitude (degrees East, -180 to 180)
  .lat           → Geodetic latitude (degrees North, -90 to 90)
  .depth         → Depth below ellipsoid surface (km, positive down)

Symbolic Coordinates (for equations):
  .λ_lon         → Symbolic longitude
  .λ_lat         → Symbolic latitude
  .λ_d           → Symbolic depth
  [:]            → Tuple of (λ_lon, λ_lat, λ_d)

Unit Vectors (symbolic, vary with position):
  Primary (physical direction):
    .unit_down     → Into planet (geodetic normal, NOT radial)
    .unit_SN       → South to North (meridional)
    .unit_WE       → West to East (azimuthal)

  Directional aliases:
    .unit_east, .unit_north, .unit_up (= -.unit_down)
    .unit_west, .unit_south

  Coordinate direction aliases:
    .unit_lon, .unit_lat, .unit_depth

Conversion Methods:
  .to_cartesian(lon, lat, depth)    → (x, y, z) in km
  .from_cartesian(x, y, z)          → (lon, lat, depth)
  .points_to_cartesian(arr)         → Convert (N,3) array
  .points_from_cartesian(arr)       → Convert (N,3) array

Rotation Matrix:
  mesh.CoordinateSystem.geoRotN     → 3×3 matrix
    Transforms Cartesian vectors to geographic frame:
    Row 0: geodetic up (NOT radial - accounts for ellipticity)
    Row 1: north (meridional)
    Row 2: east (azimuthal)

Notes:
  - 'Up' is the geodetic normal, perpendicular to the ellipsoid surface
  - At mid-latitudes, this differs from radial by ~10-11 arcminutes
  - This matters for regional models at 10-100 km scale
  - For spherical geometry instead, use RegionalSphericalBox
"""
        print(info)


class SphericalCoordinateAccessor:
    r"""
    Spherical/polar coordinates for spherical and cylindrical meshes.

    This class provides natural coordinate access for spherical (3D) and
    polar/cylindrical (2D) meshes, including radius, angle data arrays,
    symbolic coordinates for equations, and basis vectors.

    Access via ``mesh.X.spherical`` on SPHERICAL or CYLINDRICAL2D meshes.
    Use ``.view()`` for a complete summary of available properties and methods.

    Attributes
    ----------
    r : ndarray
        Radial distance from origin
    theta : ndarray
        Angle in radians:
        - 3D: Colatitude (0 at north pole, π at south pole)
        - 2D: Polar angle from x-axis (standard θ)
    phi : ndarray (3D only)
        Longitude/azimuth in radians (-π to π). Not available for 2D meshes.
    coords : ndarray
        Combined array: [r, θ, φ] for 3D, [r, θ] for 2D

    Examples
    --------
    >>> sph = mesh.X.spherical
    >>> sph.view()               # Show all available properties
    >>> sph.coords               # (N, 3) or (N, 2) coordinate array
    >>> r, theta = sph[:2]       # Works for both 2D and 3D
    >>> sph.unit_r               # Radial unit vector (outward)

    Notes
    -----
    The coordinate convention follows physics conventions:

    - :math:`r`: Radial distance from origin
    - :math:`\theta`: Angle (colatitude in 3D, polar angle in 2D)
    - :math:`\phi`: Longitude/azimuth (3D only)

    For Earth-like applications with geodetic (ellipsoidal) geometry,
    use a GEOGRAPHIC mesh with ``mesh.X.geo`` instead.
    """

    def __init__(self, coordinate_system):
        """
        Initialize spherical/polar coordinate accessor.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The coordinate system object with SPHERICAL or CYLINDRICAL2D type
        """
        self.cs = coordinate_system
        self.mesh = coordinate_system.mesh
        self._dim = self.mesh.dim  # 2 for polar, 3 for spherical

        # Cache for coordinate arrays
        self._r_cache = None
        self._theta_cache = None
        self._phi_cache = None
        self._cache_valid = False

    def _invalidate_cache(self):
        """Mark coordinate cache as invalid (call when mesh coordinates change)."""
        self._cache_valid = False

    def _compute_coordinates(self):
        """Compute spherical/polar coordinates from Cartesian mesh coordinates."""
        if self._cache_valid:
            return

        import numpy as np

        # Get Cartesian coordinates
        coords = self.mesh.CoordinateSystem.coords

        if self._dim == 2:
            # 2D polar: (x, y) → (r, θ)
            x, y = coords[:, 0], coords[:, 1]
            self._r_cache = np.sqrt(x**2 + y**2)
            self._theta_cache = np.arctan2(y, x)
            self._phi_cache = None
        else:
            # 3D spherical: (x, y, z) → (r, θ, φ)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            self._r_cache = np.sqrt(x**2 + y**2 + z**2)
            self._theta_cache = np.arccos(np.clip(z / np.maximum(self._r_cache, 1e-30), -1, 1))
            self._phi_cache = np.arctan2(y, x)

        self._cache_valid = True

    @property
    def r(self):
        """Radial distance from origin."""
        self._compute_coordinates()
        return self._r_cache

    @property
    def theta(self):
        """Colatitude in radians (0 at north pole, π at south pole)."""
        self._compute_coordinates()
        return self._theta_cache

    @property
    def phi(self):
        """Longitude/azimuth in radians (-π to π). Only available for 3D meshes."""
        if self._dim == 2:
            raise AttributeError(
                "phi (azimuthal angle) is not available for 2D polar meshes.\n"
                "2D meshes have only (r, θ) coordinates."
            )
        self._compute_coordinates()
        return self._phi_cache

    @property
    def coords(self):
        r"""
        Spherical/polar coordinates as array.

        Returns mesh node coordinates in spherical/polar form:
        - 3D: (N, 3) array [r, θ, φ]
        - 2D: (N, 2) array [r, θ]

        Returns
        -------
        numpy.ndarray
            For 3D (spherical): Shape (N, 3) array where columns are:

            - Column 0: Radius (same units as mesh)
            - Column 1: Colatitude θ in radians (0 at north pole)
            - Column 2: Longitude φ in radians (-π to π)

            For 2D (polar): Shape (N, 2) array where columns are:

            - Column 0: Radius (same units as mesh)
            - Column 1: Polar angle θ in radians

        Examples
        --------
        >>> sph_coords = mesh.X.spherical.coords
        >>> print(f"Node 0: r={sph_coords[0,0]:.3f}, θ={np.degrees(sph_coords[0,1]):.1f}°")

        See Also
        --------
        mesh.X.coords : Cartesian coordinates (x, y) or (x, y, z)
        r, theta, phi : Individual coordinate arrays
        """
        import numpy as np
        self._compute_coordinates()
        if self._dim == 2:
            return np.column_stack([self._r_cache, self._theta_cache])
        else:
            return np.column_stack([self._r_cache, self._theta_cache, self._phi_cache])

    def __getitem__(self, idx):
        """
        Access symbolic spherical/polar coordinates.

        Returns
        -------
        tuple or scalar
            For 3D: r, θ, φ symbolic coordinates
            For 2D: r, θ symbolic coordinates
        """
        if idx == slice(None, None, None):  # mesh.spherical[:]
            if self._dim == 2:
                return self.cs.R[0, 0], self.cs.R[0, 1]
            else:
                return self.cs.R[0, 0], self.cs.R[0, 1], self.cs.R[0, 2]
        else:
            return self.cs.R[0, idx]

    # === Symbolic coordinate access ===

    @property
    def r_sym(self):
        """Symbolic radial coordinate."""
        return self.cs.R[0, 0]

    @property
    def theta_sym(self):
        """Symbolic polar/colatitude coordinate."""
        return self.cs.R[0, 1]

    @property
    def phi_sym(self):
        """Symbolic longitude coordinate (3D only)."""
        if self._dim == 2:
            raise AttributeError(
                "phi_sym is not available for 2D polar meshes.\n"
                "2D meshes have only (r, θ) coordinates."
            )
        return self.cs.R[0, 2]

    # === Unit Vectors ===

    @property
    def unit_r(self):
        r"""Radial unit vector (outward from origin)."""
        return self.cs.unit_e_0

    @property
    def unit_theta(self):
        r"""Tangential unit vector (direction of increasing θ)."""
        return self.cs.unit_e_1

    @property
    def unit_phi(self):
        r"""Azimuthal unit vector (eastward, direction of increasing φ). 3D only."""
        if self._dim == 2:
            raise AttributeError(
                "unit_phi is not available for 2D polar meshes.\n"
                "2D meshes have only unit_r and unit_theta."
            )
        return self.cs.unit_e_2

    # === Directional aliases ===

    @property
    def unit_radial(self):
        """Alias for unit_r."""
        return self.cs.unit_e_0

    @property
    def unit_outward(self):
        """Outward radial unit vector (alias for unit_r)."""
        return self.cs.unit_e_0

    @property
    def unit_inward(self):
        """Inward radial unit vector (opposite of unit_r)."""
        return -self.cs.unit_e_0

    # === Coordinate conversion methods ===

    def to_cartesian(self, r, theta, phi):
        """
        Convert spherical coordinates to Cartesian (x, y, z).

        Parameters
        ----------
        r : float or array_like
            Radial distance
        theta : float or array_like
            Colatitude in radians (0 at north pole)
        phi : float or array_like
            Longitude in radians

        Returns
        -------
        tuple
            (x, y, z) Cartesian coordinates
        """
        import numpy as np
        r = np.asarray(r)
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def from_cartesian(self, x, y, z):
        """
        Convert Cartesian coordinates to spherical (r, θ, φ).

        Parameters
        ----------
        x : float or array_like
            X coordinate
        y : float or array_like
            Y coordinate
        z : float or array_like
            Z coordinate

        Returns
        -------
        tuple
            (r, theta, phi) where:

            - r: Radial distance
            - theta: Colatitude in radians (0 at north pole)
            - phi: Longitude in radians (-π to π)
        """
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / np.maximum(r, 1e-30), -1, 1))
        phi = np.arctan2(y, x)
        return r, theta, phi

    def points_to_cartesian(self, points_sph):
        """
        Convert array of spherical points to Cartesian coordinates.

        Parameters
        ----------
        points_sph : array_like
            Array of shape (N, 3) with columns [r, theta, phi]

        Returns
        -------
        ndarray
            Array of shape (N, 3) with columns [x, y, z]
        """
        import numpy as np
        points_sph = np.asarray(points_sph)
        r, theta, phi = points_sph[:, 0], points_sph[:, 1], points_sph[:, 2]
        x, y, z = self.to_cartesian(r, theta, phi)
        return np.column_stack([x, y, z])

    def points_from_cartesian(self, points_xyz):
        """
        Convert array of Cartesian points to spherical coordinates.

        Parameters
        ----------
        points_xyz : array_like
            Array of shape (N, 3) with columns [x, y, z]

        Returns
        -------
        ndarray
            Array of shape (N, 3) with columns [r, theta, phi]
        """
        import numpy as np
        points_xyz = np.asarray(points_xyz)
        x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
        r, theta, phi = self.from_cartesian(x, y, z)
        return np.column_stack([r, theta, phi])

    def __repr__(self):
        """String representation showing available coordinates and basis vectors."""
        return (
            f"SphericalCoordinates(\n"
            f"  Coordinates: r, theta, phi (data arrays in radians)\n"
            f"              r_sym, theta_sym, phi_sym (symbolic)\n"
            f"  Basis vectors: unit_r, unit_theta, unit_phi\n"
            f"                unit_radial, unit_outward, unit_inward (aliases)\n"
            f")"
        )

    def view(self):
        r"""
        Display a formatted summary of available properties and methods.

        This method prints a helpful guide to the spherical coordinate system,
        showing all available data arrays, symbolic coordinates, unit vectors,
        and conversion methods.
        """
        info = r"""
Spherical Coordinates
=====================

Data Arrays (numpy):
  .coords        → (N, 3) array of [r, θ, φ] (radians)
  .r             → Radial distance array
  .theta         → Colatitude array (0 at north pole, π at south)
  .phi           → Longitude/azimuth array (-π to π)

Symbolic Coordinates (for equations):
  .r_sym         → Symbolic radius
  .theta_sym     → Symbolic colatitude (θ)
  .phi_sym       → Symbolic longitude (φ)
  [:]            → Tuple of (r_sym, theta_sym, phi_sym)

Unit Vectors (symbolic, vary with position):
  .unit_r        → Radial (outward from origin)
  .unit_theta    → Colatitude direction (southward, increasing θ)
  .unit_phi      → Azimuthal direction (eastward, increasing φ)

  Aliases:
    .unit_radial   → Same as unit_r
    .unit_outward  → Same as unit_r
    .unit_inward   → Opposite of unit_r (-unit_r)

Conversion Methods:
  .to_cartesian(r, θ, φ)        → (x, y, z)
  .from_cartesian(x, y, z)      → (r, θ, φ)
  .points_to_cartesian(arr)     → Convert (N,3) array
  .points_from_cartesian(arr)   → Convert (N,3) array

Rotation Matrix:
  mesh.CoordinateSystem.rRotN   → 3×3 matrix
    Transforms Cartesian vectors to spherical frame:
    Row 0: radial (outward)
    Row 1: theta (southward)
    Row 2: phi (eastward)

Notes:
  - θ is colatitude (physics convention), not latitude
  - Convert: latitude = 90° - θ
  - φ is in radians, use np.degrees() for degrees
"""
        print(info)


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
        if isinstance(mesh, (list, tuple)) or not hasattr(mesh, "r"):
            raise TypeError(
                f"CoordinateSystem requires a mesh object, got {type(mesh).__name__}. "
                "This object is not meant to be reconstructed by SymPy."
            )

        self.mesh = mesh
        self.coordinate_type = system

        # are the mesh coordinates XYZ or have they been replaced by
        # "Natural" coordinates like r, theta, z ?
        self.CartesianDM = True

        # Get the raw BaseScalars from the mesh
        base_scalars = self.mesh.r  # (N.x, N.y, N.z)

        # === Key Architecture Decision (December 2025): ===
        #
        # TWO coordinate representations:
        #
        # 1. _N: Raw BaseScalar objects (N.x, N.y, N.z)
        #    - Used for: jacobian(), derivatives, JIT compilation
        #    - Required by SymPy's differentiation machinery
        #    - Access via: CoordinateSystem.N
        #
        # 2. _X: UWCoordinate wrapper objects
        #    - Used for: User-facing expressions like r - inner_radius
        #    - Recognized by type in unwrap_for_evaluate (NOT nondimensionalized)
        #    - Access via: CoordinateSystem.X
        #
        # This split solves the r_prime bug while preserving jacobian compatibility.

        # _N contains raw BaseScalars for derivatives/JIT
        if self.mesh.cdim == 3:
            self._N = sympy.Matrix([[base_scalars[0], base_scalars[1], base_scalars[2]]])
        else:
            self._N = sympy.Matrix([[base_scalars[0], base_scalars[1]]])

        # Create UWCoordinate objects wrapping the BaseScalars
        # These go in _X for user-facing access
        # New signature: UWCoordinate(index, system, pretty_str, latex_str, mesh, axis_index)
        coord_x = UWCoordinate(0, self.mesh.N, pretty_str="x", latex_str="x", mesh=self.mesh, axis_index=0)
        coord_y = UWCoordinate(1, self.mesh.N, pretty_str="y", latex_str="y", mesh=self.mesh, axis_index=1)

        if self.mesh.cdim == 3:
            coord_z = UWCoordinate(2, self.mesh.N, pretty_str="z", latex_str="z", mesh=self.mesh, axis_index=2)
            self._X = sympy.Matrix([[coord_x, coord_y, coord_z]])
        else:
            self._X = sympy.Matrix([[coord_x, coord_y]])

        # Store CoordinateSystem back-references on UWCoordinates
        self._X[0].CS = self
        self._X[1].CS = self
        if self.mesh.cdim == 3:
            self._X[2].CS = self

        # Also store on the underlying BaseScalars for legacy compatibility
        base_scalars[0].CS = self
        base_scalars[1].CS = self
        if self.mesh.cdim == 3:
            base_scalars[2].CS = self

        self._R = self._X.copy()

        # We need this to define zeros in the coordinate transforms
        # (since they need to indicate they are coordinate functions even
        # if they are independent of all coordinates)
        #
        # IMPORTANT: Use raw BaseScalars (mesh.r) here, NOT UWCoordinates (self._N)!
        # This is used by lambdify in rbf_evaluate which substitutes N.base_scalars().
        # If we use UWCoordinates, they won't be substituted by lambdify.

        if self.mesh.cdim == 3:
            self.independent_of_N = expression(
                r"\vec{0}",
                underworld3.maths.functions.vanishing * base_scalars[0] * base_scalars[1] * base_scalars[2],
                "independent of N0, N1, N2",
            )
        else:
            self.independent_of_N = expression(
                r"\vec{0}",
                underworld3.maths.functions.vanishing * base_scalars[0] * base_scalars[1],
                "independent of N0, N1",
            )

        ## Change specific coordinates systems as required

        if system == CoordinateSystemType.CYLINDRICAL2D and self.mesh.dim == 2:
            """
            This describes the situation for an annulus mesh with base coordinates
            in Cartesian (x,y,z).
            """

            self.type = "Cylindrical 2D"

            # _X already contains UWCoordinates, _x is a copy for the "lowercase" alias
            self._x = self._X.copy()

            # Use UWCoordinates to build derived coordinates
            # These will be unwrapped to BaseScalars during evaluation
            x, y = self.X
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

            # _X already contains UWCoordinates, _x is a copy for the "lowercase" alias
            self._x = self._X.copy()

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, z")], real=True)

            # Use UWCoordinates to build derived coordinates
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

            # _X already contains UWCoordinates, _x is a copy for the "lowercase" alias
            self._x = self._X.copy()

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, \phi")])

            # Use UWCoordinates to build derived coordinates
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
            if not hasattr(self, "ellipsoid"):
                self.ellipsoid = ELLIPSOIDS["WGS84"].copy()

            # _X already contains UWCoordinates, _x is a copy for the "lowercase" alias
            self._x = self._X.copy()

            # Define symbolic geographic coordinates (λ notation as per user request)
            self._geo = sympy.Matrix([sympy.symbols(R"\lambda_{lon}, \lambda_{lat}, \lambda_d")])

            # Use UWCoordinates to build derived coordinates
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
            depth_approx = self.ellipsoid["a"] - r

            # Geographic coordinate expressions (approximations for symbolic work)
            lambda_lon = expression(R"\lambda_{lon}", lon_deg, "Longitude (degrees East)")
            lambda_lat = expression(R"\lambda_{lat}", lat_approx, "Latitude (degrees North)")
            lambda_d = expression(R"\lambda_d", depth_approx, "Depth (km below surface)")

            self._geo_coords = sympy.Matrix([[lambda_lon, lambda_lat, lambda_d]])

            # Basis vectors for geographic system (ELLIPSOIDAL)
            # Following user's naming: unit_WE (West to East), unit_SN (South to North), unit_down
            #
            # For an ellipsoid x²/a² + y²/a² + z²/b² = 1, the geodetic normal is
            # perpendicular to the ellipsoid surface, NOT radial.
            #
            # This matters at regional scales (10-100 km) where the difference
            # between geodetic and geocentric latitude (~10 arcmin) is significant.

            a = sympy.sympify(self.ellipsoid["a"])
            b = sympy.sympify(self.ellipsoid["b"])

            # Geodetic normal components (gradient of ellipsoid equation)
            # ∇F = (2x/a², 2y/a², 2z/b²) - we drop the factor of 2 since we normalize
            nx = x / a**2
            ny = y / a**2
            nz = z / b**2
            n_mag = sympy.sqrt(nx**2 + ny**2 + nz**2)

            # Unit up vector (geodetic normal, outward from ellipsoid surface)
            unit_up_x = nx / n_mag
            unit_up_y = ny / n_mag
            unit_up_z = nz / n_mag

            # Unit east vector (azimuthal, in horizontal plane)
            # Perpendicular to meridian plane, pointing east
            # Same as spherical case: (-y, x, 0) / √(x² + y²)
            rxy = sympy.sqrt(x**2 + y**2)
            unit_east_x = -y / rxy
            unit_east_y = x / rxy
            unit_east_z = self.independent_of_N  # Zero, but symbolic

            # Unit north vector (tangent to meridian, perpendicular to both up and east)
            # Computed via cross product: north = up × east
            # This gives the direction along the meridian, pointing north
            unit_north_x = unit_up_y * unit_east_z - unit_up_z * unit_east_y
            unit_north_y = unit_up_z * unit_east_x - unit_up_x * unit_east_z
            unit_north_z = unit_up_x * unit_east_y - unit_up_y * unit_east_x

            # Note: The cross product may not be exactly unit length due to
            # the symbolic zero handling. We normalize to be safe.
            north_mag = sympy.sqrt(unit_north_x**2 + unit_north_y**2 + unit_north_z**2)
            unit_north_x = unit_north_x / north_mag
            unit_north_y = unit_north_y / north_mag
            unit_north_z = unit_north_z / north_mag

            # Geographic basis vectors (stored as row vectors)
            # unit_down: into planet (negative of geodetic normal)
            self._unit_down = sympy.Matrix([[-unit_up_x, -unit_up_y, -unit_up_z]])

            # unit_SN: South to North (along meridian)
            self._unit_SN = sympy.Matrix([[unit_north_x, unit_north_y, unit_north_z]])

            # unit_WE: West to East (along latitude circle)
            self._unit_WE = sympy.Matrix([[unit_east_x, unit_east_y, unit_east_z]])

            # Ellipsoidal rotation matrix: transforms Cartesian vectors to geographic frame
            # Row 0: up (geodetic normal)
            # Row 1: north (meridional)
            # Row 2: east (azimuthal)
            # This is the equivalent of "xRotR" for geographic coordinates
            geoRotN = sympy.Matrix(
                [
                    [unit_up_x, unit_up_y, unit_up_z],
                    [unit_north_x, unit_north_y, unit_north_z],
                    [unit_east_x, unit_east_y, unit_east_z],
                ]
            )
            self._geoRotN = geoRotN

            # For backward compatibility with mesh.R (spherical coords),
            # also compute the spherical rotation matrix
            rz = sympy.sqrt(x**2 + y**2)
            r_x_rz = sympy.sqrt((x**2 + y**2 + z**2) * (x**2 + y**2))

            rRotN = sympy.Matrix(
                [
                    [x / r, y / r, z / r],
                    [(x * z) / r_x_rz, (y * z) / r_x_rz, -(x**2 + y**2) / r_x_rz],
                    [-y / rz, +x / rz, self.independent_of_N],
                ]
            )
            self._rRotN = rRotN
            self._xRotN = sympy.eye(self.mesh.dim)

        else:  # Cartesian by default
            self.type = f"Cartesian {self.mesh.dim}D"

            # For Cartesian, _X already contains UWCoordinates (set above)
            # _x is a lowercase alias
            self._x = self._X.copy()

            self._xRotN = sympy.eye(self.mesh.dim)
            self._rRotN = sympy.eye(self.mesh.dim)

        # For all meshes: Mark as scaled if the mesh has a model with units
        # Note: Coordinates are already in model units, but we set _scaled flag
        # to indicate the mesh is unit-aware
        self._apply_units_scaling()

        # Create coordinate accessors based on coordinate system type
        if system == CoordinateSystemType.GEOGRAPHIC:
            self._geo_accessor = GeographicCoordinateAccessor(self)
            self._spherical_accessor = None
        elif system == CoordinateSystemType.SPHERICAL:
            self._spherical_accessor = SphericalCoordinateAccessor(self)
            self._geo_accessor = None
        elif system == CoordinateSystemType.CYLINDRICAL2D:
            # Polar/cylindrical 2D uses same accessor as spherical (r, θ only)
            self._spherical_accessor = SphericalCoordinateAccessor(self)
            self._geo_accessor = None
        else:
            self._geo_accessor = None
            self._spherical_accessor = None

        return

    def _apply_units_scaling(self):
        """Mark coordinate system as scaled if model has units."""
        try:
            # Get the model from the mesh
            if hasattr(self.mesh, "_model") and self.mesh._model is not None:
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
            if "length" in scales:
                length_scale = scales["length"]

                # Get scale factor as dimensionless number in base SI units
                if hasattr(length_scale, "to_base_units"):
                    # Convert to base SI units first, then get magnitude
                    scale_factor = length_scale.to_base_units().magnitude
                elif hasattr(length_scale, "magnitude"):
                    scale_factor = length_scale.magnitude
                else:
                    scale_factor = float(length_scale)

                # Mark coordinate system as scaled
                # Note: We don't transform mesh.X because coordinates are already in model units
                self._scaled = True
                self._length_scale = scale_factor

                # Store scale factors for potential debugging
                self._scale_factors = {
                    "length": scale_factor,
                    "source": f"model '{model.name}' length scale",
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
        if hasattr(self, "_scaled") and self._scaled:
            scale_factor = self._length_scale
            coords = model_coords * scale_factor
        else:
            coords = model_coords

        # Wrap with unit-aware array if units are specified
        if self.mesh.units is not None:
            from underworld3.utilities.unit_aware_array import UnitAwareArray

            # Coordinates are scaled to SI base units (meters), not the reference unit
            # The scale factor (self._length_scale) converts dimensionless (0-1) to meters
            # So we label the result as "meter" regardless of the original reference unit
            return UnitAwareArray(coords, units="meter")

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
    def with_units(self):
        """
        Coordinate symbols with unit information.

        DEPRECATED (2025-11-26): Following the Transparent Container Principle,
        this now returns raw coordinate symbols. Units are derived on demand via
        uw.get_units() which finds the _units attribute on coordinate atoms.

        Examples:
            >>> x, y = mesh.X.with_units  # Same as mesh.X[0], mesh.X[1]
            >>> area = x * y  # Raw SymPy Mul; uw.get_units(area) → km**2
            >>> uw.get_units(x)  # → kilometer (derived from _units attribute)

        Returns:
            tuple: Coordinate symbols (x, y) or (x, y, z)
        """
        # TRANSPARENT CONTAINER PRINCIPLE (2025-11-26):
        # Just return raw coordinates. They have _units attributes from patching,
        # and uw.get_units() can derive units on demand. No wrapping needed.
        return tuple(self._X)

    @property
    def shape(self):
        """Shape of the symbolic coordinate matrix."""
        return self._X.shape

    # === SymPy Integration (for mathematical operations) ===

    def _sympy_(self):
        """
        Tell SymPy how to convert this object to a SymPy expression.

        Note: Uses _sympy_() protocol (not _sympify_()) for SymPy 1.14+ compatibility.
        This is required for proper symbolic algebra in strict mode (matrix operations).

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
        return self._X.is_commutative if hasattr(self._X, "is_commutative") else True

    def __getattr__(self, name):
        """
        Delegate SymPy-specific attributes to the underlying symbolic matrix.

        This allows CoordinateSystem to be used transparently in SymPy operations
        by forwarding attribute access to _X when the attribute doesn't exist
        on CoordinateSystem itself.

        Note: When properties like 'geo' or 'spherical' raise AttributeError
        (because the coordinate system doesn't support them), Python falls
        through to __getattr__. We detect this and re-invoke the property
        to get the helpful error message.
        """
        # Prevent infinite recursion for _X access
        if name == "_X":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_X'")

        # For coordinate accessor properties, re-invoke the property to get
        # the helpful error message (they raise AttributeError with guidance)
        if name in ("geo", "spherical"):
            # Access the property descriptor directly to get its error message
            prop = type(self).__dict__.get(name)
            if prop is not None:
                return prop.__get__(self, type(self))

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
        return self._X**other

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
        r"""
        Geographic coordinates for GEOGRAPHIC meshes.

        Provides access to longitude, latitude, depth coordinates
        and geographic basis vectors on ellipsoidal (WGS84) meshes.

        Returns
        -------
        GeographicCoordinateAccessor
            Object with .lon, .lat, .depth, .coords, unit vectors, etc.
            Use .view() for a complete summary of available properties.

        Raises
        ------
        AttributeError
            If coordinate system is not GEOGRAPHIC. The error message
            indicates what coordinate system IS available for this mesh.

        Examples
        --------
        >>> lon = mesh.X.geo.lon         # Longitude data array
        >>> geo_coords = mesh.X.geo.coords  # (N, 3) array [lon, lat, depth]
        >>> mesh.X.geo.view()            # Show all available properties
        """
        if self._geo_accessor is None:
            # Provide helpful error message indicating what IS available
            if self._spherical_accessor is not None:
                hint = (
                    f"\n\nThis mesh uses SPHERICAL coordinates. Use mesh.X.spherical instead:\n"
                    f"  mesh.X.spherical.coords  → (r, θ, φ) coordinates\n"
                    f"  mesh.X.spherical.view()  → See all available properties\n\n"
                    f"Cartesian coordinates are always available:\n"
                    f"  mesh.X.coords            → (x, y, z) data array\n"
                    f"  mesh.X[0], mesh.X[1], mesh.X[2]  → Symbolic x, y, z"
                )
            else:
                hint = (
                    f"\n\nThis mesh uses CARTESIAN coordinates only.\n\n"
                    f"Cartesian coordinates are always available:\n"
                    f"  mesh.X.coords            → (x, y, z) data array\n"
                    f"  mesh.X[0], mesh.X[1], mesh.X[2]  → Symbolic x, y, z"
                )
            raise AttributeError(
                f"Geographic coordinates (.geo) are only available for GEOGRAPHIC meshes.\n"
                f"Current mesh coordinate system: {self.coordinate_type}"
                f"{hint}"
            )
        return self._geo_accessor

    @property
    def spherical(self):
        r"""
        Spherical/polar coordinates for SPHERICAL and CYLINDRICAL2D meshes.

        Provides access to radius and angle coordinates, plus basis vectors:
        - 3D (SPHERICAL): r, θ (colatitude), φ (longitude)
        - 2D (CYLINDRICAL2D/Annulus): r, θ (polar angle)

        Returns
        -------
        SphericalCoordinateAccessor
            Object with .r, .theta, .coords, unit vectors, etc.
            For 3D also .phi. Use .view() for a complete summary.

        Raises
        ------
        AttributeError
            If coordinate system is not SPHERICAL or CYLINDRICAL2D.
            The error message indicates what IS available for this mesh.

        Examples
        --------
        >>> r = mesh.X.spherical.r           # Radius data array
        >>> sph_coords = mesh.X.spherical.coords  # (N, 2) or (N, 3) array
        >>> mesh.X.spherical.view()          # Show all available properties
        """
        if self._spherical_accessor is None:
            # Provide helpful error message indicating what IS available
            if self._geo_accessor is not None:
                hint = (
                    f"\n\nThis mesh uses GEOGRAPHIC (ellipsoidal) coordinates. Use mesh.X.geo instead:\n"
                    f"  mesh.X.geo.coords  → (lon, lat, depth) coordinates\n"
                    f"  mesh.X.geo.view()  → See all available properties\n\n"
                    f"Cartesian coordinates are always available:\n"
                    f"  mesh.X.coords            → (x, y, z) data array\n"
                    f"  mesh.X[0], mesh.X[1], mesh.X[2]  → Symbolic x, y, z"
                )
            else:
                hint = (
                    f"\n\nThis mesh uses CARTESIAN coordinates only.\n\n"
                    f"Cartesian coordinates are always available:\n"
                    f"  mesh.X.coords            → (x, y, z) data array\n"
                    f"  mesh.X[0], mesh.X[1], mesh.X[2]  → Symbolic x, y, z"
                )
            raise AttributeError(
                f"Spherical coordinates (.spherical) are only available for SPHERICAL meshes.\n"
                f"Current mesh coordinate system: {self.coordinate_type}"
                f"{hint}"
            )
        return self._spherical_accessor

    @property
    def rRotN(self) -> sympy.Matrix:
        return self._rRotN

    @property
    def xRotN(self) -> sympy.Matrix:
        return self._xRotN

    @property
    def geoRotN(self) -> sympy.Matrix:
        r"""
        Ellipsoidal rotation matrix for GEOGRAPHIC coordinate systems.

        Transforms Cartesian vectors to the local geographic frame:
        - Row 0: geodetic up (perpendicular to ellipsoid surface)
        - Row 1: north (meridional, along ellipsoid surface)
        - Row 2: east (azimuthal, along ellipsoid surface)

        This is the ellipsoidal equivalent of ``rRotN`` for spherical coordinates.
        For a sphere (a=b), this reduces to the spherical rotation matrix.

        For an ellipsoid, the geodetic normal differs from the radial direction
        by up to ~10 arcminutes at mid-latitudes, which is significant for
        regional models at scales of 10-100 km.

        Returns
        -------
        sympy.Matrix
            3×3 rotation matrix, or None if not GEOGRAPHIC coordinate system.

        Examples
        --------
        Transform a Cartesian velocity to geographic components:

        >>> v_cartesian = sympy.Matrix([[vx, vy, vz]])
        >>> v_geo = mesh.CoordinateSystem.geoRotN * v_cartesian.T
        >>> v_up, v_north, v_east = v_geo[0], v_geo[1], v_geo[2]
        """
        if hasattr(self, "_geoRotN"):
            return self._geoRotN
        return None

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
            raise NotImplementedError(
                f"unit_vertical not defined for coordinate system {self.coordinate_type}"
            )

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
            raise NotImplementedError(
                f"unit_horizontal not defined for coordinate system {self.coordinate_type}"
            )

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
            raise NotImplementedError(
                f"unit_horizontal_1 not defined for coordinate system {self.coordinate_type}"
            )

    @property
    def unit_radial(self) -> sympy.Matrix:
        """Radial direction (for cylindrical/spherical coordinate systems)"""
        if self.coordinate_type in [
            CoordinateSystemType.CYLINDRICAL2D,
            CoordinateSystemType.CYLINDRICAL3D,
        ]:
            return self.unit_e_0
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_0
        else:
            raise NotImplementedError(
                f"unit_radial not defined for coordinate system {self.coordinate_type}"
            )

    @property
    def unit_tangential(self) -> sympy.Matrix:
        """Tangential direction (for cylindrical coordinate systems)"""
        if self.coordinate_type in [
            CoordinateSystemType.CYLINDRICAL2D,
            CoordinateSystemType.CYLINDRICAL3D,
        ]:
            return self.unit_e_1
        else:
            raise NotImplementedError(
                f"unit_tangential not defined for coordinate system {self.coordinate_type}"
            )

    @property
    def unit_meridional(self) -> sympy.Matrix:
        """Meridional direction (for spherical coordinate systems)"""
        if self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_1
        else:
            raise NotImplementedError(
                f"unit_meridional not defined for coordinate system {self.coordinate_type}"
            )

    @property
    def unit_azimuthal(self) -> sympy.Matrix:
        """Azimuthal direction (for spherical coordinate systems)"""
        if self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return self.unit_e_2
        else:
            raise NotImplementedError(
                f"unit_azimuthal not defined for coordinate system {self.coordinate_type}"
            )

    @property
    def geometric_dimension_names(self) -> list:
        """Names of geometric dimensions for this coordinate system"""
        if self.coordinate_type in [CoordinateSystemType.CARTESIAN]:
            if self.mesh.dim == 2:
                return ["horizontal", "vertical"]
            else:
                return ["horizontal_x", "horizontal_y", "vertical"]
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL2D]:
            return ["radial", "tangential"]
        elif self.coordinate_type in [CoordinateSystemType.CYLINDRICAL3D]:
            return ["radial", "tangential", "vertical"]
        elif self.coordinate_type in [CoordinateSystemType.SPHERICAL]:
            return ["radial", "meridional", "azimuthal"]
        else:
            return [f"dimension_{i}" for i in range(self.mesh.dim)]

    @property
    def primary_directions(self) -> dict:
        """Dictionary of all available geometric directions for this mesh type"""
        directions = {
            "unit_e_0": self.unit_e_0,
            "unit_e_1": self.unit_e_1,
        }
        if self.mesh.dim >= 3:
            directions["unit_e_2"] = self.unit_e_2

        # Add coordinate-system-specific directions
        try:
            directions["unit_horizontal"] = self.unit_horizontal
            directions["unit_horizontal_0"] = self.unit_horizontal_0
        except NotImplementedError:
            pass

        try:
            directions["unit_horizontal_1"] = self.unit_horizontal_1
        except (NotImplementedError, ValueError):
            pass

        try:
            directions["unit_vertical"] = self.unit_vertical
        except NotImplementedError:
            pass

        try:
            directions["unit_radial"] = self.unit_radial
        except NotImplementedError:
            pass

        try:
            directions["unit_tangential"] = self.unit_tangential
        except NotImplementedError:
            pass

        try:
            directions["unit_meridional"] = self.unit_meridional
        except NotImplementedError:
            pass

        try:
            directions["unit_azimuthal"] = self.unit_azimuthal
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
            raise ValueError(
                f"Start point must have {self.mesh.dim} coordinates for {self.mesh.dim}D mesh"
            )

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
            "cartesian_coords": cartesian_coords,
            "natural_coords": natural_coords,
            "parameters": t_values,
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
            raise NotImplementedError(
                f"Profile sampling not implemented for coordinate system {self.coordinate_type}"
            )

    def _create_cartesian_profile(self, profile_type, **params):
        """Create profiles for Cartesian coordinate systems"""
        import numpy as np

        num_points = params.get("num_points", 50)

        if profile_type == "horizontal":
            # Horizontal line at specified y-position
            y_position = params.get("y_position", 0.5)
            x_range = params.get("x_range", (0.0, 1.0))

            x_values = np.linspace(x_range[0], x_range[1], num_points)
            if self.mesh.dim == 2:
                cartesian_coords = np.column_stack([x_values, np.full(num_points, y_position)])
            else:  # 3D
                z_position = params.get("z_position", 0.5)
                cartesian_coords = np.column_stack(
                    [x_values, np.full(num_points, y_position), np.full(num_points, z_position)]
                )

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": cartesian_coords.copy(),  # Same for Cartesian
                "parameters": x_values,
            }

        elif profile_type == "vertical":
            # Vertical line at specified x-position
            x_position = params.get("x_position", 0.5)
            if self.mesh.dim == 2:
                y_range = params.get("y_range", (0.0, 1.0))
                y_values = np.linspace(y_range[0], y_range[1], num_points)
                cartesian_coords = np.column_stack([np.full(num_points, x_position), y_values])
                return {
                    "cartesian_coords": cartesian_coords,
                    "natural_coords": cartesian_coords.copy(),
                    "parameters": y_values,
                }
            else:  # 3D
                y_position = params.get("y_position", 0.5)
                z_range = params.get("z_range", (0.0, 1.0))
                z_values = np.linspace(z_range[0], z_range[1], num_points)
                cartesian_coords = np.column_stack(
                    [np.full(num_points, x_position), np.full(num_points, y_position), z_values]
                )
                return {
                    "cartesian_coords": cartesian_coords,
                    "natural_coords": cartesian_coords.copy(),
                    "parameters": z_values,
                }

        elif profile_type == "diagonal":
            # Diagonal line from start to end point
            start_point = params.get("start_point", [0.0] * self.mesh.dim)
            end_point = params.get("end_point", [1.0] * self.mesh.dim)

            start_point = np.array(start_point)
            end_point = np.array(end_point)

            t_values = np.linspace(0, 1, num_points)
            cartesian_coords = np.array(
                [start_point + t * (end_point - start_point) for t in t_values]
            )

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": cartesian_coords.copy(),
                "parameters": t_values,
            }

        else:
            raise ValueError(f"Unknown Cartesian profile type: {profile_type}")

    def _create_cylindrical_profile(self, profile_type, **params):
        """Create profiles for cylindrical coordinate systems"""
        import numpy as np

        num_points = params.get("num_points", 50)

        if profile_type == "radial":
            # Radial line at specified angle
            theta = params.get("theta", 0.0)  # Angle in radians
            r_range = params.get("r_range", (0.5, 1.0))

            r_values = np.linspace(r_range[0], r_range[1], num_points)

            # Convert to Cartesian coordinates
            x_values = r_values * np.cos(theta)
            y_values = r_values * np.sin(theta)
            cartesian_coords = np.column_stack([x_values, y_values])

            # Natural coordinates
            natural_coords = np.column_stack([r_values, np.full(num_points, theta)])

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": natural_coords,
                "parameters": r_values,
            }

        elif profile_type == "tangential":
            # Tangential (circular arc) at specified radius
            radius = params.get("radius", 0.75)
            theta_range = params.get("theta_range", (0.0, 2 * np.pi))

            theta_values = np.linspace(theta_range[0], theta_range[1], num_points)

            # Convert to Cartesian coordinates
            x_values = radius * np.cos(theta_values)
            y_values = radius * np.sin(theta_values)
            cartesian_coords = np.column_stack([x_values, y_values])

            # Natural coordinates
            natural_coords = np.column_stack([np.full(num_points, radius), theta_values])

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": natural_coords,
                "parameters": theta_values,
            }

        elif profile_type == "vertical":
            # Vertical line in Cartesian y-direction
            x_position = params.get("x_position", 0.0)
            y_range = params.get("y_range", (0.0, 1.0))

            y_values = np.linspace(y_range[0], y_range[1], num_points)
            cartesian_coords = np.column_stack([np.full(num_points, x_position), y_values])

            # Convert to natural coordinates
            natural_coords = self._cartesian_to_natural_coords(cartesian_coords)

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": natural_coords,
                "parameters": y_values,
            }

        else:
            raise ValueError(f"Unknown cylindrical profile type: {profile_type}")

    def _create_spherical_profile(self, profile_type, **params):
        """Create profiles for spherical coordinate systems"""
        import numpy as np

        num_points = params.get("num_points", 50)

        if profile_type == "radial":
            # Radial line at specified theta, phi
            theta = params.get("theta", np.pi / 2)  # Colatitude (0 to pi)
            phi = params.get("phi", 0.0)  # Azimuth (-pi to pi)
            r_range = params.get("r_range", (0.5, 1.0))

            r_values = np.linspace(r_range[0], r_range[1], num_points)

            # Convert to Cartesian coordinates
            x_values = r_values * np.sin(theta) * np.cos(phi)
            y_values = r_values * np.sin(theta) * np.sin(phi)
            z_values = r_values * np.cos(theta)
            cartesian_coords = np.column_stack([x_values, y_values, z_values])

            # Natural coordinates
            natural_coords = np.column_stack(
                [r_values, np.full(num_points, theta), np.full(num_points, phi)]
            )

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": natural_coords,
                "parameters": r_values,
            }

        elif profile_type == "meridional":
            # Meridional line (constant phi, varying theta) at specified radius
            radius = params.get("radius", 0.75)
            phi = params.get("phi", 0.0)
            theta_range = params.get("theta_range", (0.0, np.pi))

            theta_values = np.linspace(theta_range[0], theta_range[1], num_points)

            # Convert to Cartesian coordinates
            x_values = radius * np.sin(theta_values) * np.cos(phi)
            y_values = radius * np.sin(theta_values) * np.sin(phi)
            z_values = radius * np.cos(theta_values)
            cartesian_coords = np.column_stack([x_values, y_values, z_values])

            # Natural coordinates
            natural_coords = np.column_stack(
                [np.full(num_points, radius), theta_values, np.full(num_points, phi)]
            )

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": natural_coords,
                "parameters": theta_values,
            }

        elif profile_type == "azimuthal":
            # Azimuthal line (constant theta, varying phi) at specified radius
            radius = params.get("radius", 0.75)
            theta = params.get("theta", np.pi / 2)
            phi_range = params.get("phi_range", (0.0, 2 * np.pi))

            phi_values = np.linspace(phi_range[0], phi_range[1], num_points)

            # Convert to Cartesian coordinates
            x_values = radius * np.sin(theta) * np.cos(phi_values)
            y_values = radius * np.sin(theta) * np.sin(phi_values)
            z_values = radius * np.full(num_points, np.cos(theta))
            cartesian_coords = np.column_stack([x_values, y_values, z_values])

            # Natural coordinates
            natural_coords = np.column_stack(
                [np.full(num_points, radius), np.full(num_points, theta), phi_values]
            )

            return {
                "cartesian_coords": cartesian_coords,
                "natural_coords": natural_coords,
                "parameters": phi_values,
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
