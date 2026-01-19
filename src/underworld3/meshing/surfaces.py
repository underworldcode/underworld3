"""
Surface Module for Underworld3.

This module provides classes for representing and manipulating embedded discretized
surfaces within computational meshes. Surfaces are represented as:
- 3D: triangulated 2D manifolds (triangle meshes)
- 2D: polylines (connected line segments)

Surfaces can represent:

- Fault surfaces (tectonic, shear zones)
- Geological horizons (Moho, sediment layers)
- Phase boundaries (solid-liquid interfaces)
- Material boundaries (compositional interfaces)
- Subduction zones (slab interfaces)

Key features:
- pyvista PolyData as the storage backend (geometry + variable data)
- SurfaceVariable for per-vertex data with .sym access for expressions
- Lazy evaluation via stale flags (distance field, proxy MeshVariables)
- VTK I/O with automatic inclusion of all variable data
- Global/redundant storage (all ranks have full copy, ~10 MB for 100k vertices)

Example:
    >>> # Create a surface from points
    >>> surface = uw.meshing.Surface("fault1", mesh, points)
    >>> surface.discretize()
    >>>
    >>> # Add variables
    >>> friction = surface.add_variable("friction", size=1)
    >>> friction.data[:] = 0.6
    >>>
    >>> # Use in expressions via .sym
    >>> eta_weak = surface.influence_function(
    ...     width=0.05,
    ...     value_near=friction.sym,
    ...     value_far=1.0,
    ...     profile="gaussian",
    ... )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import mpi

if TYPE_CHECKING:
    from underworld3.discretisation import Mesh, MeshVariable


def _require_pyvista():
    """Check pyvista availability with helpful error message."""
    try:
        import pyvista

        return pyvista
    except ImportError:
        raise ImportError(
            "Surface discretization and operations require pyvista. "
            "Install with: pixi install -e runtime"
        )


class SurfaceVariable:
    """Variable defined on surface vertices, stored in pyvista point_data.

    Provides direct access to per-vertex data via .data property, and
    symbolic access for expressions via .sym property.

    The .sym property uses a proxy MeshVariable that is lazily interpolated
    from surface vertices to mesh nodes when accessed.

    Unit awareness is provided via UnitAwareArray wrapper on .data access.
    Distance-based masking is available via .mask property when mask_width is set.

    Attributes:
        name: Variable name (also the key in pyvista point_data)
        surface: Parent Surface object
        size: Number of components per vertex
        units: Optional units for this variable (e.g., "Pa", "m/s")

    Example:
        >>> friction = surface.add_variable("friction", size=1, mask_width=0.1)
        >>> friction.data[:] = 0.6  # Set values on surface vertices
        >>> friction.data[weak_mask] = 0.3  # Set weak zone values
        >>>
        >>> # Use in expressions with explicit masking
        >>> eta = friction.sym[0] * friction.mask
    """

    def __init__(
        self,
        name: str,
        surface: "Surface",
        size: int = 1,
        proxy_degree: int = 1,
        existing: bool = False,
        units: Optional[str] = None,
        mask_width: Optional[float] = None,
        mask_profile: str = "gaussian",
    ):
        """Create a variable on surface vertices.

        Args:
            name: Variable name (key in pyvista point_data)
            surface: Parent Surface object
            size: Number of components per vertex (1 for scalar, 3 for vector)
            proxy_degree: Degree of proxy MeshVariable for .sym access
            existing: If True, wraps existing point_data (for loading from VTK)
            units: Optional units for this variable (e.g., "Pa", "m/s")
            mask_width: Width for distance-based mask (enables .mask property)
            mask_profile: Profile for mask function ("step", "linear", "gaussian", "smoothstep")
        """
        self.name = name
        self.surface = surface
        self.size = size
        self._proxy_degree = proxy_degree
        self._units = units
        self._mask_width = mask_width
        self._mask_profile = mask_profile

        # Create array in pyvista's point_data (unless wrapping existing)
        if not existing:
            n_verts = surface.n_vertices
            if size == 1:
                surface._pv_mesh.point_data[name] = np.zeros(n_verts)
            else:
                surface._pv_mesh.point_data[name] = np.zeros((n_verts, size))

        # Proxy MeshVariable for .sym access (created lazily)
        self._proxy: Optional[uw.discretisation.MeshVariable] = None
        self._proxy_stale = True

    @property
    def units(self) -> Optional[str]:
        """Units for this variable (None if dimensionless)."""
        return self._units

    @property
    def has_units(self) -> bool:
        """Check if this variable has units."""
        return self._units is not None

    @property
    def data(self):
        """Direct access to vertex data with optional unit awareness.

        Returns:
            UnitAwareArray if units are set, otherwise plain numpy array.
            Shape is (n_vertices,) for size=1, or (n_vertices, size) otherwise.

        Note:
            When units are set, modifications via array operations automatically
            sync to pyvista storage and mark proxy as stale. For raw numpy arrays,
            call mark_stale() after modifications.
        """
        raw = self.surface._pv_mesh.point_data[self.name]

        if self._units is not None:
            # Import here to avoid circular imports
            from underworld3.utilities.unit_aware_array import UnitAwareArray

            # Create callback to sync changes to pyvista and mark stale
            def sync_to_pyvista(array, info):
                self.surface._pv_mesh.point_data[self.name] = np.asarray(array)
                self._proxy_stale = True

            return UnitAwareArray(
                raw,
                units=self._units,
                callback=sync_to_pyvista,
                owner=self,
            )

        return raw

    @data.setter
    def data(self, values) -> None:
        """Set vertex data and mark proxy as stale.

        If values have units (magnitude attribute), the magnitude is extracted
        for storage in pyvista. Units are tracked separately.
        """
        # Strip units if present
        if hasattr(values, 'magnitude'):
            values = values.magnitude

        self.surface._pv_mesh.point_data[self.name] = np.asarray(values)
        self._proxy_stale = True

    def mark_stale(self) -> None:
        """Mark the proxy as stale so it will be recomputed on next .sym access."""
        self._proxy_stale = True

    @property
    def mask(self) -> sympy.Expr:
        """Distance-based mask: 1 near surface, 0 far away.

        Uses the surface's signed distance field with the configured profile.
        Must be explicitly applied in expressions: use `var.sym * var.mask`.

        Returns:
            sympy.Expr representing the mask (1 near, 0 far)

        Raises:
            ValueError: If mask_width was not set in add_variable()

        Example:
            >>> friction = surface.add_variable("friction", mask_width=0.1)
            >>> friction.data[:] = 0.6
            >>> eta = friction.sym[0] * friction.mask  # Masked value
        """
        if self._mask_width is None:
            raise ValueError(
                f"SurfaceVariable '{self.name}' has no mask_width set. "
                "Set mask_width in add_variable() or use surface.influence_function() directly."
            )

        return self.surface.influence_function(
            width=self._mask_width,
            value_near=1.0,
            value_far=0.0,
            profile=self._mask_profile,
        )

    @property
    def sym(self) -> sympy.Matrix:
        """Symbolic representation for use in expressions.

        Returns:
            sympy.Matrix that can be used in Underworld expressions.

        Note:
            On first access (or after mark_stale()), this triggers interpolation
            from surface vertices to mesh nodes. The interpolation uses inverse
            distance weighting from nearby surface vertices.
        """
        if self._proxy is None:
            self._create_proxy()

        if self._proxy_stale:
            self._interpolate_to_proxy()
            self._proxy_stale = False

        return self._proxy.sym

    def _create_proxy(self) -> None:
        """Create the proxy MeshVariable for .sym access."""
        self._proxy = uw.discretisation.MeshVariable(
            f"surf_{self.surface.name}_{self.name}",
            self.surface.mesh,
            self.size,
            degree=self._proxy_degree,
        )

    def _interpolate_to_proxy(self) -> None:
        """Interpolate surface vertex data to mesh nodes.

        Each rank populates its LOCAL mesh nodes only using inverse distance
        weighting from surface vertices.
        """
        if self._proxy is None:
            self._create_proxy()

        # Get local mesh node coordinates
        mesh_coords = self.surface.mesh.X.coords
        if hasattr(mesh_coords, 'magnitude'):
            mesh_coords = mesh_coords.magnitude
        elif hasattr(mesh_coords, '__array__'):
            mesh_coords = np.asarray(mesh_coords)

        # Get surface vertex data
        surface_coords = self.surface.vertices
        surface_values = self.data

        # For 2D surfaces, use only x,y components for KDTree
        if self.surface.is_2d:
            surface_coords = np.ascontiguousarray(surface_coords[:, :2])

        # Build KDTree for surface vertices
        kdtree = uw.kdtree.KDTree(surface_coords)

        # Find nearest surface vertex for each mesh node
        # (Simple nearest-neighbor for now; could use inverse distance weighting)
        _, nearest_idx = kdtree.query(mesh_coords)

        # Transfer values
        with uw.synchronised_array_update():
            if self.size == 1:
                self._proxy.data[:, 0] = surface_values[nearest_idx.flatten()]
            else:
                self._proxy.data[:] = surface_values[nearest_idx.flatten()]

    def __repr__(self) -> str:
        parts = [
            f"SurfaceVariable(name='{self.name}'",
            f"surface='{self.surface.name}'",
            f"size={self.size}",
        ]
        if self._units is not None:
            parts.append(f"units='{self._units}'")
        if self._mask_width is not None:
            parts.append(f"mask_width={self._mask_width}")
        # Get raw data length to avoid triggering UnitAwareArray
        raw = self.surface._pv_mesh.point_data[self.name]
        parts.append(f"n_vertices={len(raw)}")
        return ", ".join(parts) + ")"


class Surface:
    """A discretized embedded surface with variable storage.

    Represents a surface embedded in a computational mesh using pyvista PolyData
    for storage. In 3D, surfaces are triangulated meshes; in 2D, they are polylines.
    Supports per-vertex variables with symbolic access for use in Underworld expressions.

    The surface uses lazy evaluation with stale flags:
    - Discretization is computed when first accessed
    - Distance field is computed when first accessed
    - Proxy MeshVariables are updated when .sym is accessed

    Attributes:
        name: Identifier for this surface
        mesh: Associated computational mesh (for proxy MeshVariables)
        vertices: (N, 3) array of vertex positions (via pyvista)
        normals: (N, 3) array of vertex normals (via pyvista)
        n_vertices: Number of vertices
        n_triangles: Number of triangles (3D only)

    Example:
        >>> # Create from points and discretize
        >>> surface = uw.meshing.Surface("fault", mesh, points)
        >>> surface.discretize()
        >>>
        >>> # Add a variable
        >>> friction = surface.add_variable("friction")
        >>> friction.data[:] = 0.6
        >>>
        >>> # Use distance field in influence function
        >>> eta_weak = surface.influence_function(
        ...     width=0.05,
        ...     value_near=0.01,
        ...     value_far=1.0,
        ...     profile="gaussian",
        ... )
    """

    def __init__(
        self,
        name: str,
        mesh: "Mesh" = None,
        control_points: np.ndarray = None,
        symbol: str = None,
    ):
        """Create a surface.

        Args:
            name: Identifier for this surface
            mesh: Computational mesh (required for .sym access and distance field)
            control_points: (N, 3) array of 3D points defining the surface.
                           If None, the surface is empty and must be loaded or
                           have points set later.
            symbol: Short LaTeX-friendly symbol for math display (e.g., "F" for "fault").
                   If None, defaults to first letter of name capitalized.
                   Used in expressions like d_F instead of {surf_fault_distance}.
        """
        self.name = name
        self.mesh = mesh

        # Register with mesh for adaptation notifications
        if mesh is not None and hasattr(mesh, 'register_surface'):
            mesh.register_surface(self)

        # Math symbol for clean LaTeX display
        # Default: first letter capitalized (e.g., "main_fault" -> "M")
        if symbol is not None:
            self._symbol = symbol
        else:
            # Extract first letter, capitalize
            self._symbol = name[0].upper() if name else "S"

        # Level 1: Control points (primary for evolving surfaces)
        self._control_points = None

        # Level 2: pyvista PolyData (global/redundant on all ranks)
        self._pv_mesh = None

        # SurfaceVariable wrappers
        self._variables: Dict[str, SurfaceVariable] = {}

        # Stale flags for lazy evaluation
        self._discretization_stale = True
        self._distance_stale = True

        # Level 3: Cached proxy MeshVariable for distance
        self._distance_var: Optional[uw.discretisation.MeshVariable] = None

        # Dimension (2 or 3) - detected from mesh or control points
        self._dim = None

        # Set control points if provided
        if control_points is not None:
            self.set_control_points(control_points)

    @property
    def dim(self) -> int:
        """Spatial dimension (2 or 3).

        Detected from mesh.dim if available, otherwise from control points shape.
        """
        if self._dim is not None:
            return self._dim

        # Try to get from mesh
        if self.mesh is not None and hasattr(self.mesh, 'dim'):
            self._dim = self.mesh.dim
            return self._dim

        # Infer from control points
        if self._control_points is not None:
            if self._control_points.shape[1] == 2:
                self._dim = 2
            else:
                self._dim = 3
            return self._dim

        # Default to 3D
        return 3

    @property
    def is_2d(self) -> bool:
        """True if this is a 2D surface (1D curve in 2D space)."""
        return self.dim == 2

    @property
    def symbol(self) -> str:
        """Short LaTeX-friendly symbol for math expressions.

        Used in distance field expressions like $d_F$ instead of
        the full variable name {surf_fault_distance}.
        """
        return self._symbol

    @symbol.setter
    def symbol(self, value: str) -> None:
        """Set the math symbol. Marks distance field as stale if changed."""
        if value != self._symbol:
            self._symbol = value
            # If distance var exists, it needs to be recreated with new symbol
            if self._distance_var is not None:
                self._distance_var = None
                self._distance_stale = True

    @property
    def control_points(self) -> Optional[np.ndarray]:
        """(N, 3) array of control points defining the surface."""
        return self._control_points

    def set_control_points(self, points: np.ndarray) -> None:
        """Set control points and mark discretization as stale.

        Args:
            points: (N, 2) or (N, 3) array of points. Can include Pint units,
                   which will be converted using the model's scaling system.
                   For 2D points, z-coordinate is set to 0.
        """
        # Handle unit conversion using standard scaling system (ndim)
        # This works with both legacy (get_coefficients) and modern patterns
        if hasattr(points, 'magnitude'):
            # Points have Pint units - convert using scaling system
            points = uw.scaling.non_dimensionalise(points)

        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                f"Points must be (N, 2) or (N, 3) array, got shape {points.shape}"
            )

        # For 2D points, pad with z=0
        if points.shape[1] == 2:
            self._dim = 2
            points = np.column_stack([points, np.zeros(len(points))])

        self._control_points = points
        self._discretization_stale = True
        self._distance_stale = True
        self._mark_all_proxies_stale()

    def _mark_all_proxies_stale(self) -> None:
        """Mark all SurfaceVariable proxies as stale."""
        for var in self._variables.values():
            var.mark_stale()

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """(N, 3) array of vertex positions."""
        if self._pv_mesh is None:
            return None
        return np.array(self._pv_mesh.points)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the discretized surface."""
        if self._pv_mesh is None:
            return 0
        return self._pv_mesh.n_points

    @property
    def n_triangles(self) -> int:
        """Number of triangles in the surface."""
        if self._pv_mesh is None:
            return 0
        return self._pv_mesh.n_cells

    @property
    def normals(self) -> Optional[np.ndarray]:
        """(N, 3) array of vertex normals (point normals from pyvista)."""
        self._ensure_discretized()
        if self._pv_mesh is None:
            return None
        return np.array(self._pv_mesh.point_normals)

    @property
    def triangles(self) -> Optional[np.ndarray]:
        """(M, 3) array of triangle vertex indices."""
        if self._pv_mesh is None:
            return None
        faces = self._pv_mesh.faces
        if len(faces) == 0:
            return None
        return faces.reshape(-1, 4)[:, 1:4]

    @property
    def face_centers(self) -> Optional[np.ndarray]:
        """(M, 3) array of triangle centroids."""
        self._ensure_discretized()
        if self._pv_mesh is None:
            return None
        return np.array(self._pv_mesh.cell_centers().points)

    @property
    def face_normals(self) -> Optional[np.ndarray]:
        """(M, 3) array of face normals (cell normals from pyvista)."""
        self._ensure_discretized()
        if self._pv_mesh is None:
            return None
        return np.array(self._pv_mesh.cell_normals)

    @property
    def pv_mesh(self):
        """PyVista PolyData mesh (None if not discretized)."""
        return self._pv_mesh

    @property
    def is_discretized(self) -> bool:
        """Whether the surface has been discretized."""
        return self._pv_mesh is not None and self._pv_mesh.n_cells > 0

    # --- Discretization ---

    def _ensure_discretized(self) -> None:
        """Ensure discretization is computed (lazy evaluation)."""
        if self._discretization_stale and self._control_points is not None:
            self.discretize()

    def discretize(self, offset: float = 0.01, n_segments: int = None) -> None:
        """Discretize control points into a surface mesh.

        For 3D surfaces: Uses pyvista delaunay_2d to create a triangulated mesh.
        For 2D surfaces: Uses scipy to fit a spline and create a polyline.

        Args:
            offset: (3D only) Height offset for delaunay_2d (controls curvature tolerance).
            n_segments: (2D only) Number of line segments. If None, uses number
                       of control points minus 1.

        Raises:
            ImportError: If pyvista not available
            ValueError: If points too sparse for discretization
            RuntimeError: If discretization fails
        """
        if self._control_points is None or len(self._control_points) == 0:
            raise ValueError(f"Surface '{self.name}' has no control points to discretize")

        if self.is_2d:
            self._discretize_2d(n_segments=n_segments)
        else:
            self._discretize_3d(offset=offset)

        # Clear stale flags
        self._discretization_stale = False
        self._distance_stale = True

    def _discretize_3d(self, offset: float = 0.01) -> None:
        """Discretize 3D control points into triangulated mesh using pyvista delaunay_2d."""
        pv = _require_pyvista()

        if len(self._control_points) < 3:
            raise ValueError(
                f"Surface '{self.name}' has only {len(self._control_points)} points. "
                "Need at least 3 points for 3D discretization."
            )

        # Check for degenerate cases (all points nearly collinear)
        extents = self._control_points.max(axis=0) - self._control_points.min(axis=0)
        sorted_extents = np.sort(extents)
        if sorted_extents[0] < 1e-10 * sorted_extents[2] and sorted_extents[1] < 1e-10 * sorted_extents[2]:
            raise ValueError(
                f"Surface '{self.name}' points appear to be nearly collinear. "
                "Cannot create a 2D surface from a 1D line."
            )

        # Create PolyData from points and triangulate
        polydata = pv.PolyData(self._control_points)
        self._pv_mesh = polydata.delaunay_2d(offset=offset)

        if self._pv_mesh.n_cells == 0:
            raise RuntimeError(
                f"Triangulation failed for surface '{self.name}'. "
                "Try adjusting the offset parameter or check point distribution."
            )

        # Compute normals (both point and cell)
        self._pv_mesh.compute_normals(inplace=True)

    def _discretize_2d(self, n_segments: int = None) -> None:
        """Create 2D surface as ordered line segments using scipy spline fitting.

        For 2D, a "surface" is a 1D curve (polyline) embedded in 2D space.
        """
        pv = _require_pyvista()
        from scipy.interpolate import splprep, splev

        if len(self._control_points) < 2:
            raise ValueError(
                f"Surface '{self.name}' has only {len(self._control_points)} points. "
                "Need at least 2 points for 2D line segments."
            )

        # Get 2D coordinates
        points_2d = self._control_points[:, :2]

        # For just 2 points, connect them directly
        if len(points_2d) == 2:
            vertices_2d = points_2d.copy()
        else:
            # Use scipy to fit a parametric spline through points
            # This naturally orders points along the curve
            try:
                # s=0 means interpolate exactly through points
                tck, u = splprep([points_2d[:, 0], points_2d[:, 1]], s=0)

                # Determine number of output points
                if n_segments is None:
                    n_out = len(points_2d)
                else:
                    n_out = n_segments + 1

                # Evaluate spline at uniform parameter values
                u_new = np.linspace(0, 1, n_out)
                x, y = splev(u_new, tck)
                vertices_2d = np.column_stack([x, y])

            except Exception as e:
                # Fall back to original points if spline fitting fails
                # (e.g., for nearly collinear points)
                vertices_2d = points_2d.copy()

        # Compute normals using geometry_tools
        from underworld3.utilities.geometry_tools import linesegment_normals_2d
        _, vertex_normals_2d = linesegment_normals_2d(vertices_2d)

        # Pad to 3D for pyvista storage (z=0)
        vertices_3d = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])
        vertex_normals_3d = np.column_stack([
            vertex_normals_2d, np.zeros(len(vertex_normals_2d))
        ])

        # Create pyvista line mesh
        self._pv_mesh = pv.lines_from_points(vertices_3d)

        # Store vertex normals in point_data (since compute_normals doesn't work for lines)
        self._pv_mesh.point_data["Normals"] = vertex_normals_3d

        # Store 2D vertices for distance computation (avoid z=0 overhead)
        self._vertices_2d = vertices_2d

    def deform_vertices(self, displacement: np.ndarray) -> None:
        """Deform surface vertices in-place (no re-discretization).

        This modifies vertex positions while keeping topology fixed.
        Normals are automatically recomputed.

        Args:
            displacement: (n_vertices, 3) array of displacements to add

        Note:
            This does NOT update control points. If you want topology changes,
            use set_control_points() instead and call discretize().
        """
        self._ensure_discretized()

        if self._pv_mesh is None:
            raise RuntimeError(f"Surface '{self.name}' must be discretized before deforming")

        displacement = np.asarray(displacement)
        if displacement.shape != self._pv_mesh.points.shape:
            raise ValueError(
                f"Displacement shape {displacement.shape} doesn't match "
                f"vertex shape {self._pv_mesh.points.shape}"
            )

        self._pv_mesh.points += displacement
        self._pv_mesh.compute_normals(inplace=True)
        self._distance_stale = True
        self._mark_all_proxies_stale()

    # --- Distance field ---

    @property
    def distance(self) -> "uw.discretisation.MeshVariable":
        """Signed distance from mesh nodes to surface (lazily computed).

        The signed distance is positive on one side of the surface and
        negative on the other. Use sympy.Abs(surface.distance.sym[0]) for
        unsigned distance, or use influence_function() which does this
        automatically.

        Returns:
            MeshVariable with signed distance values at each mesh node.
            Access .sym[0] for use in expressions.

        Raises:
            RuntimeError: If mesh not set or surface not discretized

        Example:
            >>> # Use signed distance for different properties on each side
            >>> d = surface.distance.sym[0]
            >>> prop = sympy.Piecewise((upper_value, d > 0), (lower_value, True))
            >>>
            >>> # Use absolute distance for symmetric influence
            >>> mask = sympy.Piecewise((1, sympy.Abs(d) < width), (0, True))
        """
        if self.mesh is None:
            raise RuntimeError(
                f"Surface '{self.name}' requires a mesh to compute distance field. "
                "Set mesh in constructor or via surface.mesh = mesh"
            )

        self._ensure_discretized()

        if self._distance_stale:
            self._compute_distance_field()
            self._distance_stale = False

        return self._distance_var

    def _compute_distance_field(self) -> None:
        """Compute signed distance field from mesh nodes to surface.

        The signed distance is positive on one side of the surface and
        negative on the other. Helper functions like influence_function()
        use sympy.Abs() when unsigned distance is needed.

        For 3D surfaces: Uses pyvista's compute_implicit_distance.
        For 2D surfaces: Uses geometry_tools signed_distance_pointcloud_polyline_2d.
        """
        if self._distance_var is None:
            # Use varsymbol for clean LaTeX display: d_{F} instead of {surf_fault_distance}
            # Always wrap symbol in braces for proper LaTeX grouping (e.g., d_{F_1} not d_F_1)
            self._distance_var = uw.discretisation.MeshVariable(
                f"surf_{self.name}_distance",
                self.mesh,
                1,
                degree=self.mesh.degree,
                varsymbol=f"d_{{{self._symbol}}}",
            )

        # Get mesh coordinates
        coords = self.mesh.X.coords
        if hasattr(coords, 'magnitude'):
            coords = coords.magnitude
        elif hasattr(coords, '__array__'):
            coords = np.asarray(coords)

        if self.is_2d:
            # 2D: Use geometry_tools for signed distance to polyline
            from underworld3.utilities.geometry_tools import (
                signed_distance_pointcloud_polyline_2d
            )

            # Get 2D coordinates
            coords_2d = coords[:, :2]

            # Use stored 2D vertices for distance computation
            if hasattr(self, '_vertices_2d') and self._vertices_2d is not None:
                vertices_2d = self._vertices_2d
            else:
                # Fall back to pyvista mesh points
                vertices_2d = self._pv_mesh.points[:, :2]

            distances = signed_distance_pointcloud_polyline_2d(coords_2d, vertices_2d)

        else:
            # 3D: Use pyvista's compute_implicit_distance
            pv = _require_pyvista()
            pv_mesh = pv.PolyData(coords)
            dist_result = pv_mesh.compute_implicit_distance(self._pv_mesh)
            # Keep signed distance - helpers use sympy.Abs() when needed
            distances = dist_result.point_data["implicit_distance"]

        with uw.synchronised_array_update():
            self._distance_var.data[:, 0] = distances

    # --- Influence function ---

    def influence_function(
        self,
        width: float,
        value_near: Union[float, sympy.Expr] = 1.0,
        value_far: Union[float, sympy.Expr] = 0.0,
        profile: str = "step",
    ) -> sympy.Expr:
        """Create level-set-like influence function based on distance.

        Creates a sympy expression that varies from value_near (at the surface)
        to value_far (far from the surface) based on the chosen profile.

        Uses the absolute value of the signed distance field, so the influence
        is symmetric on both sides of the surface. For asymmetric behavior,
        access the signed distance directly via surface.distance.sym[0].

        Args:
            width: Characteristic width of the transition zone. Can include
                   Pint units (e.g., 500 * u.meter), which will be converted
                   using the model's scaling system.
            value_near: Value at/near the surface (can be a scalar or sympy expression)
            value_far: Value far from the surface (can be a scalar or sympy expression)
            profile: Transition profile type:
                - "step": Sharp transition at distance = width
                - "linear": Linear ramp from 0 to width
                - "gaussian": Smooth Gaussian decay
                - "smoothstep": C1-continuous Hermite interpolation

        Returns:
            sympy.Expr that can be used in Underworld expressions

        Example:
            >>> # Step function for fault zone viscosity
            >>> eta = surface.influence_function(
            ...     width=0.05,
            ...     value_near=0.01,
            ...     value_far=1.0,
            ...     profile="step",
            ... )
            >>>
            >>> # Gaussian decay for smooth transitions
            >>> eta = surface.influence_function(
            ...     width=0.1,
            ...     value_near=friction.sym,  # Variable on surface
            ...     value_far=1.0,
            ...     profile="gaussian",
            ... )
        """
        # Handle unit conversion for width using standard scaling system (ndim)
        # This works with both legacy (get_coefficients) and modern patterns
        if hasattr(width, 'magnitude'):
            width = float(uw.scaling.non_dimensionalise(width))

        # Use absolute distance - influence is symmetric about surface
        d = sympy.Abs(self.distance.sym[0])

        if profile == "step":
            return sympy.Piecewise(
                (value_near, d < width),
                (value_far, True),
            )
        elif profile == "linear":
            t = sympy.Max(0, 1 - d / width)
            return value_far + (value_near - value_far) * t
        elif profile == "gaussian":
            return value_far + (value_near - value_far) * sympy.exp(-(d / width) ** 2)
        elif profile == "smoothstep":
            # Hermite smoothstep: 3t^2 - 2t^3 for t in [0,1]
            t = sympy.Max(0, sympy.Min(1, 1 - d / width))
            smooth = 3 * t**2 - 2 * t**3
            return value_far + (value_near - value_far) * smooth
        else:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                "Choose from: step, linear, gaussian, smoothstep"
            )

    # --- Variables ---

    def add_variable(
        self,
        name: str,
        size: int = 1,
        proxy_degree: int = 1,
        units: Optional[str] = None,
        mask_width: Optional[float] = None,
        mask_profile: str = "gaussian",
    ) -> SurfaceVariable:
        """Add a variable on surface vertices.

        Creates a SurfaceVariable stored in pyvista point_data with
        symbolic access via .sym for use in expressions.

        Args:
            name: Variable name
            size: Number of components (1 for scalar, 3 for vector)
            proxy_degree: Degree of proxy MeshVariable for .sym access
            units: Optional units for this variable (e.g., "Pa", "m/s")
            mask_width: Width for distance-based mask (enables .mask property)
            mask_profile: Profile for mask function ("step", "linear", "gaussian", "smoothstep")

        Returns:
            SurfaceVariable that can be modified via .data and used via .sym

        Example:
            >>> # Variable with units and mask
            >>> friction = surface.add_variable("friction", size=1, mask_width=0.1)
            >>> friction.data[:] = 0.6
            >>>
            >>> # Use in expressions with explicit masking
            >>> eta = friction.sym[0] * friction.mask
        """
        self._ensure_discretized()

        if name in self._variables:
            raise ValueError(
                f"Variable '{name}' already exists on surface '{self.name}'"
            )

        var = SurfaceVariable(
            name, self, size, proxy_degree,
            units=units, mask_width=mask_width, mask_profile=mask_profile
        )
        self._variables[name] = var
        return var

    def get_variable(self, name: str) -> SurfaceVariable:
        """Get an existing variable by name.

        Args:
            name: Variable name

        Returns:
            SurfaceVariable

        Raises:
            KeyError: If variable doesn't exist
        """
        return self._variables[name]

    @property
    def variables(self) -> Dict[str, SurfaceVariable]:
        """Dictionary of all variables on this surface."""
        return self._variables

    # --- VTK I/O ---

    def save(self, filename: str) -> None:
        """Save surface with all variables to VTK file.

        All SurfaceVariable data is automatically included in the VTK file
        as point_data arrays.

        Args:
            filename: Output path (.vtk or .vtp)
        """
        self._ensure_discretized()

        if self._pv_mesh is None:
            raise RuntimeError(
                f"Surface '{self.name}' must be discretized before saving"
            )

        self._pv_mesh.save(str(filename))

    # --- Mesh Adaptation Support ---

    def _on_mesh_adapted(self, adapted_mesh: "Mesh") -> None:
        """Called by mesh.adapt() to update after mesh adaptation.

        Marks the distance field as stale so it will be recomputed on next access.
        The surface geometry (control points, pyvista mesh) is unchanged -
        only the cached distance values need updating.

        The distance MeshVariable itself is reinitialized by mesh.adapt() along
        with all other MeshVariables - we just need to mark the data as stale.

        Args:
            adapted_mesh: The mesh (same object, updated internals)
        """
        # Mark distance as stale - will be recomputed on next access
        # The MeshVariable stays in mesh._vars and gets reinitialized by adapt()
        # just like any other variable (same pattern as swarm proxy variables)
        self._distance_stale = True

        # Mark all variable proxies as stale (they project to mesh nodes)
        self._mark_all_proxies_stale()

    def refinement_metric(
        self,
        h_near: float,
        h_far: float,
        width: float = None,
        profile: str = "linear",
        name: str = None,
    ) -> "MeshVariable":
        r"""Create a metric field for mesh adaptation based on distance from this surface.

        Returns a MeshVariable containing refinement metric values that can
        be passed directly to mesh.adapt(). Higher metric values produce finer
        mesh (smaller elements).

        Parameters
        ----------
        h_near : float
            Target edge length near the surface (smaller = finer mesh).
            Smaller values produce more refinement near the surface.
        h_far : float
            Target edge length far from the surface (larger = coarser mesh).
            Larger values allow coarser mesh far from the surface.
        width : float, optional
            Distance over which to transition from h_near to h_far.
            If None, defaults to 2 * h_far.
        profile : str, optional
            Transition profile: "linear", "smoothstep", or "gaussian".
            Default is "linear".
        name : str, optional
            Name for the metric MeshVariable. Defaults to "{surface_name}_metric".

        Returns
        -------
        MeshVariable
            Scalar MeshVariable containing refinement metric values.

        Notes
        -----
        **Metric Tensor Mathematics**

        For isotropic mesh adaptation, MMG/PETSc uses a metric tensor:

        .. math::

            M = h^{-2} \cdot I

        where :math:`h` is the target edge length and :math:`I` is the identity
        matrix. This relationship is **dimension-independent** - the same formula
        applies in 2D and 3D because the metric defines edge lengths, not areas
        or volumes.

        The adaptation algorithm seeks to make all edges have unit length in the
        metric space (i.e., :math:`\mathbf{e}^T M \mathbf{e} = 1` for edge vector
        :math:`\mathbf{e}`). Higher metric values produce smaller elements.

        **Refinement Ratio**

        The refinement ratio is ``h_far / h_near``. For example, if ``h_near=0.01``
        and ``h_far=0.1``, the mesh will be ~10× finer near the surface.

        **Element Count Control**

        To maintain approximately the same total element count while refining
        near the surface, the far-field should use similar h to the original
        mesh's cell size. The refined region is small, so coarsening the far-field
        slightly can compensate.

        References
        ----------
        .. [1] MMG Platform documentation: http://www.mmgtools.org/
        .. [2] Alauzet, F. "Metric-based anisotropic mesh adaptation" (2010)

        Examples
        --------
        >>> fault = uw.meshing.Surface("fault", mesh, fault_points)
        >>> fault.discretize()
        >>>
        >>> # 10x refinement near fault, maintain original density far away
        >>> metric = fault.refinement_metric(h_near=0.005, h_far=0.05)
        >>> mesh.adapt(metric)
        """
        if self.mesh is None:
            raise RuntimeError(
                f"Surface '{self.name}' must be attached to a mesh to create refinement metric"
            )

        if width is None:
            width = 2.0 * h_far

        # Create metric MeshVariable
        if name is None:
            name = f"{self.name}_metric"

        metric = uw.discretisation.MeshVariable(name, self.mesh, 1, degree=1)

        # Get distance values directly from the distance MeshVariable
        dist_var = self.distance
        with self.mesh.access(dist_var, metric):
            dist_values = np.abs(dist_var.data[:, 0])

            # Compute target edge lengths based on distance profile
            if profile == "linear":
                t = np.minimum(dist_values / width, 1.0)
                h_values = h_near + (h_far - h_near) * t

            elif profile == "smoothstep":
                t = np.minimum(dist_values / width, 1.0)
                smooth_t = 3 * t**2 - 2 * t**3
                h_values = h_near + (h_far - h_near) * smooth_t

            elif profile == "gaussian":
                sigma = width / 3.0
                gaussian = np.exp(-(dist_values**2) / (2 * sigma**2))
                h_values = h_far - (h_far - h_near) * gaussian

            else:
                raise ValueError(f"Unknown profile: {profile}. Use 'linear', 'smoothstep', or 'gaussian'")

            # Convert to metric tensor: M = 1/h² × I (isotropic)
            # This is dimension-independent: same formula for 2D and 3D
            # The metric defines edge lengths, not areas/volumes
            # Higher metric values → finer mesh (smaller elements)
            metric.data[:, 0] = 1.0 / (h_values ** 2)

        return metric

    @classmethod
    def from_vtk(
        cls,
        filename: str,
        mesh: "Mesh" = None,
        name: str = None,
    ) -> "Surface":
        """Load surface from VTK file.

        All point_data arrays in the VTK file are automatically wrapped
        as SurfaceVariables.

        Args:
            filename: Path to VTK file (.vtk or .vtp)
            mesh: Computational mesh (required for .sym access)
            name: Name for the surface. If None, uses filename stem.

        Returns:
            Surface with all variables from VTK file

        Raises:
            FileNotFoundError: If file doesn't exist
            ImportError: If pyvista not available
        """
        pv = _require_pyvista()

        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"VTK file not found: {filename}")

        if name is None:
            name = filepath.stem

        surface = cls(name, mesh)
        surface._pv_mesh = pv.read(str(filepath))
        surface._discretization_stale = False
        surface._distance_stale = True

        # Compute normals if not present
        if "Normals" not in surface._pv_mesh.point_data:
            surface._pv_mesh.compute_normals(inplace=True)

        # Wrap existing point_data as SurfaceVariables
        for data_name in surface._pv_mesh.point_data.keys():
            if data_name not in ["Normals"]:  # Skip built-in normals
                data = surface._pv_mesh.point_data[data_name]
                size = 1 if data.ndim == 1 else data.shape[1]
                surface._variables[data_name] = SurfaceVariable(
                    data_name, surface, size, existing=True
                )

        return surface

    # --- Compatibility ---

    def compute_normals(self, consistent_normals: bool = True) -> None:
        """Recompute vertex and face normals.

        Args:
            consistent_normals: If True, attempt to make normals consistently oriented
        """
        self._ensure_discretized()

        if self._pv_mesh is not None:
            self._pv_mesh.compute_normals(
                inplace=True, consistent_normals=consistent_normals
            )
            self._mark_all_proxies_stale()

    def flip_normals(self) -> None:
        """Flip the direction of all normals.

        This directly negates the normal vectors. Note that pyvista's
        compute_normals() uses consistent orientation so we can't rely
        on flip_faces() to reverse normals - we must negate them explicitly.
        """
        self._ensure_discretized()

        if self._pv_mesh is not None:
            # Ensure normals are computed
            if "Normals" not in self._pv_mesh.point_data:
                self._pv_mesh.compute_normals(inplace=True)

            # Directly negate the point normals
            if "Normals" in self._pv_mesh.point_data:
                self._pv_mesh.point_data["Normals"] = (
                    -self._pv_mesh.point_data["Normals"]
                )

            # Also negate cell normals if present
            if "Normals" in self._pv_mesh.cell_data:
                self._pv_mesh.cell_data["Normals"] = (
                    -self._pv_mesh.cell_data["Normals"]
                )

            self._mark_all_proxies_stale()

    # --- Representation ---

    def __repr__(self) -> str:
        status = "discretized" if self.is_discretized else "not discretized"
        mesh_name = self.mesh.name if self.mesh is not None else "None"
        n_vars = len(self._variables)
        return (
            f"Surface(name='{self.name}', "
            f"mesh='{mesh_name}', "
            f"n_vertices={self.n_vertices}, "
            f"n_triangles={self.n_triangles}, "
            f"n_variables={n_vars}, "
            f"status={status})"
        )


class SurfaceCollection:
    """Collection of surfaces for combined operations.

    Manages multiple Surface objects and provides methods to:
    - Compute minimum distance from mesh points to any surface
    - Transfer surface normals to mesh variables via nearest-neighbor
    - Create combined influence functions

    Example:
        >>> surfaces = uw.meshing.SurfaceCollection()
        >>> surfaces.add(fault1)
        >>> surfaces.add(fault2)
        >>>
        >>> # Compute combined distance field
        >>> dist = surfaces.compute_distance_field(mesh)
        >>>
        >>> # Combined influence function
        >>> eta = surfaces.influence_function(
        ...     width=0.05,
        ...     value_near=0.01,
        ...     value_far=1.0,
        ... )
    """

    def __init__(self):
        """Create an empty surface collection."""
        self.surfaces: Dict[str, Surface] = {}
        self._distance_var: Optional[uw.discretisation.MeshVariable] = None
        self._distance_stale = True

    def add(self, surface: Surface) -> None:
        """Add a surface to the collection.

        Args:
            surface: Surface to add

        Raises:
            ValueError: If surface with same name already exists
        """
        if surface.name in self.surfaces:
            raise ValueError(
                f"Surface '{surface.name}' already exists in collection. "
                "Use a different name or remove the existing surface."
            )
        self.surfaces[surface.name] = surface
        self._distance_stale = True

    def add_from_vtk(
        self,
        filename: str,
        mesh: "Mesh" = None,
        name: str = None,
    ) -> Surface:
        """Load and add a surface from VTK file.

        Args:
            filename: Path to VTK file
            mesh: Computational mesh
            name: Name for the surface. If None, uses filename stem.

        Returns:
            The loaded Surface
        """
        surface = Surface.from_vtk(filename, mesh, name)
        self.add(surface)
        return surface

    def remove(self, name: str) -> Surface:
        """Remove and return a surface from the collection.

        Args:
            name: Name of surface to remove

        Returns:
            The removed Surface

        Raises:
            KeyError: If surface not found
        """
        surface = self.surfaces.pop(name)
        self._distance_stale = True
        return surface

    def __getitem__(self, name: str) -> Surface:
        """Get a surface by name."""
        return self.surfaces[name]

    def __iter__(self):
        """Iterate over surface names."""
        return iter(self.surfaces)

    def __len__(self):
        """Number of surfaces in collection."""
        return len(self.surfaces)

    @property
    def names(self) -> List[str]:
        """List of surface names."""
        return list(self.surfaces.keys())

    def compute_distance_field(
        self,
        mesh: "Mesh",
        distance_var: "MeshVariable" = None,
        variable_name: str = "surface_distance",
    ) -> "MeshVariable":
        """Compute minimum distance from mesh points to any surface.

        Args:
            mesh: The mesh to compute distances on
            distance_var: Optional existing MeshVariable to store results
            variable_name: Name for new variable if distance_var is None

        Returns:
            MeshVariable with distance values (scalar)
        """
        pv = _require_pyvista()

        if len(self.surfaces) == 0:
            raise ValueError("Cannot compute distance field: no surfaces in collection")

        # Ensure all surfaces are discretized
        for name, surface in self.surfaces.items():
            surface._ensure_discretized()
            if not surface.is_discretized:
                raise ValueError(
                    f"Surface '{name}' must be discretized before computing distances"
                )

        # Create or use existing variable
        if distance_var is None:
            if self._distance_var is None or self._distance_stale:
                self._distance_var = uw.discretisation.MeshVariable(
                    variable_name, mesh, 1, degree=mesh.degree
                )
            distance_var = self._distance_var

        # Get mesh coordinates
        coords = mesh.X.coords
        if hasattr(coords, 'magnitude'):
            coords = coords.magnitude
        elif hasattr(coords, '__array__'):
            coords = np.asarray(coords)

        pv_mesh = pv.PolyData(coords)

        # Initialize with large distance
        with uw.synchronised_array_update():
            distance_var.data[:, 0] = 1e10

            # Compute unsigned distance to each surface, take minimum.
            # Unlike single Surface (which stores signed distance), collections
            # store unsigned because signed distance is ambiguous with multiple surfaces.
            for surface in self.surfaces.values():
                dist_result = pv_mesh.compute_implicit_distance(surface._pv_mesh)
                surface_dist = np.abs(dist_result.point_data["implicit_distance"])
                distance_var.data[:, 0] = np.minimum(
                    distance_var.data[:, 0], surface_dist
                )

        self._distance_stale = False
        return distance_var

    def transfer_normals(
        self,
        mesh: "Mesh",
        coords: np.ndarray = None,
        normal_var: "MeshVariable" = None,
        variable_name: str = "surface_normals",
    ) -> "MeshVariable":
        """Transfer surface normals to mesh points via nearest-neighbor.

        For each mesh point, finds the closest surface face and copies
        that face's normal vector.

        Args:
            mesh: The mesh to transfer normals to
            coords: Optional coordinates to query. If None, uses mesh.data
            normal_var: Optional existing MeshVariable
            variable_name: Name for new variable

        Returns:
            MeshVariable with normal vectors (3 components)
        """
        if len(self.surfaces) == 0:
            raise ValueError("Cannot transfer normals: no surfaces in collection")

        # Ensure all surfaces are discretized
        for name, surface in self.surfaces.items():
            surface._ensure_discretized()

        # Get query coordinates
        if coords is None:
            coords = mesh.X.coords

        if hasattr(coords, 'magnitude'):
            coords = coords.magnitude
        elif hasattr(coords, '__array__'):
            coords = np.asarray(coords)

        # Create or validate output variable
        if normal_var is None:
            normal_var = uw.discretisation.MeshVariable(
                variable_name, mesh, 3, degree=mesh.degree
            )

        # Build combined arrays of all face centers and normals
        all_centers = []
        all_normals = []
        for surface in self.surfaces.values():
            all_centers.append(surface.face_centers)
            all_normals.append(surface.face_normals)

        combined_centers = np.vstack(all_centers)
        combined_normals = np.vstack(all_normals)

        # Build KDTree and query
        kdtree = uw.kdtree.KDTree(combined_centers)
        _, closest_idx = kdtree.query(coords)

        # Transfer normals
        with uw.synchronised_array_update():
            normal_var.data[:] = combined_normals[closest_idx.flatten()]

        return normal_var

    def influence_function(
        self,
        mesh: "Mesh",
        width: float,
        value_near: Union[float, sympy.Expr] = 1.0,
        value_far: Union[float, sympy.Expr] = 0.0,
        profile: str = "step",
    ) -> sympy.Expr:
        """Create combined influence function from all surfaces.

        Args:
            mesh: Computational mesh
            width: Characteristic width of the transition zone
            value_near: Value at/near surfaces
            value_far: Value far from surfaces
            profile: Transition profile (step, linear, gaussian, smoothstep)

        Returns:
            sympy.Expr based on combined distance field
        """
        distance_var = self.compute_distance_field(mesh)
        d = distance_var.sym[0]

        if profile == "step":
            return sympy.Piecewise(
                (value_near, d < width),
                (value_far, True),
            )
        elif profile == "linear":
            t = sympy.Max(0, 1 - d / width)
            return value_far + (value_near - value_far) * t
        elif profile == "gaussian":
            return value_far + (value_near - value_far) * sympy.exp(-(d / width) ** 2)
        elif profile == "smoothstep":
            t = sympy.Max(0, sympy.Min(1, 1 - d / width))
            smooth = 3 * t**2 - 2 * t**3
            return value_far + (value_near - value_far) * smooth
        else:
            raise ValueError(f"Unknown profile '{profile}'")

    # --- Backward compatibility ---

    def create_weakness_function(
        self,
        distance_var: "MeshVariable",
        fault_width: float,
        eta_weak: float = 0.01,
        eta_background: float = 1.0,
    ) -> sympy.Expr:
        """Create Piecewise viscosity function for fault weakness.

        DEPRECATED: Use influence_function() instead.

        Args:
            distance_var: MeshVariable containing distances
            fault_width: Width of the weak zone
            eta_weak: Viscosity within weak zone
            eta_background: Viscosity outside weak zone

        Returns:
            sympy.Piecewise expression
        """
        import warnings
        warnings.warn(
            "create_weakness_function is deprecated. Use influence_function() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return sympy.Piecewise(
            (eta_weak, distance_var.sym[0] < fault_width),
            (eta_background, True),
        )

    def __repr__(self) -> str:
        surface_strs = [
            f"  {name}: {surface}" for name, surface in self.surfaces.items()
        ]
        surfaces_repr = "\n".join(surface_strs) if surface_strs else "  (empty)"
        return f"SurfaceCollection(\n{surfaces_repr}\n)"
