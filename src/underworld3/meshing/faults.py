"""
Fault Surface Module for Underworld3.

This module provides classes for representing and manipulating fault surfaces
within 3D meshes. Faults are represented as triangulated 2D manifolds that can
be used for:

- Computing distance fields from mesh points to fault surfaces
- Transferring fault orientations (normals) to mesh variables
- Applying anisotropic rheology in fault zones

The workflow typically involves:
1. Creating FaultSurface objects from point clouds or VTK files
2. Triangulating point clouds using pyvista (optional dependency)
3. Collecting faults into a FaultCollection
4. Computing distance fields and transferring normals to mesh variables
5. Using the data with TransverseIsotropicFlowModel for fault-weakened rheology

Example:
    >>> # Load faults from VTK files
    >>> faults = uw.meshing.FaultCollection()
    >>> faults.add_from_vtk("fault1.vtk")
    >>> faults.add_from_vtk("fault2.vtk")
    >>>
    >>> # Compute distance field
    >>> fault_distance = faults.compute_distance_field(mesh)
    >>>
    >>> # Transfer normals to mesh
    >>> fault_normals = faults.transfer_normals(mesh)
    >>>
    >>> # Apply to rheology
    >>> stokes.constitutive_model.Parameters.director = fault_normals.sym

Notes:
    - pyvista is required for triangulation and distance computation
    - All pyvista operations run redundantly on each MPI rank
    - VTK files can be loaded/saved without pyvista
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
            "Fault triangulation and distance computation require pyvista. "
            "Install with: pixi install -e runtime"
        )


class FaultSurface:
    """A triangulated fault surface with orientation data.

    Represents a single fault segment as a 2D surface embedded in 3D space.
    Can be created from:
    - A point cloud (requires triangulation via pyvista)
    - A VTK file (pre-triangulated surface)

    Attributes:
        name: Identifier for this fault segment
        points: (N, 3) array of surface points
        triangles: (M, 3) array of triangle vertex indices (after triangulation)
        normals: (M, 3) array of face normals (after triangulation)
        pv_mesh: PyVista PolyData object (if pyvista available)
        is_triangulated: Whether the surface has been triangulated

    Example:
        >>> # Create from points and triangulate
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]])
        >>> fault = uw.meshing.FaultSurface("fault1", points)
        >>> fault.triangulate()
        >>> fault.to_vtk("fault1.vtk")
        >>>
        >>> # Load from VTK
        >>> fault2 = uw.meshing.FaultSurface.from_vtk("fault1.vtk")
    """

    def __init__(self, name: str, points: np.ndarray = None):
        """Create a fault surface.

        Args:
            name: Identifier for this fault segment
            points: (N, 3) array of 3D points defining the fault surface.
                   If None, the fault is empty and must be loaded or
                   have points added later.
        """
        self.name = name
        self._points = None
        self._triangles = None
        self._normals = None
        self._pv_mesh = None
        self._kdtree = None

        if points is not None:
            self.points = points

    @property
    def points(self) -> Optional[np.ndarray]:
        """(N, 3) array of surface points."""
        return self._points

    @points.setter
    def points(self, value: np.ndarray):
        """Set points and invalidate cached data."""
        if value is not None:
            value = np.asarray(value)
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(
                    f"Points must be (N, 3) array, got shape {value.shape}"
                )
        self._points = value
        # Invalidate derived data
        self._triangles = None
        self._normals = None
        self._pv_mesh = None
        self._kdtree = None

    @property
    def triangles(self) -> Optional[np.ndarray]:
        """(M, 3) array of triangle vertex indices."""
        return self._triangles

    @property
    def normals(self) -> Optional[np.ndarray]:
        """(M, 3) array of face normals."""
        return self._normals

    @property
    def pv_mesh(self):
        """PyVista PolyData mesh (None if not triangulated or pyvista unavailable)."""
        return self._pv_mesh

    @property
    def is_triangulated(self) -> bool:
        """Whether the surface has been triangulated."""
        return self._triangles is not None and self._normals is not None

    @property
    def n_points(self) -> int:
        """Number of points in the surface."""
        return 0 if self._points is None else self._points.shape[0]

    @property
    def n_triangles(self) -> int:
        """Number of triangles in the surface."""
        return 0 if self._triangles is None else self._triangles.shape[0]

    @classmethod
    def from_vtk(cls, filename: str, name: str = None) -> "FaultSurface":
        """Load fault surface from VTK file.

        Args:
            filename: Path to VTK file (.vtk or .vtp)
            name: Name for the fault. If None, uses filename stem.

        Returns:
            FaultSurface: Loaded fault surface with triangulation and normals

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

        fault = cls(name)

        # Load the VTK file
        mesh = pv.read(str(filepath))
        fault._pv_mesh = mesh
        fault._points = np.array(mesh.points)

        # Extract triangles from faces
        if mesh.n_cells > 0:
            # VTK faces format: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
            faces = mesh.faces
            if len(faces) > 0:
                # Reshape to extract triangles (assumes all triangles)
                n_faces = mesh.n_cells
                fault._triangles = faces.reshape(-1, 4)[:, 1:4]

        # Extract or compute normals
        if "Normals" in mesh.cell_data:
            fault._normals = np.array(mesh.cell_data["Normals"])
        elif mesh.n_cells > 0:
            mesh.compute_normals(inplace=True)
            fault._normals = np.array(mesh.cell_data["Normals"])

        return fault

    def triangulate(self, offset: float = 0.01) -> None:
        """Triangulate point cloud using pyvista delaunay_2d.

        This creates a triangulated surface from the point cloud by projecting
        points onto a best-fit plane, performing 2D Delaunay triangulation,
        and mapping back to 3D.

        Args:
            offset: Height offset for delaunay_2d (controls curvature tolerance).
                   Larger values allow more curved surfaces.

        Raises:
            ImportError: If pyvista not available
            ValueError: If points too sparse for triangulation (< 3 points)
            RuntimeError: If triangulation fails
        """
        pv = _require_pyvista()

        if self._points is None or self.n_points == 0:
            raise ValueError(f"Fault '{self.name}' has no points to triangulate")

        if self.n_points < 3:
            raise ValueError(
                f"Fault '{self.name}' has only {self.n_points} points. "
                "Need at least 3 points for triangulation."
            )

        # Check for degenerate cases (all points nearly collinear)
        if self.n_points >= 3:
            # Compute bounding box extent
            extents = self._points.max(axis=0) - self._points.min(axis=0)
            sorted_extents = np.sort(extents)
            # If smallest extent is negligible compared to largest, points may be collinear
            if sorted_extents[0] < 1e-10 * sorted_extents[2] and sorted_extents[1] < 1e-10 * sorted_extents[2]:
                raise ValueError(
                    f"Fault '{self.name}' points appear to be nearly collinear. "
                    "Cannot create a 2D surface from a 1D line."
                )

        # Create PolyData from points and triangulate
        # This runs on all ranks redundantly (pyvista doesn't work in parallel)
        polydata = pv.PolyData(self._points)
        self._pv_mesh = polydata.delaunay_2d(offset=offset)

        if self._pv_mesh.n_cells == 0:
            raise RuntimeError(
                f"Triangulation failed for fault '{self.name}'. "
                "Try adjusting the offset parameter or check point distribution."
            )

        # Compute normals
        self._pv_mesh.compute_normals(inplace=True)

        # Extract numpy arrays
        faces = self._pv_mesh.faces
        self._triangles = faces.reshape(-1, 4)[:, 1:4]
        self._normals = np.array(self._pv_mesh.cell_data["Normals"])

    def compute_normals(self, consistent_normals: bool = True) -> None:
        """Recompute face normals for triangulated surface.

        Args:
            consistent_normals: If True, attempt to make normals consistently oriented
        """
        if not self.is_triangulated:
            raise RuntimeError(
                f"Fault '{self.name}' must be triangulated before computing normals"
            )

        pv = _require_pyvista()

        if self._pv_mesh is not None:
            self._pv_mesh.compute_normals(
                inplace=True, consistent_normals=consistent_normals
            )
            self._normals = np.array(self._pv_mesh.cell_data["Normals"])

    def flip_normals(self) -> None:
        """Flip the direction of all face normals."""
        if self._normals is not None:
            self._normals = -self._normals
            if self._pv_mesh is not None:
                self._pv_mesh.cell_data["Normals"] = self._normals

    def to_vtk(self, filename: str) -> None:
        """Export triangulated surface to VTK file.

        Args:
            filename: Output path (.vtk or .vtp)

        Raises:
            RuntimeError: If surface not triangulated
            ImportError: If pyvista not available
        """
        pv = _require_pyvista()

        if not self.is_triangulated:
            raise RuntimeError(
                f"Fault '{self.name}' must be triangulated before saving to VTK"
            )

        # Ensure we have a pyvista mesh
        if self._pv_mesh is None:
            # Reconstruct from arrays
            faces = np.column_stack([
                np.full(self.n_triangles, 3),
                self._triangles
            ]).flatten()
            self._pv_mesh = pv.PolyData(self._points, faces)
            self._pv_mesh.cell_data["Normals"] = self._normals

        self._pv_mesh.save(str(filename))

    def build_kdtree(self) -> "uw.kdtree.KDTree":
        """Build KDTree for nearest-neighbor queries on face centers.

        Returns:
            KDTree built from triangle centroids

        Raises:
            RuntimeError: If surface not triangulated
        """
        if not self.is_triangulated:
            raise RuntimeError(
                f"Fault '{self.name}' must be triangulated before building KDTree"
            )

        if self._kdtree is None:
            # Compute triangle centroids
            centroids = self.face_centers
            self._kdtree = uw.kdtree.KDTree(centroids)

        return self._kdtree

    @property
    def face_centers(self) -> np.ndarray:
        """(M, 3) array of triangle centroids."""
        if not self.is_triangulated:
            raise RuntimeError(
                f"Fault '{self.name}' must be triangulated to get face centers"
            )

        if self._pv_mesh is not None:
            return np.array(self._pv_mesh.cell_centers().points)
        else:
            # Compute manually from triangles
            v0 = self._points[self._triangles[:, 0]]
            v1 = self._points[self._triangles[:, 1]]
            v2 = self._points[self._triangles[:, 2]]
            return (v0 + v1 + v2) / 3.0

    def __repr__(self) -> str:
        status = "triangulated" if self.is_triangulated else "not triangulated"
        return (
            f"FaultSurface(name='{self.name}', "
            f"n_points={self.n_points}, "
            f"n_triangles={self.n_triangles}, "
            f"status={status})"
        )


class FaultCollection:
    """Collection of fault surfaces for mesh integration.

    Manages multiple FaultSurface objects and provides methods to:
    - Compute minimum distance from mesh points to any fault
    - Transfer fault normals to mesh variables via nearest-neighbor
    - Create rheology functions for fault-weakened zones

    Example:
        >>> faults = uw.meshing.FaultCollection()
        >>> faults.add_from_vtk("fault1.vtk")
        >>> faults.add_from_vtk("fault2.vtk")
        >>>
        >>> # Compute distance field
        >>> fault_distance = faults.compute_distance_field(mesh)
        >>>
        >>> # Transfer normals
        >>> fault_normals = faults.transfer_normals(mesh)
        >>>
        >>> # Create weakness function for rheology
        >>> eta_weak = faults.create_weakness_function(
        ...     fault_distance,
        ...     fault_width=mesh.get_min_radius() * 5,
        ...     eta_weak=0.01,
        ... )
    """

    def __init__(self):
        """Create an empty fault collection."""
        self.faults: Dict[str, FaultSurface] = {}

    def add(self, fault: FaultSurface) -> None:
        """Add a fault surface to the collection.

        Args:
            fault: FaultSurface to add

        Raises:
            ValueError: If fault with same name already exists
        """
        if fault.name in self.faults:
            raise ValueError(
                f"Fault '{fault.name}' already exists in collection. "
                "Use a different name or remove the existing fault."
            )
        self.faults[fault.name] = fault

    def add_from_vtk(self, filename: str, name: str = None) -> FaultSurface:
        """Load and add a fault from VTK file.

        Args:
            filename: Path to VTK file
            name: Name for the fault. If None, uses filename stem.

        Returns:
            The loaded FaultSurface
        """
        fault = FaultSurface.from_vtk(filename, name)
        self.add(fault)
        return fault

    def remove(self, name: str) -> FaultSurface:
        """Remove and return a fault from the collection.

        Args:
            name: Name of fault to remove

        Returns:
            The removed FaultSurface

        Raises:
            KeyError: If fault not found
        """
        return self.faults.pop(name)

    def __getitem__(self, name: str) -> FaultSurface:
        """Get a fault by name."""
        return self.faults[name]

    def __iter__(self):
        """Iterate over fault names."""
        return iter(self.faults)

    def __len__(self):
        """Number of faults in collection."""
        return len(self.faults)

    @property
    def names(self) -> List[str]:
        """List of fault names."""
        return list(self.faults.keys())

    def compute_distance_field(
        self,
        mesh: "Mesh",
        distance_var: "MeshVariable" = None,
        variable_name: str = "fault_distance",
    ) -> "MeshVariable":
        """Compute minimum distance from mesh points to any fault surface.

        Uses pyvista's compute_implicit_distance for accurate signed distance
        computation. The returned field contains the absolute distance to the
        nearest fault surface at each mesh point.

        Args:
            mesh: The mesh to compute distances on
            distance_var: Optional existing MeshVariable to store results.
                         If None, creates a new variable.
            variable_name: Name for new variable if distance_var is None

        Returns:
            MeshVariable with distance values (scalar, 1 component)

        Raises:
            ValueError: If collection is empty or no faults are triangulated
            ImportError: If pyvista not available
        """
        pv = _require_pyvista()

        if len(self.faults) == 0:
            raise ValueError("Cannot compute distance field: no faults in collection")

        # Check all faults are triangulated
        for name, fault in self.faults.items():
            if not fault.is_triangulated:
                raise ValueError(
                    f"Fault '{name}' must be triangulated before computing distances"
                )

        # Create or validate output variable
        if distance_var is None:
            distance_var = uw.discretisation.MeshVariable(
                variable_name, mesh, 1, degree=mesh.degree
            )

        # Get mesh coordinates and create pyvista point cloud
        # (avoids visualisation module which initializes trame)
        coords = mesh.X.coords
        if hasattr(coords, 'magnitude'):
            coords = coords.magnitude
        elif hasattr(coords, '__array__'):
            coords = np.asarray(coords)

        pv_mesh = pv.PolyData(coords)

        # Initialize with large distance
        with uw.synchronised_array_update():
            distance_var.data[:, 0] = 1e10

            # Compute distance to each fault, take minimum
            for fault in self.faults.values():
                dist_result = pv_mesh.compute_implicit_distance(fault.pv_mesh)
                fault_dist = np.abs(dist_result.point_data["implicit_distance"])
                distance_var.data[:, 0] = np.minimum(
                    distance_var.data[:, 0], fault_dist
                )

        return distance_var

    def transfer_normals(
        self,
        mesh: "Mesh",
        coords: np.ndarray = None,
        normal_var: "MeshVariable" = None,
        variable_name: str = "fault_normals",
    ) -> "MeshVariable":
        """Transfer fault normals to mesh points via nearest-neighbor lookup.

        For each mesh point, finds the closest fault face (from any fault in
        the collection) and copies that face's normal vector.

        Args:
            mesh: The mesh to transfer normals to
            coords: Optional coordinates to query. If None, uses mesh.X.coords
            normal_var: Optional existing MeshVariable to store results.
                       If None, creates a new variable.
            variable_name: Name for new variable if normal_var is None

        Returns:
            MeshVariable with normal vectors (3 components)

        Raises:
            ValueError: If collection is empty or no faults are triangulated
        """
        if len(self.faults) == 0:
            raise ValueError("Cannot transfer normals: no faults in collection")

        # Check all faults are triangulated
        for name, fault in self.faults.items():
            if not fault.is_triangulated:
                raise ValueError(
                    f"Fault '{name}' must be triangulated before transferring normals"
                )

        # Get query coordinates
        if coords is None:
            coords = mesh.X.coords

        # Handle UnitAwareArray by extracting raw values
        if hasattr(coords, 'magnitude'):
            coords = coords.magnitude
        elif hasattr(coords, '__array__'):
            coords = np.asarray(coords)

        # Create or validate output variable
        if normal_var is None:
            normal_var = uw.discretisation.MeshVariable(
                variable_name, mesh, 3, degree=mesh.degree
            )

        # Build combined arrays of all fault face centers and normals
        all_centers = []
        all_normals = []
        for fault in self.faults.values():
            all_centers.append(fault.face_centers)
            all_normals.append(fault.normals)

        combined_centers = np.vstack(all_centers)
        combined_normals = np.vstack(all_normals)

        # Build KDTree and query
        kdtree = uw.kdtree.KDTree(combined_centers)
        _, closest_idx = kdtree.query(coords)

        # Transfer normals
        with uw.synchronised_array_update():
            normal_var.data[:] = combined_normals[closest_idx.flatten()]

        return normal_var

    def create_weakness_function(
        self,
        distance_var: "MeshVariable",
        fault_width: float,
        eta_weak: float = 0.01,
        eta_background: float = 1.0,
    ) -> sympy.Expr:
        """Create Piecewise viscosity function for fault weakness.

        Creates a sympy Piecewise expression that gives:
        - eta_weak when distance < fault_width
        - eta_background otherwise

        This can be used directly with TransverseIsotropicFlowModel.Parameters.eta_1
        for creating anisotropic weakness along fault zones.

        Args:
            distance_var: MeshVariable containing fault distances
            fault_width: Width of the weak zone around faults
            eta_weak: Viscosity within fault zone (default 0.01)
            eta_background: Viscosity outside fault zone (default 1.0)

        Returns:
            sympy.Piecewise expression for use in constitutive models

        Example:
            >>> eta_1 = faults.create_weakness_function(
            ...     fault_distance,
            ...     fault_width=mesh.get_min_radius() * 5,
            ...     eta_weak=0.01,
            ... )
            >>> stokes.constitutive_model.Parameters.eta_1 = eta_1
        """
        return sympy.Piecewise(
            (eta_weak, distance_var.sym[0] < fault_width),
            (eta_background, True),
        )

    def __repr__(self) -> str:
        fault_strs = [f"  {name}: {fault}" for name, fault in self.faults.items()]
        faults_repr = "\n".join(fault_strs) if fault_strs else "  (empty)"
        return f"FaultCollection(\n{faults_repr}\n)"
