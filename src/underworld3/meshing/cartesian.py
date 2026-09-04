"""
Cartesian mesh generation functions for Underworld3.

This module contains mesh generation functions for rectangular/box meshes 
in Cartesian coordinates.
"""

from typing import Optional, Tuple
from enum import Enum

import tempfile
import numpy as np
import petsc4py
from petsc4py import PETSc
import os
import math

import underworld3 as uw
from underworld3.meshing._mesh_files import mesh_file_dir, write_gmsh
from underworld3.discretisation import Mesh
from underworld3 import VarType
from underworld3.coordinates import CoordinateSystemType
from underworld3.discretisation import _from_gmsh as gmsh2dmplex
import underworld3.timing as timing
import underworld3.cython.petsc_discretisation

import sympy

# Note: Mesh coordinates are always in non-dimensional (scaled) units
# The uw.scaling.non_dimensionalise() function handles unit conversion


@timing.routine_timer_decorator
def UnstructuredSimplexBox(
    minCoords: Tuple = (0.0, 0.0),
    maxCoords: Tuple = (1.0, 1.0),
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    regular: bool = False,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    units=None,
    verbose=False,
):
    r"""
    Create an unstructured simplex mesh on a rectangular box domain.

    Generates a triangular (2D) or tetrahedral (3D) mesh using Gmsh,
    with named boundary labels for applying boundary conditions.

    Parameters
    ----------
    minCoords : tuple of float
        Minimum corner coordinates ``(x_min, y_min)`` for 2D or
        ``(x_min, y_min, z_min)`` for 3D. Supports plain numbers
        (model units) or UWQuantity objects (auto-converted).
    maxCoords : tuple of float
        Maximum corner coordinates ``(x_max, y_max)`` for 2D or
        ``(x_max, y_max, z_max)`` for 3D. Supports plain numbers
        or UWQuantity objects.
    cellSize : float
        Target mesh element size. Controls mesh density; smaller
        values produce finer meshes with more elements.
    degree : int, default=1
        Polynomial degree of finite element basis functions.
        Use ``degree=1`` for linear elements, ``degree=2`` for quadratic.
    qdegree : int, default=2
        Quadrature degree for numerical integration. Should typically
        be at least ``2 * degree`` for accuracy.
    regular : bool, default=False
        If True, use transfinite meshing for a more structured layout.
        Currently only works for 2D meshes.
    filename : str, optional
        Path to save the mesh file. If None, generates a unique name
        in the mesh-file directory (``.meshes/`` by default, or
        ``UW_MESH_CACHE_DIR``) based on mesh parameters.
    refinement : int, optional
        Number of uniform refinement levels to apply after mesh
        generation. Each level approximately quadruples element count.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level. 0 is silent, higher values
        produce more diagnostic output.
    units : str, optional
        Deprecated and ignored (DeprecationWarning). Mesh coordinates are
        always in the model's units (``model.set_reference_quantities``).
    verbose : bool, default=False
        If True, print additional diagnostic information during
        mesh construction.

    Returns
    -------
    Mesh
        An Underworld mesh object with the following boundaries defined:

        **2D boundaries** (accessible via ``mesh.boundaries``):

        - ``Bottom``: :math:`y = y_{min}` edge
        - ``Top``: :math:`y = y_{max}` edge
        - ``Right``: :math:`x = x_{max}` edge
        - ``Left``: :math:`x = x_{min}` edge

        **3D boundaries**:

        - ``Bottom``: :math:`z = z_{min}` face
        - ``Top``: :math:`z = z_{max}` face
        - ``Right``: :math:`x = x_{max}` face
        - ``Left``: :math:`x = x_{min}` face
        - ``Front``: :math:`y = y_{min}` face
        - ``Back``: :math:`y = y_{max}` face

    See Also
    --------
    StructuredQuadBox : For quadrilateral/hexahedral meshes.
    BoxInternalBoundary : For box meshes with an internal interface.

    Examples
    --------
    Create a 2D unit square mesh:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.UnstructuredSimplexBox(
    ...     minCoords=(0.0, 0.0),
    ...     maxCoords=(1.0, 1.0),
    ...     cellSize=0.1
    ... )
    >>> mesh.dim
    2

    Create a 3D box with finer resolution:

    >>> mesh3d = uw.meshing.UnstructuredSimplexBox(
    ...     minCoords=(0.0, 0.0, 0.0),
    ...     maxCoords=(2.0, 1.0, 1.0),
    ...     cellSize=0.05,
    ...     degree=2
    ... )

    Access boundary labels for boundary conditions:

    >>> mesh.boundaries.Bottom
    <boundaries_2D.Bottom: 11>

    Notes
    -----
    Mesh coordinates are always in non-dimensional (scaled) units (set via
    ``uw.scaling.get_coefficients()``). If UWQuantity objects with
    physical units are passed, they are automatically converted using
    ``uw.scaling.non_dimensionalise()``.

    The ``regular=True`` option produces a more structured mesh layout
    but currently only works for 2D meshes.

    """

    class boundaries_2D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14

    class boundaries_3D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14
        Front = 15
        Back = 16

    # Enum is not quite natural but matches the above

    class boundary_normals_2D(Enum):
        Bottom = sympy.Matrix([0, 1])
        Top = sympy.Matrix([0, 1])
        Right = sympy.Matrix([1, 0])
        Left = sympy.Matrix([1, 0])

    class boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1])
        Top = sympy.Matrix([0, 0, 1])
        Right = sympy.Matrix([1, 0, 0])
        Left = sympy.Matrix([1, 0, 0])
        Front = sympy.Matrix([0, 1, 0])
        Back = sympy.Matrix([0, 1, 0])

    # Convert coordinates to non-dimensional units (handles UWQuantity objects)
    minCoords = tuple(uw.scaling.non_dimensionalise(c) for c in minCoords)
    maxCoords = tuple(uw.scaling.non_dimensionalise(c) for c in maxCoords)
    cellSize = uw.scaling.non_dimensionalise(cellSize)

    dim = len(minCoords)
    if dim == 2:
        boundaries = boundaries_2D
        boundary_normals = boundary_normals_2D
    else:
        boundaries = boundaries_3D
        boundary_normals = boundary_normals_3D

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(mesh_file_dir(), exist_ok=True)

        uw_filename = f"{mesh_file_dir()}/uw_simplexbox_minC{minCoords}_maxC{maxCoords}_csize{cellSize}_reg{regular}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Box")

        if dim == 2:
            xmin, ymin = minCoords
            xmax, ymax = maxCoords

            p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
            p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
            p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
            p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

            l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries.Bottom.value)
            l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries.Right.value)
            l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries.Top.value)
            l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries.Left.value)

            cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
            surface = gmsh.model.geo.add_plane_surface([cl])

            gmsh.model.geo.synchronize()

            # Add Physical groups for boundaries
            gmsh.model.add_physical_group(1, [l1], l1)
            gmsh.model.set_physical_name(1, l1, boundaries.Bottom.name)
            gmsh.model.add_physical_group(1, [l2], l2)
            gmsh.model.set_physical_name(1, l2, boundaries.Right.name)
            gmsh.model.add_physical_group(1, [l3], l3)
            gmsh.model.set_physical_name(1, l3, boundaries.Top.name)
            gmsh.model.add_physical_group(1, [l4], l4)
            gmsh.model.set_physical_name(1, l4, boundaries.Left.name)

            gmsh.model.addPhysicalGroup(2, [surface], 99999)
            gmsh.model.setPhysicalName(2, 99999, "Elements")

            if regular:
                gmsh.model.mesh.set_transfinite_surface(surface, cornerTags=[p1, p2, p3, p4])

        else:
            boundaries = boundaries_3D
            boundary_normals = boundary_normals_3D

            xmin, ymin, zmin = minCoords
            xmax, ymax, zmax = maxCoords

            p1 = gmsh.model.geo.add_point(xmin, ymin, zmin, meshSize=cellSize)
            p2 = gmsh.model.geo.add_point(xmax, ymin, zmin, meshSize=cellSize)
            p3 = gmsh.model.geo.add_point(xmin, ymax, zmin, meshSize=cellSize)
            p4 = gmsh.model.geo.add_point(xmax, ymax, zmin, meshSize=cellSize)
            p5 = gmsh.model.geo.add_point(xmin, ymin, zmax, meshSize=cellSize)
            p6 = gmsh.model.geo.add_point(xmax, ymin, zmax, meshSize=cellSize)
            p7 = gmsh.model.geo.add_point(xmin, ymax, zmax, meshSize=cellSize)
            p8 = gmsh.model.geo.add_point(xmax, ymax, zmax, meshSize=cellSize)

            l1 = gmsh.model.geo.add_line(p1, p2)
            l2 = gmsh.model.geo.add_line(p2, p4)
            l3 = gmsh.model.geo.add_line(p4, p3)
            l4 = gmsh.model.geo.add_line(p3, p1)
            l5 = gmsh.model.geo.add_line(p5, p6)
            l6 = gmsh.model.geo.add_line(p6, p8)
            l7 = gmsh.model.geo.add_line(p8, p7)
            l8 = gmsh.model.geo.add_line(p7, p5)
            l9 = gmsh.model.geo.add_line(p5, p1)
            l10 = gmsh.model.geo.add_line(p2, p6)
            l11 = gmsh.model.geo.add_line(p7, p3)
            l12 = gmsh.model.geo.add_line(p4, p8)

            cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
            bottom = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Bottom.value)

            cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
            top = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Top.value)

            cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
            right = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Right.value)

            cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
            left = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Left.value)

            cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
            front = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Front.value)

            cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
            back = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Back.value)

            sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
            volume = gmsh.model.geo.add_volume([sloop])

            gmsh.model.geo.synchronize()

            # Add Physical groups
            for b in boundaries:
                tag = b.value
                name = b.name
                gmsh.model.add_physical_group(2, [tag], tag)
                gmsh.model.set_physical_name(2, tag, name)

            gmsh.model.addPhysicalGroup(3, [volume], 99999)
            gmsh.model.setPhysicalName(3, 99999, "Elements")

        # Generate Mesh
        gmsh.model.mesh.generate(dim)
        write_gmsh(uw_filename)
        gmsh.finalize()

    def box_return_coords_to_bounds(coords):

        epsilon = 1.0e-3

        x00s = coords[:, 0] < minCoords[0]
        x01s = coords[:, 0] > maxCoords[0]
        x10s = coords[:, 1] < minCoords[1]
        x11s = coords[:, 1] > maxCoords[1]

        if dim == 3:
            x20s = coords[:, 2] < minCoords[2]
            x21s = coords[:, 2] > maxCoords[2]

        coords[x00s, 0] = minCoords[0] + epsilon
        coords[x01s, 0] = maxCoords[0] - epsilon
        coords[x10s, 1] = minCoords[1] + epsilon
        coords[x11s, 1] = maxCoords[1] - epsilon

        if dim == 3:
            coords[x20s, 2] = minCoords[2] + epsilon
            coords[x21s, 2] = maxCoords[2] - epsilon

        return coords

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        boundaries=boundaries,
        boundary_normals=boundary_normals,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        refinement=refinement,
        refinement_callback=None,
        return_coords_to_bounds=box_return_coords_to_bounds,
        units=units,
        verbose=verbose,
    )

    # Bounding-surface objects for tangent slip: axis-aligned box faces are
    # planes (see docs/developer/design/boundary-slip-strategy.md).
    from underworld3.meshing.bounding_surface import register_box_face_surfaces
    register_box_face_surfaces(new_mesh, minCoords, maxCoords)

    return new_mesh


@timing.routine_timer_decorator
def BoxInternalBoundary(
    elementRes: Optional[Tuple[int, int, int]] = (8, 8, 8),
    zelementRes: Optional[Tuple[int, int]] = (4, 4),
    cellSize: float = 0.1,
    minCoords: Optional[Tuple[float, float, float]] = (0, 0, 0),
    maxCoords: Optional[Tuple[float, float, float]] = (1, 1, 1),
    zintCoord: float = 0.5,
    simplex: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    units=None,
    verbose=False,
):
    r"""
    Create a box mesh with an internal horizontal boundary.

    Generates a 2D or 3D mesh with an embedded internal boundary surface,
    useful for problems with material interfaces, phase boundaries, or
    layered domains that require flux calculations across the interface.

    Parameters
    ----------
    elementRes : tuple of int, default=(8, 8, 8)
        Number of elements in each direction ``(nx, ny)`` for 2D or
        ``(nx, ny, nz)`` for 3D. Used when ``simplex=False`` (structured).
    zelementRes : tuple of int, default=(4, 4)
        Number of elements ``(n_below, n_above)`` in the vertical direction
        below and above the internal boundary. Allows different resolution
        in each layer.
    cellSize : float, default=0.1
        Target element size for unstructured meshing (``simplex=True``).
        Ignored for structured meshes.
    minCoords : tuple of float, default=(0, 0, 0)
        Minimum corner coordinates. Length determines dimensionality:
        2-tuple for 2D, 3-tuple for 3D.
    maxCoords : tuple of float, default=(1, 1, 1)
        Maximum corner coordinates.
    zintCoord : float, default=0.5
        Vertical coordinate of the internal boundary surface.
        In 2D this is the y-coordinate; in 3D the z-coordinate.
    simplex : bool, default=False
        If False, create a structured quadrilateral/hexahedral mesh.
        If True, create an unstructured triangular/tetrahedral mesh.
    degree : int, default=1
        Polynomial degree of finite element basis functions.
    qdegree : int, default=2
        Quadrature degree for numerical integration.
    filename : str, optional
        Path to save the mesh file. If None, auto-generates in the mesh-file
        directory (``.meshes/`` by default, or ``UW_MESH_CACHE_DIR``).
    refinement : int, optional
        Number of uniform refinement levels to apply.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level.
    units : str, optional
        Deprecated and ignored (DeprecationWarning). Mesh coordinates are
        always in the model's units (``model.set_reference_quantities``).
    verbose : bool, default=False
        Print diagnostic information during mesh construction.

    Returns
    -------
    Mesh
        An Underworld mesh object with boundaries including an internal
        interface:

        **2D boundaries**:

        - ``Bottom``: :math:`y = y_{min}` edge
        - ``Top``: :math:`y = y_{max}` edge
        - ``Right``: :math:`x = x_{max}` edge
        - ``Left``: :math:`x = x_{min}` edge
        - ``Internal``: :math:`y = z_{int}` interface

        **3D boundaries**:

        - ``Bottom``: :math:`z = z_{min}` face
        - ``Top``: :math:`z = z_{max}` face
        - ``Right``: :math:`x = x_{max}` face
        - ``Left``: :math:`x = x_{min}` face
        - ``Front``: :math:`y = y_{min}` face
        - ``Back``: :math:`y = y_{max}` face
        - ``Internal``: :math:`z = z_{int}` interface

    See Also
    --------
    UnstructuredSimplexBox : Box mesh without internal boundary.
    AnnulusInternalBoundary : Annular mesh with internal boundary.

    Examples
    --------
    Create a 2D layered domain with an interface at y=0.5:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.BoxInternalBoundary(
    ...     minCoords=(0.0, 0.0),
    ...     maxCoords=(1.0, 1.0),
    ...     zintCoord=0.5,
    ...     elementRes=(16, 16),
    ...     zelementRes=(8, 8)
    ... )

    Access the internal boundary for flux calculations:

    >>> mesh.boundaries.Internal
    <boundaries_2D.Internal: 15>

    Notes
    -----
    The internal boundary is useful for:

    - Calculating heat flux across a thermal boundary layer
    - Tracking mass flux between mantle and crust
    - Applying different material properties in each layer

    """

    class boundaries_2D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14
        Internal = 15

    class regions_2D(Enum):
        Inner = 101   # Below internal boundary
        Outer = 102   # Above internal boundary

    # TODO(BUG): the external entries below are INWARD-pointing while
    # Mesh.canonical_normal documents the declared normal as outward —
    # audit consumers before flipping them. The Internal entry follows the
    # internal-boundary convention: it points from region Inner to region
    # Outer (+y here; +z in 3D; radially outward, unit_e_0, on annulus and
    # spherical shells).
    class boundary_normals_2D(Enum):
        Bottom = sympy.Matrix([0, 1])
        Top = sympy.Matrix([0, -1])
        Right = sympy.Matrix([-1, 0])
        Left = sympy.Matrix([1, 0])
        Internal = sympy.Matrix([0, 1])

    class boundaries_3D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14
        Front = 15
        Back = 16
        Internal = 17

    class regions_3D(Enum):
        Inner = 101   # Below internal boundary
        Outer = 102   # Above internal boundary

    class boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1])
        Top = sympy.Matrix([0, 0, -1])
        Right = sympy.Matrix([-1, 0, 0])
        Left = sympy.Matrix([1, 0, 0])
        Front = sympy.Matrix([0, -1, 0])
        Back = sympy.Matrix([0, 1, 0])
        Internal = sympy.Matrix([0, 0, 1])

    dim = len(minCoords)

    # Every rank passes these Enums to Mesh() below, so they must be bound
    # on every rank — only the gmsh geometry construction is rank-0 work.
    if dim == 2:
        boundaries = boundaries_2D
        boundary_normals = boundary_normals_2D
    else:
        boundaries = boundaries_3D
        boundary_normals = boundary_normals_3D

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(mesh_file_dir(), exist_ok=True)
        if not simplex:
            # structuredQuadBoxIB
            uw_filename = f"{mesh_file_dir()}/uw_sqbIB_minC{minCoords}_maxC{maxCoords}.msh"
        else:
            uw_filename = f"{mesh_file_dir()}/uw_usbIB_minC{minCoords}_maxC{maxCoords}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Box")

        if dim == 2:
            xmin, ymin = minCoords
            xmax, ymax = maxCoords
            yint = zintCoord

            if not simplex:
                cellSize = 0.0
                nx, ny = elementRes
                ny_a, ny_b = zelementRes

            p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, cellSize)
            p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, cellSize)
            p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, cellSize)
            p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, cellSize)
            p5 = gmsh.model.geo.add_point(xmin, yint, 0.0, cellSize)
            p6 = gmsh.model.geo.add_point(xmax, yint, 0.0, cellSize)

            l1 = gmsh.model.geo.add_line(p1, p2)
            l2 = gmsh.model.geo.add_line(p3, p4)
            l3 = gmsh.model.geo.add_line(p1, p5)
            l4 = gmsh.model.geo.add_line(p5, p3)
            l5 = gmsh.model.geo.add_line(p2, p6)
            l6 = gmsh.model.geo.add_line(p6, p4)
            l7 = gmsh.model.geo.add_line(p5, p6)
            l34 = gmsh.model.geo.add_line(p1, p3)
            l56 = gmsh.model.geo.add_line(p2, p4)

            cl1 = gmsh.model.geo.add_curve_loop((l1, l5, -l7, -l3))
            cl2 = gmsh.model.geo.add_curve_loop((-l2, -l4, l7, l6))
            surface1 = gmsh.model.geo.add_plane_surface([cl1])
            surface2 = gmsh.model.geo.add_plane_surface([cl2])

            gmsh.model.geo.synchronize()

            # Add Physical groups for boundaries
            gmsh.model.add_physical_group(
                1,
                [
                    l1,
                ],
                boundaries.Bottom.value,
            )
            gmsh.model.set_physical_name(1, l1, boundaries.Bottom.name)
            gmsh.model.add_physical_group(1, [l2], boundaries.Top.value)
            gmsh.model.set_physical_name(1, l2, boundaries.Top.name)
            gmsh.model.add_physical_group(1, [l3, l4], boundaries.Left.value)
            gmsh.model.set_physical_name(1, l34, boundaries.Left.name)
            gmsh.model.add_physical_group(1, [l5, l6], boundaries.Right.value)
            gmsh.model.set_physical_name(1, l56, boundaries.Right.name)
            gmsh.model.add_physical_group(1, [l7], boundaries.Internal.value)
            gmsh.model.set_physical_name(1, l7, boundaries.Internal.name)
            # Region physical groups — surface1 is below, surface2 is above
            gmsh.model.addPhysicalGroup(2, [surface1], regions_2D.Inner.value, name=regions_2D.Inner.name)
            gmsh.model.addPhysicalGroup(2, [surface2], regions_2D.Outer.value, name=regions_2D.Outer.name)
            gmsh.model.addPhysicalGroup(2, [surface1, surface2], 99999)
            gmsh.model.setPhysicalName(2, 99999, "Elements")

            if not simplex:
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l1, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l2, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l3, numNodes=ny_b + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l4, numNodes=ny_a + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l5, numNodes=ny_b + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l6, numNodes=ny_a + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    tag=l7, numNodes=nx + 1, meshType="Progression", coef=1.0
                )

                gmsh.model.mesh.set_transfinite_surface(
                    tag=surface1, arrangement="Left", cornerTags=[p1, p2, p5, p6]
                )
                gmsh.model.mesh.set_recombine(2, surface1)
                gmsh.model.mesh.set_transfinite_surface(
                    tag=surface2, arrangement="Left", cornerTags=[p5, p6, p3, p4]
                )
                gmsh.model.mesh.set_recombine(2, surface2)

            gmsh.model.mesh.generate(dim)
            write_gmsh(uw_filename)
            gmsh.finalize()

        if dim == 3:
            xmin, ymin, zmin = minCoords
            xmax, ymax, zmax = maxCoords
            zint = zintCoord

            if not simplex:
                cellSize = 0.0
                nx, ny, nz = elementRes
                nzt, nzb = zelementRes

            p1t = gmsh.model.geo.add_point(xmin, ymin, zmax, cellSize)
            p2t = gmsh.model.geo.add_point(xmax, ymin, zmax, cellSize)
            p3t = gmsh.model.geo.add_point(xmin, ymax, zmax, cellSize)
            p4t = gmsh.model.geo.add_point(xmax, ymax, zmax, cellSize)
            p1b = gmsh.model.geo.add_point(xmin, ymin, zmin, cellSize)
            p2b = gmsh.model.geo.add_point(xmax, ymin, zmin, cellSize)
            p3b = gmsh.model.geo.add_point(xmin, ymax, zmin, cellSize)
            p4b = gmsh.model.geo.add_point(xmax, ymax, zmin, cellSize)
            p1i = gmsh.model.geo.add_point(xmin, ymin, zint, cellSize)
            p2i = gmsh.model.geo.add_point(xmax, ymin, zint, cellSize)
            p3i = gmsh.model.geo.add_point(xmin, ymax, zint, cellSize)
            p4i = gmsh.model.geo.add_point(xmax, ymax, zint, cellSize)

            l1t = gmsh.model.geo.add_line(p1t, p2t)
            l2t = gmsh.model.geo.add_line(p2t, p4t)
            l3t = gmsh.model.geo.add_line(p4t, p3t)
            l4t = gmsh.model.geo.add_line(p3t, p1t)
            l1b = gmsh.model.geo.add_line(p1b, p2b)
            l2b = gmsh.model.geo.add_line(p2b, p4b)
            l3b = gmsh.model.geo.add_line(p4b, p3b)
            l4b = gmsh.model.geo.add_line(p3b, p1b)
            l1i = gmsh.model.geo.add_line(p1i, p2i)
            l2i = gmsh.model.geo.add_line(p2i, p4i)
            l3i = gmsh.model.geo.add_line(p4i, p3i)
            l4i = gmsh.model.geo.add_line(p3i, p1i)

            l5 = gmsh.model.geo.add_line(p1b, p1t)
            l6 = gmsh.model.geo.add_line(p2b, p2t)
            l7 = gmsh.model.geo.add_line(p3b, p3t)
            l8 = gmsh.model.geo.add_line(p4b, p4t)
            l5t = gmsh.model.geo.add_line(p1i, p1t)
            l6t = gmsh.model.geo.add_line(p2i, p2t)
            l7t = gmsh.model.geo.add_line(p3i, p3t)
            l8t = gmsh.model.geo.add_line(p4i, p4t)
            l5b = gmsh.model.geo.add_line(p1b, p1i)
            l6b = gmsh.model.geo.add_line(p2b, p2i)
            l7b = gmsh.model.geo.add_line(p3b, p3i)
            l8b = gmsh.model.geo.add_line(p4b, p4i)

            cl = gmsh.model.geo.add_curve_loop((l1b, l2b, l3b, l4b))
            bottom = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l1t, l2t, l3t, l4t))
            top = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l1i, l2i, l3i, l4i))
            internal = gmsh.model.geo.add_plane_surface([cl])

            cl = gmsh.model.geo.add_curve_loop((l6, l2t, -l8, -l2b))
            right = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l6t, l2t, -l8t, -l2i))
            right_t = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l6b, l2i, -l8b, -l2b))
            right_b = gmsh.model.geo.add_plane_surface([cl])

            cl = gmsh.model.geo.add_curve_loop((l5, -l4t, -l7, l4b))
            left = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l5t, -l4t, -l7t, l4i))
            left_t = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l5b, -l4i, -l7b, l4b))
            left_b = gmsh.model.geo.add_plane_surface([cl])

            cl = gmsh.model.geo.add_curve_loop((l5, l1t, -l6, -l1b))
            front = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l5t, l1t, -l6t, -l1i))
            front_t = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l5b, l1i, -l6b, -l1b))
            front_b = gmsh.model.geo.add_plane_surface([cl])

            cl = gmsh.model.geo.add_curve_loop((l8, l3t, -l7, -l3b))
            back = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l8t, l3t, -l7t, -l3i))
            back_t = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l8b, l3i, -l7b, -l3b))
            back_b = gmsh.model.geo.add_plane_surface([cl])

            sloop1 = gmsh.model.geo.add_surface_loop(
                [front_t, right_t, back_t, top, left_t, internal]
            )
            volume_t = gmsh.model.geo.add_volume([sloop1])
            sloop2 = gmsh.model.geo.add_surface_loop(
                [front_b, right_b, back_b, internal, left_b, bottom]
            )
            volume_b = gmsh.model.geo.add_volume([sloop2])

            gmsh.model.geo.synchronize()

            gmsh.model.add_physical_group(2, [bottom], boundaries.Bottom.value)
            gmsh.model.set_physical_name(2, bottom, boundaries.Bottom.name)
            gmsh.model.add_physical_group(2, [top], boundaries.Top.value)
            gmsh.model.set_physical_name(2, top, boundaries.Top.name)
            gmsh.model.add_physical_group(2, [internal], boundaries.Internal.value)
            gmsh.model.set_physical_name(2, internal, boundaries.Internal.name)
            gmsh.model.add_physical_group(2, [left_t, left_b], boundaries.Left.value)
            gmsh.model.set_physical_name(2, left, boundaries.Left.name)
            gmsh.model.add_physical_group(2, [right_t, right_b], boundaries.Right.value)
            gmsh.model.set_physical_name(2, right, boundaries.Right.name)
            gmsh.model.add_physical_group(2, [front_t, front_b], boundaries.Front.value)
            gmsh.model.set_physical_name(2, front, boundaries.Front.name)
            gmsh.model.add_physical_group(2, [back_t, back_b], boundaries.Back.value)
            gmsh.model.set_physical_name(2, back, boundaries.Back.name)

            # Region physical groups — volume_b is below, volume_t is above
            gmsh.model.addPhysicalGroup(3, [volume_b], regions_3D.Inner.value, name=regions_3D.Inner.name)
            gmsh.model.addPhysicalGroup(3, [volume_t], regions_3D.Outer.value, name=regions_3D.Outer.name)
            gmsh.model.addPhysicalGroup(3, [volume_t, volume_b], 99999)
            gmsh.model.setPhysicalName(3, 99999, "Elements")

            if not simplex:
                gmsh.model.mesh.set_transfinite_curve(
                    l1t, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l2t, numNodes=ny + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l3t, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l4t, numNodes=ny + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l1i, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l2i, numNodes=ny + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l3i, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l4i, numNodes=ny + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l1b, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l2b, numNodes=ny + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l3b, numNodes=nx + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l4b, numNodes=ny + 1, meshType="Progression", coef=1.0
                )

                gmsh.model.mesh.set_transfinite_curve(
                    l5t, numNodes=nzt + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l6t, numNodes=nzt + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l7t, numNodes=nzt + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l8t, numNodes=nzt + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l5b, numNodes=nzb + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l6b, numNodes=nzb + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l7b, numNodes=nzb + 1, meshType="Progression", coef=1.0
                )
                gmsh.model.mesh.set_transfinite_curve(
                    l8b, numNodes=nzb + 1, meshType="Progression", coef=1.0
                )

                gmsh.model.mesh.set_transfinite_surface(
                    tag=bottom, arrangement="Left", cornerTags=[p1b, p2b, p4b, p3b]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=top, arrangement="Left", cornerTags=[p1t, p2t, p4t, p3t]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=internal, arrangement="Left", cornerTags=[p1i, p2i, p4i, p3i]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=front_t, arrangement="Left", cornerTags=[p1i, p2i, p2t, p1t]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=back_t, arrangement="Left", cornerTags=[p3i, p4i, p4t, p3t]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=right_t, arrangement="Left", cornerTags=[p2i, p4i, p4t, p2t]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=left_t, arrangement="Left", cornerTags=[p1i, p3i, p3t, p1t]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=front_b, arrangement="Left", cornerTags=[p1b, p2b, p2i, p1i]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=back_b, arrangement="Left", cornerTags=[p3b, p4b, p4i, p3i]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=right_b, arrangement="Left", cornerTags=[p2b, p4b, p4i, p2i]
                )
                gmsh.model.mesh.set_transfinite_surface(
                    tag=left_b, arrangement="Left", cornerTags=[p1b, p3b, p3i, p1i]
                )
                gmsh.model.mesh.set_recombine(2, bottom)
                gmsh.model.mesh.set_recombine(2, top)
                gmsh.model.mesh.set_recombine(2, internal)
                gmsh.model.mesh.set_recombine(2, front_t)
                gmsh.model.mesh.set_recombine(2, back_t)
                gmsh.model.mesh.set_recombine(2, right_t)
                gmsh.model.mesh.set_recombine(2, left_t)
                gmsh.model.mesh.set_recombine(2, front_b)
                gmsh.model.mesh.set_recombine(2, back_b)
                gmsh.model.mesh.set_recombine(2, right_b)
                gmsh.model.mesh.set_recombine(2, left_b)

                gmsh.model.mesh.set_transfinite_volume(
                    volume_t, cornerTags=[p1i, p2i, p4i, p3i, p1t, p2t, p4t, p3t]
                )
                gmsh.model.mesh.set_transfinite_volume(
                    volume_b, cornerTags=[p1b, p2b, p4b, p3b, p1i, p2i, p4i, p3i]
                )
                gmsh.model.mesh.set_recombine(3, volume_t)
                gmsh.model.mesh.set_recombine(3, volume_b)

            gmsh.model.mesh.generate(dim)
            write_gmsh(uw_filename)
            gmsh.finalize()

    def box_return_coords_to_bounds(coords):
        x00s = coords[:, 0] < minCoords[0]
        x01s = coords[:, 0] > maxCoords[0]
        x10s = coords[:, 1] < minCoords[1]
        x11s = coords[:, 1] > maxCoords[1]

        coords[x00s, :] = minCoords[0]
        coords[x01s, :] = maxCoords[0]
        coords[x10s, :] = minCoords[1]
        coords[x11s, :] = maxCoords[1]

        if dim == 3:
            x20s = coords[:, 1] < minCoords[2]
            x21s = coords[:, 1] > maxCoords[2]
            coords[x20s, :] = minCoords[2]
            coords[x21s, :] = maxCoords[2]

        return coords

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        boundaries=boundaries,
        boundary_normals=boundary_normals,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=False,  # BoxInternalBoundary uses _dm_unstack_bcs instead
        markVertices=True,
        refinement=0.0,
        refinement_callback=None,
        return_coords_to_bounds=box_return_coords_to_bounds,
        units=units,
        verbose=verbose,
    )
    uw.adaptivity._dm_unstack_bcs(new_mesh.dm, new_mesh.boundaries, "Face Sets")

    # Create region labels by classifying cells based on centroid position
    # relative to the internal boundary coordinate
    if dim == 2:
        new_mesh.regions = regions_2D
    else:
        new_mesh.regions = regions_3D

    dm = new_mesh.dm
    depth_label = dm.getLabel("depth")
    cell_is = depth_label.getStratumIS(dim)

    if cell_is:
        cells = cell_is.getIndices()
        coord_sec = dm.getCoordinateSection()
        coord_vec = dm.getCoordinatesLocal()
        coord_arr = coord_vec.array

        for region in new_mesh.regions:
            dm.createLabel(region.name)

        inner_label = dm.getLabel(new_mesh.regions.Inner.name)
        outer_label = dm.getLabel(new_mesh.regions.Outer.name)

        # z-coordinate index: 1 for 2D (y), 2 for 3D (z)
        z_idx = dim - 1

        for cell in cells:
            # Compute centroid from cell vertex coordinates
            closure = dm.getTransitiveClosure(cell)[0]
            vert_coords = []
            for pt in closure:
                ndof = coord_sec.getDof(pt)
                if ndof > 0:
                    off = coord_sec.getOffset(pt)
                    vert_coords.append(coord_arr[off + z_idx])

            if vert_coords:
                centroid_z = sum(vert_coords) / len(vert_coords)
                if centroid_z < zintCoord:
                    inner_label.setValue(cell, new_mesh.regions.Inner.value)
                else:
                    outer_label.setValue(cell, new_mesh.regions.Outer.value)

    return new_mesh


@timing.routine_timer_decorator
def BoxInternalPatch(
    cellSize: float = 0.1,
    minCoords: Optional[Tuple[float, float, float]] = (0, 0, 0),
    maxCoords: Optional[Tuple[float, float, float]] = (1, 1, 1),
    patch_points=None,
    patch_name: str = "Fault",
    patch_cellSize: Optional[float] = None,
    grading_distance: Optional[float] = None,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    units=None,
    verbose=False,
):
    r"""
    Create a 3-D simplex box with an interior conforming surface patch.

    The patch is a planar polygon embedded in the tetrahedral mesh so that
    a set of mesh FACES coincides with it exactly, labelled ``patch_name``.
    Its rim stays strictly inside the box — this is the geometry a
    split-node fault needs (:func:`underworld3.utilities.fault_split.
    split_fault` duplicates everything inside the rim, and the rim is where
    slip must taper to zero). Unlike :func:`BoxInternalBoundary`, the patch
    does not span the domain and does not partition it into regions.

    Parameters
    ----------
    cellSize : float
        Target element size.
    minCoords, maxCoords : tuple of float
        Box corners.
    patch_points : array_like, shape (N, 3)
        Corners of the planar patch polygon, in order around its rim,
        N >= 3. All points must be coplanar and strictly inside the box.
        A disc (penny) patch is a many-sided polygon: sides of length
        ``cellSize`` resolve the rim as well as the mesh can.
    patch_name : str
        Physical name for the patch — the fault name ``split_fault`` and
        ``add_fault_bc`` will use.
    patch_cellSize : float, optional
        Target element size AT the patch. When set, the mesh grades from
        ``patch_cellSize`` on the fault to ``cellSize`` in the far field
        (gmsh Distance/Threshold size field from the patch surface) —
        resolution belongs at the fault, exactly as the 2-D practice
        refines on the cuts. Without it the box is uniform at
        ``cellSize``.
    grading_distance : float, optional
        Distance from the patch at which the size reaches ``cellSize``
        (default ``8 * patch_cellSize``). Only used with
        ``patch_cellSize``.

    Example
    -------
    >>> patch = np.array([[0.5, 0.3, 0.3], [0.5, 0.7, 0.3],
    ...                   [0.5, 0.7, 0.7], [0.5, 0.3, 0.7]])
    >>> mesh = uw.meshing.BoxInternalPatch(cellSize=0.1,
    ...                                    patch_points=patch,
    ...                                    patch_name="FltA")
    >>> child = uw.utilities.fault_split.split_fault(mesh, "FltA")
    """

    class boundaries_3D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14
        Front = 15
        Back = 16

    class boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1])
        Top = sympy.Matrix([0, 0, 1])
        Right = sympy.Matrix([1, 0, 0])
        Left = sympy.Matrix([1, 0, 0])
        Front = sympy.Matrix([0, 1, 0])
        Back = sympy.Matrix([0, 1, 0])

    minCoords = tuple(uw.scaling.non_dimensionalise(c) for c in minCoords)
    maxCoords = tuple(uw.scaling.non_dimensionalise(c) for c in maxCoords)
    cellSize = uw.scaling.non_dimensionalise(cellSize)

    if len(minCoords) != 3:
        raise ValueError("BoxInternalPatch is 3-D; in 2-D a fault is a "
                         "polyline — use Mesh.add_fault on any box mesh.")
    # a FaultSurface is the preferred input: it names itself, its rim
    # polygon becomes the conforming patch, and it is stored on the mesh
    # so add_fault_bc(..., normal="surface") can read the constraint
    # frame from the surface's own face normals. A LIST of patches
    # (FaultSurfaces and/or (name, polygon) pairs) embeds a NETWORK of
    # DISJOINT patches — run prepare_fault_surfaces first; crossing
    # patches are refused here.
    def _one_patch(entry, default_name):
        if hasattr(entry, "rim_polygon") and hasattr(entry, "name"):
            return str(entry.name), np.asarray(entry.rim_polygon(),
                                               dtype=float), entry
        if isinstance(entry, tuple) and len(entry) == 2 \
                and isinstance(entry[0], str):
            return str(entry[0]), np.asarray(entry[1],
                                             dtype=float)[:, :3], None
        return default_name, np.asarray(entry, dtype=float), None

    if isinstance(patch_points, (list,)) and len(patch_points) > 0 \
            and not np.isscalar(patch_points[0]) \
            and (hasattr(patch_points[0], "rim_polygon")
                 or isinstance(patch_points[0], tuple)):
        raw_entries = list(patch_points)
    else:
        raw_entries = [patch_points]
    patches = []
    for k, entry in enumerate(raw_entries):
        nm = patch_name if len(raw_entries) == 1 else f"{patch_name}{k}"
        patches.append(_one_patch(entry, nm))
    if len({nm for nm, _p, _f in patches}) != len(patches):
        raise ValueError(
            f"duplicate patch names: {[nm for nm, _p, _f in patches]}")

    lo, hi = np.asarray(minCoords), np.asarray(maxCoords)
    for nm, patch, _fs in patches:
        if patch.ndim != 2 or patch.shape[0] < 3 or patch.shape[1] != 3:
            raise ValueError(f"patch {nm!r} must be an (N, 3) polygon, "
                             "N >= 3.")
        if not ((patch > lo).all() and (patch < hi).all()):
            raise ValueError(
                f"patch {nm!r} must sit strictly inside the box — a "
                "fault that daylights is not yet supported.")
    if len(patches) > 1:
        from .fault_network_3d import crossing_segment
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                if crossing_segment(patches[i][1], patches[j][1]) \
                        is not None:
                    raise ValueError(
                        f"patches {patches[i][0]!r} and "
                        f"{patches[j][0]!r} cross — the embed takes "
                        "DISJOINT patches; run prepare_fault_surfaces "
                        "(or FaultNetwork.prepare) first.")

    # single-patch variables kept for the legacy filename/store paths
    patch_name = patches[0][0]
    patch = patches[0][1]
    fault_surface = patches[0][2]

    members = {b.name: b.value for b in boundaries_3D}
    patch_tags = {}
    for nm, _p, _fs in patches:
        tag = max(members.values()) + 4
        members[nm] = tag
        patch_tags[nm] = tag
    boundaries = Enum("boundaries", members)
    patch_tag = patch_tags[patch_name]

    if patch_cellSize is not None:
        patch_cellSize = uw.scaling.non_dimensionalise(patch_cellSize)
        if not 0 < patch_cellSize <= cellSize:
            raise ValueError("patch_cellSize must be in (0, cellSize].")
    if grading_distance is None and patch_cellSize is not None:
        grading_distance = 8.0 * patch_cellSize

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(mesh_file_dir(), exist_ok=True)
        grade = ("" if patch_cellSize is None
                 else f"_pcs{patch_cellSize}_gd{grading_distance}")
        tagline = "_".join(f"{nm}{len(p)}" for nm, p, _f in patches)
        uw_filename = (f"{mesh_file_dir()}/uw_boxpatch_minC{minCoords}_"
                       f"maxC{maxCoords}_csize{cellSize}{grade}_"
                       f"{tagline}.msh")
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("BoxInternalPatch")

        box = gmsh.model.occ.addBox(
            minCoords[0], minCoords[1], minCoords[2],
            maxCoords[0] - minCoords[0], maxCoords[1] - minCoords[1],
            maxCoords[2] - minCoords[2])

        patch_surfaces = {}
        for nm, ppoly, _fs in patches:
            corner_tags = [gmsh.model.occ.addPoint(
                *p, patch_cellSize if patch_cellSize is not None
                else cellSize) for p in ppoly]
            line_tags = [gmsh.model.occ.addLine(
                corner_tags[i], corner_tags[(i + 1) % len(ppoly)])
                for i in range(len(ppoly))]
            loop = gmsh.model.occ.addCurveLoop(line_tags)
            patch_surfaces[nm] = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()

        # The embed makes the volume mesh conform to the patches: their
        # faces become tet faces, shared by one cell on each side.
        gmsh.model.mesh.embed(2, list(patch_surfaces.values()), 3, box)

        # OCC numbers the box walls itself; identify them by centre of mass.
        centre = {
            "Bottom": (None, None, minCoords[2]),
            "Top": (None, None, maxCoords[2]),
            "Left": (minCoords[0], None, None),
            "Right": (maxCoords[0], None, None),
            "Front": (None, minCoords[1], None),
            "Back": (None, maxCoords[1], None),
        }
        patch_surface_tags = set(patch_surfaces.values())
        for dim2, tag in gmsh.model.getEntities(2):
            if tag in patch_surface_tags:
                continue
            com = gmsh.model.occ.getCenterOfMass(dim2, tag)
            for wall, probe in centre.items():
                if all(p is None or abs(c - p) < 1e-9
                       for c, p in zip(com, probe)):
                    value = boundaries[wall].value
                    gmsh.model.addPhysicalGroup(2, [tag], value)
                    gmsh.model.setPhysicalName(2, value, wall)
                    break

        for nm, stag in patch_surfaces.items():
            gmsh.model.addPhysicalGroup(2, [stag], patch_tags[nm])
            gmsh.model.setPhysicalName(2, patch_tags[nm], nm)
        gmsh.model.addPhysicalGroup(3, [box], 99999)
        gmsh.model.setPhysicalName(3, 99999, "Elements")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cellSize)
        if patch_cellSize is not None:
            # grade from the fault outward: Distance field measured from
            # the patch surfaces, thresholded to [patch_cellSize, cellSize]
            df = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                df, "SurfacesList", list(patch_surfaces.values()))
            gmsh.model.mesh.field.setNumber(df, "Sampling", 100)
            tf = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tf, "InField", df)
            gmsh.model.mesh.field.setNumber(tf, "SizeMin", patch_cellSize)
            gmsh.model.mesh.field.setNumber(tf, "SizeMax", cellSize)
            gmsh.model.mesh.field.setNumber(tf, "DistMin", patch_cellSize)
            gmsh.model.mesh.field.setNumber(tf, "DistMax",
                                            grading_distance)
            gmsh.model.mesh.field.setAsBackgroundMesh(tf)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # TODO(BUG): patch sets the crossing pre-check cannot classify
        # (coplanar overlaps, patches touching along an edge) can still
        # make the 3-D mesher fail with a raw "PLC Error" from tetgen;
        # the embed should catch that and re-raise with its own
        # message. Fault-session follow-up.
        try:
            gmsh.model.mesh.generate(3)
            write_gmsh(uw_filename)
        finally:
            # A mesher failure must not leave the gmsh session
            # initialized: a poisoned session makes the NEXT mesh
            # constructor in the same process write an invalid .msh,
            # which PETSc rejects with error 79 ("expecting $Nodes").
            gmsh.finalize()

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        boundaries=boundaries,
        boundary_normals=boundary_normals_3D,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        refinement=refinement,
        refinement_callback=None,
        units=units,
        verbose=verbose,
    )

    from underworld3.meshing.bounding_surface import register_box_face_surfaces
    register_box_face_surfaces(new_mesh, minCoords, maxCoords)

    stored = {nm: fs for nm, _p, fs in patches if fs is not None}
    if stored:
        new_mesh._fault_surfaces = stored

    return new_mesh


@timing.routine_timer_decorator
def StructuredQuadBox(
    elementRes: Optional[Tuple[int, int, int]] = (16, 16),
    minCoords: Optional[Tuple[float, float, float]] = None,
    maxCoords: Optional[Tuple[float, float, float]] = None,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    units=None,
    verbose=False,
):
    r"""
    Create a structured quadrilateral or hexahedral box mesh.

    Generates a mesh with regular rectangular (2D) or brick (3D) elements
    using transfinite meshing. Provides precise control over element count
    in each direction.

    Parameters
    ----------
    elementRes : tuple of int, default=(16, 16)
        Number of elements in each direction. Use ``(nx, ny)`` for 2D
        or ``(nx, ny, nz)`` for 3D. This tuple also determines the
        mesh dimensionality.
    minCoords : tuple of float, optional
        Minimum corner coordinates. Defaults to ``(0.0, 0.0)`` for 2D
        or ``(0.0, 0.0, 0.0)`` for 3D based on ``elementRes`` length.
        Supports plain numbers or UWQuantity objects.
    maxCoords : tuple of float, optional
        Maximum corner coordinates. Defaults to ``(1.0, 1.0)`` for 2D
        or ``(1.0, 1.0, 1.0)`` for 3D. Supports UWQuantity objects.
    degree : int, default=1
        Polynomial degree of finite element basis functions.
        Use ``degree=1`` for bilinear/trilinear elements,
        ``degree=2`` for biquadratic/triquadratic.
    qdegree : int, default=2
        Quadrature degree for numerical integration.
    filename : str, optional
        Path to save the mesh file. If None, auto-generates in the mesh-file
        directory (``.meshes/`` by default, or ``UW_MESH_CACHE_DIR``).
    refinement : int, optional
        Number of uniform refinement levels to apply.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level.
    units : str, optional
        Deprecated and ignored (DeprecationWarning). Mesh coordinates are
        always in the model's units (``model.set_reference_quantities``).
    verbose : bool, default=False
        Print diagnostic information during mesh construction.

    Returns
    -------
    Mesh
        An Underworld mesh object with structured elements and boundaries:

        **2D boundaries** (same as UnstructuredSimplexBox):

        - ``Bottom``: :math:`y = y_{min}` edge
        - ``Top``: :math:`y = y_{max}` edge
        - ``Right``: :math:`x = x_{max}` edge
        - ``Left``: :math:`x = x_{min}` edge

        **3D boundaries**:

        - ``Bottom``: :math:`z = z_{min}` face
        - ``Top``: :math:`z = z_{max}` face
        - ``Right``: :math:`x = x_{max}` face
        - ``Left``: :math:`x = x_{min}` face
        - ``Front``: :math:`y = y_{min}` face
        - ``Back``: :math:`y = y_{max}` face

    See Also
    --------
    UnstructuredSimplexBox : For triangular/tetrahedral meshes.
    BoxInternalBoundary : For box meshes with an internal interface.

    Examples
    --------
    Create a 2D structured mesh with 32x32 elements:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.StructuredQuadBox(
    ...     elementRes=(32, 32),
    ...     minCoords=(0.0, 0.0),
    ...     maxCoords=(1.0, 1.0)
    ... )

    Create a 3D mesh (note the 3-element tuple):

    >>> mesh3d = uw.meshing.StructuredQuadBox(
    ...     elementRes=(16, 16, 8),
    ...     maxCoords=(2.0, 2.0, 1.0)
    ... )

    Notes
    -----
    Structured meshes have predictable element layouts which can be
    advantageous for:

    - Consistent interpolation behaviour
    - Benchmark problems with known analytical solutions
    - Simpler mesh-to-mesh comparisons in convergence studies

    The mesh dimensionality is determined by the length of ``elementRes``:
    2-tuple creates a 2D mesh, 3-tuple creates a 3D mesh.

    """
    if minCoords == None:
        minCoords = len(elementRes) * (0.0,)
    if maxCoords == None:
        maxCoords = len(elementRes) * (1.0,)

    import gmsh

    # boundaries = {"Bottom": 1, "Top": 2, "Right": 3, "Left": 4, "Front": 5, "Back": 6}

    class boundaries_2D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14

    class boundaries_3D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14
        Front = 15
        Back = 16

    # Enum is not quite natural but matches the above

    class boundary_normals_2D(Enum):
        Bottom = sympy.Matrix([0, 1])
        Top = sympy.Matrix([0, -1])
        Right = sympy.Matrix([-1, 0])
        Left = sympy.Matrix([1, 0])

    class boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1])
        Top = sympy.Matrix([0, 0, 1])
        Right = sympy.Matrix([1, 0, 0])
        Left = sympy.Matrix([1, 0, 0])
        Front = sympy.Matrix([0, 1, 0])
        Back = sympy.Matrix([0, 1, 0])

    # Convert coordinates to non-dimensional units (handles UWQuantity objects)
    if minCoords is not None:
        minCoords = tuple(uw.scaling.non_dimensionalise(c) for c in minCoords)
    if maxCoords is not None:
        maxCoords = tuple(uw.scaling.non_dimensionalise(c) for c in maxCoords)

    dim = len(minCoords)
    if dim == 2:
        boundaries = boundaries_2D
        boundary_normals = boundary_normals_2D
    else:
        boundaries = boundaries_3D
        boundary_normals = boundary_normals_3D

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(mesh_file_dir(), exist_ok=True)

        # `elementRes` is part of the name because it is part of the mesh.
        # Without it every StructuredQuadBox on the same box — 16x16, 32x32,
        # a resolution sweep — writes and reads ONE path, so two processes at
        # different resolutions race on it and one reads the other's mesh
        # (#618). The annulus already carries its cell size for this reason:
        # `uw_annulus_ro1.0_ri0.5_csize0.05.msh`.
        uw_filename = (
            f"{mesh_file_dir()}/uw_structuredQuadBox_minC{minCoords}"
            f"_maxC{maxCoords}_res{tuple(elementRes)}.msh"
        )
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Box")

        # Create Box Geometry

        if dim == 2:
            xmin, ymin = minCoords
            xmax, ymax = maxCoords

            p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0)
            p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0)
            p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0)
            p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0)

            l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries.Bottom.value)
            l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries.Right.value)
            l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries.Top.value)
            l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries.Left.value)

            cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
            surface = gmsh.model.geo.add_plane_surface([cl])

            gmsh.model.geo.synchronize()

            # Add Physical groups for boundaries
            gmsh.model.add_physical_group(1, [l1], l1)
            gmsh.model.set_physical_name(1, l1, boundaries.Bottom.name)
            gmsh.model.add_physical_group(1, [l2], l2)
            gmsh.model.set_physical_name(1, l2, boundaries.Right.name)
            gmsh.model.add_physical_group(1, [l3], l3)
            gmsh.model.set_physical_name(1, l3, boundaries.Top.name)
            gmsh.model.add_physical_group(1, [l4], l4)
            gmsh.model.set_physical_name(1, l4, boundaries.Left.name)

            gmsh.model.add_physical_group(2, [surface], 99999)
            gmsh.model.set_physical_name(2, 99999, "Elements")

            nx, ny = elementRes
            print("Structured box element resolution", nx, ny)

            gmsh.model.mesh.set_transfinite_curve(
                tag=l1, numNodes=nx + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                tag=l2, numNodes=ny + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                tag=l3, numNodes=nx + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                tag=l4, numNodes=ny + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_surface(
                tag=surface, arrangement="Left", cornerTags=[p1, p2, p3, p4]
            )
            gmsh.model.mesh.set_recombine(2, surface)

        else:
            xmin, ymin, zmin = minCoords
            xmax, ymax, zmax = maxCoords

            p1 = gmsh.model.geo.add_point(xmin, ymin, zmin)
            p2 = gmsh.model.geo.add_point(xmax, ymin, zmin)
            p3 = gmsh.model.geo.add_point(xmin, ymax, zmin)
            p4 = gmsh.model.geo.add_point(xmax, ymax, zmin)
            p5 = gmsh.model.geo.add_point(xmin, ymin, zmax)
            p6 = gmsh.model.geo.add_point(xmax, ymin, zmax)
            p7 = gmsh.model.geo.add_point(xmin, ymax, zmax)
            p8 = gmsh.model.geo.add_point(xmax, ymax, zmax)

            l1 = gmsh.model.geo.add_line(p1, p2)
            l2 = gmsh.model.geo.add_line(p2, p4)
            l3 = gmsh.model.geo.add_line(p4, p3)
            l4 = gmsh.model.geo.add_line(p3, p1)
            l5 = gmsh.model.geo.add_line(p5, p6)
            l6 = gmsh.model.geo.add_line(p6, p8)
            l7 = gmsh.model.geo.add_line(p8, p7)
            l8 = gmsh.model.geo.add_line(p7, p5)
            l9 = gmsh.model.geo.add_line(p5, p1)
            l10 = gmsh.model.geo.add_line(p2, p6)
            l11 = gmsh.model.geo.add_line(p7, p3)
            l12 = gmsh.model.geo.add_line(p4, p8)

            cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
            bottom = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Bottom.value)

            cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
            top = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Top.value)

            cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
            right = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Right.value)

            cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
            left = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Left.value)

            cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
            front = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Front.value)

            cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
            back = gmsh.model.geo.add_plane_surface([cl], tag=boundaries.Back.value)

            sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
            volume = gmsh.model.geo.add_volume([sloop])

            gmsh.model.geo.synchronize()

            nx, ny, nz = elementRes

            gmsh.model.mesh.set_transfinite_curve(
                l1, numNodes=nx + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l2, numNodes=ny + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l3, numNodes=nx + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l4, numNodes=ny + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l5, numNodes=nx + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l6, numNodes=ny + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l7, numNodes=nx + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l8, numNodes=ny + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l9, numNodes=nz + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l10, numNodes=nz + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l11, numNodes=nz + 1, meshType="Progression", coef=1.0
            )
            gmsh.model.mesh.set_transfinite_curve(
                l12, numNodes=nz + 1, meshType="Progression", coef=1.0
            )

            gmsh.model.mesh.set_transfinite_surface(
                tag=bottom, arrangement="Left", cornerTags=[p1, p2, p4, p3]
            )
            gmsh.model.mesh.set_transfinite_surface(
                tag=top, arrangement="Left", cornerTags=[p5, p6, p8, p7]
            )
            gmsh.model.mesh.set_transfinite_surface(
                tag=front, arrangement="Left", cornerTags=[p1, p2, p6, p5]
            )
            gmsh.model.mesh.set_transfinite_surface(
                tag=back, arrangement="Left", cornerTags=[p3, p4, p8, p7]
            )
            gmsh.model.mesh.set_transfinite_surface(
                tag=right, arrangement="Left", cornerTags=[p2, p6, p8, p4]
            )
            gmsh.model.mesh.set_transfinite_surface(
                tag=left, arrangement="Left", cornerTags=[p5, p1, p3, p7]
            )

            gmsh.model.mesh.set_recombine(2, front)
            gmsh.model.mesh.set_recombine(2, back)
            gmsh.model.mesh.set_recombine(2, bottom)
            gmsh.model.mesh.set_recombine(2, top)
            gmsh.model.mesh.set_recombine(2, right)
            gmsh.model.mesh.set_recombine(2, left)

            gmsh.model.mesh.set_transfinite_volume(
                volume, cornerTags=[p1, p2, p4, p3, p5, p6, p8, p7]
            )

            # Add Physical groups
            for b in boundaries:
                tag = b.value
                name = b.name
                gmsh.model.add_physical_group(2, [tag], tag)
                gmsh.model.set_physical_name(2, tag, name)

            gmsh.model.addPhysicalGroup(3, [volume], 99999)
            gmsh.model.setPhysicalName(3, 99999, "Elements")

        # Generate Mesh
        gmsh.model.mesh.generate(dim)
        write_gmsh(uw_filename)
        gmsh.finalize()

    def box_return_coords_to_bounds(coords):

        x00s = coords[:, 0] < minCoords[0]
        x01s = coords[:, 0] > maxCoords[0]
        x10s = coords[:, 1] < minCoords[1]
        x11s = coords[:, 1] > maxCoords[1]

        if dim == 3:
            x20s = coords[:, 2] < minCoords[2]
            x21s = coords[:, 2] > maxCoords[2]

        coords[x00s, 0] = minCoords[0]
        coords[x01s, 0] = maxCoords[0]
        coords[x10s, 1] = minCoords[1]
        coords[x11s, 1] = maxCoords[1]

        if dim == 3:
            coords[x20s, 2] = minCoords[2]
            coords[x21s, 2] = maxCoords[2]

        return coords

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        boundaries=boundaries,
        boundary_normals=boundary_normals,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        refinement=refinement,
        refinement_callback=None,
        return_coords_to_bounds=box_return_coords_to_bounds,
        units=units,
        verbose=verbose,
    )

    # Bounding-surface objects for tangent slip: axis-aligned box faces are
    # planes (see docs/developer/design/boundary-slip-strategy.md).
    from underworld3.meshing.bounding_surface import register_box_face_surfaces
    register_box_face_surfaces(new_mesh, minCoords, maxCoords)

    return new_mesh
