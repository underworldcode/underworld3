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
from underworld3.discretisation import Mesh
from underworld3 import VarType
from underworld3.coordinates import CoordinateSystemType
from underworld3.discretisation import _from_gmsh as gmsh2dmplex
import underworld3.timing as timing
import underworld3.cython.petsc_discretisation

import sympy

# Note: Mesh coordinates are always in model units
# The model.to_model_magnitude() method handles unit conversion


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
        in the ``.meshes/`` directory based on mesh parameters.
    refinement : int, optional
        Number of uniform refinement levels to apply after mesh
        generation. Each level approximately quadruples element count.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level. 0 is silent, higher values
        produce more diagnostic output.
    units : str, optional
        **Deprecated**. Mesh coordinates are always in model reference
        units. This parameter is retained for backward compatibility.
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
    Mesh coordinates are always in model reference units (set via
    ``model.set_reference_quantities()``). If UWQuantity objects with
    physical units are passed, they are automatically converted using
    ``model.to_model_magnitude()``.

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

    # Convert coordinates to model units (handles UWQuantity objects)
    # Mesh coordinates are ALWAYS in model reference units
    model = uw.get_default_model()

    # Detect units from UWQuantity inputs (if not explicitly specified)
    if units is None:
        # Try to detect units from maxCoords (most likely to have units)
        if hasattr(maxCoords, "__iter__"):
            for coord in maxCoords:
                if hasattr(coord, "units"):  # UWQuantity
                    units = str(coord.units)
                    break
                elif hasattr(coord, "_pint_qty"):  # Direct Pint Quantity
                    units = str(coord._pint_qty.units)
                    break

    minCoords = model.to_model_magnitude(minCoords)
    maxCoords = model.to_model_magnitude(maxCoords)
    cellSize = model.to_model_magnitude(cellSize)

    dim = len(minCoords)
    if dim == 2:
        boundaries = boundaries_2D
        boundary_normals = boundary_normals_2D
    else:
        boundaries = boundaries_3D
        boundary_normals = boundary_normals_3D

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_simplexbox_minC{minCoords}_maxC{maxCoords}_csize{cellSize}_reg{regular}.msh"
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
        gmsh.write(uw_filename)
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
        Path to save the mesh file. If None, auto-generates in ``.meshes/``.
    refinement : int, optional
        Number of uniform refinement levels to apply.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level.
    units : str, optional
        Coordinate units for unit-aware arrays.
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

    class boundary_normals_2D(Enum):
        Bottom = sympy.Matrix([0, 1])
        Top = sympy.Matrix([0, -1])
        Right = sympy.Matrix([-1, 0])
        Left = sympy.Matrix([1, 0])
        Internal = sympy.Matrix([0, -1])

    class boundaries_3D(Enum):
        Bottom = 11
        Top = 12
        Right = 13
        Left = 14
        Front = 15
        Back = 16
        Internal = 17

    class boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1])
        Top = sympy.Matrix([0, 0, -1])
        Right = sympy.Matrix([-1, 0, 0])
        Left = sympy.Matrix([1, 0, 0])
        Front = sympy.Matrix([0, -1, 0])
        Back = sympy.Matrix([0, 1, 0])
        Internal = sympy.Matrix([0, 0, 1])

    dim = len(minCoords)

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        if not simplex:
            # structuredQuadBoxIB
            uw_filename = f".meshes/uw_sqbIB_minC{minCoords}_maxC{maxCoords}.msh"
        else:
            uw_filename = f".meshes/uw_usbIB_minC{minCoords}_maxC{maxCoords}.msh"
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
            boundaries = boundaries_2D
            boundary_normals = boundary_normals_2D

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
            gmsh.write(uw_filename)
            gmsh.finalize()

        if dim == 3:
            xmin, ymin, zmin = minCoords
            xmax, ymax, zmax = maxCoords
            zint = zintCoord
            boundaries = boundaries_3D
            boundary_normals = boundary_normals_3D

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
            gmsh.write(uw_filename)
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
        useRegions=False,
        markVertices=True,
        refinement=0.0,
        refinement_callback=None,
        return_coords_to_bounds=box_return_coords_to_bounds,
        units=units,
        verbose=verbose,
    )
    uw.adaptivity._dm_unstack_bcs(new_mesh.dm, new_mesh.boundaries, "Face Sets")
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
        Path to save the mesh file. If None, auto-generates in ``.meshes/``.
    refinement : int, optional
        Number of uniform refinement levels to apply.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level.
    units : str, optional
        **Deprecated**. Mesh coordinates are always in model reference units.
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

    # Convert coordinates to model units (handles UWQuantity objects)
    # Mesh coordinates are ALWAYS in model reference units
    model = uw.get_default_model()

    # Detect units from UWQuantity inputs (if not explicitly specified)
    if units is None:
        # Try to detect units from maxCoords (most likely to have units)
        if maxCoords is not None and hasattr(maxCoords, "__iter__"):
            for coord in maxCoords:
                if hasattr(coord, "units"):  # UWQuantity
                    units = str(coord.units)
                    break
                elif hasattr(coord, "_pint_qty"):  # Direct Pint Quantity
                    units = str(coord._pint_qty.units)
                    break

    if minCoords is not None:
        minCoords = model.to_model_magnitude(minCoords)
    if maxCoords is not None:
        maxCoords = model.to_model_magnitude(maxCoords)

    dim = len(minCoords)
    if dim == 2:
        boundaries = boundaries_2D
        boundary_normals = boundary_normals_2D
    else:
        boundaries = boundaries_3D
        boundary_normals = boundary_normals_3D

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_structuredQuadBox_minC{minCoords}_maxC{maxCoords}.msh"
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
        gmsh.write(uw_filename)
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

    return new_mesh
