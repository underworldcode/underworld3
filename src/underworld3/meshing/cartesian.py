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

from ._utils import extract_coordinates, extract_scalar


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
    verbose=False,
):
    """
    Generates a 2 or 3-dimensional box mesh.

    Parameters
    ----------
    minCoord:
        Tuple specifying minimum mesh location.
    maxCoord:
        Tuple specifying maximum mesh location.

    regular option works in 2D but not (currently) in 3D

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

    # Extract numerical values from coordinates (handles UWQuantity objects)
    minCoords = extract_coordinates(minCoords)
    maxCoords = extract_coordinates(maxCoords)
    cellSize = extract_scalar(cellSize)

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
                gmsh.model.mesh.set_transfinite_surface(
                    surface, cornerTags=[p1, p2, p3, p4]
                )

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

            sloop = gmsh.model.geo.add_surface_loop(
                [front, right, back, top, left, bottom]
            )
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
    verbose=False,
):
    """
    Generates a 2 or 3-dimensional box mesh with internal boundary.

    Parameters
    ----------
    elementRes:
        Tuple specifying number of elements in each axis direction.
    zelementRes:
        Tuple specifying number of elements in the vertical axis direction (half top part and bottom part).
    cellSize : float, optional
        The target size for the mesh elements. This controls the density of the unstructuredSimplexBox mesh.
    minCoord:
        Optional. Tuple specifying minimum mesh location.
    maxCoord:
        Optional. Tuple specifying maximum mesh location.
    zintCoord:
        float specifying internal boundary location.
    simplex: bool, optional
        If True, build structuredQuadBox; if not, build unstructuredSimplexBox. Default is False.
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
    verbose=False,
):
    """
    Generates a 2 or 3-dimensional box mesh.

    Parameters
    ----------
    elementRes:
        Tuple specifying number of elements in each axis direction.
    minCoord:
        Optional. Tuple specifying minimum mesh location.
    maxCoord:
        Optional. Tuple specifying maximum mesh location.
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

    # Extract numerical values from coordinates (handles UWQuantity objects)
    if minCoords is not None:
        minCoords = extract_coordinates(minCoords)
    if maxCoords is not None:
        maxCoords = extract_coordinates(maxCoords)

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

        uw_filename = (
            f".meshes/uw_structuredQuadBox_minC{minCoords}_maxC{maxCoords}.msh"
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

            sloop = gmsh.model.geo.add_surface_loop(
                [front, right, back, top, left, bottom]
            )
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
        verbose=verbose,
    )

    return new_mesh