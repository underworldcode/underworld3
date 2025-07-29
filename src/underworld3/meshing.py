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

import sympy


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


@timing.routine_timer_decorator
def SphericalShell(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 11
        Upper = 12
        Centre = 1

    import gmsh

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_spherical_shell_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Sphere")

        p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)

        ball1_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusOuter)

        if radiusInner > 0.0:
            ball2_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusInner)
            gmsh.model.occ.cut(
                [(3, ball1_tag)], [(3, ball2_tag)], removeObject=True, removeTool=True
            )

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cellSize)
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(2)
        volume = gmsh.model.getEntities(3)[0]

        if radiusInner > 0.0:
            for surface in surfaces:
                if np.isclose(
                    gmsh.model.get_bounding_box(surface[0], surface[1])[-1], radiusInner
                ):
                    gmsh.model.addPhysicalGroup(
                        surface[0],
                        [surface[1]],
                        boundaries.Lower.value,
                        name=boundaries.Lower.name,
                    )
                    print("Created inner boundary surface")
                elif np.isclose(
                    gmsh.model.get_bounding_box(surface[0], surface[1])[-1], radiusOuter
                ):
                    gmsh.model.addPhysicalGroup(
                        surface[0],
                        [surface[1]],
                        boundaries.Upper.value,
                        name=boundaries.Upper.name,
                    )
                    print("Created outer boundary surface")

            gmsh.model.addPhysicalGroup(volume[0], [volume[1]], 99999)
            gmsh.model.setPhysicalName(volume[1], 99999, "Elements")

        else:
            outerSurface = surfaces[0]
            gmsh.model.addPhysicalGroup(
                outerSurface[0],
                [outerSurface[1]],
                boundaries.Upper.value,
                name=boundaries.Upper.name,
            )
            gmsh.model.addPhysicalGroup(volume[0], [volume[1]], 99999)
            gmsh.model.setPhysicalName(volume[1], 99999, "Elements")
            gmsh.model.addPhysicalGroup(0, [p1], tag=boundaries.Centre.value)
            gmsh.model.setPhysicalName(
                0,
                boundaries.Centre.value,
                boundaries.Centre.name,
            )

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(3)
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Ensure boundaries conform (if refined)
    # This is equivalent to a partial function because it already
    # knows the configuration of THIS spherical mesh and
    # is called if the general mesh routine does some refinement
    # and the new dm object needs some tweeks

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = 11
        Upper = 12
        Centre = 1

    return new_mesh


@timing.routine_timer_decorator
def SphericalShellInternalBoundary(
    radiusOuter: float = 1.0,
    radiusInternal: float = 0.8,
    radiusInner: float = 0.547,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    """
    Generates a spherical shell with an internal boundary using Gmsh. The function creates a 3D mesh of a spherical shell
    defined by outer, internal, and inner radii. Mesh size, polynomial degree, and Gmsh verbosity can be customized.

    Parameters:
    -----------
    radiusOuter : float, optional
        The outer radius of the spherical shell. Default is 1.0.

    radiusInternal : float, optional
        The radius of the internal boundary within the spherical shell. Default is 0.8.

    radiusInner : float, optional
        The inner radius of the spherical shell. Default is 0.547.

    cellSize : float, optional
        The target size for the mesh elements. This controls the density of the mesh. Default is 0.1.

    degree : int, optional
        The polynomial degree of the finite elements used in the mesh. Default is 1.

    qdegree : int, optional
        The quadrature degree for integration. Higher values may improve accuracy but increase computation time. Default is 2.

    filename : str, optional
        The name of the file where the mesh will be saved. If None, a default name is generated based on the radii and mesh size. Default is None.

    refinement : optional
        Refinement level or method for the mesh. Used to increase the resolution of the mesh in certain regions. Default is None.

    gmsh_verbosity : int, optional
        Controls the verbosity of Gmsh output. Set to 0 for minimal output, higher numbers for more detailed logs. Default is 0.

    verbose : bool, optional
        If True, the function prints additional information during execution. Default is False.

    Returns:
    --------
    None
        The function generates and saves a mesh file according to the specified parameters.

    Example:
    --------
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=2.0,
        radiusInternal=1.5,
        radiusInner=1.0,
        cellSize=0.05,
        degree=2,
        qdegree=3,
        filename="custom_spherical_shell.msh",
        gmsh_verbosity=1,
        verbose=True
    )
    """

    class boundaries(Enum):
        Centre = 1
        Lower = 11
        Internal = 12
        Upper = 13

    import gmsh

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_spherical_shell_ro{radiusOuter}_rint{radiusInternal}_ri{radiusInner}_csize{cellSize}.msh"
    else:
        uw_filename = filename

    # Check if r_i is greater than 0
    if radiusInner <= 0:
        raise ValueError("The inner radius must be greater than 0.")

    if uw.mpi.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("SphereShell_with_Internal_Surface")

        p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)

        ball1_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusOuter)
        ball2_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusInner)
        # Cut the inner sphere from the outer sphere to create a shell
        gmsh.model.occ.cut(
            [(3, ball1_tag)], [(3, ball2_tag)], removeObject=True, removeTool=True
        )

        ball3_tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radiusInternal)
        ball4_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusInner)
        # Create another inner sphere with radius r_i (for the internal sphere)
        gmsh.model.occ.cut(
            [(3, ball3_tag)], [(3, ball4_tag)], removeObject=True, removeTool=True
        )

        # Set the maximum characteristic length (mesh size) for the mesh elements
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cellSize)
        gmsh.model.occ.synchronize()

        # Embed a 2D surface into a 3D volume
        # Here, 2D entities with tag 6 are embedded into a 3D entity with tag 1
        gmsh.model.mesh.embed(2, [6], 3, 1)
        # Remove specific entities from the model (these repetitions)
        gmsh.model.remove_entities([(3, 2)], [(2, 5)])
        gmsh.model.occ.remove([(3, 2)], [(2, 5)])

        # Get all surface entities (2D) and the first volume entity (3D)
        surfaces = gmsh.model.getEntities(2)
        volume = gmsh.model.getEntities(3)[0]

        # Loop through all surface entities to categorize them based on their bounding box
        for surface in surfaces:
            if np.isclose(
                gmsh.model.get_bounding_box(surface[0], surface[1])[-1], radiusInner
            ):
                gmsh.model.addPhysicalGroup(
                    surface[0],
                    [surface[1]],
                    boundaries.Lower.value,
                    name=boundaries.Lower.name,
                )
                print("Created inner boundary surface")
            elif np.isclose(
                gmsh.model.get_bounding_box(surface[0], surface[1])[-1], radiusOuter
            ):
                gmsh.model.addPhysicalGroup(
                    surface[0],
                    [surface[1]],
                    boundaries.Upper.value,
                    name=boundaries.Upper.name,
                )
                print("Created outer boundary surface")
            elif np.isclose(
                gmsh.model.get_bounding_box(surface[0], surface[1])[-1], radiusInternal
            ):
                gmsh.model.addPhysicalGroup(
                    surface[0],
                    [surface[1]],
                    boundaries.Internal.value,
                    name=boundaries.Internal.name,
                )
                print("Created internal boundary surface")

        # Add the volume entity to a physical group with a high tag number (99999) and name it "Elements"
        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], 99999)
        gmsh.model.setPhysicalName(volume[1], 99999, "Elements")

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(3)
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Ensure boundaries conform (if refined)
    # This is equivalent to a partial function because it already
    # knows the configuration of THIS spherical mesh and
    # is called if the general mesh routine does some refinement
    # and the new dm object needs some tweeks

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = 11
        Internal = 12
        Upper = 13
        Centre = 1

    return new_mesh


@timing.routine_timer_decorator
def SegmentofSphere(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    longitudeExtent: float = 90.0,
    latitudeExtent: float = 90.0,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
    centroid: Tuple = (0.0, 0.0, 0.0),
):
    """
    Generates a segment of a sphere using Gmsh. This function creates a 3D mesh of a spherical segment defined by outer and inner radii,
    and the extent in longitude and latitude. The mesh can be customized in terms of size, polynomial degree, and verbosity.

    Parameters:
    -----------
    radiusOuter : float, optional
        The outer radius of the spherical segment. Default is 1.0.

    radiusInner : float, optional
        The inner radius of the spherical segment. Default is 0.547.

    longitudeExtent : float, optional
        The angular extent of the segment in the longitudinal direction (in degrees). Default is 90.0.

    latitudeExtent : float, optional
        The angular extent of the segment in the latitudinal direction (in degrees). Default is 90.0.

    cellSize : float, optional
        The target size for the mesh elements. This controls the density of the mesh. Default is 0.1.

    degree : int, optional
        The polynomial degree of the finite elements used in the mesh. Default is 1.

    qdegree : int, optional
        The quadrature degree for integration. Higher values may improve accuracy but increase computation time. Default is 2.

    filename : str, optional
        The name of the file where the mesh will be saved. If None, a default name is generated based on the parameters. Default is None.

    refinement : optional
        Refinement level or method for the mesh. Used to increase the resolution of the mesh in certain regions. Default is None.

    gmsh_verbosity : int, optional
        Controls the verbosity of Gmsh output. Set to 0 for minimal output, higher numbers for more detailed logs. Default is 0.

    verbose : bool, optional
        If True, the function prints additional information during execution. Default is False.

    centroid : Tuple[float, float, float], optional
        The coordinates of the centroid (center) of the sphere segment. Default is (0.0, 0.0, 0.0).

    Returns:
    --------
    None
        The function generates and saves a mesh file according to the specified parameters.

    Example:
    --------
    mesh = uw.meshing.SegmentofSphere(
        radiusOuter=2.0,
        radiusInner=1.0,
        longitudeExtent=120.0,
        latitudeExtent=60.0,
        cellSize=0.05,
        degree=2,
        qdegree=3,
        filename="custom_sphere_segment.msh",
        centroid=(0.0, 0.0, 0.0),
        gmsh_verbosity=1,
        verbose=True
    )
    """

    class boundaries(Enum):
        Lower = 11
        Upper = 12
        East = 13
        West = 14
        South = 15
        North = 16

    import gmsh

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_segmentofsphere_ro{radiusOuter}_ri{radiusInner}_longext{longitudeExtent}_latext{latitudeExtent}_csize{cellSize}.msh"
    else:
        uw_filename = filename

    if (
        radiusInner <= 0
        or not (0 < longitudeExtent < 180)
        or not (0 < latitudeExtent < 180)
    ):
        raise ValueError(
            "Invalid input parameters: "
            "radiusInner must be greater than 0, "
            "and longitudeExtent and latitudeExtent must be within the range (0, 180)."
        )

    if uw.mpi.rank == 0:

        def getSphericalXYZ(point):
            """
            Perform Cubed-sphere projection on coordinates.
            Converts (radius, lon, lat) in spherical region to (x, y, z) in spherical region.

            Parameters
            ----------
            Input:
                Coordinates in rthetaphi format (radius, lon, lat)
            Output
                Coordinates in XYZ format (x, y, z)
            """

            (x, y) = (
                math.tan(point[1] * math.pi / 180.0),
                math.tan(point[2] * math.pi / 180.0),
            )
            d = point[0] / math.sqrt(x**2 + y**2 + 1)
            coordX, coordY, coordZ = (
                centroid[0] + d * x,
                centroid[1] + d * y,
                centroid[2] + d,
            )

            return (coordX, coordY, coordZ)

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("SegmentOfSphere")

        p0 = gmsh.model.geo.addPoint(
            centroid[0], centroid[1], centroid[2], meshSize=cellSize
        )

        # Create segment of sphere
        dim = 3

        long_half = longitudeExtent / 2
        lat_half = latitudeExtent / 2

        pt1 = getSphericalXYZ((radiusInner, -long_half, -lat_half))
        pt2 = getSphericalXYZ((radiusInner, long_half, -lat_half))
        pt3 = getSphericalXYZ((radiusInner, long_half, lat_half))
        pt4 = getSphericalXYZ((radiusInner, -long_half, lat_half))
        pt5 = getSphericalXYZ((radiusOuter, -long_half, -lat_half))
        pt6 = getSphericalXYZ((radiusOuter, long_half, -lat_half))
        pt7 = getSphericalXYZ((radiusOuter, long_half, lat_half))
        pt8 = getSphericalXYZ((radiusOuter, -long_half, lat_half))

        p1 = gmsh.model.geo.addPoint(pt1[0], pt1[1], pt1[2], meshSize=cellSize)
        p2 = gmsh.model.geo.addPoint(pt2[0], pt2[1], pt2[2], meshSize=cellSize)
        p3 = gmsh.model.geo.addPoint(pt3[0], pt3[1], pt3[2], meshSize=cellSize)
        p4 = gmsh.model.geo.addPoint(pt4[0], pt4[1], pt4[2], meshSize=cellSize)
        p5 = gmsh.model.geo.addPoint(pt5[0], pt5[1], pt5[2], meshSize=cellSize)
        p6 = gmsh.model.geo.addPoint(pt6[0], pt6[1], pt6[2], meshSize=cellSize)
        p7 = gmsh.model.geo.addPoint(pt7[0], pt7[1], pt7[2], meshSize=cellSize)
        p8 = gmsh.model.geo.addPoint(pt8[0], pt8[1], pt8[2], meshSize=cellSize)

        l1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
        l2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
        l3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
        l4 = gmsh.model.geo.addCircleArc(p4, p0, p1)
        l5 = gmsh.model.geo.addCircleArc(p5, p0, p6)
        l6 = gmsh.model.geo.addCircleArc(p6, p0, p7)
        l7 = gmsh.model.geo.addCircleArc(p7, p0, p8)
        l8 = gmsh.model.geo.addCircleArc(p8, p0, p5)
        l9 = gmsh.model.geo.addLine(p5, p1)
        l10 = gmsh.model.geo.addLine(p2, p6)
        l11 = gmsh.model.geo.addLine(p7, p3)
        l12 = gmsh.model.geo.addLine(p4, p8)

        cl = gmsh.model.geo.addCurveLoop((l1, l2, l3, l4))
        lower = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Lower.value)

        cl = gmsh.model.geo.addCurveLoop((l5, l6, l7, l8))
        upper = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Upper.value)

        cl = gmsh.model.geo.addCurveLoop((l10, l6, l11, -l2))
        east = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.East.value)

        cl = gmsh.model.geo.addCurveLoop((l9, -l4, l12, l8))
        west = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.West.value)

        cl = gmsh.model.geo.addCurveLoop((l1, l10, -l5, l9))
        south = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.South.value)

        cl = gmsh.model.geo.addCurveLoop((-l3, -l11, l7, -l12))
        north = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.North.value)

        sloop = gmsh.model.geo.addSurfaceLoop([south, east, north, upper, west, lower])
        volume = gmsh.model.geo.addVolume([sloop])

        gmsh.model.geo.synchronize()

        # Add Physical groups
        for b in boundaries:
            tag = b.value
            name = b.name
            gmsh.model.addPhysicalGroup(2, [tag], tag)
            gmsh.model.setPhysicalName(2, tag, name)

        # Add the volume entity to a physical group with a high tag number (99999) and name it "Elements"
        gmsh.model.addPhysicalGroup(3, [volume], 99999)
        gmsh.model.setPhysicalName(3, 99999, "Elements")

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(3)
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Ensure boundaries conform (if refined)
    # This is equivalent to a partial function because it already
    # knows the configuration of THIS spherical mesh and
    # is called if the general mesh routine does some refinement
    # and the new dm object needs some tweeks

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = 11
        Upper = 12
        East = 13
        West = 14
        South = 15
        North = 16

    return new_mesh


@timing.routine_timer_decorator
def QuarterAnnulus(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    angle: float = 45,
    cellSize: float = 0.1,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 1
        Upper = 2
        Left = 3
        Right = 4
        Centre = 10

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = (
            f"uw_QuarterAnnulus_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}.msh"
        )
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("QuarterAnnulus")

        p0 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)

        loops = []

        if radiusInner > 0.0:
            p1 = gmsh.model.geo.add_point(radiusInner, 0.0, 0.0, meshSize=cellSize)
            p4 = gmsh.model.geo.add_point(0.0, radiusInner, 0.0, meshSize=cellSize)

        print("add points")

        p2 = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0, meshSize=cellSize)
        p3 = gmsh.model.geo.add_point(0.0, radiusOuter, 0.0, meshSize=cellSize)

        # gmsh.model.geo.rotate([(p2, p3)], 0, 0, 0, 0, 0.3, 0, math.pi / 2)

        if radiusInner > 0.0:
            gmsh.model.geo.rotate([(0, p2)], 0.0, 0.0, 0.0, 0, 0, 1, np.deg2rad(angle))
            gmsh.model.geo.rotate([(0, p3)], 0.0, 0.0, 0.0, 0, 0, 1, np.deg2rad(angle))

            gmsh.model.geo.rotate([(0, p1)], 0.0, 0.0, 0.0, 0, 0, 1, np.deg2rad(angle))
            gmsh.model.geo.rotate([(0, p2)], 0.0, 0.0, 0.0, 0, 0, 1, np.deg2rad(angle))

            l1 = gmsh.model.geo.add_line(p1, p2)
            l3 = gmsh.model.geo.add_line(p3, p4)

            print("add lines")

            c_upper = gmsh.model.geo.add_circle_arc(p2, p0, p3)
            c_lower = gmsh.model.geo.add_circle_arc(p4, p0, p1)

            print("add circles")

            loops = [l1, c_upper, l3, c_lower]

        else:
            gmsh.model.geo.rotate([(0, p2)], 0.0, 0.0, 0.0, 0, 0, 1, np.deg2rad(angle))
            gmsh.model.geo.rotate([(0, p3)], 0.0, 0.0, 0.0, 0, 0, 1, np.deg2rad(angle))

            l1 = gmsh.model.geo.add_line(p0, p2)
            l3 = gmsh.model.geo.add_line(p3, p0)

            c_upper = gmsh.model.geo.add_circle_arc(p2, p0, p3)

            loops = [l1, c_upper, l3]

        loop = gmsh.model.geo.add_curve_loop(loops)

        print("add loop")

        s = gmsh.model.geo.add_plane_surface([loop])

        print("add plane surface")

        gmsh.model.geo.synchronize()

        print("synchronize")

        gmsh.model.mesh.embed(0, [p0], 2, s)

        print("embed")

        if radiusInner > 0.0:
            gmsh.model.addPhysicalGroup(
                1,
                [c_lower],
                boundaries.Lower.value,
                name=boundaries.Lower.name,
            )
        else:
            gmsh.model.addPhysicalGroup(
                0, [p1], tag=boundaries.Centre.value, name=boundaries.Centre.name
            )

        gmsh.model.addPhysicalGroup(
            1, [c_upper], boundaries.Upper.value, name=boundaries.Upper.name
        )

        gmsh.model.addPhysicalGroup(
            1, [l1], boundaries.Left.value, name=boundaries.Left.name
        )

        gmsh.model.addPhysicalGroup(
            1, [l3], boundaries.Right.value, name=boundaries.Right.name
        )

        print("add physical groups")

        gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

        print("add elements")

        gmsh.model.geo.synchronize()

        print("synchronize")

        gmsh.model.mesh.generate(2)

        print("generate")

        gmsh.write(uw_filename)
        gmsh.finalize()

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        verbose=verbose,
    )

    # add boundary normal information to the new mesh
    # this is done now because it requires the coordinate system to be
    # instantiated already (could/should this be done before the mesh is constructed ?)

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0
        Left = new_mesh.CoordinateSystem.unit_e_1
        Right = new_mesh.CoordinateSystem.unit_e_1
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def Annulus(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    cellSize: float = 0.1,
    cellSizeOuter: float = None,
    cellSizeInner: float = None,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 1
        Upper = 2
        Centre = 10

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = (
            f".meshes/uw_annulus_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}.msh"
        )
    else:
        uw_filename = filename

    if cellSizeInner is None:
        cellSizeInner = cellSize

    if cellSizeOuter is None:
        cellSizeOuter = cellSize

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Annulus")

        p1 = gmsh.model.geo.add_point(0.00, 0.00, 0.00, meshSize=cellSizeInner)

        loops = []

        if radiusInner > 0.0:
            p2 = gmsh.model.geo.add_point(radiusInner, 0.0, 0.0, meshSize=cellSizeInner)
            p3 = gmsh.model.geo.add_point(
                -radiusInner, 0.0, 0.0, meshSize=cellSizeInner
            )

            c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
            c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

            cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries.Lower.value)

            loops = [cl1] + loops

        p4 = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0, meshSize=cellSizeOuter)
        p5 = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0, meshSize=cellSizeOuter)

        c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
        c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

        # l1 = gmsh.model.geo.add_line(p5, p4)

        cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries.Upper.value)

        loops = [cl2] + loops

        s = gmsh.model.geo.add_plane_surface(loops)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [p1], 2, s)

        if radiusInner > 0.0:
            gmsh.model.addPhysicalGroup(
                1,
                [c1, c2],
                boundaries.Lower.value,
                name=boundaries.Lower.name,
            )
        else:
            if centre is True:
                gmsh.model.addPhysicalGroup(
                    0, [p1], tag=boundaries.Centre.value, name=boundaries.Centre.name
                )

        gmsh.model.addPhysicalGroup(
            1, [c3, c4], boundaries.Upper.value, name=boundaries.Upper.name
        )
        gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Ensure boundaries conform (if refined)
    # This is equivalent to a partial function because it already
    # knows the configuration of THIS spherical mesh and
    # is called if the general mesh routine does some refinement
    # and the new dm object needs some tweeks

    def annulus_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 2)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    # This needs to respect the size of the elements so it
    # does not flag points that are actually in the mesh.

    def annulus_return_coords_to_bounds(coords):

        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        coords[outside, :] *= 0.99 * radiusOuter / (Rsq[outside] ** 0.5).reshape(-1, 1)
        coords[inside, :] *= 1.01 * radiusInner / (Rsq[inside] ** 0.5).reshape(-1, 1)

        return coords

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        refinement=refinement,
        refinement_callback=annulus_mesh_refinement_callback,
        return_coords_to_bounds=annulus_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def SegmentofAnnulus(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    angleExtent: float = 45,
    cellSize: float = 0.1,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    """
    Generates a segment of an annulus using Gmsh. This function creates a 2D mesh of an annular segment defined by outer and inner radii,
    and the extent of the angle. The mesh can be customized with various parameters like cell size, element degree, and verbosity.

    Parameters:
    -----------
    radiusOuter : float, optional
        The outer radius of the annular segment. Default is 1.0.

    radiusInner : float, optional
        The inner radius of the annular segment. Default is 0.547.

    angleExtent : float, optional
        The angular extent of the segment in degrees. Default is 45.

    cellSize : float, optional
        The target size for the mesh elements. This controls the density of the mesh. Default is 0.1.

    centre : bool, optional
        If True, the segment will be centered at the origin. If False, the segment is positioned based on the radii. Default is False.

    degree : int, optional
        The polynomial degree of the finite elements used in the mesh. Default is 1.

    qdegree : int, optional
        The quadrature degree for integration. Higher values may improve accuracy but increase computation time. Default is 2.

    filename : str, optional
        The name of the file where the mesh will be saved. If None, a default name is generated based on the parameters. Default is None.

    refinement : optional
        Refinement level or method for the mesh. Used to increase the resolution of the mesh in certain regions. Default is None.

    gmsh_verbosity : int, optional
        Controls the verbosity of Gmsh output. Set to 0 for minimal output, higher numbers for more detailed logs. Default is 0.

    verbose : bool, optional
        If True, the function prints additional information during execution. Default is False.

    Returns:
    --------
    None
        The function generates and saves a mesh file according to the specified parameters.

    Example:
    --------
    mesh = uw.meshing.SegmentofAnnulus(
        radiusOuter=2.0,
        radiusInner=1.0,
        angleExtent=90.0,
        cellSize=0.05,
        centre=True,
        degree=2,
        qdegree=3,
        filename="custom_annulus_segment.msh",
        gmsh_verbosity=1,
        verbose=True
    )
    """

    class boundaries(Enum):
        Lower = 1
        Upper = 2
        Left = 3
        Right = 4
        Centre = 10

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f"uw_SegmentOfAnnulus_ro{radiusOuter}_ri{radiusInner}_extent{angleExtent}_csize{cellSize}.msh"
    else:
        uw_filename = filename

    # error checks
    if radiusInner <= 0 or not (0 < angleExtent < 180):
        raise ValueError(
            "Invalid input parameters: "
            "radiusInner must be greater than 0, "
            "and angleExtent must be within the range (0, 180)."
        )

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("SegmentofAnnulus")

        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, meshSize=cellSize)

        # angle Extent in radian
        angleExtentRadian = np.deg2rad(angleExtent)
        theta1 = (np.pi - angleExtentRadian) / 2
        theta2 = theta1 + angleExtentRadian

        loops = []

        if radiusInner > 0.0:
            p1 = gmsh.model.geo.addPoint(
                radiusInner * np.cos(theta1),
                radiusInner * np.sin(theta1),
                0.0,
                meshSize=cellSize,
            )
            p4 = gmsh.model.geo.addPoint(
                radiusInner * np.cos(theta2),
                radiusInner * np.sin(theta2),
                0.0,
                meshSize=cellSize,
            )

        p2 = gmsh.model.geo.addPoint(
            radiusOuter * np.cos(theta1),
            radiusOuter * np.sin(theta1),
            0.0,
            meshSize=cellSize,
        )
        p3 = gmsh.model.geo.addPoint(
            radiusOuter * np.cos(theta2),
            radiusOuter * np.sin(theta2),
            0.0,
            meshSize=cellSize,
        )

        if radiusInner > 0.0:
            l_right = gmsh.model.geo.addLine(p1, p2)
            l_left = gmsh.model.geo.addLine(p3, p4)
            c_upper = gmsh.model.geo.addCircleArc(p2, p0, p3)
            c_lower = gmsh.model.geo.addCircleArc(p4, p0, p1)
            loops = [c_lower, l_right, c_upper, l_left]
        else:
            l_right = gmsh.model.geo.addLine(p0, p2)
            l_left = gmsh.model.geo.addLine(p3, p0)
            c_upper = gmsh.model.geo.addCircleArc(p2, p0, p3)
            loops = [l_right, c_upper, l_left]

        loop = gmsh.model.geo.addCurveLoop(loops)
        s = gmsh.model.geo.addPlaneSurface([loop])

        # gmsh.model.mesh.embed(0, [p0], 2, s) # not sure use of this line

        if radiusInner > 0.0:
            gmsh.model.addPhysicalGroup(
                1, [c_lower], boundaries.Lower.value, name=boundaries.Lower.name
            )
        else:
            gmsh.model.addPhysicalGroup(
                0, [p0], tag=boundaries.Centre.value, name=boundaries.Centre.name
            )

        gmsh.model.addPhysicalGroup(
            1, [c_upper], boundaries.Upper.value, name=boundaries.Upper.name
        )
        gmsh.model.addPhysicalGroup(
            1, [l_left], boundaries.Left.value, name=boundaries.Left.name
        )
        gmsh.model.addPhysicalGroup(
            1, [l_right], boundaries.Right.value, name=boundaries.Right.name
        )
        gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Ensure boundaries conform (if refined)
    # This is equivalent to a partial function because it already
    # knows the configuration of THIS spherical mesh and
    # is called if the general mesh routine does some refinement
    # and the new dm object needs some tweeks

    def annulus_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 2)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    # This needs to respect the size of the elements so it
    # does not flag points that are actually in the mesh.

    def annulus_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        refinement=refinement,
        refinement_callback=annulus_mesh_refinement_callback,
        return_coords_to_bounds=annulus_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def AnnulusWithSpokes(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    cellSizeOuter: float = 0.1,
    cellSizeInner: float = None,
    centre: bool = False,
    spokes: int = 3,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 10
        LowerPlus = 11
        Upper = 20
        UpperPlus = 21
        Centre = 1
        Spokes = 99

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_annulus_ro{radiusOuter}_ri{radiusInner}_csize{cellSizeOuter}.msh"
    else:
        uw_filename = filename

    if cellSizeInner is None:
        cellSizeInner = cellSizeOuter

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Annulus")

        theta = 2 * np.pi / spokes

        p0 = gmsh.model.geo.add_point(0.00, 0.00, 0.00, meshSize=cellSizeInner)

        loops = []
        outer_segments = []
        inner_segments = []
        spoke_segments = []

        if radiusInner > 0.0:
            p1 = gmsh.model.geo.add_point(0.0, radiusInner, 0.0, meshSize=cellSizeInner)
            p2 = gmsh.model.geo.add_point(
                radiusInner * np.sin(theta),
                radiusInner * np.cos(theta),
                0.0,
                meshSize=cellSizeInner,
            )
            c1 = gmsh.model.geo.add_circle_arc(p1, p0, p2)

            inner_segments.append(c1)

        p3 = gmsh.model.geo.add_point(0.0, radiusOuter, 0.0, meshSize=cellSizeOuter)
        p4 = gmsh.model.geo.add_point(
            radiusOuter * np.sin(theta),
            radiusOuter * np.cos(theta),
            0.0,
            meshSize=cellSizeOuter,
        )

        c2 = gmsh.model.geo.add_circle_arc(p3, p0, p4)

        outer_segments.append(c2)

        if radiusInner > 0.0:
            l1 = gmsh.model.geo.add_line(p1, p3)
            l2 = gmsh.model.geo.add_line(p2, p4)

            cl1 = gmsh.model.geo.add_curve_loop([c1, l2, -c2, -l1])

        else:
            l1 = gmsh.model.geo.add_line(p0, p3)
            l2 = gmsh.model.geo.add_line(p0, p4)

            cl1 = gmsh.model.geo.add_curve_loop([l2, -c2, -l1])

        spoke_segments.append(l1)
        spoke_segments.append(l2)

        gmsh.model.geo.addPlaneSurface([cl1], tag=-1)
        loops.append(cl1)

        # Now copy / rotate

        for i in range(1, spokes):
            new_slice = gmsh.model.geo.copy([(2, cl1)])
            gmsh.model.geo.rotate(new_slice, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, i * theta)
            gmsh.model.geo.synchronize()

            _, new_lines = gmsh.model.get_adjacencies(2, new_slice[0][1])

            loops.append(new_slice[0][1])

            if radiusInner > 0:
                inner_segments.append(new_lines[0])
                outer_segments.append(new_lines[2])
                spoke_segments.append(new_lines[1])
                spoke_segments.append(new_lines[3])
            else:
                outer_segments.append(new_lines[1])
                spoke_segments.append(new_lines[0])
                spoke_segments.append(new_lines[2])

        gmsh.model.geo.synchronize()
        # We finally generate and save the mesh:

        gmsh.model.addPhysicalGroup(1, outer_segments, boundaries.Upper.value)
        gmsh.model.setPhysicalName(1, boundaries.Upper.value, boundaries.Upper.name)

        gmsh.model.addPhysicalGroup(1, inner_segments, boundaries.Lower.value)
        gmsh.model.setPhysicalName(1, boundaries.Lower.value, boundaries.Lower.name)

        gmsh.model.addPhysicalGroup(1, spoke_segments, boundaries.Spokes.value)
        gmsh.model.setPhysicalName(1, boundaries.Spokes.value, boundaries.Spokes.name)

        gmsh.model.addPhysicalGroup(2, loops, 30)
        gmsh.model.setPhysicalName(2, 30, "Elements")

        if radiusInner > 0.0:
            gmsh.model.remove_entities([(0, p0)])

        gmsh.model.mesh.generate(2)

        gmsh.write(uw_filename)

        # We need to build the plex here in order to make some changes
        # before the mesh gets built

        plex_0 = gmsh2dmplex(
            uw_filename,
            useMultipleTags=True,
            useRegions=True,
            markVertices=True,
            comm=PETSc.COMM_SELF,
        )

        # Composite label - upper + wedge slices

        ul = plex_0[1].getLabel(boundaries.Upper.name)
        sl = plex_0[1].getLabel(boundaries.Spokes.name)

        ul_is = ul.getStratumIS(boundaries.Upper.value)
        sl_is = sl.getStratumIS(boundaries.Spokes.value)

        new_is = ul_is.union(sl_is)

        plex_0[1].createLabel(boundaries.UpperPlus.name)
        both_lab = plex_0[1].getLabel(boundaries.UpperPlus.name)
        both_lab.setStratumIS(boundaries.UpperPlus.value, new_is)

        if radiusInner > 0.0:
            ll = plex_0[1].getLabel(boundaries.Lower.name)
            sl = plex_0[1].getLabel(boundaries.Spokes.name)

            ll_is = ll.getStratumIS(boundaries.Lower.value)
            sl_is = sl.getStratumIS(boundaries.Spokes.value)

            new_is = ll_is.union(sl_is)

            plex_0[1].createLabel(boundaries.LowerPlus.name)
            both_lab = plex_0[1].getLabel(boundaries.LowerPlus.name)
            both_lab.setStratumIS(boundaries.LowerPlus.value, new_is)

        ####

        viewer = PETSc.ViewerHDF5().create(
            uw_filename + ".h5", "w", comm=PETSc.COMM_SELF
        )

        viewer(plex_0[1])

    # Ensure boundaries conform (if refined)
    # This is equivalent to a partial function because it already
    # knows the configuration of THIS spherical mesh and
    # is called if the general mesh routine does some refinement
    # and the new dm object needs some tweeks

    def annulus_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 2)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    def annulus_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    new_mesh = Mesh(
        uw_filename + ".h5",
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        refinement=refinement,
        refinement_callback=annulus_mesh_refinement_callback,
        return_coords_to_bounds=annulus_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0 * sympy.Piecewise(
            (1.0, new_mesh.CoordinateSystem.R[0] < 1.01 * radiusInner),
            (0.0, True),
        )
        Upper = new_mesh.CoordinateSystem.unit_e_0 * sympy.Piecewise(
            (1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radiusOuter),
            (0.0, True),
        )
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def AnnulusInternalBoundary(
    radiusOuter: float = 1.5,
    radiusInternal: float = 1.0,
    radiusInner: float = 0.547,
    cellSize: float = 0.1,
    cellSize_Outer: float = None,
    cellSize_Inner: float = None,
    cellSize_Internal: float = None,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 1
        Internal = 2
        Upper = 3
        Centre = 10

    if cellSize_Inner is None:
        cellSize_Inner = cellSize

    if cellSize_Outer is None:
        cellSize_Outer = cellSize

    if cellSize_Internal is None:
        cellSize_Internal = cellSize

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_annulus_internalBoundary_rO{radiusOuter}rInt{radiusInternal}_rI{radiusInner}_csize{cellSize}_csizefs{cellSize_Outer}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("AnnulusFS")

        p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize_Inner)

        loops = []

        if radiusInner > 0.0:
            p2 = gmsh.model.geo.add_point(
                radiusInner, 0.0, 0.0, meshSize=cellSize_Inner
            )
            p3 = gmsh.model.geo.add_point(
                -radiusInner, 0.0, 0.0, meshSize=cellSize_Inner
            )

            c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
            c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

            cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries.Lower.value)

            loops = [cl1] + loops

        p4 = gmsh.model.geo.add_point(
            radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal
        )
        p5 = gmsh.model.geo.add_point(
            -radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal
        )

        c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
        c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

        cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries.Internal.value)

        ### adding this curve loop results in the mesh not being generated correctly
        ### although the internal boundary is still defined in the mesh dm
        # loops = [cl2] + loops

        # Outermost mesh

        p6 = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0, meshSize=cellSize_Outer)
        p7 = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0, meshSize=cellSize_Outer)

        c5 = gmsh.model.geo.add_circle_arc(p6, p1, p7)
        c6 = gmsh.model.geo.add_circle_arc(p7, p1, p6)

        cl3 = gmsh.model.geo.add_curve_loop([c5, c6], tag=boundaries.Upper.value)

        loops = [cl3] + loops

        s = gmsh.model.geo.add_plane_surface(loops)

        gmsh.model.geo.synchronize()

        if radiusInner == 0.0:
            gmsh.model.mesh.embed(0, [p1], 2, s)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [c3, c4], 2, s)

        gmsh.model.geo.synchronize()

        if radiusInner > 0.0:
            gmsh.model.addPhysicalGroup(
                1, [c1, c2], boundaries.Lower.value, name=boundaries.Lower.name
            )
        else:
            if centre is True:
                gmsh.model.addPhysicalGroup(
                    0, [p1], tag=boundaries.Centre.value, name=boundaries.Centre.name
                )

        gmsh.model.addPhysicalGroup(
            1,
            [c3, c4],
            boundaries.Internal.value,
            name=boundaries.Internal.name,
        )
        gmsh.model.addPhysicalGroup(
            1,
            [c5, c6],
            boundaries.Upper.value,
            name=boundaries.Upper.name,
        )

        gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(uw_filename)
        gmsh.finalize()

    ## This is the same as the simple annulus
    def annulus_internal_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    ## This has an additional step to move the inner boundary
    def annulus_internal_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner
        r_int = radiusInternal

        import underworld3 as uw

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 2)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        internalIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Internal"
            )
        )

        coords[internalIndices] *= r_int / (1.0e-16 + R[internalIndices].reshape(-1, 1))

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        refinement_callback=annulus_internal_mesh_refinement_callback,
        return_coords_to_bounds=annulus_internal_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0
        Internal = new_mesh.CoordinateSystem.unit_e_0
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def DiscInternalBoundaries(
    radiusUpper: float = 1.5,
    radiusInternal: float = 1.0,
    radiusLower: float = 0.547,
    cellSize: float = 0.1,
    cellSize_Upper: float = None,
    cellSize_Lower: float = None,
    cellSize_Internal: float = None,
    cellSize_Centre: float = None,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 1
        Internal = 2
        Upper = 3
        Centre = 10

    if cellSize_Lower is None:
        cellSize_Lower = cellSize

    if cellSize_Upper is None:
        cellSize_Upper = cellSize

    if cellSize_Internal is None:
        cellSize_Internal = cellSize

    if cellSize_Centre is None:
        cellSize_Centre = cellSize

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_disc_internalBoundaries_rO{radiusUpper}rInt{radiusInternal}_rI{radiusLower}_csize{cellSize}_csizefs{cellSize_Upper}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("AnnulusFS")

        p1 = gmsh.model.geo.add_point(1.0e-16, 0.0, 0.0, meshSize=cellSize_Centre)

        loops = []

        p2 = gmsh.model.geo.add_point(radiusLower, 0.0, 0.0, meshSize=cellSize_Lower)
        p3 = gmsh.model.geo.add_point(-radiusLower, 0.0, 0.0, meshSize=cellSize_Lower)

        c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
        c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

        cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries.Lower.value)

        # loops = [cl1] + loops

        p4 = gmsh.model.geo.add_point(
            radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal
        )
        p5 = gmsh.model.geo.add_point(
            -radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal
        )

        c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
        c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

        cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries.Internal.value)

        # Outermost mesh

        p6 = gmsh.model.geo.add_point(radiusUpper, 0.0, 0.0, meshSize=cellSize_Upper)
        p7 = gmsh.model.geo.add_point(-radiusUpper, 0.0, 0.0, meshSize=cellSize_Upper)

        c5 = gmsh.model.geo.add_circle_arc(p6, p1, p7)
        c6 = gmsh.model.geo.add_circle_arc(p7, p1, p6)

        cl3 = gmsh.model.geo.add_curve_loop([c5, c6], tag=boundaries.Upper.value)

        loops = [cl3] + loops

        s = gmsh.model.geo.add_plane_surface(loops)

        ## Now add embedded surfaces

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [p1], 2, s)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [c1, c2], 2, s)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [c3, c4], 2, s)

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(
            1, [c1, c2], boundaries.Lower.value, name=boundaries.Lower.name
        )

        gmsh.model.addPhysicalGroup(
            0, [p1], tag=boundaries.Centre.value, name=boundaries.Centre.name
        )

        gmsh.model.addPhysicalGroup(
            1,
            [c3, c4],
            boundaries.Internal.value,
            name=boundaries.Internal.name,
        )
        gmsh.model.addPhysicalGroup(
            1,
            [c5, c6],
            boundaries.Upper.value,
            name=boundaries.Upper.name,
        )

        gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(uw_filename)
        gmsh.finalize()

    ## This is the same as the simple annulus
    def annulus_internal_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    ## This has an additional step to move the inner boundary
    def annulus_internal_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner
        r_int = radiusInternal

        import underworld3 as uw

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 2)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        internalIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Internal"
            )
        )

        coords[internalIndices] *= r_int / (1.0e-16 + R[internalIndices].reshape(-1, 1))

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        refinement_callback=annulus_internal_mesh_refinement_callback,
        return_coords_to_bounds=annulus_internal_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0
        Internal = new_mesh.CoordinateSystem.unit_e_0
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


# # ToDo: Not sure if this works really ...


@timing.routine_timer_decorator
def CubedSphere(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    numElements: int = 5,
    degree: int = 1,
    qdegree: int = 2,
    simplex: bool = False,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    """Cubed Sphere mesh in hexahedra (which can be left uncombined to produce a simplex-based mesh
    The number of elements is the edge of each cube"""

    class boundaries(Enum):
        Lower = 1
        Upper = 2

    r1 = radiusInner / np.sqrt(3)
    r2 = radiusOuter / np.sqrt(3)

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = f".meshes/uw_cubed_spherical_shell_ro{radiusOuter}_ri{radiusInner}_elts{numElements}_plex{simplex}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.add("Cubed Sphere")

        center_point = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=1)

        gmsh.model.geo.addPoint(r2, r2, -r2, tag=2)
        gmsh.model.geo.addPoint(-r2, r2, -r2, tag=3)
        gmsh.model.geo.addPoint(-r2, -r2, -r2, tag=4)
        gmsh.model.geo.addPoint(r2, -r2, -r2, tag=5)

        gmsh.model.geo.addCircleArc(3, 1, 2, tag=1)
        gmsh.model.geo.addCircleArc(2, 1, 5, tag=2)
        gmsh.model.geo.addCircleArc(5, 1, 4, tag=3)
        gmsh.model.geo.addCircleArc(4, 1, 3, tag=4)

        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)
        gmsh.model.geo.addSurfaceFilling([1], tag=1, sphereCenterTag=1)

        gmsh.model.geo.addPoint(r1, r1, -r1, tag=6)
        gmsh.model.geo.addPoint(-r1, r1, -r1, tag=7)
        gmsh.model.geo.addPoint(-r1, -r1, -r1, tag=8)
        gmsh.model.geo.addPoint(r1, -r1, -r1, tag=9)

        gmsh.model.geo.addCircleArc(7, 1, 6, tag=5)
        gmsh.model.geo.addCircleArc(6, 1, 9, tag=6)
        gmsh.model.geo.addCircleArc(9, 1, 8, tag=7)
        gmsh.model.geo.addCircleArc(8, 1, 7, tag=8)

        gmsh.model.geo.addCurveLoop([5, 6, 7, 8], tag=2)
        gmsh.model.geo.addSurfaceFilling([2], tag=2, sphereCenterTag=1)

        gmsh.model.geo.addLine(2, 6, tag=9)
        gmsh.model.geo.addLine(3, 7, tag=10)
        gmsh.model.geo.addLine(5, 9, tag=11)
        gmsh.model.geo.addLine(4, 8, tag=12)

        gmsh.model.geo.addCurveLoop([3, 12, -7, -11], tag=3)
        gmsh.model.geo.addSurfaceFilling([3], tag=3)

        gmsh.model.geo.addCurveLoop([10, 5, -9, -1], tag=4)
        gmsh.model.geo.addSurfaceFilling([4], tag=4)

        gmsh.model.geo.addCurveLoop([9, 6, -11, -2], tag=5)
        gmsh.model.geo.addSurfaceFilling([5], tag=5)

        gmsh.model.geo.addCurveLoop([12, 8, -10, -4], tag=6)
        gmsh.model.geo.addSurfaceFilling([6], tag=6)

        gmsh.model.geo.addSurfaceLoop([2, 4, 6, 3, 1, 5], tag=1)
        gmsh.model.geo.addVolume([1], tag=1)

        # Make copies
        gmsh.model.geo.rotate(
            gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.pi / 2.0
        )
        gmsh.model.geo.rotate(
            gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.pi
        )
        gmsh.model.geo.rotate(
            gmsh.model.geo.copy([(3, 1)]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            3.0 * np.pi / 2.0,
        )
        gmsh.model.geo.rotate(
            gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.pi / 2.0
        )
        gmsh.model.geo.rotate(
            gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -np.pi / 2.0
        )

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(
            2, [1, 34, 61, 88, 115, 137], boundaries.Upper.value
        )
        gmsh.model.setPhysicalName(2, boundaries.Upper.value, "Upper")

        gmsh.model.addPhysicalGroup(2, [2, 14, 41, 68, 95, 117], boundaries.Lower.value)
        gmsh.model.setPhysicalName(2, boundaries.Lower.value, "Lower")

        gmsh.model.addPhysicalGroup(3, [1, 13, 40, 67, 94, 116], 99999)
        gmsh.model.setPhysicalName(3, 99999, "Elements")

        for _, line in gmsh.model.get_entities(1):
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=numElements + 1)

        for _, surface in gmsh.model.get_entities(2):
            gmsh.model.mesh.setTransfiniteSurface(surface)
            if not simplex:
                gmsh.model.mesh.set_recombine(2, surface)

        if not simplex:
            for _, volume in gmsh.model.get_entities(3):
                gmsh.model.mesh.set_transfinite_volume(volume)
                # if not simplex:
                gmsh.model.mesh.set_recombine(3, volume)

        # Generate Mesh
        gmsh.model.mesh.generate(3)
        gmsh.write(uw_filename)
        gmsh.finalize()

    def sphere_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        ## Note these numbers should not be hard-wired

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        return_coords_to_bounds=sphere_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


# ToDo: if keeping, we need to add boundaries etc.


@timing.routine_timer_decorator
def RegionalSphericalBox(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    SWcorner=[-45, -45],
    NEcorner=[+45, +45],
    numElementsLon: int = 5,
    numElementsLat: int = 5,
    numElementsDepth: int = 5,
    degree: int = 1,
    qdegree: int = 2,
    simplex: bool = False,
    filename=None,
    refinement=None,
    coarsening=None,
    gmsh_verbosity=0,
    verbose=False,
):
    """One section of the cube-sphere mesh - currently there is no choice of the lateral extent"""

    class boundaries(Enum):
        Lower = 1
        Upper = 2
        North = 4
        South = 3
        East = 6
        West = 5

    lt_min = np.radians(SWcorner[1])
    lt_max = np.radians(NEcorner[1])
    ln_min = np.radians(SWcorner[0])
    ln_max = np.radians(NEcorner[0])

    p2 = (
        radiusOuter * np.cos(lt_max) * np.cos(ln_max),
        radiusOuter * np.cos(lt_max) * np.sin(ln_max),
        radiusOuter * np.sin(lt_max),
    )

    p3 = (
        radiusOuter * np.cos(lt_max) * np.cos(ln_min),
        radiusOuter * np.cos(lt_max) * np.sin(ln_min),
        radiusOuter * np.sin(lt_max),
    )

    p4 = (
        radiusOuter * np.cos(lt_min) * np.cos(ln_min),
        radiusOuter * np.cos(lt_min) * np.sin(ln_min),
        radiusOuter * np.sin(lt_min),
    )

    p5 = (
        radiusOuter * np.cos(lt_min) * np.cos(ln_max),
        radiusOuter * np.cos(lt_min) * np.sin(ln_max),
        radiusOuter * np.sin(lt_min),
    )

    p6 = (
        radiusInner * np.cos(lt_max) * np.cos(ln_max),
        radiusInner * np.cos(lt_max) * np.sin(ln_max),
        radiusInner * np.sin(lt_max),
    )

    p7 = (
        radiusInner * np.cos(lt_max) * np.cos(ln_min),
        radiusInner * np.cos(lt_max) * np.sin(ln_min),
        radiusInner * np.sin(lt_max),
    )

    p8 = (
        radiusInner * np.cos(lt_min) * np.cos(ln_min),
        radiusInner * np.cos(lt_min) * np.sin(ln_min),
        radiusInner * np.sin(lt_min),
    )

    p9 = (
        radiusInner * np.cos(lt_min) * np.cos(ln_max),
        radiusInner * np.cos(lt_min) * np.sin(ln_max),
        radiusInner * np.sin(lt_min),
    )

    # lat_south = np.radians(centralLatitude - latitudeExtent/2)
    # lat_north = np.radians(centralLatitude + latitudeExtent/2)

    # ss = min(longitudeExtent / 90, 1.99) * np.cos(lat_south)/np.cos(np.pi/4)
    # sn = min(longitudeExtent / 90, 1.99) * np.cos(lat_north)/np.cos(np.pi/4)
    # t = min(latitudeExtent / 90, 1.99)

    # r1 = radiusInner / np.sqrt(3)
    # r2 = radiusOuter / np.sqrt(3)

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = f".meshes/uw_cubed_spherical_shell_ro{radiusOuter}_ri{radiusInner}_elts{numElementsDepth}_plex{simplex}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.add("Cubed Sphere")

        center_point = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=1)

        gmsh.model.geo.addPoint(p2[0], p2[1], p2[2], tag=2)
        gmsh.model.geo.addPoint(p3[0], p3[1], p3[2], tag=3)
        gmsh.model.geo.addPoint(p4[0], p4[1], p4[2], tag=4)
        gmsh.model.geo.addPoint(p5[0], p5[1], p5[2], tag=5)

        gmsh.model.geo.addCircleArc(3, 1, 2, tag=1)
        gmsh.model.geo.addCircleArc(2, 1, 5, tag=2)
        gmsh.model.geo.addCircleArc(5, 1, 4, tag=3)
        gmsh.model.geo.addCircleArc(4, 1, 3, tag=4)

        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)
        gmsh.model.geo.addSurfaceFilling([1], tag=1, sphereCenterTag=1)

        gmsh.model.geo.addPoint(p6[0], p6[1], p6[2], tag=6)
        gmsh.model.geo.addPoint(p7[0], p7[1], p7[2], tag=7)
        gmsh.model.geo.addPoint(p8[0], p8[1], p8[2], tag=8)
        gmsh.model.geo.addPoint(p9[0], p9[1], p9[2], tag=9)

        gmsh.model.geo.addCircleArc(7, 1, 6, tag=5)
        gmsh.model.geo.addCircleArc(6, 1, 9, tag=6)
        gmsh.model.geo.addCircleArc(9, 1, 8, tag=7)
        gmsh.model.geo.addCircleArc(8, 1, 7, tag=8)

        gmsh.model.geo.addCurveLoop([5, 6, 7, 8], tag=2)
        gmsh.model.geo.addSurfaceFilling([2], tag=2, sphereCenterTag=1)

        gmsh.model.geo.addLine(2, 6, tag=9)
        gmsh.model.geo.addLine(3, 7, tag=10)
        gmsh.model.geo.addLine(5, 9, tag=11)
        gmsh.model.geo.addLine(4, 8, tag=12)

        gmsh.model.geo.addCurveLoop([3, 12, -7, -11], tag=3)
        gmsh.model.geo.addSurfaceFilling([3], tag=3)

        gmsh.model.geo.addCurveLoop([10, 5, -9, -1], tag=4)
        gmsh.model.geo.addSurfaceFilling([4], tag=4)

        gmsh.model.geo.addCurveLoop([9, 6, -11, -2], tag=5)
        gmsh.model.geo.addSurfaceFilling([5], tag=5)

        gmsh.model.geo.addCurveLoop([12, 8, -10, -4], tag=6)
        gmsh.model.geo.addSurfaceFilling([6], tag=6)

        gmsh.model.geo.addSurfaceLoop([2, 4, 6, 3, 1, 5], tag=1)
        gmsh.model.geo.addVolume([1], tag=1)

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], boundaries.Upper.value)
        gmsh.model.setPhysicalName(2, boundaries.Upper.value, "Upper")

        gmsh.model.addPhysicalGroup(2, [2], boundaries.Lower.value)
        gmsh.model.setPhysicalName(2, boundaries.Lower.value, "Lower")

        ## These probably have the wrong names ... check ordering

        gmsh.model.addPhysicalGroup(2, [4], boundaries.North.value)
        gmsh.model.setPhysicalName(2, boundaries.North.value, "North")

        gmsh.model.addPhysicalGroup(2, [6], boundaries.West.value)
        gmsh.model.setPhysicalName(2, boundaries.West.value, "West")

        gmsh.model.addPhysicalGroup(2, [3], boundaries.South.value)
        gmsh.model.setPhysicalName(2, boundaries.South.value, "South")

        gmsh.model.addPhysicalGroup(2, [5], boundaries.East.value)
        gmsh.model.setPhysicalName(2, boundaries.East.value, "East")

        gmsh.model.addPhysicalGroup(3, [1, 2], 99999)
        gmsh.model.setPhysicalName(3, 99999, "Elements")

        ## We need to know which surface !!
        # for _, line in gmsh.model.get_entities(1):
        #     gmsh.model.mesh.setTransfiniteCurve(line, numNodes=numElements + 1)
        #
        #
        #

        gmsh.model.mesh.setTransfiniteCurve(1, numNodes=numElementsLon + 1)
        gmsh.model.mesh.setTransfiniteCurve(2, numNodes=numElementsLat + 1)
        gmsh.model.mesh.setTransfiniteCurve(3, numNodes=numElementsLon + 1)
        gmsh.model.mesh.setTransfiniteCurve(4, numNodes=numElementsLat + 1)

        gmsh.model.mesh.setTransfiniteCurve(5, numNodes=numElementsLon + 1)
        gmsh.model.mesh.setTransfiniteCurve(6, numNodes=numElementsLat + 1)
        gmsh.model.mesh.setTransfiniteCurve(7, numNodes=numElementsLon + 1)
        gmsh.model.mesh.setTransfiniteCurve(8, numNodes=numElementsLat + 1)

        gmsh.model.mesh.setTransfiniteCurve(9, numNodes=numElementsDepth + 1)
        gmsh.model.mesh.setTransfiniteCurve(10, numNodes=numElementsDepth + 1)
        gmsh.model.mesh.setTransfiniteCurve(11, numNodes=numElementsDepth + 1)
        gmsh.model.mesh.setTransfiniteCurve(12, numNodes=numElementsDepth + 1)

        for _, surface in gmsh.model.get_entities(2):
            gmsh.model.mesh.setTransfiniteSurface(surface)
            if not simplex:
                gmsh.model.mesh.set_recombine(2, surface)

        if not simplex:
            for _, volume in gmsh.model.get_entities(3):
                gmsh.model.mesh.set_transfinite_volume(volume)
                # if not simplex:
                gmsh.model.mesh.set_recombine(3, volume)

        # Generate Mesh
        gmsh.model.mesh.generate(3)
        gmsh.write(uw_filename)
        gmsh.finalize()

    ## This needs a side-boundary capture routine as well

    def sphere_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        ## Note these numbers should not be hard-wired

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        coarsening=coarsening,
        coarsening_callback=spherical_mesh_refinement_callback,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        return_coords_to_bounds=sphere_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = sympy.UnevaluatedExpr(
            new_mesh.CoordinateSystem.unit_e_0
        ) * sympy.UnevaluatedExpr(
            sympy.Piecewise(
                (1.0, new_mesh.CoordinateSystem.R[0] < 1.01 * radiusInner), (0.0, True)
            )
        )
        Upper = sympy.UnevaluatedExpr(
            new_mesh.CoordinateSystem.unit_e_0
        ) * sympy.UnevaluatedExpr(
            sympy.Piecewise(
                (1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radiusOuter), (0.0, True)
            )
        )

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


# ToDo: if keeping, we need to add boundaries etc.


@timing.routine_timer_decorator
def SegmentedSphericalSurface2D(
    radius: float = 1.0,
    cellSize: float = 0.05,
    numSegments: int = 6,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    gmsh_verbosity=0,
    verbose=False,
):
    num_segments = numSegments
    meshRes = cellSize

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = f".meshes/uw_segmented_spherical_surface_r{radius}_csize{cellSize}_segs{num_segments}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        options = PETSc.Options()
        options["dm_plex_gmsh_multiple_tags"] = None
        options["dm_plex_gmsh_spacedim"] = 2
        options["dm_plex_gmsh_use_regions"] = None
        options["dm_plex_gmsh_mark_vertices"] = None

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Segmented Sphere 2D Surface")

        # Mesh like an orange

        surflist = []
        longitudesN = []
        longitudesS = []
        segments_clps = []
        segments_surfs = []
        equator_pts = []

        centre = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=-1)
        poleN = gmsh.model.geo.addPoint(0.0, 0.0, 1.0, tag=-1, meshSize=0.5 * meshRes)
        poleS = gmsh.model.geo.addPoint(0.0, 0.0, -1.0, tag=-1, meshSize=0.5 * meshRes)

        dtheta = 2 * np.pi / num_segments

        for i in range(num_segments):
            theta = i * 2 * np.pi / num_segments
            x1 = np.cos(theta)
            y1 = np.sin(theta)
            equator_pts.append(
                gmsh.model.geo.addPoint(x1, y1, 0.0, tag=-1, meshSize=meshRes)
            )

        for i in range(num_segments):
            pEq = equator_pts[i]
            longitudesN.append(gmsh.model.geo.addCircleArc(poleN, centre, pEq, tag=-1))
            longitudesS.append(gmsh.model.geo.addCircleArc(pEq, centre, poleS, tag=-1))

        gmsh.model.geo.synchronize()

        # Curve loops:

        for i in range(num_segments):
            loops = [
                longitudesN[i],
                longitudesS[i],
                longitudesS[np.mod(i + 1, num_segments)],
                longitudesN[np.mod(i + 1, num_segments)],
            ]
            segments_clps.append(
                gmsh.model.geo.addCurveLoop(loops[::-1], tag=-1, reorient=True)
            )

        gmsh.model.geo.synchronize()

        # Surfaces

        for i in range(num_segments):
            segments_surfs.append(
                gmsh.model.geo.addSurfaceFilling(
                    [segments_clps[i]], tag=-1, sphereCenterTag=centre
                )
            )

        gmsh.model.geo.synchronize()

        # Add some physical labels etc.

        gmsh.model.addPhysicalGroup(0, [poleN], 1000)
        gmsh.model.addPhysicalGroup(0, [poleS], 2000)
        gmsh.model.addPhysicalGroup(0, [poleN, poleS], 3000)
        gmsh.model.setPhysicalName(0, 1000, "NPole")
        gmsh.model.setPhysicalName(0, 2000, "SPole")
        gmsh.model.setPhysicalName(0, 3000, "Poles")

        gmsh.model.addPhysicalGroup(2, segments_surfs, 10000)
        gmsh.model.setPhysicalName(2, 10000, "Elements")

        gmsh.model.mesh.remove_duplicate_nodes()
        gmsh.model.remove_entities([(0, centre)])

        # Generate Mesh
        gmsh.model.mesh.generate(2)
        gmsh.write(uw_filename)

        # xyz coordinates of the mesh
        xyz = gmsh.model.mesh.get_nodes()[1].reshape(-1, 3)
        gmsh.write(uw_filename)
        gmsh.finalize()

        plex_0 = gmsh2dmplex(
            uw_filename,
            useMultipleTags=True,
            useRegions=True,
            markVertices=True,
            comm=PETSc.COMM_SELF,
        )

        # Re-interpret the DM coordinates
        lonlat_vec = plex_0[1].getCoordinates()
        lonlat = np.empty_like(xyz[:, 0:2])
        lonlat[:, 0] = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi) - np.pi
        lonlat[:, 1] = np.arcsin(xyz[:, 2])
        lonlat_vec.array[...] = lonlat.reshape(-1)
        plex_0[1].setCoordinates(lonlat_vec)

        # Does this get saved by the viewer ?
        uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
            plex_0[1], [np.pi, 0.0], [-np.pi, 0.0], [np.pi * 2, 0.0]
        )

        viewer = PETSc.ViewerHDF5().create(
            uw_filename + ".h5", "w", comm=PETSc.COMM_SELF
        )
        viewer(plex_0[1])

    # Now do this collectively

    # TODO: add callbacks for refinement and out-of-box returns
    new_mesh = Mesh(
        uw_filename + ".h5",
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERE_SURFACE_NATIVE,
        verbose=verbose,
    )

    #### May have been causing the script to hang - BK

    # # This may not be needed
    # uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    #     new_mesh.dm, [np.pi, 0.0], [-np.pi, 0.0], [np.pi * 2, 0.0]
    # )

    return new_mesh


@timing.routine_timer_decorator
def SegmentedSphericalShell(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    cellSize: float = 0.1,
    numSegments: int = 6,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    coordinatesNative=False,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Lower = 20
        LowerPlus = 21
        Upper = 30
        UpperPlus = 31
        Centre = 1
        Slices = 40

    meshRes = cellSize
    num_segments = numSegments

    if coordinatesNative == True:
        coordinate_system = CoordinateSystemType.SPHERICAL_NATIVE
    else:
        coordinate_system = CoordinateSystemType.SPHERICAL

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = f".meshes/uw_segmented_sphere_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}_segs{num_segments}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        options = PETSc.Options()
        options["dm_plex_gmsh_multiple_tags"] = None
        options["dm_plex_gmsh_use_regions"] = None
        options["dm_plex_gmsh_mark_vertices"] = None

        ## Follow the lead of the cubed sphere and make copies of a segment

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.add("Segmented Sphere 3D")

        centre = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=-1)

        poleNo = gmsh.model.geo.addPoint(
            0.0, 0.0, radiusOuter, tag=-1, meshSize=meshRes
        )
        poleSo = gmsh.model.geo.addPoint(
            0.0, 0.0, -radiusOuter, tag=-1, meshSize=meshRes
        )
        poleNi = gmsh.model.geo.addPoint(
            0.0, 0.0, radiusInner, tag=-1, meshSize=meshRes
        )
        poleSi = gmsh.model.geo.addPoint(
            0.0, 0.0, -radiusInner, tag=-1, meshSize=meshRes
        )

        dtheta = 2 * np.pi / num_segments

        r = radiusOuter
        equator_pts_0o = gmsh.model.geo.addPoint(r, 0.0, 0.0, tag=-1, meshSize=meshRes)
        equator_pts_1o = gmsh.model.geo.addPoint(
            r * np.cos(dtheta), r * np.sin(dtheta), 0.0, tag=-1, meshSize=meshRes
        )
        r = radiusInner
        equator_pts_0i = gmsh.model.geo.addPoint(r, 0.0, 0.0, tag=-1, meshSize=meshRes)
        equator_pts_1i = gmsh.model.geo.addPoint(
            r * np.cos(dtheta), r * np.sin(dtheta), 0.0, tag=-1, meshSize=meshRes
        )

        gmsh.model.geo.synchronize()

        # Make edges

        edgeWo = gmsh.model.geo.addCircleArc(poleNo, centre, equator_pts_0o, tag=-1)
        edgeEqo = gmsh.model.geo.addCircleArc(
            equator_pts_0o, centre, equator_pts_1o, tag=-1
        )
        edgeEo = gmsh.model.geo.addCircleArc(equator_pts_1o, centre, poleNo, tag=-1)

        edgeWi = gmsh.model.geo.addCircleArc(poleNi, centre, equator_pts_0i, tag=-1)
        edgeEqi = gmsh.model.geo.addCircleArc(
            equator_pts_0i, centre, equator_pts_1i, tag=-1
        )
        edgeEi = gmsh.model.geo.addCircleArc(equator_pts_1i, centre, poleNi, tag=-1)

        ## Struts

        radialW = gmsh.model.geo.addLine(equator_pts_0o, equator_pts_0i, tag=-1)
        radialE = gmsh.model.geo.addLine(equator_pts_1o, equator_pts_1i, tag=-1)
        radialN = gmsh.model.geo.addLine(poleNo, poleNi, tag=-1)

        # Make boundaries

        faceLoopo = gmsh.model.geo.addCurveLoop(
            [edgeWo, edgeEqo, edgeEo], tag=-1, reorient=True
        )
        faceLoopi = gmsh.model.geo.addCurveLoop(
            [edgeWi, edgeEqi, edgeEi], tag=-1, reorient=True
        )
        faceLoopW = gmsh.model.geo.addCurveLoop(
            [edgeWo, radialW, edgeWi, radialN], tag=-1, reorient=True
        )
        faceLoopE = gmsh.model.geo.addCurveLoop(
            [edgeEo, radialE, edgeEi, radialN], tag=-1, reorient=True
        )
        faceLoopS = gmsh.model.geo.addCurveLoop(
            [edgeEqo, radialW, edgeEqi, radialE], tag=-1, reorient=True
        )

        # Make surfaces

        face_o = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopo,
            ],
            tag=-1,
            sphereCenterTag=centre,
        )
        face_i = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopi,
            ],
            tag=-1,
            sphereCenterTag=centre,
        )
        face_W = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopW,
            ],
            tag=-1,
        )
        face_E = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopE,
            ],
            tag=-1,
        )
        face_S = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopS,
            ],
            tag=-1,
        )

        outer_faces = [face_o]
        inner_faces = [face_i]
        wedge_slices = [face_E, face_S, face_W]

        # Make volume

        wedge_surf = gmsh.model.geo.addSurfaceLoop(
            [face_o, face_i, face_W, face_E, face_S], tag=-1
        )

        wedge_vol = gmsh.model.geo.addVolume([wedge_surf], tag=-1)
        wedges = [wedge_vol]

        gmsh.model.geo.synchronize()

        # Make copies

        for i in range(1, num_segments):
            new_wedge = gmsh.model.geo.copy([(3, 1)])
            gmsh.model.geo.rotate(new_wedge, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, i * dtheta)

            gmsh.model.geo.synchronize()

            _, new_faces = gmsh.model.get_adjacencies(3, new_wedge[0][1])

            wedges.append(new_wedge[0][1])
            outer_faces.append(new_faces[0])
            inner_faces.append(new_faces[1])
            wedge_slices.append(new_faces[2])
            wedge_slices.append(new_faces[3])
            wedge_slices.append(new_faces[4])

        mirror_wedge = gmsh.model.geo.copy([(3, 1)])
        gmsh.model.geo.mirror(mirror_wedge, 0.0, 0.0, 1.0, 0.0)

        gmsh.model.geo.synchronize()
        _, mirror_faces = gmsh.model.get_adjacencies(3, mirror_wedge[0][1])
        wedges.append(mirror_wedge[0][1])
        outer_faces.append(mirror_faces[0])
        inner_faces.append(mirror_faces[1])
        wedge_slices.append(mirror_faces[2])
        wedge_slices.append(mirror_faces[3])
        wedge_slices.append(mirror_faces[4])

        _, mirror_edges_w = gmsh.model.get_adjacencies(2, mirror_faces[2])
        _, mirror_edges_e = gmsh.model.get_adjacencies(2, mirror_faces[3])

        radialS = tuple(set(mirror_edges_e).intersection(set(mirror_edges_w)))[0]

        for i in range(1, num_segments):
            new_wedge = gmsh.model.geo.copy(mirror_wedge)
            gmsh.model.geo.rotate(new_wedge, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, i * dtheta)

            gmsh.model.geo.synchronize()

            _, new_faces = gmsh.model.get_adjacencies(3, new_wedge[0][1])

            wedges.append(new_wedge[0][1])
            outer_faces.append(new_faces[0])
            inner_faces.append(new_faces[1])
            wedge_slices.append(new_faces[2])
            wedge_slices.append(new_faces[3])
            wedge_slices.append(new_faces[4])

        gmsh.model.addPhysicalGroup(0, [poleNo], 1)
        gmsh.model.setPhysicalName(0, 1, "PolePtNo")

        gmsh.model.addPhysicalGroup(0, [poleNi], 2)
        gmsh.model.setPhysicalName(0, 2, "PolePtNi")

        gmsh.model.addPhysicalGroup(0, [poleSo], 3)
        gmsh.model.setPhysicalName(0, 3, "PolePtSo")

        gmsh.model.addPhysicalGroup(0, [poleSi], 4)
        gmsh.model.setPhysicalName(0, 4, "PolePtSi")

        gmsh.model.addPhysicalGroup(1, [radialN], 10)
        gmsh.model.setPhysicalName(1, 10, "PoleAxisN")

        gmsh.model.addPhysicalGroup(1, [radialS], 11)
        gmsh.model.setPhysicalName(1, 11, "PoleAxisS")

        gmsh.model.addPhysicalGroup(2, outer_faces, boundaries.Upper.value)
        gmsh.model.setPhysicalName(2, boundaries.Upper.value, boundaries.Upper.name)

        gmsh.model.addPhysicalGroup(2, inner_faces, boundaries.Lower.value)
        gmsh.model.setPhysicalName(2, boundaries.Lower.value, boundaries.Lower.name)

        gmsh.model.addPhysicalGroup(2, wedge_slices, boundaries.Slices.value)
        gmsh.model.setPhysicalName(2, boundaries.Slices.value, boundaries.Slices.name)

        gmsh.model.addPhysicalGroup(3, wedges, 30)
        gmsh.model.setPhysicalName(3, 30, "Elements")

        gmsh.model.remove_entities([(0, centre)])

        gmsh.model.mesh.generate(3)

        gmsh.write(uw_filename)
        gmsh.finalize()

        # We need to build the plex here in order to make some changes
        # before the mesh gets built
        plex_0 = gmsh2dmplex(
            uw_filename,
            useMultipleTags=True,
            useRegions=True,
            markVertices=True,
            comm=PETSc.COMM_SELF,
        )

        if coordinatesNative:
            xyz_vec = plex_0[1].getCoordinates()
            xyz = xyz_vec.array.reshape(-1, 3)

            rthph = np.empty_like(xyz)
            rthph[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
            rthph[:, 1] = np.arccos((xyz[:, 2]) / (rthph[:, 0] + 1.0e-6))
            rthph[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0] + 1.0e-6) - np.pi

            rthph_vec = xyz_vec.copy()
            rthph_vec.array[...] = rthph.reshape(-1)[...]
            plex_0[1].setCoordinates(rthph_vec)

            uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
                plex_0[1], [0.0, 0.0, np.pi], [0.0, 0.0, -np.pi], [0.0, 0.0, np.pi * 2]
            )

        # Composite label - upper + wedge slices

        ul = plex_0[1].getLabel(boundaries.Upper.name)
        sl = plex_0[1].getLabel(boundaries.Slices.name)

        ul_is = ul.getStratumIS(boundaries.Upper.value)
        sl_is = sl.getStratumIS(boundaries.Slices.value)

        new_is = ul_is.union(sl_is)

        plex_0[1].createLabel(boundaries.UpperPlus.name)
        both_lab = plex_0[1].getLabel(boundaries.UpperPlus.name)
        both_lab.setStratumIS(boundaries.UpperPlus.value, new_is)

        # Composite label - lower + wedge slices  (Combine with above to eliminate sl)

        ll = plex_0[1].getLabel(boundaries.Lower.name)
        sl = plex_0[1].getLabel(boundaries.Slices.name)

        ll_is = ll.getStratumIS(boundaries.Lower.value)
        sl_is = sl.getStratumIS(boundaries.Slices.value)

        new_is = ll_is.union(sl_is)

        plex_0[1].createLabel(boundaries.LowerPlus.name)
        both_lab = plex_0[1].getLabel(boundaries.LowerPlus.name)
        both_lab.setStratumIS(boundaries.LowerPlus.value, new_is)

        ####

        viewer = PETSc.ViewerHDF5().create(
            uw_filename + ".h5", "w", comm=PETSc.COMM_SELF
        )

        viewer(plex_0[1])

    ## Are these needed for native coordinates ?
    def sphere_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        ## Note these numbers should not be hard-wired

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename + ".h5",
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        coordinate_system_type=coordinate_system,
        return_coords_to_bounds=sphere_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = sympy.UnevaluatedExpr(
            new_mesh.CoordinateSystem.unit_e_0
        ) * sympy.UnevaluatedExpr(
            sympy.Piecewise(
                (1.0, new_mesh.CoordinateSystem.R[0] < 1.01 * radiusInner), (0.0, True)
            )
        )
        Upper = sympy.UnevaluatedExpr(
            new_mesh.CoordinateSystem.unit_e_0
        ) * sympy.UnevaluatedExpr(
            sympy.Piecewise(
                (1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radiusOuter), (0.0, True)
            )
        )
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def SegmentedSphericalBall(
    radius: float = 1.0,
    cellSize: float = 0.1,
    numSegments: int = 6,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    coordinatesNative=False,
    verbosity=0,
    gmsh_verbosity=0,
    verbose=False,
):
    class boundaries(Enum):
        Upper = 30
        UpperPlus = 31
        Centre = 1
        Slices = 40
        Null_Boundary = 666

    meshRes = cellSize
    num_segments = numSegments

    if coordinatesNative == True:
        coordinate_system = CoordinateSystemType.SPHERICAL_NATIVE
    else:
        coordinate_system = CoordinateSystemType.SPHERICAL

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = f".meshes/uw_segmented_ball_ro{radius}_csize{cellSize}_segs{num_segments}.msh"
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

        options = PETSc.Options()
        options["dm_plex_gmsh_multiple_tags"] = None
        options["dm_plex_gmsh_use_regions"] = None
        options["dm_plex_gmsh_mark_vertices"] = None

        ## Follow the lead of the cubed sphere and make copies of a segment

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.add("Segmented Sphere 3D")

        centre = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=-1, meshSize=meshRes)

        poleNo = gmsh.model.geo.addPoint(0.0, 0.0, radius, tag=-1, meshSize=meshRes)
        poleSo = gmsh.model.geo.addPoint(0.0, 0.0, -radius, tag=-1, meshSize=meshRes)

        dtheta = 2 * np.pi / num_segments

        r = radius
        equator_pts_0o = gmsh.model.geo.addPoint(r, 0.0, 0.0, tag=-1, meshSize=meshRes)
        equator_pts_1o = gmsh.model.geo.addPoint(
            r * np.cos(dtheta), r * np.sin(dtheta), 0.0, tag=-1, meshSize=meshRes
        )

        gmsh.model.geo.synchronize()

        # Make edges

        edgeWo = gmsh.model.geo.addCircleArc(poleNo, centre, equator_pts_0o, tag=-1)
        edgeEqo = gmsh.model.geo.addCircleArc(
            equator_pts_0o, centre, equator_pts_1o, tag=-1
        )
        edgeEo = gmsh.model.geo.addCircleArc(equator_pts_1o, centre, poleNo, tag=-1)

        ## Struts

        radialW = gmsh.model.geo.addLine(equator_pts_0o, centre, tag=-1)
        radialE = gmsh.model.geo.addLine(equator_pts_1o, centre, tag=-1)
        radialN = gmsh.model.geo.addLine(poleNo, centre, tag=-1)

        # Make boundaries

        faceLoopo = gmsh.model.geo.addCurveLoop(
            [edgeWo, edgeEqo, edgeEo], tag=-1, reorient=True
        )
        # faceLoopi = gmsh.model.geo.addCurveLoop(
        #     [edgeWi, edgeEqi, edgeEi], tag=-1, reorient=True
        # )
        faceLoopW = gmsh.model.geo.addCurveLoop(
            [edgeWo, radialW, radialN], tag=-1, reorient=True
        )
        faceLoopE = gmsh.model.geo.addCurveLoop(
            [edgeEo, radialE, radialN], tag=-1, reorient=True
        )
        faceLoopS = gmsh.model.geo.addCurveLoop(
            [edgeEqo, radialW, radialE], tag=-1, reorient=True
        )

        # Make surfaces

        face_o = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopo,
            ],
            tag=-1,
            sphereCenterTag=centre,
        )
        # face_i = gmsh.model.geo.addSurfaceFilling(
        #     [
        #         faceLoopi,
        #     ],
        #     tag=-1,
        #     sphereCenterTag=centre,
        # )
        face_W = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopW,
            ],
            tag=-1,
        )
        face_E = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopE,
            ],
            tag=-1,
        )
        face_S = gmsh.model.geo.addSurfaceFilling(
            [
                faceLoopS,
            ],
            tag=-1,
        )

        outer_faces = [face_o]
        wedge_slices = [face_E, face_S, face_W]

        # Make volume

        wedge_surf = gmsh.model.geo.addSurfaceLoop(
            [face_o, face_W, face_E, face_S], tag=-1
        )

        wedge_vol = gmsh.model.geo.addVolume([wedge_surf], tag=-1)
        wedges = [wedge_vol]

        gmsh.model.geo.synchronize()

        # Make copies

        for i in range(1, num_segments):
            new_wedge = gmsh.model.geo.copy([(3, 1)])
            gmsh.model.geo.rotate(new_wedge, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, i * dtheta)

            gmsh.model.geo.synchronize()

            _, new_faces = gmsh.model.get_adjacencies(3, new_wedge[0][1])

            wedges.append(new_wedge[0][1])
            outer_faces.append(new_faces[0])
            wedge_slices.append(new_faces[1])
            wedge_slices.append(new_faces[2])
            wedge_slices.append(new_faces[3])

        mirror_wedge = gmsh.model.geo.copy([(3, 1)])
        gmsh.model.geo.mirror(mirror_wedge, 0.0, 0.0, 1.0, 0.0)

        gmsh.model.geo.synchronize()
        _, mirror_faces = gmsh.model.get_adjacencies(3, mirror_wedge[0][1])
        wedges.append(mirror_wedge[0][1])
        outer_faces.append(mirror_faces[0])
        wedge_slices.append(mirror_faces[1])
        wedge_slices.append(mirror_faces[2])
        wedge_slices.append(mirror_faces[3])

        _, mirror_edges_w = gmsh.model.get_adjacencies(2, mirror_faces[2])
        _, mirror_edges_e = gmsh.model.get_adjacencies(2, mirror_faces[3])

        radialS = tuple(set(mirror_edges_e).intersection(set(mirror_edges_w)))[0]

        for i in range(1, num_segments):
            new_wedge = gmsh.model.geo.copy(mirror_wedge)
            gmsh.model.geo.rotate(new_wedge, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, i * dtheta)

            gmsh.model.geo.synchronize()

            _, new_faces = gmsh.model.get_adjacencies(3, new_wedge[0][1])

            wedges.append(new_wedge[0][1])
            outer_faces.append(new_faces[0])
            wedge_slices.append(new_faces[1])
            wedge_slices.append(new_faces[2])
            wedge_slices.append(new_faces[3])

        gmsh.model.addPhysicalGroup(0, [poleNo], 1)
        gmsh.model.setPhysicalName(0, 1, "PolePtNo")

        gmsh.model.addPhysicalGroup(0, [poleSo], 3)
        gmsh.model.setPhysicalName(0, 3, "PolePtSo")

        gmsh.model.addPhysicalGroup(1, [radialN], 10)
        gmsh.model.setPhysicalName(1, 10, "PoleAxisN")

        gmsh.model.addPhysicalGroup(1, [radialS], 11)
        gmsh.model.setPhysicalName(1, 11, "PoleAxisS")

        gmsh.model.addPhysicalGroup(2, outer_faces, boundaries.Upper.value)
        gmsh.model.setPhysicalName(2, boundaries.Upper.value, boundaries.Upper.name)

        gmsh.model.addPhysicalGroup(2, wedge_slices, boundaries.Slices.value)
        gmsh.model.setPhysicalName(2, boundaries.Slices.value, boundaries.Slices.name)

        gmsh.model.addPhysicalGroup(3, wedges, 30)
        gmsh.model.setPhysicalName(3, 30, "Elements")

        # gmsh.model.remove_entities([(0, centre)])

        gmsh.model.mesh.generate(3)

        gmsh.write(uw_filename)
        gmsh.finalize()

        # We need to build the plex here in order to make some changes
        # before the mesh gets built
        plex_0 = gmsh2dmplex(
            uw_filename,
            useMultipleTags=True,
            useRegions=True,
            markVertices=True,
            comm=PETSc.COMM_SELF,
        )

        if coordinatesNative:
            xyz_vec = plex_0[1].getCoordinates()
            xyz = xyz_vec.array.reshape(-1, 3)

            rthph = np.empty_like(xyz)
            rthph[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
            rthph[:, 1] = np.arccos((xyz[:, 2]) / (rthph[:, 0] + 1.0e-6))
            rthph[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0] + 1.0e-6) - np.pi

            rthph_vec = xyz_vec.copy()
            rthph_vec.array[...] = rthph.reshape(-1)[...]
            plex_0[1].setCoordinates(rthph_vec)

            uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
                plex_0[1], [0.0, 0.0, np.pi], [0.0, 0.0, -np.pi], [0.0, 0.0, np.pi * 2]
            )

        # Composite label - upper + wedge slices

        ul = plex_0[1].getLabel(boundaries.Upper.name)
        sl = plex_0[1].getLabel(boundaries.Slices.name)

        ul_is = ul.getStratumIS(boundaries.Upper.value)
        sl_is = sl.getStratumIS(boundaries.Slices.value)

        new_is = ul_is.union(sl_is)

        plex_0[1].createLabel(boundaries.UpperPlus.name)
        both_lab = plex_0[1].getLabel(boundaries.UpperPlus.name)
        both_lab.setStratumIS(boundaries.UpperPlus.value, new_is)

        # # Composite label - lower + wedge slices  (Combine with above to eliminate sl)

        # ll = plex_0[1].getLabel(boundaries.Lower.name)
        # sl = plex_0[1].getLabel(boundaries.Slices.name)

        # ll_is = ll.getStratumIS(boundaries.Lower.value)
        # sl_is = sl.getStratumIS(boundaries.Slices.value)

        # new_is = ll_is.union(sl_is)

        # plex_0[1].createLabel(boundaries.LowerPlus.name)
        # both_lab = plex_0[1].getLabel(boundaries.LowerPlus.name)
        # both_lab.setStratumIS(boundaries.LowerPlus.value, new_is)

        ####

        viewer = PETSc.ViewerHDF5().create(
            uw_filename + ".h5", "w", comm=PETSc.COMM_SELF
        )

        viewer(plex_0[1])

    ## Are these needed for native coordinates ?
    def sphere_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2

        outside = Rsq > radiusOuter**2
        inside = Rsq < radiusInner**2

        ## Note these numbers should not be hard-wired

        coords[outside, :] *= 0.99 * radiusOuter / np.sqrt(Rsq[outside].reshape(-1, 1))
        coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))

        return coords

    def spherical_mesh_refinement_callback(dm):
        r_o = radiusOuter
        r_i = radiusInner

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Upper"
            )
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = (
            uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
                dm, "Lower"
            )
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
        # print(f"Refinement callback - Lower {len(lowerIndices)}", flush=True)

        c2.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c2)

        return

    new_mesh = Mesh(
        uw_filename + ".h5",
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_mesh_refinement_callback,
        coordinate_system_type=coordinate_system,
        return_coords_to_bounds=sphere_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Upper = sympy.UnevaluatedExpr(
            new_mesh.CoordinateSystem.unit_e_0
        ) * sympy.UnevaluatedExpr(
            sympy.Piecewise(
                (1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radius), (0.0, True)
            )
        )
        Centre = None

    new_mesh.boundary_normals = boundary_normals

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
