"""
Annulus mesh generation functions for Underworld3.

This module contains mesh generation functions for 2D cylindrical/annular geometries.
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

        uw_filename = f"uw_QuarterAnnulus_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}.msh"
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

        gmsh.model.addPhysicalGroup(1, [l1], boundaries.Left.value, name=boundaries.Left.name)

        gmsh.model.addPhysicalGroup(1, [l3], boundaries.Right.value, name=boundaries.Right.name)

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

        uw_filename = f".meshes/uw_annulus_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}.msh"
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
            p3 = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0, meshSize=cellSizeInner)

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

        gmsh.model.addPhysicalGroup(1, [c3, c4], boundaries.Upper.value, name=boundaries.Upper.name)
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

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Lower"
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
        gmsh.model.addPhysicalGroup(1, [l_left], boundaries.Left.value, name=boundaries.Left.name)
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

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Lower"
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

        viewer = PETSc.ViewerHDF5().create(uw_filename + ".h5", "w", comm=PETSc.COMM_SELF)

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

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Lower"
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
            p2 = gmsh.model.geo.add_point(radiusInner, 0.0, 0.0, meshSize=cellSize_Inner)
            p3 = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0, meshSize=cellSize_Inner)

            c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
            c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

            cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries.Lower.value)

            loops = [cl1] + loops

        p4 = gmsh.model.geo.add_point(radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal)
        p5 = gmsh.model.geo.add_point(-radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal)

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

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Lower"
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        internalIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Internal"
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

        p4 = gmsh.model.geo.add_point(radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal)
        p5 = gmsh.model.geo.add_point(-radiusInternal, 0.0, 0.0, meshSize=cellSize_Internal)

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

        gmsh.model.addPhysicalGroup(1, [c1, c2], boundaries.Lower.value, name=boundaries.Lower.name)

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

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)

        lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Lower"
        )

        coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))

        internalIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Internal"
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
