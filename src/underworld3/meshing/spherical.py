"""
Spherical mesh generation functions for Underworld3.

This module contains mesh generation functions for spherical shells and
3D spherical geometries.
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
