"""
Segmented mesh generation functions for Underworld3.

This module contains mesh generation functions for multi-segment and
complex spherical geometries.
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


# Functions moved from meshing_legacy.py:


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
            equator_pts.append(gmsh.model.geo.addPoint(x1, y1, 0.0, tag=-1, meshSize=meshRes))

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
            segments_clps.append(gmsh.model.geo.addCurveLoop(loops[::-1], tag=-1, reorient=True))

        gmsh.model.geo.synchronize()

        # Surfaces

        for i in range(num_segments):
            segments_surfs.append(
                gmsh.model.geo.addSurfaceFilling([segments_clps[i]], tag=-1, sphereCenterTag=centre)
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

        viewer = PETSc.ViewerHDF5().create(uw_filename + ".h5", "w", comm=PETSc.COMM_SELF)
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

        poleNo = gmsh.model.geo.addPoint(0.0, 0.0, radiusOuter, tag=-1, meshSize=meshRes)
        poleSo = gmsh.model.geo.addPoint(0.0, 0.0, -radiusOuter, tag=-1, meshSize=meshRes)
        poleNi = gmsh.model.geo.addPoint(0.0, 0.0, radiusInner, tag=-1, meshSize=meshRes)
        poleSi = gmsh.model.geo.addPoint(0.0, 0.0, -radiusInner, tag=-1, meshSize=meshRes)

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
        edgeEqo = gmsh.model.geo.addCircleArc(equator_pts_0o, centre, equator_pts_1o, tag=-1)
        edgeEo = gmsh.model.geo.addCircleArc(equator_pts_1o, centre, poleNo, tag=-1)

        edgeWi = gmsh.model.geo.addCircleArc(poleNi, centre, equator_pts_0i, tag=-1)
        edgeEqi = gmsh.model.geo.addCircleArc(equator_pts_0i, centre, equator_pts_1i, tag=-1)
        edgeEi = gmsh.model.geo.addCircleArc(equator_pts_1i, centre, poleNi, tag=-1)

        ## Struts

        radialW = gmsh.model.geo.addLine(equator_pts_0o, equator_pts_0i, tag=-1)
        radialE = gmsh.model.geo.addLine(equator_pts_1o, equator_pts_1i, tag=-1)
        radialN = gmsh.model.geo.addLine(poleNo, poleNi, tag=-1)

        # Make boundaries

        faceLoopo = gmsh.model.geo.addCurveLoop([edgeWo, edgeEqo, edgeEo], tag=-1, reorient=True)
        faceLoopi = gmsh.model.geo.addCurveLoop([edgeWi, edgeEqi, edgeEi], tag=-1, reorient=True)
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

        wedge_surf = gmsh.model.geo.addSurfaceLoop([face_o, face_i, face_W, face_E, face_S], tag=-1)

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

        viewer = PETSc.ViewerHDF5().create(uw_filename + ".h5", "w", comm=PETSc.COMM_SELF)

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

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Lower"
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
        Lower = sympy.UnevaluatedExpr(new_mesh.CoordinateSystem.unit_e_0) * sympy.UnevaluatedExpr(
            sympy.Piecewise((1.0, new_mesh.CoordinateSystem.R[0] < 1.01 * radiusInner), (0.0, True))
        )
        Upper = sympy.UnevaluatedExpr(new_mesh.CoordinateSystem.unit_e_0) * sympy.UnevaluatedExpr(
            sympy.Piecewise((1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radiusOuter), (0.0, True))
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
        edgeEqo = gmsh.model.geo.addCircleArc(equator_pts_0o, centre, equator_pts_1o, tag=-1)
        edgeEo = gmsh.model.geo.addCircleArc(equator_pts_1o, centre, poleNo, tag=-1)

        ## Struts

        radialW = gmsh.model.geo.addLine(equator_pts_0o, centre, tag=-1)
        radialE = gmsh.model.geo.addLine(equator_pts_1o, centre, tag=-1)
        radialN = gmsh.model.geo.addLine(poleNo, centre, tag=-1)

        # Make boundaries

        faceLoopo = gmsh.model.geo.addCurveLoop([edgeWo, edgeEqo, edgeEo], tag=-1, reorient=True)
        # faceLoopi = gmsh.model.geo.addCurveLoop(
        #     [edgeWi, edgeEqi, edgeEi], tag=-1, reorient=True
        # )
        faceLoopW = gmsh.model.geo.addCurveLoop([edgeWo, radialW, radialN], tag=-1, reorient=True)
        faceLoopE = gmsh.model.geo.addCurveLoop([edgeEo, radialE, radialN], tag=-1, reorient=True)
        faceLoopS = gmsh.model.geo.addCurveLoop([edgeEqo, radialW, radialE], tag=-1, reorient=True)

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

        wedge_surf = gmsh.model.geo.addSurfaceLoop([face_o, face_W, face_E, face_S], tag=-1)

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

        viewer = PETSc.ViewerHDF5().create(uw_filename + ".h5", "w", comm=PETSc.COMM_SELF)

        viewer(plex_0[1])

    ## Are these needed for native coordinates ?
    def sphere_return_coords_to_bounds(coords):
        Rsq = coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2

        outside = Rsq > radius**2
        # inside = Rsq < radiusInner**2  # Not applicable for solid ball

        ## Note these numbers should not be hard-wired

        coords[outside, :] *= 0.99 * radius / np.sqrt(Rsq[outside].reshape(-1, 1))
        # coords[inside, :] *= 1.01 * radiusInner / np.sqrt(Rsq[inside].reshape(-1, 1))  # Not applicable

        return coords

    def spherical_mesh_refinement_callback(dm):
        r_o = radius
        # r_i = radiusInner  # Not applicable for solid ball

        import underworld3 as uw

        # print(f"Refinement callback - spherical", flush=True)

        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 3)
        R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
            dm, "Upper"
        )
        coords[upperIndices] *= r_o / R[upperIndices].reshape(-1, 1)
        # print(f"Refinement callback - Upper {len(upperIndices)}", flush=True)

        # lowerIndices = (  # Not applicable for solid ball
        #     uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        #         dm, "Lower"
        #     )
        # )

        # coords[lowerIndices] *= r_i / (1.0e-16 + R[lowerIndices].reshape(-1, 1))
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
        Upper = sympy.UnevaluatedExpr(new_mesh.CoordinateSystem.unit_e_0) * sympy.UnevaluatedExpr(
            sympy.Piecewise((1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radius), (0.0, True))
        )
        Centre = None

    new_mesh.boundary_normals = boundary_normals

    return new_mesh
