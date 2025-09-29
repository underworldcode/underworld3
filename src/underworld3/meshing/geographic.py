"""
Geographic mesh generation functions for Underworld3.

This module contains mesh generation functions for geographic/geodetic coordinate systems.
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