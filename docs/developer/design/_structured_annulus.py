"""Structured-quadrilateral annulus with an internal boundary.

Builds the same geometry as `uw.meshing.AnnulusInternalBoundary`
(three concentric arcs at r_inner, r_internal, r_outer plus the
two annular regions between them) but with a transfinite-quad
mesh: cells form a regular polar grid with quad elements aligned
to (r, θ).

We split each radial layer into two half-annuli (θ ∈ [0, π] and
θ ∈ [-π, 0]) so each surface has the four corners needed by
gmsh's transfinite surface. The radial node count and the
angular node count are exposed as parameters.

Usage:
    from _structured_annulus import AnnulusInternalBoundaryStructured
    mesh = AnnulusInternalBoundaryStructured(
        radiusOuter=1.5, radiusInternal=1.0, radiusInner=0.5,
        nRadialInner=8, nRadialOuter=8, nAngular=64,
    )

The resulting mesh has the same boundary names as the unstructured
version: Lower (r=r_inner), Internal (r=r_int), Upper (r=r_outer).
"""

import os
from enum import Enum

import numpy as np
import underworld3 as uw
from underworld3.discretisation import Mesh


def AnnulusInternalBoundaryStructured(
    radiusOuter: float = 1.5,
    radiusInternal: float = 1.0,
    radiusInner: float = 0.5,
    nRadialInner: int = 8,
    nRadialOuter: int = 8,
    nAngular: int = 64,
    degree: int = 1,
    qdegree: int = 2,
    filename: str = None,
    gmsh_verbosity: int = 0,
    verbose: bool = False,
):
    """Annulus with an internal boundary, meshed by transfinite quads.

    Parameters
    ----------
    radiusOuter, radiusInternal, radiusInner
        Three concentric radii.
    nRadialInner
        Number of *cells* in the radial direction between r_inner and
        r_internal.
    nRadialOuter
        Number of *cells* in the radial direction between r_internal
        and r_outer.
    nAngular
        Number of *cells* in the angular direction around the full
        2π. Must be even (we split into two half-annuli).
    """
    if nAngular % 2 != 0:
        raise ValueError("nAngular must be even.")

    class boundaries(Enum):
        Lower = 1
        Internal = 2
        Upper = 3

    class regions(Enum):
        Inner = 101
        Outer = 102

    if filename is None:
        os.makedirs(".meshes", exist_ok=True)
        filename = (f".meshes/uw_annulus_struct_rO{radiusOuter}"
                    f"_rInt{radiusInternal}_rI{radiusInner}"
                    f"_nRi{nRadialInner}_nRo{nRadialOuter}"
                    f"_nA{nAngular}.msh")

    if uw.mpi.rank == 0:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        # Quad-friendly meshing
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.model.add("AnnulusFSStructured")

        # Centre point (used as arc centre; not a mesh node)
        c = gmsh.model.geo.add_point(0.0, 0.0, 0.0)

        # Six end points: at θ=0 and θ=π on each of the three circles
        p_iL = gmsh.model.geo.add_point(radiusInner,  0.0, 0.0)
        p_iR = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0)
        p_xL = gmsh.model.geo.add_point(radiusInternal, 0.0, 0.0)
        p_xR = gmsh.model.geo.add_point(-radiusInternal, 0.0, 0.0)
        p_oL = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0)
        p_oR = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0)

        # Six arcs: top half and bottom half of each circle
        a_iT = gmsh.model.geo.add_circle_arc(p_iL, c, p_iR)  # inner θ∈[0,π]
        a_iB = gmsh.model.geo.add_circle_arc(p_iR, c, p_iL)  # inner θ∈[π,2π]
        a_xT = gmsh.model.geo.add_circle_arc(p_xL, c, p_xR)
        a_xB = gmsh.model.geo.add_circle_arc(p_xR, c, p_xL)
        a_oT = gmsh.model.geo.add_circle_arc(p_oL, c, p_oR)
        a_oB = gmsh.model.geo.add_circle_arc(p_oR, c, p_oL)

        # Four radial spokes (θ=0 right side, θ=π left side, both
        # sub-divided across the two layers):
        #   r_iL → r_xL → r_oL  (θ=0)
        #   r_iR → r_xR → r_oR  (θ=π)
        s_inn_R = gmsh.model.geo.add_line(p_iL, p_xL)  # right inner spoke
        s_out_R = gmsh.model.geo.add_line(p_xL, p_oL)  # right outer spoke
        s_inn_L = gmsh.model.geo.add_line(p_iR, p_xR)  # left  inner spoke
        s_out_L = gmsh.model.geo.add_line(p_xR, p_oR)  # left  outer spoke

        # Four surfaces (two layers × two halves):
        #   inner-top:    spokes s_inn_R, s_inn_L; arcs a_iT, a_xT
        #   inner-bot:    spokes s_inn_R, s_inn_L; arcs a_iB, a_xB
        #   outer-top:    spokes s_out_R, s_out_L; arcs a_xT, a_oT
        #   outer-bot:    spokes s_out_R, s_out_L; arcs a_xB, a_oB
        # Each curve_loop must traverse the boundary in order.
        cl_inn_T = gmsh.model.geo.add_curve_loop(
            [a_iT, s_inn_L, -a_xT, -s_inn_R])
        cl_inn_B = gmsh.model.geo.add_curve_loop(
            [a_iB, s_inn_R, -a_xB, -s_inn_L])
        cl_out_T = gmsh.model.geo.add_curve_loop(
            [a_xT, s_out_L, -a_oT, -s_out_R])
        cl_out_B = gmsh.model.geo.add_curve_loop(
            [a_xB, s_out_R, -a_oB, -s_out_L])

        s_inn_T = gmsh.model.geo.add_plane_surface([cl_inn_T])
        s_inn_B = gmsh.model.geo.add_plane_surface([cl_inn_B])
        s_out_T = gmsh.model.geo.add_plane_surface([cl_out_T])
        s_out_B = gmsh.model.geo.add_plane_surface([cl_out_B])

        gmsh.model.geo.synchronize()

        # Physical groups: boundaries
        gmsh.model.addPhysicalGroup(
            1, [a_iT, a_iB], boundaries.Lower.value,
            name=boundaries.Lower.name)
        gmsh.model.addPhysicalGroup(
            1, [a_xT, a_xB], boundaries.Internal.value,
            name=boundaries.Internal.name)
        gmsh.model.addPhysicalGroup(
            1, [a_oT, a_oB], boundaries.Upper.value,
            name=boundaries.Upper.name)

        # Region physical groups (2D) — labels cells by region
        gmsh.model.addPhysicalGroup(
            2, [s_inn_T, s_inn_B], tag=regions.Inner.value,
            name=regions.Inner.name)
        gmsh.model.addPhysicalGroup(
            2, [s_out_T, s_out_B], tag=regions.Outer.value,
            name=regions.Outer.name)
        gmsh.model.addPhysicalGroup(
            2, [s_inn_T, s_inn_B, s_out_T, s_out_B], 666666, "Elements")

        gmsh.model.geo.synchronize()

        # Transfinite curves & surfaces (set AFTER physical groups,
        # immediately before mesh.generate — matches the working
        # pattern used by underworld3.meshing.geographic).
        nA_half = nAngular // 2 + 1
        nRi = nRadialInner + 1
        nRo = nRadialOuter + 1

        for arc in (a_iT, a_iB, a_xT, a_xB, a_oT, a_oB):
            gmsh.model.mesh.setTransfiniteCurve(arc, numNodes=nA_half)
        for spoke in (s_inn_R, s_inn_L):
            gmsh.model.mesh.setTransfiniteCurve(spoke, numNodes=nRi)
        for spoke in (s_out_R, s_out_L):
            gmsh.model.mesh.setTransfiniteCurve(spoke, numNodes=nRo)

        for surf in (s_inn_T, s_inn_B, s_out_T, s_out_B):
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
        gmsh.finalize()

    if verbose and uw.mpi.rank == 0:
        print(f"  wrote structured annulus mesh: {filename}",
              flush=True)

    new_mesh = Mesh(
        filename,
        degree=degree,
        qdegree=qdegree,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D,
        useMultipleTags=True,
        useRegions=True,
    )

    return new_mesh


def AnnulusStructured(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.5,
    nRadial: int = 8,
    nAngular: int = 64,
    degree: int = 1,
    qdegree: int = 2,
    filename: str = None,
    gmsh_verbosity: int = 0,
    verbose: bool = False,
):
    """Single-layer rock annulus with Lower (r=r_inner, no-slip) and
    Upper (r=r_outer, free surface) boundaries — no internal boundary,
    no air layer. Polar-quad transfinite mesh with quad cells.
    """
    if nAngular % 2 != 0:
        raise ValueError("nAngular must be even.")

    class boundaries(Enum):
        Lower = 1
        Upper = 3

    if filename is None:
        os.makedirs(".meshes", exist_ok=True)
        filename = (f".meshes/uw_annulus_struct_rock_rO{radiusOuter}"
                    f"_rI{radiusInner}_nR{nRadial}_nA{nAngular}.msh")

    if uw.mpi.rank == 0:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.model.add("AnnulusStructured")

        c = gmsh.model.geo.add_point(0.0, 0.0, 0.0)
        p_iL = gmsh.model.geo.add_point(radiusInner,  0.0, 0.0)
        p_iR = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0)
        p_oL = gmsh.model.geo.add_point(radiusOuter,  0.0, 0.0)
        p_oR = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0)

        a_iT = gmsh.model.geo.add_circle_arc(p_iL, c, p_iR)
        a_iB = gmsh.model.geo.add_circle_arc(p_iR, c, p_iL)
        a_oT = gmsh.model.geo.add_circle_arc(p_oL, c, p_oR)
        a_oB = gmsh.model.geo.add_circle_arc(p_oR, c, p_oL)

        s_R = gmsh.model.geo.add_line(p_iL, p_oL)  # right spoke
        s_L = gmsh.model.geo.add_line(p_iR, p_oR)  # left spoke

        cl_T = gmsh.model.geo.add_curve_loop([a_iT, s_L, -a_oT, -s_R])
        cl_B = gmsh.model.geo.add_curve_loop([a_iB, s_R, -a_oB, -s_L])

        s_T = gmsh.model.geo.add_plane_surface([cl_T])
        s_B = gmsh.model.geo.add_plane_surface([cl_B])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(
            1, [a_iT, a_iB], boundaries.Lower.value,
            name=boundaries.Lower.name)
        gmsh.model.addPhysicalGroup(
            1, [a_oT, a_oB], boundaries.Upper.value,
            name=boundaries.Upper.name)
        gmsh.model.addPhysicalGroup(2, [s_T, s_B], 666666, "Elements")

        gmsh.model.geo.synchronize()

        nA_half = nAngular // 2 + 1
        nR = nRadial + 1
        for arc in (a_iT, a_iB, a_oT, a_oB):
            gmsh.model.mesh.setTransfiniteCurve(arc, numNodes=nA_half)
        for spoke in (s_R, s_L):
            gmsh.model.mesh.setTransfiniteCurve(spoke, numNodes=nR)
        for surf in (s_T, s_B):
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
        gmsh.finalize()

    if verbose and uw.mpi.rank == 0:
        print(f"  wrote rock-only structured annulus mesh: {filename}",
              flush=True)

    new_mesh = Mesh(
        filename,
        degree=degree,
        qdegree=qdegree,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D,
        useMultipleTags=True,
        useRegions=True,
    )

    return new_mesh


if __name__ == "__main__":
    # Smoke test
    m = AnnulusInternalBoundaryStructured(
        radiusOuter=1.5, radiusInternal=1.0, radiusInner=0.5,
        nRadialInner=6, nRadialOuter=6, nAngular=48,
        verbose=True,
    )
    print(f"Internal-boundary mesh: {m.X.coords.shape[0]} nodes, "
          f"{m._centroids.shape[0]} cells")
    m2 = AnnulusStructured(
        radiusOuter=1.0, radiusInner=0.5,
        nRadial=8, nAngular=64,
        verbose=True,
    )
    print(f"Rock-only mesh: {m2.X.coords.shape[0]} nodes, "
          f"{m2._centroids.shape[0]} cells")
