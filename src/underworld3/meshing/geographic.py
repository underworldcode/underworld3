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
    r"""
    Create a regional spherical box mesh (cubed-sphere section).

    Generates a 3D structured mesh for a regional section of a spherical shell,
    using a cubed-sphere projection. The domain is defined by corner coordinates
    in degrees (longitude, latitude) and radial bounds.

    Parameters
    ----------
    radiusOuter : float, default=1.0
        Outer radius of the spherical shell.
    radiusInner : float, default=0.547
        Inner radius of the spherical shell.
    SWcorner : list of float, default=[-45, -45]
        Southwest corner as [longitude, latitude] in degrees.
    NEcorner : list of float, default=[+45, +45]
        Northeast corner as [longitude, latitude] in degrees.
    numElementsLon : int, default=5
        Number of elements in the longitude direction.
    numElementsLat : int, default=5
        Number of elements in the latitude direction.
    numElementsDepth : int, default=5
        Number of elements in the radial (depth) direction.
    degree : int, default=1
        Polynomial degree of finite elements.
    qdegree : int, default=2
        Quadrature degree for numerical integration.
    simplex : bool, default=False
        If True, use tetrahedral elements; if False, use hexahedral.
    filename : str, optional
        Path to save the mesh file.
    refinement : int, optional
        Number of uniform refinement levels to apply.
    coarsening : int, optional
        Number of coarsening levels to apply.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level.
    verbose : bool, default=False
        Print diagnostic information.

    Returns
    -------
    Mesh
        A 3D mesh with boundaries:

        - ``Lower``: Inner surface at :math:`r = r_{inner}`
        - ``Upper``: Outer surface at :math:`r = r_{outer}`
        - ``North``: Northern boundary at :math:`\phi = \phi_{max}`
        - ``South``: Southern boundary at :math:`\phi = \phi_{min}`
        - ``East``: Eastern boundary at :math:`\lambda = \lambda_{max}`
        - ``West``: Western boundary at :math:`\lambda = \lambda_{min}`

        The mesh uses a SPHERICAL coordinate system and includes a refinement
        callback that snaps boundary nodes to true spherical geometry.

    See Also
    --------
    CubedSphere : Full cubed-sphere mesh.
    RegionalGeographicBox : Geographic mesh with ellipsoidal geometry.
    SphericalShell : Unstructured spherical shell.

    Examples
    --------
    Create a regional mesh for the Australian region:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.RegionalSphericalBox(
    ...     radiusOuter=1.0,
    ...     radiusInner=0.9,
    ...     SWcorner=[110, -45],
    ...     NEcorner=[155, -10],
    ...     numElementsLon=10,
    ...     numElementsLat=8,
    ...     numElementsDepth=5
    ... )

    Notes
    -----
    This mesh uses a cubed-sphere projection, which provides more uniform
    element sizes than a latitude-longitude grid. The structured mesh is
    suitable for regional mantle convection models where boundary-aligned
    elements are beneficial.

    The coordinate system provides unit vectors via ``mesh.CoordinateSystem``:

    - ``unit_e_0``: radial direction :math:`(r)`
    - ``unit_e_1``: colatitude direction :math:`(\theta)`
    - ``unit_e_2``: longitude direction :math:`(\phi)`
    """

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
        Lower = sympy.UnevaluatedExpr(new_mesh.CoordinateSystem.unit_e_0) * sympy.UnevaluatedExpr(
            sympy.Piecewise((1.0, new_mesh.CoordinateSystem.R[0] < 1.01 * radiusInner), (0.0, True))
        )
        Upper = sympy.UnevaluatedExpr(new_mesh.CoordinateSystem.unit_e_0) * sympy.UnevaluatedExpr(
            sympy.Piecewise((1.0, new_mesh.CoordinateSystem.R[0] > 0.99 * radiusOuter), (0.0, True))
        )

    new_mesh.boundary_normals = boundary_normals

    return new_mesh


@timing.routine_timer_decorator
def RegionalGeographicBox(
    lon_range: Tuple[float, float] = (135.0, 140.0),
    lat_range: Tuple[float, float] = (-35.0, -30.0),
    depth_range: Tuple[float, float] = (0.0, 400.0),
    ellipsoid="WGS84",
    numElements: Tuple[int, int, int] = (10, 10, 10),
    degree: int = 1,
    qdegree: int = 2,
    simplex: bool = True,
    filename: Optional[str] = None,
    refinement: Optional[int] = None,
    coarsening: Optional[int] = None,
    gmsh_verbosity: int = 0,
    verbose: bool = False,
):
    """
    Create a regional geographic mesh with ellipsoidal geometry.

    This function creates a structured 3D mesh in geographic coordinates
    (longitude, latitude, depth) on an ellipsoidal planet. The mesh uses
    geodetic latitude (perpendicular to ellipsoid surface) and measures
    depth below the reference ellipsoid surface.

    Parameters
    ----------
    lon_range : tuple of float, optional
        Longitude range in degrees East (lon_min, lon_max).
        Default: (135.0, 140.0) for southeastern Australia.
    lat_range : tuple of float, optional
        Latitude range in degrees North (lat_min, lat_max).
        Geodetic latitude (perpendicular to ellipsoid).
        Default: (-35.0, -30.0) for southeastern Australia.
    depth_range : tuple of float, optional
        Depth range in km below ellipsoid surface (depth_min, depth_max).
        Positive downward. depth_min=0 means surface.
        Default: (0.0, 400.0) for 0-400 km depth.
    ellipsoid : str, tuple, or bool, optional
        Ellipsoid specification:
        - str: Name from ELLIPSOIDS dict ('WGS84', 'Mars', 'Moon', 'Venus', 'sphere')
        - tuple: (semi_major_axis_km, semi_minor_axis_km) for custom ellipsoid
        - True: Use WGS84 (default)
        - False or 'sphere': Use perfect sphere with Earth mean radius
        Default: 'WGS84'
    numElements : tuple of int, optional
        Number of elements in (lon, lat, depth) directions.
        Default: (10, 10, 10)
    degree : int, optional
        Polynomial degree for finite elements (1=linear, 2=quadratic).
        Default: 1
    qdegree : int, optional
        Quadrature degree for numerical integration.
        Default: 2
    simplex : bool, optional
        If True, use tetrahedral elements. If False, use hexahedral elements.
        Default: True
    filename : str, optional
        Path to save generated mesh file. If None, uses automatic naming.
        Default: None
    refinement : int, optional
        Number of uniform refinement steps to apply.
        Default: None
    coarsening : int, optional
        Number of coarsening steps to apply.
        Default: None
    gmsh_verbosity : int, optional
        Gmsh output verbosity level (0=quiet, 5=very verbose).
        Default: 0
    verbose : bool, optional
        If True, print mesh generation details.
        Default: False

    Returns
    -------
    Mesh
        Underworld3 mesh object with GEOGRAPHIC coordinate system.
        Access geographic coordinates via mesh.geo:
        - mesh.geo.lon, mesh.geo.lat, mesh.geo.depth (data arrays)
        - mesh.geo[:] for symbolic coordinates (λ_lon, λ_lat, λ_d)
        - mesh.geo.unit_east, mesh.geo.unit_north, mesh.geo.unit_down (basis vectors)

    Examples
    --------
    # Create mesh for southeastern Australia, 0-400 km depth
    mesh = uw.meshing.RegionalGeographicBox(
        lon_range=(135, 140),
        lat_range=(-35, -30),
        depth_range=(0, 400),
        ellipsoid='WGS84',
        numElements=(20, 20, 10),
    )

    # Access geographic coordinates
    lon = mesh.geo.lon         # Longitude array (degrees East)
    lat = mesh.geo.lat         # Latitude array (degrees North)
    depth = mesh.geo.depth     # Depth array (km below surface)

    # Use in equations
    λ_lon, λ_lat, λ_d = mesh.geo[:]
    T = 1600 - 0.5 * λ_d       # Temperature decreasing with depth

    # Basis vectors for boundary conditions
    v_surface = 0 * mesh.geo.unit_up     # No vertical flow at surface
    v_bottom = 10 * mesh.geo.unit_down   # Downward flow at bottom

    # Mars example
    mesh_mars = uw.meshing.RegionalGeographicBox(
        lon_range=(0, 45),
        lat_range=(-22.5, 22.5),
        depth_range=(0, 200),
        ellipsoid='Mars',
        numElements=(15, 15, 8),
    )

    Notes
    -----
    - Uses geodetic latitude (GPS/map standard), not geocentric latitude
    - Depth is measured from reference ellipsoid surface, not from center
    - mesh.R provides spherical coordinates $(r, \\theta, \\phi)$ for backward compatibility
    - mesh.geo provides geographic coordinates (lon, lat, depth) with ellipsoid geometry
    - Right-handed coordinate system: WE × SN = down
    """
    from underworld3.coordinates import ELLIPSOIDS, geographic_to_cartesian
    from underworld3.units import (
        require_units_if_active,
        convert_angle_to_degrees,
        has_units,
    )

    # Check if units system is active
    model = uw.get_default_model()
    units_active = model is not None and model.has_units()

    # Process input parameters with unit awareness
    # Angles: accept quantities or raw (degrees)
    lon_min = convert_angle_to_degrees(lon_range[0], "lon_min")
    lon_max = convert_angle_to_degrees(lon_range[1], "lon_max")
    lat_min = convert_angle_to_degrees(lat_range[0], "lat_min")
    lat_max = convert_angle_to_degrees(lat_range[1], "lat_max")

    # Depths: require quantities when units active
    depth_min_nd = require_units_if_active(
        depth_range[0], "depth_min",
        expected_dimensionality="[length]",
        default_unit="km"
    )
    depth_max_nd = require_units_if_active(
        depth_range[1], "depth_max",
        expected_dimensionality="[length]",
        default_unit="km"
    )

    # Parse ellipsoid parameter
    if ellipsoid is True or ellipsoid == "WGS84":
        ellipsoid_dict = ELLIPSOIDS["WGS84"].copy()
    elif ellipsoid is False or ellipsoid == "sphere":
        ellipsoid_dict = ELLIPSOIDS["sphere"].copy()
    elif isinstance(ellipsoid, str):
        if ellipsoid not in ELLIPSOIDS:
            raise ValueError(
                f"Unknown ellipsoid '{ellipsoid}'. " f"Available: {list(ELLIPSOIDS.keys())}"
            )
        ellipsoid_dict = ELLIPSOIDS[ellipsoid].copy()
    elif isinstance(ellipsoid, (tuple, list)) and len(ellipsoid) == 2:
        # Custom ellipsoid (a, b)
        a, b = ellipsoid
        f = (a - b) / a if a != b else 0.0
        ellipsoid_dict = {
            "a": float(a),
            "b": float(b),
            "f": f,
            "planet": "Custom",
            "description": f"Custom ellipsoid (a={a} km, b={b} km)",
        }
    else:
        raise ValueError(
            f"Invalid ellipsoid parameter: {ellipsoid}. "
            "Use str name, (a, b) tuple, True for WGS84, or False for sphere."
        )

    # Get ellipsoid parameters (in km)
    a_km = ellipsoid_dict["a"]
    b_km = ellipsoid_dict["b"]

    # Nondimensionalize ellipsoid if units active
    if units_active:
        # Get reference length in km
        ref_length = model.get_fundamental_scales().get("length")
        if ref_length is not None:
            # Convert reference length to km
            if hasattr(ref_length, "to"):
                L_ref_km = float(ref_length.to("km").magnitude)
            elif hasattr(ref_length, "magnitude"):
                # Assume it's in base SI (meters)
                L_ref_km = float(ref_length.magnitude) / 1000.0
            else:
                L_ref_km = float(ref_length) / 1000.0

            # Nondimensional ellipsoid parameters
            a_nd = a_km / L_ref_km
            b_nd = b_km / L_ref_km

            # Store both in ellipsoid dict
            ellipsoid_dict["a_nd"] = a_nd
            ellipsoid_dict["b_nd"] = b_nd
            ellipsoid_dict["L_ref_km"] = L_ref_km

            # Use nondimensional values for mesh generation
            a = a_nd
            b = b_nd
            depth_min = depth_min_nd
            depth_max = depth_max_nd
        else:
            # No reference length available - use km
            a = a_km
            b = b_km
            depth_min = depth_min_nd
            depth_max = depth_max_nd
    else:
        # No units - use km directly
        a = a_km
        b = b_km
        depth_min = depth_min_nd
        depth_max = depth_max_nd

    # Unpack element counts (angles already processed above)
    numLon, numLat, numDepth = numElements

    # Validate inputs
    if not (-180 <= lon_min < lon_max <= 360):
        raise ValueError(f"Invalid longitude range: ({lon_min}, {lon_max}). Must be in [-180, 360].")
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(f"Invalid latitude range: ({lat_min}, {lat_max}). Must be in [-90, 90].")
    if not (0 <= depth_min < depth_max):
        raise ValueError(
            f"Invalid depth range: ({depth_min}, {depth_max}). Must be positive with depth_min < depth_max."
        )

    # Define boundary enum
    class boundaries(Enum):
        Surface = 1  # depth = depth_min (top)
        Bottom = 2  # depth = depth_max (bottom)
        North = 3  # lat = lat_max
        South = 4  # lat = lat_min
        East = 5  # lon = lon_max
        West = 6  # lon = lon_min

    # Generate mesh filename if not provided
    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = (
            f".meshes/uw_geographic_{ellipsoid_dict['planet']}_"
            f"lon{lon_min:.1f}_{lon_max:.1f}_"
            f"lat{lat_min:.1f}_{lat_max:.1f}_"
            f"d{depth_min:.0f}_{depth_max:.0f}_"
            f"n{numLon}x{numLat}x{numDepth}_"
            f"deg{degree}_{'simplex' if simplex else 'hex'}.msh"
        )
    else:
        uw_filename = filename

    # Generate mesh using gmsh (only on rank 0)
    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("Geographic Box")

        # Create grid of points in geographic coordinates
        # We create points at corners and let gmsh interpolate

        # Corner points in geographic coordinates
        geo_corners = [
            (lon_min, lat_min, depth_min),  # SW surface
            (lon_max, lat_min, depth_min),  # SE surface
            (lon_max, lat_max, depth_min),  # NE surface
            (lon_min, lat_max, depth_min),  # NW surface
            (lon_min, lat_min, depth_max),  # SW bottom
            (lon_max, lat_min, depth_max),  # SE bottom
            (lon_max, lat_max, depth_max),  # NE bottom
            (lon_min, lat_max, depth_max),  # NW bottom
        ]

        # Convert to Cartesian coordinates
        cart_corners = []
        for lon, lat, depth in geo_corners:
            x, y, z = geographic_to_cartesian(lon, lat, depth, a, b)
            cart_corners.append((x, y, z))

        # Add points to gmsh
        for i, (x, y, z) in enumerate(cart_corners, start=1):
            gmsh.model.geo.addPoint(x, y, z, tag=i)

        # Create box topology using lines and surfaces
        # Bottom face (depth=depth_min, surface)
        gmsh.model.geo.addLine(1, 2, tag=1)  # South edge
        gmsh.model.geo.addLine(2, 3, tag=2)  # East edge
        gmsh.model.geo.addLine(3, 4, tag=3)  # North edge
        gmsh.model.geo.addLine(4, 1, tag=4)  # West edge

        # Top face (depth=depth_max, bottom)
        gmsh.model.geo.addLine(5, 6, tag=5)
        gmsh.model.geo.addLine(6, 7, tag=6)
        gmsh.model.geo.addLine(7, 8, tag=7)
        gmsh.model.geo.addLine(8, 5, tag=8)

        # Vertical edges
        gmsh.model.geo.addLine(1, 5, tag=9)  # SW vertical
        gmsh.model.geo.addLine(2, 6, tag=10)  # SE vertical
        gmsh.model.geo.addLine(3, 7, tag=11)  # NE vertical
        gmsh.model.geo.addLine(4, 8, tag=12)  # NW vertical

        # Create surfaces (use Ruled Surface for compatibility with PETSc)
        # Surface (depth=depth_min)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)
        gmsh.model.geo.addSurfaceFilling([1], tag=1)

        # Bottom (depth=depth_max)
        gmsh.model.geo.addCurveLoop([5, 6, 7, 8], tag=2)
        gmsh.model.geo.addSurfaceFilling([2], tag=2)

        # South face (lat=lat_min)
        gmsh.model.geo.addCurveLoop([1, 10, -5, -9], tag=3)
        gmsh.model.geo.addSurfaceFilling([3], tag=3)

        # East face (lon=lon_max)
        gmsh.model.geo.addCurveLoop([2, 11, -6, -10], tag=4)
        gmsh.model.geo.addSurfaceFilling([4], tag=4)

        # North face (lat=lat_max)
        gmsh.model.geo.addCurveLoop([3, 12, -7, -11], tag=5)
        gmsh.model.geo.addSurfaceFilling([5], tag=5)

        # West face (lon=lon_min)
        gmsh.model.geo.addCurveLoop([4, 9, -8, -12], tag=6)
        gmsh.model.geo.addSurfaceFilling([6], tag=6)

        # Create volume
        gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4, 5, 6], tag=1)
        gmsh.model.geo.addVolume([1], tag=1)

        gmsh.model.geo.synchronize()

        # Set physical groups for boundaries
        gmsh.model.addPhysicalGroup(2, [1], boundaries.Surface.value)
        gmsh.model.setPhysicalName(2, boundaries.Surface.value, "Surface")

        gmsh.model.addPhysicalGroup(2, [2], boundaries.Bottom.value)
        gmsh.model.setPhysicalName(2, boundaries.Bottom.value, "Bottom")

        gmsh.model.addPhysicalGroup(2, [4], boundaries.South.value)
        gmsh.model.setPhysicalName(2, boundaries.South.value, "South")

        gmsh.model.addPhysicalGroup(2, [5], boundaries.East.value)
        gmsh.model.setPhysicalName(2, boundaries.East.value, "East")

        gmsh.model.addPhysicalGroup(2, [3], boundaries.North.value)
        gmsh.model.setPhysicalName(2, boundaries.North.value, "North")

        gmsh.model.addPhysicalGroup(2, [6], boundaries.West.value)
        gmsh.model.setPhysicalName(2, boundaries.West.value, "West")

        gmsh.model.addPhysicalGroup(3, [1], 99999)
        gmsh.model.setPhysicalName(3, 99999, "Elements")

        # Set transfinite meshing for structured grid
        # Edges in longitude direction
        for edge in [1, 3, 5, 7]:
            gmsh.model.mesh.setTransfiniteCurve(edge, numNodes=numLon + 1)

        # Edges in latitude direction
        for edge in [2, 4, 6, 8]:
            gmsh.model.mesh.setTransfiniteCurve(edge, numNodes=numLat + 1)

        # Edges in depth direction
        for edge in [9, 10, 11, 12]:
            gmsh.model.mesh.setTransfiniteCurve(edge, numNodes=numDepth + 1)

        # Set transfinite surfaces
        for surface in range(1, 7):
            gmsh.model.mesh.setTransfiniteSurface(surface)
            if not simplex:
                gmsh.model.mesh.set_recombine(2, surface)

        # Set transfinite volume
        if not simplex:
            gmsh.model.mesh.set_transfinite_volume(1)
            gmsh.model.mesh.set_recombine(3, 1)
        else:
            # For simplex, don't use transfinite volume, just generate
            pass

        # Generate mesh
        gmsh.model.mesh.generate(3)
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Load mesh on all ranks
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
        coarsening=coarsening,
        coordinate_system_type=CoordinateSystemType.GEOGRAPHIC,
        verbose=verbose,
    )

    # Store ellipsoid parameters in coordinate system
    new_mesh.CoordinateSystem.ellipsoid = ellipsoid_dict

    # Recreate geographic accessor with updated ellipsoid (in case default was used)
    from underworld3.coordinates import GeographicCoordinateAccessor

    new_mesh.CoordinateSystem._geo_accessor = GeographicCoordinateAccessor(
        new_mesh.CoordinateSystem
    )

    # Define boundary normals using geographic basis vectors
    class boundary_normals(Enum):
        Surface = new_mesh.CoordinateSystem.geo.unit_up  # Outward at surface
        Bottom = new_mesh.CoordinateSystem.geo.unit_down  # Downward at bottom
        North = new_mesh.CoordinateSystem.geo.unit_north  # Northward at north boundary
        South = new_mesh.CoordinateSystem.geo.unit_south  # Southward at south boundary
        East = new_mesh.CoordinateSystem.geo.unit_east  # Eastward at east boundary
        West = new_mesh.CoordinateSystem.geo.unit_west  # Westward at west boundary

    new_mesh.boundary_normals = boundary_normals

    return new_mesh
