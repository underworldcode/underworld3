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
    r"""
    Create a 3D spherical shell mesh.

    Generates a tetrahedral mesh on a full spherical shell (or solid
    sphere if ``radiusInner=0``) using Gmsh's OpenCASCADE geometry
    kernel. Provides :math:`(r, \theta, \phi)` coordinates for
    convenient representation of radial problems.

    Parameters
    ----------
    radiusOuter : float, default=1.0
        Outer radius of the spherical shell.
    radiusInner : float, default=0.547
        Inner radius of the spherical shell. Set to 0 for a solid
        sphere extending to the centre.
    cellSize : float, default=0.1
        Target mesh element size.
    degree : int, default=1
        Polynomial degree of finite element basis functions.
    qdegree : int, default=2
        Quadrature degree for numerical integration.
    filename : str, optional
        Path to save the mesh file.
    refinement : int, optional
        Number of uniform refinement levels to apply. Each level
        approximately octruples element count.
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
        - ``Centre``: Centre point (if radiusInner=0)

        The mesh includes a refinement callback that snaps boundary
        nodes back to true spherical geometry after refinement.

    See Also
    --------
    Annulus : 2D equivalent.
    SphericalShellInternalBoundary : Spherical shell with internal surface.
    CubedSphere : Structured spherical shell mesh.

    Examples
    --------
    Create a spherical shell for mantle convection:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.SphericalShell(
    ...     radiusOuter=1.0,
    ...     radiusInner=0.547,
    ...     cellSize=0.1
    ... )

    Create a solid sphere:

    >>> mesh = uw.meshing.SphericalShell(
    ...     radiusOuter=1.0,
    ...     radiusInner=0.0,
    ...     cellSize=0.1
    ... )

    Notes
    -----
    The inner radius default of 0.547 corresponds approximately to the
    Earth's core-mantle boundary radius ratio (3480/6371).

    The mesh coordinate system provides unit vectors via
    ``mesh.CoordinateSystem``:

    - ``unit_e_0``: radial direction :math:`(r)`
    - ``unit_e_1``: colatitude direction :math:`(\theta)`
    - ``unit_e_2``: longitude direction :math:`(\phi)`

    For free-slip boundary conditions on vector problems (e.g., Stokes),
    use a penalty on the normal velocity component::

        Gamma_N = mesh.Gamma  # discrete normal, or
        Gamma_N = mesh.CoordinateSystem.unit_e_0  # analytic radial
        stokes.add_natural_bc(
            penalty * Gamma_N.dot(v_soln.sym) * Gamma_N, "Upper"
        )

    """
    class boundaries(Enum):
        Lower = 11
        Upper = 12
        Centre = 1

    import gmsh

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = (
            f".meshes/uw_spherical_shell_ro{radiusOuter}_ri{radiusInner}_csize{cellSize}.msh"
        )
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
                if np.isclose(gmsh.model.get_bounding_box(surface[0], surface[1])[-1], radiusInner):
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

    # Full spherical shell: 3 rigid rotation modes
    x, y, z = new_mesh.X
    new_mesh._nullspace_rotations = [
        sympy.Matrix([0, -z, y]),   # rotation about x
        sympy.Matrix([z, 0, -x]),   # rotation about y
        sympy.Matrix([-y, x, 0]),   # rotation about z
    ]

    # Bounding-surface objects for tangent slip: Upper/Lower are radial about the
    # origin at the known radii (see docs/developer/design/boundary-slip-strategy.md).
    from underworld3.meshing.bounding_surface import register_radial_surfaces
    register_radial_surfaces(
        new_mesh, centre=(0.0, 0.0, 0.0),
        label_radius={"Upper": radiusOuter, "Lower": radiusInner})

    return new_mesh


@timing.routine_timer_decorator
def SphericalManifold(
    radius: float = 1.0,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
):
    r"""
    Create a 2-manifold mesh of a spherical surface, embedded in 3-D.

    Produces a triangular surface mesh with topological dimension
    ``dim=2`` and coordinate dimension ``cdim=3``: a 2-sphere lying in
    Euclidean 3-space. This is the manifold sibling of
    :func:`SphericalShell` (which builds a 3-D volume between two
    radii); both share the spherical coordinate system so
    :math:`(r, \theta, \phi)` accessors are available on either.

    The mesh is generated by building a 3-D ball in gmsh, extracting
    its boundary surface, and meshing only the surface — no volume
    cells are produced. The resulting DM has a clean three-stratum
    chart (cells, edges, vertices) and supports PDEs solved on the
    manifold (Laplace–Beltrami, surface advection-diffusion, surface
    flow).

    Parameters
    ----------
    radius : float, default=1.0
        Radius of the sphere.
    cellSize : float, default=0.1
        Target mesh element size (chord length).
    degree : int, default=1
        Polynomial degree of finite element basis functions.
    qdegree : int, default=2
        Quadrature degree for numerical integration.
    filename : str, optional
        Path to save the gmsh ``.msh`` file.
    refinement : int, optional
        Number of uniform refinement levels to apply.
    gmsh_verbosity : int, default=0
        Gmsh output verbosity level.
    verbose : bool, default=False
        Print diagnostic information.

    Returns
    -------
    Mesh
        A 2-manifold mesh with ``dim=2``, ``cdim=3``. The closed
        sphere has no boundary curve; ``boundaries`` carries a single
        ``Surface`` label covering the whole manifold for
        convenience. The mesh registers spherical coordinate
        accessors (``mesh.X.spherical.r``, ``.theta``, ``.phi``;
        symbolic ``mesh.r`` etc.) the same way :func:`SphericalShell`
        does.

    See Also
    --------
    SphericalShell : 3-D volume sibling.
    extract_surface : extract a manifold from an existing volume mesh's
        boundary (alternative construction path).

    Notes
    -----
    The closed sphere has 3 rigid-body rotation modes (about each
    Cartesian axis) which are recorded on ``mesh._nullspace_rotations``
    just as for ``SphericalShell``. These tangent-to-surface vector
    fields lie in the kernel of the surface Stokes / shallow-water
    operators and must be projected out in solvers.

    Examples
    --------
    Construct a unit-sphere manifold mesh and inspect its dimensions::

        >>> import underworld3 as uw
        >>> sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.1)
        >>> sphere.dim, sphere.cdim
        (2, 3)

    Access spherical coordinates::

        >>> r     = sphere.X.spherical.r        # constant == radius
        >>> theta = sphere.X.spherical.theta    # co-latitude
        >>> phi   = sphere.X.spherical.phi      # longitude
    """

    class boundaries(Enum):
        Surface = 12  # the manifold itself (closed sphere has no boundary curve)

    import gmsh

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)
        uw_filename = (
            f".meshes/uw_spherical_manifold_r{radius}_csize{cellSize}.msh"
        )
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        gmsh.model.add("SphericalManifold")

        ball_tag = gmsh.model.occ.addSphere(0, 0, 0, radius)
        gmsh.model.occ.synchronize()

        # Extract the 2-D boundary surfaces of the ball, then discard
        # the volume — we want a surface-only mesh.
        surfaces = gmsh.model.getEntities(2)
        gmsh.model.occ.remove([(3, ball_tag)], recursive=False)
        gmsh.model.occ.synchronize()

        # Tag the surface so it has a name we can refer to.
        surface_dim_tags = [s[1] for s in surfaces]
        gmsh.model.addPhysicalGroup(
            2,
            surface_dim_tags,
            boundaries.Surface.value,
            name=boundaries.Surface.name,
        )

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cellSize)
        gmsh.model.mesh.generate(2)  # 2-D mesh in 3-D space
        gmsh.write(uw_filename)
        gmsh.finalize()

    # Force the PETSc gmsh reader to preserve the 3-D embedding when
    # loading the 2-D surface mesh. Without this, DMPlexCreateGmsh
    # defaults coordDim = dim (= 2) and drops the z-coordinate, which
    # collapses the manifold to its xy-projection. We load the DM
    # directly here (rather than going through ``_from_gmsh``'s
    # h5 round-trip, which currently loses the explicit coordinate
    # dimension), then hand the DM to Mesh() which accepts either a
    # filename or a DMPlex object.
    PETSc.Options().setValue("dm_plex_gmsh_spacedim", 3)
    surface_dm = PETSc.DMPlex().createFromFile(
        uw_filename, interpolate=True, comm=PETSc.COMM_WORLD,
    )
    surface_dm.setName("uw_mesh")
    surface_dm.markBoundaryFaces("All_Boundaries", 1001)
    assert surface_dm.getCoordinateDim() == 3, (
        f"SphericalManifold expected cdim=3, got "
        f"cdim={surface_dm.getCoordinateDim()} after gmsh load"
    )

    def spherical_manifold_refinement_callback(dm):
        """Snap refined vertices back onto the true sphere."""
        c = dm.getCoordinatesLocal()
        coords = c.array.reshape(-1, 3)
        R = np.sqrt((coords ** 2).sum(axis=1))
        # Every vertex should lie on the sphere — push it back exactly.
        coords *= (radius / np.maximum(R, 1.0e-30)).reshape(-1, 1)
        c.array[...] = coords.reshape(-1)
        dm.setCoordinatesLocal(c)

    def sphere_manifold_return_coords_to_bounds(coords):
        """Project arbitrary R^3 points onto the sphere surface.

        Plugged into ``Mesh.return_coords_to_bounds`` so the SLCN
        semi-Lagrangian trace-back machinery automatically maps each
        Euler-step launch point back to the manifold after sampling
        a tangent velocity field. The Euler step in R^3 leaves the
        surface by O((|v| dt)^2 / R) per substep; this projection
        wipes that drift between substeps. The closed-form sphere
        projection is exact — no kd-tree or per-cell barycentric
        needed for the spherical case.

        Coords are modified in-place AND returned, matching the
        contract used by box/annulus/geographic factories.
        """
        R = np.sqrt((coords ** 2).sum(axis=1))
        scale = radius / np.maximum(R, 1.0e-30)
        coords *= scale.reshape(-1, 1)
        return coords

    new_mesh = Mesh(
        surface_dm,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        boundary_normals=None,
        refinement=refinement,
        refinement_callback=spherical_manifold_refinement_callback,
        return_coords_to_bounds=sphere_manifold_return_coords_to_bounds,
        verbose=verbose,
    )

    # The closed sphere has 3 rigid-body rotation modes — same as
    # SphericalShell. These tangent-to-surface vector fields live in
    # the kernel of surface Stokes / shallow-water operators and need
    # projecting out at solve time.
    x, y, z = new_mesh.X
    new_mesh._nullspace_rotations = [
        sympy.Matrix([0, -z, y]),   # rotation about x
        sympy.Matrix([z, 0, -x]),   # rotation about y
        sympy.Matrix([-y, x, 0]),   # rotation about z
    ]

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

    Parameters
    ----------
    radiusOuter : float, optional
        The outer radius of the spherical shell. Default is 1.0.
    radiusInternal : float, optional
        The radius of the internal boundary. Default is 0.8.
    radiusInner : float, optional
        The inner radius of the spherical shell. Default is 0.547.
    cellSize : float, optional
        The target size for the mesh elements. Default is 0.1.
    degree : int, optional
        The polynomial degree of the finite elements. Default is 1.
    qdegree : int, optional
        The quadrature degree for integration. Default is 2.
    filename : str, optional
        The name of the file where the mesh will be saved. Default is None.
    refinement : optional
        Refinement level for the mesh. Default is None.
    gmsh_verbosity : int, optional
        Gmsh output verbosity (0=quiet). Default is 0.
    verbose : bool, optional
        If True, print additional information. Default is False.

    Returns
    -------
    Mesh
        The generated spherical shell mesh with internal boundary.

    Examples
    --------
    >>> mesh = uw.meshing.SphericalShellInternalBoundary(
    ...     radiusOuter=2.0,
    ...     radiusInternal=1.5,
    ...     radiusInner=1.0,
    ...     cellSize=0.05,
    ... )
    """

    class boundaries(Enum):
        Centre = 1
        Lower = 11
        Internal = 12
        Upper = 13

    class regions(Enum):
        Inner = 101
        Outer = 102

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

        # Create the spherical shell volume.
        outer = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radiusOuter)
        inner = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radiusInner)
        gmsh.model.occ.cut(
            [(3, outer)],
            [(3, inner)],
            removeObject=True,
            removeTool=True,
        )

        # Create an internal shell only to obtain a clean spherical surface at
        # radiusInternal. That surface is embedded into the shell volume below;
        # the duplicate volume and duplicate lower surface are removed before
        # meshing.
        internal = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radiusInternal)
        inner_copy = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radiusInner)
        gmsh.model.occ.cut(
            [(3, internal)],
            [(3, inner_copy)],
            removeObject=True,
            removeTool=True,
        )

        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cellSize)

        def bbox_radius(dimtag):
            """Estimate a concentric sphere radius from the bounding box."""
            bb = gmsh.model.get_bounding_box(dimtag[0], dimtag[1])
            return np.sqrt(bb[3]**2 + bb[4]**2 + bb[5]**2) / np.sqrt(3.0)

        volumes = gmsh.model.getEntities(3)
        shell_vols = [vol for vol in volumes if np.isclose(bbox_radius(vol), radiusOuter, atol=cellSize * 0.5)]
        duplicate_vols = [vol for vol in volumes if np.isclose(bbox_radius(vol), radiusInternal, atol=cellSize * 0.5)]

        if len(shell_vols) != 1:
            raise RuntimeError(
                "Could not identify the spherical-shell volume while building "
                "SphericalShellInternalBoundary."
            )

        shell_vol = shell_vols[0]
        shell_boundary = {
            dimtag[1]
            for dimtag in gmsh.model.getBoundary([shell_vol], oriented=False, recursive=False)
            if dimtag[0] == 2
        }

        lower_surface_tags = []
        internal_surface_tags = []
        upper_surface_tags = []
        duplicate_lower_tags = []

        for surface in gmsh.model.getEntities(2):
            surface_tag = surface[1]
            r_est = bbox_radius(surface)
            if np.isclose(r_est, radiusInner, atol=cellSize * 0.5):
                if surface_tag in shell_boundary:
                    lower_surface_tags.append(surface_tag)
                else:
                    duplicate_lower_tags.append(surface_tag)
            elif np.isclose(r_est, radiusOuter, atol=cellSize * 0.5):
                upper_surface_tags.append(surface_tag)
            elif np.isclose(r_est, radiusInternal, atol=cellSize * 0.5):
                internal_surface_tags.append(surface_tag)

        if not lower_surface_tags or not upper_surface_tags or not internal_surface_tags:
            raise RuntimeError(
                "Could not identify Lower, Internal, and Upper spherical surfaces "
                "while building SphericalShellInternalBoundary."
            )

        gmsh.model.mesh.embed(2, internal_surface_tags, shell_vol[0], shell_vol[1])

        remove_dimtags = duplicate_vols + [(2, tag) for tag in duplicate_lower_tags]
        if remove_dimtags:
            gmsh.model.remove_entities(remove_dimtags, recursive=False)
            gmsh.model.occ.remove(remove_dimtags, recursive=False)
            gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(
            2, lower_surface_tags, boundaries.Lower.value, name=boundaries.Lower.name
        )
        gmsh.model.addPhysicalGroup(
            2, internal_surface_tags, boundaries.Internal.value, name=boundaries.Internal.name
        )
        gmsh.model.addPhysicalGroup(
            2, upper_surface_tags, boundaries.Upper.value, name=boundaries.Upper.name
        )

        gmsh.model.addPhysicalGroup(shell_vol[0], [shell_vol[1]], 99999, "Elements")

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

    # boundary_normals deprecated — use mesh.Gamma_P1 for boundary normals

    new_mesh.regions = regions

    # Full spherical shell with internal boundary: 3 rigid rotation modes
    x, y, z = new_mesh.X
    new_mesh._nullspace_rotations = [
        sympy.Matrix([0, -z, y]),
        sympy.Matrix([z, 0, -x]),
        sympy.Matrix([-y, x, 0]),
    ]

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

    Parameters
    ----------
    radiusOuter : float, optional
        The outer radius of the spherical segment. Default is 1.0.
    radiusInner : float, optional
        The inner radius of the spherical segment. Default is 0.547.
    longitudeExtent : float, optional
        The angular extent in longitude (degrees). Default is 90.0.
    latitudeExtent : float, optional
        The angular extent in latitude (degrees). Default is 90.0.
    cellSize : float, optional
        The target size for the mesh elements. Default is 0.1.
    degree : int, optional
        The polynomial degree of the finite elements. Default is 1.
    qdegree : int, optional
        The quadrature degree for integration. Default is 2.
    filename : str, optional
        The name of the file where the mesh will be saved. Default is None.
    refinement : optional
        Refinement level for the mesh. Default is None.
    gmsh_verbosity : int, optional
        Gmsh output verbosity (0=quiet). Default is 0.
    verbose : bool, optional
        If True, print additional information. Default is False.
    centroid : tuple of float, optional
        The coordinates of the centroid of the sphere segment.
        Default is (0.0, 0.0, 0.0).

    Returns
    -------
    Mesh
        The generated spherical segment mesh.

    Examples
    --------
    >>> mesh = uw.meshing.SegmentofSphere(
    ...     radiusOuter=2.0,
    ...     radiusInner=1.0,
    ...     longitudeExtent=120.0,
    ...     latitudeExtent=60.0,
    ...     cellSize=0.05,
    ... )
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

    if radiusInner <= 0 or not (0 < longitudeExtent < 180) or not (0 < latitudeExtent < 180):
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

        p0 = gmsh.model.geo.addPoint(centroid[0], centroid[1], centroid[2], meshSize=cellSize)

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
    r"""
    Create a cubed-sphere mesh with structured elements.

    Generates a spherical shell mesh using the cubed-sphere projection,
    which maps six cube faces onto the sphere. Produces hexahedral
    elements (or tetrahedra if ``simplex=True``) with uniform resolution.

    Parameters
    ----------
    radiusOuter : float, default=1.0
        Outer radius of the spherical shell.
    radiusInner : float, default=0.547
        Inner radius of the spherical shell.
    numElements : int, default=5
        Number of elements along each edge of each cube face. Total
        elements approximately ``6 * numElements^2`` per radial layer.
    degree : int, default=1
        Polynomial degree of finite element basis functions.
    qdegree : int, default=2
        Quadrature degree for numerical integration.
    simplex : bool, default=False
        If False, use hexahedral elements. If True, subdivide into
        tetrahedra (simplex elements).
    filename : str, optional
        Path to save the mesh file.
    refinement : int, optional
        Number of uniform refinement levels to apply.
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

    See Also
    --------
    SphericalShell : Unstructured tetrahedral spherical mesh.
    SegmentofSphere : Partial spherical shell.

    Examples
    --------
    Create a cubed-sphere mesh with 10 elements per edge:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.CubedSphere(
    ...     radiusOuter=1.0,
    ...     radiusInner=0.55,
    ...     numElements=10
    ... )

    Notes
    -----
    The cubed-sphere projection avoids polar singularities present in
    latitude-longitude grids, providing quasi-uniform element sizes
    across the sphere. This is particularly advantageous for global
    geodynamics simulations.

    The mesh coordinate system provides unit vectors via
    ``mesh.CoordinateSystem``:

    - ``unit_e_0``: radial direction :math:`(r)`
    - ``unit_e_1``: colatitude direction :math:`(\theta)`
    - ``unit_e_2``: longitude direction :math:`(\phi)`

    """

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
        gmsh.model.geo.rotate(gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.pi)
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

        gmsh.model.addPhysicalGroup(2, [1, 34, 61, 88, 115, 137], boundaries.Upper.value)
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
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        return_coords_to_bounds=sphere_return_coords_to_bounds,
        verbose=verbose,
    )

    class boundary_normals(Enum):
        Lower = new_mesh.CoordinateSystem.unit_e_0
        Upper = new_mesh.CoordinateSystem.unit_e_0

    # boundary_normals deprecated — use mesh.Gamma_P1 for boundary normals

    # Full cubed sphere: 3 rigid rotation modes
    x, y, z = new_mesh.X
    new_mesh._nullspace_rotations = [
        sympy.Matrix([0, -z, y]),
        sympy.Matrix([z, 0, -x]),
        sympy.Matrix([-y, x, 0]),
    ]

    # Bounding-surface objects for tangent slip: Upper/Lower are radial about the
    # origin at the known radii (see docs/developer/design/boundary-slip-strategy.md).
    from underworld3.meshing.bounding_surface import register_radial_surfaces
    register_radial_surfaces(
        new_mesh, centre=(0.0, 0.0, 0.0),
        label_radius={"Upper": radiusOuter, "Lower": radiusInner})

    return new_mesh
