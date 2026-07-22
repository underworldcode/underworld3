from typing import Optional, Tuple, Union
from enum import Enum
from contextlib import contextmanager

import os
import weakref
import threading
from mpi4py.MPI import Info
import numpy
import sympy
from sympy.matrices.expressions.blockmatrix import bc_dist
import sympy.vector
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities._api_tools import Stateful
from underworld3.utilities._api_tools import uw_object
from underworld3.utilities._utils import gather_data
from underworld3.utilities.nd_array_callback import (
    fire_canonical_callbacks,
    register_collective_flush,
)

from underworld3.coordinates import CoordinateSystem, CoordinateSystemType

# from underworld3.cython import petsc_discretisation
import underworld3.cython
import underworld3.timing as timing

## Introduce these two specific types of coordinate tracking vector objects

from sympy.vector import CoordSys3D


@contextmanager
def _temporary_petsc_option(key, value):
    opts = PETSc.Options()
    try:
        old_value = opts.getString(key)
    except KeyError:
        old_value = None
    opts[key] = value
    try:
        yield
    finally:
        if old_value is None:
            opts.delValue(key)
        else:
            opts[key] = old_value


# The ``dm_plex_gmsh_*`` options configure how PETSc imports a Gmsh ``.msh``
# file (coordinate space dimension, label handling, …). They are GLOBAL on the
# PETSc options database and are *import-time scratch* — meaningful only for the
# single import that follows. A mesh constructor that sets one (notably the
# manifold-surface generators set ``dm_plex_gmsh_spacedim = 2``) and then raises
# before cleanup leaks it, and the stale option silently corrupts the NEXT gmsh
# import (e.g. a subsequent SphericalShell is read with spacedim=2 → a 2-D mesh
# in a 3-D cdim → a "cannot reshape" crash, or a corrupt cache). ``_from_gmsh``
# therefore clears the whole namespace after every import (see below).
_GMSH_IMPORT_OPTION_KEYS = (
    "dm_plex_gmsh_spacedim",
    "dm_plex_gmsh_multiple_tags",
    "dm_plex_gmsh_use_regions",
    "dm_plex_gmsh_mark_vertices",
)


def _clear_gmsh_import_options():
    """Remove all ``dm_plex_gmsh_*`` import-scratch options from the global
    PETSc options database (idempotent)."""
    opts = PETSc.Options()
    for key in _GMSH_IMPORT_OPTION_KEYS:
        try:
            if opts.hasName(key):
                opts.delValue(key)
        except Exception:
            pass


## Add the ability to inherit an Enum, so we can add standard boundary
## types to ones that are supplied by the users / the meshing module
## https://stackoverflow.com/questions/46073413/python-enum-combination


def extend_enum(inherited):
    def wrapper(final):
        joined = {}
        inherited.append(final)
        for i in inherited:
            for j in i:
                joined[j.name] = j.value
        return Enum(final.__name__, joined)

    return wrapper


@timing.routine_timer_decorator
def _from_gmsh(filename, comm=None, markVertices=False, useRegions=True, useMultipleTags=True):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """

    ## NOTE: - this should be smart enough to serialise the msh conversion
    ## and then read back in parallel via h5.  This is currently done
    ## by every gmesh mesh

    comm = comm or PETSc.COMM_WORLD
    options = PETSc.Options()
    options["dm_plex_hash_location"] = None

    # This option allows objects to be in multiple physical groups
    # Rather than just the first one found.
    if useMultipleTags:
        options.setValue("dm_plex_gmsh_multiple_tags", True)
    else:
        options.setValue("dm_plex_gmsh_multiple_tags", False)

    # This is usually True because dmplex then contains
    # Labels for physical groups
    if useRegions:
        options["dm_plex_gmsh_use_regions"] = None

    else:
        options.delValue("dm_plex_gmsh_use_regions")

    # Marking the vertices may be necessary to constrain isolated points
    # but it means that the labels will have a mix of points, and edges / faces
    if markVertices:
        options.setValue("dm_plex_gmsh_mark_vertices", True)
    else:
        options.delValue("dm_plex_gmsh_mark_vertices")

    # this process is more efficient done on the root process and then distributed
    # we do this by saving the mesh as h5 which is more flexible to re-use later

    try:
        if uw.mpi.rank == 0:
            plex_0 = PETSc.DMPlex().createFromFile(filename, interpolate=True, comm=PETSc.COMM_SELF)

            plex_0.setName("uw_mesh")
            plex_0.markBoundaryFaces("All_Boundaries", 1001)

            viewer = PETSc.ViewerHDF5().create(filename + ".h5", "w", comm=PETSc.COMM_SELF)
            viewer(plex_0)
            viewer.destroy()
    finally:
        # The gmsh import options are import-time scratch — meaningful only for
        # the createFromFile above. Clear the whole namespace so a value set by
        # the caller for THIS import (notably ``dm_plex_gmsh_spacedim`` for a
        # manifold-surface generator) cannot leak into the next gmsh import and
        # silently produce a wrong-dimension mesh (e.g. a later SphericalShell
        # read as 2-D). Runs on success or failure.
        _clear_gmsh_import_options()

    # Now we have an h5 file and we can hand this to _from_plexh5

    return _from_plexh5(filename + ".h5", comm, return_sf=True)


@timing.routine_timer_decorator
def _from_plexh5(
    filename,
    comm=None,
    return_sf=False,
):
    """Read a dmplex .h5 file from `filename` provided.

    comm: Optional communicator to build the mesh on (defaults to
    COMM_WORLD).
    """

    if comm == None:
        comm = PETSc.COMM_WORLD

    viewer = PETSc.ViewerHDF5().create(filename, "r", comm=comm)
    h5plex = PETSc.DMPlex().create(comm=comm)
    h5plex.setName("uw_mesh")
    viewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
    try:
        sf0 = h5plex.topologyLoad(viewer)
        h5plex.coordinatesLoad(viewer, sf0)
        h5plex.labelsLoad(viewer, sf0)
        h5plex.markBoundaryFaces("All_Boundaries", 1001)
    finally:
        viewer.popFormat()
        viewer.destroy()

    if not return_sf:
        return h5plex
    else:
        return sf0, h5plex


def _hierarchy_sidecar_name(mesh_filename):
    """Filename for the coarse hierarchy sidecar of a mesh checkpoint.

    The geometric-multigrid (FMG) hierarchy is persisted as a single extra
    single-DM HDF5 file holding the **coarsest** level beside the main mesh
    checkpoint — PETSc's ``HDF5_PETSC`` format does not support several
    DMPlex objects in one file. The intermediate coarse levels are rebuilt
    by refinement on reload, so only ``mymesh.hierarchy.L0.h5`` is ever
    written (the ``L0`` suffix names that coarsest level).
    """
    base, ext = os.path.splitext(mesh_filename)
    return f"{base}.hierarchy.L0{ext}"


def _mesh_coords_update_callback(array, change_context):
    """Setter callback for a mesh's canonical coordinate array.

    Routes every user write to ``mesh.X.coords`` through the full
    ``_deform_mesh`` geometry rebuild and bumps ``_mesh_version`` so
    registered swarms see the coordinate change. Installed by
    :meth:`Mesh._install_coords_array` — the ONE callback shared by all
    sites that (re)create the coordinate array (construction, submesh
    re-extraction, adaptation), so the teardown guard and the identity
    gate below cannot silently diverge between copies again.

    Verbosity is read from the owning mesh's ``_coords_callback_verbose``
    flag (set at install time).
    """
    mesh = array.owner
    if mesh is None:
        # This guard handles cases where the array is accessed during
        # object teardown (e.g. at application exit or during mesh
        # replacement), where the owning Python mesh object has already
        # been garbage collected but the NDArray proxy still exists.
        return

    # ``NDArray_With_Callback.__array_finalize__`` propagates this
    # callback (and the owner) to every view / fancy-index copy of the
    # coordinate array. Only the mesh's *canonical* coordinate array
    # represents an actual coordinate update; a derived sub-array (e.g.
    # a boundary subset built inside the tangent-slip / bounding-surface
    # / mover machinery) that merely inherited this callback must NOT
    # trigger a full-mesh deform. Identity-gate on the canonical
    # ``_coords`` — note this is NOT a size filter, so a genuinely
    # malformed full coordinate update still reaches ``_deform_mesh``
    # and surfaces loudly rather than being silently dropped.
    if array is not mesh._coords:
        return

    verbose = getattr(mesh, "_coords_callback_verbose", False)
    if verbose:
        uw.pprint(f"Mesh update callback - mesh deform")

    coords = array.reshape(-1, mesh.cdim)
    try:
        mesh._deform_mesh(coords, verbose=verbose)
    except Exception:
        # The user's write landed in the canonical array BEFORE this
        # callback ran. A rejected/failed deform must not leave
        # mesh.X.coords disagreeing with the DM — the guard's message
        # says the write "is rejected", so make that true by restoring
        # the canonical from the DM, then let the error surface.
        # (np.copyto bypasses callbacks; this restore must not re-fire.)
        dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)
        cached = numpy.asarray(mesh._coords).reshape(-1, mesh.cdim)
        if cached.shape == dm_coords.shape:
            numpy.copyto(cached, dm_coords)
        # else: the failure replaced the coordinate Vec at a different size
        # (mid-rebuild) — restoring is impossible and a broadcast error here
        # would mask the original exception (round-2 review).
        raise

    # Increment mesh version to notify registered swarms of coordinate changes
    with mesh._mesh_update_lock:
        mesh._mesh_version += 1
        if verbose:
            uw.pprint(f"Mesh version incremented to {mesh._mesh_version}")

    return


class Mesh(Stateful, uw_object):
    r"""
    Unstructured mesh with PETSc DMPlex backend.

    The Mesh class provides the spatial discretisation for finite element
    computations. It wraps PETSc's DMPlex for unstructured mesh management,
    supporting various cell types (triangles, quadrilaterals, tetrahedra,
    hexahedra) and coordinate systems.

    Parameters
    ----------
    plex_or_meshfile : PETSc.DMPlex or str
        Either a PETSc DMPlex object or path to a mesh file (gmsh, exodus).
    degree : int, optional
        Polynomial degree for the coordinate field (default 1).
    simplex : bool, optional
        True for simplicial elements (triangles/tets), False for quads/hexes.
    coordinate_system_type : CoordinateSystemType, optional
        Coordinate system for vector calculus (Cartesian, cylindrical, etc.).
    qdegree : int, optional
        Quadrature degree for numerical integration (default 2).
    boundaries : list of NamedTuple, optional
        Boundary region definitions with names and values.
    boundary_normals : dict, optional
        Outward normal vectors for each boundary.
    units : str or pint.Unit, optional
        Deprecated and ignored (DeprecationWarning). Mesh coordinate units
        always come from the model's reference quantities
        (``model.set_reference_quantities``); query them via ``mesh.units``.
    verbose : bool, optional
        Print mesh construction information.

    Examples
    --------
    Meshes are typically created via the meshing module::

        >>> mesh = uw.meshing.UnstructuredSimplexBox(
        ...     minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
        ... )
        >>> T = mesh.add_variable("T", vtype=uw.VarType.SCALAR)

    See Also
    --------
    underworld3.meshing : Mesh generation utilities.
    underworld3.discretisation.MeshVariable : Field variables on meshes.
    """

    mesh_instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        plex_or_meshfile,
        degree=1,
        simplex=True,
        coordinate_system_type=None,
        qdegree=2,
        markVertices=None,
        useRegions=None,
        useMultipleTags=None,
        filename=None,
        refinement=None,
        refinement_callback=None,
        coarsening=None,
        coarsening_callback=None,
        return_coords_to_bounds=None,
        boundaries=None,
        boundary_normals=None,
        name=None,
        units=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        self.instance = Mesh.mesh_instances
        Mesh.mesh_instances += 1

        # Coordinate units come from the model (not a user parameter):
        # the model owns the unit system so all meshes and variables agree.
        import underworld3 as uw

        model = uw.get_default_model()

        # Deprecated 2026-07 (WA-23/WC-11, units-family ruling D7): the
        # `units=` kwarg was always ignored in favour of the model units;
        # now it says so.
        if units is not None:
            import warnings

            warnings.warn(
                f"The 'units' mesh-constructor parameter is deprecated and "
                f"ignored. Mesh coordinates use the model's units "
                f"('{model.get_coordinate_unit()}', set via "
                "model.set_reference_quantities). Remove the argument.",
                DeprecationWarning,
                stacklevel=3,
            )

        # Set units from model
        self.units = model.get_coordinate_unit()

        # Lock model units now that a mesh has been created
        # This prevents changing reference quantities after mesh exists
        model._lock_units()

        # === LENGTH SCALE FOR NON-DIMENSIONALIZATION ===
        # The length scale is IMMUTABLE after mesh creation to ensure
        # synchronization with all spatial operators (grad, div, curl)
        self._derive_length_scale_from_model(model)

        # Mesh coordinate version tracking for swarm coordination
        self._mesh_version = 0
        self._registered_swarms = weakref.WeakSet()
        self._registered_surfaces = weakref.WeakSet()  # Surfaces using this mesh
        self._registered_submeshes = weakref.WeakSet()  # Submeshes from extract_region
        self._registered_children = weakref.WeakSet()    # SBR refinement children from adapt()

        # _mesh_update_lock: Re-entrant lock to coordinate mesh deformation.
        # Held by mesh_update_callback during _deform_mesh(). Checked by
        # MeshVariable callbacks (blocking=False) to skip PETSc sync during
        # sensitive coordinate changes.
        self._mesh_update_lock = threading.RLock()

        name, boundaries, coordinate_system_type, regions = self._load_dm_from_file(
            plex_or_meshfile,
            boundaries,
            coordinate_system_type,
            markVertices,
            useRegions,
            useMultipleTags,
            verbose,
            kwargs,
        )

        self._patch_boundary_enum(
            plex_or_meshfile, boundaries, boundary_normals, regions, filename
        )

        ## ---
        ## Note - coarsening callback is tricky because the coarse meshes do not have the labels
        ##

        self.refinement_callback = refinement_callback
        self.coarsening_callback = coarsening_callback
        self.name = name
        self.sf1 = None
        self.return_coords_to_bounds = return_coords_to_bounds

        ## This is where we can refine the dm if required, and rebuild / redistribute

        if verbose and uw.mpi.rank == 0:
            print(
                f"Mesh refinement levels: {refinement}",
                flush=True,
            )
            print(
                f"Mesh coarsening levels: {coarsening}",
                flush=True,
            )

        uw.mpi.barrier()

        # Default: no navigation-only auxiliary DM. Only the
        # no-refinement / no-coarsening branch sets up a non-None
        # _nav_dm on manifold meshes. Other branches leave it as None
        # which means the navigation indices use self.dm directly.
        self._nav_dm = None

        if getattr(self, "_sidecar_coarsest", None) is not None:
            self._splice_hierarchy_from_sidecar()

        elif not refinement is None and refinement > 0:
            self._build_refined_hierarchy(refinement, refinement_callback)

        elif not coarsening is None and coarsening > 0:
            self._build_coarsened_hierarchy(coarsening)

        else:
            if not self.dm.isDistributed():
                self.sf1 = self.dm.distribute()

            # On manifold meshes (dim != cdim — e.g. SphericalManifold
            # and future bounded-surface patches) we want each rank to
            # see its neighbours' partition-boundary cells so that
            # surface query points near the seam can be located by
            # local navigation rather than ending up orphaned. Apply
            # the 1-cell overlap on a *clone* of the DM and use that
            # clone solely for the navigation kdtree / in-cell test.
            # The solver / FE assembly DM stays non-overlapped — PETSc
            # FE assembly with overlap double-counts contributions at
            # the partition seam via LocalToGlobal+ADD_VALUES, breaking
            # accuracy. Volume meshes don't enter this branch.
            if uw.mpi.size > 1 and self.dm.getDimension() != self.dm.getCoordinateDim():
                self._nav_dm = self.dm.clone()
                self._nav_dm.distributeOverlap(1)

            self.dm_hierarchy = [self.dm]
            self.dm_h = self.dm.clone()

        # This will be done anyway - the mesh maybe in a
        # partially adapted state

        if self.sf1 and self.sf0:
            self.sf = self.sf0.compose(self.sf1)
        else:
            self.sf = self.sf0  # could be None !

        if self.name is None:
            self.name = "mesh"
            self.dm.setName("uw_mesh")
        else:
            self.dm.setName(f"uw_{self.name}")

        if verbose and uw.mpi.rank == 0:
            print(
                f"PETSc dmplex set-up complete",
                flush=True,
            )

        self._install_coordinate_array(verbose)

        self._setup_symbolic_coordinates(coordinate_system_type)

        try:
            self.isSimplex = self.dm.isSimplex()
        except:
            self.isSimplex = simplex

        # Using WeakValueDictionary to prevent circular references
        self._vars = weakref.WeakValueDictionary()
        self._block_vars = {}

        # a list of equation systems that will
        # need to be rebuilt if the mesh coordinates change

        self._equation_systems_register = []

        # Operator on_remesh(ctx) hooks (SemiLagrangian / Lagrangian DDt,
        # solver-coupled history transfers). Stored as weakrefs so a
        # forgotten operator does not keep the mesh holding it alive.
        # The adapt op (smooth_mesh_interior / follow_metric)
        # fires these after the generic per-variable REMAP pass; see
        # discretisation/remesh.py.
        self._remesh_hooks = []

        # Capability gate for coordinate mutation. `_deform_mesh` (the raw
        # primitive that moves nodes WITHOUT the field/SL-history transfer)
        # is only permitted inside a sanctioned context: a remesh transaction
        # (`_in_remesh_transfer`, set by remesh_with_field_transfer) or a
        # `_coord_mutation()` scope (depth>0, opened by `deform`,
        # `ephemeral_coords`, and trusted internal movers). Outside those,
        # a bare call on a mesh that already carries variables/history raises
        # with a pointer to the public methods — so the field/history transfer
        # can never be silently skipped. See `deform`/`ephemeral_coords`.
        self._in_remesh_transfer = getattr(self, "_in_remesh_transfer", False)
        self._coord_mutation_depth = 0

        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        self._dm_initialized = False
        self._quadrature = False
        self._stale_lvec = True
        self._lvec = None
        self.petsc_fe = None

        # Rigid-body rotation null modes for this geometry.
        # Mesh factories override this for closed surfaces (annulus, sphere).
        # Each entry is a SymPy Matrix velocity field in mesh coordinates.
        self._nullspace_rotations = []

        self.degree = degree
        self.qdegree = qdegree

        # Populate the element information for this mesh. This is intended to be
        # human readable because the mesh is quite simple: either quads / tris in 2D
        # tetrahedra / hexahedra in 3D

        from dataclasses import dataclass

        @dataclass
        class ElementInfo:
            type: str
            entities: tuple
            face_entities: tuple

        if self.dm.isSimplex():
            if self.dim == 2:
                self._element = ElementInfo("triangle", (1, 3, 3), (0, 1, 2))
            else:
                self._element = ElementInfo("tetrahedron", (1, 4, 6, 4), (0, 1, 3, 3))
        else:
            if self.dim == 2:
                self._element = ElementInfo("quadrilateral", (1, 4, 4), (0, 1, 2))
            else:
                self._element = ElementInfo("hexahedron", (1, 6, 12, 8), (0, 1, 4, 4))

        # Initialize generic parameters property - mesh factories can set this
        self.parameters = None

        # Initialize DMInterpolation caching system
        from underworld3.function.dminterpolation_cache import DMInterpolationCache
        self._topology_version = 0  # Track mesh topology changes
        self._dminterpolation_cache = DMInterpolationCache(self, name=self.name)
        self.enable_dminterpolation_cache = True  # User can disable if needed

        if verbose and uw.mpi.rank == 0:
            print(
                f"PETSc spatial discretisation",
                flush=True,
            )

        # Navigation / coordinates etc
        self.nuke_coords_and_rebuild(verbose)

        # Apply a deferred FMG-hierarchy deformed-coordinate stamp (set in the
        # reload/splice branch above). The hierarchy + working dm were rebuilt
        # with reference coordinates; now the mesh is fully constructed we map
        # each local vertex to its canonical twin by an EXACT reference-coordinate
        # lookup and apply the saved deformed coordinates through _deform_mesh().
        _pending = getattr(self, "_pending_hierarchy_stamp", None)
        if _pending is not None:
            self._pending_hierarchy_stamp = None
            if os.environ.get("UW_NOSTAMP") != "1":
                from underworld3 import kdtree as _kdtree

                _ref_canon, _def_canon, _cdim = _pending
                _ref_local = numpy.ascontiguousarray(
                    self.dm.getCoordinatesLocal().array.reshape(-1, _cdim)
                )
                _tree = _kdtree.KDTree(numpy.ascontiguousarray(_ref_canon))
                _tree.build_index()
                _idx, _d2, _found = _tree.find_closest_point(_ref_local)
                if _d2.size and float(numpy.sqrt(_d2.max())) > 1.0e-8:
                    raise RuntimeError(
                        "FMG hierarchy restore: deformed-coordinate stamp is not "
                        f"exact (max lookup distance "
                        f"{float(numpy.sqrt(_d2.max())):.2e}); the reloaded "
                        "reference geometry does not match the saved fine."
                    )
                # Restoring saved deformed geometry; fields are reloaded
                # separately from the checkpoint, so this is a sanctioned
                # internal coordinate move (no live-state transfer needed).
                with self._coord_mutation():
                    self._deform_mesh(_def_canon[_idx].reshape(-1, _cdim))

        if verbose and uw.mpi.rank == 0:
            print(
                f"Populating mesh coordinates {coordinate_system_type}",
                flush=True,
            )

        ## Coordinate System

        if False:  # NATIVE coordinate systems deprecated
            self.vector = uw.maths.vector_calculus_cylindrical(
                mesh=self,
            )
        elif False:  # SPHERICAL_NATIVE deprecated
            self.vector = uw.maths.vector_calculus_spherical(
                mesh=self,
            )  ## Not yet complete or tested

        elif False:  # SPHERE_SURFACE_NATIVE deprecated
            self.vector = uw.maths.vector_calculus_spherical_surface2D_lonlat(
                mesh=self,
            )

        else:
            self.vector = uw.maths.vector_calculus(mesh=self)

        super().__init__()

        # Register with default model for orchestration and store reference
        self._model = uw.get_default_model()
        self._model._register_mesh(self)

    # --- Mesh.__init__ construction phases -------------------------------
    # The eight methods below are the named phases of mesh construction,
    # called in sequence from __init__ (pure code motion from the former
    # monolithic constructor). They are internal: each assumes the state
    # established by the preceding phases and is not safe to call again
    # on a fully constructed mesh.

    def _derive_length_scale_from_model(self, model):
        """Set the immutable length scale used for non-dimensionalisation.

        The length scale ties mesh coordinates to the model's reference
        quantities so that all spatial operators (grad, div, curl) share
        one consistent scaling. Priority order: ``domain_depth`` over
        ``length``; default 1.0 (no scaling) when the model defines
        neither.

        Parameters
        ----------
        model : uw.Model
            The orchestration model whose reference quantities define
            the scale.
        """
        self._length_scale = 1.0  # Default: no scaling
        self._length_units = (
            self.units if self.units else "dimensionless"
        )  # Same as coordinate units

        # Derive length scale from model reference quantities if available.
        # Priority order: domain_depth > length (first match wins).
        if hasattr(model, "_reference_quantities") and model._reference_quantities:
            for key in ("domain_depth", "length"):
                if key not in model._reference_quantities:
                    continue
                ref_qty = model._reference_quantities[key]
                # Convert to base units (SI: meters) for consistent scaling
                try:
                    base_qty = ref_qty.to_base_units()
                    self._length_scale = float(base_qty.magnitude)
                    self._length_units = str(base_qty.units)
                except (AttributeError, TypeError, ValueError):
                    # Sanctioned fallback rather than raise: a reference
                    # quantity stored as a plain number (no .to_base_units,
                    # AttributeError) or with a unit pint cannot reduce
                    # (pint errors subclass TypeError/AttributeError) still
                    # yields a usable scale in its OWN units — mesh
                    # construction must not fail on unit bookkeeping.
                    self._length_scale = float(ref_qty.magnitude)
                    self._length_units = (
                        str(ref_qty.units) if hasattr(ref_qty, "units") else "dimensionless"
                    )
                break

    def _load_dm_from_file(
        self,
        plex_or_meshfile,
        boundaries,
        coordinate_system_type,
        markVertices,
        useRegions,
        useMultipleTags,
        verbose,
        kwargs,
    ):
        """Build ``self.dm`` from a DMPlex object or a mesh file.

        Dispatches on the constructor's first argument: an existing
        ``PETSc.DMPlex`` is wrapped as-is; a ``.msh`` file is imported via
        gmsh; a ``.h5`` DMPlex checkpoint is reloaded together with any
        boundary / coordinate-system / region / ellipsoid metadata and the
        FMG coarse-hierarchy sidecar it carries.

        Returns
        -------
        tuple
            ``(name, boundaries, coordinate_system_type, regions)`` — the
            mesh name derived from the source, plus the metadata possibly
            restored from an ``.h5`` checkpoint (unchanged from the passed
            arguments otherwise).
        """
        comm = PETSc.COMM_WORLD
        regions = None  # May be set from h5 metadata or mesh generator

        if isinstance(plex_or_meshfile, PETSc.DMPlex):
            isDistributed = plex_or_meshfile.isDistributed()
            if verbose and uw.mpi.rank == 0:
                print(
                    f"Constructing UW mesh from DMPlex object (distributed == {isDistributed})",
                    flush=True,
                )
            if verbose:
                plex_or_meshfile.view()

            name = "plexmesh"
            self.dm = plex_or_meshfile
            self.sf0 = None  # Should we build one ?

            # Don't set from options — don't want to redistribute the dm
            # or change any settings as this should be left to the user

        else:
            comm = kwargs.get("comm", PETSc.COMM_WORLD)
            name = plex_or_meshfile
            basename, ext = os.path.splitext(plex_or_meshfile)

            # Note: should be able to handle a .geo as well on this pathway
            if ext.lower() == ".msh":
                if verbose and uw.mpi.rank == 0:
                    print(f"Constructing UW mesh from gmsh {plex_or_meshfile}", flush=True)

                self.sf0, self.dm = _from_gmsh(
                    plex_or_meshfile,
                    comm,
                    markVertices=markVertices,
                    useRegions=useRegions,
                    useMultipleTags=useMultipleTags,
                )

            elif ext.lower() == ".h5":
                if verbose and uw.mpi.rank == 0:
                    print(
                        f"Constructing UW mesh from DMPlex h5 file {plex_or_meshfile}",
                        flush=True,
                    )
                self.sf0, self.dm = _from_plexh5(plex_or_meshfile, PETSc.COMM_WORLD, return_sf=True)

                ## We can check if there is boundary metadata in the h5 file and we
                ## should use it if it is present.

                import h5py, json

                f = h5py.File(plex_or_meshfile, "r")

                try:
                    json_str = f["metadata"].attrs["boundaries"]
                    bdr_dict = json.loads(json_str)
                    boundaries = Enum("Boundaries", bdr_dict)
                except KeyError:
                    pass

                try:
                    json_str = f["metadata"].attrs["coordinate_system_type"]
                    coord_type_dict = json.loads(json_str)
                    coordinate_system_type = CoordinateSystemType(coord_type_dict["value"])
                except KeyError:
                    pass

                regions = None
                try:
                    json_str = f["metadata"].attrs["regions"]
                    rgn_dict = json.loads(json_str)
                    regions = Enum("Regions", rgn_dict)
                except KeyError:
                    pass

                # Restore ellipsoid with quantities for geographic meshes
                self._checkpoint_ellipsoid = None
                try:
                    json_str = f["metadata"].attrs["ellipsoid"]
                    ellipsoid_raw = json.loads(json_str)
                    for k, v in ellipsoid_raw.items():
                        if isinstance(v, dict) and "value" in v and "unit" in v:
                            ellipsoid_raw[k] = uw.quantity(v["value"], v["unit"])
                    self._checkpoint_ellipsoid = ellipsoid_raw
                except KeyError:
                    pass

                f.close()

                # Restore the geometric-multigrid (FMG) coarse hierarchy from the
                # sidecar, if present. We keep the *undistributed* coarsest level
                # and the refinement count; the hierarchy is rebuilt by
                # _splice_hierarchy_from_sidecar exactly the way a fresh
                # refinement mesh is built — distribute the coarsest, then
                # refine() locally — which is robust in serial and at any
                # parallel decomposition.
                self._sidecar_coarsest = None
                try:
                    with h5py.File(plex_or_meshfile, "r") as fh:
                        n_coarse = int(
                            fh["metadata"].attrs.get("hierarchy_coarse_levels", 0)
                        )
                except (KeyError, OSError):
                    n_coarse = 0
                if n_coarse > 0:
                    sidecar = _hierarchy_sidecar_name(plex_or_meshfile)
                    if os.path.isfile(sidecar):
                        self._sidecar_coarsest = _from_plexh5(
                            sidecar, PETSc.COMM_WORLD
                        )
                        self._sidecar_n_coarse = n_coarse
                        self._sidecar_meshfile = plex_or_meshfile

                # Do not call setFromOptions() here. DMPlexTopologyLoad()
                # returns the topology SF needed to reload checkpoint fields.
                # setFromOptions() can repartition/reorder the DM before UW
                # composes that SF with any later redistribution SF, leaving
                # checkpoint field reloads mapped to stale point numbering.

            else:
                raise RuntimeError(
                    "Mesh file %s has unknown format '%s'." % (plex_or_meshfile, ext[1:])
                )

        return name, boundaries, coordinate_system_type, regions

    def _patch_boundary_enum(
        self, plex_or_meshfile, boundaries, boundary_normals, regions, filename
    ):
        """Extend the boundary enum and rebuild the DM's boundary labels.

        Patches the user-supplied boundary enum with the members every UW
        mesh needs (``Null_Boundary`` — every vertex, value 666 — and
        ``All_Boundaries``, value 1001), records the boundary / region
        metadata attributes on the mesh, rebuilds named boundary labels for
        wrapped DMPlex imports that only expose stacked Gmsh label sets,
        and builds the ``Null_Boundary`` and stacked ``UW_Boundaries``
        labels used by the boundary-condition machinery.
        """
        ## Patch up the boundaries to include the additional
        ## definitions that we do / might need. Note: the
        ## extend_enum decorator will replace existing members with
        ## the new ones.

        if boundaries is None:

            class replacement_boundaries(Enum):
                Null_Boundary = 666
                All_Boundaries = 1001

            boundaries = replacement_boundaries
        else:

            @extend_enum([boundaries])
            class replacement_boundaries(Enum):
                Null_Boundary = 666
                All_Boundaries = 1001

            boundaries = replacement_boundaries

        self.filename = filename
        self.boundaries = boundaries
        # Bounding-surface objects (tangent-slip + restore), keyed by boundary
        # label. SEPARATE from self.boundaries (the persisted gmsh/DMPlex
        # labelling, untouched). Populated by analytic-geometry constructors;
        # see docs/developer/design/boundary-slip-strategy.md.
        self._bounding_surfaces = {}
        self.boundary_normals = boundary_normals
        self.regions = regions
        self.parent = None       # Set by extract_region()/adapt() for children
        self.subpoint_is = None  # IS mapping submesh points -> parent points
        # How this mesh was derived from ``self.parent`` (the parent/child DAG):
        #   None        -- a base mesh (no parent)
        #   "submesh"   -- a subset extracted via extract_region (DOFs coincide)
        #   "refinement"-- an SBR adapt-on-top child (parent ⊂ child, DOFs differ)
        # copy_into/restrict/prolongate dispatch on this kind.
        self._relationship_kind = None
        # Mesh-owned custom-P geometric-MG hierarchy (set on refinement children
        # by adapt()): the static coarse-mesh tail every solver on this mesh
        # consumes. ``None`` => no mesh-owned hierarchy (solvers fall back to
        # their own preconditioner / an explicit set_custom_fmg).
        self._custom_mg_coarse_meshes = None
        self._custom_mg_builder = "barycentric"

        # Wrapped imported DMPlex meshes may only expose generic Gmsh labels
        # such as "Face Sets". Rebuild named boundary labels from those sets so
        # boundary APIs behave the same way as the standard Mesh(mesh_file) path.
        if isinstance(plex_or_meshfile, PETSc.DMPlex) and self.boundaries is not None:
            has_named_boundary_labels = any(self.dm.getLabel(b.name) for b in self.boundaries)
            if not has_named_boundary_labels:
                for stacked_label_name in ("Face Sets", "Edge Sets", "Vertex Sets"):
                    if self.dm.getLabel(stacked_label_name):
                        uw.adaptivity._dm_unstack_bcs(self.dm, self.boundaries, stacked_label_name)
                        break

        # Null_Boundary marks EVERY vertex with the reserved value 666 —
        # the catch-all stratum used when a condition applies mesh-wide.
        # Note: getStratumIS(0) on the "depth" label fetches the depth-0
        # points, i.e. VERTICES (the old name "all_edges" was wrong).
        depth_label = self.dm.getLabel("depth")
        self.dm.createLabel("Null_Boundary")
        null_boundary_label = self.dm.getLabel("Null_Boundary")
        if depth_label and null_boundary_label:
            vertex_stratum_is = depth_label.getStratumIS(0)
            if vertex_stratum_is:
                null_boundary_label.setStratumIS(
                    boundaries.Null_Boundary.value, vertex_stratum_is
                )

        ## --- UW_Boundaries label
        if self.boundaries is not None:

            self.dm.removeLabel("UW_Boundaries")
            uw.mpi.barrier()
            self.dm.createLabel("UW_Boundaries")

            stacked_bc_label = self.dm.getLabel("UW_Boundaries")

            for b in self.boundaries:
                bc_label_name = b.name
                label = self.dm.getLabel(bc_label_name)

                if label:
                    label_is = label.getStratumIS(b.value)

                    # Load this up on the stacked BC label
                    if label_is:
                        stacked_bc_label.setStratumIS(b.value, label_is)

            uw.mpi.barrier()

    def _splice_hierarchy_from_sidecar(self):
        """Rebuild the FMG hierarchy of a reloaded mesh from its sidecar.

        Reloaded mesh with a persisted FMG hierarchy: rebuild the geometric
        multigrid hierarchy exactly as a fresh refinement mesh does —
        distribute the coarsest level, then refine() it locally. refine()
        never moves points across the decomposition (only distribute()
        does), so a coarse cell and all of its children are guaranteed
        co-resident on one rank — precisely what the nested interpolator
        needs, and robust at any np. (Independently distributing pre-built
        levels misaligns at uneven np and aborts inside
        DMPlexComputeInterpolatorNested.)

        The refine-built fine carries *reference* coordinates, so the saved
        (deformed) fine coordinates are stamped onto it afterwards. The
        reference geometry rebuilt here (refine-of-coarsest) is bit-identical
        to the one at save time, so every distributed fine vertex maps to
        exactly one canonical vertex by an *exact* nearest-reference lookup,
        and the deformed value is read straight from the saved fine's
        canonical-ordered coordinates. (See the checkpoint-hierarchy design
        note.)
        """
        n_coarse = self._sidecar_n_coarse
        cdim = self._sidecar_coarsest.getCoordinateDim()

        # --- canonical (reference, deformed) coordinate pair, on rank 0 ---
        # BOTH arrays are built rank-locally on COMM_SELF so they share ONE
        # canonical ordering (serial .h5 load order == serial refine order,
        # verified). That shared ordering is what makes deformed_canon[k] and
        # reference_canon[k] the *same physical vertex* — the stamp pairs them
        # by that index. (Reading the deformed coords from the COMM_WORLD
        # undistributed load instead can use a different vertex ordering and
        # silently scrambles the stamp.) COMM_SELF work is rank-local, so it
        # cannot perturb the collective distribute of self._sidecar_coarsest.
        if uw.mpi.rank == 0:
            _df = _from_plexh5(self._sidecar_meshfile, PETSc.COMM_SELF)
            deformed_canon = (
                _df.getCoordinatesLocal().array.reshape(-1, cdim).copy()
            )
            _cs = _from_plexh5(
                _hierarchy_sidecar_name(self._sidecar_meshfile),
                PETSc.COMM_SELF,
            )
            for _ in range(n_coarse):
                _cs.setRefinementUniform()
                _cs = _cs.refine()
            reference_canon = (
                _cs.getCoordinatesLocal().array.reshape(-1, cdim).copy()
            )
        else:
            deformed_canon = None
            reference_canon = None
        deformed_canon = uw.mpi.comm.bcast(deformed_canon, root=0)
        reference_canon = uw.mpi.comm.bcast(reference_canon, root=0)

        # --- aligned hierarchy: distribute the coarsest, then local refine,
        #     EXACTLY as _build_refined_hierarchy does (proven to
        #     build a correct nested geometric-MG hierarchy at any np):
        #     setRefinementUniform() on the base before distribute(), then a
        #     plain refine() loop. refine() flags the regular refinement
        #     itself — setting it by hand on a non-uniformly-refined DM
        #     instead corrupts the nested interpolator and the solve diverges.
        self._sidecar_coarsest.setRefinementUniform()
        if not self._sidecar_coarsest.isDistributed():
            self.sf1 = self._sidecar_coarsest.distribute()
        self.dm_hierarchy = [self._sidecar_coarsest]
        for i in range(n_coarse):
            dm_refined = self.dm_hierarchy[i].refine()
            dm_refined.setCoarseDM(self.dm_hierarchy[i])
            self.dm_hierarchy.append(dm_refined)

        self.dm_h = self.dm_hierarchy[-1]
        self.dm_h.setName("uw_hierarchical_dm")

        # Working dm is a link-free clone of the finest level (mirrors the
        # refinement branch). It must NOT carry a coarse-DM link or
        # mesh.update_lvec()'s createFieldDecomposition recurses into the
        # 0-field coarse levels and fails.
        self.dm = self.dm_h.clone()

        # Defer the deformed-coordinate stamp: the hierarchy and working dm
        # are built with REFERENCE coordinates here, and the saved deformed
        # coordinates are applied through the normal _deform_mesh() path at
        # the END of __init__ (once self._coords and the rebuild machinery
        # exist). A raw setCoordinatesLocal() at this point — before the
        # coordinate cache/callbacks are set up — leaves the mesh in an
        # inconsistent state and the geometric multigrid solve diverges. Stash
        # the canonical (reference, deformed) pair for the deferred apply.
        self._pending_hierarchy_stamp = (reference_canon, deformed_canon, cdim)
        self._sidecar_coarsest = None

    def _build_refined_hierarchy(self, refinement, refinement_callback):
        """Distribute the DM and build a uniformly refined FMG hierarchy.

        Each level is a plain ``refine()`` of the previous one (never
        ``refineHierarchy`` — the per-level loop lets the optional
        ``refinement_callback`` repair each refined mesh, e.g. snapping
        new boundary vertices back onto a curved surface). The working
        ``self.dm`` becomes a link-free clone of the finest level.
        """
        self.dm.setRefinementUniform()

        if not self.dm.isDistributed():
            self.sf1 = self.dm.distribute()

        # This is preferable to the refineHierarchy call
        # because we can repair the refined mesh at each
        # step along the way

        self.dm_hierarchy = [self.dm]
        for i in range(refinement):
            dm_refined = self.dm_hierarchy[i].refine()
            dm_refined.setCoarseDM(self.dm_hierarchy[i])

            if callable(refinement_callback):
                refinement_callback(dm_refined)

            self.dm_hierarchy.append(dm_refined)

        self.dm_h = self.dm_hierarchy[-1]
        self.dm_h.setName("uw_hierarchical_dm")

        # Is this needed here, after the above calls ?
        if callable(refinement_callback):
            for dm in self.dm_hierarchy:
                refinement_callback(dm)

        # Single level equivalent dm (needed for aux vars ?? Check this - LM)
        self.dm = self.dm_h.clone()

    def _build_coarsened_hierarchy(self, coarsening):
        """Distribute the DM and build a coarsened FMG hierarchy.

        Builds ``coarsening`` successively coarser levels below the input
        mesh, then reverses the list so the coarsest mesh is first —
        consistent with the ordering the refinement path produces. The
        working ``self.dm`` becomes a link-free clone of the finest level.

        Note: coarsening callbacks are not supported — the coarse meshes
        do not carry the boundary labels a callback would need.
        """
        # Does this have any effect on a coarsening strategy ?
        self.dm.setRefinementUniform()

        if not self.dm.isDistributed():
            self.sf1 = self.dm.distribute()

        self.dm_hierarchy = [self.dm]
        for i in range(coarsening):
            dm_coarsened = self.dm_hierarchy[i].coarsen()
            self.dm_hierarchy[i].setCoarseDM(dm_coarsened)
            self.dm_hierarchy.append(dm_coarsened)

        # Coarsest mesh should be first in the hierarchy to be consistent
        # with the way we manage refinements
        self.dm_hierarchy.reverse()

        self.dm_h = self.dm_hierarchy[-1]
        self.dm_h.setName("uw_hierarchical_dm")

        # Single level equivalent dm (needed for aux vars ?? Check this - LM)
        self.dm = self.dm_h.clone()

    def _install_coordinate_array(self, verbose):
        """Expose the DM's coordinates as the mesh's canonical coord array.

        Validates the raw coordinate buffer against ``cdim``, wraps it as
        an ``NDArray_With_Callback`` (``self._coords``) whose setter
        callback routes every user coordinate write through the full
        ``_deform_mesh`` rebuild, and builds the navigation-only coord
        view used by the point-location indices on manifold meshes.
        """
        # Validate that the DM's coordinate array is consistent with the
        # coordinate dimension before reshaping. A mismatch here almost always
        # means a STALE or CORRUPT cached mesh file — e.g. a cached .h5 that was
        # written as a lower-dimension mesh (the classic symptom of a leaked
        # ``dm_plex_gmsh_spacedim`` during generation; see _from_gmsh). Raise a
        # clear, actionable error instead of an opaque numpy "cannot reshape"
        # failure.
        _coord_size = self.dm.getCoordinatesLocal().array.size
        if self.cdim and _coord_size % self.cdim != 0:
            _src = f" ('{self.filename}')" if getattr(self, "filename", None) else ""
            raise RuntimeError(
                f"Mesh coordinate array (size {_coord_size}) is not divisible by "
                f"the coordinate dimension cdim={self.cdim}, so it cannot be a "
                f"valid set of {self.cdim}-D node coordinates. This usually "
                f"indicates a stale or corrupt cached mesh file{_src} (e.g. a "
                f"cache written at a different space dimension). Delete the "
                f"cached '.meshes/*.msh' and '.msh.h5' for this mesh and "
                f"regenerate."
            )

        # Expose mesh points through special numpy array class with a callback
        # on all setter operations (see _mesh_coords_update_callback).
        self._install_coords_array(verbose=verbose)

        # Navigation-only coord view. On manifold meshes the nav DM is
        # a 1-cell-overlap clone with extra ghost vertices; navigation
        # indices (kdtree, in-cell control points) read from these
        # coords. On volume meshes _nav_dm is None and we reuse the
        # main coords.
        if self._nav_dm is not None:
            # distributeOverlap grew the coordinate section to the ghost
            # vertices but left their coordinates unscattered, so build a
            # section-consistent local array (incl. ghost rows) here — a plain
            # getCoordinatesLocal() is short by the ghost rows (issue #360).
            self._nav_coords = self._navigation_coords_from_dm(self._nav_dm)
        else:
            self._nav_coords = self._coords

    def _install_coords_array(self, verbose=False):
        """(Re)wrap the DM's local coordinates as the canonical coord array.

        Creates ``self._coords`` as an ``NDArray_With_Callback`` view of
        the DM's local coordinate buffer and attaches the shared
        module-level :func:`_mesh_coords_update_callback`. Every site
        that replaces the coordinate buffer uses this ONE installer —
        initial construction (:meth:`_install_coordinate_array`), submesh
        re-extraction (:meth:`_re_extract_from_parent`) and adaptation
        (:meth:`adapt`) — so the callback's teardown guard and identity
        gate are identical everywhere.
        """
        self._coords_callback_verbose = bool(verbose)
        self._coords = uw.utilities.NDArray_With_Callback(
            numpy.ndarray.view(self.dm.getCoordinatesLocal().array.reshape(-1, self.cdim)),
            owner=self,
        )
        # Canonical registration: the guard keeps derived views/copies from
        # firing a full-mesh deform (#376-class), and — because the deform
        # is COLLECTIVE — the mesh joins the synchronised-update flush
        # registry so coordinate writes inside uw.synchronised_array_update
        # defer to the single rank-agreed flush instead of replaying a
        # rank-local deform at exit.
        self._coords.add_canonical_callback(_mesh_coords_update_callback)
        if not hasattr(self, "_collective_flush_id"):
            # Register once per mesh: re-installs (submesh re-extraction,
            # adaptation) replace the array, not the mesh's flush identity.
            self._collective_flush_id = register_collective_flush(self)

    def _deferred_canonical_flush(self):
        """Collective flush target for ``uw.synchronised_array_update``.

        Coordinate writes made inside the context land in the canonical
        array immediately; the deform they imply runs here, once, on every
        rank in the agreed flush order.
        """
        fire_canonical_callbacks(self._coords)

    def _setup_symbolic_coordinates(self, coordinate_system_type):
        """Create the sympy coordinate systems and their JIT code bindings.

        Builds the generic Cartesian ``CoordSys3D`` (``N``, mapped to
        ``petsc_x`` in generated kernels) and the boundary-normal system
        (``Gamma``, mapped to ``petsc_n`` in surface integrals), attaches
        the mesh's natural :class:`CoordinateSystem`, the time symbol
        ``mesh.t`` (``petsc_t``), and patches unit awareness onto the
        coordinate symbols.
        """
        # Set sympy constructs. First a generic, symbolic, Cartesian coordinate system
        # A unique set of vectors / names for each mesh instance
        #

        self.CoordinateSystemType = coordinate_system_type

        from sympy.vector import CoordSys3D

        self._N = CoordSys3D(f"N")

        # Tidy some of this printing without changing the
        # underlying vector names (as these are part of the code generation system)

        self._N.x._latex_form = r"\mathrm{\xi_0}"
        self._N.y._latex_form = r"\mathrm{\xi_1}"
        self._N.z._latex_form = r"\mathrm{\xi_2}"
        self._N.i._latex_form = r"\mathbf{\hat{\mathbf{e}}_0}"
        self._N.j._latex_form = r"\mathbf{\hat{\mathbf{e}}_1}"
        self._N.k._latex_form = r"\mathbf{\hat{\mathbf{e}}_2}"

        self._Gamma = CoordSys3D(r"\Gamma")

        self._Gamma.x._latex_form = r"\Gamma_x"
        self._Gamma.y._latex_form = r"\Gamma_y"
        self._Gamma.z._latex_form = r"\Gamma_z"

        # Now add the appropriate coordinate system for the mesh's natural geometry
        # This step will usually over-write the defaults we just defined
        # For geographic meshes loaded from checkpoint, pre-set the ellipsoid
        # so the CoordinateSystem __init__ picks it up.
        if hasattr(self, "_checkpoint_ellipsoid") and self._checkpoint_ellipsoid is not None:
            self._checkpoint_ellipsoid_pending = self._checkpoint_ellipsoid

        self._CoordinateSystem = CoordinateSystem(self, coordinate_system_type)

        # This was in the _jit extension but ... if
        # not here then the tests fail sometimes (caching ?)

        self._N.x._ccodestr = "petsc_x[0]"
        self._N.y._ccodestr = "petsc_x[1]"
        self._N.z._ccodestr = "petsc_x[2]"

        # Surface integrals also have normal vector information as petsc_n

        self._Gamma.x._ccodestr = "petsc_n[0]"
        self._Gamma.y._ccodestr = "petsc_n[1]"
        self._Gamma.z._ccodestr = "petsc_n[2]"

        # Time coordinate — PETSc passes this as petsc_t to all pointwise
        # functions. Solvers set dm.time before each solve via solve(time=t).
        # Users reference it as mesh.t in expressions (e.g. V0 * sympy.sin(omega * mesh.t))
        from ..utilities.unit_aware_coordinates import TimeSymbol

        self._t = TimeSymbol("t")
        self._t._units = None  # patched below by _patch_time_units

        # Add unit awareness to coordinate symbols if mesh has units or model has scales
        from ..utilities.unit_aware_coordinates import patch_coordinate_units

        patch_coordinate_units(self)

    @property
    def dim(self) -> int:
        """Topological dimension of the mesh.

        Returns
        -------
        int
            The mesh dimension (2 for 2D, 3 for 3D).
        """
        return self.dm.getDimension()

    @property
    def cdim(self) -> int:
        """Coordinate dimension (embedding space dimension).

        For most meshes, ``cdim == dim``. For surface meshes embedded in 3D
        (e.g., a 2D spherical shell), ``dim=2`` but ``cdim=3``.

        Returns
        -------
        int
            The coordinate dimension.
        """
        return self.dm.getCoordinateDim()

    @property
    def element(self) -> dict:
        """Element type information for the mesh.

        Contains details about the finite element discretization including
        cell type, polynomial degree, and quadrature order.

        Returns
        -------
        dict
            Element information dictionary.

        Notes
        -----
        UW3 does not support mixed-element meshes; this applies uniformly
        to all cells.
        """

        return self._element

    @property
    def length_scale(self) -> float:
        """
        Length scale for non-dimensionalization.

        This property is IMMUTABLE after mesh creation to ensure synchronization
        with all spatial operators (gradient, divergence, curl, etc.).

        The length scale is derived from model reference quantities at mesh creation:
        - Priority 1: `domain_depth` from `model.set_reference_quantities()`
        - Priority 2: `length` from `model.set_reference_quantities()`
        - Default: 1.0 (no scaling)

        Returns
        -------
        float
            Length scale value for non-dimensionalization

        Examples
        --------
        >>> model.set_reference_quantities(domain_depth=uw.quantity(100, "km"))
        >>> mesh = uw.meshing.UnstructuredSimplexBox(...)
        >>> mesh.length_scale
        100000.0  # meters

        See Also
        --------
        length_units : Units string for length scale
        """
        return self._length_scale

    @property
    def length_units(self) -> str:
        """
        Unit string for the length scale.

        Returns
        -------
        str
            Units for the length scale (e.g., "meter", "kilometer")

        Examples
        --------
        >>> mesh.length_units
        'kilometer'
        """
        return self._length_units

    def quality(self, per_cell=False):
        r"""Cell-quality diagnostics relevant to FE / solver conditioning.

        Bulk volume ratios (min/mean) hide the handful of
        near-degenerate cells that nonetheless dominate
        stiffness-matrix conditioning — a Stokes / saddle-point
        solve line-search-fails on the *worst* element, not the
        mean. This reports the tail metrics that actually predict
        that. For a 2-D simplex (triangle) mesh, per cell:

        * shape quality ``q = 4√3·A / Σℓ²``  (1 = equilateral,
          → 0 = sliver; folds skew + stretch into one number)
        * largest interior angle (→ 180° is the conditioning killer)
        * aspect ratio ``ℓ_max² / (2A)``  (longest edge / shortest
          altitude)
        * neighbour size-jump (adjacent-cell area ratio — the mesh
          gradation the solver actually sees)

        The conditioning-relevant numbers are the *worst* cell
        (``q_min``, ``angle_max_deg``, ``aspect_max``) and the
        poor-cell counts, not the means. Non-2-D-simplex meshes get
        the dimension-agnostic cell-volume-spread subset only.

        Parameters
        ----------
        per_cell : bool, default False
            Also return per-cell arrays (``q``, ``angle_deg``,
            ``aspect``, ``volume``) under ``"per_cell"`` — for
            plotting or locating the bad cells.

        Returns
        -------
        dict
            Aggregate + tail stats. Headline scalars (min/max/counts)
            are MPI-reduced so they are correct in parallel;
            percentiles and the neighbour size-jump are rank-local
            estimates (exact in serial — the convention for the
            mesh-redistribution tooling).

        Examples
        --------
        >>> q = mesh.quality()
        >>> q["q_min"], q["n_q_lt_0p3"], q["aspect_max"]
        >>> mesh.quality(per_cell=True)["per_cell"]["q"]  # to plot
        """
        import numpy as np

        dm = self.dm
        cdim = self.cdim
        cStart, cEnd = dm.getHeightStratum(0)
        pStart, pEnd = dm.getDepthStratum(0)
        vertex_coords = np.asarray(
            dm.getCoordinatesLocal().array).reshape(-1, cdim)

        def _reduce(val, op):
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                return uw.mpi.comm.allreduce(
                    val, op=getattr(_MPI, op))
            return val

        tri_vertex_lists = []
        is_simplex2d = cdim == 2
        if is_simplex2d:
            for cell_id in range(cStart, cEnd):
                closure_points = dm.getTransitiveClosure(cell_id)[0]
                cell_vertices = [p - pStart for p in closure_points
                                 if pStart <= p < pEnd]
                if len(cell_vertices) != 3:
                    is_simplex2d = False
                    break
                tri_vertex_lists.append(cell_vertices)

        if not is_simplex2d or not tri_vertex_lists:
            try:
                volume = np.abs(np.array(
                    [dm.computeCellGeometryFVM(cell_id)[0]
                     for cell_id in range(cStart, cEnd)]))
            except Exception:
                volume = np.array([1.0])
            if not volume.size:
                volume = np.array([1.0])
            n_cells = _reduce(int(volume.size), "SUM")
            vol_min = _reduce(float(volume.min()), "MIN")
            vol_sum = _reduce(float(volume.sum()), "SUM")
            metrics = dict(
                n_cells=n_cells, element="non-2D-simplex",
                vol_min_over_mean=vol_min / (vol_sum / max(n_cells, 1)),
                shape_metrics=None,
                note="shape quality / angle / aspect need a 2-D "
                     "triangle mesh; only volume spread reported")
            if per_cell:
                metrics["per_cell"] = dict(volume=volume)
            return metrics

        tri = np.asarray(tri_vertex_lists, dtype=np.int64)
        v0, v1, v2 = (vertex_coords[tri[:, 0]],
                      vertex_coords[tri[:, 1]],
                      vertex_coords[tri[:, 2]])
        # Triangle side lengths, each named for the vertex it is opposite
        edge_a = np.linalg.norm(v1 - v2, axis=1)
        edge_b = np.linalg.norm(v2 - v0, axis=1)
        edge_c = np.linalg.norm(v0 - v1, axis=1)
        # 2-D triangle area from the z-component of the edge cross product
        # (numpy 2.0 removed the 2-D np.cross that returned this scalar).
        _e1, _e2 = v1 - v0, v2 - v0
        _cross_z = _e1[:, 0] * _e2[:, 1] - _e1[:, 1] * _e2[:, 0]
        area = np.maximum(0.5 * np.abs(_cross_z), 1.0e-300)
        shape_q = 4.0 * np.sqrt(3.0) * area / (
            edge_a * edge_a + edge_b * edge_b + edge_c * edge_c)

        def _angle_deg(opposite, side1, side2):
            # Interior angle opposite the side `opposite`, law of cosines
            return np.degrees(np.arccos(np.clip(
                (side1 * side1 + side2 * side2 - opposite * opposite)
                / (2.0 * side1 * side2),
                -1.0, 1.0)))
        largest_angle = np.maximum.reduce(
            [_angle_deg(edge_a, edge_b, edge_c),
             _angle_deg(edge_b, edge_c, edge_a),
             _angle_deg(edge_c, edge_a, edge_b)])
        longest_edge = np.maximum.reduce([edge_a, edge_b, edge_c])
        aspect = longest_edge * longest_edge / (2.0 * area)
        rel_area = area / area.mean()

        # Neighbour size-jump: map each (undirected) edge to the triangles
        # sharing it; interior edges (exactly two triangles) contribute the
        # adjacent-cell area ratio — the gradation the solver actually sees.
        edge_to_tris = {}
        for tri_idx, (i, j, k) in enumerate(tri):
            for u, w in ((i, j), (j, k), (k, i)):
                edge_to_tris.setdefault((min(u, w), max(u, w)),
                                        []).append(tri_idx)
        size_jump = np.array([max(area[t]) / min(area[t])
                              for t in edge_to_tris.values() if len(t) == 2]
                             or [1.0])

        n_cells = _reduce(int(tri.shape[0]), "SUM")
        q_sum = _reduce(float(shape_q.sum()), "SUM")
        area_sum = _reduce(float(area.sum()), "SUM")
        metrics = dict(
            n_cells=n_cells, element="2D-simplex",
            q_min=_reduce(float(shape_q.min()), "MIN"),
            q_mean=q_sum / max(n_cells, 1),
            q_p01=float(np.percentile(shape_q, 1)),
            q_p05=float(np.percentile(shape_q, 5)),
            n_q_lt_0p3=_reduce(int((shape_q < 0.3).sum()), "SUM"),
            n_q_lt_0p2=_reduce(int((shape_q < 0.2).sum()), "SUM"),
            angle_max_deg=_reduce(float(largest_angle.max()), "MAX"),
            n_angle_gt_150=_reduce(int((largest_angle > 150).sum()), "SUM"),
            n_angle_gt_165=_reduce(int((largest_angle > 165).sum()), "SUM"),
            aspect_max=_reduce(float(aspect.max()), "MAX"),
            aspect_p99=float(np.percentile(aspect, 99)),
            sizejump_max=float(size_jump.max()),
            sizejump_p99=float(np.percentile(size_jump, 99)),
            n_big_thin=_reduce(
                int(((rel_area > 2.0) & (aspect > 4.0)).sum()), "SUM"),
            vol_min_over_mean=(_reduce(float(area.min()), "MIN")
                               / (area_sum / max(n_cells, 1))))
        if per_cell:
            metrics["per_cell"] = dict(
                q=shape_q, angle_deg=largest_angle, aspect=aspect, volume=area)
        return metrics

    def _print_variable_table(self):
        """Print the variable name / components / degree / type table (rank 0)."""
        if len(self.vars) > 0:
            uw.pprint(f"| Variable Name       | component | degree |     type        |")
            uw.pprint(f"| ---------------------------------------------------------- |")
            for vname in self.vars.keys():
                v = self.vars[vname]
                uw.pprint(
                    f"| {v.clean_name:<20}|{v.num_components:^10} |{v.degree:^7} | {v.vtype.name:^15} |"
                )

            uw.pprint(f"| ---------------------------------------------------------- |")
            uw.pprint("\n")
        else:
            uw.pprint(f"No variables are defined on the mesh\n")

    def _print_boundary_table(self, with_sizes=False):
        """Print the boundary-label table (rank 0).

        ``with_sizes=True`` adds the min/max per-rank stratum sizes; those
        gathers are collective, so in that mode every rank must call this
        together.
        """
        import numpy as np

        if len(self.boundaries) > 0:
            if with_sizes:
                uw.pprint(f"| Boundary Name            | ID    | Min Size | Max Size |")
                uw.pprint(f"| ------------------------------------------------------ |")
            else:
                uw.pprint(f"| Boundary Name            | ID    |")
                uw.pprint(f"| -------------------------------- |")
        else:
            uw.pprint(f"No boundary labels are defined on the mesh\n")

        i = 0
        for bd in self.boundaries:
            l = self.dm.getLabel(bd.name)
            if l:
                i = l.getStratumSize(bd.value)
            else:
                i = 0

            if with_sizes:
                ii = uw.utilities.gather_data(np.array([i]), dtype="int")
                uw.pprint(f"| {bd.name:<20}     | {bd.value:<5} | {ii.min():<8} | {ii.max():<8} |")
            else:
                uw.pprint(f"| {bd.name:<20}     | {bd.value:<5} |")

        if with_sizes:
            # TODO(DESIGN): this All_Boundaries row re-gathers the FINAL loop
            # iteration's stratum size (quirk preserved from the original
            # inline table) instead of querying the All_Boundaries label.
            ii = uw.utilities.gather_data(np.array([i]), dtype="int")
            uw.pprint(f"| {'All_Boundaries':<20}     | 1001  | {ii.min():<8} | {ii.max():<8} |")
        else:
            uw.pprint(f"| {'All_Boundaries':<20}     | 1001  |")

        ## UW_Boundaries (stacked label): total of the per-boundary strata
        l = self.dm.getLabel("UW_Boundaries")
        i = 0
        if l:
            for bd in self.boundaries:
                i += l.getStratumSize(bd.value)

        if with_sizes:
            ii = uw.utilities.gather_data(np.array([i]), dtype="int")
            uw.pprint(f"| {'UW_Boundaries':<20}     | --    | {ii.min():<8} | {ii.max():<8} |")
            uw.pprint(f"| ------------------------------------------------------ |")
        else:
            uw.pprint(f"| {'UW_Boundaries':<20}     | --    |")
            uw.pprint(f"| -------------------------------- |")

        uw.pprint("\n")

    def view(self, level=0):
        """
        Displays mesh information at different levels.

        Parameters
        ----------
        level : int (0 default)
            The display level.
            0, for basic mesh information (variables and boundaries), while level=1 displays detailed mesh information (including PETSc information)
        """

        import numpy as np

        if level == 0:
            uw.pprint(f"\n")
            uw.pprint(f"Mesh # {self.instance}: {self.name}\n")

            # Display coordinate units if set
            if hasattr(self, "units") and self.units is not None:
                uw.pprint(f"Coordinate units: {self.units}\n")
                uw.pprint(f"  Access unit-aware coordinates via: mesh.X.coords\n")
                uw.pprint(f"  Query units with: uw.get_units(mesh.X.coords)\n")

            # Display length scale for non-dimensionalization
            if hasattr(self, "_length_scale"):
                if self._length_scale != 1.0:
                    uw.pprint(
                        f"Length scale (non-dimensionalization): {self._length_scale} {self._length_units}\n"
                    )
                else:
                    uw.pprint(f"Length scale: 1.0 (no scaling)\n")

            # Display coordinate system information
            coord_sys = self.CoordinateSystem
            coord_type = coord_sys.coordinate_type
            uw.pprint(f"Coordinate system: {coord_type.name}\n")

            # Show available coordinate accessors
            accessors = ["mesh.X.coords (Cartesian)"]  # Always available
            if coord_sys._spherical_accessor is not None:
                if self.dim == 2:
                    accessors.append("mesh.X.spherical (r, θ)")
                else:
                    accessors.append("mesh.X.spherical (r, θ, φ)")
            if coord_sys._geo_accessor is not None:
                accessors.append("mesh.X.geo (lon, lat, depth)")

            uw.pprint(f"Coordinate access:\n")
            for acc in accessors:
                uw.pprint(f"  • {acc}\n")

            # Only if notebook and serial
            if uw.is_notebook and uw.mpi.size == 1:
                uw.visualisation.plot_mesh(self, window_size=(600, 400))

            # Total number of cells
            nstart, nend = self.dm.getHeightStratum(0)
            num_cells = nend - nstart

            uw.pprint(f"Number of cells: {num_cells}\n")

            # Cell-quality summary (the conditioning-relevant tail;
            # full metrics + per-cell arrays via mesh.quality()).
            try:
                Q = self.quality()
                if Q.get("element") == "2D-simplex":
                    uw.pprint(
                        f"Cell quality: q_min={Q['q_min']:.3f} "
                        f"mean={Q['q_mean']:.2f} | poor(q<0.3): "
                        f"{Q['n_q_lt_0p3']} | worst aspect "
                        f"{Q['aspect_max']:.1f} | max size-jump "
                        f"{Q['sizejump_max']:.1f}\n")
                    if Q["n_q_lt_0p2"] > 0:
                        uw.pprint(
                            f"  ! {Q['n_q_lt_0p2']} cell(s) "
                            f"q<0.2 (near-degenerate — solver "
                            f"conditioning hazard)\n")
                else:
                    uw.pprint(
                        f"Cell quality: vol_min/mean="
                        f"{Q['vol_min_over_mean']:.3f} "
                        f"(2-D triangle mesh needed for shape "
                        f"metrics)\n")
                uw.pprint("  (full metrics: mesh.quality())\n")
            except Exception:
                pass

            self._print_variable_table()

            ## Boundary information — sizes are omitted at level 0, so no
            ## collective gathers are needed (they were dead results here).
            self._print_boundary_table(with_sizes=False)

            uw.pprint(f"Use view(1) to view detailed mesh information.\n")

        elif level == 1:
            if uw.mpi.rank == 0:
                print(f"\n")
                print(f"Mesh # {self.instance}: {self.name}\n")
                uw.visualisation.plot_mesh(self)

                # Total number of cells
                nstart, nend = self.dm.getHeightStratum(0)
                num_cells = nend - nstart
                print(f"Number of cells: {num_cells}\n")

            self._print_variable_table()

            ## Boundary information (with collective size gathers)
            self._print_boundary_table(with_sizes=True)

            ## Information on the mesh DM
            self.dm.view()

        else:
            print(
                f"\n Please use view() or view(0) for default view and view(1) for a detailed view of the mesh."
            )

    def view_parallel(self):
        """
        returns the break down of boundary labels from each processor
        """

        import numpy as np

        uw.pprint(f"\n")
        uw.pprint(f"Mesh # {self.instance}: {self.name}\n")

        self._print_variable_table()

        ## Boundary information on each proc

        if len(self.boundaries) > 0:
            uw.pprint(f"| Boundary Name            | ID    | Size | Proc ID      |")
            uw.pprint(f"| ------------------------------------------------------ |")
        else:
            uw.pprint(f"No boundary labels are defined on the mesh\n")

        ### goes through each processor and gets the label size
        with uw.mpi.call_pattern(pattern="sequential"):
            for bd in self.boundaries:
                l = self.dm.getLabel(bd.name)
                if l:
                    i = l.getStratumSize(bd.value)
                else:
                    i = 0
                print(f"| {bd.name:<20}     | {bd.value:<5} | {i:<8} | {uw.mpi.rank:<8} |")

        uw.mpi.barrier()

        if uw.mpi.rank == 0:
            print(f"| ------------------------------------------------------ |")
            print("\n", flush=True)

        ## Information on the mesh DM
        # self.dm.view()

    def clone_dm_hierarchy(self):
        """
        Clone the dm hierarchy on the mesh
        """

        dm_hierarchy = self.dm_hierarchy

        new_dm_hierarchy = []
        for dm in dm_hierarchy:
            new_dm_hierarchy.append(dm.clone())

        for i, dm in enumerate(new_dm_hierarchy[:-1]):
            new_dm_hierarchy[i + 1].setCoarseDM(new_dm_hierarchy[i])

        return new_dm_hierarchy

    def _surviving_labels(self, subdm, candidates):
        """Return the candidate labels that survive on an extracted submesh DM.

        Both submesh flavours (:meth:`extract_region`, :meth:`extract_surface`)
        need the subset of the parent's boundary / region labels that still
        mark points on the filtered DM, to build the submesh's boundary enum.

        Uses the safe probe order for submesh DMs: enumerate the SUBMESH's
        labels by index (probing parent labels by name on a submesh DM can
        hard-abort PETSc), and check each label's live value set via
        ``getValueIS()`` BEFORE asking for a stratum — calling
        ``getStratumIS(v)`` for a value not in the live set can also
        hard-abort on some labels (cf. the "Centre" pseudo-label).

        Parameters
        ----------
        subdm : PETSc.DMPlex
            The filtered / extracted submesh DM.
        candidates : dict
            Parent label names mapped to their stratum values, in the order
            the surviving enum members should keep.

        Returns
        -------
        dict
            The subset of ``candidates`` present on ``subdm`` with a
            non-empty stratum, preserving candidate order.
        """
        present = set()
        for i in range(subdm.getNumLabels()):
            name = subdm.getLabelName(i)
            if name not in candidates:
                continue  # internal PETSc label (celltype, depth, ...)
            lab = subdm.getLabel(name)
            if lab is None:
                continue
            try:
                vis = lab.getValueIS()
                vals = (
                    set(int(v) for v in vis.getIndices())
                    if vis is not None else set()
                )
            except Exception:
                # Sanctioned swallow: a label whose value set cannot be
                # queried on the submesh is treated as not-surviving rather
                # than aborting the whole extraction.
                continue
            value = candidates[name]
            if value not in vals:
                continue
            lsis = lab.getStratumIS(value)
            if lsis is not None and lsis.getSize() > 0:
                present.add(name)
        return {name: v for name, v in candidates.items() if name in present}

    def extract_region(self, label_name, label_value=None):
        """Extract a submesh containing only cells with the given region label.

        Uses ``DMPlexFilter`` to create a new mesh sharing exact node
        positions with the parent. The submesh carries a ``subpoint_is``
        mapping back to the parent for restrict/prolongate operations,
        and a ``parent`` reference.

        Boundary labels from the parent survive the filter. For example,
        an "Internal" boundary on the parent becomes an exterior boundary
        on the submesh and can be referenced by the same name.

        Parameters
        ----------
        label_name : str
            DM label name identifying the region (e.g., ``"Inner"``).
        label_value : int, optional
            Stratum value within the label. If ``None``, uses
            ``mesh.regions.<label_name>.value`` when available.

        Returns
        -------
        Mesh
            A new mesh covering only the specified region.

        Examples
        --------
        >>> full_mesh = uw.meshing.AnnulusInternalBoundary(...)
        >>> rock_mesh = full_mesh.extract_region("Inner")
        >>> rock_mesh.parent is full_mesh
        True
        """
        from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label

        # Resolve label value
        if label_value is None:
            if self.regions is not None:
                try:
                    label_value = self.regions[label_name].value
                except KeyError:
                    raise ValueError(
                        f"Region '{label_name}' not found. "
                        f"Available: {[r.name for r in self.regions]}"
                    )
            else:
                raise ValueError(
                    "No regions defined on this mesh. Provide label_value explicitly."
                )

        # Filter the DM
        subdm = petsc_dm_filter_by_label(self.dm, label_name, label_value)
        subdm.markBoundaryFaces("All_Boundaries", 1001)

        # Build boundaries enum from labels that survived the filter
        # (DMPlexFilter preserves parent labels on the submesh)
        candidates = {}
        if self.boundaries is not None:
            for b in self.boundaries:
                if b.name in ("Null_Boundary", "All_Boundaries"):
                    continue
                candidates[b.name] = b.value
        if self.regions is not None:
            for r in self.regions:
                candidates[r.name] = r.value

        surviving = self._surviving_labels(subdm, candidates)
        sub_boundaries = Enum("Boundaries", surviving) if surviving else None

        # Get the subpoint IS before wrapping (the Mesh constructor may modify the DM)
        subpoint_is = subdm.getSubpointIS()

        # Construct the submesh
        sub_mesh = Mesh(
            subdm,
            degree=self.degree,
            qdegree=self.qdegree,
            boundaries=sub_boundaries,
            coordinate_system_type=self.CoordinateSystemType,
            verbose=False,
        )

        # Store lineage
        sub_mesh.parent = self
        sub_mesh._relationship_kind = "submesh"
        sub_mesh.subpoint_is = subpoint_is
        sub_mesh._parent_mesh_version = self._mesh_version
        sub_mesh._extract_label_name = label_name
        sub_mesh._extract_label_value = label_value

        # Inherit regions from parent (for nested extraction)
        sub_mesh.regions = self.regions

        # Cache for DOF mappings (built lazily on first restrict/prolongate)
        sub_mesh._dof_maps = {}

        # Build and cache the vertex map now (before any deformation)
        sub_mesh._build_vertex_map()

        # Register with parent for coordinate sync notifications
        self._registered_submeshes.add(sub_mesh)

        return sub_mesh

    def extract_surface(self, label_name, label_value=None, verbose=False):
        """Extract the codimension-1 surface marked by ``label_name`` as a mesh.

        The third submesh flavour (alongside :meth:`extract_region`, which
        filters cells of the *same* dimension): a *surface submesh* is a real
        :class:`Mesh` for the parent's codim-1 boundary stratum, sharing exact
        vertex positions with the parent. On a 3D ``SphericalShell``,
        ``shell.extract_surface("Upper")`` returns a 2-manifold embedded in
        3-space (``dim = parent.dim - 1``, ``cdim = parent.cdim``).

        Mechanism: ``DMPlexCreateSubmesh`` on the face label produces a cd-1
        DM but retains an upward-DAG phantom stratum (one point per parent
        volume cell) that breaks closure-based navigation; ``DMPlexFilter`` on
        ``(depth, dim-1)`` strips it, leaving a clean standalone manifold. The
        two subpoint IS's compose into a single surface→parent point map.

        Parent ↔ submesh DOF transfer reuses :meth:`restrict` / :meth:`prolongate`
        (the same KDTree coordinate-match-at-1e-10 path as ``extract_region``);
        surface vertices are an *exact* subset of the parent's, so it is
        bit-exact.

        Parameters
        ----------
        label_name : str
            Name of the parent boundary label whose marked faces become the
            cells of the surface submesh (e.g. ``"Upper"``).
        label_value : int, optional
            Stratum value within the label. If ``None``, resolved from
            ``self.boundaries[label_name].value``.

        Returns
        -------
        Mesh
            A surface mesh with ``parent`` set to this mesh and the standard
            submesh lineage (``subpoint_is``, registration with
            ``_registered_submeshes``).

        Raises
        ------
        ValueError
            If ``label_name`` is missing from the parent or its face stratum
            is empty (loud-fail contract — no degenerate-mesh fallback).
        """
        from underworld3.cython.petsc_discretisation import (
            petsc_dm_create_submesh_from_label,
            petsc_dm_filter_by_label,
        )

        # --- resolve the label value (from boundaries, not regions) ---
        if label_value is None:
            if self.boundaries is None:
                raise ValueError(
                    "No boundaries defined on this mesh. "
                    "Provide label_value explicitly."
                )
            try:
                label_value = self.boundaries[label_name].value
            except KeyError:
                raise ValueError(
                    f"Boundary '{label_name}' not found on parent mesh. "
                    f"Available: {[b.name for b in self.boundaries]}"
                )

        # --- loud-fail if the parent has no faces marked with this label ---
        # NB: never call getStratumIS(v) for a value not in the live value set
        # — that hard-aborts PETSc on some labels (cf. the "Centre" pseudo-
        # label). Probe getValueIS() first.
        label = self.dm.getLabel(label_name)
        if label is None:
            raise ValueError(
                f"Parent DM has no label '{label_name}'. "
                f"Cannot extract a surface submesh from it."
            )
        vals_is = label.getValueIS()
        live_values = (
            set(int(v) for v in vals_is.getIndices())
            if vals_is is not None else set()
        )
        if int(label_value) not in live_values:
            raise ValueError(
                f"Label '{label_name}' has no stratum with value {label_value} "
                f"on the parent (live values: {sorted(live_values)}). "
                f"There is no surface to extract."
            )
        sis = label.getStratumIS(label_value)
        if sis is None or sis.getSize() == 0:
            raise ValueError(
                f"Label '{label_name}' (value {label_value}) has an empty face "
                f"stratum on the parent. There is no surface to extract."
            )

        # Stage 1: cd-1 DM (with the phantom upward-DAG stratum).
        sub_with_phantoms = petsc_dm_create_submesh_from_label(
            self.dm, label_name, label_value, marked_faces=True,
        )
        # Stage 2: strip the phantom — keep only the surface cells (height-0
        # of the cd-1 chart, i.e. depth dim-1) and their downward closure.
        surf_dm = petsc_dm_filter_by_label(
            sub_with_phantoms, "depth", self.dim - 1,
        )

        # Compose the two subpoint maps (surf -> sub1 -> parent) BEFORE the
        # Mesh constructor wraps the DM (constructor side-effects can
        # invalidate cached IS handles).
        stage1_sp = sub_with_phantoms.getSubpointIS()
        stage2_sp = surf_dm.getSubpointIS()
        composed_indices = stage1_sp.getIndices()[stage2_sp.getIndices()]
        subpoint_is = PETSc.IS().createGeneral(
            composed_indices, comm=surf_dm.getComm(),
        )

        # Surviving boundaries on the surface submesh (safe getValueIS-first
        # probe; see _surviving_labels). "Centre" is additionally excluded
        # here: it is a pseudo-label with no persisted stratum.
        sub_boundaries = None
        if self.boundaries is not None:
            candidates = {
                b.name: b.value
                for b in self.boundaries
                if b.name not in ("Null_Boundary", "All_Boundaries", "Centre")
            }
            surviving = self._surviving_labels(surf_dm, candidates)
            sub_boundaries = Enum("Boundaries", surviving) if surviving else None

        # Construct the surface Mesh (dim = parent.dim - 1, cdim preserved).
        surf_mesh = Mesh(
            surf_dm,
            degree=self.degree,
            qdegree=self.qdegree,
            boundaries=sub_boundaries,
            coordinate_system_type=self.CoordinateSystemType,
            verbose=verbose,
        )

        # Submesh lineage — same shape as extract_region
        surf_mesh.parent = self
        surf_mesh._relationship_kind = "submesh"
        surf_mesh.subpoint_is = subpoint_is
        surf_mesh._parent_mesh_version = self._mesh_version
        surf_mesh._extract_label_name = label_name
        surf_mesh._extract_label_value = label_value
        surf_mesh._is_surface_submesh = True  # disambiguates from extract_region
        surf_mesh.regions = self.regions
        surf_mesh._dof_maps = {}

        # Vertex map (sub_rows -> parent_rows for coincident vertices) — the
        # same coordinate-coincidence build as extract_region. Surface
        # vertices are an exact subset of the parent's, so the 1e-10 match
        # is bit-exact. (The issue-#197 breakage that once forced an inline
        # copy here was fixed in _build_vertex_map itself.)
        surf_mesh._build_vertex_map()

        # Register with parent for coordinate sync notifications
        self._registered_submeshes.add(surf_mesh)

        return surf_mesh

    def _build_vertex_map(self):
        """Build vertex index mapping between submesh and parent.

        Uses coordinate matching at extraction time (before any
        deformation). Cached permanently since topology doesn't change.
        """
        if hasattr(self, "_vertex_map") and self._vertex_map is not None:
            return self._vertex_map

        # Build a KDTree directly on the coordinate arrays rather than
        # ``self.X._get_kdtree()`` — ``mesh.X`` is a CoordinateSystem, which has
        # no ``_get_kdtree`` (that lives on MeshVariable/swarm vars), so the old
        # call raised AttributeError on every extract_region (UW3 issue #197).
        # This mirrors the proven inline path in ``extract_surface``: submesh
        # vertices are an exact subset of the parent's, so the 1e-10 coincidence
        # match is bit-exact.
        import underworld3 as _uw

        sub_coords = numpy.asarray(self._coords)
        parent_coords = numpy.asarray(self.parent._coords)
        tree = _uw.kdtree.KDTree(sub_coords)
        dists, indices = tree.query(parent_coords, sqr_dists=False)
        dists = numpy.asarray(dists).reshape(-1)
        indices = numpy.asarray(indices).reshape(-1)
        matched = dists < 1.0e-10

        # parent_rows[i] -> sub_rows[i]: matched vertex pairs
        parent_rows = numpy.where(matched)[0]
        sub_rows = indices[matched]

        self._vertex_map = (sub_rows, parent_rows)
        return self._vertex_map

    def sync_coordinates_from_parent(self):
        """Update submesh coordinates from the parent mesh.

        Called automatically when the parent mesh deforms. Uses the
        cached vertex map to copy parent vertex positions to the
        submesh, then calls ``_deform_mesh`` to rebuild geometry.

        Raises
        ------
        ValueError
            If this mesh has no parent.
        """
        if self.parent is None:
            raise ValueError("sync_coordinates_from_parent requires a submesh")

        sub_rows, parent_rows = self._build_vertex_map()

        new_sub_coords = numpy.array(self.X.coords)
        new_sub_coords[sub_rows] = self.parent.X.coords[parent_rows]

        # Submesh follows its parent's geometry; this is a sanctioned
        # internal coordinate move (the parent owns the transfer).
        with self._coord_mutation():
            self._deform_mesh(new_sub_coords)
        self._parent_mesh_version = self.parent._mesh_version

    # --- shared DM-replacement helpers (submesh re-extraction + adapt) ---

    def _destroy_variable_petsc_state(self, var):
        """Destroy a variable's PETSc vectors and drop its cached data views.

        Required before re-initialising variables on a REPLACED DM (submesh
        re-extraction, adaptation): a stale lvec/gvec still carries field_ids
        from the old DM, and createSubDM on the new DM fails while any
        variable holds one (#48).
        """
        if var._lvec is not None:
            var._lvec.destroy()
            var._lvec = None
        if var._gvec is not None:
            var._gvec.destroy()
            var._gvec = None
        if hasattr(var, '_canonical_data'):
            var._canonical_data = None
        if hasattr(var, '_cached_data_array'):
            var._cached_data_array = None

    def _reinit_variable_on_new_dm(self, var):
        """Re-create a variable's discretisation and vectors on the current DM.

        The variable's data comes back zeroed; transferring old values is the
        caller's responsibility — see :meth:`_idw_transfer_to_var` (submesh
        re-extraction) and :meth:`adapt`'s symbol re-evaluation, the two
        deliberately different transfer strategies.
        """
        var._setup_ds()
        var._set_vec(available=True)

    def _idw_transfer_to_var(self, old_coords, old_data, var):
        """Inverse-distance-weighted transfer of backed-up DOF values.

        Submesh re-extraction's transfer strategy: each new DOF value is the
        1/d-weighted average of its dim+1 nearest old DOFs. (:meth:`adapt`
        instead re-evaluates each variable's SYMBOL at the new DOF
        coordinates — an intentional difference, do not merge the two.)
        """
        tree = uw.kdtree.KDTree(old_coords)
        nnn = 3 if self.dim == 2 else 4
        dists, indices = tree.query(var.coords, k=nnn, sqr_dists=False)

        # Inverse distance weighting
        weights = 1.0 / (dists + 1e-30)
        weights /= weights.sum(axis=1, keepdims=True)
        new_data = numpy.zeros_like(var.data)
        for i in range(nnn):
            new_data += weights[:, i:i+1] * old_data[indices[:, i]]

        var.pack_raw_data_to_petsc(new_data, sync=True)

    def _invalidate_caches_after_dm_change(self, reason):
        """Mark solvers for rebuild and drop geometry-keyed caches.

        Shared tail of every DM replacement (submesh re-extraction,
        adaptation): solvers must not trust their assembled SNES/DM, and the
        evaluation / DMInterpolation caches keyed on the old geometry must
        not serve stale results.
        """
        for solver in self._equation_systems_register:
            if solver is not None and hasattr(solver, 'is_setup'):
                solver.is_setup = False

        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        if hasattr(self, '_dminterpolation_cache'):
            self._dminterpolation_cache.invalidate_all(reason=reason)

    def _re_extract_from_parent(self, verbose=False):
        """Re-extract this submesh from the adapted parent mesh.

        Called automatically when the parent mesh adapts. Replaces the
        DM, rebuilds coordinates and vertex map, and reinitialises all
        MeshVariables on the new submesh (reset to zero).

        The Python object is updated in-place — external references
        to this submesh remain valid.
        """
        import underworld3 as uw
        from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label

        if self.parent is None:
            raise ValueError("_re_extract_from_parent requires a submesh")

        # Find which region label this submesh was extracted from
        # (stored at extraction time)
        if not hasattr(self, '_extract_label_name') or not hasattr(self, '_extract_label_value'):
            raise RuntimeError(
                "Cannot re-extract: submesh doesn't know its extraction label. "
                "Was it created with extract_region()?"
            )

        label_name = self._extract_label_name
        label_value = self._extract_label_value

        if verbose:
            uw.pprint(f"Re-extracting submesh '{label_name}' from adapted parent...")

        # Extract new DM
        new_subdm = petsc_dm_filter_by_label(self.parent.dm, label_name, label_value)
        new_subdm.markBoundaryFaces("All_Boundaries", 1001)

        # Back up old variable data and coordinates for interpolation
        old_vars = {}
        old_var_backups = {}
        for var_name, var in self._vars.items():
            if var is not None:
                old_vars[var_name] = var
                try:
                    if var._lvec is not None and var.data.size > 0:
                        old_var_backups[var_name] = (
                            numpy.array(var.coords),  # old DOF coords
                            numpy.array(var.data),     # old DOF values
                        )
                except Exception:
                    pass

        # Update DM in-place
        with self._mesh_update_lock:
            self.dm = new_subdm
            self.subpoint_is = new_subdm.getSubpointIS()

            # Rebuild coordinates with the shared coordinate-update callback
            self._install_coords_array(verbose=False)

            self._mesh_version += 1
            self._topology_version += 1
            self.nuke_coords_and_rebuild(verbose=False)

        # Rebuild vertex map (for restrict/prolongate)
        self._vertex_map = None
        self._build_vertex_map()

        # Invalidate DOF maps
        self._dof_maps = {}

        # Reinitialise variables on the new DM.
        # TODO(DESIGN): adapt() destroys ALL variable vectors upfront before
        # any _setup_ds (#48); this per-variable destroy order predates that
        # fix and has not bitten on submeshes — align when next touched.
        for var_name, old_var in old_vars.items():
            try:
                self._destroy_variable_petsc_state(old_var)
                self._reinit_variable_on_new_dm(old_var)

                # Interpolate from backed-up data via kd-tree IDW
                if var_name in old_var_backups:
                    try:
                        old_coords, old_data = old_var_backups[var_name]
                        self._idw_transfer_to_var(old_coords, old_data, old_var)
                        if verbose:
                            uw.pprint(f"  Submesh variable '{var_name}' transferred")
                    except Exception as e2:
                        if verbose:
                            uw.pprint(f"  Submesh variable '{var_name}' reset (transfer failed: {e2})")
                else:
                    if verbose:
                        uw.pprint(f"  Submesh variable '{var_name}' reset")
            except Exception as e:
                if verbose:
                    uw.pprint(f"  Warning: failed to reinitialise '{var_name}': {e}")

        # Mark solvers for rebuild and clear geometry-keyed caches
        self._invalidate_caches_after_dm_change(reason="submesh_re_extraction")

        self._parent_mesh_version = self.parent._mesh_version

        if verbose:
            uw.pprint(f"  Submesh re-extracted: {self.dm.getChart()}")

    def _build_dof_map(self, parent_var, sub_var):
        """Build a DOF-level index mapping between parent and submesh variables.

        Uses coordinate matching on DOF coordinates (exact match from
        DMPlexFilter shared nodes). Cached per variable pair.

        Returns (sub_rows, parent_rows) — numpy arrays of matching DOF indices.
        """
        import numpy as np

        key = (id(parent_var), id(sub_var))
        if key in self._dof_maps:
            return self._dof_maps[key]

        tree = sub_var._get_kdtree()
        dists, indices = tree.query(parent_var.coords_nd, sqr_dists=False)
        matched = dists < 1.0e-10

        # indices[matched] maps parent row → sub row
        parent_rows = np.where(matched)[0]
        sub_rows = indices[matched]

        if len(sub_rows) != sub_var.data.shape[0]:
            import warnings
            warnings.warn(
                f"DOF mapping: matched {len(sub_rows)} of "
                f"{sub_var.data.shape[0]} submesh DOFs"
            )

        result = (sub_rows, parent_rows)
        self._dof_maps[key] = result
        return result

    def restrict(self, parent_var, sub_var, mode="replace"):
        """Copy data from a parent-mesh variable to a submesh variable.

        Parameters
        ----------
        parent_var : MeshVariable
            Source variable on the parent mesh.
        sub_var : MeshVariable
            Destination variable on this (sub)mesh.
        mode : str
            ``"replace"`` overwrites submesh values (INSERT_VALUES).
            ``"add"`` adds parent values into submesh (ADD_VALUES).

        Raises
        ------
        ValueError
            If this mesh has no parent, or the variable meshes don't match.
        """
        if self.parent is None:
            raise ValueError("restrict requires a submesh (parent is None)")
        if parent_var.mesh is not self.parent:
            raise ValueError("parent_var must be on this mesh's parent")
        if sub_var.mesh is not self:
            raise ValueError("sub_var must be on this mesh")

        sub_rows, parent_rows = self._build_dof_map(parent_var, sub_var)

        # Copy, modify, then write through pack_raw_data_to_petsc
        # to properly sync the PETSc Vec without callback issues
        new_data = numpy.array(sub_var.data)

        if mode == "replace":
            new_data[sub_rows] = parent_var.data[parent_rows]
        elif mode == "add":
            new_data[sub_rows] += parent_var.data[parent_rows]
        else:
            raise ValueError(f"mode must be 'replace' or 'add', got '{mode}'")

        sub_var.pack_raw_data_to_petsc(new_data, sync=True)

    def prolongate(self, sub_var, parent_var, mode="replace"):
        """Copy data from a submesh variable to a parent-mesh variable.

        Parameters
        ----------
        sub_var : MeshVariable
            Source variable on this (sub)mesh.
        parent_var : MeshVariable
            Destination variable on the parent mesh.
        mode : str
            ``"replace"`` overwrites parent values at submesh DOFs.
            ``"add"`` adds submesh values into parent.

        Raises
        ------
        ValueError
            If this mesh has no parent, or the variable meshes don't match.
        """
        if self.parent is None:
            raise ValueError("prolongate requires a submesh (parent is None)")
        if parent_var.mesh is not self.parent:
            raise ValueError("parent_var must be on this mesh's parent")
        if sub_var.mesh is not self:
            raise ValueError("sub_var must be on this mesh")

        sub_rows, parent_rows = self._build_dof_map(parent_var, sub_var)

        new_data = numpy.array(parent_var.data)

        if mode == "replace":
            new_data[parent_rows] = sub_var.data[sub_rows]
        elif mode == "add":
            new_data[parent_rows] += sub_var.data[sub_rows]
        else:
            raise ValueError(f"mode must be 'replace' or 'add', got '{mode}'")

        parent_var.pack_raw_data_to_petsc(new_data, sync=True)

        parent_var._data_is_dirty = True

    # ------------------------------------------------------------------ #
    #  Refinement-child transfers (SBR adapt-on-top; self is the CHILD)
    #
    #  Unlike the submesh restrict/prolongate above (DOFs coincide via
    #  ``subpoint_is``), an adapt() child is *bigger* than its parent and the
    #  DOFs do not coincide. Direction therefore flips:
    #    parent (coarse) -> child (fine)  = PROLONGATE  (FE-exact custom-P)
    #    child  (fine)   -> parent (coarse) = RESTRICT  (injection at shared nodes)
    #  Serial uses the structured barycentric custom-P / nearest-node injection
    #  (FE-exact, the quality path); parallel (np>1) falls back to the
    #  partition-agnostic ``global_evaluate`` REMAP (swarm-migration based).
    # ------------------------------------------------------------------ #
    def _refine_prolongate(self, parent_var, child_var, mode="replace"):
        """Coarse parent -> fine child interpolation (this mesh is the child)."""
        if self._relationship_kind != "refinement":
            raise ValueError("_refine_prolongate requires an adapt() refinement child")
        pv = getattr(parent_var, "_base_var", parent_var)
        cv = getattr(child_var, "_base_var", child_var)

        if uw.mpi.size > 1:
            out = numpy.asarray(
                uw.function.global_evaluate(pv.sym, numpy.asarray(cv.coords))
            ).reshape(cv.data.shape)
        else:
            from underworld3.utilities import custom_mg
            cc = numpy.asarray(self.parent._get_coords_for_basis(pv.degree, pv.continuous))
            fc = numpy.asarray(self._get_coords_for_basis(cv.degree, cv.continuous))
            P = custom_mg.barycentric_prolongation(cc, fc)
            out = (P @ numpy.asarray(pv.data)).reshape(cv.data.shape)

        new = numpy.array(cv.data)
        if mode == "replace":
            new[:] = out
        elif mode == "add":
            new[:] += out
        else:
            raise ValueError(f"mode must be 'replace' or 'add', got '{mode}'")
        cv.pack_raw_data_to_petsc(new, sync=True)

    def _refine_restrict(self, child_var, parent_var, mode="replace"):
        """Fine child -> coarse parent injection (this mesh is the child)."""
        if self._relationship_kind != "refinement":
            raise ValueError("_refine_restrict requires an adapt() refinement child")
        pv = getattr(parent_var, "_base_var", parent_var)
        cv = getattr(child_var, "_base_var", child_var)

        if uw.mpi.size > 1:
            out = numpy.asarray(
                uw.function.global_evaluate(cv.sym, numpy.asarray(pv.coords))
            ).reshape(pv.data.shape)
        else:
            from scipy.spatial import cKDTree
            cc = numpy.asarray(self.parent._get_coords_for_basis(pv.degree, pv.continuous))
            fc = numpy.asarray(self._get_coords_for_basis(cv.degree, cv.continuous))
            # nested SBR: every coarse DOF coincides with a fine DOF (P1) or sits
            # on a fine element edge (P2) -> nearest fine node is exact / near-exact.
            _, idx = cKDTree(fc).query(cc)
            out = numpy.asarray(cv.data)[idx].reshape(pv.data.shape)

        new = numpy.array(pv.data)
        if mode == "replace":
            new[:] = out
        elif mode == "add":
            new[:] += out
        else:
            raise ValueError(f"mode must be 'replace' or 'add', got '{mode}'")
        pv.pack_raw_data_to_petsc(new, sync=True)
        pv._data_is_dirty = True

    def nuke_coords_and_rebuild(
        self,
        verbose=False,
        active_vars=None,
    ):
        """Rebuild DM/DS, the kd-tree, mesh sizes, and per-variable DOF
        coordinate caches after a coordinate change.

        ``active_vars`` (optional set/list of MeshVariables): restrict
        the per-variable DOF coordinate-cache recomputation to this set.
        When ``None`` (default) every registered variable is
        recomputed eagerly, matching the BUGFIX(#130) collective-safe
        behaviour. Movers that thread their own work-vars through
        ``_deform_mesh(..., active_vars=...)`` skip recomputing the
        non-mover variables n_outer× during the inner sweep; the
        wrapper does one final ``_deform_mesh`` (or a direct
        ``nuke_coords_and_rebuild()``) with ``active_vars=None`` at
        sweep exit to bring the full cache back into sync.

        Naming note: "nuke and rebuild" historically referred to the
        DS+DM tear-down/recreate; what was called "refill" of the
        per-variable cache (line 1890 in the old code) is in fact
        *recomputation* of each variable's DOF coordinates from the new
        mesh coordinates — the storage is per-variable, the values are
        derived. The ``active_vars`` whitelist controls which of those
        recomputations runs now versus deferring to the next full
        rebuild.
        """
        # This is a reversion to the old version (3.15 compatible which seems to work in 3.16 too)
        #
        #

        self.dm.clearDS()
        self.dm.createDS()

        if verbose:
            uw.pprint(f"PETScDS - (re) initialised")

        self._coord_array = {}

        # let's go ahead and do an initial projection from linear (the default)
        # to linear. this really is a nothing operation, but a
        # side effect of this operation is that coordinate DM DMField is
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates.

        # Dual-space options control node placement on simplices and must be set
        # before createDefault(). Currently only P1 coordinate meshes are used,
        # but these are needed for higher-order (curved) coordinate meshes.
        options = PETSc.Options()
        options.setValue(f"meshproj_{self.mesh_instances}_petscspace_degree", self.degree)
        options.setValue(f"meshproj_{self.mesh_instances}_petscdualspace_lagrange_continuity", True)
        options.setValue(f"meshproj_{self.mesh_instances}_petscdualspace_lagrange_node_endpoints", False)

        self.petsc_fe = PETSc.FE().createDefault(
            self.dim,
            self.cdim,
            self.isSimplex,
            self.qdegree,
            f"meshproj_{self.mesh_instances}_",
        )

        if verbose and uw.mpi.rank == 0:
            print(
                f"PETScFE - (re) initialised",
                flush=True,
            )

        if PETSc.Sys.getVersion() <= (3, 20, 5) and PETSc.Sys.getVersionInfo()["release"] == True:
            self.dm.projectCoordinates(self.petsc_fe)
        elif hasattr(self.dm, "createCoordinateSpace"):
            # Use createCoordinateSpace rather than setCoordinateDisc.
            # setCoordinateDisc with a user-created FE leaves the coordinate
            # dual space without proper point subspaces, causing
            # DMPlexComputeBdIntegral to segfault/deadlock (issue #96).
            # createCoordinateSpace builds the FE internally with correct
            # subspace initialisation.
            self.dm.createCoordinateSpace(self.degree, False, True)

            # Issue #96 fix: Force coordinate field creation and strip
            # boundary labels from the coordinate DM. createCoordinateSpace
            # clears the coordinate field cache. Without this, BdIntegral
            # lazily recreates the field by cloning mesh.dm (with boundary
            # labels), causing DMCompleteBCLabels_Internal MPI errors.
            from underworld3.cython.petsc_maths import dm_force_coordinate_field
            dm_force_coordinate_field(self.dm)
        elif PETSc.Sys.getVersion() >= (3, 24, 0):
            self.dm.setCoordinateDisc(disc=self.petsc_fe, localized=False, project=False)
        else:
            self.dm.setCoordinateDisc(disc=self.petsc_fe, project=False)

        if verbose and uw.mpi.rank == 0:
            print(
                f"PETSc DM - coordinates",
                flush=True,
            )

        # now set copy of this array into dictionary
        arr = self.dm.getCoordinatesLocal().array

        key = (
            self.isSimplex,
            self.degree,
            True,
        )  # True here assumes continuous basis for coordinates ...

        self._coord_array[key] = arr.reshape(-1, self.cdim).copy()

        # invalidate the cell-search k-d tree and the mesh centroid data / rebuild
        #

        if verbose and uw.mpi.rank == 0:
            print(
                f"UW kD-Tree",
                flush=True,
            )

        # The navigation coords (used to build the kd-tree control points and
        # for point location) were captured as a reference to the ORIGINAL
        # coords in __init__ and are not updated by the DM rebuild above, so
        # on an adapted/deformed mesh they still describe the old geometry.
        # They MUST be refreshed BEFORE _build_kd_tree_index() runs: that
        # rebuild indexes _nav_coords with the NEW nav-DM point range, and a
        # stale (old-sized) array raises IndexError when the mesh grew — and
        # silently mislocates points otherwise (issue #286).
        if getattr(self, "_nav_dm", None) is None:
            self._nav_coords = numpy.asarray(
                self.dm.getCoordinatesLocal().array
            ).reshape(-1, self.cdim)
        else:
            # manifold mesh: the nav clone carries its own (ghosted) coords.
            # Push the rebuilt/deformed geometry onto the clone via its GLOBAL
            # coordinate vector (owned points share numbering with the main
            # DM), then rebuild the section-consistent local array incl. the
            # ghost rows. setCoordinatesLocal(main-local) is size-mismatched on
            # an overlapped clone and leaves ghost rows unfilled (issue #360).
            try:
                self._nav_dm.setCoordinates(self.dm.getCoordinates())
                self._nav_coords = self._navigation_coords_from_dm(self._nav_dm)
            except Exception:
                pass

        self._index = None
        self._build_kd_tree_index()

        if verbose and uw.mpi.rank == 0:
            print(
                f"UW kD-Tree - constructed",
                flush=True,
            )

        (
            self._min_size,
            self._radii,
            self._centroids,
            self._search_lengths,
        ) = self._get_mesh_sizes()

        # Skip self-copy when hierarchy is trivial (issue #96 investigation)
        if self.dm is not self.dm_hierarchy[-1]:
            self.dm.copyDS(self.dm_hierarchy[-1])

        # Invalidate projected boundary normals (rebuilt lazily on access)
        self._projected_normals = None
        # Per-boundary deformation-tracking normals are stale now too. The
        # variables persist (we keep the name->var map); their DATA is refilled
        # eagerly by Mesh.deform() after the remesh completes, so BCs that
        # captured boundary_normal(...).sym at setup read the new geometry.

        # BUGFIX(#135): invalidate the per-cell face control-point arrays.
        # These are populated lazily by _get_mesh_face_control_points, sized
        # (num_faces, num_local_cells, dim). After mesh.adapt() the new mesh
        # has a different cell count, so the stale arrays from the old mesh
        # would be indexed with new-mesh cell IDs in
        # _test_if_points_in_cells_internal — producing IndexError when the
        # new cell count exceeds the old one (and silent corruption otherwise).
        self.faces_outer_control_points = None
        self.faces_inner_control_points = None

        # BUGFIX (deformed-domain membership): also invalidate the boundary-
        # skeleton kd-tree used by points_in_domain() (and the SL out-of-domain
        # restore). It is cached from the boundary geometry and was only rebuilt
        # on adapt. After a DEFORM the surface has moved, so the stale control
        # points (at the OLD boundary) wrongly flag a bulged-out region (r>r_o
        # on a free surface) as EXTERIOR — stranding semi-Lagrangian trace-back
        # feet there and mis-locating evaluations, which injects the cold
        # boundary value at the topographic highs (upwellings). Rebuilt lazily
        # from the deformed boundary on next points_in_domain() access.
        self.boundary_face_control_points_kdtree = None
        self.boundary_face_control_points_sign = None
        self._domain_radius_squared = float("inf")

        # NB: the _nav_coords refresh for the deformed/adapted geometry happens
        # ABOVE, before _build_kd_tree_index() — see the issue #286 note there.

        # BUGFIX(#130): recompute the DOF coordinate cache for every
        # already-registered variable. Variables created before this
        # rebuild would otherwise have their cache entry (from __init__)
        # wiped above and recompute lazily from rank-local code paths
        # (rbf_interpolate), which deadlocks when the collectives
        # inside _get_coords_for_basis are reached by only a subset of
        # ranks.
        #
        # ``active_vars`` (Phase-1 remesh redesign): when set, restrict
        # the recompute to the listed variables. Movers in
        # smoothing.py thread their work-vars during the inner sweep so
        # the n_outer× recompute of user fields (T, V, P, every
        # psi_star, ...) is paid only once at sweep exit. The collective
        # safety property (#130) is preserved because the mover is
        # itself collective — every rank passes the same active set —
        # and the sweep wrapper does one full recompute at exit.
        if active_vars is None:
            _recompute = list(self.vars.values())
        else:
            # Map identity-equal lookup; tolerate either base or wrapper
            # variables in the whitelist (mesh.vars stores base vars but
            # callers commonly pass the user-visible wrapper).
            _ids = set()
            for v in active_vars:
                _ids.add(id(v))
                _ids.add(id(getattr(v, "_base_var", None)))
            _recompute = [v for v in self.vars.values() if id(v) in _ids]
        for _var in _recompute:
            self._get_coords_for_var(_var)

        if verbose and uw.mpi.rank == 0:
            print(
                f"Mesh Spatial Discretisation Complete",
                flush=True,
            )

        return

    def _update_projected_normals(self):
        """Project PETSc face normals (Gamma) onto a P1 field and normalise.

        Creates ``_projected_normals`` on first call, updates in-place
        thereafter. The result is a smooth, consistently-oriented unit
        normal field that works well for penalty and Nitsche BCs on
        curved boundaries.

        NOTE: this GLOBAL field point-evaluates ``mesh.Gamma`` whose petsc_n
        only exists in surface-integral kernels, so it falls back to the
        coordinate (radial for a circle) and does NOT track a deformed
        surface. For deformation-aware, corner-correct normals use the
        per-boundary :meth:`boundary_normal` (which ``add_nitsche_bc`` now
        uses). This global field is retained unchanged for back-compat.
        """
        import underworld3 as uw

        Gamma = self.Gamma

        if not hasattr(self, '_projected_normals') or self._projected_normals is None:
            existing = self.vars.get("_n_proj")
            if existing is not None:
                self._projected_normals = existing
            else:
                self._projected_normals = uw.discretisation.MeshVariable(
                    "_n_proj", self, self.cdim, degree=1,
                    remesh_policy="reinit",
                )

        n = self._projected_normals
        for i in range(self.cdim):
            n.data[:, i] = uw.function.evaluate(Gamma[i], n.coords).flatten()

        mag = numpy.sqrt(numpy.sum(n.data ** 2, axis=1))
        nonzero = mag > 1.0e-30
        n.data[nonzero] /= mag[nonzero, numpy.newaxis]

    def boundary_normal(self, boundary):
        """Outward unit normal of a single boundary, tracking deformation.

        Assembles the EXACT, outward, area-weighted PETSc facet normals
        (``dm.computeCellGeometryFVM``) from ONLY this boundary's facets onto
        its P1 vertices. Because each boundary is assembled independently,
        a vertex shared by two boundaries (a sharp corner) is NOT averaged
        across the discontinuity — each boundary keeps its own normal. On a
        smooth boundary (e.g. a free surface) the result is the smooth
        deformed normal. Cached per boundary; rebuilt lazily after a deform.

        Returns a sympy Matrix (row) of the P1 normal-field components, for
        use as the constraint direction in Nitsche/penalty BCs.

        Parameters
        ----------
        boundary : str or enum
            Boundary label name (or a ``mesh.boundaries`` enum member).
        """
        import underworld3 as uw

        name = getattr(boundary, "name", str(boundary))
        if not hasattr(self, "_boundary_normal_vars") or self._boundary_normal_vars is None:
            self._boundary_normal_vars = {}
        var = self._boundary_normal_vars.get(name)
        if var is None:
            existing = self.vars.get(f"_n_bd_{name}")
            var = existing if existing is not None else uw.discretisation.MeshVariable(
                f"_n_bd_{name}", self, self.cdim, degree=1, remesh_policy="reinit")
            self._boundary_normal_vars[name] = var
        self._assemble_boundary_normal(var, name)
        return var.sym

    def _assemble_boundary_normal(self, var, name):
        """Fill ``var`` with the area-weighted outward facet normal assembled
        from the faces of boundary ``name`` only (see :meth:`boundary_normal`)."""
        from scipy.spatial import cKDTree
        cdim = self.cdim
        dm = self.dm
        coords = numpy.ascontiguousarray(var.coords)
        accum = numpy.zeros((coords.shape[0], cdim))

        # faces carrying this boundary label: DM label named after the boundary,
        # stratum keyed by the boundary's value (same access the BC code uses).
        bvalue = None
        for b in (self.boundaries or []):
            if b.name == name:
                bvalue = b.value
                break
        face_pts = []
        label = dm.getLabel(name) if dm.hasLabel(name) else None
        if label is not None and bvalue is not None:
            # NB: getStratumIS(value) for a value NOT in this rank's live value
            # set can hard-abort the interpreter (e.g. a rank holding no faces
            # of this boundary in parallel). Only query a live value.
            try:
                vis = label.getValueIS()
                live = set(int(x) for x in vis.getIndices()) if vis.getSize() else set()
            except Exception:
                live = set()
            if int(bvalue) in live:
                pis = label.getStratumIS(bvalue)
                if pis is not None and pis.getSize():
                    fS, fE = dm.getHeightStratum(1)
                    for p in pis.getIndices():
                        if fS <= int(p) < fE:
                            face_pts.append(int(p))

        tree = cKDTree(coords)
        # P1 vertices per facet, counted from the facet's own closure so this
        # works for non-simplex facets too (2D edge=2, 3D tri=3, 3D quad=4).
        vStart, vEnd = dm.getDepthStratum(0)
        for f in face_pts:
            if dm.getSupportSize(f) != 1:
                continue
            area, cent, nrm = dm.computeCellGeometryFVM(f)
            nrm = numpy.asarray(nrm)[:cdim]
            cell = dm.getSupport(f)[0]
            _, ccent, _ = dm.computeCellGeometryFVM(cell)
            if numpy.dot(nrm, numpy.asarray(cent)[:cdim]
                         - numpy.asarray(ccent)[:cdim]) < 0:
                nrm = -nrm
            _clo = dm.getTransitiveClosure(f)[0]
            nverts = int(numpy.count_nonzero((_clo >= vStart) & (_clo < vEnd))) or cdim
            # Accumulate to the facet's P1 DOFs (its vertices) — found as the
            # nearest DOFs to the facet centroid. This avoids indexing the local
            # coordinate array by (vertex_point - vStart), which is only valid
            # in serial (the parallel coordinate layout differs → out-of-range).
            # Normalisation at the end makes the per-DOF weight (full vs share)
            # irrelevant to the resulting direction.
            _, idxs = tree.query(numpy.asarray(cent)[:cdim], k=nverts)
            for idx in numpy.atleast_1d(idxs):
                accum[idx] += area * nrm

        # TODO(parallel): a boundary vertex on a partition interface should
        # ADD-reduce the UNnormalised facet contributions from both ranks before
        # normalising (DMLocalToGlobal ADD_VALUES on the variable's section),
        # so its normal is the full-stencil average rather than this rank's
        # partial stencil. This is parallel-SAFE as-is (rank-interior boundary
        # vertices are exact; only the handful of partition-seam surface
        # vertices get a slightly-rotated unit normal). A first ADD-reduce
        # attempt SEGV'd on the lazily-built work variable's global vec; deferred
        # to a focused follow-up with the right vec/section plumbing.
        mag = numpy.sqrt(numpy.sum(accum ** 2, axis=1))
        nonzero = mag > 1.0e-30
        accum[nonzero] /= mag[nonzero, numpy.newaxis]
        var.data[...] = accum

    def cell_size(self):
        """Local, per-cell characteristic mesh size as a scalar field symbol.

        Returns the ``.sym`` of a cell-constant (degree-0, discontinuous)
        scalar MeshVariable holding each cell's characteristic length (the
        ``volume**(1/dim)`` equivalent radius, i.e. ``self._radii``). Unlike
        the single *global* scalar from :meth:`get_min_radius` (the smallest
        cell anywhere), this varies cell to cell, so a stabilisation that
        scales as :math:`1/h` — e.g. the Nitsche free-slip penalty
        :math:`\\gamma\\mu/h` — is correctly scaled on every facet of a
        non-uniform or adaptively-refined mesh rather than using the global
        minimum (which over-penalises coarse cells and drifts as refinement
        changes the global min).

        On a boundary integral the kernel sees the value of the cell adjacent
        to the facet. The field is cached and rebuilt lazily; its data is
        refreshed when the mesh deforms or is adapted (see :meth:`deform`),
        so it tracks a moving / re-refined mesh — a stale size on a deformed
        mesh would re-introduce the mis-scaling.

        On a uniform mesh every cell is the same size, so this reduces to the
        global ``get_min_radius`` value everywhere and existing behaviour is
        preserved to tolerance.

        Returns
        -------
        sympy scalar
            The cell-size field symbol, for use in JIT-compiled residuals.
        """
        var = self._cell_size_var()
        return var.sym[0]

    def _cell_size_var(self):
        """Lazily create / fetch the per-cell size MeshVariable (filled).

        Mirrors the per-boundary normal machinery (:meth:`boundary_normal`):
        a small ``reinit`` MeshVariable owned by the mesh, refreshed from the
        current geometry. The reinit callback re-fills it during any remesh
        transaction (deform / mover sweep), and :meth:`deform` / :meth:`adapt`
        re-fill it explicitly so BCs that captured ``cell_size()`` at setup
        read the new geometry at solve time.
        """
        import underworld3 as uw

        if getattr(self, "_cell_size_variable", None) is None:
            existing = self.vars.get("_h_cell")
            var = existing if existing is not None else uw.discretisation.MeshVariable(
                "_h_cell", self, 1, degree=0, continuous=False,
                remesh_policy="reinit")
            self._cell_size_variable = var

            def _refresh():
                try:
                    self._assemble_cell_size(var)
                except Exception:
                    # Sanctioned swallow: this fires inside a remesh
                    # transaction where the variable may be mid-teardown
                    # (vecs destroyed, sizes transiently inconsistent).
                    # deform()/adapt() re-fill the field explicitly right
                    # after, so a failed opportunistic refresh here only
                    # delays the update — it must not abort the remesh.
                    pass

            var._remesh_reinit_callback = _refresh

        self._assemble_cell_size(self._cell_size_variable)
        return self._cell_size_variable

    def _assemble_cell_size(self, var):
        """Fill ``var`` (degree-0 scalar) with each cell's characteristic size.

        Uses the per-cell characteristic lengths ``self._radii`` computed by
        :meth:`_get_mesh_sizes` on the *current* geometry. A degree-0
        discontinuous variable's local DOFs and ``self._radii`` are BOTH
        indexed by this rank's cell-stratum order, so a direct assignment is
        correct on every rank.

        This is deliberately a purely RANK-LOCAL operation (no ``var.coords``
        access, no collective): mixing a rank-local fast path with a
        collective fallback would diverge across ranks and deadlock, because
        ``var.coords`` triggers the collective ``_get_coords_for_basis``."""
        radii = numpy.asarray(self._radii).reshape(-1)
        # Empty partition (no local cells): nothing to fill on this rank.
        if radii.size == 0 or var.data.shape[0] == 0:
            return
        # Assign over the common length. In practice these match exactly (same
        # local cell set / ordering); the slice only guards a stray off-by-ghost
        # mismatch without ever taking a collective path on a subset of ranks.
        n = min(var.data.shape[0], radii.shape[0])
        var.data[:n, 0] = radii[:n]

    @property
    def Gamma_P1(self):
        """Projected P1 boundary normals as a sympy Matrix.

        Returns the normalised, vertex-averaged PETSc face normals
        as a smooth P1 field. Preferred over :attr:`Gamma_N` for
        penalty and Nitsche BCs on curved boundaries — gives
        consistent orientation and better convergence in 3D.

        Automatically updated when the mesh deforms.
        """
        if not hasattr(self, '_projected_normals') or self._projected_normals is None:
            self._update_projected_normals()
        return self._projected_normals.sym

    # ===================================================================
    #  Bounding surfaces — per-surface tangent-slip + restore.
    #  See docs/developer/design/boundary-slip-strategy.md. SEPARATE from
    #  self.boundaries (the persisted gmsh/DMPlex labelling, untouched).
    # ===================================================================
    @property
    def bounding_surfaces(self):
        """Mapping ``{label: BoundingSurface}`` of this mesh's registered
        bounding-surface objects (tangent-slip + restore).

        This is a NEW collection, separate from and additional to
        :attr:`boundaries` (the persisted gmsh/DMPlex label ``Enum``, left
        untouched). Populated by analytic-geometry constructors (Annulus,
        SphericalShell, CubedSphere, box meshes); user-extendable via
        :meth:`register_tangent_slip_provider`.
        """
        if not hasattr(self, "_bounding_surfaces") or self._bounding_surfaces is None:
            self._bounding_surfaces = {}
        return self._bounding_surfaces

    def register_tangent_slip_provider(self, label, surface):
        """Install a :class:`BoundingSurface` object for a boundary ``label``
        (separate from the persisted ``mesh.boundaries`` labelling).

        Lets a user declare a custom analytic surface (e.g. an ellipsoid) the
        constructors don't know about, or replace one (free-surface release).
        """
        from underworld3.meshing.bounding_surface import BoundingSurface
        if not isinstance(surface, BoundingSurface):
            raise TypeError(
                "surface must be a BoundingSurface; got "
                f"{type(surface).__name__}")
        self.bounding_surfaces[str(label)] = surface
        return surface

    def _resolve_slip_spec(self, slip_spec):
        """Resolve a ``slip_spec`` to ``(slip_labels tuple, free_labels set)``.

        Back-compatible forms: ``True``/``"all"``/``"ring"``/``"box"`` → all
        geometric boundary labels; ``False``/``None`` → none; a label name; a
        list of labels; a ``dict {label: snap_bool}`` (``False`` = free
        surface, slip but do not restore).
        """
        from underworld3.meshing.smoothing import _auto_pinned_labels
        geo = _auto_pinned_labels(self)
        if slip_spec is None or slip_spec is False:
            return (), set()
        if slip_spec is True:
            return tuple(geo), set()
        if isinstance(slip_spec, dict):
            return tuple(slip_spec.keys()), {k for k, v in slip_spec.items() if not v}
        if isinstance(slip_spec, str):
            s = slip_spec.strip().lower()
            if s in ("true", "on", "1", "all", "ring", "box", "axes", "axis"):
                return tuple(geo), set()
            if s in ("false", "off", "0", "none", ""):
                return (), set()
            return (slip_spec,), set()
        return tuple(slip_spec), set()

    def restore_to_surface(self, coords, label):
        """Snap ``coords`` onto the named bounding surface (delegates to the
        surface object's ``restore``)."""
        return self.bounding_surfaces[str(label)].restore(
            numpy.asarray(coords, dtype=float))

    def tangent_project(self, coords, label, reference):
        """Tangent-slide ``coords`` (displacement measured from ``reference``)
        on the named bounding surface (delegates to the surface object)."""
        return self.bounding_surfaces[str(label)].tangent_project(
            numpy.asarray(coords, dtype=float),
            numpy.asarray(reference, dtype=float))

    def boundary_slip(self, slip_spec=True, reference_coords=None,
                      boundary_labels=None):
        """Build ``(is_pinned, project)`` for tangent slip on this mesh's
        bounding surfaces — the orchestrator the metric movers call.

        See ``docs/developer/design/boundary-slip-strategy.md``. The mesh
        classifies which vertices slip vs pin (the cross-surface concern); each
        surface object owns its tangent-project + restore.

        A vertex **slips** iff it lies on **exactly one** slip surface that has a
        registered analytic :class:`BoundingSurface`; non-boundary, junction
        (≥2 surfaces), unregistered-surface, or degenerate-normal vertices are
        **pinned** (the step-1 safe default — ``facet`` restore is a follow-up).

        Parameters
        ----------
        slip_spec :
            See :meth:`_resolve_slip_spec`. Default ``True`` (all surfaces).
        reference_coords : ndarray, optional
            Fixed reference vertex positions (local-chart vertex order) that
            displacements are measured from. Defaults to ``mesh.X.coords``.
        boundary_labels : iterable of str, optional
            Boundary labels defining the boundary (``is_bnd``). Defaults to all
            geometric boundary labels; pass a mover's pinned set for parity.

        Returns
        -------
        (is_pinned, project) : (ndarray[bool], callable)
            ``is_pinned`` is the per-vertex pinned mask (local-chart order);
            ``project(Y)`` slides+restores the slip vertices of ``Y`` in place
            and returns it.
        """
        from underworld3.meshing.smoothing import (
            _pinned_mask, _auto_pinned_labels, _owned_vertex_mask)
        from underworld3.meshing.smoothing import _boundary_facets
        from underworld3.meshing.bounding_surface import BoundingSurface

        dm = self.dm
        cdim = self.cdim
        pStart, pEnd = dm.getDepthStratum(0)
        n_verts = pEnd - pStart
        if reference_coords is None:
            reference_coords = numpy.asarray(self.X.coords, dtype=float)
        ref = numpy.asarray(reference_coords, dtype=float)

        all_labels = (tuple(boundary_labels) if boundary_labels is not None
                      else _auto_pinned_labels(self))
        # TODO(follow-up): _pinned_mask expands labels through vertices/edges
        # only, so a 3D boundary label that tags FACES alone (a mesh loaded with
        # markVertices=False) leaves its boundary vertices unmarked. This is a
        # pre-existing limitation of the shared helper used by every mover; the
        # fix (close faces→edges→vertices) belongs with _pinned_mask itself.
        is_bnd = _pinned_mask(dm, all_labels)

        slip_labels, free_labels = self._resolve_slip_spec(slip_spec)
        # Per-label vertex masks (closure of each label's tagged facets).
        masks = {lab: _pinned_mask(dm, (lab,)) for lab in slip_labels}

        # Resolve a BoundingSurface for every slip label. Constructor-registered
        # labels (radial / plane) restore analytically; a slip label with NO
        # registered surface (a loaded mesh, an internal boundary) gets a
        # *transient* ``facet`` surface built from THIS call's reference facets
        # — nearest-reference-facet restore, matching the mover's
        # ``_build_slip_projector`` facet fallback rather than pinning. FREE
        # labels (dict ``False``) still slide-without-restore regardless of
        # kind (handled in ``project`` below). A label with no boundary facets
        # at all stays unusable → its vertices pin (the safe default).
        surf = dict(self.bounding_surfaces)
        unreg = [lab for lab in slip_labels if lab not in surf]
        if unreg:
            facets, _opp = _boundary_facets(self, cdim)
            if facets is not None and facets.size:
                for lab in unreg:
                    fac_in = masks[lab][facets].all(axis=1)
                    if fac_in.any():
                        surf[lab] = BoundingSurface(
                            self, lab, "facet",
                            reference_facets=ref[facets[fac_in]])
        usable = [lab for lab in slip_labels if lab in surf]
        masks = {lab: masks[lab] for lab in usable}
        count = numpy.zeros(n_verts, dtype=int)
        for m in masks.values():
            count += m.astype(int)
        slip_mask = is_bnd & (count == 1)
        is_pinned = is_bnd & ~slip_mask
        vert_label = numpy.empty(n_verts, dtype=object)
        for lab, m in masks.items():
            vert_label[m & slip_mask] = lab

        # Project only OWNED slip vertices: the movers halo-sync owned→ghost
        # after calling project(), so a leaf/ghost receives its owner's
        # projected value — modifying non-owned coordinates here is both
        # wasteful and a parallel-safety hazard. (Serial: every vertex is
        # owned, so this is a no-op.) is_pinned stays the full geometric
        # classification, which is rank-consistent for shared vertices.
        slip_b = numpy.nonzero(slip_mask & _owned_vertex_mask(dm))[0]
        if slip_b.size == 0:
            return is_pinned, (lambda Y: Y)
        old_slip = ref[slip_b]
        labels_b = vert_label[slip_b]

        # Precompute each slip vertex's tangent-slide normal ONCE, at the fixed
        # reference (see the re-solve-vs-cached trade-off in the DESIGN NOTE on
        # ``BoundingSurface.normals``). The metric movers call ``project``
        # repeatedly inside their line-search backtrack; re-deriving the
        # projected normal (a ``Gamma_P1`` re-solve via ``_slip_normals``) on
        # every call would be a severe regression. The normal is taken at the
        # reference and is constant
        # across the backtrack — matching ``_build_slip_projector``, which also
        # fixes the normal per build. A slip vertex with a degenerate normal
        # (``valid`` False — e.g. a corner the junction rule missed) keeps its
        # reference position under the slide; the surface restore still applies.
        normals_b = numpy.zeros((slip_b.size, cdim))
        valid_b = numpy.zeros(slip_b.size, dtype=bool)
        for lab in usable:
            sel = labels_b == lab
            if not sel.any():
                continue
            nrm, val = surf[lab].normals(old_slip[sel])
            normals_b[sel] = nrm
            valid_b[sel] = val

        def project(Y):
            Y = numpy.asarray(Y, dtype=float)
            # tangent slide with the precomputed reference normals
            disp = Y[slip_b] - old_slip
            dn = (disp * normals_b).sum(axis=1, keepdims=True)
            slid = numpy.where(valid_b[:, None],
                               old_slip + (disp - dn * normals_b), old_slip)
            for lab in usable:
                sel = labels_b == lab
                if not sel.any():
                    continue
                idx = slip_b[sel]
                # FREE surfaces (dict spec False) slide but do not restore.
                Y[idx] = (slid[sel] if lab in free_labels
                          else surf[lab].restore(slid[sel]))
            return Y

        return is_pinned, project

    def project_to_slip_surface(self, coords, slip_spec=True,
                                reference_coords=None, boundary_labels=None):
        """In-place convenience over :meth:`boundary_slip`: slide+restore the
        slip vertices of ``coords`` (a full local-chart vertex array) and return
        it. For callers that just want coordinates snapped back (a checkpoint
        reload, a diagnostic, the free-surface module)."""
        _is_pinned, project = self.boundary_slip(
            slip_spec, reference_coords=reference_coords,
            boundary_labels=boundary_labels)
        return project(numpy.asarray(coords, dtype=float))

    @timing.routine_timer_decorator
    def update_lvec(self, swarm_sync=True):
        """
        This method creates and/or updates the mesh variable local vector.
        If the local vector is already up to date, this method will do nothing.

        ``swarm_sync=False`` skips the swarm-dependency hook below. It MUST
        be passed at call sites that only a SUBSET of ranks reach (e.g. the
        refresh calls inside ``petsc_interpolate`` — ranks with zero interior
        points skip that function entirely): the hook performs collective
        reductions, so running it on a subset of ranks deadlocks. Such sites
        rely on an earlier all-ranks ``update_lvec()`` for freshness, exactly
        as they already do for the collective ``globalToLocal`` below.
        """

        # Swarm dependencies first (issues #215 Bug 3 / #289): run any
        # deferred particle migration and refresh stale swarm-variable
        # proxies BEFORE the staleness check below — the refresh writes
        # proxy MeshVariable data, which is what sets _stale_lvec. Solvers
        # read the proxy DM directly, so the lazy `.sym` refresh alone
        # cannot guarantee freshness at assembly. Collective; flag-guarded
        # no-op when nothing changed. Ordered deterministically so all
        # ranks act on swarms in the same sequence.
        if swarm_sync and not getattr(self, "_swarm_sync_in_progress", False):
            self._swarm_sync_in_progress = True
            try:
                for swarm in sorted(
                    self._registered_swarms,
                    key=lambda s: s._instance_number,
                ):
                    swarm._sync_before_assembly()
            finally:
                self._swarm_sync_in_progress = False

        if self._stale_lvec:
            if not self._lvec:
                self.dm.clearDS()
                self.dm.createDS()
                # create the local vector (memory chunk) and attach to original dm
                self._lvec = self.dm.createLocalVec()

            # push avar arrays into the parent dm array
            a_global = self.dm.getGlobalVec()

            # The field decomposition seems to fail if coarse DMs are present
            names, isets, dms = self.dm.createFieldDecomposition()

            # traverse subdms, taking user generated data in the subdm
            # local vec, pushing it into a global sub vec
            for var, subiset, subdm in zip(self.vars.values(), isets, dms):
                # var.vec lazily creates the PETSc local vector on first access
                lvec = var.vec
                subvec = a_global.getSubVector(subiset)
                subdm.localToGlobal(lvec, subvec, addv=False)
                a_global.restoreSubVector(subiset, subvec)

            for iset in isets:
                iset.destroy()
            for dm in dms:
                dm.destroy()

            self.dm.globalToLocal(a_global, self._lvec)
            self.dm.restoreGlobalVec(a_global)
            self._stale_lvec = False

    @property
    def lvec(self) -> PETSc.Vec:
        """
        Returns a local Petsc vector containing the flattened array
        of all the mesh variables.
        """
        if self._stale_lvec:
            raise RuntimeError("Mesh `lvec` needs to be updated using the update_lvec()` method.")
        return self._lvec

    def __del__(self):
        if hasattr(self, "_lvec") and self._lvec:
            self._lvec.destroy()

    def register_remesh_hook(self, op):
        """Register an operator's ``on_remesh(ctx)`` callback.

        Called by the adapt op (``smooth_mesh_interior``,
        ``follow_metric``) after the generic per-variable REMAP pass.
        ``op`` must expose an ``on_remesh(ctx)`` method; ``ctx`` is a
        :class:`~underworld3.discretisation.remesh.RemeshContext` with
        the old/new coords, total displacement, ``dt``, and a scratch
        dict for stashing things like ``v_mesh`` for the next solve.

        Stored as a weak reference so operators that go out of scope are
        cleaned up automatically. Idempotent: registering the same
        operator twice is a no-op.
        """
        import weakref as _wr
        # Drop any dead refs while we are here.
        self._remesh_hooks = [r for r in self._remesh_hooks
                              if (r() if isinstance(r, _wr.ReferenceType)
                                  else r) is not None]
        # De-dupe by identity.
        for r in self._remesh_hooks:
            cur = r() if isinstance(r, _wr.ReferenceType) else r
            if cur is op:
                return
        try:
            self._remesh_hooks.append(_wr.ref(op))
        except TypeError:
            # Some objects can't be weak-referenced (e.g. certain C
            # extension types). Store strongly as a fallback — the
            # caller takes responsibility for unregistering.
            self._remesh_hooks.append(op)

    def unregister_remesh_hook(self, op):
        """Drop an operator's ``on_remesh`` registration. Idempotent."""
        import weakref as _wr
        kept = []
        for r in self._remesh_hooks:
            cur = r() if isinstance(r, _wr.ReferenceType) else r
            if cur is None or cur is op:
                continue
            kept.append(r)
        self._remesh_hooks = kept

    # ------------------------------------------------------------------
    # Coordinate-mutation capability gate + sanctioned entry points
    # ------------------------------------------------------------------
    def _assert_coord_mutation_allowed(self):
        """Guard for :meth:`_deform_mesh`.

        Moving coordinates with the raw primitive skips the field /
        SL-DDt-history transfer. That is only safe (a) before the mesh
        carries any variables/history (construction, restart-before-solvers)
        or (b) inside a sanctioned scope — a remesh transaction
        (``_in_remesh_transfer``) or a ``_coord_mutation()`` scope opened by
        :meth:`deform`, :meth:`ephemeral_coords`, or a trusted internal mover.
        Outside those, on a live mesh, raise with a pointer to the public API.
        """
        allowed = (getattr(self, "_in_remesh_transfer", False)
                   or getattr(self, "_coord_mutation_depth", 0) > 0)
        if allowed:
            return
        has_state = bool(getattr(self, "vars", None)) or bool(
            getattr(self, "_remesh_hooks", None))
        if not has_state:
            return
        raise RuntimeError(
            "Mesh._deform_mesh() is an internal primitive — it moves nodes "
            "WITHOUT transferring fields or solver/DDt history onto the new "
            "layout, which corrupts the solution. This mesh already carries "
            "variables, so a direct call (or an in-place write to "
            "`mesh.X.coords`) is rejected.\n"
            "  Use instead:\n"
            "    • mesh.deform(new_coords, dt=…)        — impose an arbitrary "
            "node displacement (free surface / prescribed motion)\n"
            "    • mesh.remesh(metric) / mesh.adapt(metric_field)\n"
            "    • uw.meshing.smooth_mesh_interior(…) / uw.meshing.follow_metric(…)\n"
            "  These route the field + SL/DDt-history transfer "
            "(remesh_with_field_transfer). For trusted scheme-internal trial "
            "meshes that will be discarded, use `with mesh.ephemeral_coords(): …`."
        )

    @contextmanager
    def _coord_mutation(self):
        """Internal: sanction direct ``_deform_mesh`` calls within this scope.

        Re-entrant. Does NOT itself transfer fields — callers either are the
        transfer transaction (REMAP + ``on_remesh`` already run by
        ``remesh_with_field_transfer``), restore saved state separately, or
        intend an ephemeral trial (see :meth:`ephemeral_coords`).
        """
        self._coord_mutation_depth += 1
        try:
            yield
        finally:
            self._coord_mutation_depth -= 1

    @contextmanager
    def ephemeral_coords(self):
        """Trusted scheme-internal trial coordinate moves, restored on exit.

        For schemes (e.g. RK4 surface stages) that probe trial meshes to
        compute velocities/increments and must NOT commit a transfer. The
        coordinates are snapshotted on enter and restored on exit, so the
        intermediate meshes are genuinely ephemeral — only the final
        committed move (via :meth:`deform`) updates fields + history.
        """
        saved = numpy.asarray(self.X.coords).copy()
        self._coord_mutation_depth += 1
        try:
            yield
        finally:
            try:
                self._deform_mesh(saved)
            finally:
                self._coord_mutation_depth -= 1

    def deform(self, new_coords, *, dt=None, zero=None, verbose=False):
        """Move the mesh to ``new_coords``, transferring all fields + history.

        The public, foolproof way to impose an arbitrary node displacement
        (free surface, prescribed mesh motion). Wraps the remesh transaction
        so that REMAP variables are re-interpolated onto the new layout and
        every registered ``on_remesh`` hook fires — in particular the
        SemiLagrangian DDt's coherent ALE (carry its history stack + a
        one-step ``v_mesh = Δx/dt`` correction consumed by the next solve).

        Parameters
        ----------
        new_coords : ndarray
            Target vertex coordinates (shape of ``mesh.X.coords``).
        dt : float, optional
            Timestep for the ALE ``v_mesh = Δx/dt`` correction. Required when
            SemiLagrangian history is present and the move is advective (a
            free-surface step); omit for a pure geometric re-mesh.
        zero : list of MeshVariable, optional
            Variables to zero after the move (e.g. V, P for a cold restart).
        verbose : bool

        Returns
        -------
        bool
            True if the mesh moved (geometry changed), False for a no-op.
        """
        from underworld3.discretisation.remesh import remesh_with_field_transfer

        _nc = numpy.asarray(new_coords)

        def _do_move():
            self._deform_mesh(_nc)

        result = remesh_with_field_transfer(
            self, _do_move, dt=dt, extra_zero=zero, verbose=verbose)

        # Notify registered swarms: solve-entry sync compares this version
        # and marks them for deferred migration (#379 item 1 retired the
        # read-trigger that consumed this channel). The bump lives HERE,
        # not in _deform_mesh: the internal primitive also runs on
        # snapshot restore, whose integrity check treats a moved version
        # as invalidation and would refuse its own recovery.
        if result:
            self._mesh_version += 1

        # Refresh deformation-tracking per-boundary normals so Nitsche/penalty
        # BCs that captured ``boundary_normal(...).sym`` at setup read the new
        # geometry (the JIT reads the variable's .data, which would otherwise
        # hold the setup-time normal). Re-assemble each cached boundary normal.
        if getattr(self, "_boundary_normal_vars", None):
            for _nm, _var in list(self._boundary_normal_vars.items()):
                try:
                    self._assemble_boundary_normal(_var, _nm)
                except Exception:
                    # Sanctioned swallow: a normal refresh can fail for a
                    # boundary whose label vanished from the current DM
                    # (e.g. after region extraction); the deform itself is
                    # complete and must not be rolled back for one BC aid.
                    # Consequence of skipping: that Nitsche/penalty BC
                    # keeps its setup-time normal until next refresh.
                    pass
        # Likewise refresh the local cell-size field (Nitsche penalty scaling)
        # so its cell-constant data tracks the deformed geometry.
        if getattr(self, "_cell_size_variable", None) is not None:
            try:
                self._assemble_cell_size(self._cell_size_variable)
            except Exception:
                # Sanctioned swallow: same contract as the normal refresh
                # above — a failed size re-fill leaves the penalty scaling
                # one geometry behind rather than aborting a completed
                # deform. The next deform/adapt re-fills it.
                pass
        return result

    def _deform_mesh(self, new_coords: numpy.ndarray, verbose=False,
                     active_vars=None):
        """
        This method will update the mesh coordinates and reset any cached coordinates in
        the mesh and in equation systems that are registered on the mesh.

        The coord array that is passed in should match the shape of self.data

        ``active_vars`` (optional): restrict the per-variable DOF
        coordinate-cache recomputation in
        :meth:`nuke_coords_and_rebuild` to this set of variables. The
        default ``None`` preserves today's behaviour — every registered
        variable's coord cache is recomputed eagerly, which is the
        BUGFIX(#130) collective-safe path. Movers that opt in pass their
        own work-vars during the inner sweep (skipping non-mover-var
        recompute n_outer×); the wrapper does a full recompute once at
        sweep exit by calling ``_deform_mesh`` again with
        ``active_vars=None``.

        .. warning::

           This is the **internal** coordinate-mutation primitive. It moves
           nodes WITHOUT transferring fields or solver/DDt history onto the
           new layout. It may only be called inside a sanctioned coordinate-
           mutation scope (see :meth:`deform`, :meth:`ephemeral_coords`, and
           ``remesh_with_field_transfer``). A bare call on a mesh that already
           carries variables raises — use the public methods instead.
        """

        self._assert_coord_mutation_allowed()

        with self._mesh_update_lock:
            coord_vec = self.dm.getCoordinatesLocal()
            coords = coord_vec.array.reshape(-1, self.cdim)
            coords[...] = new_coords[...]

            self.dm.setCoordinatesLocal(coord_vec)
            self.nuke_coords_and_rebuild(active_vars=active_vars)

            # Rebuild the _coords array view.  nuke_coords_and_rebuild may
            # replace the coordinate vector internally (createCoordinateSpace),
            # leaving self._coords as a stale numpy view of the old buffer.
            import underworld3.utilities
            old_callbacks = getattr(self._coords, "_callbacks", [])
            self._coords = underworld3.utilities.NDArray_With_Callback(
                numpy.ndarray.view(
                    self.dm.getCoordinatesLocal().array.reshape(-1, self.cdim)
                ),
                owner=self,
            )
            for cb in old_callbacks:
                original = getattr(cb, "_wrapped", None)
                if original is not None:
                    # Canonical dispatch wrappers are bound (by weakref) to
                    # the OLD array's identity — copied verbatim they never
                    # fire on the replacement, so a SECOND coords write
                    # would silently do nothing (round-2 review; same
                    # class as the sync_data re-homing). Re-register the
                    # original function against the new canonical.
                    self._coords.add_canonical_callback(original)
                else:
                    self._coords.add_callback(cb)

            # BUGFIX(#122): mark registered solvers for rebuild. Since PR #127
            # ("Trust JIT cache: skip DM rebuild on constant-only parameter
            # changes") a solver with is_setup=True trusts its cached PETSc DM
            # / SNES assembly and skips rebuild on the next solve(). After a
            # coordinate change the cached DM still carries pre-deform
            # coordinates, so F(v_prev) ≈ 0 and the solver converges in 0
            # iterations without updating the solution. mesh.adapt() already
            # does this; _deform_mesh must match.
            for solver in self._equation_systems_register:
                if solver is not None and hasattr(solver, "is_setup"):
                    solver.is_setup = False

            # Invalidate caches whose contents become stale when mesh
            # coordinates change. Matches the cache hygiene already
            # performed by mesh.adapt() and _legacy_access. Without
            # these, uw.function.evaluate (and any user code that keys
            # lookups off _topology_version) can return values
            # computed against the pre-deform mesh.
            self._evaluation_hash = None
            self._evaluation_interpolated_results = None
            if hasattr(self, '_dminterpolation_cache'):
                self._dminterpolation_cache.invalidate_all(
                    reason="mesh deformed")
            self._topology_version += 1

            # Propagate coordinate changes to registered submeshes
            for submesh in self._registered_submeshes:
                submesh.sync_coordinates_from_parent()

        return

    def _legacy_access(self, *writeable_vars: "MeshVariable"):
        """
        This context manager makes the underlying mesh variables data available to
        the user. The data should be accessed via the variables `data` handle.

        As default, all data is read-only. To enable writeable data, the user should
        specify which variable they wish to modify.

        Parameters
        ----------
        writeable_vars
            The variables for which data write access is required.

        Example
        -------
        Legacy pattern only — new code writes ``var.data[...]`` directly
        (see docs/developer/subsystems/data-access.md)::

            >>> import underworld3 as uw
            >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
            >>> T = uw.discretisation.MeshVariable("T", mesh, 1)
            >>> with mesh._legacy_access(T):
            ...     T.data[:, 0] = 1.0
        """

        import time

        timing._incrementDepth()
        stime = time.time()

        if writeable_vars is not None:
            self._evaluation_hash = None
            self._evaluation_interpolated_results = None

            # Invalidate DMInterpolation cache when DM structure changes
            self._topology_version += 1
            self._dminterpolation_cache.invalidate_all("DM rebuilt with new variables")

        self._dm_initialized = True
        deaccess_list = []
        for var in self.vars.values():
            # if already accessed within higher level context manager, continue.
            if var._is_accessed == True:
                continue

            # set flag so variable status can be known elsewhere
            var._is_accessed = True
            # add to de-access list to rewind this later
            deaccess_list.append(var)

            # create & set vec
            var._set_vec(available=True)

            # grab numpy object, setting read only if necessary
            var._data = var.vec.array.reshape(-1, var.num_components)

            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()

            # make view for each var component

            for i in range(0, var.shape[0]):
                for j in range(0, var.shape[1]):
                    # var._data_ij[i, j] = var.data[:, var._data_layout(i, j)]
                    var._data_container[i, j] = var._data_container[i, j]._replace(
                        data=var.data[:, var._data_layout(i, j)],
                    )

        class exit_manager:
            def __init__(self, mesh):
                self.mesh = mesh

            def __enter__(self):
                pass

            def __exit__(self, *args):
                for var in self.mesh.vars.values():
                    # only de-access variables we have set access for.
                    if var not in deaccess_list:
                        continue
                    # set this back, although possibly not required.
                    if var not in writeable_vars:
                        var._data.flags.writeable = var._old_data_flag
                    # perform sync for any modified vars.

                    if var in writeable_vars:
                        indexset, subdm = self.mesh.dm.createSubDM(var.field_id)

                        # sync ghost values
                        subdm.localToGlobal(var.vec, var._gvec, addv=False)
                        subdm.globalToLocal(var._gvec, var.vec, addv=False)

                        indexset.destroy()
                        subdm.destroy()
                        self.mesh._stale_lvec = True

                    var._data = None
                    var._set_vec(available=False)
                    var._is_accessed = False

                    for i in range(0, var.shape[0]):
                        for j in range(0, var.shape[1]):
                            var._data_container[i, j] = var._data_container[i, j]._replace(
                                data=f"MeshVariable[...].data is only available within mesh.access() context",
                            )

                timing._decrementDepth()
                timing.log_result(time.time() - stime, "Mesh.access", 1)

        return exit_manager(self)

    def access(self, *writeable_vars: "MeshVariable"):
        """
        Dummy access manager that provides deferred sync for backward compatibility.
        Uses NDArray_With_Callback.delay_callbacks_global() internally.

        This is a compatibility wrapper that allows existing code using the access()
        context manager to work with the new direct-access variable interfaces.
        All variable modifications are deferred and synchronized at context exit.

        Parameters
        ----------
        writeable_vars
            Variables that will be modified (ignored - all variables are writable
            with the new interface, this parameter is kept for API compatibility)

        Returns
        -------
        Context manager that defers variable synchronization until exit

        Notes
        -----
        This method is deprecated. New code should access variable.data or
        variable.array directly without requiring an access context.
        """
        import underworld3.utilities

        class DummyAccessContext:
            def __init__(self, mesh, writeable_vars):
                self.mesh = mesh
                self.writeable_vars = writeable_vars
                self.delay_context = None

            def __enter__(self):
                # Use NDArray_With_Callback global delay context for deferred sync
                self.delay_context = (
                    underworld3.utilities.NDArray_With_Callback.delay_callbacks_global(
                        "mesh.access compatibility"
                    )
                )
                return self.delay_context.__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                # This triggers all accumulated callbacks from all variables
                if self.delay_context:
                    return self.delay_context.__exit__(exc_type, exc_val, exc_tb)
                return False

        return DummyAccessContext(self, writeable_vars)

    @property
    def N(self) -> sympy.vector.CoordSys3D:
        r"""SymPy coordinate system for symbolic calculus.

        The base coordinate system used for gradient, divergence, and
        curl operations. Access base scalars via ``mesh.N.x``, ``mesh.N.y``,
        ``mesh.N.z`` and base vectors via ``mesh.N.i``, ``mesh.N.j``, ``mesh.N.k``.

        Returns
        -------
        sympy.vector.CoordSys3D
            The SymPy coordinate system object.

        See Also
        --------
        X : Coordinate system with data access.
        r : Tuple of coordinate scalars.
        """
        return self._N

    @property
    def Gamma_N(self) -> sympy.Matrix:
        r"""Normalised boundary/surface normal as a row matrix.

        Returns ``Gamma / |Gamma|`` so that the result is a unit normal
        regardless of element size. Use this for penalty and Nitsche BCs
        where mesh-independent scaling is required.

        Returns
        -------
        sympy.Matrix
            Row matrix of normalised boundary normal components.
        """
        G = self.Gamma
        return G / sympy.sqrt(G.dot(G))

    @property
    def Gamma(self) -> sympy.Matrix:
        r"""Raw (un-normalised) boundary coordinate scalars as a row matrix.

        The magnitude scales with face edge length (2D) or face area (3D).
        For a unit normal, use :attr:`Gamma_N` instead.

        Returns
        -------
        sympy.Matrix
            Row matrix of boundary coordinate scalars.
        """
        return sympy.Matrix(self._Gamma.base_scalars()[0 : self.cdim]).T

    @property
    def X(self):
        r"""Coordinate system with symbolic coordinates and data access.

        The primary interface for mesh coordinates, providing both symbolic
        expressions for equations and numerical data for evaluation.

        Returns
        -------
        CoordinateSystem
            Coordinate system object with:

            - ``mesh.X[0]``, ``mesh.X[1]``: Symbolic coordinate functions
            - ``mesh.X.coords``: Coordinate data array (vertex positions)
            - ``mesh.X.units``: Coordinate units
            - ``x, y = mesh.X``: Unpack symbolic coordinates

        Examples
        --------
        >>> x, y = mesh.X  # Symbolic coordinates for equations
        >>> coords = mesh.X.coords  # Numerical vertex positions

        See Also
        --------
        N : SymPy coordinate system for vector calculus.
        """
        return self._CoordinateSystem

    @property
    def CoordinateSystem(self) -> CoordinateSystem:
        r"""Alias for :attr:`X` (the coordinate system object)."""
        return self._CoordinateSystem

    @property
    def t(self):
        r"""Symbolic time coordinate.

        PETSc passes a time value (``petsc_t``) to all pointwise residual
        and Jacobian functions. Use ``mesh.t`` in expressions to reference
        this time without forcing JIT recompilation each timestep.

        The low-level PETSc solver accepts ``time=t`` to set the value
        of ``petsc_t`` for pointwise functions. If not provided, ``petsc_t``
        defaults to 0. Note: the high-level Python ``solve()`` wrappers
        do not yet pass ``time=`` through — set it directly via
        ``UW_DMSetTime`` at the Cython level if needed.

        When the scaling system is active, ``mesh.t`` carries time units
        (derived from the model's time scale) so that dimensional analysis
        works correctly in expressions.

        Examples
        --------
        >>> omega = 2 * np.pi / period
        >>> stokes.add_dirichlet_bc((V0 * sympy.sin(omega * mesh.t), 0.0), "Top")
        >>> stokes.solve(time=current_time)   # sets petsc_t before SNES
        """
        return self._t

    @property
    def nullspace_rotations(self):
        """Symbolic velocity fields for rigid-body rotation null modes.

        Returns a list of SymPy Matrix expressions in mesh Cartesian
        coordinates. Empty for meshes with no rotation nullspace (boxes,
        wedge segments with walls). Set by mesh factory functions for
        closed surfaces (annulus, spherical shell, etc.).

        Each entry represents a rigid rotation: v = omega x r.

        Returns
        -------
        list of sympy.Matrix
            Velocity fields for each independent rotation mode.

        Examples
        --------
        >>> annulus = uw.meshing.Annulus(...)
        >>> annulus.nullspace_rotations  # [Matrix([-y, x])]
        >>> shell = uw.meshing.SphericalShell(...)
        >>> shell.nullspace_rotations   # 3 rotation matrices
        """
        return self._nullspace_rotations

    @property
    def r(self) -> Tuple[sympy.vector.BaseScalar]:
        r"""Tuple of coordinate scalars :math:`(x, y)` or :math:`(x, y, z)`.

        Returns
        -------
        tuple
            Tuple of SymPy base scalars ``(N.x, N.y[, N.z])``.

        See Also
        --------
        rvec : Position vector form.
        """
        return self._N.base_scalars()[0 : self.cdim]

    @property
    def rvec(self) -> sympy.vector.Vector:
        r"""Position vector :math:`\mathbf{r} = x\hat{i} + y\hat{j} [+ z\hat{k}]`.

        Returns
        -------
        sympy.vector.Vector
            The position vector in the mesh coordinate system.
        """
        N = self.N

        r_vec = sympy.vector.Vector.zero

        N_s = N.base_scalars()
        N_v = N.base_vectors()
        for i in range(self.cdim):
            r_vec += N_s[i] * N_v[i]

        return r_vec

    @property
    def data(self) -> numpy.ndarray:
        """
        The array of mesh element vertex coordinates.

        .. deprecated:: 0.99.0
            Use :attr:`X.coords` instead.
            ``mesh.data`` is deprecated in favor of ``mesh.X.coords``
            (coordinate-system-aware interface).

        This is an alias for mesh.points (which is also deprecated).
        """
        import warnings

        warnings.warn(
            "mesh.data is deprecated, use mesh.X.coords instead", DeprecationWarning, stacklevel=2
        )
        return self.X.coords

    @property
    def points(self):
        """
        Mesh node coordinates in physical units.

        .. deprecated:: 0.99.0
            Use :attr:`X.coords` instead.
            ``mesh.points`` is deprecated in favor of ``mesh.X.coords``
            (coordinate-system-aware interface).

        When the mesh has coordinate scaling applied (via model units),
        this property automatically converts from internal model coordinates
        to physical coordinates for user access.

        When the mesh has coordinate units specified, returns a unit-aware array.

        Returns:
            numpy.ndarray or UnitAwareArray: Node coordinates (with units if specified)
        """
        import warnings

        warnings.warn(
            "mesh.points is deprecated, use mesh.X.coords instead", DeprecationWarning, stacklevel=2
        )

        model_coords = self._coords

        # Apply scaling to convert model coordinates to physical coordinates
        if hasattr(self.CoordinateSystem, "_scaled") and self.CoordinateSystem._scaled:
            scale_factor = self.CoordinateSystem._length_scale
            coords = model_coords * scale_factor
        else:
            coords = model_coords

        # Wrap with unit-aware array if units are specified
        if self.units is not None:
            from underworld3.utilities.unit_aware_array import UnitAwareArray

            return UnitAwareArray(coords, units=self.units)

        return coords

    @points.setter
    def points(self, value):
        """Removed. Move mesh nodes with :meth:`deform`.

        The deprecated setter rebound ``self._coords`` to a plain ndarray,
        silently discarding the ``NDArray_With_Callback`` wrapper: the
        deform callback never fired, so PETSc coordinates, kd-trees and
        dependent caches were never updated — writes looked accepted but
        changed nothing downstream (2026-07 audit, BF-18 / READ-43).
        Rather than repair an already-deprecated write path, it now
        refuses loudly.
        """
        raise AttributeError(
            "Assigning to mesh.points has been removed: it never propagated "
            "the new coordinates to PETSc (2026-07 audit, BF-18). "
            "Use mesh.deform(new_coords) to move mesh nodes; read "
            "coordinates via mesh.X.coords."
        )

    @property
    def physical_coordinates(self):
        """
        Mesh coordinates in physical units.

        Returns the mesh coordinate array scaled to physical units using
        the model's length scale. Requires the mesh to be associated with
        a model that has reference quantities set.

        Returns
        -------
        UWQuantity or None
            Coordinates in physical units, or None if no model scaling available

        Examples
        --------
        >>> model.set_reference_quantities(domain_length=1000*uw.units.km, ...)
        >>> mesh = uw.meshing.StructuredQuadBox(...)
        >>> physical_coords = mesh.physical_coordinates  # In kilometers
        """
        if not hasattr(self, "_model") or self._model is None:
            return None

        return self._model.scale_to_physical(self.points, dimension="length")

    @property
    def physical_bounds(self):
        """
        Mesh bounds in physical units.

        Returns the mesh bounding box scaled to physical units using
        the model's length scale.

        Returns
        -------
        tuple of UWQuantity or None
            (min_coords, max_coords) in physical units, or None if no model scaling

        Examples
        --------
        >>> physical_min, physical_max = mesh.physical_bounds
        >>> print(f"Domain: {physical_min} to {physical_max}")
        """
        if not hasattr(self, "_model") or self._model is None:
            return None

        import numpy as np

        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)

        return (
            self._model.scale_to_physical(min_coords, dimension="length"),
            self._model.scale_to_physical(max_coords, dimension="length"),
        )

    @property
    def physical_extent(self):
        """
        Mesh spatial extent in physical units.

        Returns the mesh size (max - min) in each dimension scaled to physical units.

        Returns
        -------
        UWQuantity or None
            Extent in physical units, or None if no model scaling

        Examples
        --------
        >>> extent = mesh.physical_extent
        >>> print(f"Domain size: {extent}")
        """
        if not hasattr(self, "_model") or self._model is None:
            return None

        import numpy as np

        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        extent = max_coords - min_coords

        return self._model.scale_to_physical(extent, dimension="length")

    @timing.routine_timer_decorator
    def write_timestep(
        self,
        filename: str,
        index: int,
        outputPath: Optional[str] = "",
        meshVars: Optional[list] = [],
        swarmVars: Optional[list] = [],
        meshUpdates: bool = False,
        create_xdmf: bool = True,
        petsc_reload: bool = False,
    ):
        """
        Write mesh and selected variables for timestep output.

        This is the standard mesh output method. It always writes:

        - one mesh HDF5 file, shared across timesteps unless ``meshUpdates=True``
        - one HDF5 file per mesh variable
        - raw coordinate/value datasets under ``/fields`` for coordinate-based
          reload with ``MeshVariable.read_timestep()``

        The optional payloads are controlled explicitly:

        - ``create_xdmf=True`` writes ParaView/XDMF output. Variable files also
          receive ``/vertex_fields`` or ``/cell_fields`` compatibility groups,
          and rank 0 writes the companion ``.xdmf`` file.
        - ``petsc_reload=True`` writes PETSc DMPlex section/vector metadata into
          the same per-variable HDF5 files. These files can then be loaded with
          ``MeshVariable.read_checkpoint()`` for PETSc-native same-mesh reload.

        Common choices are:

        - visualisation/remap only:
          ``create_xdmf=True, petsc_reload=False``
        - PETSc-native reload only:
          ``create_xdmf=False, petsc_reload=True``
        - unified visualisation/remap and PETSc reload:
          ``create_xdmf=True, petsc_reload=True``

        With both flags enabled, the same variable HDF5 file can be used by
        ``MeshVariable.read_timestep()`` for coordinate/KDTree remapping and by
        ``MeshVariable.read_checkpoint()`` for exact PETSc-native reload.

        Parameters
        ----------
        filename
            Output filename base. Files are written as
            ``<filename>.mesh.<index>.h5`` and
            ``<filename>.mesh.<variable>.<index>.h5``.
        index
            Timestep/output index used in generated filenames.
        outputPath
            Directory where output files are written.
        meshVars
            Mesh variables to write.
        swarmVars
            Swarm variables to write as proxy fields.
        meshUpdates
            If ``False``, reuse ``<filename>.mesh.00000.h5`` when it already
            exists. If ``True``, write an indexed mesh file for this timestep.
        create_xdmf
            Write ParaView/XDMF-compatible datasets and companion XDMF file.
        petsc_reload
            Write PETSc DMPlex section/vector metadata for reload with
            ``MeshVariable.read_checkpoint()``.

        """
        options = PETSc.Options()
        options.setValue("viewer_hdf5_sp_output", True)
        options.setValue("viewer_hdf5_collective", False)

        output_base_name = os.path.join(outputPath, filename)

        # Fail early, with the absolute path in the message, if the output
        # directory is missing or read-only — clearer than a mid-write error.
        abs_dir = os.path.abspath(os.path.dirname(output_base_name))
        if not os.path.exists(abs_dir):
            raise RuntimeError(f"{abs_dir} does not exist")
        if not os.access(abs_dir, os.W_OK):
            raise RuntimeError(f"No write access to {abs_dir}")

        # Checkpoint the mesh file itself if required

        if not meshUpdates:
            from pathlib import Path

            mesh_file = output_base_name + ".mesh.00000.h5"
            path = Path(mesh_file)
            if not path.is_file():
                self.write(mesh_file)

        else:
            mesh_file = output_base_name + f".mesh.{index:05}.h5"
            self.write(mesh_file)

        variables = []
        if meshVars is not None:
            for var in meshVars:
                save_location = output_base_name + f".mesh.{var.clean_name}.{index:05}.h5"
                var.write(save_location)
                if create_xdmf:
                    _write_compat_groups(self, var, save_location)
                variables.append((var, save_location))

        if swarmVars is not None:
            for svar in swarmVars:
                save_location = output_base_name + f".proxy.{svar.clean_name}.{index:05}.h5"
                svar.write_proxy(save_location)
                if petsc_reload:
                    variables.append((svar._meshVar, save_location))

        if petsc_reload:
            for var, save_location in variables:
                self._write_petsc_reload_file(save_location, [var], mode="a")

        if create_xdmf and uw.mpi.rank == 0:
            checkpoint_xdmf(
                output_base_name,
                meshUpdates,
                meshVars,
                swarmVars,
                index,
            )

        return

    @timing.routine_timer_decorator
    def petsc_save_checkpoint(
        self,
        index: int,
        meshVars: Optional[list] = [],
        outputPath: Optional[str] = "",
    ):
        """Save the mesh and mesh variables to HDF5 with XDMF.

        This is a convenience wrapper around ``write_timestep()`` that
        provides the simpler interface used by earlier Underworld3 code.
        Output uses the same per-variable file layout and XDMF generation
        (including vertex/cell compatibility groups, field projection, and
        tensor repacking) as ``write_timestep()``.

        Parameters
        ----------
        meshVars :
            List of UW mesh variables to save. If left empty then just
            the mesh is saved.
        index :
            An index which might correspond to the timestep or output
            number (for example).
        outputPath :
            Path to save the data. If left empty it will save the data
            in the current working directory.
        """

        if meshVars is not None and not isinstance(meshVars, list):
            raise RuntimeError("`meshVars` does not appear to be a list.")

        # Split outputPath into directory and filename base for write_timestep().
        # Old callers pass outputPath like './output/' or './output/run_name'.
        import os

        outputPath = outputPath or ""
        if outputPath.endswith(os.sep) or outputPath.endswith("/"):
            # Directory only — use 'checkpoint' as the file base name
            directory = outputPath
            filename = "checkpoint"
        elif os.sep in outputPath or "/" in outputPath:
            # Path with filename component
            directory = os.path.dirname(outputPath)
            filename = os.path.basename(outputPath)
        else:
            # Bare name, no directory
            directory = ""
            filename = outputPath if outputPath else "checkpoint"

        self.write_timestep(
            filename=filename,
            index=index,
            outputPath=directory,
            meshVars=meshVars if meshVars is not None else [],
            swarmVars=[],
            meshUpdates=False,
            create_xdmf=True,
        )

    def _write_petsc_reload_variable(self, viewer, var):
        """Write one variable's PETSc DMPlex reload metadata to ``viewer``."""

        if var._lvec is None:
            var._set_vec(available=True)

        iset, subdm = self.dm.createSubDM(var.field_id)
        subdm.setName(var.clean_name)
        old_lvec_name = var._lvec.getName()

        try:
            var._lvec.setName(var.clean_name)
            self.dm.sectionView(viewer, subdm)
            self.dm.localVectorView(viewer, subdm, var._lvec)
        finally:
            var._lvec.setName(old_lvec_name)
            iset.destroy()
            subdm.destroy()

    def _write_petsc_reload_file(self, checkpoint_file, variables, mode="w"):
        """Write PETSc DMPlex section/vector reload metadata."""

        old_dm_name = self.dm.getName()
        self.dm.setName("uw_mesh")

        viewer = PETSc.ViewerHDF5().create(
            checkpoint_file, mode, comm=PETSc.COMM_WORLD
        )
        viewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
        try:
            self.dm.sectionView(viewer, self.dm)

            for var in variables:
                self._write_petsc_reload_variable(viewer, var)

            uw.mpi.barrier()
        finally:
            viewer.popFormat()
            viewer.destroy()
            if old_dm_name is not None:
                self.dm.setName(old_dm_name)

    @timing.routine_timer_decorator
    def write_checkpoint(
        self,
        filename: str,
        outputPath: str = "",
        meshUpdates: bool = True,
        meshVars: Optional[list] = [],
        swarmVars: Optional[list] = [],
        index: Optional[int] = 0,
        unique_id: Optional[bool] = False,
        separate_variable_files: bool = True,
        create_xdmf: bool = False,
    ):
        """Compatibility wrapper for PETSc DMPlex reload output.

        This method is retained for existing callers. New code should use
        ``write_timestep(..., petsc_reload=True)`` so all mesh-variable output
        goes through the standard timestep writer. By default this compatibility
        method writes PETSc DMPlex section/vector metadata required for exact
        parallel reload and does not write XDMF or vertex-field visualisation
        datasets. Use ``create_xdmf=True`` to route through the unified
        timestep-style output path.

        Parameters
        ----------
        filename
            Checkpoint base filename. With ``outputPath`` unset, this may include
            a directory. With ``outputPath`` set, it is joined to that directory.
        outputPath
            Optional output directory, matching the ``write_timestep()`` style.
        meshUpdates
            If ``False``, write the mesh checkpoint only when it does not already
            exist. If ``True``, always write the indexed mesh checkpoint.
        meshVars, swarmVars
            Variables to write into checkpoint files.
        index
            Checkpoint index used in output filenames.
        unique_id
            Preserve existing unique-rank filename behaviour for checkpoint data.
        separate_variable_files
            If ``True`` (default), write one file per variable:
            ``<base>.<variable>.<index>.h5``. If ``False``, write all variables
            into one file: ``<base>.checkpoint.<index>.h5``.
        create_xdmf
            If ``True``, route through ``write_timestep()`` and write XDMF,
            vertex/cell compatibility groups, coordinate/KDTree remap data,
            and PETSc reload metadata. The output uses the timestep filename
            convention ``<base>.mesh.<variable>.<index>.h5``. This mode does
            not support ``unique_id=True`` or ``separate_variable_files=False``.
        """
        import warnings

        warnings.warn(
            "write_checkpoint() is deprecated and retained for compatibility. "
            "Use write_timestep(..., petsc_reload=True) for PETSc reload output; "
            "set create_xdmf=True when visualization/remap payloads are also "
            "needed.",
            FutureWarning,
            stacklevel=2,
        )

        if outputPath:
            filename = os.path.join(outputPath, filename)

        if create_xdmf:
            if unique_id:
                raise RuntimeError(
                    "write_checkpoint(create_xdmf=True) uses write_timestep() "
                    "layout and does not support unique_id=True."
                )
            if not separate_variable_files:
                raise RuntimeError(
                    "write_checkpoint(create_xdmf=True) uses per-variable "
                    "timestep files and does not support "
                    "separate_variable_files=False."
                )
            output_dir = os.path.dirname(filename)
            output_name = os.path.basename(filename)
            self.write_timestep(
                output_name,
                index=index,
                outputPath=output_dir,
                meshVars=meshVars,
                swarmVars=swarmVars,
                meshUpdates=meshUpdates,
                create_xdmf=True,
                petsc_reload=True,
            )
            return

        def _checkpoint_filename(var_name=None):
            variable_part = f".{var_name}" if var_name is not None else ".checkpoint"
            if unique_id:
                return filename + f"{uw.mpi.unique}{variable_part}.{index:05}.h5"
            return filename + f"{variable_part}.{index:05}.h5"

        old_dm_name = self.dm.getName()
        self.dm.setName("uw_mesh")

        try:
            with _temporary_petsc_option("dm_plex_view_hdf5_storage_version", "3.0.0"):
                # The mesh checkpoint is the same as the one required for visualisation

                if not meshUpdates:
                    from pathlib import Path

                    mesh_file = filename + f".mesh.{index:05}.h5"
                    path = Path(mesh_file)
                    if not path.is_file():
                        self.write(mesh_file, petsc_format=True)

                else:
                    self.write(filename + f".mesh.{index:05}.h5", petsc_format=True)

                variables = []
                if meshVars is not None:
                    variables.extend(meshVars)
                if swarmVars is not None:
                    variables.extend(svar._meshVar for svar in swarmVars)

                if separate_variable_files:
                    for var in variables:
                        self._write_petsc_reload_file(
                            _checkpoint_filename(var.clean_name), [var], mode="w"
                        )
                else:
                    self._write_petsc_reload_file(
                        _checkpoint_filename(), variables, mode="w"
                    )
        finally:
            if old_dm_name is not None:
                self.dm.setName(old_dm_name)

    # ----- Unitary snapshot / restore -----
    #
    # See ``src/underworld3/checkpoint/snapshot.py`` and
    # ``docs/developer/design/in_memory_checkpoint_design.md``. v1
    # captures deformed coords + per-MV global-vector DOFs; v1.2 will
    # add topology / section capture so the DM can be rebuilt on
    # restore after ``mesh.adapt()``.

    def snapshot_payload(self) -> dict:
        """Return a self-contained dict describing this mesh's state.

        The returned dict is consumed by
        :mod:`underworld3.checkpoint.snapshot` capture. Keys:

        - ``name``: stable string identifier for the mesh.
        - ``mesh_version``: current ``_mesh_version`` integer.
        - ``coords``: deformed mesh coordinates (numpy array).
        - ``vars``: ``{var.clean_name: gvec_array.copy()}`` for every
          mesh variable on this mesh.

        v1.2 will additionally populate a ``topology`` key with
        section / DM-topology data sufficient to rebuild the DM on
        restore.
        """
        coords = numpy.asarray(self.X.coords).copy()
        var_arrays: dict[str, numpy.ndarray] = {}
        for var in self.vars.values():
            var._sync_lvec_to_gvec()
            # Variables created but never touched have _gvec=None (lazy
            # allocation in MeshVariable). They carry no data so they
            # contribute nothing to the snapshot — skip cleanly.
            if var._gvec is None:
                continue
            var_arrays[var.clean_name] = numpy.asarray(var._gvec.array).copy()
        return {
            "name": self.name,
            "mesh_version": int(getattr(self, "_mesh_version", 0)),
            "coords": coords,
            "vars": var_arrays,
        }

    def apply_snapshot_payload(self, payload: dict) -> None:
        """Restore this mesh from a payload produced by :meth:`snapshot_payload`.

        v1 implementation writes coordinates and per-variable DOFs
        back in place. The captured DOF arrays must match the current
        section, which means ``_mesh_version`` must equal the captured
        value — mesh-adapt during the interval would have resized the
        section and is detected as a v1 refusal here.

        v1.2 will replace the ``_mesh_version`` refusal with a
        rebuild-from-payload path: destroy the current DM, rebuild
        from ``payload["topology"]``, allocate vectors, write DOFs,
        and re-bind MeshVariable / Swarm wrappers. The interface stays
        the same; only this method's body changes.
        """
        from underworld3.checkpoint.snapshot import SnapshotInvalidatedError

        current_version = int(getattr(self, "_mesh_version", 0))
        captured_version = int(payload["captured_mesh_version"])
        if current_version != captured_version:
            raise SnapshotInvalidatedError(
                f"mesh {self.name!r}: _mesh_version moved from "
                f"{captured_version} to {current_version} since snapshot. "
                f"mesh.adapt() rebuild on restore is scheduled for v1.2; "
                f"v1 refuses rather than corrupt the DOF arrays"
            )

        coords = numpy.asarray(payload["coords"])
        expected_shape = numpy.asarray(self.X.coords).shape
        if coords.shape != expected_shape:
            raise SnapshotInvalidatedError(
                f"mesh {self.name!r}: coordinate shape changed "
                f"({coords.shape} vs current {expected_shape}); programming "
                f"error since _mesh_version matched"
            )
        # Snapshot restore: variables are reloaded just below, so this is a
        # sanctioned internal coordinate move (no live-state transfer needed).
        with self._coord_mutation():
            self._deform_mesh(coords)

        current_vars = {var.clean_name: var for var in self.vars.values()}
        for var_clean_name, saved_array in payload["vars"].items():
            var = current_vars.get(var_clean_name)
            if var is None:
                raise SnapshotInvalidatedError(
                    f"mesh {self.name!r}: variable {var_clean_name!r} "
                    f"from snapshot is no longer present"
                )
            var._sync_lvec_to_gvec()
            current_shape = numpy.asarray(var._gvec.array).shape
            if saved_array.shape != current_shape:
                raise SnapshotInvalidatedError(
                    f"mesh {self.name!r}: variable {var_clean_name!r} gvec "
                    f"shape changed ({saved_array.shape} vs current "
                    f"{current_shape})"
                )
            var._gvec.array[...] = saved_array
            iset, subdm = self.dm.createSubDM(var.field_id)
            subdm.globalToLocal(var._gvec, var._lvec, addv=False)
            iset.destroy()
            subdm.destroy()
            self._stale_lvec = True

    @timing.routine_timer_decorator
    def write(
        self,
        filename: str,
        index: Optional[int] = None,
        petsc_format: Optional[bool] = None,
    ):
        """
        Save mesh data to the specified hdf5 file.


        Parameters
        ----------
        filename :
            The filename for the mesh checkpoint file.
        index :
            Not yet implemented. An optional index which might
            correspond to the timestep (for example).
        petsc_format :
            If True, force PETSc DMPlex HDF5 checkpoint/restart topology.
            If False, force PETSc HDF5_VIZ topology only.
            If None, use PETSc's default HDF5 layout, which includes the
            restart-style topology and labels as well as visualization
            topology for XDMF.

        """

        if index is not None:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via
            ## the PETSc xdmf script.
            # viewer.pushTimestepping(viewer)
            # viewer.setTimestep(index)

        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=PETSc.COMM_WORLD)
        try:
            if petsc_format is not None:
                viewer_format = (
                    PETSc.Viewer.Format.HDF5_PETSC
                    if petsc_format
                    else PETSc.Viewer.Format.HDF5_VIZ
                )
                viewer.pushFormat(viewer_format)
            viewer(self.dm)
        finally:
            if petsc_format is not None:
                viewer.popFormat()
            viewer.destroy()

        ## Add boundary metadata to the file

        import h5py, json

        # Use preferred selective_ranks pattern for metadata operations
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                f = h5py.File(filename, "a")
                g = f.create_group("metadata")

                boundaries_dict = {i.name: i.value for i in self.boundaries}
                g.attrs["boundaries"] = json.dumps(boundaries_dict)

                if self.regions is not None:
                    regions_dict = {i.name: i.value for i in self.regions}
                    g.attrs["regions"] = json.dumps(regions_dict)

                coordinates_type_dict = {
                    "name": self.CoordinateSystemType.name,
                    "value": self.CoordinateSystemType.value,
                }
                g.attrs["coordinate_system_type"] = json.dumps(coordinates_type_dict)

                # Save ellipsoid metadata for geographic meshes
                if hasattr(self.CoordinateSystem, "ellipsoid"):
                    ellipsoid_ser = {}
                    for k, v in self.CoordinateSystem.ellipsoid.items():
                        if hasattr(v, "to"):  # uw.quantity
                            ellipsoid_ser[k] = {
                                "value": float(v.magnitude),
                                "unit": str(v.units),
                            }
                        else:
                            ellipsoid_ser[k] = v
                    g.attrs["ellipsoid"] = json.dumps(ellipsoid_ser)

                # Add coordinate units metadata
                if hasattr(self, "coordinate_units"):
                    coord_units_dict = {
                        "coordinate_units": str(self.coordinate_units),
                        "coordinate_dimensionality": (
                            str(self.coordinate_dimensionality)
                            if hasattr(self, "coordinate_dimensionality")
                            else None
                        ),
                        "length_scale": (
                            str(self.length_scale) if hasattr(self, "length_scale") else None
                        ),
                        "mesh_type": type(self).__name__,
                        "dimension": self.dim,
                    }
                    g.attrs["coordinate_units"] = json.dumps(coord_units_dict)

                # Number of coarse multigrid levels in the hierarchy (= number
                # of refinements from the stored coarsest level up to the fine
                # mesh). Used on reload to rebuild the intermediate levels.
                g.attrs["hierarchy_coarse_levels"] = len(self.dm_hierarchy) - 1

                f.close()

        # Persist the geometric-multigrid (FMG) hierarchy as a SINGLE sidecar
        # holding the coarsest level only. On reload the intermediate coarse
        # levels are rebuilt by refining it (they come back canonically numbered,
        # which is all the co-located nested interpolation needs). Without this
        # file a reloaded mesh has a single level and falls back to GAMG. One
        # single-DM HDF5 file (PETSc's HDF5_PETSC format holds one DMPlex per
        # file). Collective write. See _hierarchy_sidecar_name and the .h5 reload.
        if len(self.dm_hierarchy) > 1:
            coarse_dm = self.dm_hierarchy[0]
            sidecar = _hierarchy_sidecar_name(filename)
            cviewer = PETSc.ViewerHDF5().create(sidecar, "w", comm=PETSc.COMM_WORLD)
            cviewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
            saved_name = coarse_dm.getName()
            coarse_dm.setName("uw_mesh")  # _from_plexh5 loads the DM named "uw_mesh"
            try:
                cviewer(coarse_dm)
            finally:
                coarse_dm.setName(saved_name)
                cviewer.popFormat()
                cviewer.destroy()

    def vtk(self, filename: str):
        """
        Save mesh to the specified file
        """

        viewer = PETSc.Viewer().createVTK(filename, "w", comm=PETSc.COMM_WORLD)
        viewer(self.dm)
        viewer.destroy()

    def generate_xdmf(self, filename: str):
        """
        This method generates an xdmf schema for the specified file.

        The filename of the generated file will be the same as the hdf5 file
        but with the `xmf` extension.

        Parameters
        ----------
        filename :
            File name of the checkpointed hdf5 file for which the
            xdmf schema will be written.
        """
        from underworld3.utilities import generateXdmf

        if uw.mpi.rank == 0:
            generateXdmf(filename)

        return

    # ToDo: rename this so it does not clash with the vars built in
    @property
    def vars(self):
        """
        A list of variables recorded on the mesh.
        """
        return self._vars

        # ToDo: rename this so it does not clash with the vars built in

    @property
    def block_vars(self):
        """
        A list of variables recorded on the mesh.
        """
        return self._block_vars

    def _get_coords_for_var(self, var):
        """
        This function returns the vertex array for the
        provided variable. If the array does not already exist,
        it is first created and then returned.
        """
        key = (self.isSimplex, var.degree, var.continuous)

        # if array already created, return.
        if key in self._coord_array:
            return self._coord_array[key]
        else:
            self._coord_array[key] = self._get_coords_for_basis(var.degree, var.continuous)
            return self._coord_array[key]

    def _get_coords_for_basis(self, degree, continuous):
        """
        This function returns the vertex array for the
        provided variable. If the array does not already exist,
        it is first created and then returned.
        """

        dmold = self.dm.getCoordinateDM()
        dmold.createDS()
        dmnew = dmold.clone()

        options = PETSc.Options()
        options["coordinterp_petscspace_degree"] = degree
        options["coordinterp_petscdualspace_lagrange_continuity"] = continuous
        options["coordinterp_petscdualspace_lagrange_node_endpoints"] = False

        dmfe = PETSc.FE().createDefault(
            self.dim,
            self.cdim,
            self.isSimplex,
            self.qdegree,
            "coordinterp_",
            PETSc.COMM_SELF,
        )

        dmnew.setField(0, dmfe)
        dmnew.createDS()

        matInterp, vecScale = dmold.createInterpolation(dmnew)
        coordsOld = self.dm.getCoordinates()
        coordsNewL = dmnew.getLocalVec()
        coordsNewG = dmnew.getGlobalVec()
        matInterp.mult(coordsOld, coordsNewG)
        dmnew.globalToLocal(coordsNewG, coordsNewL)

        arr = coordsNewL.array
        arrcopy = arr.reshape(-1, self.cdim).copy()

        dmnew.restoreGlobalVec(coordsNewG)
        dmnew.restoreLocalVec(coordsNewL)
        # Clean up the PETSc interpolation objects built above. Without this
        # they accumulate until Python GC runs — noticeable in long adapt
        # loops that re-fill the coord cache per variable.
        matInterp.destroy()
        if vecScale is not None:
            vecScale.destroy()
        dmnew.destroy()
        dmfe.destroy()

        return arrcopy

    def _build_kd_tree_index_DS(self):

        if hasattr(self, "_index") and self._index is not None:
            return

        # Build this from the PETScDS rather than the SWARM

        centroids = self._get_coords_for_basis(0, False)
        index_coords = self._get_coords_for_basis(2, False)

        points_per_cell = index_coords.shape[0] // centroids.shape[0]

        cell_id = numpy.empty(index_coords.shape[0])
        for i in range(cell_id.shape[0]):
            cell_id[i] = i // points_per_cell

        self._indexCoords = index_coords
        self._index = uw.kdtree.KDTree(self._indexCoords)
        # self._index.build_index()
        self._indexMap = numpy.array(cell_id, dtype=numpy.int64)

        return

    def _coord_rows_for_points(self, nav_dm, points):
        """Row indices into the navigation coordinate array (``_nav_coords``)
        for the given vertex plex points, via the coordinate PetscSection
        offsets.

        Correct in parallel: on the 1-cell-overlap navigation clone (manifold
        meshes) the local coordinate array is laid out by the coordinate
        section, so mapping a vertex with the affine shift ``plex_point -
        pStart`` runs negative or past the end across ghost/halo vertices
        (issue #360). The section offset is the authoritative vertex->row map.
        In serial the section offset equals ``(plex_point - pStart) * cdim``,
        so the rows returned here are bit-identical to the previous affine
        indexing (verified) and serial navigation is unchanged.
        """
        coord_section = nav_dm.getCoordinateSection()
        cdim = self.cdim
        return numpy.fromiter(
            (coord_section.getOffset(p) // cdim for p in points),
            dtype=numpy.int64,
            count=len(points),
        )

    def _navigation_coords_from_dm(self, nav_dm):
        """Section-consistent LOCAL coordinate rows for a navigation DM.

        On the 1-cell-overlap navigation clone (manifold meshes, parallel)
        ``distributeOverlap`` expands the coordinate section to cover the
        ghost vertices but does NOT scatter their coordinates into the local
        vector, so a plain ``getCoordinatesLocal()`` is short by the ghost
        rows and any vertex->row map overruns it (issue #360). Rebuild the
        full local coordinate vector through the coordinate DM's
        global->local scatter so every ghost vertex referenced by a local
        cell closure carries coordinates. In serial (no overlap) this equals
        ``getCoordinatesLocal()``.
        """
        from petsc4py import PETSc

        coord_dm = nav_dm.getCoordinateDM()
        local_coords = coord_dm.createLocalVec()
        coord_dm.globalToLocal(
            nav_dm.getCoordinates(), local_coords, addv=PETSc.InsertMode.INSERT_VALUES
        )
        return numpy.array(local_coords.array).reshape(-1, self.cdim)

    def _build_kd_tree_index(self):

        if hasattr(self, "_index") and self._index is not None:
            return

        dim = self.dim
        # Navigation indices build from the nav DM (a 1-cell-overlap
        # clone on manifold meshes; identical to self.dm on volume
        # meshes). Cell indices in the resulting _indexMap and
        # _centroid_index correspond to nav-DM local cell ordering.
        nav_dm = self._nav_dm if self._nav_dm is not None else self.dm
        nav_coords = self._nav_coords

        cStart, cEnd = nav_dm.getHeightStratum(0)
        fStart, fEnd = nav_dm.getHeightStratum(1)
        pStart, pEnd = nav_dm.getDepthStratum(0)
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        control_points_list = []
        control_points_cell_list = []
        centroids_list = []

        for cell, cell_id in enumerate(range(cStart, cEnd)):

            cell_faces = nav_dm.getCone(cell_id)
            points = nav_dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
            # Use raw internal array for KD-tree construction (avoid unit-aware wrapping)
            cell_point_coords = nav_coords[self._coord_rows_for_points(nav_dm, points)]
            cell_centroid = cell_point_coords.mean(axis=0)
            centroids_list.append(cell_centroid)

            # for face in range(cell_num_faces):

            #     points = self.dm.getTransitiveClosure(cell_faces[face])[0][
            #         -face_num_points:
            #     ]
            #     point_coords = self.data[points - pStart]

            #     face_centroid = point_coords.mean(axis=0)
            #     cell_centroid = cell_point_coords.mean(axis=0)

            #     # 2D case
            #     if self.dim == 2:
            #         vector = point_coords[1] - point_coords[0]
            #         normal = numpy.array((-vector[1], vector[0]))

            #     # 3D simplex case (probably also OK for hexes)
            #     else:
            #         normal = numpy.cross(
            #             (point_coords[1] - point_coords[0]),
            #             (point_coords[2] - point_coords[0]),
            #         )

            #     inward_outward = numpy.sign(normal.dot(face_centroid - cell_centroid))
            #     normal *= inward_outward / numpy.sqrt(normal.dot(normal))

            #     inside_control_point = -1e-3 * normal + face_centroid

            #     control_points_list.append(inside_control_point)
            #     control_points_cell_list.append(cell_id)
            #     control_points_list.append(cell_centroid)
            #     control_points_cell_list.append(cell_id)

            # Add points near the cell vertices

            for i in range(cell_point_coords.shape[0]):
                control_points_list.append(0.99 * cell_point_coords[i] + 0.01 * cell_centroid)
                control_points_cell_list.append(cell_id)

            # Add centroid
            control_points_list.append(cell_centroid)
            control_points_cell_list.append(cell_id)

        self._indexCoords = numpy.array(control_points_list)
        self._index = uw.kdtree.KDTree(self._indexCoords)
        # self._index.build_index()
        self._indexMap = numpy.array(control_points_cell_list, dtype=numpy.int64)

        # Cell-centroid kdtree, built from the nav-DM cells in the
        # same enumeration order as _indexMap, so the indices it
        # returns can be used directly as nav-DM cell indices.
        # We keep _nav_centroids separate from _centroids (which is
        # the main-DM cell centroids set in __init__) so the FE-side
        # ``_centroids`` semantics are unchanged on manifold meshes.
        self._nav_centroids = numpy.array(centroids_list)
        self._centroid_index = uw.kdtree.KDTree(self._nav_centroids)

        return

    def _build_kd_tree_index_PIC(self):

        if hasattr(self, "_index") and self._index is not None:
            return

        ## Bootstrapping - the kd-tree is needed to build the index but
        ## the index is also used in the kd-tree.

        from underworld3.swarm import Swarm, SwarmPICLayout

        # Create a temp swarm which we'll use to populate particles
        # at gauss points. These will then be used as basis for
        # kd-tree indexing back to owning cells.

        from petsc4py import PETSc

        tempSwarm = PETSc.DMSwarm().create()
        tempSwarm.setDimension(self.dim)
        tempSwarm.setCellDM(self.dm)
        tempSwarm.setType(PETSc.DMSwarm.Type.PIC)
        tempSwarm.finalizeFieldRegister()

        # 3^dim or 4^dim pop is used. This number may need to be considered
        # more carefully, or possibly should be coded to be set dynamically.

        tempSwarm.insertPointUsingCellDM(PETSc.DMSwarm.PICLayoutType.LAYOUT_GAUSS, 3)

        # We can't use our own populate function since this needs THIS kd_tree to exist
        # We will need to use a standard layout instead

        ## ?? is this required given no migration ??
        # tempSwarm.migrate(remove_sent_points=True)

        PIC_coords = tempSwarm.getField("DMSwarmPIC_coor").reshape(-1, self.dim)
        PIC_cellid = tempSwarm.getField("DMSwarm_cellid")

        self._indexCoords = PIC_coords.copy()
        self._index = uw.kdtree.KDTree(self._indexCoords)
        self._indexMap = numpy.array(PIC_cellid, dtype=numpy.int64)
        # self._index.build_index()

        # We don't need an indexMap for this one because there is only one point per cell
        # and the returned kdtree value IS the index.
        # Note: self._centroids is not yet defined:

        self._centroid_index = uw.kdtree.KDTree(self._get_coords_for_basis(0, False))
        # self._centroid_index.build_index()

        tempSwarm.restoreField("DMSwarmPIC_coor")
        tempSwarm.restoreField("DMSwarm_cellid")  #

        tempSwarm.destroy()

        return

    # Note - need to add this to the mesh rebuilding triggers
    def _facet_outward_unit_normal(
        self, facet_point_coords, facet_centroid, cell_centroid,
        cell_point_coords=None,
    ):
        """Outward unit normal of a cell facet, for each dimension/embedding case.

        Used by the in-cell / in-domain control-point builders
        (:meth:`_mark_faces_inside_and_out`,
        :meth:`_mark_local_boundary_faces_inside_and_out`): the returned
        normal orients the mirrored inner/outer control-point pair each
        facet gets. The four cases are:

        - ``dim == 1``: 1-manifold (annulus / shell boundary loop as a
          surface submesh) — a cell is an edge, its facets are end
          vertices. The outward direction is along the edge, away from
          the cell centroid (the sign fix below orients it).
        - ``dim == 2, cdim == 2``: 2-D volume mesh — perpendicular to the
          edge in the plane of the mesh.
        - ``dim == 2, cdim == 3``: 2-manifold in 3-space — perpendicular
          to the edge, lying in the cell's tangent plane (the natural
          generalisation of the 2-D rule, where the implicit z-hat is
          replaced by the explicit cell normal). Needs
          ``cell_point_coords`` (three cell vertices) to build the cell
          normal.
        - otherwise: 3-D simplex / hex face — face normal from two
          in-face edges.

        Parameters
        ----------
        facet_point_coords : ndarray
            Coordinates of the facet's vertices.
        facet_centroid, cell_centroid : ndarray
            Centroids used to orient the normal outward.
        cell_point_coords : ndarray, optional
            Coordinates of the owning cell's vertices; required only for
            the 2-manifold-in-3-space case.

        Returns
        -------
        ndarray
            Unit normal pointing from the cell centroid out through the
            facet.
        """
        if self.dim == 1:
            normal = facet_centroid - cell_centroid
        elif self.dim == 2 and self.cdim == 2:
            vector = facet_point_coords[1] - facet_point_coords[0]
            normal = numpy.array((-vector[1], vector[0]))
        elif self.dim == 2 and self.cdim == 3:
            cell_normal = numpy.cross(
                cell_point_coords[1] - cell_point_coords[0],
                cell_point_coords[2] - cell_point_coords[0],
            )
            edge_vector = facet_point_coords[1] - facet_point_coords[0]
            normal = numpy.cross(cell_normal, edge_vector)
        else:
            normal = numpy.cross(
                (facet_point_coords[1] - facet_point_coords[0]),
                (facet_point_coords[2] - facet_point_coords[0]),
            )

        # Orient outward: flip if the normal points from the facet centroid
        # back toward the cell centroid; normalise to unit length.
        inward_outward = numpy.sign(normal.dot(facet_centroid - cell_centroid))
        normal *= inward_outward / numpy.sqrt(normal.dot(normal))
        return normal

    def _mark_faces_inside_and_out(self):
        """
        Create a collection of control point pairs that are slightly inside
        and slightly outside each mesh face (mirrors to each other). This
        allows a fast lookup of whether we on the inside or outside of the plane
        defined by a face (i.e. same side or other side as the cell centroid). If we are inside
        for all faces in a convex polyhedron, then we are inside the cell.

        Internal Coordinate System Access Pattern
        ------------------------------------------
        This method uses `self._coords` (raw PETSc array) instead of `self.data`
        or `self.X.coords` (unit-wrapped properties) for performance and correctness:

        1. **Guard at boundaries**: External interfaces use unit-aware properties
        2. **Raw access internally**: Internal geometric calculations use `self._coords`
        3. **Performance**: Avoids UnitAwareArray overhead in tight loops
        4. **Correctness**: Prevents unit conversion issues in geometric operations

        This is the recommended pattern for internal mesh operations that manipulate
        coordinates directly.
        """

        if (
            hasattr(self, "faces_inner_control_points")
            and self.faces_inner_control_points is not None
            and hasattr(self, "faces_outer_control_points")
            and self.faces_outer_control_points is not None
        ):
            return

        dim = self.dim
        # Build face control points from the nav DM (includes ghost
        # cells on manifold meshes). Volume meshes have _nav_dm is
        # None and we use self.dm directly.
        nav_dm = self._nav_dm if self._nav_dm is not None else self.dm
        nav_coords = self._nav_coords

        cStart, cEnd = nav_dm.getHeightStratum(0)
        fStart, fEnd = nav_dm.getHeightStratum(1)
        pStart, pEnd = nav_dm.getDepthStratum(0)
        num_local_cells = cEnd - cStart
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        # All elements in our mesh are a single type

        mesh_cell_outer_control_points = numpy.ndarray(
            shape=(cell_num_faces, num_local_cells, self.cdim)
        )
        mesh_cell_inner_control_points = numpy.ndarray(
            shape=(cell_num_faces, num_local_cells, self.cdim)
        )

        for cell, cell_id in enumerate(range(cStart, cEnd)):
            cell_faces = nav_dm.getCone(cell_id)
            points = nav_dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
            # Use raw internal array for internal mesh operations (avoid unit-aware wrapping)
            cell_point_coords = nav_coords[self._coord_rows_for_points(nav_dm, points)]

            for face in range(cell_num_faces):

                points = nav_dm.getTransitiveClosure(cell_faces[face])[0][-face_num_points:]
                # Use raw internal array for internal mesh operations (avoid unit-aware wrapping)
                point_coords = nav_coords[self._coord_rows_for_points(nav_dm, points)]

                face_centroid = point_coords.mean(axis=0)
                cell_centroid = cell_point_coords.mean(axis=0)

                # Face normal from point coordinates (already plain numpy
                # arrays); dimension-case dispatch lives in the helper.
                normal = self._facet_outward_unit_normal(
                    point_coords, face_centroid, cell_centroid,
                    cell_point_coords=cell_point_coords,
                )

                # Compute control points (all arrays are already plain numpy, no units)
                outside_control_point = 1e-3 * normal + face_centroid
                inside_control_point = -1e-3 * normal + face_centroid

                mesh_cell_outer_control_points[face, cell, :] = outside_control_point
                mesh_cell_inner_control_points[face, cell, :] = inside_control_point

        self.faces_inner_control_points = mesh_cell_inner_control_points
        self.faces_outer_control_points = mesh_cell_outer_control_points

        return

    def _get_owned_cells_mask(self):
        """Return a boolean array of length n_local_cells (NAV-DM
        cells) where True means the cell is owned by this rank, False
        means it's a ghost cell brought in by
        ``DMPlexDistributeOverlap``.

        On a non-overlapped DM (the default for volume meshes), every
        cell is owned and the mask is all True — the downstream filter
        is a no-op. On the nav DM of a manifold mesh, ghost cells
        appear as leaves of the point SF in the cell range
        ``[cStart, cEnd)``.

        Cached on the mesh; rebuilt only when ``_mesh_version`` changes.
        """
        version = self._mesh_version
        cache = getattr(self, "_owned_cells_mask_cache", None)
        if cache is not None and cache.get("version") == version:
            return cache["mask"]

        nav_dm = self._nav_dm if self._nav_dm is not None else self.dm
        cStart, cEnd = nav_dm.getHeightStratum(0)
        n_local = cEnd - cStart
        mask = numpy.ones(n_local, dtype=bool)

        sf = nav_dm.getPointSF()
        if sf is not None:
            try:
                _, leaves, _ = sf.getGraph()
            except Exception:
                leaves = None
            if leaves is not None and len(leaves) > 0:
                leaves = numpy.asarray(leaves)
                # Leaves are global point IDs; cells live in [cStart, cEnd).
                cell_leaves = leaves[
                    (leaves >= cStart) & (leaves < cEnd)
                ] - cStart
                if cell_leaves.size > 0:
                    mask[cell_leaves] = False

        self._owned_cells_mask_cache = {"version": version, "mask": mask}
        return mask

    def _test_if_points_in_cells_internal(self, points, cells,
                                          on_boundary=True, tol=0.0):
        """
        Determine if the given points lie in the suggested cells.
        Uses a mesh skeletonization array to determine whether the point is
        with the convex polygon / polyhedron defined by a cell.

        Exact if applied to a linear mesh, approximate otherwise.

        On an overlapped DM (manifold meshes), a query point may land in
        a *ghost* cell — a cell owned by another rank and present locally
        only as part of the partition halo. Ghost cells are explicitly
        rejected so a single rank claims each point cleanly; the migrate
        loop's iterative fallback then routes the rejected point to the
        actual owning rank (where the same cell is genuinely owned).

        Parameters
        ----------
        points : numpy.ndarray
            Coordinate array, assumed already in model units (this internal
            helper does not perform unit conversion — use the public
            `test_if_points_in_cells` for unit-aware input).
        cells : numpy.ndarray
            1-D cell indices to test, one per point.
        on_boundary : bool, default True
            If True (the default), a point exactly on a cell face counts as
            inside that cell — the natural semantics for FE evaluation,
            where the basis at a shared face/vertex is consistent across
            the adjacent cells. A query point lying on a face shared by N
            cells passes the test for any of those N cells.

            If False, a point exactly on a face is reported as NOT inside —
            strict-inside semantics. Use this when uniqueness matters (a
            strict-ownership scheme where a shared-face point being claimed
            by all adjacent cells would be a bug).

            The implementation compares the squared distance from the query
            to a mirrored inner/outer control-point pair placed ±1e-3 along
            the face normal; a point exactly on the face has zero distance
            difference. With on_boundary=True the test accepts diff >= -1e-12
            (well below the 1e-3 control-point offset, well above 64-bit
            float roundoff); with on_boundary=False the test requires diff > 0.
        tol : float, default 0.0
            Face-RELATIVE tolerance — overrides ``on_boundary``'s absolute
            -1e-12 floor when nonzero. The test becomes
            ``diff > -tol * |O - I|^2`` (i.e. relative to the
            control-point separation squared). With ``tol == 0`` (the
            default) ``on_boundary`` controls the test as documented above.
            With ``tol > 0`` the test is relaxed to admit points within
            roughly ``tol`` of the face (relative to the face-normal
            separation), while still rejecting points that lie inside a
            *different* cell (whose diff is strongly negative). The
            parallel evaluation locator
            (``Mesh._robust_owning_cells``) uses ``tol=1e-2`` to admit
            on-face / partition-seam node queries that ``on_boundary=True``'s
            absolute 1e-12 floor is too tight to accept — for query
            coordinates slightly off the face (e.g. RBF-shifted node
            points), the geometric scale of the test needs to match the
            *mesh* spacing, not float roundoff. See memory
            project_pr207_loose_boundary_clash.
        """
        # Internal version - points assumed to already be in model units
        self._mark_faces_inside_and_out()

        cells = cells.reshape(-1)
        assert points.shape[0] == cells.shape[0]

        cStart, cEnd = self.dm.getHeightStratum(0)
        num_cell_faces = self.dm.getConeSize(cStart)

        insiders = numpy.ndarray(shape=(cells.shape[0], num_cell_faces), dtype=bool)

        # One loop computes the per-face inside/outside discriminator
        # (squared distance to the outer control point minus squared
        # distance to the inner one — positive means the query is on the
        # cell's side of the face); the acceptance test then depends on
        # the tolerance mode. A non-zero tol takes precedence over
        # on_boundary: it expresses a geometric tolerance relative to the
        # face-normal separation² (parallel evaluation locator), far wider
        # than on_boundary=True's absolute -1e-12 on-face floor, while
        # on_boundary=False is the strict-inside test (diff > 0).
        for f in range(num_cell_faces):
            control_points_o = self.faces_outer_control_points[f, cells]
            control_points_i = self.faces_inner_control_points[f, cells]
            diff = (
                ((control_points_o - points) ** 2).sum(axis=1)
                - ((control_points_i - points) ** 2).sum(axis=1)
            )
            if tol > 0.0:
                sep2 = ((control_points_o - control_points_i) ** 2).sum(axis=1)
                insiders[:, f] = diff > -tol * sep2
            elif on_boundary:
                insiders[:, f] = diff >= -1e-12
            else:
                insiders[:, f] = diff > 0

        result = numpy.all(insiders, axis=1)

        # Reject ghost-cell claims so ownership remains unique. No-op on
        # non-overlapped DMs where every cell is owned.
        owned_mask = self._get_owned_cells_mask()
        valid_cell = (cells >= 0) & (cells < owned_mask.shape[0])
        result[valid_cell] = result[valid_cell] & owned_mask[cells[valid_cell]]

        return result

    def _mark_local_boundary_faces_inside_and_out(self):
        """
        Create a collection of control point pairs that are slightly inside
        and slightly outside each boundary-defining face (mirrors to each other). This
        allows a fast lookup of whether we on the inside or outside of the domain.
        We cannot ensure convexity, so this is approximate when close to the boundary
        """

        if (
            hasattr(self, "boundary_face_control_points_kdtree")
            and self.boundary_face_control_points_kdtree is not None
            and hasattr(self, "boundary_face_control_points_sign")
            and self.boundary_face_control_points_sign is not None
        ):
            return

        # Build boundary control points from the nav DM (sees the
        # ghost cells on manifold meshes). On volume meshes nav_dm is
        # self.dm.
        nav_dm = self._nav_dm if self._nav_dm is not None else self.dm
        nav_coords = self._nav_coords

        cStart, cEnd = nav_dm.getHeightStratum(0)
        fStart, fEnd = nav_dm.getHeightStratum(1)
        pStart, pEnd = nav_dm.getDepthStratum(0)
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        # On an overlapped DM (manifold meshes with the partition halo),
        # the outer edge of the halo masquerades as a boundary: faces
        # there have ``getJoin(face).shape[0] == 1`` because only one
        # of the two adjacent cells is in this rank's local view —
        # and that one is a ghost. Filter to faces whose single
        # bounding cell is OWNED locally; otherwise we'd build
        # control points along the partition seam and reject legitimate
        # interior points. No-op on non-overlapped DMs.
        owned_mask = self._get_owned_cells_mask()
        boundary_faces = []
        for face in range(fStart, fEnd):
            join = nav_dm.getJoin(face)
            if join.shape[0] != 1:
                continue
            cell = int(join[0])
            if cell < cStart or cell >= cEnd:
                continue
            if not owned_mask[cell - cStart]:
                continue
            boundary_faces.append(face)

        boundary_faces = numpy.array(boundary_faces)

        # Closed manifolds (e.g. SphericalManifold) have no boundary
        # faces — the kdtree path is honestly empty. Let the caller's
        # closest-local-cell short-circuit handle on-surface queries.
        if len(boundary_faces) == 0:
            self.boundary_face_control_points_kdtree = None
            self.boundary_face_control_points_sign = None
            self._domain_radius_squared = float("inf")
            return

        control_points_list = []
        control_point_sign_list = []

        # Pick the right centroid source: _nav_centroids if it's been
        # built (set in _build_kd_tree_index from nav-DM cells), else
        # the main-DM _centroids. On volume meshes these are equal.
        nav_centroids = getattr(self, "_nav_centroids", None)
        if nav_centroids is None:
            nav_centroids = self._centroids

        for face in boundary_faces:
            cell = nav_dm.getJoin(face)[0]
            points = nav_dm.getTransitiveClosure(face)[0][-face_num_points:]
            point_coords = nav_coords[self._coord_rows_for_points(nav_dm, points)]  # raw array for internal calculations
            face_centroid = point_coords.mean(axis=0)
            cell_centroid = nav_centroids[cell - cStart]

            # Dimension-case dispatch lives in _facet_outward_unit_normal;
            # only the tangent-plane case (bounded 2-manifold in 3-space,
            # e.g. a partial-surface patch) needs the owning cell's vertex
            # coordinates, fetched lazily to keep the common cases cheap.
            cell_point_coords = None
            if self.dim == 2 and self.cdim == 3:
                cell_points = nav_dm.getTransitiveClosure(cell)[0][-cell_num_points:]
                cell_point_coords = nav_coords[self._coord_rows_for_points(nav_dm, cell_points)]

            normal = self._facet_outward_unit_normal(
                point_coords, face_centroid, cell_centroid,
                cell_point_coords=cell_point_coords,
            )

            # Control points near centroid

            outside_control_point = 1e-8 * normal + face_centroid
            control_points_list.append(outside_control_point)
            control_point_sign_list.append(-1)

            inside_control_point = -1e-8 * normal + face_centroid
            control_points_list.append(inside_control_point)
            control_point_sign_list.append(1)

            # Control points closer to face nodes

            for pt in range(0, face_num_points):

                outside_control_point = 1e-8 * normal + 0.8 * point_coords[pt] + 0.2 * face_centroid
                control_points_list.append(outside_control_point)
                control_point_sign_list.append(-1)

                inside_control_point = -1e-8 * normal + 0.8 * point_coords[pt] + 0.2 * face_centroid
                control_points_list.append(inside_control_point)
                control_point_sign_list.append(1)

        control_points_array = numpy.array(control_points_list)
        control_point_kdtree = uw.kdtree.KDTree(control_points_array)
        control_point_sign = numpy.array(control_point_sign_list)

        self.boundary_face_control_points_kdtree = control_point_kdtree
        self.boundary_face_control_points_sign = control_point_sign

        # Domain bounding radius (squared): distance from centroid to farthest
        # control point. Points beyond this distance from their nearest control
        # point cannot be inside the domain.
        domain_centroid = control_points_array.mean(axis=0)
        radii_sq = numpy.sum((control_points_array - domain_centroid) ** 2, axis=1)
        self._domain_radius_squared = float(radii_sq.max())

        return

    def points_in_domain(self, points, strict_validation=True):
        """
        Determine if the given points lie in this domain.
        Uses a mesh-boundary skeletonization array to determine whether the point is
        inside the boundary or outside. If close to the boundary, it checks if points
        are in a cell.

        Parameters
        ----------
        points : array-like
            Coordinate array in any physical unit system (will be auto-converted).
            Plain numbers are assumed to be in model coordinates.
        strict_validation : bool
            Whether to perform strict validation near boundaries

        """
        # Convert points to model coordinates using the unified conversion function
        # This handles all coordinate formats: plain numbers, unit-aware coordinates, lists, tuples, arrays
        import underworld3 as uw
        from underworld3.function.unit_conversion import _convert_coords_to_si

        # _convert_coords_to_si now converts to model coordinates (despite the name)
        # and handles all the complexity of extracting values from unit-aware coordinates
        model_points = _convert_coords_to_si(points)

        self._mark_local_boundary_faces_inside_and_out()

        max_radius = self.get_max_radius()

        if model_points.shape[0] == 0:
            return numpy.array([], dtype=bool)

        # Cd-1 surface mesh: no boundary-face control points exist
        # (see _mark_local_boundary_faces_inside_and_out). Per the
        # surface-mesh contract, query points are assumed to lie on
        # the manifold; the closest-local-cell test is the right
        # filter, not an inside/outside split.
        if self.boundary_face_control_points_kdtree is None:
            return self._get_closest_local_cells_internal(model_points) != -1

        dist2, closest_control_points_ext = self.boundary_face_control_points_kdtree.query(
            model_points, k=1, sqr_dists=True
        )
        dist2 = numpy.asarray(dist2).ravel()  # kd-tree returns (n,1) for k=1
        in_or_not = self.boundary_face_control_points_sign[closest_control_points_ext] > 0

        # Points very far from the nearest boundary face are definitely exterior.
        # The sign heuristic only works for points within the domain's neighbourhood;
        # beyond that, "nearest control point" is arbitrary.
        far_from_domain = dist2 > self._domain_radius_squared
        in_or_not[far_from_domain] = False

        # Points close to the boundary need the expensive cell-location check.
        #
        # The plain cell-wall test (_get_closest_local_cells_internal) returns
        # -1 for points sitting exactly on a cell face/edge OR on the domain
        # boundary — even though an on-boundary point is in the (closed) domain.
        # That rejection is what strands on-face / partition-seam / domain-
        # boundary NODE points in swarm migration (they are never "claimed", so
        # the domain-centroid routing leaves them on a non-containing rank) and
        # what routes them to rank-local RBF in evaluation. On parallel simplex
        # / manifold meshes (mesh._eval_use_robust_location()) defer instead to
        # the bulletproof barycentric locator, which returns a valid adjacent
        # cell (>= 0) for any point genuinely in/on the mesh and -1 only for
        # true exterior. Serial / non-simplex keep the cell-wall test
        # (bit-identical to the validated baseline).
        near_boundary = numpy.where(dist2 < 2 * max_radius**2)[0]
        near_boundary_points = model_points[near_boundary]

        if self._eval_use_robust_location():
            in_or_not[near_boundary] = self._robust_owning_cells(near_boundary_points) >= 0
        else:
            in_or_not[near_boundary] = (
                self._get_closest_local_cells_internal(near_boundary_points) != -1
            )

        if strict_validation:
            chosen_ones = numpy.where(in_or_not == True)[0]
            chosen_points = model_points[chosen_ones]
            if self._eval_use_robust_location():
                in_or_not[chosen_ones] = self._robust_owning_cells(chosen_points) >= 0
            else:
                in_or_not[chosen_ones] = self._get_closest_local_cells_internal(chosen_points) != -1

        return in_or_not

    @timing.routine_timer_decorator
    def get_closest_cells(self, coords: numpy.ndarray) -> numpy.ndarray:
        """
        This method uses a kd-tree algorithm to find the closest
        cells to the provided coords. For a regular mesh, this should
        be exactly the owning cell, but if the mesh is deformed, this
        is not guaranteed. Note, the nearest point may not be all
        that close by - use get_closest_local_cells to filter out points
        that are (probably) not within any local cell.

        Parameters:
        -----------
        coords:
            An array of the coordinates for which we wish to determine the
            closest cells. This should be a 2-dimensional array of
            shape (n_coords,dim) in any physical unit system (will be auto-converted).
            Plain numbers are assumed to be in model coordinates.

        Returns:
        --------
        closest_cells:
            An array of indices representing the cells closest to the provided
            coordinates. This will be a 1-dimensional array of
            shape (n_coords).
        """
        import numpy as np

        # Convert coords to model coordinates
        # Simply extract raw values - np.asarray handles unit-aware objects correctly
        model_coords = np.asarray(coords)

        self._build_kd_tree_index()

        if len(model_coords) > 0:
            dist, closest_points = self._index.query(model_coords, k=1, sqr_dists=False)
            if np.any(closest_points > self._index.n):
                raise RuntimeError(
                    "An error was encountered attempting to find the closest cells to the provided coordinates."
                )
            return self._indexMap[closest_points]
        else:
            ### returns an empty 1D array if no coords are provided
            # CRITICAL: Must return 1D array, not 2D, for Cython buffer compatibility
            return numpy.array([], dtype=numpy.int64)

    def _get_closest_local_cells_internal(
        self,
        coords: numpy.ndarray,
        on_boundary: bool = True,
        tol: float = 0.0,
    ) -> numpy.ndarray:
        """
        This method uses a kd-tree algorithm to find the closest
        cells to the provided coords. For a regular mesh, this should
        be exactly the owning cell, but if the mesh is deformed, this
        is not guaranteed. Also compares the distance from the cell to the
        point - if this is larger than the "cell size" then returns -1

        ``on_boundary`` and ``tol`` are forwarded to the in-cell
        containment test (see ``_test_if_points_in_cells_internal``).
        Default ``(on_boundary=True, tol=0.0)`` admits on-face queries
        at PR #207's absolute -1e-12 floor — the FE-evaluation-natural
        semantics. Pass ``on_boundary=False`` for strict-inside (a
        point exactly on a face returns -1). ``tol > 0`` admits on-face
        points at a face-relative tolerance, taking precedence over
        ``on_boundary``.

        Parameters:
        -----------
        coords:
            An array of the coordinates for which we wish to determine the
            closest cells. This should be a 2-dimensional array of
            shape (n_coords,dim), assumed already in model units (this
            internal helper does not perform unit conversion — use the
            public `get_closest_local_cells` for unit-aware input).
        on_boundary : bool, default True
            Forwarded to `_test_if_points_in_cells_internal`. If True (the
            default), queries exactly on a cell face count as inside that
            cell — the natural semantics for FE-evaluation hints (every mesh
            vertex sits on the faces of every cell containing it). If False,
            strict-inside semantics; boundary queries come back as -1.
        tol : float, default 0.0
            Face-relative tolerance, forwarded to
            `_test_if_points_in_cells_internal`. When > 0, the in-cell
            test becomes ``diff > -tol * |O-I|²`` and takes precedence
            over ``on_boundary``. Used by the parallel evaluation
            locator (`Mesh._robust_owning_cells`) to admit on-face /
            partition-seam node queries at a mesh-spacing-relative
            tolerance — wider than ``on_boundary=True``'s absolute
            -1e-12 floor.

        Returns:
        --------
        closest_cells:
            An array of indices representing the cells closest to the provided
            coordinates. This will be a 1-dimensional array of
            shape (n_coords).


        """
        import numpy as np

        # Internal version - coords assumed to already be in model units
        # Create index if required
        self._build_kd_tree_index()

        if len(coords) > 0:
            dist, closest_points = self._index.query(coords, k=1, sqr_dists=False)
            if np.any(closest_points > self._index.n):
                raise RuntimeError(
                    "An error was encountered attempting to find the closest cells to the provided coordinates."
                )
        else:
            return np.zeros((0,))

        cells = self._indexMap[closest_points]
        cStart, cEnd = self.dm.getHeightStratum(0)

        # We need to filter points that lie outside the mesh but
        # still are allocated a nearby element by this distance-only check.
        # On a 2-manifold in 3-space the in-cell test is the
        # in-tangent-plane half-space rule (Site A in
        # _mark_faces_inside_and_out generalises the perpendicular
        # construction to ``cell_normal × edge``), so this works
        # uniformly on volume meshes and cd-1 manifolds.

        inside = self._test_if_points_in_cells_internal(
            coords, cells, on_boundary=on_boundary, tol=tol)
        cells[~inside] = -1
        lost_points = np.where(inside == False)[0]

        # Part 2 - try to find the lost points by walking nearby cells

        # Size by the nav-DM cell count, which is what _centroid_index
        # was built from (includes ghost cells on manifold meshes).
        nav_centroids = getattr(self, "_nav_centroids", None)
        if nav_centroids is None:
            nav_centroids = self._centroids
        num_local_cells = nav_centroids.shape[0]
        num_testable_neighbours = min(num_local_cells, 50)

        dist2, closest_centroids = self._centroid_index.query(
            coords[lost_points], k=num_testable_neighbours, sqr_dists=False
        )

        # This number is close to the point-point coordination value in 3D unstructured
        # grids (by inspection)

        for i in range(0, num_testable_neighbours):

            inside = self._test_if_points_in_cells_internal(
                coords[lost_points], closest_centroids[:, i],
                on_boundary=on_boundary, tol=tol,
            )
            cells[lost_points[inside]] = closest_centroids[inside, i]

            if np.count_nonzero(cells == -1) == 0:
                break

        return cells

    def test_if_points_in_cells(self, points, cells, on_boundary=True, tol=0.0):
        """
        Determine if the given points lie in the suggested cells.
        Uses a mesh skeletonization array to determine whether the point is
        with the convex polygon / polyhedron defined by a cell.

        Exact if applied to a linear mesh, approximate otherwise.

        Parameters
        ----------
        points : array-like
            Coordinate array in any physical unit system (will be auto-converted)
        cells : array-like
            Cell indices to test
        on_boundary : bool, default True
            If True (the default), points exactly on a cell face count as
            inside the cell (natural for FE evaluation, where the basis at
            a shared face/vertex is consistent across adjacent cells). If
            False, points on the closure of a cell are reported as NOT in
            it (strict-inside semantics — useful when uniqueness matters).
        tol : float, default 0.0
            Face-relative tolerance forwarded to
            `_test_if_points_in_cells_internal`. When ``> 0`` takes
            precedence over ``on_boundary``: the test admits points
            within ``tol`` of the face relative to the control-point
            separation² — used by the parallel evaluation locator
            for on-face / near-face queries at the mesh-spacing scale.

        Returns
        -------
        numpy.ndarray
            Boolean array indicating if points are in cells
        """
        # Convert points to model units using the elegant protocol
        import underworld3 as uw

        model = uw.get_default_model()
        model_quantity = model.to_model_units(points)

        # Extract numerical values for internal mesh operations
        if hasattr(model_quantity, "_pint_qty"):
            model_points = model_quantity._pint_qty.magnitude
        else:
            model_points = model_quantity

        # Coerce cells to a 1-D numpy array — accept list/tuple input as the
        # docstring promises ("array-like") even though the internal helper
        # calls cells.reshape(-1) directly.
        cells = numpy.asarray(cells).reshape(-1)

        # Call internal implementation
        return self._test_if_points_in_cells_internal(
            model_points, cells, on_boundary=on_boundary, tol=tol,
        )

    def get_closest_local_cells(
        self,
        coords: numpy.ndarray,
        on_boundary: bool = True,
        tol: float = 0.0,
    ) -> numpy.ndarray:
        """
        This method uses a kd-tree algorithm to find the closest
        cells to the provided coords. For a regular mesh, this should
        be exactly the owning cell, but if the mesh is deformed, this
        is not guaranteed. Also compares the distance from the cell to the
        point - if this is larger than the "cell size" then returns -1

        Parameters:
        -----------
        coords:
            An array of the coordinates for which we wish to determine the
            closest cells. This should be a 2-dimensional array of
            shape (n_coords,dim) in any physical unit system (will be auto-converted).
        on_boundary : bool, default True
            If True (the default), queries exactly on a cell face are
            treated as inside that cell (natural for FE-evaluation hints —
            mesh vertices sit on cell faces by definition). If False,
            strict-inside semantics; boundary queries return -1.
        tol : float, default 0.0
            Face-relative tolerance forwarded to
            `_test_if_points_in_cells_internal`. When ``> 0`` takes
            precedence over ``on_boundary``: the test admits points
            within ``tol`` of the face relative to the control-point
            separation² — used by the parallel evaluation locator
            for on-face / near-face queries at the mesh-spacing scale.

        Returns:
        --------
        closest_cells:
            An array of indices representing the cells closest to the provided
            coordinates. This will be a 1-dimensional array of
            shape (n_coords).
        """
        # Convert coords to model units using the elegant protocol
        import underworld3 as uw

        model = uw.get_default_model()
        model_quantity = model.to_model_units(coords)

        # Extract numerical values for internal mesh operations
        if hasattr(model_quantity, "_pint_qty"):
            model_coords = model_quantity._pint_qty.magnitude
        else:
            model_coords = model_quantity

        # Call internal implementation
        return self._get_closest_local_cells_internal(
            model_coords, on_boundary=on_boundary, tol=tol,
        )

    # Face tolerance for the parallel evaluation locator (relative to the
    # control-point separation). Tight: it admits points genuinely on a
    # cell face/edge (containment value ~0) but rejects points sitting inside
    # a *different* cell (value strongly negative), so a rank never claims a
    # point another rank owns. A point in the (closed) domain is then found by
    # exactly one rank (verified: 360/360, owners==1), and the migration routes
    # it to that owner.
    _EVAL_FACE_TOL = 1.0e-2

    def _robust_owning_cells(self, coords: numpy.ndarray) -> numpy.ndarray:
        """Per-point owning cell for parallel evaluation (coords in model units).

        This is the strict barycentric/cell-wall locator
        (:meth:`_get_closest_local_cells_internal`) with a *tight* face
        tolerance (``_EVAL_FACE_TOL``): it returns the containing cell for
        interior points and a valid sharing cell for genuinely-on-face points,
        and ``-1`` for points that lie inside a *different* cell or outside the
        local mesh. Crucially it does **not** fall back to a bounding-sphere
        "nearest cell" — that earlier fallback let a rank claim a point that
        another rank actually owns, so the eval-swarm migration stranded it on
        the wrong rank and ξ-clamp-evaluated it in an adjacent cell (the
        partition-seam hotspots). With the strict+tol locator the point is
        found only by its true owner, the migration delivers it there, and it
        evaluates exactly.

        Never calls PETSc ``DMLocatePoints`` (slow, raises out-of-domain), and
        is purely kd-tree / Euclidean — manifold-safe, no manifold branch.
        """
        coords = numpy.asarray(coords)
        if coords.shape[0] == 0:
            return numpy.array([], dtype=numpy.int64)
        return numpy.asarray(
            self._get_closest_local_cells_internal(coords, tol=self._EVAL_FACE_TOL),
            dtype=numpy.int64,
        )

    def _audit_cell_face_geometry(self):
        """Measure the two quantities that bound the cell-wall estimator's
        authority on quad/hex meshes: worst face non-planarity (sagitta) and
        worst outward half-space violation by a cell's own vertices, both
        relative to the face diameter.

        The in-cell test (:meth:`_test_if_points_in_cells_internal`) is a
        half-space intersection over the face planes the control points
        define. It is exact when every face is planar and every cell is
        convex; a warped face confines misclassification to a slab of
        thickness ~sagitta at that face (and the misassigned cell is the
        face-adjacent neighbour, where continuous FE interpolants agree).
        The violation term catches non-convex / tangled cells, whose
        half-space intersection no longer represents the cell at all.

        Returns ``(max_sagitta_rel, max_violation_rel)`` over local cells;
        ``(0.0, 0.0)`` on an empty local mesh.
        """
        nav_dm = self._nav_dm if self._nav_dm is not None else self.dm
        nav_coords = self._nav_coords

        cStart, cEnd = nav_dm.getHeightStratum(0)
        if cEnd == cStart:
            return 0.0, 0.0
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        max_sag = 0.0
        max_viol = 0.0
        for cell_id in range(cStart, cEnd):
            cell_faces = nav_dm.getCone(cell_id)
            cpoints = nav_dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
            cell_point_coords = nav_coords[self._coord_rows_for_points(nav_dm, cpoints)]
            cell_centroid = cell_point_coords.mean(axis=0)
            for face in range(cell_num_faces):
                fpoints = nav_dm.getTransitiveClosure(cell_faces[face])[0][-face_num_points:]
                point_coords = nav_coords[self._coord_rows_for_points(nav_dm, fpoints)]
                face_centroid = point_coords.mean(axis=0)
                normal = self._facet_outward_unit_normal(
                    point_coords, face_centroid, cell_centroid,
                    cell_point_coords=cell_point_coords,
                )
                diam = float(numpy.linalg.norm(
                    point_coords.max(axis=0) - point_coords.min(axis=0)))
                if diam == 0.0:
                    continue
                sag = float(numpy.abs((point_coords - face_centroid) @ normal).max())
                viol = float(((cell_point_coords - face_centroid) @ normal).max())
                max_sag = max(max_sag, sag / diam)
                max_viol = max(max_viol, viol / diam)
        return max_sag, max_viol

    # Capability thresholds. EXACT demands machine-planar faces and convex
    # cells; CONTINUOUS admits face warp up to 5% of the face diameter (the
    # cubed sphere measures ~1e-2 on its spherical faces; a smoothly deformed
    # hex box ~1e-2). Beyond that the estimator's error bound is no longer
    # small against per-cell field variation and it loses authority entirely.
    _LOCATION_EXACT_TOL = 1.0e-9
    _LOCATION_CONTINUOUS_TOL = 5.0e-2

    def _location_capability(self) -> str:
        """Measured point-location capability of this mesh's cell-wall
        estimator: ``"exact"``, ``"continuous"``, or ``"none"``.

        * ``"exact"`` — the estimator is authoritative for every field type:
          simplex meshes, manifold meshes (dim != cdim, where PETSc's own
          in-cell test is the unreliable party), and any quad/hex mesh whose
          measured face sagitta and convexity violation are at machine level
          (rectilinear boxes, affine images, and *all valid 2-D quad meshes* —
          straight edges are always planar and convexity is equivalent to an
          untangled mesh).
        * ``"continuous"`` — authoritative for continuous fields only: warped
          hexes within the sagitta tolerance (cubed-sphere class). A point in
          the misclassification slab lands in the face-adjacent neighbour,
          where continuous FE interpolants agree to O(sagitta x gradient) —
          but a field with a face-aligned jump would see O(jump) errors, so
          discontinuous evaluation must not trust the estimator here.
        * ``"none"`` — badly warped or non-convex cells: the estimator carries
          no authority and dropped points take the RBF fallback.

        The audit is computed per rank over local cells and cached against
        ``(_mesh_version, _topology_version)`` — ``deform()`` and adaptation
        both bump these. It is deliberately NOT reduced across ranks: the
        evaluator runs on COMM_SELF, so a (pathological) capability split at
        a threshold is rank-local and safe, while a collective reduction here
        could deadlock (petsc_interpolate is reached only by ranks holding
        points).
        """
        key = (self._mesh_version, self._topology_version)
        cached = getattr(self, "_location_capability_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        if bool(self.dm.isSimplex()) or (self.dim != self.cdim):
            cap = "exact"
        else:
            sag_rel, viol_rel = self._audit_cell_face_geometry()
            worst = max(sag_rel, viol_rel)
            if worst < self._LOCATION_EXACT_TOL:
                cap = "exact"
            elif worst < self._LOCATION_CONTINUOUS_TOL:
                cap = "continuous"
            else:
                cap = "none"
        self._location_capability_cache = (key, cap)
        return cap

    def _hint_is_authoritative(self, all_fields_continuous: bool = True) -> bool:
        """Whether the cell-wall hint may bypass ``DMLocatePoints`` for an
        evaluation involving the given field continuity. See
        :meth:`_location_capability` for the regimes; ``"continuous"``
        capability is only authoritative when every interpolated variable is
        continuous.
        """
        cap = self._location_capability()
        return cap == "exact" or (cap == "continuous" and bool(all_fields_continuous))

    def _eval_use_robust_location(self) -> bool:
        """Single switch for the parallel evaluation cell-location strategy.

        Returns True when ``uw.function`` evaluation should locate cells with
        the bulletproof barycentric hint (:meth:`_robust_owning_cells`) and the
        ``DMLocatePoints`` bypass (``petsc_tools.c``), rather than PETSc's
        ``DMLocatePoints``. This is the *one place* the parallel policy lives;
        the evaluate_nd classifier, the petsc_interpolate hint, and the
        DMInterpolation wrapper all defer to it so the three stay consistent.

        Two conditions, both required:

        * **parallel only** (``uw.mpi.size > 1``) — in serial the on-face/edge
          node points go to RBF-at-node (exact) and PETSc/the cell-wall test
          are reliable with a single domain, so serial keeps the validated path
          bit-for-bit. The parallel-only failure is the rank-local RBF / wrong-
          region value at partition-seam node points.
        * **the estimator carries geometric authority** — measured capability
          ``"exact"`` or ``"continuous"`` (:meth:`_location_capability`). For
          in/out *classification* the continuous regime is sufficient: the
          only ambiguity is within the sagitta slab of the domain boundary,
          which is exactly the tolerance the classifier already accepts.
          Capability ``"none"`` (badly warped / non-convex quad meshes) keeps
          PETSc's DMLocatePoints search.
        """
        return (uw.mpi.size > 1) and (self._location_capability() != "none")

    def _get_mesh_sizes(self, verbose=False):
        """
        Obtain the (local) mesh radii and centroids using kdtree distances
        This routine is called when the mesh is built / rebuilt
        """

        centroids = self._get_coords_for_basis(0, False)
        centroids_kd_tree = uw.kdtree.KDTree(centroids)

        import numpy as np

        cStart, cEnd = self.dm.getHeightStratum(0)
        pStart, pEnd = self.dm.getDepthStratum(0)
        cell_length = np.empty(centroids.shape[0])
        cell_min_r = np.empty(centroids.shape[0])
        cell_r = np.empty(centroids.shape[0])

        for cell in range(cEnd - cStart):
            cell_num_points = self.dm.getConeSize(cell)
            cell_points = self.dm.getTransitiveClosure(cell)[0][-cell_num_points:]
            # Use raw internal array for internal mesh operations (avoid unit-aware wrapping)
            cell_coords = self._coords[cell_points - pStart]

            distsq, _ = centroids_kd_tree.query(cell_coords, k=1, sqr_dists=True)

            cell_length[cell] = np.sqrt(distsq.max())
            cell_r[cell] = np.sqrt(distsq.mean())
            cell_min_r[cell] = np.sqrt(distsq.min())

        return cell_min_r, cell_r, centroids, cell_length

    # ==========

    # Deprecated in favour of _get_mesh_sizes (above)
    def _get_mesh_centroids(self):
        """
        Obtain and cache the (local) mesh centroids using underworld swarm technology.
        This routine is called when the mesh is built / rebuilt

        The global cell number corresponding to a centroid is (supposed to be)
        self.dm.getCellNumbering().array.min() + index

        """

        # (
        #     sizes,
        #     centroids,
        # ) = petsc_discretisation.petsc_fvm_get_local_cell_sizes(self)

        centroids = self._get_coords_for_basis(0, False)

        return centroids

    def _get_domain_centroids(self):

        import numpy as np
        from underworld3.utilities import gather_data

        domain_centroid = self._centroids.mean(axis=0)
        all_centroids = gather_data(domain_centroid, bcast=True).reshape(-1, self.cdim)
        return all_centroids

    def _get_domain_kdtree(self):
        import underworld3 as uw
        if (
            not hasattr(self, "_domain_kdtree")
            or self._domain_kdtree is None
            or getattr(self, "_domain_kdtree_version", -1) != self._mesh_version
        ):
            centroids = self._get_domain_centroids()
            self._domain_kdtree = uw.kdtree.KDTree(centroids)
            self._domain_kdtree_version = self._mesh_version
        return self._domain_kdtree

    def get_min_radius_old(self) -> float:
        """
        This method returns the global minimum distance from any cell centroid to a face.
        It wraps to the PETSc `DMPlexGetMinRadius` routine. The petsc4py equivalent always
        returns zero.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        from underworld3.cython.petsc_discretisation import petsc_fvm_get_min_radius

        if (not hasattr(self, "_min_radius")) or (self._min_radius == None):
            self._min_radius = petsc_fvm_get_min_radius(self)

        return self._min_radius

    @uw.collective_operation
    def get_min_radius(self) -> float:
        """
        This method returns the global minimum distance from any cell centroid to a face.
        It wraps to the PETSc `DMPlexGetMinRadius` routine. The petsc4py equivalent always
        returns zero.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        import numpy as np

        all_min_radii = uw.utilities.gather_data(np.array((self._radii.min(),)), bcast=True)

        return all_min_radii.min()

    @uw.collective_operation
    def get_max_radius(self) -> float:
        """
        This method returns the global maximum distance from any cell centroid to a face.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        import numpy as np

        all_max_radii = uw.utilities.gather_data(np.array((self._radii.max(),)), bcast=True)

        return all_max_radii.max()

    @uw.collective_operation
    def get_mean_radius(self) -> float:
        """
        Global mean of the characteristic cell length scale
        (``volume^(1/dim)``, i.e. the equivalent radius derived from each
        cell's volume — the same quantity averaged by ``get_min_radius``
        and ``get_max_radius`` to obtain global min/max). Parallel-safe
        via MPI allreduce of the local sum and count.

        Together with :meth:`get_min_radius` / :meth:`get_max_radius`
        this is the canonical "mesh length" API. Use this anywhere you
        need a representative h0 (smoothing-length defaults, diffusion-
        stability heuristics, problem-scale normalisation) rather than
        reaching for the rank-local ``self._radii`` array, which gives
        different answers on different MPI ranks and leaks downstream
        (e.g. into JIT C source via per-rank pointwise-function inputs).
        """

        import numpy as np
        from mpi4py import MPI

        radii = np.asarray(self._radii)
        local_sum = float(radii.sum())
        local_n = int(radii.size)
        if uw.mpi.size > 1:
            local_sum = uw.mpi.comm.allreduce(local_sum, op=MPI.SUM)
            local_n = uw.mpi.comm.allreduce(local_n, op=MPI.SUM)
        return local_sum / max(local_n, 1)

    # This should be deprecated in favour of using integrals
    def stats(self, uw_function, uw_meshVariable, basis=None):
        """
        Returns various norms on the mesh for the provided function.
          - size
          - mean
          - min
          - max
          - sum
          - L2 norm
          - rms

          NOTE: this currently assumes scalar variables !
        """

        #       This uses a private work MeshVariable and the various norms defined there but
        #       could either be simplified to just use petsc vectors, or extended to
        #       compute integrals over the elements which is in line with uw1 and uw2

        if basis is None:
            basis = self.N

        from petsc4py.PETSc import NormType

        tmp = uw_meshVariable
        tmp.data[...] = uw.function.evaluate(uw_function, tmp.coords, basis).reshape(-1, 1)

        vsize = tmp._gvec.getSize()
        vmean = tmp.mean()
        vmax = tmp.max()[1]
        vmin = tmp.min()[1]
        vsum = tmp.sum()
        vnorm2 = tmp.norm(NormType.NORM_2)
        vrms = vnorm2 / numpy.sqrt(vsize)

        return vsize, vmean, vmin, vmax, vsum, vnorm2, vrms

    def meshVariable_mask_from_label(self, label_name, label_value):
        """Extract single label value and make a point mask - note: this produces a mask on the mesh points and
        assumes a 1st order mesh. Cell labels are not respected in this function."""

        meshVar = MeshVariable(
            f"Mask_{label_name}_{label_value}",
            self,
            vtype=uw.VarType.SCALAR,
            degree=1,
            continuous=True,
            varsymbol=rf"\cal{{M}}^{{[{label_name:.4}]}}",
        )

        point_indices = petsc_dm_find_labeled_points_local(
            self.dm,
            label_name,
            label_value,
            sectionIndex=False,
        )

        meshVar.data[...] = 0.0
        if point_indices is not None:
            meshVar.data[point_indices] = 1.0

        return meshVar

    def register_swarm(self, swarm):
        """Register swarm as dependent on this mesh for coordinate change notifications"""
        self._registered_swarms.add(swarm)

    def unregister_swarm(self, swarm):
        """Unregister swarm (called during swarm cleanup)"""
        # WeakSet handles weak references internally, just remove the swarm directly
        self._registered_swarms.discard(swarm)

    def register_surface(self, surface):
        """Register surface as dependent on this mesh for adaptation notifications."""
        self._registered_surfaces.add(surface)

    def unregister_surface(self, surface):
        """Unregister surface (called during surface cleanup)."""
        self._registered_surfaces.discard(surface)

    def _increment_mesh_version(self):
        """
        Manually increment mesh version to notify swarms of coordinate changes.
        This is called automatically when mesh.points is modified, but can be
        called manually if coordinates are changed through other means.
        """
        with self._mesh_update_lock:
            self._mesh_version += 1
            print(f"Mesh version manually incremented to {self._mesh_version}")

    def OT_adapt(self, *args, **kwargs):
        """RETIRED (2026-07). The optimal-transport reset adapt was
        superseded by the variational MMPDE mover; calling this raises
        RuntimeError.

        Use ``uw.meshing.follow_metric(mesh, field, refinement=...)`` (the
        two-knob gradient-following adapter) or
        ``uw.meshing.smooth_mesh_interior(mesh, metric=..., method="mmpde")``
        for fixed-topology node redistribution; use :meth:`adapt` when more
        resolution than a fixed node budget can provide is needed.
        """
        raise RuntimeError(
            "mesh.OT_adapt was retired (2026-07): the OT reset adapt was "
            "superseded by the variational MMPDE mover. Use "
            "uw.meshing.follow_metric(mesh, field, refinement=...) or "
            "uw.meshing.smooth_mesh_interior(mesh, metric=..., "
            "method='mmpde'); for a topology change use mesh.adapt(...).")

    @timing.routine_timer_decorator
    def _wrap_coarse_level(self, dm):
        """Wrap a (static) coarse-hierarchy DM as a UW Mesh carrying this mesh's
        boundary labels — a coarse level for the custom-P geometric-MG hierarchy."""
        return Mesh(
            dm.clone(),
            simplex=self.dm.isSimplex(),
            coordinate_system_type=self.CoordinateSystem.coordinate_type,
            qdegree=self.qdegree,
            boundaries=self.boundaries,
            verbose=False,
        )

    def _coarse_level_meshes(self):
        """The static coarse-mesh tail (one Mesh per base hierarchy level,
        coarsest..base-finest), built once and cached — they never change
        because the base hierarchy is static across adapts."""
        cached = getattr(self, "_coarse_level_meshes_cache", None)
        if cached is None:
            cached = [self._wrap_coarse_level(d) for d in self.dm_hierarchy]
            self._coarse_level_meshes_cache = cached
        return cached

    def redistribute_nodes(self, metric, *, verbose=False, **kwargs):
        r"""Move this mesh's nodes so cell sizes follow ``metric``, with
        the topology fixed.

        Node **redistribution** concentrates the existing node budget
        where the metric demands resolution: vertex count, vertex ids,
        DOF layout and the parallel partition are all preserved — only
        coordinates move. Mesh variables (and solver / transport
        history) are transferred onto the moved nodes automatically.
        Contrast :meth:`adapt`, which *refines* (returns a child mesh
        with more cells), and :meth:`remesh`, which regenerates the
        mesh in place.

        This method is how each mesh type controls whether (and how) it
        can be modified: the base implementation supports **2D simplex
        (triangle) meshes**, where it drives the Huang–Kamenski
        variational MMPDE mover (non-folding by construction,
        parallel-safe; scalar metric → isotropic equidistribution,
        tensor metric → anisotropic clustering and alignment).
        Quadrilateral / hexahedral meshes, 3D meshes and constrained
        manifolds raise ``NotImplementedError`` — no mover is
        implemented for them yet.

        Parameters
        ----------
        metric : sympy expression, MeshVariable, or sympy Matrix
            Target density :math:`\rho(x)` (scalar, larger ⇒ finer
            cells) or a full :math:`d \times d` SPD metric tensor
            (anisotropic: small across a feature, base along it).
        verbose : bool, default False
            Print mover progress.
        **kwargs
            Forwarded to
            :func:`underworld3.meshing.smooth_mesh_interior` — e.g.
            ``pinned_labels``, ``slip_surfaces``, ``skip_threshold``,
            ``method_kwargs`` (the mover tunables).

        Examples
        --------
        >>> x, y = mesh.CoordinateSystem.X
        >>> rho = 1 + 8 * sympy.exp(-(((x - 0.5)**2 + (y - 0.5)**2) / 0.05))
        >>> mesh.redistribute_nodes(rho)

        See Also
        --------
        underworld3.meshing.node_redistribution : The free-function
            spelling of this operation.
        underworld3.meshing.follow_metric : Two-knob adapter that
            builds the metric from a field gradient.
        adapt : Add resolution (topology change, returns a child mesh).
        """
        if self.dim != self.cdim:
            raise NotImplementedError(
                "node redistribution is not implemented for constrained-"
                f"manifold meshes (mesh.dim={self.dim} != mesh.cdim="
                f"{self.cdim}): every node would have to be constrained "
                "to the surface. Implemented today: 2D simplex (triangle) "
                "meshes, via the MMPDE mover.")
        if not bool(self.dm.isSimplex()) or self.cdim != 2:
            kind = "simplex" if bool(self.dm.isSimplex()) else "tensor-product (quad/hex)"
            raise NotImplementedError(
                f"node redistribution is not implemented for {self.cdim}D "
                f"{kind} meshes. Implemented today: 2D simplex (triangle) "
                "meshes, via the MMPDE mover (its 3D / quad discretization "
                "does not exist yet). To add resolution instead, use "
                "mesh.adapt(metric_field, max_levels=...) — a topology "
                "change.")
        from underworld3.meshing.smoothing import smooth_mesh_interior
        smooth_mesh_interior(self, metric=metric, method="mmpde",
                             verbose=verbose, **kwargs)

    def adapt(self, metric_field, max_levels=None, node_budget=None,
              builder=None, adapter=None, engine=None, verbose=False):
        r"""
        Nested **adapt-on-top**: return a refined **child** mesh.

        Locally refine the static base finest where the metric demands resolution,
        **on top of** the existing uniform hierarchy, and return a new child mesh
        (``child.parent is self``). The base mesh is **not modified** — this is
        *adapt / re-adapt*, not node movement: each call re-marks from the static
        base finest, so successive adapts are non-cumulative (cf. :meth:`remesh`,
        which regenerates the mesh in place via MMG and may redistribute).

        The default call needs no engine choice —
        ``mesh.adapt(metric, max_levels=...)`` refines with the graded
        newest-vertex-bisection engine, on **2D triangle and 3D tetrahedral
        meshes, serial and parallel** (the refined mesh is
        partition-independent: the same mesh at any communicator size).
        ``engine=`` remains available as an **advanced / internal selector**
        (the algorithm names live here, not in the everyday call):

        * ``"nvb"`` (default) — newest-vertex bisection, a **graded** engine with a
          *bounded* conforming closure: a marked cell adds O(1) cells locally, so
          successive levels grade (a level+1 ring around a finer core) and DOFs
          concentrate near the feature. Runs **in parallel** in both
          dimensions via the native ``uwnvb`` ``DMPlexTransform`` driver
          (in-place, co-partitioned with the parent, bit-confluent
          serial↔parallel; in 3D the per-cell refinement state is seeded
          identically on every rank from geometry). When the compiled
          extension is absent it falls back to the serial cell-list engines
          (``NVBMesh`` / ``TaggedBisectionMesh``; ``NotImplementedError`` at
          np>1). Bisects 1→2 (volume halves), so one isotropic-equivalent
          ``max_levels`` is run as **dim** bisection generations.
        * ``"sbr"`` — PETSc skeleton-based (longest-edge) bisection, **2D
          only** (PETSc's SBR transform cannot handle tetrahedra). Each
          pass refines marked cells isotropically (1→4). Its conforming closure is
          *unbounded for region marking*, so it produces a **uniform-finest patch**,
          not a graded mesh (a marked cell drains the longest-edge path to the patch
          edge). Robust and fine for the MG hierarchy.

        The child owns a custom-P geometric-MG hierarchy with **one level per
        refinement step** — ``[base L0 … base finest, refine-1, …, refine-n]`` —
        so every solver built on it drives geometric multigrid on the refined
        operator with no per-solver setup. (Each ``max_levels`` pass adds its
        own MG level; the transfers between consecutive levels each span a single
        refinement.)

        .. note::
            ``mesh.adapt(metric)`` with **no other keyword** raises a
            ``TypeError``: that call shape used to be the in-place MMG remesher
            (now :meth:`remesh`), and a legacy caller would silently discard
            the returned child. Pass any keyword — e.g.
            ``adapt(metric, max_levels=2)`` — to opt in to the new semantics.

        Parameters
        ----------
        metric_field : MeshVariable, sympy expression, or callable
            Scalar metric ``M = 1/h²`` (target edge length ``h``); larger ⇒ finer.
            A **MeshVariable** or **sympy/UWexpression** is sampled through
            ``uw.function.evaluate`` (so anything it references is interpolated
            from the *base* mesh). A **callable** ``metric(centroids) -> M`` is
            evaluated directly at each refined level's centroids — use this for a
            metric built from exact geometry (e.g.
            :meth:`Surface.refinement_metric_function`) so a thin feature refines
            to a clean, uniform-width band instead of a P1-aliased *patchy* one.
            Same interface as :meth:`remesh` / ``adaptivity.create_metric``.
        max_levels : int, default 2
            Maximum refinement depth applied on top of the base finest (bounds
            the on-rank imbalance). Each level re-marks against the metric.
        node_budget : int or None
            Optional cap on the number of *seed* cells marked per level (highest-
            metric first). **Caveat:** this caps the marked seeds, *not* the
            resulting DOFs — SBR's conforming closure re-refines the whole
            connected patch from any seed in it, so a seed cap does not bound the
            added DOFs and cannot, on its own, concentrate the finest level near a
            feature. To make the **finest level hug a feature** (a funnel) with
            bounded per-level growth, shape the **metric** so element size grows
            with distance (e.g. ``Surface.refinement_metric(..., profile="linear")``,
            a wedge), not this budget. A *flat-core* metric (e.g. a Gaussian)
            refines the whole core uniformly at every level (no funnel).
            ``None`` ⇒ uncapped.
        builder : {"barycentric", "rbf"}
            Per-level node-prolongation builder for the child's custom-P hierarchy.
        adapter : {"sbr", "mmg"}
            ``"sbr"`` (default) is the nested adapt-on-top path (the refinement
            engine is then chosen by ``engine``). ``"mmg"`` is a **deprecated shim**
            that forwards to :meth:`remesh` (in-place, returns ``self``).
        engine : {"nvb", "sbr"}, optional
            Advanced selector for the nested refinement engine (ignored when
            ``adapter="mmg"``). Default ``"nvb"`` — graded newest-vertex
            bisection; ``"sbr"`` is longest-edge bisection (uniform patch,
            still the right choice when a uniform-finest MG patch is wanted).
            See above.
        verbose : bool

        Returns
        -------
        Mesh
            The refined child (or ``self`` when ``adapter='mmg'``).

        Notes
        -----
        **Controlling the grading (funnel toward a feature).** Each SBR pass
        refines every cell that is still coarser than the metric target, so the
        final grading *is* whatever the metric specifies. To make the **finest**
        level hug a feature (a funnel), use a metric whose target size grows with
        distance from the feature — a *wedge*, e.g.
        ``Surface.refinement_metric(h_near, h_far, width, profile="linear")`` with
        a small ``width``. A flat-core metric (Gaussian) instead refines the whole
        core uniformly at every level, so every level has the same width.

        SBR's conforming closure re-refines the whole connected patch from any
        marked seed, so the funnel must come from the metric shape, not from
        capping seed counts (see ``node_budget``). For a 1-D feature in 2-D the
        per-level DOFs still grow (the along-feature resolution doubles each
        level); the wedge keeps the total finite and feature-concentrated.
        """
        import warnings

        # Legacy-call guard. Before 2026-07 ``mesh.adapt(metric)`` was the
        # IN-PLACE MMG remesher (now :meth:`remesh`); the nested adapt-on-top
        # returns a NEW child mesh instead, which a legacy caller would silently
        # discard. A bare call with no new-API keyword is therefore ambiguous
        # and refused rather than redirected (the return semantics differ).
        if (max_levels is None and node_budget is None and builder is None
                and adapter is None and engine is None):
            raise TypeError(
                "mesh.adapt(metric) has changed meaning: the in-place metric "
                "adaptation this call shape used to perform is now "
                "mesh.remesh(metric). mesh.adapt(...) performs nested "
                "adapt-on-top refinement and RETURNS A NEW child mesh (the "
                "base mesh is not modified) — opt in explicitly by passing "
                "any of its keywords, e.g. mesh.adapt(metric, max_levels=2) "
                "or mesh.adapt(metric, engine='nvb'), and keep the returned "
                "child."
            )
        if max_levels is None:
            max_levels = 2
        if builder is None:
            builder = "barycentric"
        if adapter is None:
            adapter = "sbr"
        if engine is None:
            # NVB is the default refinement engine (2026-07 naming ruling):
            # graded, bounded conforming closure, parallel (2D) via the
            # native uwnvb transform. In 3D the serial tagged-simplex engine
            # serves np=1; an np>1 3D call raises honestly inside the engine
            # guard (the parallel tetrahedral transform is capstone stage
            # 1c). Note engine='sbr' is NOT a 3D fallback: PETSc's SBR
            # transform handles triangles only (error 56 on tetrahedra).
            engine = "nvb"

        if adapter == "mmg":
            warnings.warn(
                "mesh.adapt(adapter='mmg') is deprecated; the in-place MMG "
                "remesher is now mesh.remesh(). Call mesh.remesh(metric) instead.",
                DeprecationWarning, stacklevel=2,
            )
            self.remesh(metric_field, verbose=verbose)
            return self
        if adapter != "sbr":
            raise ValueError(f"adapter must be 'sbr' or 'mmg', got {adapter!r}")
        if engine not in ("sbr", "nvb"):
            raise ValueError(f"engine must be 'sbr' or 'nvb', got {engine!r}")

        return self._adapt_nested(
            metric_field, max_levels=max_levels, node_budget=node_budget,
            builder=builder, engine=engine, verbose=verbose,
        )

    def _adapt_nested(self, metric_field, max_levels=2, node_budget=None,
                      builder="barycentric", engine="nvb", verbose=False):
        """Core nested adapt-on-top (SBR or NVB engine). See :meth:`adapt`."""
        import math
        from underworld3.utilities import custom_mg

        if self.parent is not None:
            raise NotImplementedError(
                "adapt() refines a BASE mesh; chaining adapt on an already-adapted "
                "child is not yet supported."
            )
        if getattr(self, "dm_hierarchy", None) is None or len(self.dm_hierarchy) < 2:
            raise RuntimeError(
                "Nested adapt-on-top needs a base mesh built with refinement>=1 (a "
                "dm_hierarchy of coarse levels supplies the geometric-MG tail). "
                "Build the mesh with e.g. refinement=2."
            )
        # The native uwnvb DMPlexTransform (Route B) is the parallel NVB engine:
        # in-place (co-partitioned with the parent), graded, and bit-confluent
        # serial<->parallel. It bisects triangles only, so it serves the 2D
        # path; 3D (tetrahedra) runs the serial dimension-general tagged-
        # simplex engine until the native transform adopts the tagged rule
        # (adaptivity capstone stage 1c).
        _nvbx = None
        if engine == "nvb":
            if not bool(self.dm.isSimplex()) or self.dim not in (2, 3):
                raise NotImplementedError(
                    "adapt(engine='nvb') supports 2D triangle and 3D "
                    "tetrahedral meshes."
                )
            if self.dim == 2:
                try:
                    from underworld3.utilities import _nvb_transform as _nvbx
                except ImportError:
                    _nvbx = None
                if _nvbx is None and uw.mpi.size > 1:
                    raise NotImplementedError(
                        "adapt(engine='nvb') at np>1 needs the native uwnvb "
                        "transform (underworld3.utilities._nvb_transform), "
                        "which is not built in this environment. Build the "
                        "custom-PETSc/amr env, or use engine='sbr' at np>1."
                    )
            else:
                # 3D: prefer the native driver when built (same preference
                # order as 2D); the serial cell-list engine is the np=1
                # fallback. The native driver needs the per-cell refinement
                # state seeded on the base mesh first (partition-independent
                # by construction — see write_tagged_state_label).
                try:
                    from underworld3.utilities import _nvb_transform as _nvbx
                except ImportError:
                    _nvbx = None
                if _nvbx is None and uw.mpi.size > 1:
                    raise NotImplementedError(
                        "adapt(engine='nvb') at np>1 needs the native uwnvb "
                        "transform (underworld3.utilities._nvb_transform), "
                        "which is not built in this environment. Build the "
                        "custom-PETSc/amr env."
                    )
                if _nvbx is not None:
                    from underworld3.utilities.nvb import (
                        write_tagged_state_label)
                    write_tagged_state_label(self.dm_hierarchy[-1])

        dim = self.dim
        DM_ADAPT_REFINE = 1                  # PETSc DMAdaptFlag: refine this cell

        # The metric is normalised to a single callable `eval_metric(centroids)
        # -> M`, re-evaluated at each refined level's centroids. There is only one
        # code path; the metric *kind* just decides which callable we build:
        #
        #   * a plain CALLABLE metric(centroids) -> M is used as-is;
        #   * a MeshVariable (.sym) or sympy/UWexpression is wrapped in an
        #     `uw.function.(global_)evaluate` adapter — i.e. fn.evaluate IS a
        #     callable in this framework, just the default one for a field/expr.
        #
        # The distinction is about *where the metric resolves*. The evaluate
        # adapter samples the base-mesh interpolant, so anything the field/expr
        # references (e.g. a Surface.distance P1 field, or a peaked M = 1/h²)
        # aliases across a base cell → *patchy* levels along a thin feature. A
        # callable built from EXACT geometry (Surface.refinement_metric_function)
        # instead resolves itself at the refined resolution — a clean band — and,
        # being coordinate-driven, is partition-independent (no swarm-migration
        # global_evaluate). A user callable is free to call global_evaluate itself
        # when the metric genuinely depends on a base field (e.g. |∇T| for
        # convection); use global_evaluate, not evaluate, at np>1.
        import sympy as _sympy
        metric_is_callable = (
            callable(metric_field)
            and not hasattr(metric_field, "sym")
            and not isinstance(metric_field, _sympy.Basic)
        )

        if metric_is_callable:
            def eval_metric(centroids):
                return numpy.asarray(metric_field(centroids), dtype=float).reshape(-1)
        else:
            # Wrap the field/expression as a callable over the (global_)evaluate
            # sampler — the same framework, with fn.evaluate as the adapter.
            metric_sym = getattr(metric_field, "sym", metric_field)
            _sampler = (uw.function.global_evaluate if uw.mpi.size > 1
                        else uw.function.evaluate)

            def eval_metric(centroids):
                return numpy.asarray(_sampler(metric_sym, centroids)).reshape(-1)

        def cell_geometry(dm):
            """Per-cell centroid and size for every cell of a simplicial DM.

            A simplex centroid is the vertex mean and its volume is
            |det(edge vectors)|/dim!, so h = (dim!·vol)^(1/dim) reduces to
            |det|^(1/dim) — one vectorised det over the cell list. (The
            per-cell petsc4py ``computeCellGeometryFVM`` calls this replaces
            dominated the marking cost at bisection depth ≥ 3.)
            """
            cs, ce = dm.getHeightStratum(0)
            n = ce - cs
            if n == 0:
                return numpy.empty((0, self.cdim)), numpy.empty(0), cs
            vS, vE = dm.getDepthStratum(0)
            verts = numpy.empty((n, dim + 1), dtype=numpy.int64)
            for i, c in enumerate(range(cs, ce)):
                verts[i] = [p for p in dm.getTransitiveClosure(c)[0]
                            if vS <= p < vE]
            X = dm.getCoordinatesLocal().array.reshape(-1, self.cdim)[
                verts - vS]
            e = X[:, 1:, :] - X[:, :1, :]
            if self.cdim == dim:
                vol_scaled = numpy.abs(numpy.linalg.det(e))
            else:                        # manifold: Gram determinant
                G = e @ e.transpose(0, 2, 1)
                vol_scaled = numpy.sqrt(numpy.abs(numpy.linalg.det(G)))
            return X.mean(axis=1), vol_scaled ** (1.0 / dim), cs

        # Refine from the mesh's CURRENT geometry. Node redistribution
        # (redistribute_nodes) moves mesh.dm's coordinates while the static
        # base hierarchy keeps the originals — without this carry, an adapt
        # after a redistribution would silently refine the unmoved mesh.
        # When moved, the base-finest MG tail level is swapped for the moved
        # copy below; the coarser tail levels keep their original geometry
        # (the coordinate-based custom-P transfers accept non-nested pairs).
        base_finest = self.dm_hierarchy[-1]
        _cur_coords = self.dm.getCoordinatesLocal().array
        _moved = not numpy.array_equal(
            _cur_coords, base_finest.getCoordinatesLocal().array)
        if uw.mpi.size > 1:
            from mpi4py import MPI
            _moved = uw.mpi.comm.allreduce(_moved, op=MPI.LOR)
        if _moved:
            base_finest = base_finest.clone()
            # DMClone shares the coordinates Vec by reference — writing
            # through getCoordinatesLocal().array here would silently move
            # the "static" hierarchy level (self.dm_hierarchy[-1]) under
            # the parent mesh. Install a duplicate instead; the clone gets
            # the moved geometry, the hierarchy keeps the original.
            _v = base_finest.getCoordinatesLocal().duplicate()
            _v.array[:] = _cur_coords
            base_finest.setCoordinatesLocal(_v)
            if engine == "nvb" and self.dim == 3 and _nvbx is not None:
                # the refinement-state seed was written on the unmoved base
                # (in the guard above); re-seed on the moved geometry
                from underworld3.utilities.nvb import write_tagged_state_label
                write_tagged_state_label(base_finest)

        markers_per_level = []
        level_dms = []                       # one DM per refinement level

        if engine == "nvb" and _nvbx is not None:
            # Native uwnvb DMPlexTransform. Each pass marks the cells whose current
            # size still exceeds the metric target and bisects them once (with the
            # bounded newest-vertex conforming closure); the transform is in-place so
            # the output stays co-partitioned with the parent and carries the
            # boundary/region labels forward automatically. A single bisection halves
            # the cell volume (h shrinks by 2^(1/dim)), so one isotropic-equivalent
            # max_levels is dim bisection passes.
            current_dm = base_finest             # base finest, current geometry
            n_gen = dim * max_levels
            for level in range(n_gen):
                centroids, cur_h, cs = cell_geometry(current_dm)
                if cur_h.size:
                    M = numpy.clip(eval_metric(centroids), 1e-30, None)
                    h_target = 1.0 / numpy.sqrt(M)
                    sel = numpy.where(cur_h > h_target)[0]
                    if node_budget is not None and sel.size > node_budget:
                        order = numpy.argsort(M[sel])[::-1]
                        sel = sel[order[:node_budget]]
                else:
                    sel = numpy.empty(0, dtype=int)   # rank owns no cells this level

                # Collective stop: refine while ANY rank still has cells to split
                # (a rank with none may still bisect via the cross-rank closure).
                # mpi4py allreduce defaults to MPI.SUM.
                if uw.mpi.comm.allreduce(int(sel.size)) == 0:
                    if verbose:
                        uw.pprint(0, f"[adapt] nvb pass {level}: nothing to refine")
                    break

                marked = [int(cs + j) for j in sel]
                markers_per_level.append(marked)
                d = current_dm.clone()
                d.createLabel("adapt")
                lab = d.getLabel("adapt")
                lab.setDefaultValue(0)
                for cidx in marked:
                    lab.setValue(cidx, DM_ADAPT_REFINE)
                current_dm = _nvbx.refine(d, "adapt")
                level_dms.append(current_dm)
                if verbose:
                    fs, fe = current_dm.getHeightStratum(0)
                    uw.pprint(0, f"[adapt] nvb pass {level}: marked {len(marked)} "
                                 f"-> {fe - fs} cells (rank-local)")
            if not level_dms:
                current_dm = base_finest.clone()
        elif engine == "nvb":
            # Serial cell-list engines: the slot-based NVBMesh in 2D (until
            # the native transform adopts the tagged rule — capstone stage
            # 1e) and the dimension-general TaggedBisectionMesh in 3D. The
            # refinement-edge state propagates parent→child across
            # generations, preserving the similarity-class (shape-regularity)
            # bound. A bisection halves the cell volume, so h shrinks by
            # 2^(1/dim) per generation and one isotropic-equivalent
            # ``max_levels`` is dim generations.
            from underworld3.utilities.nvb import NVBMesh, TaggedBisectionMesh
            _Engine = TaggedBisectionMesh if self.dim == 3 else NVBMesh
            carry = [(b.name, b.value) for b in self.boundaries
                     if b.name not in ("Null_Boundary", "All_Boundaries")]
            rcarry = ([(r.name, r.value) for r in self.regions]
                      if self.regions is not None else [])
            nvb = _Engine.from_dm(base_finest, boundaries=carry,
                                  regions=rcarry)
            n_gen = dim * max_levels
            for level in range(n_gen):
                centroids, cur_h, cids = nvb.centroids_h()
                M = numpy.clip(eval_metric(centroids), 1e-30, None)
                h_target = 1.0 / numpy.sqrt(M)
                sel = numpy.where(cur_h > h_target)[0]
                if sel.size == 0:
                    if verbose:
                        uw.pprint(0, f"[adapt] nvb gen {level}: nothing to refine")
                    break
                if node_budget is not None and sel.size > node_budget:
                    order = numpy.argsort(M[sel])[::-1]
                    sel = sel[order[:node_budget]]
                marked = [int(cids[j]) for j in sel]
                markers_per_level.append(marked)
                nvb.refine(set(marked))
                level_dms.append(nvb.to_dm(boundaries=carry, regions=rcarry,
                                           comm=self.dm.comm))
                if verbose:
                    uw.pprint(0, f"[adapt] nvb gen {level}: marked {len(marked)} "
                                 f"-> {len(nvb.cells)} cells")
            current_dm = level_dms[-1] if level_dms else base_finest.clone()
        else:
            current_dm = base_finest             # base finest, current geometry
            for level in range(max_levels):
                centroids, cur_h, cs = cell_geometry(current_dm)
                if cur_h.size == 0:
                    break

                # Metric M = 1/h_target² at the cell centroids (parent field).
                # A callable is evaluated directly on the centroids; a field/expr
                # goes through global_evaluate (parallel) or evaluate (serial).
                M = numpy.clip(eval_metric(centroids), 1e-30, None)
                h_target = 1.0 / numpy.sqrt(M)

                refine = numpy.where(cur_h > h_target)[0]
                if refine.size == 0:
                    if verbose:
                        uw.pprint(0, f"[adapt] level {level}: no cells need refinement")
                    break

                if node_budget is not None and refine.size > node_budget:
                    # keep the highest-metric (finest-demand) cells first
                    order = numpy.argsort(M[refine])[::-1]
                    refine = refine[order[:node_budget]]

                cell_ids = [int(cs + j) for j in refine]
                markers_per_level.append(cell_ids)
                if verbose:
                    uw.pprint(0, f"[adapt] level {level}: refining {len(cell_ids)} "
                                 f"of {ncells} cells")
                current_dm = custom_mg.sbr_refine(current_dm, cell_ids)
                level_dms.append(current_dm)

        if verbose:
            base_n = base_finest.getHeightStratum(0)
            fin_n = current_dm.getHeightStratum(0)
            uw.pprint(0, f"[adapt] base finest {base_n[1]-base_n[0]} -> "
                         f"child {fin_n[1]-fin_n[0]} cells "
                         f"({len(markers_per_level)} {engine} level(s))")

        # Wrap the refined finest as the child mesh (on-rank; no redistribute).
        child = Mesh(
            current_dm.clone(),
            simplex=self.dm.isSimplex(),
            coordinate_system_type=self.CoordinateSystem.coordinate_type,
            qdegree=self.qdegree,
            boundaries=self.boundaries,
            verbose=False,
        )

        # Lineage (parent/child DAG) and mesh-owned custom-P hierarchy.
        child.parent = self
        child._relationship_kind = "refinement"
        child.regions = self.regions
        child._parent_mesh_version = self._mesh_version
        # Markers per refinement level (in each level's cell numbering) — the
        # checkpoint-by-marker payload (design only; storage is a follow-up).
        child._adapt_markers = markers_per_level
        child._adapt_engine = engine
        # Mesh-owned custom-P geometric-MG tail. EVERY refinement level is its own
        # MG level (one custom-P transfer per refinement step), not a single
        # base-finest -> child jump: the tail is
        #   [base L0 … base finest]  +  [refine level 1 … refine level n-1]
        # and the solver appends its own mesh (the finest level = child). Each
        # intermediate level is wrapped here (transient, lives on the child); the
        # static base levels reuse the parent's cached wraps. (NVB snapshots and
        # SBR levels are both just coordinate sets — the custom-P transfers are
        # coordinate-based, so the hierarchy is engine-agnostic.)
        intermediate = [
            self._wrap_coarse_level(d) for d in level_dms[:-1]
        ]
        coarse_tail = self._coarse_level_meshes()
        if _moved:
            # the finest base level of the MG tail must carry the SAME moved
            # geometry the child was refined from; coarser levels keep their
            # original coordinates (custom-P transfers accept non-nested pairs)
            coarse_tail = coarse_tail[:-1] + [self._wrap_coarse_level(base_finest)]
        child._custom_mg_coarse_meshes = coarse_tail + intermediate
        child._custom_mg_builder = builder

        self._registered_children.add(child)
        return child

    def remesh(self, metric_field, verbose=False):
        r"""
        Re-mesh (regenerate) the discretization in place from a metric field.

        This is the **MMG / topology-changing** remesher: it regenerates the
        mesh in place (cells are created/destroyed and the partition may change),
        automatically transferring all attached MeshVariables, updating Surfaces,
        and marking Solvers for rebuild on their next solve() call.

        Contrast :meth:`adapt`, which performs *nested* skeleton-based refinement
        on top of the static base and **returns a refined child** (the mesh is
        not modified in place). ``remesh`` is the in-place, redistributing path;
        prefer ``adapt`` when you want a parent/child geometric-MG hierarchy.

        This method was formerly called ``adapt``; ``adapt`` now performs the
        nested SBR adapt-on-top.

        Parameters
        ----------
        metric_field : MeshVariable
            A scalar MeshVariable containing metric values (1/h² where h is
            target edge length). Larger values mean finer mesh (smaller elements).
            Use Surface.refinement_metric() to create this field from distance.
        verbose : bool, optional
            If True, print progress and statistics during adaptation.

        Notes
        -----
        The adaptation uses PETSc's mesh adaptation with MMG/pragmatic backend.

        **What happens automatically:**

        - MeshVariables are interpolated to the new mesh
        - Surfaces recompute their distance fields
        - Swarms are marked as stale (particle-element associations invalidated)
        - Solvers are marked for rebuild (happens lazily on next solve())

        Examples
        --------
        >>> # Define metric from fault distance
        >>> metric = uw.discretisation.MeshVariable("H", mesh, 1)
        >>> # Smaller H near fault, larger far away
        >>> metric.data[:, 0] = 0.01 + 0.09 * fault.distance_from(mesh.X.coords)
        >>> mesh.remesh(metric, verbose=True)
        >>> stokes.solve()  # Solver rebuilds automatically
        """
        import underworld3 as uw
        from underworld3 import adaptivity

        # Store old state for transfer
        old_dm = self.dm

        # Notify surfaces to mark their distance fields as stale
        # Surface distance variables are just regular MeshVariables with lazy
        # recomputation - they get reinitialized along with all other variables
        for surface_ref in list(self._registered_surfaces):
            surface = surface_ref() if callable(surface_ref) else surface_ref
            if surface is not None:
                if hasattr(surface, '_on_mesh_adapted'):
                    if verbose:
                        print(f"[{uw.mpi.rank}] Notifying surface '{surface.name}' (marking distance stale)...", flush=True)
                    surface._on_mesh_adapted(self)

        # Capture all user-supplied variables for reinitialization on the new mesh.
        # The metric field is included — it's a user-created variable that may
        # have external references and be reused in subsequent adaptation cycles.
        old_vars_data = {}
        for var_name, var in self._vars.items():
            if var is not None:
                old_vars_data[var_name] = var

        # Stack boundary labels for adaptation
        adaptivity._dm_stack_bcs(self.dm, self.boundaries, "CombinedBoundaries")

        # Create cell region label if regions exist — this tells MMG to
        # preserve the interface between regions during adaptation
        rgLabel_name = None
        if self.regions is not None:
            depth_label = self.dm.getLabel("depth")
            cell_is = depth_label.getStratumIS(self.dim)
            if cell_is:
                cells = cell_is.getIndices()
                self.dm.createLabel("_CellRegions_")
                rg = self.dm.getLabel("_CellRegions_")
                for region in self.regions:
                    lab = self.dm.getLabel(region.name)
                    if lab:
                        region_is = lab.getStratumIS(region.value)
                        if region_is:
                            region_cells = set(region_is.getIndices()) & set(cells)
                            for c in region_cells:
                                rg.setValue(c, region.value)
                rgLabel_name = "_CellRegions_"

        # Create the metric from the field
        hvec = metric_field._lvec
        metric_vec = self.dm.metricCreateIsotropic(hvec, metric_field.field_id)

        if verbose:
            n_nodes_old = self.dm.getChart()[1] - self.dm.getChart()[0]
            print(f"[{uw.mpi.rank}] Mesh adaptation starting (nodes: ~{n_nodes_old})...", flush=True)

        # Perform the actual mesh adaptation
        new_dm = self.dm.adaptMetric(
            metric_vec,
            bdLabel="CombinedBoundaries",
            rgLabel=rgLabel_name,
        )

        # Unstack boundary labels on the new dm
        adaptivity._dm_unstack_bcs(new_dm, self.boundaries, "CombinedBoundaries")

        # Reconstruct region labels from cell tags on the adapted mesh
        if rgLabel_name and self.regions is not None:
            rg_new = new_dm.getLabel(rgLabel_name)
            if rg_new:
                for region in self.regions:
                    new_dm.createLabel(region.name)
                    region_label = new_dm.getLabel(region.name)
                    region_is = rg_new.getStratumIS(region.value)
                    if region_is:
                        region_label.setStratumIS(region.value, region_is)

        if verbose:
            n_nodes_new = new_dm.getChart()[1] - new_dm.getChart()[0]
            print(f"[{uw.mpi.rank}] Mesh adapted (nodes: ~{n_nodes_new})", flush=True)

        # Create temporary mesh for interpolation
        # (We need a full Mesh to use mesh2mesh_meshVariable)
        temp_mesh = Mesh(
            new_dm,
            simplex=self.dm.isSimplex(),
            coordinate_system_type=self.CoordinateSystem.coordinate_type,
            qdegree=self.qdegree,
            refinement=None,
            refinement_callback=self.refinement_callback,
            boundaries=self.boundaries,
        )

        # Transfer variable data from old mesh to new mesh via evaluate.
        # The old variables are still on ``self`` (old DM). Evaluate each old
        # variable's symbol at the *new mesh's DOF coords for that variable's
        # degree/continuity*, not at the new mesh's vertex set. The vertex
        # set only matches degree-1 continuous variables; for degree>=2 or
        # discontinuous variables the DOF count differs and the resulting
        # array would not fit into ``new_var.data``.
        #
        # Cache target coords by (degree, continuous) so meshes with many
        # variables of the same basis don't pay for repeated cloned
        # CoordinateDMs inside ``_get_coords_for_basis``.
        transferred_data = {}
        target_coords_cache = {}

        for var_name, old_var in old_vars_data.items():
            try:
                if old_var._lvec is not None and old_var.data.size > 0:
                    if verbose:
                        print(f"[{uw.mpi.rank}] Transferring '{var_name}'...", flush=True)
                    basis_key = (old_var.degree, old_var.continuous)
                    if basis_key not in target_coords_cache:
                        target_coords_cache[basis_key] = (
                            temp_mesh._get_coords_for_basis(*basis_key)
                        )
                    target_coords = target_coords_cache[basis_key]
                    transferred_data[var_name] = uw.function.evaluate(
                        old_var.sym, target_coords
                    )
            except Exception as e:
                if verbose:
                    print(f"[{uw.mpi.rank}] Warning: transfer of '{var_name}' failed: {e}", flush=True)

        del temp_mesh

        # Now update this mesh's internal state
        with self._mesh_update_lock:
            # Update the DM
            self.dm = new_dm
            self.dm.setName(f"uw_{self.name}")

            # Update coordinates array with the shared coordinate-update
            # callback. The unified callback adds the teardown guard and the
            # canonical-array identity gate this site's inline copy had
            # silently dropped (READ-16).
            self._install_coords_array(verbose=verbose)

            # Increment mesh version (marks swarms as stale)
            self._mesh_version += 1
            self._topology_version += 1

            # Rebuild coordinate navigation
            self.nuke_coords_and_rebuild(verbose=False)

        # Destroy ALL old vectors upfront before reinitializing any variable.
        # This is critical because _setup_ds() iterates mesh._vars to backup/restore
        # data — if some variables still hold lvecs with stale field_ids from the
        # pre-adaptation DM, createSubDM will fail on the new DM.  (Fixes #48)
        for old_var in old_vars_data.values():
            self._destroy_variable_petsc_state(old_var)

        # Reinitialize MeshVariables on the new mesh
        # Note: Variables are reset to zero. Users should reinitialize with data.
        for var_name, old_var in old_vars_data.items():
            try:
                self._reinit_variable_on_new_dm(old_var)

                # Restore transferred data if available
                if var_name in transferred_data:
                    try:
                        data = transferred_data[var_name]
                        # evaluate returns (N, a, b) shaped array; pack to (N, ncomp)
                        data_flat = data.reshape(old_var.data.shape)
                        old_var.pack_raw_data_to_petsc(data_flat, sync=True)
                        if verbose:
                            print(f"[{uw.mpi.rank}] Variable '{var_name}' transferred to adapted mesh", flush=True)
                    except Exception as e2:
                        if verbose:
                            print(f"[{uw.mpi.rank}] Variable '{var_name}' reset (transfer failed: {e2})", flush=True)
                else:
                    if verbose:
                        print(f"[{uw.mpi.rank}] Variable '{var_name}' reset on adapted mesh", flush=True)
            except Exception as e:
                if verbose:
                    print(f"[{uw.mpi.rank}] Warning: Failed to reinitialize '{var_name}': {e}", flush=True)

        # Note: Surfaces were already notified at the start of adapt()
        # They will lazily recompute distance fields when accessed

        # Refresh the local cell-size field from the adapted geometry. The
        # generic variable transfer above re-interpolates it (meaningless for
        # a geometric size); re-fill it from the new mesh's per-cell radii so
        # the Nitsche penalty scales correctly after re-refinement.
        if getattr(self, "_cell_size_variable", None) is not None:
            try:
                self._assemble_cell_size(self._cell_size_variable)
            except Exception as e:
                # Sanctioned swallow (verbose-only report): the adaptation is
                # already committed; a failed size re-fill leaves the Nitsche
                # penalty scaling one geometry behind rather than losing the
                # adapted mesh. The next deform/adapt re-fills it.
                if verbose:
                    print(f"[{uw.mpi.rank}] Warning: cell-size refresh failed: {e}", flush=True)

        # Mark solvers for rebuild and clear geometry-keyed caches
        self._invalidate_caches_after_dm_change(reason="mesh_adaptation")

        # Re-extract registered submeshes from the re-meshed parent. Only true
        # subset submeshes (extract_region) can be re-filtered; SBR refinement
        # children (adapt) have a different lineage and are skipped.
        for submesh in list(self._registered_submeshes):
            if getattr(submesh, "_relationship_kind", "submesh") != "submesh":
                continue
            try:
                submesh._re_extract_from_parent(verbose=verbose)
            except Exception as e:
                # Sanctioned swallow (verbose-only report): one submesh
                # failing to re-extract must not abort the parent's completed
                # adaptation or the other submeshes' re-extraction; the failed
                # submesh keeps its pre-adapt DM and will loudly mismatch on
                # first parent-version check.
                if verbose:
                    print(f"[{uw.mpi.rank}] Warning: submesh re-extraction failed: {e}", flush=True)

        if verbose:
            print(f"[{uw.mpi.rank}] Mesh adaptation complete", flush=True)

        return


def _write_compat_groups(mesh, var, var_h5_path):
    """Write ``/vertex_fields/`` or ``/cell_fields/`` compatibility groups.

    Uses ``uw.function.write_vertices_to_viewer`` (PETSc interpolation +
    ViewerHDF5) for continuous variables, and
    ``uw.function.write_cell_field_to_viewer`` for cell/DG-0 variables.
    PETSc handles all parallel I/O natively.

    Vertex coordinates are also written to ``/vertex_fields/coordinates``
    for XDMF compatibility.

    Parameters
    ----------
    mesh : Mesh
        The parent mesh.
    var : MeshVariable
        The variable whose data has already been written to *var_h5_path*
        by ``var.write()`` (so ``var._gvec`` is up-to-date).
    var_h5_path : str
        Path to the HDF5 file (already contains ``/fields/<name>``).
    """
    import underworld3 as uw

    is_cell = (not var.continuous) or (var.degree == 0)
    group = "cell_fields" if is_cell else "vertex_fields"

    # Some PETSc versions (3.21+) write /vertex_fields/ or /cell_fields/
    # automatically during var.write().  Remove any pre-existing group so
    # that our compat writer can create it afresh (otherwise PETSc error 76
    # on duplicate dataset).
    import h5py

    if uw.mpi.rank == 0:
        with h5py.File(var_h5_path, "a") as f:
            if group in f:
                del f[group]
    uw.mpi.barrier()

    viewer = PETSc.ViewerHDF5().create(
        var_h5_path, "a", comm=PETSc.COMM_WORLD,
    )

    if is_cell:
        uw.function.write_cell_field_to_viewer(var, viewer)
    else:
        uw.function.write_vertices_to_viewer(var, viewer)
        uw.function.write_coordinates_to_viewer(mesh, viewer)

    viewer.destroy()


def checkpoint_xdmf(
    filename: str,
    meshUpdates: bool = True,
    meshVars: Optional[list] = [],
    swarmVars: Optional[list] = [],
    index: Optional[int] = 0,
):
    import h5py
    import os
    import warnings

    """Create xdmf file for checkpoints"""

    ## Identify the mesh file. Use the
    ## zeroth one if this option is turned off

    if not meshUpdates:
        mesh_filename = filename + ".mesh.00000.h5"
    else:
        mesh_filename = filename + f".mesh.{index:05}.h5"

    ## Obtain the mesh information

    h5 = h5py.File(mesh_filename, "r")
    if "viz" in h5 and "geometry" in h5["viz"]:
        geomPath = "viz/geometry"
        geom = h5["viz"]["geometry"]
    else:
        geomPath = "geometry"
        geom = h5["geometry"]

    if "viz" in h5 and "topology" in h5["viz"] and "cells" in h5["viz"]["topology"]:
        topoPath = "viz/topology"
        topo = h5["viz"]["topology"]
    elif "topology" in h5 and "cells" in h5["topology"]:
        topoPath = "topology"
        topo = h5["topology"]
    else:
        h5.close()
        raise RuntimeError(
            f"Cannot generate XDMF for {mesh_filename}: no direct cell "
            "connectivity dataset found at /viz/topology/cells."
        )

    vertices = geom["vertices"]
    numVertices = vertices.shape[0]
    spaceDim = vertices.shape[1]
    cells = topo["cells"]
    if len(cells.shape) != 2:
        h5.close()
        raise RuntimeError(
            f"Cannot generate XDMF for {mesh_filename}: {topoPath}/cells has "
            f"shape {cells.shape}. XDMF requires a 2D direct cell-to-vertex "
            "connectivity dataset."
        )
    numCells = cells.shape[0]
    numCorners = cells.shape[1]
    topology_precision = cells.dtype.itemsize

    if numCorners <= 1:
        h5.close()
        raise RuntimeError(
            f"Cannot generate XDMF for {mesh_filename}: {topoPath}/cells has "
            f"shape {cells.shape}. XDMF requires direct cell-to-vertex "
            "connectivity, not PETSc DMPlex internal topology."
        )

    if topoPath == "topology":
        warnings.warn(
            "Using raw '/topology/cells' for XDMF. This may not be Paraview-compatible. "
            "Expected '/viz/topology/cells'.",
            stacklevel=2,
        )

        cells_data = cells[...]
        c_min, c_max = cells_data.min(), cells_data.max()
        if c_min < 0 or c_max >= numVertices:
            warnings.warn(
                f"XDMF connectivity is invalid! cells max {c_max} >= "
                f"numVertices {numVertices} or min {c_min} < 0. ParaView will likely crash. "
                f"Ensure cell-to-vertex connectivity is written.",
                stacklevel=2,
            )

    h5.close()

    # We only use a subset of the possible cell types
    if spaceDim == 2:
        if numCorners == 3:
            topology_type = "Triangle"
        elif numCorners == 4:
            topology_type = "Quadrilateral"
        else:
            warnings.warn(f"Unexpected numCorners={numCorners} for 2D spaceDim. Expected 3 or 4.", stacklevel=2)
            topology_type = "Quadrilateral"
        geomType = "XY"
    else:
        if numCorners == 4:
            topology_type = "Tetrahedron"
        elif numCorners == 8:
            topology_type = "Hexahedron"
        else:
            warnings.warn(f"Unexpected numCorners={numCorners} for 3D spaceDim. Expected 4 or 8.", stacklevel=2)
            topology_type = "Hexahedron"
        geomType = "XYZ"

    ## Create the header

    header = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [
<!ENTITY MeshData "{os.path.basename(mesh_filename)}">
"""
    for var in meshVars:
        var_filename = filename + f".mesh.{var.clean_name}.{index:05}.h5"
        header += f"""
<!ENTITY {var.clean_name}_Data "{os.path.basename(var_filename)}">"""

    for var in swarmVars:
        var_filename = filename + f".proxy.{var.clean_name}.{index:05}.h5"
        header += f"""
<!ENTITY {var.clean_name}_Data "{os.path.basename(var_filename)}">"""

    header += """
]>"""

    xdmf_start = f"""
<Xdmf>
  <Domain Name="domain">
    <DataItem Name="cells"
              ItemType="Uniform"
              Format="HDF"
              NumberType="Int" Precision="{topology_precision}"
              Dimensions="{numCells} {numCorners}">
      &MeshData;:/{topoPath}/cells
    </DataItem>
    <DataItem Name="vertices"
              Format="HDF"
              Dimensions="{numVertices} {spaceDim}">
      &MeshData;:/{geomPath}/vertices
    </DataItem>
    <!-- ============================================================ -->
      <Grid Name="domain" GridType="Uniform">
        <Topology
           TopologyType="{topology_type}"
           NumberOfElements="{numCells}">
          <DataItem Reference="XML">
            /Xdmf/Domain/DataItem[@Name="cells"]
          </DataItem>
        </Topology>
        <Geometry GeometryType="{geomType}">
          <DataItem Reference="XML">
            /Xdmf/Domain/DataItem[@Name="vertices"]
          </DataItem>
        </Geometry>
"""

    ## The mesh Var attributes

    def get_field_info(h5_filename, mesh_var, center):
        """
        Return (num_items, num_components, dataset_path) for a mesh variable.
        Prefers vertex/cell compatibility groups, falls back to /fields layout.
        """
        compat_name = f"{mesh_var.clean_name}_{mesh_var.clean_name}"
        candidates = []

        if center == "Cell":
            candidates = [f"cell_fields/{compat_name}", f"fields/{mesh_var.clean_name}"]
        else:
            candidates = [f"vertex_fields/{compat_name}", f"fields/{mesh_var.clean_name}"]

        with h5py.File(h5_filename, "r") as f:
            for path in candidates:
                if path in f:
                    shp = f[path].shape
                    if len(shp) == 1:
                        return shp[0], 1, path
                    return shp[0], shp[1], path

        raise RuntimeError(
            f"Could not locate data for variable '{mesh_var.clean_name}' in {h5_filename}"
        )

    attributes = ""
    for var in meshVars:
        var_filename = filename + f".mesh.{var.clean_name}.{index:05}.h5"

        # Determine if data is stored on nodes (vertex_fields) or cells (cell_fields)
        if not getattr(var, "continuous") or getattr(var, "degree") == 0:
            center = "Cell"
        else:
            center = "Node"
        numItems, numComponents, dataset_path = get_field_info(var_filename, var, center)

        if center == "Node" and numItems != numVertices:
            warnings.warn(
                f"Attribute '{var.clean_name}' Center is 'Node' but numItems "
                f"({numItems}) != numVertices ({numVertices}).",
                stacklevel=2,
            )
        elif center == "Cell" and numItems != numCells:
            warnings.warn(
                f"Attribute '{var.clean_name}' Center is 'Cell' but numItems "
                f"({numItems}) != numCells ({numCells}).",
                stacklevel=2,
            )

        # Use variable type when available, but reflect actual stored component count.
        if hasattr(var, "vtype") and var.vtype in (
            uw.VarType.TENSOR,
            uw.VarType.SYM_TENSOR,
            uw.VarType.MATRIX,
        ):
            variable_type = "Tensor"
        elif numComponents == 1:
            variable_type = "Scalar"
        else:
            variable_type = "Vector"

        data_dimensions = f"{numItems}" if numComponents == 1 else f"{numItems} {numComponents}"
        var_attribute = f"""
        <Attribute
           Name="{var.clean_name}"
           Type="{variable_type}"
           Center="{center}">
          <DataItem
             DataType="Float" Precision="8"
             Dimensions="{data_dimensions}"
             Format="HDF">
            &{var.clean_name+"_Data"};:/{dataset_path}
          </DataItem>
        </Attribute>
        """
        attributes += var_attribute

    for var in swarmVars:
        var_filename = filename + f".proxy.{var.clean_name}.{index:05}.h5"
        if var.num_components == 1:
            variable_type = "Scalar"
        else:
            variable_type = "Vector"
        # We should add a tensor type here ...

        var_attribute = f"""
        <Attribute
           Name="{var.clean_name}"
           Type="{variable_type}"
           Center="Node">
          <DataItem ItemType="HyperSlab"
        	    Dimensions="1 {numVertices} {var.num_components}"
        	    Type="HyperSlab">
            <DataItem
               Dimensions="3 3"
               Format="XML">
              0 0 0
              1 1 1
              1 {numVertices} {var.num_components}
            </DataItem>
            <DataItem
               DataType="Float" Precision="8"
               Dimensions="1 {numVertices} {var.num_components}"
               Format="HDF">
              &{var.clean_name+"_Data"};:/vertex_fields/{var.clean_name+"_P"+str(var._meshVar.degree)}
            </DataItem>
          </DataItem>
        </Attribute>
    """
        attributes += var_attribute

    xdmf_end = f"""
    </Grid>
  </Domain>
</Xdmf>
    """

    xdmf_filename = filename + f".mesh.{index:05}.xdmf"
    with open(xdmf_filename, "w") as fp:
        fp.write(header)
        fp.write(xdmf_start)
        fp.write(attributes)
        fp.write(xdmf_end)

    return


def meshVariable_lookup_by_symbol(mesh, sympy_object):
    """Given a sympy object, scan the mesh variables in `mesh` to find the
    location (meshvariable, component in the data array) corresponding to the symbol
    or return None if not found
    """

    for meshvar in mesh.vars.values():
        if meshvar.sym == sympy_object:
            return meshvar, -1
        else:
            for comp, subvar in enumerate(meshvar.sym_1d):
                if subvar == sympy_object:
                    return meshvar, comp

    return None


def petsc_dm_find_labeled_points_local(
    dm, label_name, label_value, sectionIndex=False, verbose=False
):
    """Identify local points associated with "Label"

    dm -> expects a petscDM object
    label_name -> "String Name for Label"
    sectionIndex -> False: leave points as indexed by the relevant section on the dm
                    True: index into the local coordinate array

    NOTE: Assumes uniform element types
    """

    import numpy as np

    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    fStart, fEnd = dm.getDepthStratum(2)

    # print(f"Label: {label_name} / {label_value}")
    # print(f"points: {pStart}: {pEnd}")
    # print(f"edges : {eStart}: {eEnd}")
    # print(f"faces : {fStart}: {fEnd}")
    # print(f"", flush=True)

    label = dm.getLabel(label_name)
    if not label:
        print(f"{uw.mpi.rank} Label {label_name} is not present on the dm", flush=True)
        return np.array([0])

    pointIS = dm.getStratumIS("depth", 0)
    edgeIS = dm.getStratumIS("depth", 1)
    faceIS = dm.getStratumIS("depth", 2)

    point_indices = pointIS.getIndices()
    edge_indices = edgeIS.getIndices()
    face_indices = faceIS.getIndices()

    # _, iset_lab = label.convertToSection()
    iset_lab = label.getStratumIS(label_value)
    if not iset_lab:
        return None

    # We need to associate edges and faces with their point indices to
    # build a field representation

    IndicesP = np.intersect1d(iset_lab.getIndices(), pointIS.getIndices())
    IndicesE = np.intersect1d(iset_lab.getIndices(), edgeIS.getIndices())
    IndicesF = np.intersect1d(iset_lab.getIndices(), faceIS.getIndices())

    # print(f"Label {label_name}")
    # print(f"P -> {len(IndicesP)}, E->{len(IndicesE)}, F->{len(IndicesF)},")

    IndicesFe = np.empty((IndicesF.shape[0], dm.getConeSize(fStart)), dtype=int)
    for f in range(IndicesF.shape[0]):
        IndicesFe[f] = dm.getCone(IndicesF[f])

    IndicesFE = np.union1d(IndicesE, IndicesFe)

    # All faces are now recorded as edges

    IndicesFEP = np.empty((IndicesFE.shape[0], dm.getConeSize(eStart)), dtype=int)

    for e in range(IndicesFE.shape[0]):
        IndicesFEP[e] = dm.getCone(IndicesFE[e])

    # all faces / edges are now points

    if sectionIndex:
        Indices = np.union1d(IndicesP, IndicesFEP)
    else:
        Indices = np.union1d(IndicesP, IndicesFEP) - pStart

    return Indices
