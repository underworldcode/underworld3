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
def _gmsh_to_h5(
    filename,
    markVertices=False,
    useRegions=True,
    useMultipleTags=True,
):
    """Convert a Gmsh file to PETSc HDF5 without loading the resulting DM."""

    h5_filename = filename + ".h5"
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

            # Write aside and rename, so ``filename + ".h5"`` never names a
            # half-written file. The name comes from the mesh PARAMETERS, so a
            # second process building the same geometry in this directory picks
            # the same one and would otherwise read what this one is still
            # writing (issue #563).
            # Imported here, not at module scope: the meshing package imports
            # this module, so a top-level import would close a cycle.
            from underworld3.meshing._mesh_files import _scratch_name

            scratch = _scratch_name(h5_filename)
            viewer = PETSc.ViewerHDF5().create(str(scratch), "w", comm=PETSc.COMM_SELF)
            viewer(plex_0)
            viewer.destroy()
            os.replace(scratch, h5_filename)
    finally:
        # The gmsh import options are import-time scratch — meaningful only for
        # the createFromFile above. Clear the whole namespace so a value set by
        # the caller for THIS import (notably ``dm_plex_gmsh_spacedim`` for a
        # manifold-surface generator) cannot leak into the next gmsh import and
        # silently produce a wrong-dimension mesh (e.g. a later SphericalShell
        # read as 2-D). Runs on success or failure.
        _clear_gmsh_import_options()

    # The barrier ensures every rank sees the complete atomically-renamed file.
    uw.mpi.barrier()

    return filename + ".h5"


@timing.routine_timer_decorator
def _from_gmsh(
    filename,
    comm=None,
    markVertices=False,
    useRegions=True,
    useMultipleTags=True,
):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """

    ## NOTE: - this should be smart enough to serialise the msh conversion
    ## and then read back in parallel via h5.  This is currently done
    ## by every gmesh mesh

    comm = comm or PETSc.COMM_WORLD
    h5_filename = _gmsh_to_h5(
        filename,
        markVertices=markVertices,
        useRegions=useRegions,
        useMultipleTags=useMultipleTags,
    )

    return _from_plexh5(h5_filename, comm, return_sf=True)


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



def _compose_prolongations(fine, coarse):
    """The transfer ``fine @ coarse``, as COO triplets, in numpy alone.

    Composing two prolongations is a sparse matrix product, but not one that
    needs a sparse-matrix library: every row of a bisection prolongation holds
    one or two entries (an inherited vertex, or the average of two), so expanding
    each entry of ``fine`` through the matching rows of ``coarse`` and summing
    duplicates stays small and is the whole operation.

    ``fine`` maps the middle level to the fine one and ``coarse`` maps the coarse
    level to the middle, so the result maps coarse to fine.
    """
    f_rows, f_cols, f_vals = fine
    c_rows, c_cols, c_vals = coarse

    order = numpy.argsort(c_rows, kind="stable")
    cr, cc, cv = c_rows[order], c_cols[order], c_vals[order]

    start = numpy.searchsorted(cr, f_cols, side="left")
    stop = numpy.searchsorted(cr, f_cols, side="right")
    counts = stop - start
    if counts.sum() == 0:
        return (numpy.empty(0, dtype=numpy.int64),
                numpy.empty(0, dtype=numpy.int64), numpy.empty(0))

    # Gather every (fine entry, matching coarse entry) pair without a Python loop.
    total = int(counts.sum())
    offsets = numpy.repeat(numpy.cumsum(counts) - counts, counts)
    picks = numpy.repeat(start, counts) + (numpy.arange(total) - offsets)

    rows = numpy.repeat(f_rows, counts)
    cols = cc[picks]
    vals = numpy.repeat(f_vals, counts) * cv[picks]

    # A fine vertex can reach the same coarse vertex by more than one route, so
    # duplicates are summed rather than dropped — dropping them silently loses
    # part of the weight and the transfer stops being a partition of unity.
    key = rows * (int(cols.max()) + 1) + cols
    uniq, inverse = numpy.unique(key, return_inverse=True)
    summed = numpy.bincount(inverse, weights=vals, minlength=uniq.size)
    width = int(cols.max()) + 1
    return (uniq // width, uniq % width, summed)

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
        # The mesh-specific ANALYTIC restore (radial on an annulus/sphere, box faces on a
        # Cartesian box). Valid only while the boundary still has the shape the closure
        # was written for: it is captured at construction with the ORIGINAL radii/extent.
        # Once the geometry is deformed (a moving free surface) it is silently wrong — the
        # true boundary is no longer that surface — so `return_coords_to_bounds` switches
        # to the general facet-based restore. See the property below.
        self._analytic_return_coords_to_bounds = return_coords_to_bounds
        self._geometry_deformed = False
        self._bnd_restore_cache = None

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
                    # A checkpointed mesh normally carries its coordinate
                    # system; a missing entry means the metadata was lost
                    # (or the file predates it). The construction defaults
                    # to CARTESIAN (#397) — say so, because for an annulus
                    # or spherical checkpoint that label is wrong and would
                    # otherwise propagate silently into re-written files.
                    if coordinate_system_type is None and uw.mpi.rank == 0:
                        import warnings

                        warnings.warn(
                            f"{plex_or_meshfile} has no coordinate_system_type "
                            "metadata; defaulting to CARTESIAN. Pass "
                            "coordinate_system_type= explicitly if this mesh "
                            "is not Cartesian.",
                            stacklevel=2,
                        )

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

        Patches the user-supplied boundary enum with ``All_Boundaries``
        (value 1001, every exterior facet, populated by ``markBoundaryFaces``),
        records the boundary / region metadata attributes on the mesh, rebuilds
        named boundary labels for wrapped DMPlex imports that only expose
        stacked Gmsh label sets, and builds the stacked ``UW_Boundaries`` label
        used by the boundary-condition machinery.

        ``Null_Boundary`` (value 666, every VERTEX of the mesh) used to be
        added here as well. Nothing needs it: it marks no facet, so it
        integrates nothing, and its one functional consumer — a fake natural BC
        the solver manufactured to guarantee "a surface integral term on every
        process" — has been removed and measured to change no answer at any
        rank count. What it did do was mislead: any pass that reads *labelled
        points* to find material interfaces saw every vertex of the mesh
        labelled, which silently refused 1114 of 1114 mesh-repair candidates.
        A boundaries enum supplied by a caller may still declare it, and old
        checkpoints still carry the value, so the sentinel skips downstream
        remain.
        """
        ## Patch up the boundaries to include the additional
        ## definitions that we do / might need. Note: the
        ## extend_enum decorator will replace existing members with
        ## the new ones.

        if boundaries is None:

            class replacement_boundaries(Enum):
                All_Boundaries = 1001

            boundaries = replacement_boundaries
        else:

            @extend_enum([boundaries])
            class replacement_boundaries(Enum):
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

        # A mesh constructed without an explicit coordinate system (e.g.
        # loaded directly from a .msh file, or from an h5 checkpoint with no
        # coordinate metadata) is Cartesian — the default every uw.meshing
        # constructor passes. Leaving None here crashed mesh.write()'s
        # metadata block (issue #397).
        if coordinate_system_type is None:
            coordinate_system_type = CoordinateSystemType.CARTESIAN

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

        # A rank owning zero cells contributes the identity element of each
        # reduction rather than raising on an empty array (issue #405).
        def _reduce_min(arr):
            return _reduce(float(arr.min()) if arr.size else float("inf"),
                           "MIN")

        def _reduce_max(arr):
            return _reduce(float(arr.max()) if arr.size else float("-inf"),
                           "MAX")

        def _local_percentile(arr, pct):
            # Rank-local estimate (see the docstring); a rank with no cells
            # has no local distribution to take a percentile of.
            return float(np.percentile(arr, pct)) if arr.size else float("nan")

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

        # Choose the branch COLLECTIVELY. A rank owning zero cells collects no
        # triangles and would otherwise take the volume-only branch (three
        # reductions) while its populated peers took the simplex branch
        # (eleven) — mismatched collective counts, i.e. a hang (issue #405).
        # A starved rank abstains from the vote instead.
        has_cells = cEnd > cStart
        n_simplex_ranks = _reduce(
            int(has_cells and is_simplex2d and bool(tri_vertex_lists)), "SUM")
        n_populated_ranks = _reduce(int(has_cells), "SUM")
        is_simplex2d = (n_populated_ranks > 0
                        and n_simplex_ranks == n_populated_ranks)

        if not is_simplex2d:
            try:
                volume = np.abs(np.array(
                    [dm.computeCellGeometryFVM(cell_id)[0]
                     for cell_id in range(cStart, cEnd)]))
            except Exception:
                volume = np.array([1.0])
            # A rank that owns cells but cannot compute their geometry keeps
            # the unit-volume placeholder; a rank owning NO cells contributes
            # nothing at all, so the global cell count stays honest.
            if not volume.size and has_cells:
                volume = np.array([1.0])
            n_cells = _reduce(int(volume.size), "SUM")
            vol_min = _reduce_min(volume)
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

        # reshape(-1, 3) keeps the (0, 3) shape on a rank with no cells, where
        # np.asarray([]) would be 1-D and the column indexing below would fail.
        tri = np.asarray(tri_vertex_lists, dtype=np.int64).reshape(-1, 3)
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
        # rel_area only ever masks this rank's own cells, so the rank-local
        # mean is the right scale — and an empty rank has no cells to mask.
        rel_area = area / area.mean() if area.size else area

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
            q_min=_reduce_min(shape_q),
            q_mean=q_sum / max(n_cells, 1),
            q_p01=_local_percentile(shape_q, 1),
            q_p05=_local_percentile(shape_q, 5),
            n_q_lt_0p3=_reduce(int((shape_q < 0.3).sum()), "SUM"),
            n_q_lt_0p2=_reduce(int((shape_q < 0.2).sum()), "SUM"),
            angle_max_deg=_reduce_max(largest_angle),
            n_angle_gt_150=_reduce(int((largest_angle > 150).sum()), "SUM"),
            n_angle_gt_165=_reduce(int((largest_angle > 165).sum()), "SUM"),
            aspect_max=_reduce_max(aspect),
            aspect_p99=_local_percentile(aspect, 99),
            sizejump_max=float(size_jump.max()),
            sizejump_p99=_local_percentile(size_jump, 99),
            n_big_thin=_reduce(
                int(((rel_area > 2.0) & (aspect > 4.0)).sum()), "SUM"),
            vol_min_over_mean=(_reduce_min(area)
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
            from underworld3.utilities import custom_mg
            cc = numpy.asarray(self.parent._get_coords_for_basis(pv.degree, pv.continuous))
            fc = numpy.asarray(self._get_coords_for_basis(cv.degree, cv.continuous))
            if getattr(self, "_refine_dofs_coincide", True):
                from scipy.spatial import cKDTree
                # nested SBR: every coarse DOF coincides with a fine DOF (P1) or
                # sits on a fine element edge (P2) -> nearest fine node is exact.
                _, idx = cKDTree(fc).query(cc)
                out = numpy.asarray(cv.data)[idx].reshape(pv.data.shape)
            else:
                # A child that MOVED parent nodes — adding a conforming surface
                # snaps vertices onto it — has no coincident DOF to inject from.
                # Nearest-node still returns that vertex, so the query succeeds
                # and silently reports the field at the DISPLACED position, an
                # O(snap_frac x h) error that nothing downstream can see.
                # Interpolate instead, which is what the parallel path above does.
                P = custom_mg.barycentric_prolongation(fc, cc)
                out = (P @ numpy.asarray(cv.data)).reshape(pv.data.shape)

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

        # Geometry generation counter. First call (construction) -> 0; every
        # later call means the coordinates changed (deform / adapt /
        # re-extract), which invalidates factory-declared analytic boundary
        # normals (see the boundary_normals setter and canonical_normal).
        self._geometry_version = getattr(self, "_geometry_version", -1) + 1

        self.dm.clearDS()
        self.dm.createDS()

        if verbose:
            uw.pprint(f"PETScDS - (re) initialised")

        self._coord_array = {}
        # Cleared with _coord_array because both describe the same node layout:
        # the coordinates and which cell owns each of them. Topology only, so it
        # survives node motion — but it is invalidated by the same re-creation
        # of the DS that invalidates the coordinates.
        self._cell_node_array = {}

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

        Assembles the EXACT, outward, measure-weighted PETSc facet normals
        (``dm.computeCellGeometryFVM``) from ONLY this boundary's facets onto
        its P1 vertices. Because each boundary is assembled independently,
        a vertex shared by two boundaries (a sharp corner) is NOT averaged
        across the discontinuity — each boundary keeps its own normal. On a
        smooth boundary (e.g. a free surface) the result is the smooth
        deformed normal. Cached per boundary; rebuilt lazily after a deform.

        COLLECTIVE. The per-vertex sum runs over ALL the facets meeting the
        vertex, which in parallel are split between ranks, so it is completed
        through the variable's own local↔global scatter before normalising
        (#564). Every rank must call this, including one that owns no part of
        the boundary.

        .. note:: **The 3-D answer changed at #564, in SERIAL as well as in
           parallel.** Facet contributions used to be routed to "the DOFs
           nearest the facet centroid" by a kd-tree query. On a TETRAHEDRAL
           boundary the three DOFs nearest a face centroid are not always that
           face's own three vertices, so the query silently picked up a
           neighbour and the assembled normal was wrong — by up to **0.103 in
           the unit normal (≈5.9°) on ``SphericalShell(0.55, 1.0, cs=0.35)``,
           at np=1, on a UNIFORM mesh**. Rows now come from the variable's own
           section, which is exact: measured against an independent
           global-facet-sum oracle, 1.9e-16 (Upper) and 2.2e-16 (Lower) where the
           old route was 4.7e-02 and 1.03e-01.

           2-D is unaffected (annulus 1.1e-16 old and new, box bit-identical) —
           on an edge the two nearest DOFs to the midpoint are always its own two
           vertices. So any **3-D** result that used the default ``normal=None``
           on a curved boundary moves, and moves toward the right answer. Guarded
           by ``tests/parallel/test_1069_boundary_normal_parallel.py``, which runs
           at np=1 as well as np>1 precisely so this path is covered.

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

    def _boundary_facets(self, name):
        """This rank's facet (height-1) points carrying boundary label ``name``.

        Uses the DM label named after the boundary with the stratum keyed by
        the boundary's value (the same access the BC code uses). Returns an
        empty list when this rank owns no facets of the boundary.
        """
        dm = self.dm
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
                # Sanctioned: no value IS on this rank means no facets here —
                # fall through to the empty list.
                live = set()
            if int(bvalue) in live:
                pis = label.getStratumIS(bvalue)
                if pis is not None and pis.getSize():
                    fS, fE = dm.getHeightStratum(1)
                    for p in pis.getIndices():
                        if fS <= int(p) < fE:
                            face_pts.append(int(p))
        return face_pts

    def _assemble_boundary_normal(self, var, name):
        """Fill ``var`` with the measure-weighted outward facet normal assembled
        from the faces of boundary ``name`` only (see :meth:`boundary_normal`).

        The nodal normal is ``Σ_f |f| n̂_f`` over EVERY facet of this boundary that
        meets the node, normalised at the end. In parallel that sum has to be
        completed across ranks before the normalise: a boundary facet is labelled
        on exactly one rank (measured — see the note below), so a node on a
        partition seam sees only SOME of its facets locally and normalising a
        partial stencil gives it a rotated normal. That is #564: on an
        ``Annulus(cellSize=0.12)`` the Upper arc's worst nodal normal was 3.0e-10
        from the exact radial one in serial and 5.8e-02 (3.3 degrees) at np=2,3,4 —
        exactly the error of taking one facet's normal instead of the average of
        the two — and it moved a constrained free-slip answer by 3.4 %.

        COLLECTIVE. The reduction runs on every rank, including one that owns no
        facet of this boundary (the #405 lesson: a rank-local early return here
        deadlocks the ranks that do own facets).

        Why a plain ADD is exact, with no de-duplication: no boundary facet is
        labelled on two ranks and none is labelled away from its owner. Measured
        on the annulus and the spherical shell at np=2,3,4 for BOTH label sources
        (the per-boundary label this uses and the consolidated ``UW_Boundaries``);
        the guard that keeps it true is
        ``tests/parallel/test_1069_boundary_normal_parallel.py``.

        FAILURE IS COLLECTIVE AND LOUD. A rank that cannot complete its facet walk
        raises :class:`RuntimeError` on EVERY rank, not just its own. Two failure
        modes are being avoided, and both were measured on the first version of
        this routine:

        * swallowing the failure rank-locally and carrying on gives that rank's
          OWNED boundary DOFs a ZERO normal — so the constraint direction over
          that part of the boundary is the zero vector, with a converged solve
          and no message. That is strictly worse than the 3.3-degree error this
          routine exists to remove, and indistinguishable from success;
        * raising rank-locally takes that rank out of the sub-DM collectives
          below and HANGS the others (measured: rank 1 returns, rank 0 blocks,
          killed by the launcher timeout).

        So the flag is agreed with an all-reduce first, every rank then takes the
        same branch, and every rank raises. Callers that swallow the exception
        (:meth:`deform`) are therefore safe by construction: what reaches them is
        already symmetric.
        """
        from underworld3.utilities.facet_normals import facet_measure_and_normal

        cdim = self.cdim
        dm = self.dm
        comm = dm.comm.tompi4py()
        failed = 0
        detail = ""
        ncomp = None
        accum = None

        # Rank-local set-up. Guarded like the facet walk below and for the same
        # reason: `dm.createSubDM` is COLLECTIVE, so a rank must not leave before
        # reaching it.
        try:
            ncomp = var.num_components
            # One node per DMPlex point is assumed by the `offset // ncomp` row
            # arithmetic below. True for the degree-1 variable `boundary_normal`
            # builds, but that path adopts a pre-existing `_n_bd_<name>` variable
            # if one is already registered (a checkpoint restore, or user code),
            # and a higher-degree space puts several nodes on one point — a P3
            # edge carries two in 2-D — which would land both on the first row and
            # leave the second at zero. Refuse rather than silently half-fill.
            if var.degree != 1:
                raise RuntimeError(
                    f"boundary_normal needs a degree-1 field; '{var.clean_name}' is "
                    f"degree {var.degree}. A higher-degree space carries several "
                    f"nodes per DMPlex point and this assembly writes one row per "
                    f"point.")
            # Dense over this rank's LOCAL DOFs (ghosts included), so a node whose
            # labelled facets all live on a neighbour needs no special enumeration —
            # it is simply a row that stays zero until the reduction fills it.
            accum = numpy.zeros_like(numpy.asarray(var.data))
        except Exception as exc:
            failed, detail = 1, f"{type(exc).__name__}: {exc}"

        # DOF rows come from the variable's OWN section on its sub-DM: exact on
        # every rank, and the same section the reduction below scatters through.
        # (This used to be a kd-tree lookup of the DOFs nearest the facet
        # centroid. That is a heuristic, and not only on a graded mesh: on a
        # TETRAHEDRAL boundary the three DOFs nearest a face centroid are not
        # always that face's own three vertices, so it mis-assigned on a uniform
        # spherical shell in SERIAL — see :meth:`boundary_normal`. It also cannot
        # be made to agree across ranks, because each rank's tree is built from
        # its own local coordinates.)
        indexset, subdm = dm.createSubDM(var.field_id)
        try:
            try:
                ssec = subdm.getLocalSection()
                for f in self._boundary_facets(name):
                    # Orientation needs the facet's OWN support cell, and only an
                    # exterior facet has exactly one. An internal boundary's facets
                    # have two and support[0] is arbitrary, so neighbouring facets
                    # of the same surface could be oriented oppositely and CANCEL
                    # in the sum — those facets are skipped, which is why a normal
                    # requested for an INTERNAL boundary comes back zero.
                    # (rotated_bc keeps the raw PETSc normal there instead;
                    # neither is a supported use of this routine today.)
                    measure, nrm, exterior = facet_measure_and_normal(dm, f)
                    if not exterior:
                        continue
                    for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
                        if ssec.getDof(q) <= 0:
                            continue
                        accum[ssec.getOffset(q) // ncomp] += measure * nrm[:cdim]
            except Exception as exc:
                if not failed:
                    failed, detail = 1, f"{type(exc).__name__}: {exc}"

            # Agree on the outcome BEFORE the reduction, so every rank takes the
            # same branch. Skipping the reduction on a failure is what keeps the
            # raise below from being reached by only some ranks.
            if comm.size > 1:
                failed = comm.allreduce(failed)
            if not failed:
                accum = self._sum_local_dofs_across_ranks(subdm, accum) \
                    if comm.size > 1 else accum
        finally:
            indexset.destroy()
            subdm.destroy()

        if failed:
            reports = [d for d in (comm.allgather(detail) if comm.size > 1
                                   else [detail]) if d]
            raise RuntimeError(
                f"boundary normal assembly for {name!r} failed on "
                f"{failed} of {comm.size} rank(s): {'; '.join(reports[:4])}")

        mag = numpy.sqrt(numpy.sum(accum ** 2, axis=1))
        nonzero = mag > 1.0e-30
        accum[nonzero] /= mag[nonzero, numpy.newaxis]
        var.data[...] = accum

    def _sum_local_dofs_across_ranks(self, subdm, values):
        """Complete a per-DOF sum across ranks: this rank's own contributions in,
        the sum over EVERY rank's contributions out — the same value on every rank
        that holds the DOF.

        ``values`` is dense over ``subdm``'s LOCAL DOFs, shaped ``(ndof, ncomp)``
        exactly like the variable's ``.data``. The sum rides the sub-DM's own
        local↔global scatter: ADD into the global vector accumulates every ghost
        copy onto the owner, and scattering back hands every rank the identical
        total. Work vectors are created here rather than borrowed from the
        variable, whose global vec is built lazily.

        COLLECTIVE on the DM's communicator.

        There is deliberately NO "the round trip lost this DOF, keep the local
        value" fallback. The question such a fallback wants to ask is "was this
        DOF constrained out of the global vector?"; the only thing it can cheaply
        test is "did it come back all-zero?", and those two differ on exactly the
        input where it would matter — a node whose GLOBAL contributions cancel
        (opposed facets on a degenerate or zero-thickness boundary) would have its
        rank-local PARTIAL value restored, re-introducing #564 on the one case the
        reduction exists to get right. The mesh DM carries no essential-BC
        constraints (those live on the solver DM), so nothing is lost today; if
        that ever changes, this should fail loudly rather than guess, and
        ``ssec.getConstraintDof(q)`` is the predicate to use.
        """
        ncomp = values.shape[1]
        lvec = subdm.createLocalVector()
        gvec = subdm.createGlobalVector()
        try:
            lvec.array[...] = values.reshape(-1)
            gvec.set(0.0)
            subdm.localToGlobal(lvec, gvec, addv=PETSc.InsertMode.ADD_VALUES)
            subdm.globalToLocal(gvec, lvec, addv=PETSc.InsertMode.INSERT_VALUES)
            summed = numpy.array(lvec.array, dtype=float).reshape(-1, ncomp)
        finally:
            lvec.destroy()
            gvec.destroy()
        return summed

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
        # TODO(BUG): this field is PARTITION-DEPENDENT, and so therefore is the
        # Nitsche penalty gamma*mu/h that consumes it (local_h=True, the default).
        # Not the indexing here — the values. `_get_mesh_sizes` measures a cell by
        # the distance from its vertices to the NEAREST CENTROID in a kd-tree built
        # from THIS RANK's centroids, so near a partition seam the nearest centroid
        # may simply be absent. Measured on Annulus(cellSize=0.12): the field's sum
        # is 26.0822 at np=1, 26.1211 at np=2 and 26.1386 at np=4, and its max moves
        # at np=4. End to end that is 6.6e-03 in the velocity of a Nitsche free-slip
        # annulus and it does NOT shrink with solver tolerance.
        # This is a DIFFERENT defect from the boundary normal fixed for #564 (which
        # is now clean: the same solve with local_h=False agrees to 3.6e-10 at
        # np=1..4). It is the local h that is left, and it also reaches every other
        # consumer of `cell_size()`. Not fixed here because `_get_mesh_sizes` also
        # feeds `get_min_radius`, the adaptivity metrics and the free-surface
        # relaxation, and it needs its own benchmarking.
        # Guard/measurement: tests/parallel/test_1069_boundary_normal_parallel.py
        # (_nitsche_annulus_diagnostics docstring records the numbers).
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
        """Deprecated — use :attr:`Gamma` in integrands and BCs, or
        :meth:`boundary_normal` for a per-boundary P1 normal field.

        This global field point-evaluates :attr:`Gamma` at every mesh
        vertex, but the underlying ``petsc_n`` only exists inside
        surface-integral kernels; off-kernel evaluation falls back to a
        coordinate-based direction and, on internal boundaries, averages
        oppositely-oriented facet normals into sub-unit vectors. Retained
        unchanged for back-compat (mesh-smoothing internals).

        Automatically updated when the mesh deforms.
        """
        if not hasattr(self, '_projected_normals') or self._projected_normals is None:
            self._update_projected_normals()
        return self._projected_normals.sym

    @property
    def boundary_normals(self):
        """Declared analytic boundary normals (Enum or mapping), or ``None``.

        Assigning stamps the declaration against the mesh's current
        geometry: any later coordinate change (``deform``, ``adapt``,
        direct coordinate writes) marks it stale, and
        :meth:`canonical_normal` then refuses to serve it. Re-assign after
        a deformation to re-declare normals that are valid for the new
        geometry (e.g. a radial normal after a radius-preserving remesh).
        """
        return getattr(self, "_boundary_normals", None)

    @boundary_normals.setter
    def boundary_normals(self, value):
        self._boundary_normals = value
        self._boundary_normals_geometry_version = getattr(
            self, "_geometry_version", 0)

    def canonical_normal(self, boundary_name):
        r"""Analytic outward-pointing normal for a boundary, or ``None``
        if no analytic normal was declared for that boundary.

        Sourced from the mesh factory's ``boundary_normals`` Enum: for
        axis-aligned box boundaries this is a constant sympy Matrix, for
        annulus / spherical-shell radial boundaries it is the analytic
        radial unit vector, and so on.

        The primary caller is code that needs a **partition-safe** normal
        on an *internal* boundary — see :issue:`327`. On an internal
        boundary at a partition seam, PETSc's per-quadrature ``petsc_n[]``
        (surfacing as :attr:`Gamma` / :attr:`Gamma_N`) is derived from
        ``support[0]`` of the DMPlex facet closure, which is
        partition-dependent; different ranks disagree on which cell is
        "support[0]" for the one seam facet, and the outward normal of
        that facet flips sign. A signed integral of ``Gamma[k]`` is then
        wrong by O(seam-facets / total-facets). The analytic normal
        returned here is partition-independent and does not touch
        ``petsc_n``, so it sidesteps the defect entirely for the mesh
        classes that know their internal-boundary geometry
        (:class:`BoxInternalBoundary`, :class:`AnnulusInternalBoundary`,
        :class:`SphericalShellInternalBoundary`).

        Parameters
        ----------
        boundary_name : str
            Name of the boundary label to look up (case-sensitive; must
            match one of :attr:`boundaries`).

        Returns
        -------
        sympy.Matrix or None
            Row matrix of length :attr:`cdim` giving the outward-pointing
            normal, or ``None`` if this mesh factory did not declare
            an analytic normal for ``boundary_name``.

        Raises
        ------
        RuntimeError
            If the mesh coordinates have changed since the normals were
            declared (``deform`` / ``adapt``): the declaration describes
            the original geometry and may no longer match the deformed
            surface. Re-assign :attr:`boundary_normals` to re-declare
            normals valid for the current geometry.

        See Also
        --------
        Gamma : the raw per-quadrature normal — use for external
            boundaries; may be partition-dependent on internal seams.
        Gamma_N : normalised :attr:`Gamma`.
        Gamma_P1 : projected P1 normals, useful for curved external
            boundaries.
        """
        bn = self.boundary_normals
        if bn is None:
            return None
        declared_at = getattr(self, "_boundary_normals_geometry_version", 0)
        if declared_at != getattr(self, "_geometry_version", 0):
            raise RuntimeError(
                f"The declared analytic boundary normals for this mesh "
                f"describe its original (factory) geometry, but the mesh "
                f"coordinates have changed since (deform / adapt). The "
                f"declaration for '{boundary_name}' may no longer match the "
                f"deformed surface. If the normals are still valid for the "
                f"new geometry (e.g. a radial normal after a "
                f"radius-preserving remesh), re-declare them:\n"
                f"    mesh.boundary_normals = mesh.boundary_normals\n"
                f"or assign a new Enum / dict of normal expressions. "
                f"Otherwise use an explicit normal expression in place of "
                f"mesh.Gamma / canonical_normal on this boundary."
            )
        try:
            member = bn[boundary_name]
        except (KeyError, AttributeError):
            return None
        value = getattr(member, "value", member)
        return value

    def _boundary_is_internal(self, boundary_name):
        """True if boundary ``boundary_name`` is an internal surface — its
        facets have a cell on BOTH sides (support size 2) — rather than part
        of the mesh exterior (support size 1).

        Collective on first call per boundary (the local answer is
        MAX-reduced so every rank agrees even when it owns no facets of the
        boundary); cached thereafter.
        """
        import underworld3 as uw
        from mpi4py import MPI

        if not hasattr(self, "_internal_boundary_cache") or \
                self._internal_boundary_cache is None:
            self._internal_boundary_cache = {}
        cached = self._internal_boundary_cache.get(boundary_name)
        if cached is not None:
            return cached

        dm = self.dm
        local = 0
        for f in self._boundary_facets(boundary_name):
            if dm.getSupportSize(f) == 2:
                local = 1
                break
        result = bool(uw.mpi.comm.allreduce(local, op=MPI.MAX))
        self._internal_boundary_cache[boundary_name] = result
        return result

    def _resolve_boundary_normals(self, fn, boundary_name):
        r"""Return ``fn`` with :attr:`Gamma` resolved for ``boundary_name``.

        ``mesh.Gamma`` is the single user-facing boundary-normal symbol. On
        an EXTERNAL boundary its components compile directly to PETSc's
        exact per-quadrature outward unit normal (``petsc_n[]``) and ``fn``
        is returned unchanged. On an INTERNAL boundary ``petsc_n[]`` is
        orientation-ambiguous — the facet has two support cells and the sign
        follows the partition-dependent ``support[0]`` — so a signed integral
        of ``Gamma`` components is partition-dependent (issue #327). Here the
        ``Gamma`` components are substituted with the mesh factory's declared
        analytic normal (:meth:`canonical_normal`), which is
        partition-independent by construction.

        Called by every boundary-integrand consumer that knows its boundary
        (``uw.maths.BdIntegral``, natural boundary conditions), so users
        write ``mesh.Gamma`` everywhere and never choose an implementation.

        Raises
        ------
        RuntimeError
            If ``boundary_name`` is internal and the mesh declares no
            analytic normal for it. There is no orientation convention for
            an arbitrary internal surface until the surface itself is
            oriented (PETSc ``DMPlexOrientLabel``, open-surface support
            pending upstream), so failing loudly beats integrating a
            partition-dependent sign.
        """
        import underworld3 as uw

        gamma_syms = tuple(self._Gamma.base_scalars()[0:self.cdim])
        expr = uw.function.expressions.unwrap(
            fn, keep_constants=False, return_self=False)
        if not isinstance(expr, sympy.MatrixBase):
            expr = sympy.sympify(expr)
        free = expr.free_symbols
        if not any(g in free for g in gamma_syms):
            return fn
        if not self._boundary_is_internal(boundary_name):
            return fn

        normal = self.canonical_normal(boundary_name)
        if normal is None:
            raise RuntimeError(
                f"The integrand references mesh.Gamma on the internal "
                f"boundary '{boundary_name}', but this mesh declares no "
                f"analytic normal for it. On an internal surface the "
                f"discrete facet normal has no well-defined orientation "
                f"(issue #327). Declare one in the mesh factory's "
                f"'boundary_normals' Enum, or use an explicit normal "
                f"expression in place of mesh.Gamma."
            )
        subs = {g: normal[i] for i, g in enumerate(gamma_syms)}
        return expr.xreplace(subs)

    # ===================================================================
    #  Bounding surfaces — per-surface tangent-slip + restore.
    #  See docs/developer/design/boundary-slip-strategy.md. SEPARATE from
    #  self.boundaries (the persisted gmsh/DMPlex labelling, untouched).
    # ===================================================================
    @property
    def return_coords_to_bounds(self):
        r"""Callable ``coords -> coords`` returning out-of-domain points to the mesh.

        Used by the semi-Lagrangian trace-back (``systems.ddt``) and by swarm advection
        to pull a departure point / particle that has left the domain back inside.

        Two implementations, selected by the state of the geometry:

        * **analytic** (default) — the closure installed by the mesh constructor
          (radial on an annulus/sphere, face clamps on a box). Cheap and exact **while
          the boundary keeps the shape it was written for**.
        * **general facet restore** (once :meth:`deform` has moved the geometry) — the
          nearest point on the mesh's CURRENT boundary facets, with an outward-normal
          side test (:meth:`_facet_return_coords_to_bounds`).

        The switch matters for a moving free surface: the analytic closure is captured at
        construction with the ORIGINAL radii/extent, so on a deformed mesh it tests the
        wrong surface — where the surface has moved inward a point beyond the true
        boundary is not detected as outside, is never restored, and falls through to the
        evaluator's RBF/Shepard fallback, which averages distant DOFs (hot material
        appearing in a cold boundary layer). The general restore follows the deformed
        boundary, so it stays correct.

        Assigning to this attribute overrides both (the setter replaces the analytic
        closure and is honoured until the geometry deforms).
        """
        if getattr(self, "_geometry_deformed", False):
            return self._facet_return_coords_to_bounds
        return self._analytic_return_coords_to_bounds

    @return_coords_to_bounds.setter
    def return_coords_to_bounds(self, fn):
        self._analytic_return_coords_to_bounds = fn

    def _boundary_facet_geometry(self):
        """Boundary facets of the CURRENT geometry as ``(facet_pts, outward_normals)``,
        cached against the coordinate-array version so it refreshes after a deform."""
        from underworld3.meshing.smoothing.graph import _boundary_facets

        coords = numpy.ascontiguousarray(self.data)
        stamp = (coords.shape, float(coords.sum()))
        cache = getattr(self, "_bnd_restore_cache", None)
        if cache is not None and cache[0] == stamp:
            return cache[1], cache[2]

        cdim = self.cdim
        facets, opp = _boundary_facets(self, cdim)
        if facets is None:                      # non-simplicial: no general restore
            self._bnd_restore_cache = (stamp, None, None)
            return None, None
        fpts = coords[facets]                   # (nf, k, cdim)
        if cdim == 2:
            e = fpts[:, 1] - fpts[:, 0]
            n = numpy.column_stack([e[:, 1], -e[:, 0]])
        else:
            n = numpy.cross(fpts[:, 1] - fpts[:, 0], fpts[:, 2] - fpts[:, 0])
        n = n / (numpy.linalg.norm(n, axis=1, keepdims=True) + 1e-300)
        # orient outward: away from the opposite cell vertex
        inward = coords[opp] - fpts[:, 0]
        flip = numpy.einsum("ij,ij->i", n, inward) > 0.0
        n[flip] *= -1.0
        self._bnd_restore_cache = (stamp, fpts, n)
        return fpts, n

    def _facet_return_coords_to_bounds(self, coords):
        """General restore: snap points lying OUTSIDE the current boundary to just inside
        the nearest boundary facet. Interior points are returned untouched (the
        outward-normal side test), so this is safe to apply to every trace-back foot."""
        from underworld3.meshing.smoothing.graph import (
            _nearest_on_facets_2d, _nearest_on_facets_3d)

        fpts, nrm = self._boundary_facet_geometry()
        if fpts is None:                        # fall back to the analytic closure
            fn = self._analytic_return_coords_to_bounds
            return fn(coords) if fn is not None else coords

        pts = numpy.ascontiguousarray(numpy.asarray(coords, dtype=float))
        cdim = self.cdim
        if cdim == 2:
            closest = _nearest_on_facets_2d(pts, fpts)
        else:
            closest = _nearest_on_facets_3d(pts, fpts)
        # side test against the normal of the facet owning `closest` (nearest centroid)
        tree = uw.kdtree.KDTree(numpy.ascontiguousarray(fpts.mean(axis=1)))
        owner = numpy.asarray(tree.query(numpy.ascontiguousarray(closest), 1)[1]).flatten()
        nvec = nrm[owner]
        outside = numpy.einsum("ij,ij->i", pts - closest, nvec) > 0.0
        if numpy.any(outside):
            eps = 1.0e-3 * float(self.get_min_radius())
            pts[outside] = closest[outside] - eps * nvec[outside]
        return pts

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
        # _pinned_mask closes every tagged point (edge in 2D, face in 3D)
        # down to its vertices, so face-only 3D labels classify correctly.
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
        # INTERIOR surfaces (embedded interfaces, e.g. the Internal circle/
        # sphere of the *InternalBoundary meshes) are never slip-eligible:
        # interface motion is physics-owned (deform / free-surface
        # machinery), so their nodes fall through to the pinned default —
        # exactly as when no surface was registered. Their registration
        # exists for adapt()'s refinement snapping.
        surf = {lab: s for lab, s in dict(self.bounding_surfaces).items()
                if not getattr(s, "interior", False)}
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

        # The geometry is about to move, so any ANALYTIC return_coords_to_bounds closure
        # (captured with the original radii/extent) no longer describes the boundary:
        # switch to the general facet restore, which follows the deformed surface. Also
        # release any registered analytic BoundingSurface on this mesh for the same
        # reason — a rigid `radial`/`plane` surface would keep restoring to the old shape.
        self._geometry_deformed = True
        self._bnd_restore_cache = None
        for _surf in getattr(self, "_bounding_surfaces", {}).values():
            if getattr(_surf, "kind", None) in ("radial", "plane") and hasattr(_surf, "release"):
                _surf.release()

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
            _bn_comm = self.dm.comm.tompi4py()
            for _nm, _var in list(self._boundary_normal_vars.items()):
                # The outcome is decided COLLECTIVELY, not per rank. The callee
                # already all-reduces its own failure flag and raises on every rank
                # or none, so this is belt and braces for anything raised OUTSIDE
                # its guarded region — but it is what makes "every rank takes the
                # same branch" a property of this loop rather than an inherited
                # assumption. The all-reduce is reached on the exception path too;
                # that is the whole point.
                _bn_failed, _bn_exc = 0, None
                try:
                    self._assemble_boundary_normal(_var, _nm)
                except Exception as _e:
                    _bn_failed, _bn_exc = 1, _e
                if _bn_comm.size > 1:
                    _bn_failed = _bn_comm.allreduce(_bn_failed)
                if _bn_failed:
                    _exc = _bn_exc if _bn_exc is not None else RuntimeError(
                        "failed on another rank")
                    # Sanctioned swallow: a normal refresh can fail for a
                    # boundary whose label vanished from the current DM
                    # (e.g. after region extraction); the deform itself is
                    # complete and must not be rolled back for one BC aid.
                    # Consequence of skipping: that Nitsche/penalty BC
                    # keeps its setup-time normal until next refresh — which is
                    # why it is a WARNING and not silence. A silent skip here
                    # leaves a stale constraint direction on a moved boundary.
                    #
                    # SAFE TO SWALLOW because _assemble_boundary_normal is
                    # COLLECTIVE (it completes the facet sum across ranks, #564)
                    # and its failures are collective too: it all-reduces its own
                    # failure flag and raises on EVERY rank or none. So what
                    # arrives here is already symmetric and every rank takes this
                    # branch together. Do not weaken that contract — a rank-local
                    # raise out of that routine hangs the job here rather than
                    # failing it. The dict is built by a collective accessor, so
                    # every rank iterates the same boundaries in the same order.
                    uw.mpi.pprint(
                        f"[mesh.deform] WARNING: could not refresh the boundary "
                        f"normal for {_nm!r} ({type(_exc).__name__}: {_exc}); BCs "
                        f"that captured it keep their pre-deform direction.")
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
        r"""Deprecated alias — use :attr:`Gamma`.

        Retained for back-compatibility: returns ``Gamma / |Gamma|``, which
        is numerically identical to :attr:`Gamma` (the quadrature-point
        normal is already unit length).

        Returns
        -------
        sympy.Matrix
            Row matrix of normalised boundary normal components.
        """
        G = self.Gamma
        return G / sympy.sqrt(G.dot(G))

    @property
    def Gamma(self) -> sympy.Matrix:
        r"""The boundary unit normal, as a symbolic row matrix.

        This is the single user-facing normal symbol: use it in boundary
        integrands (:class:`uw.maths.BdIntegral`) and natural boundary
        conditions on any boundary, external or internal.

        The consumer that compiles the integrand resolves the symbol per
        boundary: on an external boundary the components map to PETSc's
        exact per-quadrature outward unit normal (``petsc_n[]``); on an
        internal boundary — where the discrete facet normal has no
        well-defined orientation (issue #327) — they are substituted with
        the mesh factory's declared analytic normal (see
        :meth:`canonical_normal` and :meth:`_resolve_boundary_normals`).

        Returns
        -------
        sympy.Matrix
            Row matrix of boundary normal components.
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

        COLLECTIVE: the bounding box spans the whole mesh, so it is reduced
        across ranks (see :meth:`_global_coord_bounds`).

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

        min_coords, max_coords = self._global_coord_bounds()

        return (
            self._model.scale_to_physical(min_coords, dimension="length"),
            self._model.scale_to_physical(max_coords, dimension="length"),
        )

    @property
    def physical_extent(self):
        """
        Mesh spatial extent in physical units.

        Returns the mesh size (max - min) in each dimension scaled to physical units.

        COLLECTIVE: the extent spans the whole mesh, so it is reduced across
        ranks (see :meth:`_global_coord_bounds`).

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

        min_coords, max_coords = self._global_coord_bounds()

        return self._model.scale_to_physical(
            max_coords - min_coords, dimension="length")

    def _global_coord_bounds(self):
        """Bounding box of the mesh nodes, ``(min_coords, max_coords)``.

        COLLECTIVE. Each rank holds only its own subdomain's nodes, so a
        rank-local ``min``/``max`` describes the partition rather than the
        mesh — every rank would report a different "domain size". The
        reduction makes the answer global and identical everywhere, and lets
        a rank owning no cells (hence no nodes) contribute the identity
        elements instead of raising on an empty array (issue #405).

        Reads the same node coordinates as the deprecated ``mesh.points``
        (without its warning or unit wrapping), so the physical-bounds /
        physical-extent answers are unchanged apart from being global.

        .. TODO(BUG): ``mesh.points`` already multiplies by
           ``CoordinateSystem._length_scale`` when the coordinate system is
           scaled, and both callers then pass the result through
           ``model.scale_to_physical(..., dimension="length")`` — a second
           application of the same factor. Reproduced here deliberately so
           this fix stays behaviour-neutral; the double scaling is a separate
           question for the units owner.
        """
        import numpy as np
        from mpi4py import MPI

        coords = np.asarray(self._coords, dtype=np.float64).reshape(
            -1, self.cdim)
        if getattr(self.CoordinateSystem, "_scaled", False):
            coords = coords * self.CoordinateSystem._length_scale
        coords = np.ascontiguousarray(coords)

        if coords.shape[0] > 0:
            local_min = np.ascontiguousarray(coords.min(axis=0))
            local_max = np.ascontiguousarray(coords.max(axis=0))
        else:
            local_min = np.full(self.cdim, np.inf)
            local_max = np.full(self.cdim, -np.inf)

        if uw.mpi.size > 1:
            # Buffer (uppercase) Allreduce: the pickling `allreduce` applies
            # MPI.MIN through Python's `min()`, which is ambiguous for arrays.
            global_min = np.empty_like(local_min)
            global_max = np.empty_like(local_max)
            uw.mpi.comm.Allreduce(local_min, global_min, op=MPI.MIN)
            uw.mpi.comm.Allreduce(local_max, global_max, op=MPI.MAX)
            local_min, local_max = global_min, global_max

        return local_min, local_max

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
        - ``petsc_reload=True`` writes PETSc DMPlex metadata and an owned
          global-vector payload into the same per-variable HDF5 files. These
          files can then be loaded with ``MeshVariable.read_checkpoint()`` for
          exact same-layout reload.

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
        """Write DMPlex metadata and exact same-layout global vectors."""

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

        viewer = PETSc.ViewerHDF5().create(
            checkpoint_file, "a", comm=PETSc.COMM_WORLD
        )
        try:
            viewer.pushGroup("/uw_checkpoint")
            for var in variables:
                var._sync_lvec_to_gvec()
                checkpoint_vec = PETSc.Vec().createWithArray(
                    var._gvec.array_r, comm=PETSc.COMM_WORLD
                )
                checkpoint_vec.setName(var.clean_name)
                viewer(checkpoint_vec)
                checkpoint_vec.destroy()
            viewer.popGroup()
        finally:
            viewer.destroy()

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
                # Context manager: an exception mid-block must not leak the
                # handle (a live handle keeps the HDF5 lock held).
                with h5py.File(filename, "a") as f:
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

        # Same quiescence contract as Swarm.save (issue #330): every rank
        # waits for rank 0's metadata append, so an immediate reopen of the
        # mesh file cannot hit HDF5 file locking.
        uw.mpi.barrier()

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

    def _basis_coordinate_dm(self, degree, continuous):
        """Coordinate DM carrying a degree-``degree`` Lagrange field.

        Its local section defines the node layout that
        :meth:`_get_coords_for_basis` reads, so anything that needs to know
        WHICH node is which — as opposed to just where the nodes are — has to
        come from this same DM. The caller destroys it.
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
        dmfe.destroy()          # DMSetField took its own reference
        return dmnew

    def _cell_node_indices(self, degree, continuous):
        """Rows of each cell's degrees of freedom, indexing exactly the array
        :meth:`_get_coords_for_basis` returns for the same ``(degree,
        continuous)``.

        Returns an ``(n_cells, nodes_per_cell)`` integer array; row ``k`` lists
        cell ``cStart + k``'s DOF rows in closure order. The order within a row
        is arbitrary but self-consistent, which is all any consumer needs: the
        weights are computed from the coordinates read at these same rows, so
        no reference-element node ordering is ever assumed. (Element assembly
        WOULD care — the two DOFs on an edge follow the edge's own orientation,
        not the cell's — so do not repurpose this for that.)

        Knowing a cell's nodes is what turns the adapt parent-CELL map into an
        exact transfer at any polynomial degree: for a Lagrange element the
        basis is dual to its nodal points, so a coarse cell's own DOF
        coordinates determine the coarse interpolant inside it (#425).
        """
        from math import comb

        key = (self.isSimplex, degree, continuous)
        if key in self._cell_node_array:
            return self._cell_node_array[key]

        dmnew = self._basis_coordinate_dm(degree, continuous)
        section = dmnew.getLocalSection()
        cStart, cEnd = self.dm.getHeightStratum(0)
        cdim = self.cdim

        rows = []
        for cell in range(cStart, cEnd):
            cell_rows = []
            for point in dmnew.getTransitiveClosure(cell)[0]:
                ndof = section.getDof(point) // cdim
                if ndof:
                    offset = section.getOffset(point) // cdim
                    cell_rows.extend(range(offset, offset + ndof))
            rows.append(cell_rows)
        dmnew.destroy()

        expected = comb(degree + self.dim, self.dim)
        if not self.isSimplex or any(len(r) != expected for r in rows):
            # A tensor-product Q_k cell carries (k+1)^dim nodes, so the monomial
            # basis of total degree <= k would not be square against them and
            # the dual-basis construction does not apply. Say so here rather
            # than return a ragged array the caller has to second-guess.
            got = sorted({len(r) for r in rows})
            raise NotImplementedError(
                f"_cell_node_indices needs a simplex mesh: expected "
                f"{expected} nodes per cell for degree {degree} in {self.dim}D, "
                f"got {got}")

        self._cell_node_array[key] = numpy.asarray(rows, dtype=numpy.int64)
        return self._cell_node_array[key]

    def _get_coords_for_basis(self, degree, continuous):
        """
        This function returns the vertex array for the
        provided variable. If the array does not already exist,
        it is first created and then returned.
        """

        dmold = self.dm.getCoordinateDM()
        dmnew = self._basis_coordinate_dm(degree, continuous)

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

        return arrcopy

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
        # Largest distance from a cell centroid to one of that cell's own
        # vertices, maximised over local cells. A convex cell is the convex
        # hull of its vertices, so every point of it lies within this distance
        # of its centroid — which makes it the rejection radius the locator
        # needs (see _get_closest_local_cells_internal).
        cell_reach = 0.0

        for cell, cell_id in enumerate(range(cStart, cEnd)):

            cell_faces = nav_dm.getCone(cell_id)
            points = nav_dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
            # Use raw internal array for KD-tree construction (avoid unit-aware wrapping)
            cell_point_coords = nav_coords[self._coord_rows_for_points(nav_dm, points)]
            cell_centroid = cell_point_coords.mean(axis=0)
            centroids_list.append(cell_centroid)
            cell_reach = max(cell_reach, float(numpy.linalg.norm(
                cell_point_coords - cell_centroid, axis=1).max()))

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

        # A rank can own zero cells (small mesh / imbalanced partition):
        # shape the point arrays explicitly so an empty list becomes a
        # well-formed (0, cdim) array rather than a 1-D numpy.array([]),
        # which crashed the KDTree construction (issue #399).
        self._indexCoords = numpy.array(
            control_points_list, dtype=numpy.float64).reshape(-1, self.cdim)
        self._index = uw.kdtree.KDTree(self._indexCoords)
        # self._index.build_index()
        self._indexMap = numpy.array(control_points_cell_list, dtype=numpy.int64)

        # Cell-centroid kdtree, built from the nav-DM cells in the
        # same enumeration order as _indexMap, so the indices it
        # returns can be used directly as nav-DM cell indices.
        # We keep _nav_centroids separate from _centroids (which is
        # the main-DM cell centroids set in __init__) so the FE-side
        # ``_centroids`` semantics are unchanged on manifold meshes.
        self._nav_centroids = numpy.array(
            centroids_list, dtype=numpy.float64).reshape(-1, self.cdim)
        self._centroid_index = uw.kdtree.KDTree(self._nav_centroids)

        # Rejection radius for the lost-point walk. Rebuilt with the kd-tree
        # (i.e. invalidated by deform / adapt along with _index) — this is the
        # ONE place it is set, and the only builder of ``_index``, so a stale
        # reach cannot outlive the geometry it was measured on. Two other
        # builders (``_build_kd_tree_index_PIC``, ``_build_kd_tree_index_DS``)
        # set ``_index`` without the reach; both had zero callers and are
        # deleted rather than taught the new invariant. Pinned by
        # test_0761_point_locator.py::test_the_rejection_radius_is_rebuilt_when_the_mesh_moves.
        self._local_cell_reach = cell_reach

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
        return self._classify_points_in_domain(points, strict_validation)[0]

    def _classify_points_in_domain(self, points, strict_validation=True):
        """In/out classification, keeping the owning cells it had to locate.

        ``points_in_domain`` located the near-boundary points and threw the
        owning cells away, leaving the interpolator to locate them a second
        time. This hands them over instead, so evaluation locates each point
        once (#551 item 2).

        Parameters
        ----------
        points : array-like
            Coordinate array in any physical unit system (will be
            auto-converted).
        strict_validation : bool
            Whether to perform strict validation near boundaries.

        Returns
        -------
        in_or_not : numpy.ndarray of bool
            Exactly what :meth:`points_in_domain` returns.
        cells : numpy.ndarray of int
            Owning cell for the points the classification actually located,
            at the evaluation face tolerance (:meth:`_robust_owning_cells`).
            ``-1`` everywhere else — for an EXTERIOR point that means "not in
            the local mesh"; for an INTERIOR point it means "not looked up",
            because the boundary-sign test settled it without a search. A
            caller that needs a cell for every interior point locates the
            ``-1`` entries itself, and only when it needs them: nothing here
            searches on the classifier's behalf.
        """
        # Convert points to model coordinates using the unified conversion function
        # This handles all coordinate formats: plain numbers, unit-aware coordinates, lists, tuples, arrays
        import underworld3 as uw
        from underworld3.function.unit_conversion import _convert_coords_to_si

        # _convert_coords_to_si now converts to model coordinates (despite the name)
        # and handles all the complexity of extracting values from unit-aware coordinates
        model_points = _convert_coords_to_si(points)

        # get_max_radius() is COLLECTIVE, so it must be reached by every rank
        # before any rank takes a short-circuit below — otherwise the starved
        # ranks skip the reduction their peers are sitting in (issue #405).
        max_radius = self.get_max_radius()

        self._mark_local_boundary_faces_inside_and_out()

        if model_points.shape[0] == 0:
            return (numpy.array([], dtype=bool),
                    numpy.array([], dtype=numpy.int64))

        cells = numpy.full(model_points.shape[0], -1, dtype=numpy.int64)

        # A rank owning no cells contains no points, so the honest answer is
        # False everywhere. Its local boundary skeleton is empty too, and the
        # closest-local-cell test below would otherwise have to interrogate a
        # cell set that does not exist.
        cStart, cEnd = self.dm.getHeightStratum(0)
        if cEnd == cStart:
            return numpy.zeros(model_points.shape[0], dtype=bool)

        # Cd-1 surface mesh: no boundary-face control points exist
        # (see _mark_local_boundary_faces_inside_and_out). Per the
        # surface-mesh contract, query points are assumed to lie on
        # the manifold; the closest-local-cell test is the right
        # filter, not an inside/outside split.
        if self.boundary_face_control_points_kdtree is None:
            in_or_not = self._get_closest_local_cells_internal(model_points) != -1
            return in_or_not, cells

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
        #
        # Only the robust locator's answer is kept as a cell hint: it is the
        # same call the evaluation path makes, so keeping it saves a repeat.
        # The cell-wall test runs at a different face tolerance and its answer
        # is a classification, not a hint.
        near_boundary = numpy.where(dist2 < 2 * max_radius**2)[0]
        near_boundary_points = model_points[near_boundary]

        if self._eval_use_robust_location():
            cells[near_boundary] = self._robust_owning_cells(near_boundary_points)
            in_or_not[near_boundary] = cells[near_boundary] >= 0
        else:
            in_or_not[near_boundary] = (
                self._get_closest_local_cells_internal(near_boundary_points) != -1
            )

        if strict_validation:
            chosen_ones = numpy.where(in_or_not == True)[0]
            chosen_points = model_points[chosen_ones]
            if self._eval_use_robust_location():
                cells[chosen_ones] = self._robust_owning_cells(chosen_points)
                in_or_not[chosen_ones] = cells[chosen_ones] >= 0
            else:
                in_or_not[chosen_ones] = self._get_closest_local_cells_internal(chosen_points) != -1

        # A point demoted to exterior keeps no hint: it goes to RBF.
        cells[~in_or_not] = -1

        return in_or_not, cells

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
            # >= : valid indices are 0..n-1, and the empty-tree sentinel
            # (0 with n=0) must trip this guard, not index _indexMap (#399).
            if np.any(closest_points >= self._index.n):
                raise RuntimeError(
                    "An error was encountered attempting to find the closest cells to the provided coordinates."
                )
            return self._indexMap[closest_points]
        else:
            ### returns an empty 1D array if no coords are provided
            # CRITICAL: Must return 1D array, not 2D, for Cython buffer compatibility
            return numpy.array([], dtype=numpy.int64)

    # Safety factor on the local cell reach used to reject a query point
    # before the lost-point walk. See _get_closest_local_cells_internal.
    _LOCATOR_REACH_MARGIN = 2.0

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

        A point the first containment test rejects is looked for among the
        nearest cell centroids. Points too far from the local mesh to be in
        any of its cells are rejected before that walk starts, and a point
        leaves the walk as soon as a cell claims it, so the walk costs what is
        still lost rather than what was asked for.

        .. note:: **Which containing cell you get changed (#551).**

           A point on a shared vertex, edge or face is contained by several
           cells and this routine returns one of them; *which* one has never
           been part of the contract. It used to be the last cell to claim the
           point across up to 50 rounds of the walk — an order that depended
           on whether some unrelated point in the same batch was still lost.
           It is now the containing cell with the nearest centroid, which is
           batch-independent. Measured on a uniform 3-D simplex box, 35% of
           near-vertex queries (the population that actually enters the walk;
           exact vertices and centroids are answered before it) come back in a
           different — equally containing — cell.

           For a CONTINUOUS field that is invisible: the interpolants of the
           containing cells agree at the shared point. For a DISCONTINUOUS
           field (P0, or the P2/P0-discontinuous pressure space the fault work
           uses) the cell *is* the answer, so the evaluated value moves by
           O(jump) at such points — measured max 1.935 on a P0 field of range
           2. Both values are legitimate: each is the value of a cell that
           contains the query. Code that needs a specific side of a jump must
           say which side, not rely on the locator's tie-break.

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
            control_point_distance, closest_points = self._index.query(
                coords, k=1, sqr_dists=False)
            # >= : valid indices are 0..n-1, and the empty-tree sentinel
            # (0 with n=0) must trip this guard, not index _indexMap (#399).
            if np.any(closest_points >= self._index.n):
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

        if lost_points.shape[0] == 0:
            return cells

        # Part 2 - try to find the lost points by walking nearby cells.
        #
        # Reject what cannot possibly be found, first. Every cell contributes
        # its centroid to the control-point kd-tree, so a point lying in cell c
        # is at most |p - centroid_c| from its NEAREST control point, and a
        # convex cell puts that within the cell's vertex reach. A lost point
        # whose nearest control point is beyond the largest local reach is in
        # no local cell and the walk has nothing to find for it. Without this
        # every genuinely foreign point pays the full 50-neighbour walk — 51
        # containment tests against 1 for an owned point — and the foreign
        # fraction is exactly what grows with rank count (#551).
        #
        # The margin is deliberately loose. The in-cell test admits a thin
        # slab outside each face (``tol``), and a badly shaped cell expands
        # further under that slab than a well-shaped one; a factor of two on
        # the reach covers both with room to spare while still rejecting
        # anything more than about one cell away from the local mesh.
        #
        # Note the two scales are set differently: the slab the containment
        # test admits is ``tol`` times the face control-point separation,
        # which _mark_faces_inside_and_out fixes at an ABSOLUTE 1e-3 in model
        # units, while the radius here is a fraction of the LOCAL cell size.
        # They only cross over when the largest local cell reach falls below
        # about 5e-6 in model units — a whole domain a few microns across, at
        # which scale the containment test's own absolute floors have already
        # gone. Measured: a mesh 1e-4 across (reach 6.9e-6) and one 6371
        # across both reject nothing they should have kept.
        reach = getattr(self, "_local_cell_reach", None)
        if reach is not None and reach > 0.0:
            reach = self._LOCATOR_REACH_MARGIN * reach
            lost_points = lost_points[control_point_distance[lost_points] <= reach]
            if lost_points.shape[0] == 0:
                return cells

        # Size by the nav-DM cell count, which is what _centroid_index
        # was built from (includes ghost cells on manifold meshes).
        nav_centroids = getattr(self, "_nav_centroids", None)
        if nav_centroids is None:
            nav_centroids = self._centroids
        num_local_cells = nav_centroids.shape[0]
        num_testable_neighbours = min(num_local_cells, 50)

        centroid_distance, closest_centroids = self._centroid_index.query(
            coords[lost_points], k=num_testable_neighbours, sqr_dists=False
        )
        # The kd-tree drops the neighbour axis at k == 1 (a rank owning a
        # single cell); the walk indexes it either way.
        centroid_distance = centroid_distance.reshape(lost_points.shape[0], -1)
        closest_centroids = closest_centroids.reshape(lost_points.shape[0], -1)

        # This number is close to the point-point coordination value in 3D unstructured
        # grids (by inspection)

        # The working set shrinks: a point drops out as soon as a neighbour
        # claims it, or as soon as the neighbour distances (sorted, so
        # monotonic in i) pass the rejection radius. The nearest containing
        # centroid therefore wins, which also makes the answer independent of
        # whether some OTHER point in the same batch is findable — previously
        # a single unlocatable point kept every already-found point in the
        # test set for all 50 rounds, and a shared-face point could be
        # reassigned to a further cell in a later round.
        working = np.arange(lost_points.shape[0])
        for i in range(0, num_testable_neighbours):

            if reach is not None and reach > 0.0:
                working = working[centroid_distance[working, i] <= reach]
                if working.shape[0] == 0:
                    break

            candidate_cells = closest_centroids[working, i]
            inside = self._test_if_points_in_cells_internal(
                coords[lost_points[working]], candidate_cells,
                on_boundary=on_boundary, tol=tol,
            )
            cells[lost_points[working[inside]]] = candidate_cells[inside]

            working = working[~inside]
            if working.shape[0] == 0:
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

        For a point several cells share, the cell returned is the containing
        one with the nearest centroid — see the tie-break note on
        :meth:`_get_closest_local_cells_internal` for what that changed and
        why it is visible only to discontinuous fields.
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

        # A rank owning zero cells has no centroid; mean() of the empty
        # array is NaN. gather_data no longer strips NaN rows (issue #405
        # made that opt-in), but a NaN row would still poison the kd-tree
        # this table feeds, and _route_by_nearest_centroid would mis-route
        # particles (issue #399 review). A huge FINITE sentinel keeps the
        # row (row == rank) while a nearest-centroid search can never
        # select it, so starved ranks correctly receive no particles.
        # (Finite, not inf: infinities poison the kd-tree's bounding boxes.)
        if self._centroids.shape[0] > 0:
            domain_centroid = self._centroids.mean(axis=0)
        else:
            domain_centroid = np.full(self.cdim, 1.0e30)
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
        Global minimum of the characteristic cell length scale — the smallest
        cell anywhere in the mesh, not just on this rank. Parallel-safe via
        MPI allreduce of the local minimum.

        A rank owning zero cells contributes the identity element of the
        reduction (:math:`+\\infty`) and therefore returns the same global
        value as its populated peers. Taking ``min()`` of that rank's empty
        ``_radii`` array instead would raise on the starved rank alone, while
        its peers waited in the reduction — the rank-asymmetric raise that
        deadlocks the job (issue #405).
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        import numpy as np
        from mpi4py import MPI

        radii = np.asarray(self._radii).reshape(-1)
        local_min = float(radii.min()) if radii.size else float("inf")
        if uw.mpi.size > 1:
            local_min = uw.mpi.comm.allreduce(local_min, op=MPI.MIN)
        return local_min

    @uw.collective_operation
    def get_max_radius(self) -> float:
        """
        Global maximum of the characteristic cell length scale — the largest
        cell anywhere in the mesh. Parallel-safe via MPI allreduce of the
        local maximum; a rank owning zero cells contributes the identity
        element (:math:`-\\infty`) and still returns the global value.
        See :meth:`get_min_radius` for why the guard matters.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        import numpy as np
        from mpi4py import MPI

        radii = np.asarray(self._radii).reshape(-1)
        local_max = float(radii.max()) if radii.size else float("-inf")
        if uw.mpi.size > 1:
            local_max = uw.mpi.comm.allreduce(local_max, op=MPI.MAX)
        return local_max

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
        because the base hierarchy is static across adapts.

        Each wrap is tagged with its ``(hierarchy token, level)`` slot — and so
        is this mesh itself, whose ``dm`` is the finest hierarchy level. Two
        levels whose slots are consecutive under the same token are a native
        ``refine()`` pair, which is what lets ``custom_mg`` give that pair the
        EXACT nested prolongation instead of a point-located one (#425/#629).
        The token is a plain sentinel object: identity ties the family together
        without a reference cycle back to this mesh."""
        cached = getattr(self, "_coarse_level_meshes_cache", None)
        if cached is None:
            cached = [self._wrap_coarse_level(d) for d in self.dm_hierarchy]
            token = object()
            for k, w in enumerate(cached):
                w._refine_slot = (token, k)
            self._refine_slot = (token, len(cached) - 1)
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
        can be modified: the base implementation supports **2D
        (triangle) and 3D (tetrahedral) simplex meshes**, where it
        drives the Huang–Kamenski variational MMPDE mover (non-folding
        by construction, parallel-safe; scalar metric → isotropic
        equidistribution, tensor metric → anisotropic clustering and
        alignment). Quadrilateral / hexahedral meshes and constrained
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
                "and 3D simplex (tetrahedral) meshes, via the MMPDE mover.")
        if not bool(self.dm.isSimplex()) or self.cdim not in (2, 3):
            kind = "simplex" if bool(self.dm.isSimplex()) else "tensor-product (quad/hex)"
            raise NotImplementedError(
                f"node redistribution is not implemented for {self.cdim}D "
                f"{kind} meshes. Implemented today: 2D simplex (triangle) "
                "and 3D simplex (tetrahedral) meshes, via the MMPDE mover "
                "(no quad/hex discretization exists). To add resolution "
                "instead, use mesh.adapt(metric_field, max_levels=...) — "
                "a topology change.")
        from underworld3.meshing.smoothing import smooth_mesh_interior
        smooth_mesh_interior(self, metric=metric, method="mmpde",
                             verbose=verbose, **kwargs)

    def label_interface_band(self, surface, offset=0.0, halo=1, name=None):
        """Label the vertices of every cell an interface passes through.

        The interface is the level set ``distance(surface) == offset`` — the
        surface itself when ``offset`` is zero, or the margin of a weak zone of
        half-width ``offset``. Cells the level set cuts are the ones that cannot
        represent the material change across them, and their vertices are what
        :meth:`relax` must hold still if the refinement that placed small cells
        there is not to be undone.

        Parameters
        ----------
        surface : Surface
            Provides the exact distance field.
        offset : float, default 0.0
            Distance at which the interface sits.
        halo : int, default 1
            Extra rings of vertices to include. Pinning only the cut cells leaves
            the mover free to pull on their immediate neighbours, which drags the
            pinned ring out of shape from outside, so at least one ring is
            usually wanted.
        name : str, optional
            Label name. Defaults to ``"PinnedBand_<surface name>"``.

        Returns
        -------
        str
            The label name, ready to pass to :meth:`relax` or
            :meth:`redistribute_nodes` as part of ``pinned_labels``.

        Notes
        -----
        The test is purely geometric, so every rank labels its own copy of a
        shared vertex identically and the result does not depend on the partition.
        """
        import numpy

        dm = self.dm
        vS, vE = dm.getDepthStratum(0)
        cS, cE = dm.getHeightStratum(0)
        coords = numpy.asarray(dm.getCoordinatesLocal().array).reshape(
            -1, self.dim)
        # SIGNED distance for the surface itself, UNSIGNED for a margin. The
        # straddle test is "the level set passes between these vertices", and
        # against the unsigned distance that can never be true at offset zero
        # because the unsigned distance is never negative — the surface would
        # label nothing at all. At a non-zero offset the unsigned distance is the
        # right choice precisely because a weak zone has TWO margins, at +offset
        # and -offset, and it catches both.
        distance = (surface.signed_distance(coords) if offset == 0.0
                    else surface.unsigned_distance(coords))

        cell_vertices = [
            numpy.array([int(p) for p in dm.getTransitiveClosure(c)[0]
                         if vS <= p < vE])
            for c in range(cS, cE)]

        def _sync_across_ranks(pinned_set):
            """A vertex pinned on ANY rank is pinned on EVERY rank holding a copy.

            The straddle test and the ring growth both walk rank-LOCAL cells, and
            cells are partitioned disjointly — so a shared vertex whose cut (or
            ring) cell lives on the neighbour rank is pinned there but not here.
            If HERE is the owner, the mover moves it and the neighbour's pinned
            copy follows through the SF: measured at np=4 (review of PR #488,
            2026-08-06), two pinned leaves moved 4.2e-3 and 1.9e-3 while np=2/3
            passed on partition luck. Matching by coordinate (the same rounded
            key the parallel test uses) makes the set partition-independent.
            """
            if uw.mpi.size == 1:
                return pinned_set
            local_xy = (coords[[v - vS for v in pinned_set]]
                        if pinned_set else numpy.empty((0, self.dim)))
            global_keys = set()
            for arr in uw.mpi.comm.allgather(local_xy):
                for p in arr:
                    global_keys.add(tuple(numpy.round(p, 12)))
            out = set(pinned_set)
            for i in range(vE - vS):
                if tuple(numpy.round(coords[i], 12)) in global_keys:
                    out.add(vS + i)
            return out

        pinned = set()
        for verts in cell_vertices:
            d = distance[verts - vS]
            if d.min() < offset < d.max():
                pinned.update(int(v) for v in verts)
        pinned = _sync_across_ranks(pinned)
        for _ring in range(halo):
            grown = set()
            for verts in cell_vertices:
                vv = [int(v) for v in verts]
                if any(v in pinned for v in vv):
                    grown.update(vv)
            pinned |= grown
            pinned = _sync_across_ranks(pinned)

        # COLLECTIVE emptiness test. A rank whose subdomain the surface never
        # enters legitimately has an empty local band — only a GLOBALLY empty
        # band is a user error. The previous rank-local raise here deadlocked
        # np=4 (measured 2026-08-06, review of PR #488: a corner-confined
        # surface left three ranks raising while the fourth entered the
        # collective mover and hung to the 300 s timeout).
        n_global = uw.mpi.comm.allreduce(len(pinned))
        if n_global == 0:
            # Raised on EVERY rank, after the collective reduction.
            raise ValueError(
                f"no cell is cut by distance == {offset} on surface "
                f"{getattr(surface, 'name', surface)!r}, so there is no band to "
                f"pin. Check the offset lies inside the mesh and matches the "
                f"interface you meant (for a weak zone it is the HALF-WIDTH, not "
                f"zero).")

        # Every rank creates the label, including ranks whose local band is
        # empty: the downstream consumer (smoothing.graph._pinned_mask) is
        # documented to tolerate a present-but-empty label, and a label that
        # exists on some ranks only is the kind of asymmetry this method is
        # not allowed to produce.
        name = name or f"PinnedBand_{getattr(surface, 'name', 'surface')}"
        if not dm.hasLabel(name):
            dm.createLabel(name)
        label = dm.getLabel(name)
        for v in pinned:
            label.setValue(v, 1)
        return name

    def relax(self, metric=None, *, pin_bands=None, pin_halo=1, verbose=False,
              **kwargs):
        r"""Improve this mesh's element **shapes** without changing its
        size distribution or its topology.

        The companion to :meth:`adapt`. Refinement chooses where new
        nodes go from *combinatorics* — which edge the tagging rule
        nominated (bisection), or the cell centroid (Alfeld) — never
        from geometry, so a refined mesh carries needles and slivers
        that reflect the base mesh's arbitrary choices rather than
        anything about the problem. Relaxation moves those nodes to
        where the geometry wants them, keeping the resolution the
        refinement installed::

            child = mesh.adapt(metric, max_levels=3)
            child.relax()

        Implemented with the same MMPDE mover as
        :meth:`redistribute_nodes`, in its **ideal reference frame**
        (``reference="ideal"``): each cell's reference element is a
        regular simplex scaled to that cell's own current volume. Two
        consequences, and they are the whole point:

        * the size term starts and stays at its optimum, so the graded
          spacing is preserved rather than re-derived;
        * "distortion" is measured against *equilateral*, not against
          the mesh as supplied — so a distorted mesh is no longer its
          own optimum, which is exactly why
          ``redistribute_nodes(metric)`` cannot do this job (its
          reference IS the mesh it was handed, so under a uniform
          metric it moves nothing at all).

        Non-folding by construction, and the parallel partition, vertex
        count and DOF layout are unchanged.

        Moving the nodes can upset the custom-P geometric-MG transfers,
        which are built by geometric point location rather than from the
        refinement relation: the local-support ``barycentric`` builder can
        be left with a coarse DOF that has no fine image. The FMG build
        now retries with the global-support ``rbf`` builder before giving
        up, so the hierarchy survives (measured in 3D: GAMG at 23
        iterations before the retry, ``pc=mg`` at 2 after). See #424.

        **Two valid placements, neither dominant.** ``adapt(metric, ...,
        relax=True)`` relaxes once at the end — the recommended default.
        ``relax="per-generation"`` relaxes inside the refinement loop, so
        each generation marks from already-relaxed coordinates. Both beat
        no relaxation; which is *better* depends on which property you
        weight, and we deliberately do not rank them:

        =========================  ==========  ==========  ===============
        property                   unrelaxed   at end      per generation
        =========================  ==========  ==========  ===============
        P1 interpolation error     2.75e-2     2.56e-2     2.38e-2
        99th pct max angle         112 deg     110 deg     96 deg
        on-fault size spread       2.80        2.31        2.31
        far-field closure halo     42.8        27.7        43.9
        =========================  ==========  ==========  ===============

        Relaxing at the end keeps the cell count identical to the
        unrelaxed mesh and cuts the far-field halo most; relaxing per
        generation gives the cleanest element shapes and the lowest
        error, but spends ~3% more cells to do it.

        In **3D it improves mesh QUALITY but not interpolation error** —
        two different things, and worth keeping apart. On an adapted 3D
        mesh it halves the near-degenerate population (cells with q < 0.1:
        3.6% -> 1.8%), lifts median quality 0.32 -> 0.39 and pulls the 99th
        percentile dihedral angle back from 153 to 146 degrees; but the
        interpolation error of an isotropic feature is unchanged (+0.5%).
        That is consistent: relaxation holds the size distribution, and in
        2D the error gain came from cells ALIGNING onto the feature, which
        an isotropic metric gives no reason to do in 3D. Use it in 3D for
        conditioning and element quality, not expecting an accuracy win.

        Parameters
        ----------
        metric : sympy expression, MeshVariable, or sympy Matrix, optional
            Usually omitted, and **omitting it is what makes this a shape
            guarantee**. ``None`` (default) relaxes under a uniform metric
            — pure shape repair at fixed size.

            Passing a metric switches to the ideal-*metric* frame, which
            re-grades as well as reshapes, and it will trade element shape
            away to chase the size field. Measured on a 4-level graded box,
            the 99th-percentile max angle went 117.9 -> 113.8 degrees with
            no metric but 117.9 -> **127.4** with one. Pass a metric when
            you want the sizes corrected too, and accept that shape is no
            longer the objective.
        pin_bands : sequence, optional
            Interfaces whose cells must not move: each entry is a ``Surface``, or
            a ``(surface, offset)`` pair when the interface is a level set of the
            distance rather than the surface itself (a weak zone of half-width
            ``offset``). Their bands are labelled via
            :meth:`label_interface_band` and held fixed.

            This is the difference between relaxation helping and hurting when a
            mesh has been refined onto an interface. The mover optimises element
            shape against an equilateral reference and knows nothing about where
            the material changes, so it slides the small cells that refinement
            placed on the interface *off* it: measured on a step-edged fault, the
            manufactured stress across the interface rose 77 % and stopped being
            confined to the fault. Pinning the band leaves that quantity unchanged
            to five decimal places while the mover still reshapes everywhere else.
        pin_halo : int, default 1
            Rings of neighbouring vertices pinned alongside each band. Pinning the
            cut cells alone lets the mover pull on them from outside.
        verbose : bool, default False
            Print mover progress.
        **kwargs
            Forwarded to :meth:`redistribute_nodes` — e.g.
            ``pinned_labels``, ``slip_surfaces``, ``method_kwargs``
            (mover tunables such as ``n_outer``).

            Note that passing ``pinned_labels`` explicitly REPLACES the default,
            which is to pin every named boundary. ``pin_bands`` is merged with
            that default rather than replacing it, so it cannot silently release
            the domain boundary.

        See Also
        --------
        adapt : Add resolution (topology change, returns a child mesh).
        redistribute_nodes : Move nodes to follow a metric (changes the
            size distribution; the reference is the mesh as supplied).
        """
        import sympy

        if pin_bands:
            from underworld3.meshing.smoothing.graph import _auto_pinned_labels

            names = []
            for entry in pin_bands:
                surface, offset = entry if isinstance(entry, tuple) else (entry, 0.0)
                names.append(self.label_interface_band(
                    surface, offset=offset, halo=pin_halo))
            # MERGE with the caller's list, or with the auto default when there is
            # none. Replacing the default would quietly unpin the domain boundary.
            existing = kwargs.pop("pinned_labels", None)
            if existing is None:
                existing = list(_auto_pinned_labels(self))
            kwargs["pinned_labels"] = list(existing) + names

        method_kwargs = dict(kwargs.pop("method_kwargs", None) or {})
        # No metric  -> keep each cell's own size, repair shape only.
        # With one   -> the metric sets size (a uniform reference volume),
        #               so sizes that are themselves wrong can be fixed.
        default_ref = "ideal" if metric is None else "ideal-metric"
        ref = method_kwargs.setdefault("reference", default_ref)
        if ref not in ("ideal", "ideal-metric"):
            raise ValueError(
                "mesh.relax() is an ideal-reference-frame operation; "
                f"method_kwargs['reference']={ref!r} contradicts it. For "
                "the mesh-reference frame call "
                "mesh.redistribute_nodes(metric).")
        return self.redistribute_nodes(
            sympy.sympify(1) if metric is None else metric,
            verbose=verbose, method_kwargs=method_kwargs, **kwargs)

    def _boundaries_with(self, name):
        """This mesh's boundary enum, extended with one more named boundary.

        An ``Enum`` carrying members cannot be subclassed, so the extended enum is
        built fresh from the existing members. The new value is the first free one
        past the largest ordinary boundary.

        Excluding the sentinels ``Null_Boundary`` (666) and ``All_Boundaries``
        (1001) from that maximum is NOT enough to keep off them: a mesh whose
        largest ordinary value is 665 lands the surface exactly on 666. So the
        candidate is stepped past anything already taken. ``Enum`` would not
        complain — it would alias the two names to one value, and every facet of
        the surface would answer to ``Null_Boundary``.
        """
        from enum import Enum

        members = {b.name: b.value for b in self.boundaries}
        if name in members:
            raise ValueError(
                f"this mesh already has a boundary called {name!r}; a conforming "
                "surface needs its own name so a solver can tell them apart.")
        taken = set(members.values())
        ordinary = [v for v in taken if v < 666]
        value = (max(ordinary) + 1) if ordinary else 1
        while value in taken:
            value += 1
        members[name] = value
        return Enum("boundaries", members)

    def cells_supporting(self, name):
        """The cells in the SUPPORT of the facets labelled ``name``.

        This is the **fault zone** of a conforming surface: not a geometrically
        bounded region but the set of cells the labelled facets belong to — one
        element each side of the surface, by construction.

        The definition is worth stating plainly because the obvious alternative
        does not work. A fault one element wide has an end cap (2-D) or an edge
        band (3-D) whose extent equals the THICKNESS, so resolving it would need
        `h` much smaller than `h`. Deriving the zone from the facets instead
        needs no cap, no band and no rim: it terminates automatically where the
        chain of facets ends, it says nothing about dimension, and the zone of a
        network is simply the union of its branches' zones, with no geometry to
        reconcile where they meet.

        The price is that thickness is no longer a physical parameter — it tracks
        `h` (measured: 0.13 `h` half-thickness, constant across a 4x refinement).
        Under adapt-on-top that is the point rather than a defect: the surface
        lives at the finest level, so the zone width is whatever the adapt metric
        asks for locally, which makes fault width a *refinement* parameter.

        Parameters
        ----------
        name : str
            A boundary of this mesh, normally one added by
            :meth:`add_conforming_surface`.

        Returns
        -------
        numpy.ndarray
            Boolean, one entry per cell, in **plex cell order** — which is also
            the DOF order of a ``degree=0`` :class:`MeshVariable`, so it can be
            assigned straight across.

        Examples
        --------
        >>> zone = mesh.cells_supporting("Fault")
        >>> eta = uw.discretisation.MeshVariable("eta", mesh, 1, degree=0)
        >>> eta.array[:, 0, 0] = numpy.where(zone, 1.0e-3, 1.0)

        See Also
        --------
        add_conforming_surface : add the surface whose facets these are.
        """
        from underworld3.utilities.edge_split import _cells_on_edge

        if name not in [b.name for b in self.boundaries]:
            raise ValueError(
                f"{name!r} is not a boundary of this mesh; the surface must be "
                f"added before its zone can be read. Have: "
                f"{[b.name for b in self.boundaries]}")

        dm = self.dm
        cS, cE = dm.getHeightStratum(0)
        zone = numpy.zeros(cE - cS, dtype=bool)

        value = self.boundaries[name].value
        label = dm.getLabel(name)
        # An empty stratum hands back a null IS that segfaults in getIndices().
        # A rank owning no part of the surface is the normal case at np>2.
        if label is None or label.getStratumSize(value) == 0:
            return zone

        fS, fE = dm.getHeightStratum(1)
        for f in label.getStratumIS(value).getIndices():
            p = int(f)
            if fS <= p < fE:
                # A FACET's support is the cells, in any dimension — a 2-D
                # facet is an edge, a 3-D facet is a face. (This method only
                # ever saw 2-D meshes before the 3-D conforming sheet, and
                # the edge walk below returns NOTHING from a 3-D face,
                # silently — the same trap `_cells_on_edge` documents, one
                # level up.)
                for c in dm.getSupport(p):
                    if cS <= c < cE:
                        zone[c - cS] = True
            else:
                # A labelled EDGE in 3-D (a trace chain): cells are one
                # level further up.
                for c in _cells_on_edge(dm, p):
                    zone[c - cS] = True
        return zone

    def cells_labelled(self, name, value=None):
        """The cells carrying DM label ``name`` (optionally stratum ``value``).

        The cell-label partner of :meth:`cells_supporting`: where that method
        derives a zone from a FACET label (a conforming surface's support),
        this one reads a CELL label directly — the region a placement call
        painted (:func:`~underworld3.utilities.place_surface.place_thin_volume`
        labels its layer's cells), an imported region marker, or any other
        authored cell stratum. It is the empty-safe way to build a fault-zone
        patch key for ``set_custom_fmg(..., fac_zone=...)`` (#629): an absent
        label or an empty stratum returns an all-``False`` mask rather than
        touching the null IS that segfaults ``getIndices()`` (#589).

        Parameters
        ----------
        name : str
            The DM label name (e.g. the ``label=`` given to a placement call).
        value : int or None
            The stratum value; ``None`` takes the union over every value the
            label carries.

        Returns
        -------
        numpy.ndarray
            Boolean, one entry per cell, in **plex cell order** — also the DOF
            order of a ``degree=0`` :class:`MeshVariable`. Points of the label
            that are not cells (faces, edges) are ignored.

        Examples
        --------
        >>> zone = mesh.cells_labelled("Band")
        >>> set_custom_fmg(stokes, tail, field_id=0, fac_zone=zone)
        """
        from underworld3.utilities.dm_labels import label_stratum_indices

        dm = self.dm
        cS, cE = dm.getHeightStratum(0)
        zone = numpy.zeros(cE - cS, dtype=bool)
        label = dm.getLabel(name)
        if label is None:
            return zone
        if value is None:
            vis = label.getValueIS()
            values = [int(v) for v in vis.getIndices()] if vis is not None else []
        else:
            values = [int(value)]
        for v in values:
            pts = label_stratum_indices(label, v)
            pts = pts[(pts >= cS) & (pts < cE)]
            zone[pts - cS] = True
        return zone

    @staticmethod
    def _repair_cut(cut_dm, lines, info, reach, verbose):
        """Flip, then delete, the cells a conforming cut left thin.

        The order is measured, not assumed, and it does not commute — see
        :meth:`add_conforming_surface`. Deletion is offered only the vertices
        near the surface, because it removes degrees of freedom and the cut is
        what justifies removing these particular ones; flipping is offered the
        whole mesh, because it conserves the point set.

        ``info`` is restated rather than left as the cut wrote it: its
        ``min_angle`` describes the mesh before repair, and a caller reading it
        off a repaired mesh would be reading the wrong number.
        """
        from underworld3.utilities import reconnect
        from underworld3.utilities.line_cut import (_coords, _distance_to_lines,
                                                    _edge_vertices, _vertex_h,
                                                    min_angles)

        cut_dm, n_flips = reconnect.flip_to_reduce_max_angle(cut_dm)

        vS, vE = cut_dm.getDepthStratum(0)
        X = _coords(cut_dm)[: vE - vS]
        near = _distance_to_lines(X, lines) < reach * _vertex_h(
            X, _edge_vertices(cut_dm))
        cut_dm, n_removed = reconnect.remove_vertices(
            cut_dm, numpy.flatnonzero(near) + vS)

        angles = min_angles(cut_dm)
        info = dict(info, n_repair_flips=n_flips, n_repair_removals=n_removed,
                    min_angle=float(angles.min()) if len(angles) else 0.0)
        if verbose:
            uw.pprint(f"[surface repair] {n_flips} flips, {n_removed} vertices "
                      f"removed, min angle now {info['min_angle']:.2f} deg")
        return cut_dm, info

    def _adopt_cut_child(self, cut_dm, boundaries, info, mg_coarsening_ratio,
                         verbose):
        """Wrap a DM cut at the finest level as this mesh's child.

        Shared by :meth:`add_conforming_surface` (2-D line cut) and
        :meth:`add_conforming_sheet` (3-D placed sheet): the cut exists on
        the finest level only, so this mesh plus everything below it is the
        child's coarse multigrid tail. The coarse levels do not carry the
        surface and do not need to — see the measured note in
        :meth:`add_conforming_surface`.
        """
        child = Mesh(
            cut_dm,
            simplex=self.dm.isSimplex(),
            coordinate_system_type=self.CoordinateSystem.coordinate_type,
            qdegree=self.qdegree,
            boundaries=boundaries,
            verbose=False,
        )
        child.parent = self
        child._relationship_kind = "refinement"
        # ... but NOT a nested one. The cut moves or replaces parent vertices
        # (the 2-D cut snaps them onto the surface; the 3-D carve deletes and
        # refills), so a coarse DOF need not have a coincident fine DOF, and
        # the injection that a bisection child's restriction relies on would
        # quietly read the field at the displaced position instead.
        child._refine_dofs_coincide = False
        child.regions = self.regions
        child._parent_mesh_version = self._mesh_version
        child._surface_info = info

        # Mesh-owned custom-P geometric-MG tail. Adding a surface refines this
        # mesh, so this mesh plus everything below it is a valid coarse tail and
        # the solver appends the child as the finest level. The transfers are
        # coordinate-based and do not need the levels to nest — just as well,
        # since a cut vertex is not an edge midpoint and the exact 1/2,1/2
        # prolongation does not apply to it.
        #
        # A mesh that is ITSELF a child (a second surface, or an adapt child) has
        # to EXTEND its own tail rather than read `dm_hierarchy`, which for a child
        # holds only its own DM: reading it there would silently discard every
        # level below and leave a two-level hierarchy calling itself multigrid.
        #
        # Tested with `is not None`, not for truthiness. A child whose own tail
        # is EMPTY is still a child, and reading `dm_hierarchy` there returns
        # just its own DM — the two-level collapse this comment warns about,
        # reached by the one input the truth test cannot distinguish from a
        # parent.
        own_tail = getattr(self, "_custom_mg_coarse_meshes", None)
        tail = (list(own_tail) + [self]) if own_tail is not None \
            else self._coarse_level_meshes()

        # A cut is not necessarily a refinement. It re-represents the same grid
        # with the surface conformed, so `self` earns its place as a separate
        # level only if the child is genuinely finer — the same question `adapt`
        # asks of an engine pass, so ask it with the same routine rather than a
        # second rule that could drift from it.
        #
        # Measured on a box fault before this: nine levels, of which the two
        # added by the two cuts coarsened h by 1.11x and 1.17x on the 5th
        # percentile against a threshold of 1.8 — each costing a full Galerkin
        # RAP and smoother sweep for no correction. Worse, transfer 7->8, BETWEEN
        # those two, is where the barycentric builder ran out of coarse DOFs with
        # a fine image and fell back to the dense RBF one (#424).
        #
        # `_subsample_mg_levels` already does "replace the level below rather
        # than append to it" for its own finest generation; handing it the pair
        # (self, child) against the level beneath them puts that decision here
        # too. One level back means it kept only the child.
        if len(tail) >= 2:
            kept, _Ps, _pc = self._subsample_mg_levels(
                tail[-2].dm, [tail[-1].dm, cut_dm], [None, None], [],
                ratio=mg_coarsening_ratio, verbose=verbose)
            if len(kept) == 1:
                tail = tail[:-1]
        child._custom_mg_coarse_meshes = tail
        child._custom_mg_builder = self._custom_mg_builder

        self._registered_children.add(child)
        return child

    def add_conforming_surface(self, surface, snap_frac=0.10, verbose=False,
                               snap_quality=0.15, snap_dist=0.0,
                               mg_coarsening_ratio=2.0, repair=False,
                               repair_reach=0.6):
        r"""Add an internal surface that the mesh conforms to.

        The surface is added *on top of* an existing mesh rather than built into
        the mesh generator, so its position does not have to be known when the
        mesh is made. Every edge the surface crosses is split **at the crossing
        point**, so the surface becomes a chain of element edges: no element
        straddles it, each element lies cleanly on one side, and the edges along
        it carry a boundary label of the surface's name.

        Two things follow from conforming, and both need the surface to be a real
        mesh entity rather than a smooth field:

        * a material property can be assigned per **cell** and be exactly right.
          A property interpolated across a straddling element manufactures stress
          :math:`-2\,\mathrm{Cov}(\eta, \dot\varepsilon)` per cell, which
          refinement shrinks but never removes;
        * the surface is a **named, labelled** set of facets, so downstream passes
          can find it again: ``relax(pin_bands=[name])`` holds it, the reconnection
          pass refuses to flip across it, and the cells either side of it can be
          marked.

        .. warning::

           An **essential boundary condition** on the surface is not yet sound
           under the geometric multigrid hierarchy. The coarse levels do not carry
           the surface at all — by design, see above — so the condition constrains
           the fine level and **zero** coarse degrees of freedom, and the coarse
           operator is singular where custom-P needs it not to be. Surface
           integrals on an embedded surface do not work either, however the
           surface was created.

           A **material contrast** across the surface — a fault zone, or sticky
           air — is unaffected: it needs the surface labelled and the cells either
           side marked, and no condition applied on the facets at all. That is the
           use this method is for.

        **Nothing below the child is cut.** The surface exists on the finest level
        only; this mesh and every coarse multigrid level under it are untouched
        and are reused as the child's coarse tail. That is the whole point of the
        stack-on formulation — the surface's position is a design variable in an
        outer optimisation, so it has to be able to move and be re-added against a
        base and a hierarchy that never change.

        Nor does a coarse cut buy anything. The custom-P hierarchy sets
        ``pc_mg_galerkin=both``, so every coarse operator is
        :math:`P^\mathsf{T} A P` formed from the **fine** operator and inherits
        the material contrast whatever the coarse mesh looks like. Measured on
        SolCx at contrasts of :math:`10^2` and :math:`10^6`, cutting the coarse
        levels changed the error in the fifth significant figure and the solve
        time not at all.

        Parameters
        ----------
        surface : uw.meshing.Surface
            The surface to conform to. Its control points give the polyline and
            its ``name`` becomes a boundary of the returned mesh, so
            ``relax(pin_bands=[surface.name])`` holds it.

            A :class:`~underworld3.meshing.Surface` rather than a
            ``(points, name)`` pair because that is what the rest of the fault
            machinery already takes — ``fault_metric``, ``fault_metric_tensor``
            and ``refinement_metric_function`` all do — so the same object drives
            the refinement metric and the cut, instead of being unpacked and its
            name re-stated. It also carries ``signed_distance`` and ``director``,
            which is what a weak-plane constitutive model needs afterwards.

            The polyline must cross the mesh from boundary to boundary and must
            not cross itself.
        snap_frac : float
            A crossing landing within this fraction of an edge's length from
            either end moves that end onto the surface instead of splitting the
            edge. This is what keeps slivers out: without it an algebraic solver
            pays about 60 % more iterations on the slivers a cut leaves behind.
            The surface stays exactly where it was specified either way — a
            snapped vertex moves *onto* it, not the other way about.

            On a GRADED mesh the 0.10 default is not the best value: measured on
            a four-level adapted mesh, 0.30 took the worst angle from 4.96 to
            10.81 degrees and cells below 15 degrees from 231 to 31. The default
            is left alone because it was chosen on a uniform mesh, where the
            trade is different again — raise it deliberately, and measure.
        verbose : bool
            Report how many edges were split and the worst cell of the result.
        snap_quality : float
            Triangle-quality floor protecting the snap; see
            :func:`~underworld3.utilities.line_cut.cut_along_lines`. It does not
            bind at the recommended tolerances — it is what stops a large
            ``snap_frac`` from flattening cells onto the surface and silently
            breaking the chain.
        snap_dist : float
            Also snap any vertex within this multiple of its local h of the
            surface, whatever the crossings on its edges look like; see
            :func:`~underworld3.utilities.line_cut.cut_along_lines`.
        mg_coarsening_ratio : float
            How much finer the cut must be than the mesh it is cut from before
            that mesh is kept as a multigrid level in its own right rather than
            replaced by the child. Same meaning, and the same routine, as in
            :meth:`adapt`: a level is a coarsening ratio, not a record that an
            operation happened. A cut usually does not clear it, and should not.
        repair : bool, default False
            Repair the element shapes the cut leaves behind, by flipping and then
            deleting. Off by default for the same reason
            :meth:`adapt`'s ``repair`` is: the cut alone gives the same mesh at
            any rank count, and repair gives that up, because which cells may be
            touched depends on where the partitioner drew the seam.

            The cut can only **snap** a vertex onto the surface or **split** an
            edge it crosses, so a crossing landing near a vertex must either drag
            the vertex to it or carve a thin cell beside it — tightening
            ``snap_frac`` only trades one for the other. Repair adds the two
            operations the cut does not have. Measured on a box fault, counting
            cells whose smallest angle is under 15 degrees: 60 after the cut, 18
            after flipping, and **4** after deleting as well — while removing 242
            cells. A second round of either finds nothing.

            The order is fixed and is not symmetric: deleting first leaves the
            count at 60, because a cavity, once retriangulated, no longer
            presents the quad the flip pass was looking for.

            The surface itself is untouched — its vertex and facet counts are
            bit-identical through both passes, at every rank count, because both
            refuse to act on a labelled edge.
        repair_reach : float, default 0.6
            How far from the surface a vertex may be and still be offered for
            deletion, as a multiple of its own local h. This is the *policy*
            half of the repair and the cut is what justifies it: the vertices
            worth removing are the ones the cut had to work around. Deleting
            removes a degree of freedom, so a pass turned loose on the whole mesh
            would coarsen it wherever the shape happened to be poor. Flipping is
            not restricted this way — it conserves the point set.

        Returns
        -------
        Mesh
            A child mesh conforming to the surface, with ``surface.name`` among
            its ``boundaries``. Call again on the result to add a second,
            non-intersecting surface — that is also how a fault NETWORK is built,
            one branch at a time, since each branch wants its own label.

        Examples
        --------
        A weak fault zone one element wide, assigned per cell so the contrast
        falls exactly on the surface:

        >>> fault = uw.meshing.Surface("Fault", mesh,
        ...                            np.array([[0.5, -0.1], [0.5, 1.1]]))
        >>> mesh2 = mesh.add_conforming_surface(fault)
        >>> zone = mesh2.cells_supporting("Fault")          # boolean, per cell
        >>> eta = uw.discretisation.MeshVariable("eta", mesh2, 1, degree=0)
        >>> eta.array[:, 0, 0] = np.where(zone, 1.0e-3, 1.0)

        The same object drives the refinement, so the zone is one element wide at
        whatever resolution the metric asks for locally:

        >>> child = base.adapt(fault.refinement_metric_function(
        ...     h_near=0.01, h_far=0.08, width=0.05), max_levels=3)
        >>> cut = child.add_conforming_surface(fault)

        Notes
        -----
        Two dimensions only — in 3-D use :meth:`add_conforming_sheet`, where
        a free rim (a fault tip) is the normal case. Here a surface **ending
        inside** the mesh is refused rather than silently mis-meshed, as is a
        triangle the surface crosses three times.

        See Also
        --------
        cells_supporting : the fault zone — the cells these facets belong to.
        adapt : local refinement, which reduces the straddling error without
            removing it.
        """
        from underworld3.meshing.surfaces import Surface, _fault_collect_polylines
        from underworld3.utilities.line_cut import cut_along_lines as _cut

        if not isinstance(surface, Surface):
            raise TypeError(
                "add_conforming_surface takes a uw.meshing.Surface, not "
                f"{type(surface).__name__}. Build one with "
                "uw.meshing.Surface(name, mesh, control_points) — it is what the "
                "refinement metric takes too, so the same object can drive both.")

        name = surface.name
        boundaries = self._boundaries_with(name)
        value = boundaries[name].value
        # Reuse the machinery's own "normalise a fault argument" routine, which
        # reads control points in MODEL space — the space the DM's coordinates
        # are in. `surface.control_points` is the dimensionalised gateway and
        # would be the wrong space under an active units system.
        lines = [numpy.array([segs[0][0]] + [b for _a, b in segs])
                 for segs in _fault_collect_polylines(surface)]

        cut_dm, info = _cut(self.dm, lines, snap_frac=snap_frac,
                            label=name, label_value=value,
                            snap_quality=snap_quality, snap_dist=snap_dist)
        if repair:
            cut_dm, info = self._repair_cut(cut_dm, lines, info, repair_reach,
                                            verbose)
        if verbose:
            uw.pprint(f"[surface {name!r}] split {info['n_split']} edges, "
                      f"{info['n_on_surface']} vertices on the surface; "
                      f"{info['n_cut_edges']} surface facets, "
                      f"min angle {info['min_angle']:.2f} deg")

        return self._adopt_cut_child(cut_dm, boundaries, info,
                                     mg_coarsening_ratio, verbose)

    def add_conforming_sheet(self, points, triangles, name, *,
                             clearance=0.6, setback=0.0, size=None,
                             verbose=False, mg_coarsening_ratio=2.0):
        r"""Add a triangulated sheet that the mesh conforms to (3-D).

        The 3-D twin of :meth:`add_conforming_surface`, and the Mesh-level
        form of :func:`~underworld3.utilities.place_surface.place_sheet`:
        the sheet's points become mesh vertices, every sheet triangle an
        interior face labelled ``name``, and the rim is free inside the
        mesh — a fault tip is the normal case here, not a refusal.

        The cut runs at the finest level ONLY. This mesh and every
        multigrid level under it are untouched and become the child's
        coarse tail, exactly as in 2-D: the coarse levels do not carry the
        sheet and do not need to (custom-P sets ``pc_mg_galerkin=both``,
        so every coarse operator is :math:`P^\mathsf{T} A P` from the fine
        operator — see the measured note in :meth:`add_conforming_surface`,
        whose warning about ESSENTIAL conditions on the surface applies
        here unchanged; a material contrast across the sheet is the
        supported use).

        Everything :func:`place_sheet` documents holds: the sheet may run
        PAST the domain (it is clipped against the mesh's own boundary,
        and an outcrop trace is labelled ``<name>_trace``); ``setback``
        stops it short as a BLIND fault with the would-be intersection
        returned in ``child._surface_info["surface_trace"]``; ``size``
        re-triangulates the sheet to match the mesh it cuts.

        Parameters
        ----------
        points, triangles : array_like
            The sheet: ``(N, 3)`` vertices and ``(M, 3)`` triangle
            indices. Explicit arrays rather than an object, because a
            sheet is DATA — a slab model, an authored parameter-space
            triangulation — whose connectivity must be embedded verbatim
            (:class:`~underworld3.meshing.FaultSurface` re-derives its
            triangulation, so it cannot carry an authored one).
        name : str
            Becomes a boundary of the returned mesh, so a solver can
            resolve the facets by name and :meth:`cells_supporting`
            marks the fault zone.
        clearance, setback, size, verbose
            Passed through to :func:`place_sheet`.
        mg_coarsening_ratio : float
            As in :meth:`add_conforming_surface`: the cut replaces this
            mesh in the tail unless it is genuinely finer.

        Returns
        -------
        Mesh
            A new mesh; this one is not modified. Placement metadata is
            on ``child._surface_info``. Call again on the result to add
            another sheet — a network is built one branch at a time.

        See Also
        --------
        add_conforming_surface : the 2-D form.
        add_fault : cut AND split, for a velocity discontinuity.
        cells_supporting : the fault zone of the labelled facets.
        """
        from underworld3.utilities.place_surface import place_sheet

        if self.dim != 3:
            raise NotImplementedError(
                "add_conforming_sheet is 3-D; in 2-D use "
                "add_conforming_surface.")

        boundaries = self._boundaries_with(name)
        cut_dm, info = place_sheet(
            self.dm, points, triangles, label=name,
            label_value=boundaries[name].value, clearance=clearance,
            verbose=verbose, setback=setback, size=size)
        return self._adopt_cut_child(cut_dm, boundaries, info,
                                     mg_coarsening_ratio, verbose)

    def add_fault(self, faults, verbose=False):
        """Cut AND split one or more faults; return the split mesh.

        The split-node fault pipeline in one call: each fault becomes a
        genuine velocity discontinuity — a conforming facet chain whose
        nodes are duplicated, with boundaries ``<name>Plus`` /
        ``<name>Minus`` and the coincident DOF pairing recorded. Interface
        conditions then go through ``solver.add_fault_bc(conds, name)``
        (``conds = 0`` frictionless, ``conds`` > 0 a viscous interface) and
        an ordinary ``solve()``.

        ``faults`` is a ``Surface``, a ``(name, points)`` pair, or a
        sequence of either (a network — cut all, then split all). Each
        fault is one open polyline with both tips strictly inside the
        domain; segments must not share vertices, so branches and
        crossings are represented as OFFSET segments (a one-to-two-cell
        ligament). The result is standalone — no geometric-MG tail, since
        the coarse levels do not carry the fault (see
        :meth:`add_conforming_surface`); solvers take their
        algebraic-multigrid defaults.

        Implementation and design: ``underworld3.utilities.fault_split``
        and ``docs/developer/design/FAULT_CONTACT_DEPLOYMENT_2026-08.md``.
        """
        from underworld3.utilities.fault_split import add_fault
        return add_fault(self, faults, verbose=verbose)


    def adapt(self, metric_field, max_levels=None, node_budget=None,
              builder=None, adapter=None, engine=None, verbose=False,
              relax=False, relax_kwargs=None, repair=False,
              mg_coarsening_ratio=2.0):
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
        engine : {"nvb", "sbr", "edge_split"}, optional
            Advanced selector for the nested refinement engine (ignored when
            ``adapter="mmg"``). Default ``"nvb"`` — graded newest-vertex
            bisection; ``"sbr"`` is longest-edge bisection (uniform patch,
            still the right choice when a uniform-finest MG patch is wanted).
            See above.

            ``"edge_split"`` splits the **longest edge** of every cell coarser
            than the metric asks for, and needs no conforming closure at all
            because splitting an edge divides every cell incident on it at the
            same new vertex. Refinement therefore stays inside the marked region
            instead of a bounded halo around it, at the cost of giving up the
            similarity-class bound that makes bisection shape-safe at arbitrary
            depth. It marks on the cell **diameter** rather than
            ``(dim!·vol)^(1/dim)``; see
            :mod:`underworld3.utilities.edge_split`.
        mg_coarsening_ratio : float
            Target coarsening in cell size `h` between consecutive multigrid
            levels. A refinement engine takes as many passes as it needs to reach
            the size the metric asks for, so a pass is not a level: recording one
            level per pass gives a hierarchy of half-steps with a tail that
            coarsens nothing, which was measured 2.3-7.3x slower than one level
            per doubling at the same iteration count. ``2.0`` (halve `h` each
            level) is the standard choice and the measured default; raise it for
            fewer, cheaper levels or lower it if a problem needs a gentler
            sequence.
        repair : bool, default False
            Run a reconnection (Lawson flip) pass after each ``edge_split``
            generation, repairing the element shapes the split leaves behind. 2-D
            and ``engine="edge_split"`` only. Off by default for one specific
            reason: ``edge_split`` alone produces a **partition-independent** mesh,
            identical at any communicator size, and repair gives that up, because
            the flips it may perform depend on where the partitioner drew the seam
            (a cavity spanning two ranks cannot be flipped). Conformity,
            orientation, volume, labels and the star-forest stay exact at every
            rank count.

            Worth turning on when the base is poor — anisotropic, graded, relaxed
            or read from a file. Measured there: 41 degrees off the 99th-percentile
            maximum angle, slivers below q=0.1 from 3.84 % to 0.00 %, and 20-30 %
            lower interpolation error per degree of freedom. On a well-shaped gmsh
            base it costs a little time and changes little else. It also
            invalidates the cell-parent map used by the any-degree nested MG
            transfer (a flipped cell can straddle two coarse cells), so a degree-2
            or higher space falls back to the geometric prolongation builder; the
            exact vertex prolongation is unaffected because flips move no vertex.
            See :mod:`underworld3.utilities.reconnect`.
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
        if engine not in ("sbr", "nvb", "edge_split"):
            raise ValueError(
                f"engine must be 'sbr', 'nvb' or 'edge_split', got {engine!r}")
        if repair:
            # Refuse rather than silently ignore: a caller asking for repair has a
            # badly shaped mesh, and quietly returning an unrepaired one sends them
            # looking for the problem somewhere else.
            if engine != "edge_split":
                raise ValueError(
                    f"repair=True needs engine='edge_split', got {engine!r}. The "
                    f"bisection engines carry a similarity-class bound that keeps "
                    f"child quality tied to the base, so there is nothing for a "
                    f"flip pass to repair.")
            if self.dim != 2:
                raise NotImplementedError(
                    "repair=True is 2-D only. In 3-D no single flip is enough — "
                    "the operator set has to become quality-gated edge removal. "
                    "See docs/developer/design/"
                    "mesh-reconnection-and-delaunay-adapt.md")

        return self._adapt_nested(
            metric_field, max_levels=max_levels, node_budget=node_budget,
            builder=builder, engine=engine, verbose=verbose,
            relax=relax, relax_kwargs=relax_kwargs, repair=repair,
            mg_coarsening_ratio=mg_coarsening_ratio,
        )

    def _adapt_nested(self, metric_field, max_levels=2, node_budget=None,
                      builder="barycentric", engine="nvb", verbose=False,
                      relax=False, relax_kwargs=None, repair=False,
                      mg_coarsening_ratio=2.0):
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

        def marking_metric(centroids):
            """``eval_metric``, clipped, with "nobody has cells" decided together.

            Returns ``None`` when NO rank owns cells, which the marking loops
            read as "mark nothing". The point of routing the emptiness question
            through a reduction is that the loops below then have a collective
            fact to stop on, instead of each rank deciding for itself whether
            to leave — which is the defect this replaced (#512).

            Note on what this does NOT fix. #512 describes the rank-local
            ``if cur_h.size:`` guards as skipping a collective, on the grounds
            that ``eval_metric`` is ``global_evaluate`` for a field or
            expression metric. Measured at np=2, that is not so: with one rank
            asleep for 10 s, the other's ``global_evaluate`` returned in 0.06 s,
            both for in-domain points and for points that strand outside the
            mesh entirely. It does not wait for its peers on these shapes, so a
            cell-less rank skipping it does not hang. The guards are routed
            through here for uniformity with the stop below, not because they
            were hanging.
            """

            # mpi4py allreduce defaults to MPI.SUM, as at the collective stop
            # further down.
            if uw.mpi.comm.allreduce(int(centroids.shape[0])) == 0:
                return None

            return numpy.clip(eval_metric(centroids), 1e-30, None)

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

        # Analytic boundary surfaces to snap each generation onto. New
        # boundary vertices are CHORD midpoints, so without snapping the
        # boundary geometry of a curved domain stays frozen at base
        # resolution no matter how deep the refinement. Per the 2026-07
        # round-3b ruling, EVERY generation snaps (not just the returned
        # child): each intermediate level is a valid mesh in its own
        # right — extractable for solvers — and the metric marks on true
        # geometry. On plane surfaces (boxes) the chord midpoint already
        # lies in the plane, so the snap is exactly a no-op and the flat
        # confluence gates are untouched.
        # NOTE: surfaces must be disjoint or intersect only where each
        # projection is an exact no-op (concentric radial pairs; orthogonal
        # planes, whose corner vertices are fixed points of both). Sequential
        # restore does not converge to the intersection of two CURVED
        # surfaces — junction handling as in boundary_slip is a follow-up
        # for the day such a mesh registers surfaces.
        _snap_surfs = [s for s in dict(self.bounding_surfaces).values()
                       if getattr(s, "kind", None) in ("radial", "plane")
                       and not getattr(s, "is_free", False)]

        def snap_level_boundaries(dm):
            if not _snap_surfs:
                return
            from underworld3.meshing.smoothing import _pinned_mask

            def sync_mask(mask):
                # In 3D a boundary-edge midpoint can live on a rank whose
                # only cells containing it are INTERIOR members of the
                # edge's star — that rank sees no labelled face and would
                # skip the snap the face-owning rank applies, leaving one
                # global vertex with two coordinates. Reduce the mask over
                # the point SF (ADD ≡ logical-or here) so every rank
                # holding the vertex agrees; pre-snap coordinates are
                # rank-consistent and restore is a pure function of them,
                # so a consistent mask gives a consistent snap.
                if uw.mpi.size == 1:
                    return mask
                from underworld3.meshing.smoothing.graph import (
                    _build_scalar_dm)
                dm_s = _build_scalar_dm(dm)
                lvec = dm_s.createLocalVector()
                gvec = dm_s.createGlobalVector()
                lvec.array[:] = mask.astype(float)
                dm_s.localToGlobal(lvec, gvec, addv=True)
                dm_s.globalToLocal(gvec, lvec)
                out = numpy.asarray(lvec.array) > 0.5
                lvec.destroy(); gvec.destroy(); dm_s.destroy()
                return out

            vec = dm.getCoordinatesLocal()
            arr = vec.array.reshape(-1, self.cdim)
            pre = arr.copy()
            snapped_any = numpy.zeros(arr.shape[0], dtype=bool)
            for s in _snap_surfs:
                mask = sync_mask(_pinned_mask(dm, (s.label,)))
                if mask.any():
                    arr[mask] = s.restore(arr[mask])
                    snapped_any |= mask
            dm.setCoordinatesLocal(vec)
            # A snap moves boundary vertices by the chord sagitta
            # (~h²/8R); on a base coarse enough that h ≈ R this could
            # invert a sliver silently. Fail loudly instead: no cell
            # incident to a snapped vertex may flip orientation.
            flipped = 0
            if snapped_any.any():
                cs_, ce_ = dm.getHeightStratum(0)
                vS_, vE_ = dm.getDepthStratum(0)
                for c in range(cs_, ce_):
                    vs = [p - vS_ for p in dm.getTransitiveClosure(c)[0]
                          if vS_ <= p < vE_]
                    if not any(snapped_any[v] for v in vs):
                        continue
                    e0 = pre[vs[1:]] - pre[vs[0]]
                    e1 = arr[vs[1:]] - arr[vs[0]]
                    if (numpy.sign(numpy.linalg.det(e0))
                            != numpy.sign(numpy.linalg.det(e1))):
                        flipped += 1

            # The count is rank-local, so the REDUCTION must be reached
            # unconditionally: a rank whose partition holds no vertex on a
            # registered surface has an all-False `snapped_any`, and guarding
            # the reduction on it starves the peers that are already in it
            # (#627).
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                flipped = uw.mpi.comm.allreduce(flipped, op=_MPI.SUM)
            if flipped:
                raise RuntimeError(
                    f"adapt: snapping boundary vertices to the analytic "
                    f"surfaces inverted {flipped} cell(s) — the base mesh "
                    "is too coarse for the boundary curvature (chord "
                    "sagitta ~ cell size). Refine the base mesh or adapt "
                    "without registered bounding surfaces.")

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
        # relax=True is the RECOMMENDED default: relax once, at the end.
        # relax="per-generation" relaxes inside the refinement loop instead.
        # Both measure better than no relaxation and neither dominates the
        # other across the properties we care about (element quality, size
        # uniformity along the feature, and refinement leakage away from
        # it), so this is a genuine choice rather than a ranking.
        if relax in (False, None):
            _relax_mode = None
        elif relax is True:
            _relax_mode = "end"
        elif str(relax).lower().replace("_", "-") in ("end", "at-end"):
            _relax_mode = "end"
        elif str(relax).lower().replace("_", "-") in ("per-generation",
                                                      "generation"):
            _relax_mode = "per-generation"
        else:
            raise ValueError(
                f"adapt(relax={relax!r}); use True (relax once at the end, "
                f"the recommended default), 'per-generation' (relax inside "
                f"the refinement loop) or False.")

        # The metric handed to the relaxation, resolved ONCE. A callable
        # (numpy) metric cannot go to the mover, so those relax in the pure
        # shape frame at fixed size; a sympy / MeshVariable metric gives the
        # ideal-metric frame. NB `relax` is a MODE, never a metric.
        _relax_metric = None if callable(metric_field) else metric_field

        def _relax_generation(engine_obj, carry, rcarry):
            """Relax THIS generation in place INSIDE the refinement engine.

            The engine's own coordinates feed the next generation's marking
            and its next midpoints, so a relaxation that only touched the
            exported DM would be discarded by the following generation (the
            same reason the boundary snap is applied inside the engine).

            Coordinates are matched by POSITION, not by index: ``to_dm``
            goes through ``createFromCellList`` and PETSc renumbers the
            vertices. The match is exact because it is taken before
            anything moves.
            """
            _dmg = engine_obj.to_dm(boundaries=carry, regions=rcarry,
                                    comm=self.dm.comm)
            _mg = Mesh(_dmg, simplex=self.dm.isSimplex(),
                       coordinate_system_type=(
                           self.CoordinateSystem.coordinate_type),
                       qdegree=self.qdegree, boundaries=self.boundaries,
                       verbose=False)
            _cd = _mg.cdim
            _pre = numpy.ascontiguousarray(
                _mg.dm.getCoordinatesLocal().array.reshape(-1, _cd))
            _src = numpy.ascontiguousarray(
                numpy.asarray([numpy.asarray(c, dtype=float)
                               for c in engine_obj.coords]))
            # EXACT identification: the DM was built from these very
            # coordinates, so byte equality is the correct test. No spatial
            # index, no tolerance, no nearest-neighbour guess.
            _key = {row.tobytes(): i for i, row in enumerate(_src)}
            _idx = numpy.asarray([_key[row.tobytes()] for row in _pre],
                                 dtype=numpy.int64)
            _mg.relax(_relax_metric, **(relax_kwargs or {}))
            _post = _mg.dm.getCoordinatesLocal().array.reshape(-1, _cd)
            for _k, _i in enumerate(_idx):
                engine_obj.coords[_i] = _post[_k]

        level_dms = []                       # one DM per refinement level
        # Nested (topological) MG prolongations, one per refinement generation.
        # `from_dm` numbers engine vertices as `DM point - vS`, so the base map
        # is that offset; each generation's map comes back from `to_dm`.
        _nested_Ps = []
        _nested_parent_cells = []
        _vS0, _vE0 = base_finest.getDepthStratum(0)
        _coarse_vmap = {i: _vS0 + i for i in range(_vE0 - _vS0)}

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
                M = marking_metric(centroids)
                if M is not None and cur_h.size:
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
                _coarse_for_P = current_dm
                current_dm = _nvbx.refine(d, "adapt")
                # Capture the exact parent/child prolongation NOW, in the one
                # window where the coordinates are still pristine: the snap
                # below moves boundary midpoints off their parent edges, and
                # relaxation moves everything, after which the relation can no
                # longer be recovered by matching. See #425.
                from underworld3.utilities.nvb import (
                    nested_prolongation_from_dms as _nested_from_dms,
                    nested_cell_parents as _nested_parents)
                _vP = _nested_from_dms(_coarse_for_P, current_dm)
                _nested_Ps.append(_vP)
                # Parent CELL map as well: with it a fine DOF's weights are the
                # coarse basis evaluated inside its parent, which is exact at
                # ANY degree — the vertex map alone only covers P1. (#425)
                _nested_parent_cells.append(
                    None if _vP is None
                    else _nested_parents(_coarse_for_P, current_dm, _vP))
                snap_level_boundaries(current_dm)
                if _relax_mode == "per-generation":
                    # Relax THIS generation before the next one marks from it:
                    # the moved coordinates are what the next pass measures and
                    # bisects, which is the whole point of relaxing in the loop
                    # rather than once at the end. A clone keeps the plex
                    # numbering, so the coordinate vectors align index-for-index
                    # and no positional matching is needed.
                    _mg = Mesh(current_dm.clone(),
                               simplex=self.dm.isSimplex(),
                               coordinate_system_type=(
                                   self.CoordinateSystem.coordinate_type),
                               qdegree=self.qdegree,
                               boundaries=self.boundaries, verbose=False)
                    _mg.relax(_relax_metric, **(relax_kwargs or {}))
                    current_dm.setCoordinatesLocal(
                        _mg.dm.getCoordinatesLocal())
                level_dms.append(current_dm)
                if verbose:
                    fs, fe = current_dm.getHeightStratum(0)
                    uw.pprint(0, f"[adapt] nvb pass {level}: marked {len(marked)} "
                                 f"-> {fe - fs} cells (rank-local)")
            if not level_dms:
                current_dm = base_finest.clone()
        elif engine == "edge_split":
            # Longest-edge refinement with NO conforming closure: splitting an
            # edge divides every cell incident on it at the same new vertex, so
            # there is no hanging node to repair and refinement cannot escape the
            # marked region. Marking is on the cell DIAMETER, not (dim!·vol)^(1/dim)
            # — for bisection the two shrink together, but this engine shortens
            # the longest edge directly and the volume proxy would report the
            # target met while the mesh is still coarse across the feature.
            from underworld3.utilities import edge_split
            # Independence caps a pass (no cell may carry two split edges), so a
            # generation satisfies only some marked cells and the loop re-marks.
            # 3D needs more passes than 2D: an edge is shared by more cells there,
            # so fewer edges are independent per pass.
            n_pass = 8 * dim * max_levels
            current_dm = base_finest
            _pct5_of_dm = {}
            for level in range(n_pass):
                centroids, _proxy_h, cs = cell_geometry(current_dm)
                # The topology tables are read ONCE per pass and shared
                # with the split — the per-cell closure walk is the
                # loop's dominant cost and was paid twice (#610). The
                # metric goes through marking_metric: the "nobody has
                # cells" verdict is COLLECTIVE (rank-local branching on
                # the raw metric is the np>1 deadlock class).
                _edge_tables = None
                M = marking_metric(centroids)
                if M is not None and centroids.shape[0]:
                    h_target = 1.0 / numpy.sqrt(M)
                    diameter, _edge_tables = edge_split.cell_diameters(
                        current_dm, return_tables=True)
                    # The marking pass has just measured this dm; the MG
                    # level selection re-derives the same 5th percentile
                    # per retained generation, so it is cached here
                    # rather than re-walking every level's topology.
                    _pct5_of_dm[id(current_dm)] = (
                        float(numpy.percentile(diameter, 5))
                        if diameter.size else float("inf"))
                    sel = numpy.where(diameter > h_target)[0]
                    if node_budget is not None and sel.size > node_budget:
                        order = numpy.argsort(M[sel])[::-1]
                        sel = sel[order[:node_budget]]
                else:
                    sel = numpy.empty(0, dtype=int)   # rank owns no cells

                marked = [int(cs + j) for j in sel]
                _coarse_for_P = current_dm
                current_dm, n_split = edge_split.bisect_longest_edges(
                    current_dm, marked, tables=_edge_tables)
                # n_split is global, so this stop is collective without a further
                # reduction — a rank with nothing marked still enters the split.
                if n_split == 0:
                    if verbose:
                        uw.pprint(0, f"[adapt] edge_split pass {level}: "
                                     f"nothing to refine")
                    break
                markers_per_level.append(marked)
                # Every inserted vertex is the exact float midpoint of a parent
                # edge, so the exact parent/child prolongation applies unchanged.
                # Capture it BEFORE the snap and any relaxation move it out of
                # reach of coordinate matching (#425).
                from underworld3.utilities.nvb import (
                    nested_prolongation_from_dms as _nested_from_dms,
                    nested_cell_parents as _nested_parents)
                _vP = _nested_from_dms(_coarse_for_P, current_dm)
                _nested_Ps.append(_vP)
                if repair:
                    # Reconnection repairs the cells the split left thin. It
                    # rebuilds the DM on the SAME point chart, so the vertex
                    # prolongation just captured stays valid (a P1 section numbers
                    # DOFs from the point numbering, which is preserved) — but the
                    # cell-parent map does not: a flipped cell can straddle two
                    # coarse cells, so the any-degree transfer has to fall back to
                    # the geometric builder.
                    from underworld3.utilities import reconnect
                    current_dm, n_flips = reconnect.flip_to_reduce_max_angle(
                        current_dm)
                    _nested_parent_cells.append(None)
                    if verbose:
                        uw.pprint(0, f"[adapt] edge_split pass {level}: repaired "
                                     f"with {n_flips} flip(s)")
                else:
                    # Parent maps are DEFERRED to the retained MG levels:
                    # the subsampler discarded every per-pass map whose
                    # span was more than one pass, so building ~76 of
                    # them to keep 3-4 was almost entirely wasted work —
                    # and the retained multi-pass spans now get EXACT
                    # parents from the composed vertex transfer instead
                    # of falling back to the geometric builder.
                    _nested_parent_cells.append(None)
                snap_level_boundaries(current_dm)
                if _relax_mode == "per-generation":
                    _mg = Mesh(current_dm.clone(),
                               simplex=self.dm.isSimplex(),
                               coordinate_system_type=(
                                   self.CoordinateSystem.coordinate_type),
                               qdegree=self.qdegree,
                               boundaries=self.boundaries, verbose=False)
                    _mg.relax(_relax_metric, **(relax_kwargs or {}))
                    current_dm.setCoordinatesLocal(
                        _mg.dm.getCoordinatesLocal())
                level_dms.append(current_dm)
                if verbose:
                    fs, fe = current_dm.getHeightStratum(0)
                    uw.pprint(0, f"[adapt] edge_split pass {level}: split "
                                 f"{n_split} edge(s) -> {fe - fs} cells "
                                 f"(rank-local)")
            else:
                # Ran out of passes with cells still coarser than the metric.
                # Silence here would look like a satisfied size field.
                uw.pprint(0, f"[adapt] edge_split: stopped at the {n_pass}-pass "
                             f"cap with the metric not yet satisfied; raise "
                             f"max_levels if the feature needs to be finer.")
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
            from underworld3.utilities.nvb import (NVBMesh, TaggedBisectionMesh,
                                                   nested_prolongation)
            _Engine = TaggedBisectionMesh if self.dim == 3 else NVBMesh
            carry = [(b.name, b.value) for b in self.boundaries
                     if b.name not in ("Null_Boundary", "All_Boundaries")]
            rcarry = ([(r.name, r.value) for r in self.regions]
                      if self.regions is not None else [])
            nvb = _Engine.from_dm(base_finest, boundaries=carry,
                                  regions=rcarry)
            n_gen = dim * max_levels
            for level in range(n_gen):
                # Rank-local marking and a rank-local break, deliberately: this
                # is the pure-Python cell-list engine, reached only when the
                # native uwnvb transform is absent, and that combination raises
                # NotImplementedError at np>1 further up. It is serial by
                # construction, so the collective discipline the other engines
                # need here (#512) has nothing to protect. Port it before
                # porting this loop.
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
                _n_coarse_verts = len(nvb.coords)   # before this generation
                nvb.refine(set(marked))
                # Snap INSIDE the engine: its own coordinates feed the next
                # generation's marking AND the next midpoints, so snapping
                # only the exported DM would leave the engine's geometry on
                # the chords.
                if _snap_surfs:
                    _val2surf = {v: s for s in _snap_surfs
                                 for (nm, v) in carry if nm == s.label}
                    _sv = {}
                    for _fkey, _val in nvb.facet_label.items():
                        _s = _val2surf.get(_val)
                        if _s is not None:
                            _sv.setdefault(id(_s), (_s, set()))[1].update(_fkey)
                    for _s, _vs in _sv.values():
                        _idx = numpy.fromiter(_vs, dtype=numpy.int64)
                        _snapped = _s.restore(
                            numpy.array([nvb.coords[i] for i in _idx]))
                        for _k, _i in enumerate(_idx):
                            nvb.coords[_i] = _snapped[_k]
                if _relax_mode == "per-generation":
                    _relax_generation(nvb, carry, rcarry)
                _gen_dm = nvb.to_dm(boundaries=carry, regions=rcarry,
                                    comm=self.dm.comm)
                level_dms.append(_gen_dm)
                # Record the EXACT prolongation for this generation while the
                # engine still knows the parent/child relation. Built from
                # `edge2mid`, not by point location, so it is full rank by
                # construction and survives any later node motion (relax /
                # snap / deformation). See design/nested-vs-geometric-mg-transfers.
                _vSg, _vEg = _gen_dm.getDepthStratum(0)
                _fine_map = dict(nvb.dm_vertex_of_engine)
                _nested_Ps.append(nested_prolongation(
                    nvb, _coarse_vmap, _fine_map, _n_coarse_verts,
                    _vSg, _vEg - _vSg))
                _coarse_vmap = _fine_map
                if verbose:
                    uw.pprint(0, f"[adapt] nvb gen {level}: marked {len(marked)} "
                                 f"-> {len(nvb.cells)} cells")
            current_dm = level_dms[-1] if level_dms else base_finest.clone()
        else:
            current_dm = base_finest             # base finest, current geometry
            for level in range(max_levels):
                centroids, cur_h, cs = cell_geometry(current_dm)

                # Metric M = 1/h_target² at the cell centroids (parent field).
                # A callable is evaluated directly on the centroids; a field/expr
                # goes through global_evaluate (parallel) or evaluate (serial).
                M = marking_metric(centroids)
                if M is None:
                    break                     # no rank has cells: leaving together

                if cur_h.size:
                    h_target = 1.0 / numpy.sqrt(M)
                    refine = numpy.where(cur_h > h_target)[0]
                else:
                    refine = numpy.empty(0, dtype=int)

                # Collective stop. `refine.size == 0` is a rank-local fact: this
                # rank's cells are all fine enough, which says nothing about its
                # peers'. Breaking on it alone takes the rank out of the loop for
                # good while the others go round again into the DM refinement,
                # which IS collective — so the job hangs, and not latently.
                # Measured at np=2 with a callable metric demanding h=0.01 inside
                # r<0.15 of one corner and nothing elsewhere: the rank holding no
                # corner cells left at the first level and `mpirun` reached its
                # timeout with neither rank returning. See test_0873.
                if uw.mpi.comm.allreduce(int(refine.size)) == 0:
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
                snap_level_boundaries(current_dm)
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

        if _relax_mode == "end":
            # A sympy/MeshVariable metric gives the ideal-metric frame (the
            # metric sets size); a plain callable cannot be handed to the
            # mover, so those fall back to pure shape repair at fixed size.
            child.relax(_relax_metric, **(relax_kwargs or {}))

        # Lineage (parent/child DAG) and mesh-owned custom-P hierarchy.
        child.parent = self
        child._relationship_kind = "refinement"
        child.regions = self.regions
        child._parent_mesh_version = self._mesh_version
        # Markers per refinement level (in each level's cell numbering) — the
        # checkpoint-by-marker payload (design only; storage is a follow-up).
        child._adapt_markers = markers_per_level
        child._adapt_engine = engine
        # One multigrid level per DOUBLING OF RESOLUTION, not one per engine
        # pass. This has to happen BEFORE the prolongations are recorded on the
        # child: custom_mg indexes that list BY LEVEL, so a per-pass list against
        # a subsampled hierarchy lines the transfers up against the wrong levels.
        if level_dms:
            level_dms, _nested_Ps, _nested_parent_cells = self._subsample_mg_levels(
                base_finest, level_dms, _nested_Ps, _nested_parent_cells,
                ratio=mg_coarsening_ratio, verbose=verbose,
                resolution_hint=(_pct5_of_dm if engine == "edge_split"
                                 else None))

        # Exact per-generation prolongations when the engine could supply them
        # (cell-list path). Empty for the native transform path, which falls
        # back to the geometric builder. See #425.
        child._adapt_prolongation = _nested_Ps
        child._adapt_parent_cells = _nested_parent_cells
        # Mesh-owned custom-P geometric-MG tail: the tail is
        #   [base L0 … base finest]  +  [refine level 1 … refine level n-1]
        # and the solver appends its own mesh (the finest level = child). Each
        # intermediate level is wrapped here (transient, lives on the child); the
        # static base levels reuse the parent's cached wraps. (NVB snapshots and
        # SBR levels are both just coordinate sets — the custom-P transfers are
        # coordinate-based, so the hierarchy is engine-agnostic.)
        intermediate = [
            self._wrap_coarse_level(d) for d in level_dms[:-1]
        ]
        # A mesh that is ITSELF a child — an adapt child, or one carrying a
        # conforming surface — owns its tail; `dm_hierarchy` for such a mesh holds
        # only its own DM, so reading it here discards every level below and
        # leaves a two-level hierarchy calling itself multigrid. Tested with
        # `is not None`: an empty own-tail is still an own-tail.
        own_tail = getattr(self, "_custom_mg_coarse_meshes", None)
        coarse_tail = (list(own_tail) + [self]) if own_tail is not None \
            else self._coarse_level_meshes()
        if _moved:
            # the finest base level of the MG tail must carry the SAME moved
            # geometry the child was refined from; coarser levels keep their
            # original coordinates (custom-P transfers accept non-nested pairs)
            coarse_tail = coarse_tail[:-1] + [self._wrap_coarse_level(base_finest)]
        child._custom_mg_coarse_meshes = coarse_tail + intermediate
        child._custom_mg_builder = builder

        self._registered_children.add(child)
        return child

    _MG_RATIO_SLACK = 0.9      # a step of 1.92 counts as a doubling

    def _subsample_mg_levels(self, base_finest, level_dms, nested_Ps,
                             nested_parent_cells, ratio=2.0, verbose=False,
                             resolution_hint=None):
        """Keep one multigrid level per DOUBLING OF RESOLUTION, not one per pass.

        A refinement engine takes as many passes as it needs to reach the size
        the metric asks for — independence caps how many edges one pass may
        split, and a conforming closure cascades — so a pass is an implementation
        detail of *reaching* a size, while a multigrid level is a *coarsening
        ratio*. Recording one level per pass conflates them, and the tail of the
        iteration becomes levels that coarsen nothing: measured, ``edge_split``
        produced ten levels whose last three grew the mesh by 4 %, 1 % and 0.7 %,
        each costing a full Galerkin RAP and smoother sweep for no correction,
        and that hierarchy stopped SolCx converging at all.

        **The measure is resolution, not element count.** Under adapt-on-top the
        mesh only grows where the feature is, so a genuine halving of `h` shows up
        as a global cell-count ratio near 1: measured on a thin band, NVB grew the
        mesh by 1.06-1.11x per generation while the in-band `h` went 0.125 ->
        0.0626 -> 0.0313 -> 0.0157. A count-based rule keeps nothing and collapses
        the hierarchy; the whole-mesh median `h` is likewise flat and useless. So
        the resolution of the refined region is what decides a level.

        The exact per-generation prolongations are COMPOSED across the generations
        a level skips, so the recorded transfer stays exact rather than falling
        back to the geometric builder.

        A level's parent-cell map survives only if that level is ONE generation.
        A composed span crosses several, so a cell no longer has a single parent
        and the any-degree transfer has to fall back to the geometric builder —
        but that is a property of the individual level, not of the call. Dropping
        every map whenever any subsampling happened discards maps that are still
        valid, and it tautologised the test that told ``repair=True`` from
        ``repair=False``.
        """
        from underworld3.utilities import edge_split

        def resolution(dm):
            """The size of the cells this level actually resolves with.

            A low percentile rather than the strict minimum, so one thin cell
            cannot declare a level; reduced with MIN so the finest region counts
            wherever it happens to live. A caller that already measured a
            dm during its own pass loop supplies the value through
            ``resolution_hint`` instead of paying a second topology walk.
            """
            if resolution_hint is not None and id(dm) in resolution_hint:
                local = resolution_hint[id(dm)]
            else:
                d = edge_split.cell_diameters(dm)
                local = (float(numpy.percentile(d, 5)) if d.size
                         else float("inf"))
            return uw.mpi.comm.allreduce(local, op=min)

        # An engine lands near the target, not on it (1.92, 1.97, 2.19 measured),
        # so the test is against a slightly slack ratio; without it a 1.99 step is
        # rejected and two real levels fuse into one.
        threshold = ratio * self._MG_RATIO_SLACK
        h_ref = resolution(base_finest)
        keep = []
        for i, dm in enumerate(level_dms):
            h = resolution(dm)
            if h <= h_ref / threshold:
                keep.append(i)
                h_ref = h
        # The finest generation IS the child, so it is always a level. If the
        # level below it is within `ratio`, that level is a near-duplicate of the
        # child rather than a coarsening of it, and REPLACING it is right —
        # appending would reintroduce exactly the pair this routine exists to
        # remove (measured: a last ratio of 1.04).
        last = len(level_dms) - 1
        if not keep:
            keep = [last]
        elif keep[-1] != last:
            if resolution(level_dms[keep[-1]]) <= resolution(level_dms[last]) * threshold:
                keep[-1] = last
            else:
                keep.append(last)

        composed, parent_cells = [], []
        start = 0
        level_coarse = base_finest
        for i in keep:
            span = [P for P in nested_Ps[start:i + 1]]
            if any(P is None for P in span) or not span:
                composed.append(None)
            elif len(span) == 1:
                composed.append(span[0])
            else:
                # x_fine = P_i ... P_start x_coarse, so the product runs
                # fine-most first.
                M = span[0]
                for P in span[1:]:
                    M = _compose_prolongations(P, M)
                composed.append(M)
            # The parent-cell map, for the RETAINED pair only. A map the
            # engine recorded per pass (a single-pass span) is used as
            # recorded; otherwise it is derived from the composed vertex
            # transfer — nested_cell_parents is topological through the
            # transfer, and a descendant's referenced coarse vertices are
            # all corners of its ancestor at any depth, so multi-pass
            # spans now carry exact parents instead of None.
            recorded = (nested_parent_cells[i]
                        if i == start and i < len(nested_parent_cells)
                        else None)
            if recorded is not None:
                parent_cells.append(recorded)
            elif composed[-1] is not None:
                from underworld3.utilities.nvb import (
                    nested_cell_parents as _parents_of)
                parent_cells.append(
                    _parents_of(level_coarse, level_dms[i], composed[-1]))
            else:
                parent_cells.append(None)
            level_coarse = level_dms[i]
            start = i + 1

        if verbose:
            uw.pprint(f"[adapt] {len(level_dms)} engine pass(es) -> "
                      f"{len(keep)} multigrid level(s) "
                      f"(kept {keep}, one per {ratio:.2g}x in h)")
        return [level_dms[i] for i in keep], composed, parent_cells

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
