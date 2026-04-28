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

    if uw.mpi.rank == 0:
        plex_0 = PETSc.DMPlex().createFromFile(filename, interpolate=True, comm=PETSc.COMM_SELF)

        plex_0.setName("uw_mesh")
        plex_0.markBoundaryFaces("All_Boundaries", 1001)

        viewer = PETSc.ViewerHDF5().create(filename + ".h5", "w", comm=PETSc.COMM_SELF)
        viewer(plex_0)
        viewer.destroy()

        # ## Now add some metadata to the mesh (not sure how to do this with the Viewer)

        # import h5py, json

        # f = h5py.File('filename + ".h5",'r+')

        # boundaries_dict = {i.name: i.value for i in cs_mesh.boundaries}
        # string_repr = json.dumps(boundaries_dict)

        # g = f.create_group("metadata")
        # g.attrs["boundaries"] = string_repr

        # f.close()

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
        Physical units for mesh coordinates.
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

        # Get coordinate units from model (not user parameter)
        # The model owns the unit system - all meshes use the same units
        import underworld3 as uw

        model = uw.get_default_model()

        # Ignore user-provided units parameter, get from model instead
        if units is not None and units != model.get_coordinate_unit():
            import warnings

            warnings.warn(
                f"Ignoring units parameter '{units}'. Mesh coordinates will use "
                f"model units '{model.get_coordinate_unit()}'. The model owns the "
                "unit system to ensure consistency across all meshes and variables.",
                UserWarning,
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
        self._length_scale = 1.0  # Default: no scaling
        self._length_units = (
            self.units if self.units else "dimensionless"
        )  # Same as coordinate units

        # Derive length scale from model reference quantities if available
        if hasattr(model, "_reference_quantities") and model._reference_quantities:
            # Priority order: domain_depth > length
            if "domain_depth" in model._reference_quantities:
                ref_qty = model._reference_quantities["domain_depth"]
                # Convert to base units (SI: meters) for consistent scaling
                try:
                    base_qty = ref_qty.to_base_units()
                    self._length_scale = float(base_qty.magnitude)
                    self._length_units = str(base_qty.units)
                except:
                    # Fallback if to_base_units() fails
                    self._length_scale = float(ref_qty.magnitude)
                    self._length_units = (
                        str(ref_qty.units) if hasattr(ref_qty, "units") else "dimensionless"
                    )
            elif "length" in model._reference_quantities:
                ref_qty = model._reference_quantities["length"]
                # Convert to base units (SI: meters) for consistent scaling
                try:
                    base_qty = ref_qty.to_base_units()
                    self._length_scale = float(base_qty.magnitude)
                    self._length_units = str(base_qty.units)
                except:
                    # Fallback if to_base_units() fails
                    self._length_scale = float(ref_qty.magnitude)
                    self._length_units = (
                        str(ref_qty.units) if hasattr(ref_qty, "units") else "dimensionless"
                    )

        # Mesh coordinate version tracking for swarm coordination
        self._mesh_version = 0
        self._registered_swarms = weakref.WeakSet()
        self._registered_surfaces = weakref.WeakSet()  # Surfaces using this mesh
        self._registered_submeshes = weakref.WeakSet()  # Submeshes from extract_region

        # _mesh_update_lock: Re-entrant lock to coordinate mesh deformation.
        # Held by mesh_update_callback during _deform_mesh(). Checked by
        # MeshVariable callbacks (blocking=False) to skip PETSc sync during
        # sensitive coordinate changes.
        self._mesh_update_lock = threading.RLock()

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

                # boundaries_dict = {i.name: i.value for i in cs_mesh.boundaries}
                # string_repr = json.dumps(boundaries_dict)

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

                # Do not call setFromOptions() here. DMPlexTopologyLoad()
                # returns the topology SF needed to reload checkpoint fields.
                # setFromOptions() can repartition/reorder the DM before UW
                # composes that SF with any later redistribution SF, leaving
                # checkpoint field reloads mapped to stale point numbering.

            else:
                raise RuntimeError(
                    "Mesh file %s has unknown format '%s'." % (plex_or_meshfile, ext[1:])
                )

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
        self.boundary_normals = boundary_normals
        self.regions = regions
        self.parent = None       # Set by extract_region() for submeshes
        self.subpoint_is = None  # IS mapping submesh points -> parent points

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

        # options.delValue("dm_plex_gmsh_mark_vertices")
        # options.delValue("dm_plex_gmsh_multiple_tags")
        # options.delValue("dm_plex_gmsh_use_regions")
        #

        # Only for newly created dm (from mesh files)
        # self.dm.setFromOptions()

        # uw.adaptivity._dm_stack_bcs(self.dm, self.boundaries, "UW_Boundaries")

        all_edges_label_dm = self.dm.getLabel("depth")
        if all_edges_label_dm:
            all_edges_IS_dm = all_edges_label_dm.getStratumIS(0)
            # all_edges_IS_dm.view()

        self.dm.createLabel("Null_Boundary")
        all_edges_label = self.dm.getLabel("Null_Boundary")
        if all_edges_label and all_edges_IS_dm:
            all_edges_label.setStratumIS(boundaries.Null_Boundary.value, all_edges_IS_dm)

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

        if not refinement is None and refinement > 0:

            self.dm.setRefinementUniform()

            if not self.dm.isDistributed():
                self.sf1 = self.dm.distribute()

            # self.dm_hierarchy = self.dm.refineHierarchy(refinement)

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

            # self.dm_hierarchy = [self.dm] + self.dm_hierarchy

            self.dm_h = self.dm_hierarchy[-1]
            self.dm_h.setName("uw_hierarchical_dm")

            # Is this needed here, after the above calls ?
            if callable(refinement_callback):
                for dm in self.dm_hierarchy:
                    refinement_callback(dm)

            # Single level equivalent dm (needed for aux vars ?? Check this - LM)
            self.dm = self.dm_h.clone()

        elif not coarsening is None and coarsening > 0:

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
            # self.dm_hierarchy[0].view()

        else:
            if not self.dm.isDistributed():
                self.sf1 = self.dm.distribute()

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

        # Expose mesh points through special numpy array class with a callback
        # on all setter operations

        self._coords = uw.utilities.NDArray_With_Callback(
            numpy.ndarray.view(self.dm.getCoordinatesLocal().array.reshape(-1, self.cdim)),
            owner=self,
        )

        # The callback is to rebuild the mesh data structures - we already have a routine
        # to handle that so we just wrap it here.

        def mesh_update_callback(array, change_context):
            mesh = array.owner
            if mesh is None:
                # This guard handles cases where the array is accessed during
                # object teardown (e.g. at application exit or during mesh
                # replacement), where the owning Python mesh object has already
                # been garbage collected but the NDArray proxy still exists.
                return

            if verbose:
                uw.pprint(0, f"Mesh update callback - mesh deform")

            coords = array.reshape(-1, mesh.cdim)
            mesh._deform_mesh(coords, verbose=verbose)

            # Increment mesh version to notify registered swarms of coordinate changes
            with mesh._mesh_update_lock:
                mesh._mesh_version += 1
                if verbose:
                    uw.pprint(0, f"Mesh version incremented to {mesh._mesh_version}")

            return

        self._coords.add_callback(mesh_update_callback)

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
        # The adapt op (smooth_mesh_interior / OT_adapt / follow_metric)
        # fires these after the generic per-variable REMAP pass; see
        # discretisation/remesh.py.
        self._remesh_hooks = []

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
        X = np.asarray(
            dm.getCoordinatesLocal().array).reshape(-1, cdim)

        def _reduce(val, op):
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                return uw.mpi.comm.allreduce(
                    val, op=getattr(_MPI, op))
            return val

        tris = []
        is_simplex2d = cdim == 2
        if is_simplex2d:
            for cc in range(cStart, cEnd):
                cl = dm.getTransitiveClosure(cc)[0]
                vs = [p - pStart for p in cl
                      if pStart <= p < pEnd]
                if len(vs) != 3:
                    is_simplex2d = False
                    break
                tris.append(vs)

        if not is_simplex2d or not tris:
            try:
                vol = np.abs(np.array(
                    [dm.computeCellGeometryFVM(cc)[0]
                     for cc in range(cStart, cEnd)]))
            except Exception:
                vol = np.array([1.0])
            if not vol.size:
                vol = np.array([1.0])
            n = _reduce(int(vol.size), "SUM")
            vmin = _reduce(float(vol.min()), "MIN")
            vsum = _reduce(float(vol.sum()), "SUM")
            res = dict(
                n_cells=n, element="non-2D-simplex",
                vol_min_over_mean=vmin / (vsum / max(n, 1)),
                shape_metrics=None,
                note="shape quality / angle / aspect need a 2-D "
                     "triangle mesh; only volume spread reported")
            if per_cell:
                res["per_cell"] = dict(volume=vol)
            return res

        tri = np.asarray(tris, dtype=np.int64)
        v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        a = np.linalg.norm(v1 - v2, axis=1)
        b = np.linalg.norm(v2 - v0, axis=1)
        cl_ = np.linalg.norm(v0 - v1, axis=1)
        A = np.maximum(
            0.5 * np.abs(np.cross(v1 - v0, v2 - v0)), 1.0e-300)
        q = 4.0 * np.sqrt(3.0) * A / (a * a + b * b + cl_ * cl_)

        def _ang(o, p, r):
            return np.degrees(np.arccos(np.clip(
                (p * p + r * r - o * o) / (2.0 * p * r),
                -1.0, 1.0)))
        ang = np.maximum.reduce(
            [_ang(a, b, cl_), _ang(b, cl_, a), _ang(cl_, a, b)])
        Lmax = np.maximum.reduce([a, b, cl_])
        aspect = Lmax * Lmax / (2.0 * A)
        rel = A / A.mean()

        et = {}
        for ti, (i, j, k) in enumerate(tri):
            for u, w in ((i, j), (j, k), (k, i)):
                et.setdefault((min(u, w), max(u, w)),
                              []).append(ti)
        jr = np.array([max(A[t]) / min(A[t])
                       for t in et.values() if len(t) == 2]
                      or [1.0])

        n = _reduce(int(tri.shape[0]), "SUM")
        qsum = _reduce(float(q.sum()), "SUM")
        Asum = _reduce(float(A.sum()), "SUM")
        res = dict(
            n_cells=n, element="2D-simplex",
            q_min=_reduce(float(q.min()), "MIN"),
            q_mean=qsum / max(n, 1),
            q_p01=float(np.percentile(q, 1)),
            q_p05=float(np.percentile(q, 5)),
            n_q_lt_0p3=_reduce(int((q < 0.3).sum()), "SUM"),
            n_q_lt_0p2=_reduce(int((q < 0.2).sum()), "SUM"),
            angle_max_deg=_reduce(float(ang.max()), "MAX"),
            n_angle_gt_150=_reduce(int((ang > 150).sum()), "SUM"),
            n_angle_gt_165=_reduce(int((ang > 165).sum()), "SUM"),
            aspect_max=_reduce(float(aspect.max()), "MAX"),
            aspect_p99=float(np.percentile(aspect, 99)),
            sizejump_max=float(jr.max()),
            sizejump_p99=float(np.percentile(jr, 99)),
            n_big_thin=_reduce(
                int(((rel > 2.0) & (aspect > 4.0)).sum()), "SUM"),
            vol_min_over_mean=(_reduce(float(A.min()), "MIN")
                               / (Asum / max(n, 1))))
        if per_cell:
            res["per_cell"] = dict(
                q=q, angle_deg=ang, aspect=aspect, volume=A)
        return res

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

            ## Boundary information

            if len(self.boundaries) > 0:
                uw.pprint(f"| Boundary Name            | ID    |")
                uw.pprint(f"| -------------------------------- |")
            else:
                uw.pprint(f"No boundary labels are defined on the mesh\n")

            for bd in self.boundaries:
                l = self.dm.getLabel(bd.name)
                if l:
                    i = l.getStratumSize(bd.value)
                else:
                    i = 0

                ii = uw.utilities.gather_data(np.array([i]), dtype="int")

                uw.pprint(f"| {bd.name:<20}     | {bd.value:<5} |")

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            uw.pprint(f"| {'All_Boundaries':<20}     | 1001  |")

            ## UW_Boundaries:
            l = self.dm.getLabel("UW_Boundaries")
            i = 0
            if l:
                for bd in self.boundaries:
                    i += l.getStratumSize(bd.value)

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            uw.pprint(f"| {'UW_Boundaries':<20}     | --    |")

            uw.pprint(f"| -------------------------------- |")
            uw.pprint("\n")

            ## Information on the mesh DM
            # self.dm.view()
            print(f"Use view(1) to view detailed mesh information.\n")

        elif level == 1:
            if uw.mpi.rank == 0:
                print(f"\n")
                print(f"Mesh # {self.instance}: {self.name}\n")
                uw.visualisation.plot_mesh(self)

                # Total number of cells
                nstart, nend = self.dm.getHeightStratum(0)
                num_cells = nend - nstart
                print(f"Number of cells: {num_cells}\n")

                if len(self.vars) > 0:
                    print(f"| Variable Name       | component | degree |     type        |")
                    print(f"| ---------------------------------------------------------- |")
                    for vname in self.vars.keys():
                        v = self.vars[vname]
                        print(
                            f"| {v.clean_name:<20}|{v.num_components:^10} |{v.degree:^7} | {v.vtype.name:^15} |"
                        )

                    print(f"| ---------------------------------------------------------- |")
                    print("\n", flush=True)
                else:
                    print(f"No variables are defined on the mesh\n", flush=True)

            ## Boundary information

            if len(self.boundaries) > 0:
                uw.pprint(f"| Boundary Name            | ID    | Min Size | Max Size |")
                uw.pprint(f"| ------------------------------------------------------ |")
            else:
                uw.pprint(f"No boundary labels are defined on the mesh\n")

            for bd in self.boundaries:
                l = self.dm.getLabel(bd.name)
                if l:
                    i = l.getStratumSize(bd.value)
                else:
                    i = 0

                ii = uw.utilities.gather_data(np.array([i]), dtype="int")

                uw.pprint(f"| {bd.name:<20}     | {bd.value:<5} | {ii.min():<8} | {ii.max():<8} |")

            # ## PETSc marked boundaries:
            # l = self.dm.getLabel("All_Boundaries")
            # if l:
            #     i = l.getStratumSize(1001)
            # else:
            #     i = 0

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            uw.pprint(f"| {'All_Boundaries':<20}     | 1001  | {ii.min():<8} | {ii.max():<8} |")

            ## UW_Boundaries:
            l = self.dm.getLabel("UW_Boundaries")
            i = 0
            if l:
                for bd in self.boundaries:
                    i += l.getStratumSize(bd.value)

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            uw.pprint(f"| {'UW_Boundaries':<20}     | --    | {ii.min():<8} | {ii.max():<8} |")

            uw.pprint(f"| ------------------------------------------------------ |")
            uw.pprint("\n")

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
        surviving = {}
        if self.boundaries is not None:
            for b in self.boundaries:
                if b.name in ("Null_Boundary", "All_Boundaries"):
                    continue
                label = subdm.getLabel(b.name)
                if label:
                    sis = label.getStratumIS(b.value)
                    if sis and sis.getSize() > 0:
                        surviving[b.name] = b.value

        if self.regions is not None:
            for r in self.regions:
                label = subdm.getLabel(r.name)
                if label:
                    sis = label.getStratumIS(r.value)
                    if sis and sis.getSize() > 0:
                        surviving[r.name] = r.value

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

    def _build_vertex_map(self):
        """Build vertex index mapping between submesh and parent.

        Uses coordinate matching at extraction time (before any
        deformation). Cached permanently since topology doesn't change.
        """
        if hasattr(self, "_vertex_map") and self._vertex_map is not None:
            return self._vertex_map

        # Use cached KDTree from coordinate variable
        tree = self.X._get_kdtree()
        dists, indices = tree.query(self.parent.X.coords_nd, sqr_dists=False)
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

        self._deform_mesh(new_sub_coords)
        self._parent_mesh_version = self.parent._mesh_version

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
            uw.pprint(0, f"Re-extracting submesh '{label_name}' from adapted parent...")

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

            # Rebuild coordinates
            self._coords = uw.utilities.NDArray_With_Callback(
                numpy.ndarray.view(self.dm.getCoordinatesLocal().array.reshape(-1, self.cdim)),
                owner=self,
            )

            def mesh_update_callback(array, change_context):
                mesh = array.owner
                if mesh is None:
                    return

                coords = array.reshape(-1, mesh.cdim)
                mesh._deform_mesh(coords, verbose=False)
                with mesh._mesh_update_lock:
                    mesh._mesh_version += 1
                return

            self._coords.add_callback(mesh_update_callback)

            self._mesh_version += 1
            self._topology_version += 1
            self.nuke_coords_and_rebuild(verbose=False)

        # Rebuild vertex map (for restrict/prolongate)
        self._vertex_map = None
        self._build_vertex_map()

        # Invalidate DOF maps
        self._dof_maps = {}

        # Reinitialise variables on the new DM
        for var_name, old_var in old_vars.items():
            try:
                if old_var._lvec is not None:
                    old_var._lvec.destroy()
                    old_var._lvec = None
                if old_var._gvec is not None:
                    old_var._gvec.destroy()
                    old_var._gvec = None
                if hasattr(old_var, '_canonical_data'):
                    old_var._canonical_data = None
                if hasattr(old_var, '_cached_data_array'):
                    old_var._cached_data_array = None

                old_var._setup_ds()
                old_var._set_vec(available=True)

                # Interpolate from backed-up data via kd-tree IDW
                if var_name in old_var_backups:
                    try:
                        old_coords, old_data = old_var_backups[var_name]
                        new_coords = old_var.coords

                        tree = uw.kdtree.KDTree(old_coords)
                        nnn = 3 if self.dim == 2 else 4
                        dists, indices = tree.query(new_coords, k=nnn, sqr_dists=False)

                        # Inverse distance weighting
                        weights = 1.0 / (dists + 1e-30)
                        weights /= weights.sum(axis=1, keepdims=True)
                        new_data = numpy.zeros_like(old_var.data)
                        for i in range(nnn):
                            new_data += weights[:, i:i+1] * old_data[indices[:, i]]

                        old_var.pack_raw_data_to_petsc(new_data, sync=True)
                        if verbose:
                            uw.pprint(0, f"  Submesh variable '{var_name}' transferred")
                    except Exception as e2:
                        if verbose:
                            uw.pprint(0, f"  Submesh variable '{var_name}' reset (transfer failed: {e2})")
                else:
                    if verbose:
                        uw.pprint(0, f"  Submesh variable '{var_name}' reset")
            except Exception as e:
                if verbose:
                    uw.pprint(0, f"  Warning: failed to reinitialise '{var_name}': {e}")

        # Mark solvers for rebuild
        for solver in self._equation_systems_register:
            if solver is not None and hasattr(solver, 'is_setup'):
                solver.is_setup = False

        # Clear caches
        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        if hasattr(self, '_dminterpolation_cache'):
            self._dminterpolation_cache.invalidate_all(reason="submesh_re_extraction")

        self._parent_mesh_version = self.parent._mesh_version

        if verbose:
            uw.pprint(0, f"  Submesh re-extracted: {self.dm.getChart()}")

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

        # BUGFIX(#135): invalidate the per-cell face control-point arrays.
        # These are populated lazily by _get_mesh_face_control_points, sized
        # (num_faces, num_local_cells, dim). After mesh.adapt() the new mesh
        # has a different cell count, so the stale arrays from the old mesh
        # would be indexed with new-mesh cell IDs in
        # _test_if_points_in_cells_internal — producing IndexError when the
        # new cell count exceeds the old one (and silent corruption otherwise).
        self.faces_outer_control_points = None
        self.faces_inner_control_points = None

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
        """
        import underworld3 as uw

        Gamma = self.Gamma

        if not hasattr(self, '_projected_normals') or self._projected_normals is None:
            # nuke_coords_and_rebuild() clears this attribute on every deform,
            # but the underlying MeshVariable persists on the mesh. Recover it
            # quietly rather than re-creating (which logs a name collision).
            existing = self.vars.get("_n_proj")
            if existing is not None:
                self._projected_normals = existing
            else:
                # REINIT policy: _n_proj is a recomputed-on-access work
                # variable. The values depend on the *current* mesh
                # geometry (Gamma normals projected onto P1 nodes), so
                # the Phase-1 remesh helper should NOT REMAP — the value
                # at old node positions is wrong on the new mesh.
                # Recomputed below from Gamma; reset on every deform
                # via the _projected_normals=None clearing in
                # nuke_coords_and_rebuild.
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

    @timing.routine_timer_decorator
    def update_lvec(self):
        """
        This method creates and/or updates the mesh variable local vector.
        If the local vector is already up to date, this method will do nothing.
        """

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
                # Use access pattern to ensure vector is available
                with self.access(var):
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

        Called by the adapt op (``smooth_mesh_interior``, ``OT_adapt``,
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
        """

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
        >>> import underworld3 as uw
        >>> someMesh = uw.discretisation.FeMesh_Cartesian()
        >>> with someMesh._deform_mesh():
        ...     someMesh.data[0] = [0.1,0.1]
        >>> someMesh.data[0]
        array([ 0.1,  0.1])
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
        """
        Set mesh node coordinates from physical units.

        .. deprecated:: 0.99.0
            Use :attr:`X.coords` instead.

        When the mesh has coordinate scaling applied (via model units),
        this property automatically converts from physical coordinates
        to internal model coordinates for PETSc storage.

        Args:
            value (numpy.ndarray or UnitAwareArray): Node coordinates in physical units
        """
        import warnings
        import underworld3 as uw

        warnings.warn(
            "mesh.points is deprecated, use mesh.X.coords instead", DeprecationWarning, stacklevel=2
        )

        # PRINCIPLE (2025-11-27): When units are active, require unit-aware input
        # to avoid ambiguity about whether values are dimensional or non-dimensional.
        has_unit_info = hasattr(value, 'magnitude') or hasattr(value, 'value')
        model = uw.get_default_model()
        units_active = model.has_units() and uw.is_nondimensional_scaling_active()
        mesh_has_units = hasattr(self, 'units') and self.units is not None

        if not has_unit_info and mesh_has_units and units_active:
            # Plain array assigned when units are active - ambiguous
            mesh_units = self.units
            raise ValueError(
                f"Cannot assign plain array to mesh coordinates when units are active.\n"
                f"\n"
                f"The mesh has coordinate units '{mesh_units}', but the assigned\n"
                f"value has no unit information. This is ambiguous: should the values be\n"
                f"interpreted as dimensional (in {mesh_units}) or non-dimensional?\n"
                f"\n"
                f"Solutions:\n"
                f"  1. Wrap with units: UnitAwareArray(coords, units='{mesh_units}')\n"
                f"  2. Use uw.quantity() for coordinate values\n"
                f"  3. For non-dimensional values, assign directly to mesh._coords\n"
            )

        # Handle unit-aware input
        if has_unit_info:
            # Extract numerical value from unit-aware object
            if hasattr(value, 'magnitude'):
                coord_values = value.magnitude
            elif hasattr(value, 'value'):
                coord_values = value.value
            else:
                coord_values = value

            # Convert to non-dimensional units if needed
            if units_active and mesh_has_units:
                coord_values = uw.scaling.non_dimensionalise(value)
        else:
            coord_values = value

        # Apply inverse scaling to convert physical coordinates to model coordinates
        if hasattr(self.CoordinateSystem, "_scaled") and self.CoordinateSystem._scaled:
            scale_factor = self.CoordinateSystem._length_scale
            model_coords = coord_values / scale_factor
            self._coords = model_coords
        else:
            self._coords = coord_values

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
    ):
        """
        Write mesh and selected variables for visualisation output.

        This writes:
        - one mesh HDF5 file (shared/static or per-step, depending on ``meshUpdates``)
        - one HDF5 file per mesh variable
        - optional proxy files for swarm variables
        - optional XDMF file linking all output files

        When ``create_xdmf=True`` (the default), variable files also include
        ParaView-compatible groups (``/vertex_fields`` or ``/cell_fields``),
        and an XDMF file is generated on rank 0.

        """
        options = PETSc.Options()
        options.setValue("viewer_hdf5_sp_output", True)
        options.setValue("viewer_hdf5_collective", False)

        output_base_name = os.path.join(outputPath, filename)

        # check the directory where we will write checkpoint
        dir_path = os.path.dirname(output_base_name)  # get directory

        # check if path exists
        if os.path.exists(os.path.abspath(dir_path)):  # easier to debug abs
            pass
        else:
            raise RuntimeError(f"{os.path.abspath(dir_path)} does not exist")

        # check if we have write access
        if os.access(os.path.abspath(dir_path), os.W_OK):
            pass
        else:
            raise RuntimeError(f"No write access to {os.path.abspath(dir_path)}")

        # Checkpoint the mesh file itself if required

        if not meshUpdates:
            from pathlib import Path

            mesh_file = output_base_name + ".mesh.00000.h5"
            path = Path(mesh_file)
            if not path.is_file():
                self.write(mesh_file, petsc_format=False)

        else:
            mesh_file = output_base_name + f".mesh.{index:05}.h5"
            self.write(mesh_file, petsc_format=False)

        if create_xdmf:
            _write_mesh_viz_groups(self, mesh_file)

        if meshVars is not None:
            for var in meshVars:
                save_location = output_base_name + f".mesh.{var.clean_name}.{index:05}.h5"
                var.write(save_location)
                if create_xdmf:
                    _write_compat_groups(self, var, save_location)

        if swarmVars is not None:
            for svar in swarmVars:
                save_location = output_base_name + f".proxy.{svar.clean_name}.{index:05}.h5"
                svar.write_proxy(save_location)

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
    ):
        """Write PETSc DMPlex checkpoint files for restart/postprocessing.

        Checkpoint output stores PETSc DMPlex section/vector metadata required
        for exact parallel reload. Unlike ``write_timestep()``, this is restart
        output and does not write XDMF or vertex-field visualisation datasets.

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
        """

        if outputPath:
            filename = os.path.join(outputPath, filename)

        def _checkpoint_filename(var_name=None):
            variable_part = f".{var_name}" if var_name is not None else ".checkpoint"
            if unique_id:
                return filename + f"{uw.mpi.unique}{variable_part}.{index:05}.h5"
            return filename + f"{variable_part}.{index:05}.h5"

        def _write_variable(viewer, var):
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

        def _write_checkpoint_file(checkpoint_file, variables):
            viewer = PETSc.ViewerHDF5().create(checkpoint_file, "w", comm=PETSc.COMM_WORLD)
            viewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
            try:
                # Store the parallel-mesh section information for restoring the checkpoint.
                self.dm.sectionView(viewer, self.dm)

                for var in variables:
                    _write_variable(viewer, var)

                uw.mpi.barrier()  # should not be required
            finally:
                viewer.popFormat()
                viewer.destroy()

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
                        _write_checkpoint_file(_checkpoint_filename(var.clean_name), [var])
                else:
                    _write_checkpoint_file(_checkpoint_filename(), variables)
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
        var_arrays: Dict[str, numpy.ndarray] = {}
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
        petsc_format: bool = False,
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
            If True, write PETSc DMPlex HDF5 checkpoint/restart topology.
            If False, write PETSc HDF5_VIZ topology used by XDMF.

        """

        if index is not None:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via
            ## the PETSc xdmf script.
            # viewer.pushTimestepping(viewer)
            # viewer.setTimestep(index)

        viewer_format = (
            PETSc.Viewer.Format.HDF5_PETSC
            if petsc_format
            else PETSc.Viewer.Format.HDF5_VIZ
        )
        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=PETSc.COMM_WORLD)
        viewer.pushFormat(viewer_format)
        try:
            viewer(self.dm)
        finally:
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

                f.close()

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

    def _build_kd_tree_index(self):

        if hasattr(self, "_index") and self._index is not None:
            return

        dim = self.dim
        # def mesh_face_skeleton_kdtree(mesh):

        cStart, cEnd = self.dm.getHeightStratum(0)
        fStart, fEnd = self.dm.getHeightStratum(1)
        pStart, pEnd = self.dm.getDepthStratum(0)
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        control_points_list = []
        control_points_cell_list = []

        for cell, cell_id in enumerate(range(cStart, cEnd)):

            cell_faces = self.dm.getCone(cell_id)
            points = self.dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
            # Use raw internal array for KD-tree construction (avoid unit-aware wrapping)
            cell_point_coords = self._coords[points - pStart]
            cell_centroid = cell_point_coords.mean(axis=0)

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

        # We don't need an indexMap for this one because there is only one point per cell
        # and the returned kdtree value IS the index.
        # Note: self._centroids is not yet defined:

        self._centroid_index = uw.kdtree.KDTree(self._get_coords_for_basis(0, False))
        # self._centroid_index.build_index()

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
        # def mesh_face_skeleton_kdtree(mesh):

        cStart, cEnd = self.dm.getHeightStratum(0)
        fStart, fEnd = self.dm.getHeightStratum(1)
        pStart, pEnd = self.dm.getDepthStratum(0)
        num_local_cells = self.dm.getHeightStratum(0)[1]
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        # All elements in our mesh are a single type

        mesh_cell_outer_control_points = numpy.ndarray(
            shape=(cell_num_faces, num_local_cells, self.dim)
        )
        mesh_cell_inner_control_points = numpy.ndarray(
            shape=(cell_num_faces, num_local_cells, self.dim)
        )

        for cell, cell_id in enumerate(range(cStart, cEnd)):
            cell_faces = self.dm.getCone(cell_id)
            points = self.dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
            # Use raw internal array for internal mesh operations (avoid unit-aware wrapping)
            cell_point_coords = self._coords[points - pStart]

            for face in range(cell_num_faces):

                points = self.dm.getTransitiveClosure(cell_faces[face])[0][-face_num_points:]
                # Use raw internal array for internal mesh operations (avoid unit-aware wrapping)
                point_coords = self._coords[points - pStart]

                face_centroid = point_coords.mean(axis=0)
                cell_centroid = cell_point_coords.mean(axis=0)

                # Compute face normal from point coordinates (already plain numpy arrays)
                point_data = point_coords

                # 2D case
                if self.dim == 2:
                    vector = point_data[1] - point_data[0]
                    normal = numpy.array((-vector[1], vector[0]))

                # 3D simplex case (probably also OK for hexes)
                else:
                    normal = numpy.cross(
                        (point_data[1] - point_data[0]),
                        (point_data[2] - point_data[0]),
                    )

                inward_outward = numpy.sign(normal.dot(face_centroid - cell_centroid))
                normal *= inward_outward / numpy.sqrt(normal.dot(normal))

                # Compute control points (all arrays are already plain numpy, no units)
                outside_control_point = 1e-3 * normal + face_centroid
                inside_control_point = -1e-3 * normal + face_centroid

                mesh_cell_outer_control_points[face, cell, :] = outside_control_point
                mesh_cell_inner_control_points[face, cell, :] = inside_control_point

        self.faces_inner_control_points = mesh_cell_inner_control_points
        self.faces_outer_control_points = mesh_cell_outer_control_points

        return

    def _test_if_points_in_cells_internal(self, points, cells,
                                          on_boundary=True, tol=0.0):
        """
        Determine if the given points lie in the suggested cells.
        Uses a mesh skeletonization array to determine whether the point is
        with the convex polygon / polyhedron defined by a cell.

        Exact if applied to a linear mesh, approximate otherwise.

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

        if tol > 0.0:
            # Face-relative tolerance branch (parallel evaluation locator).
            # Takes precedence over on_boundary: a non-zero tol expresses
            # a geometric tolerance on the face-normal separation²; the
            # absolute on_boundary floor (~1e-12) is far below it and the
            # strict on_boundary=False (>0) is what tol is widening.
            for f in range(num_cell_faces):
                control_points_o = self.faces_outer_control_points[f, cells]
                control_points_i = self.faces_inner_control_points[f, cells]
                value = (
                    ((control_points_o - points) ** 2).sum(axis=1)
                    - ((control_points_i - points) ** 2).sum(axis=1)
                )
                sep2 = ((control_points_o - control_points_i) ** 2).sum(axis=1)
                insiders[:, f] = value > -tol * sep2
        elif on_boundary:
            _face_tol = -1e-12
            for f in range(num_cell_faces):
                control_points_o = self.faces_outer_control_points[f, cells]
                control_points_i = self.faces_inner_control_points[f, cells]
                insiders[:, f] = (
                    ((control_points_o - points) ** 2).sum(axis=1)
                    - ((control_points_i - points) ** 2).sum(axis=1)
                ) >= _face_tol
        else:
            for f in range(num_cell_faces):
                control_points_o = self.faces_outer_control_points[f, cells]
                control_points_i = self.faces_inner_control_points[f, cells]
                insiders[:, f] = (
                    ((control_points_o - points) ** 2).sum(axis=1)
                    - ((control_points_i - points) ** 2).sum(axis=1)
                ) > 0

        return numpy.all(insiders, axis=1)

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

        cStart, cEnd = self.dm.getHeightStratum(0)
        fStart, fEnd = self.dm.getHeightStratum(1)
        pStart, pEnd = self.dm.getDepthStratum(0)
        cell_num_faces = self.element.entities[1]
        cell_num_points = self.element.entities[self.dim]
        face_num_points = self.element.face_entities[self.dim]

        boundary_faces = []
        for face in range(fStart, fEnd):
            if self.dm.getJoin(face).shape[0] == 1:
                boundary_faces.append(face)

        boundary_faces = numpy.array(boundary_faces)

        control_points_list = []
        control_point_sign_list = []

        for face in boundary_faces:
            cell = self.dm.getJoin(face)[0]
            points = self.dm.getTransitiveClosure(face)[0][-face_num_points:]
            point_coords = self._coords[points - pStart]  # Use raw array for internal calculations
            face_centroid = point_coords.mean(axis=0)
            cell_centroid = self._centroids[cell - cStart]

            # 2D case
            if self.dim == 2:
                vector = point_coords[1] - point_coords[0]
                normal = numpy.array((-vector[1], vector[0]))

            else:
                # 3D simplex case (probably also OK for hexes)
                normal = numpy.cross(
                    (point_coords[1] - point_coords[0]),
                    (point_coords[2] - point_coords[0]),
                )

            inward_outward = numpy.sign(normal.dot(face_centroid - cell_centroid))
            normal *= inward_outward / numpy.sqrt(normal.dot(normal))

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
        points at an absolute -1e-12 floor (matches PR #207). ``tol > 0``
        admits on-face points at a face-relative tolerance, taking
        precedence over ``on_boundary``.

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

        # We need to filter points that lie outside the mesh but
        # still are allocated a nearby element by this distance-only check.

        cells = self._indexMap[closest_points]
        cStart, cEnd = self.dm.getHeightStratum(0)

        inside = self._test_if_points_in_cells_internal(
            coords, cells, on_boundary=on_boundary, tol=tol)
        cells[~inside] = -1
        lost_points = np.where(inside == False)[0]

        # Part 2 - try to find the lost points by walking nearby cells

        num_local_cells = self._centroids.shape[0]
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

    def _eval_use_robust_location(self) -> bool:
        """Single switch for the parallel evaluation cell-location strategy.

        Returns True when ``uw.function`` evaluation should locate cells with
        the bulletproof barycentric hint (:meth:`_robust_owning_cells`) and the
        ``DMLocatePoints`` bypass (``petsc_tools.c``), rather than PETSc's
        ``DMLocatePoints``. This is the *one place* the policy lives; the
        evaluate_nd classifier, the petsc_interpolate hint, and the
        DMInterpolation wrapper all defer to it so the three stay consistent.

        Two conditions, both required:

        * **parallel only** (``uw.mpi.size > 1``) — in serial the on-face/edge
          node points go to RBF-at-node (exact) and PETSc/the cell-wall test
          are reliable with a single domain, so serial keeps the validated path
          bit-for-bit. The parallel-only failure is the rank-local RBF / wrong-
          region value at partition-seam node points.
        * **hint is authoritative** — simplex cells (``dm.isSimplex()``: planar
          faces, affine reference map, exact face containment) or manifold
          meshes (``dim != cdim``: PETSc's in-cell test is unreliable near
          2-manifold simplex edges in 3-D). On non-simplex volume meshes
          (quads/hexes) deformed faces can be non-planar and the kdtree-nearest
          cell can be wrong, so those keep PETSc's DMLocatePoints search.
        """
        return (uw.mpi.size > 1) and (
            bool(self.dm.isSimplex()) or (self.dim != self.cdim)
        )

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
        all_centroids = gather_data(domain_centroid, bcast=True).reshape(-1, self.dim)
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

    @timing.routine_timer_decorator
    def OT_adapt(
        self,
        field,
        *,
        refinement=3.0,
        coarsening="auto",
        grad_smoothing_length="auto",
        metric_choice="front-following",
        fields_to_remap=None,
        fields_to_zero=None,
        skip_threshold=None,
        reference_coords=None,
        verbose=False,
    ):
        r"""Adapt the mesh in place so cell sizes track ``|∇field|``, using
        the validated optimal-transport reset pattern.

        Each call resets the mesh to a cached reference (the initial uniform
        coordinates), FE-remaps ``field`` onto that clean canvas, builds a
        gradient-density metric, runs the OT mover, and FE-remaps the
        requested fields onto the adapted positions. Resetting every event
        (rather than composing adaptations across time steps) is what keeps
        the mover sliver-free over long runs. The "reset" is internal — from
        the caller's point of view this just tracks the moving feature.

        Topology is preserved (vertex count, DOF maps, rank partition
        unchanged); only coordinates move. Registered solvers are marked for
        rebuild via ``_deform_mesh``.

        Reference coordinates
        ---------------------
        The reset target is snapshotted lazily on the **first** call as
        ``self._ot_adapt_reference_coords`` (a copy of the current
        ``mesh.X.coords``) and reused thereafter.

        .. warning::
           If the mesh is deformed by something other than ``OT_adapt``
           between calls (e.g. a manual ``mesh._deform_mesh(...)`` or a
           resume that loads a *deformed* snapshot), the cached reference no
           longer matches the intended pristine state. Use
           :meth:`OT_adapt_reset_reference` to re-baseline, or pass an
           explicit ``reference_coords`` for a one-off override.

        Parameters
        ----------
        field : MeshVariable
            Scalar field whose gradient drives refinement (typically ``T``).
            Always FE-remapped onto the adapted mesh.
        refinement : float, default 3.0
            Cell-size envelope ``h0/refinement`` for the densest cells.
            Validated range 1.5–5; 3 is the Nu sweet spot.
        coarsening : float or "auto", default "auto"
            ``"auto"`` = budget-conserving ``refinement**(1/d)``.
        grad_smoothing_length : "auto", None, float, or Pint Quantity, default "auto"
            Screened-Poisson de-noising length for ``|∇field|`` before the
            metric is built — the most effective sliver lever; without it,
            production refinement chases sub-cell gradient noise.
            ``"auto"`` (default) ≈ the mesh's uniform cell size (mean edge
            length) — the validated setting. ``None`` turns it off. A number
            or Pint length sets ``L`` explicitly; **user-supplied lengths are
            unit-aware** (non-dimensionalised via the projection), so pass a
            Pint quantity (or a non-dimensional number) — ``≈ h0`` is mild,
            ``≈ 2·h0`` stronger.
        metric_choice : {"front-following", "gradient-uniform"}, default "front-following"
        fields_to_remap : list of MeshVariable, optional
            Extra fields to FE-remap onto the adapted positions (``field`` is
            always remapped). ``None`` ⇒ just ``field``.
        fields_to_zero : list of MeshVariable, optional
            Fields to zero after the adapt (e.g. ``[V, P]`` for a cold
            restart of the flow solve).
        skip_threshold : float, optional
            If the mesh is already aligned with the metric (misalignment
            below this; see :func:`~underworld3.meshing.mesh_metric_mismatch`),
            skip the adapt and return ``False``. ``None`` ⇒ always adapt.
        reference_coords : array, optional
            One-off override of the reset target (does not update the cache).
        verbose : bool, default False

        Returns
        -------
        bool
            ``True`` if the mesh was adapted, ``False`` if the
            ``skip_threshold`` check short-circuited it.

        Notes
        -----
        Boundary nodes slide tangentially and stay on the boundary for
        radial coordinate systems (Annulus / shell), using the projected
        boundary normal ``mesh.Gamma_P1``. Cartesian boundaries are pinned
        (the vertex-evaluated normal is degenerate there).

        Constrained-manifold meshes (``mesh.dim != mesh.cdim``, e.g. a 2D
        spherical surface in 3D) are **not supported**: the OT mover would
        have to constrain *every* node to the surface, not just boundary
        nodes. See ``docs/developer/design/ot-adapt-api-proposal.md``.

        Examples
        --------
        >>> mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
        ...                           cellSize=1/16, qdegree=3)
        >>> T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
        >>> # ... initialise T ...
        >>> mesh.OT_adapt(T, refinement=3.0, fields_to_remap=[T])

        See Also
        --------
        OT_adapt_reset_reference : Re-baseline the reset reference coords.
        underworld3.meshing.follow_metric : The single-shot anisotropic mover.
        adapt : Topology-changing MMG remeshing (different mechanism).
        """
        if self.dim != self.cdim:
            raise NotImplementedError(
                "OT_adapt is not supported on constrained-manifold meshes "
                f"(mesh.dim={self.dim} != mesh.cdim={self.cdim}). The OT "
                "mover would need to constrain every node to the surface, "
                "not just boundary nodes — see "
                "docs/developer/design/ot-adapt-api-proposal.md."
            )
        if (not hasattr(self, "_ot_adapt_reference_coords")
                or self._ot_adapt_reference_coords is None):
            # Lazy snapshot of the reset target on first call.
            self._ot_adapt_reference_coords = numpy.asarray(
                self.X.coords).copy()

        from underworld3.meshing._ot_adapt import _ot_adapt_step

        return _ot_adapt_step(
            self, field,
            refinement=refinement,
            coarsening=coarsening,
            grad_smoothing_length=grad_smoothing_length,
            metric_choice=metric_choice,
            fields_to_remap=fields_to_remap,
            fields_to_zero=fields_to_zero,
            skip_threshold=skip_threshold,
            reference_coords=reference_coords,
            verbose=verbose,
        )

    def OT_adapt_reset_reference(self, coords=None):
        r"""Re-baseline the reference coordinates used by :meth:`OT_adapt`.

        ``coords=None`` re-snapshots the current ``mesh.X.coords`` as the new
        reset target; passing explicit ``coords`` (e.g. the initial uniform
        mesh loaded from a checkpoint) sets those instead. Use on resume,
        when the loaded mesh is in a deformed state and the cache would
        otherwise lazily initialise from it.
        """
        if coords is None:
            self._ot_adapt_reference_coords = numpy.asarray(
                self.X.coords).copy()
        else:
            self._ot_adapt_reference_coords = numpy.asarray(coords).copy()

    @timing.routine_timer_decorator
    def adapt(self, metric_field, verbose=False):
        r"""
        Adapt the mesh discretization based on a metric field.

        This method refines or coarsens the mesh in place, automatically
        transferring all attached MeshVariables, updating Surfaces, and
        marking Solvers for rebuild on their next solve() call.

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
        >>> with mesh.access(metric):
        ...     # Smaller H near fault, larger far away
        ...     metric.data[:, 0] = 0.01 + 0.09 * fault.distance_from(mesh.data)
        >>> mesh.adapt(metric, verbose=True)
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

            # Update coordinates array
            self._coords = uw.utilities.NDArray_With_Callback(
                numpy.ndarray.view(self.dm.getCoordinatesLocal().array.reshape(-1, self.cdim)),
                owner=self,
            )

            # Rebuild the callback for mesh deformation
            def mesh_update_callback(array, change_context):
                if verbose:
                    uw.pprint(0, f"Mesh update callback - mesh deform")

                coords = array.reshape(-1, array.owner.cdim)
                self._deform_mesh(coords, verbose=verbose)
                with self._mesh_update_lock:
                    self._mesh_version += 1
                    if verbose:
                        uw.pprint(0, f"Mesh version incremented to {self._mesh_version}")
                return

            self._coords.add_callback(mesh_update_callback)

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
            if old_var._lvec is not None:
                old_var._lvec.destroy()
                old_var._lvec = None
            if old_var._gvec is not None:
                old_var._gvec.destroy()
                old_var._gvec = None
            if hasattr(old_var, '_canonical_data'):
                old_var._canonical_data = None
            if hasattr(old_var, '_cached_data_array'):
                old_var._cached_data_array = None

        # Reinitialize MeshVariables on the new mesh
        # Note: Variables are reset to zero. Users should reinitialize with data.
        for var_name, old_var in old_vars_data.items():
            try:
                old_var._setup_ds()
                old_var._set_vec(available=True)

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

        # Mark solvers for rebuild
        for solver in self._equation_systems_register:
            if solver is not None and hasattr(solver, '_rebuild_after_mesh_update'):
                solver.is_setup = False
                if verbose:
                    print(f"[{uw.mpi.rank}] Solver marked for rebuild", flush=True)

        # Clear caches
        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        if hasattr(self, '_dminterpolation_cache'):
            self._dminterpolation_cache.invalidate_all(reason="mesh_adaptation")

        # Re-extract registered submeshes from the adapted parent
        for submesh in list(self._registered_submeshes):
            try:
                submesh._re_extract_from_parent(verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[{uw.mpi.rank}] Warning: submesh re-extraction failed: {e}", flush=True)

        if verbose:
            print(f"[{uw.mpi.rank}] Mesh adaptation complete", flush=True)

        return


## This is a temporary replacement for the PETSc xdmf generator
## Simplified to allow us to decide how we want to checkpoint


def _petsc_numbering_to_global_ids(numbering):
    """Convert PETSc numbering entries to non-negative global ids."""

    gids = numpy.asarray(numbering, dtype=numpy.int64).copy()
    negative = gids < 0
    gids[negative] = -gids[negative] - 1
    return gids


def _local_viz_cell_connectivity(mesh):
    """Return local cell-to-vertex connectivity in global vertex ids."""

    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    vertex_numbering = dm.getVertexNumbering().getIndices()
    vertex_gids = _petsc_numbering_to_global_ids(vertex_numbering)
    cell_num_points = mesh.element.entities[mesh.dim]

    cell_points_list = []
    for cell_id in range(cStart, cEnd):
        closure = dm.getTransitiveClosure(cell_id)[0]
        # Filter closure to strictly retain true vertices
        points = numpy.asarray(
            [p for p in closure if pStart <= p < pEnd],
            dtype=numpy.int64,
        )
        if len(points) != cell_num_points:
            raise RuntimeError(f"Expected {cell_num_points} vertices for cell {cell_id}, got {len(points)}.")
        cell_points_list.append(vertex_gids[points - pStart])

    if not cell_points_list:
        return numpy.empty((0, cell_num_points), dtype=numpy.int64)

    if mesh.dim == 3:
        if dm.isSimplex():
            reorder = [0, 2, 1, 3]
        else:
            reorder = [0, 3, 2, 1, 4, 5, 6, 7]
        cell_points_list = [pts[reorder] for pts in cell_points_list]

    return numpy.asarray(cell_points_list, dtype=numpy.int64)


def _write_mesh_viz_groups(mesh, mesh_h5_path):
    """Write ParaView-safe ``/viz`` geometry/topology groups into a mesh HDF5."""

    import underworld3 as uw
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    vertex_numbering = dm.getVertexNumbering().getIndices()
    vertex_gids = _petsc_numbering_to_global_ids(vertex_numbering)

    coords_local = numpy.asarray(dm.getCoordinatesLocal().array, dtype=numpy.float64).reshape(-1, mesh.dim)
    n_local_vertices = pEnd - pStart
    if coords_local.shape[0] != n_local_vertices:
        coords_local = numpy.asarray(mesh.X.coords, dtype=numpy.float64)
        if coords_local.shape[0] != n_local_vertices:
            raise RuntimeError(
                f"Could not match local coordinate rows ({coords_local.shape[0]}) "
                f"to DMPlex vertex count ({n_local_vertices}) for {mesh_h5_path}."
            )

    local_cells = _local_viz_cell_connectivity(mesh)

    # Gather GIDs and coordinates separately to prevent float64 upcasting of integer GIDs
    gathered_gids = uw.mpi.comm.gather(vertex_gids, root=0)
    gathered_coords = uw.mpi.comm.gather(coords_local, root=0)
    gathered_cells = uw.mpi.comm.gather(local_cells, root=0)
    uw.mpi.barrier()

    if uw.mpi.rank == 0:
        import h5py

        gid_blocks = [block for block in gathered_gids if block is not None and block.size > 0]
        coord_blocks = [block for block in gathered_coords if block is not None and block.size > 0]
        cell_blocks = [block for block in gathered_cells if block is not None and block.size > 0]

        if gid_blocks:
            all_gids = numpy.concatenate(gid_blocks)
            all_coords = numpy.vstack(coord_blocks)
            
            # Vectorized deduplication (numpy.unique returns sorted unique elements)
            ordered_gids, unique_indices = numpy.unique(all_gids, return_index=True)
            ordered_vertices = all_coords[unique_indices]
        else:
            ordered_gids = numpy.empty((0,), dtype=numpy.int64)
            ordered_vertices = numpy.empty((0, mesh.dim), dtype=numpy.float64)

        if cell_blocks:
            all_cells = numpy.vstack(cell_blocks)
            # Vectorized remapping using searchsorted (O(N log M) instead of Python dict lookup)
            dense_cells = numpy.searchsorted(ordered_gids, all_cells)
        else:
            all_cells = numpy.empty((0, mesh.element.entities[mesh.dim]), dtype=numpy.int64)
            dense_cells = numpy.empty_like(all_cells)

        with h5py.File(mesh_h5_path, "a") as h5:
            if "viz" in h5:
                del h5["viz"]
            viz = h5.create_group("viz")
            geom = viz.create_group("geometry")
            topo = viz.create_group("topology")
            geom.create_dataset("vertices", data=ordered_vertices)
            topo_cells = topo.create_dataset("cells", data=dense_cells)
            topo_cells.attrs["cell_dim"] = mesh.dim

    uw.mpi.barrier()


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
    cellDim = topo["cells"].attrs["cell_dim"]
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
