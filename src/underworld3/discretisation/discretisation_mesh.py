from typing import Optional, Tuple, Union
from enum import Enum

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

    # h5plex = PETSc.DMPlex().createFromFile(filename, comm=comm)
    h5plex = PETSc.DMPlex().create(comm=comm)
    sf0 = h5plex.topologyLoad(viewer)
    h5plex.coordinatesLoad(viewer, sf0)
    h5plex.labelsLoad(viewer, sf0)

    # Do this as well
    h5plex.setName("uw_mesh")
    h5plex.markBoundaryFaces("All_Boundaries", 1001)

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

    Attributes
    ----------
    N : sympy.vector.CoordSys3D
        SymPy coordinate system for symbolic expressions.
    X : UWCoordinate tuple
        Coordinate variables (x, y, z) for use in expressions.
    dim : int
        Spatial dimension of the mesh.
    dm : PETSc.DMPlex
        Underlying PETSc distributed mesh object.

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
        self._mesh_update_lock = threading.RLock()

        comm = PETSc.COMM_WORLD

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

                f.close()

                # This needs to be done when reading a dm from a checkpoint
                # or building from an imported mesh format

                self.dm.setFromOptions()

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
                self.dm.distribute()

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
                self.dm.distribute()

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
                self.dm.distribute()

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
            print(f"Mesh update callback - mesh deform")
            coords = array.reshape(-1, array.owner.cdim)
            self._deform_mesh(coords, verbose=True)

            # Increment mesh version to notify registered swarms of coordinate changes
            with self._mesh_update_lock:
                self._mesh_version += 1
                print(f"Mesh version incremented to {self._mesh_version}")

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

        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        self._dm_initialized = False
        self._quadrature = False
        self._stale_lvec = True
        self._lvec = None
        self.petsc_fe = None

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

    def nuke_coords_and_rebuild(
        self,
        verbose=False,
    ):
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

        options = PETSc.Options()
        options.setValue(f"meshproj_{self.mesh_instances}_petscspace_degree", self.degree)

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
        elif PETSc.Sys.getVersion() >= (3, 24, 0):
            # PETSc 3.24+ added 'localized' parameter (for DG coordinate spaces)
            self.dm.setCoordinateDisc(disc=self.petsc_fe, localized=False, project=False)
        else:
            # PETSc 3.21-3.23: older signature without localized parameter
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

        self.dm.copyDS(self.dm_hierarchy[-1])

        if verbose and uw.mpi.rank == 0:
            print(
                f"Mesh Spatial Discretisation Complete",
                flush=True,
            )

        return

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

    def _deform_mesh(self, new_coords: numpy.ndarray, verbose=False):
        """
        This method will update the mesh coordinates and reset any cached coordinates in
        the mesh and in equation systems that are registered on the mesh.

        The coord array that is passed in should match the shape of self.data
        """

        coord_vec = self.dm.getCoordinatesLocal()
        coords = coord_vec.array.reshape(-1, self.cdim)
        coords[...] = new_coords[...]

        self.dm.setCoordinatesLocal(coord_vec)
        self.nuke_coords_and_rebuild()

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
    def Gamma_N(self) -> sympy.vector.CoordSys3D:
        r"""SymPy coordinate system for boundary/surface coordinates.

        Returns
        -------
        sympy.vector.CoordSys3D
            The boundary coordinate system object.
        """
        return self._Gamma

    @property
    def Gamma(self) -> sympy.vector.CoordSys3D:
        r"""Boundary coordinate scalars as a row matrix.

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
    ):
        """
        Write the selected mesh, variables and swarm variables (as proxies) for later visualisation.
        An xdmf file is generated and the overall package can then be read by paraview or pyvista.
        Vertex values (on the mesh points) are stored for all variables regardless of their interpolation order
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
                self.write(mesh_file)

        else:
            self.write(output_base_name + f".mesh.{index:05}.h5")

        if meshVars is not None:
            for var in meshVars:
                save_location = output_base_name + f".mesh.{var.clean_name}.{index:05}.h5"
                var.write(save_location)

        if swarmVars is not None:
            for svar in swarmVars:
                save_location = output_base_name + f".proxy.{svar.clean_name}.{index:05}.h5"
                svar.write_proxy(save_location)

        if uw.mpi.rank == 0:
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
        """

        Use PETSc to save the mesh and mesh vars in a h5 and xdmf file.

        Parameters
        ----------
        meshVars:
            List of UW mesh variables to save. If left empty then just the mesh is saved.
        index :
            An index which might correspond to the timestep or output number (for example).
        outputPath :
            Path to save the data. If left empty it will save the data in the current working directory.
        """

        if meshVars != None and not isinstance(meshVars, list):
            raise RuntimeError("`meshVars` does not appear to be a list.")

        from underworld3.utilities import generateXdmf

        ### save mesh vars
        fname = f"./{outputPath}{'_step_'}{index:05d}.h5"
        xfname = f"./{outputPath}{'_step_'}{index:05d}.xdmf"
        #### create petsc viewer
        viewer = PETSc.ViewerHDF5().createHDF5(
            fname, mode=PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_WORLD
        )

        viewer(self.dm)

        ### Empty meshVars will save just the mesh
        if meshVars != None:
            for var in meshVars:
                viewer(var._gvec)

        viewer.destroy()

        if uw.mpi.rank == 0:
            generateXdmf(fname, xfname)

    @timing.routine_timer_decorator
    def write_checkpoint(
        self,
        filename: str,
        meshUpdates: bool = True,
        meshVars: Optional[list] = [],
        swarmVars: Optional[list] = [],
        index: Optional[int] = 0,
        unique_id: Optional[bool] = False,
    ):
        """Write data in a format that can be restored for restarting the simulation
        The difference between this and the visualisation is 1) the parallel section needs
        to be stored to reload the data correctly, and 2) the visualisation information (vertex form of fields)
        is not stored. This routines uses dmplex *VectorView and *VectorLoad functionality

        """

        # The mesh checkpoint is the same as the one required for visualisation

        if not meshUpdates:
            from pathlib import Path

            mesh_file = filename + ".mesh.0.h5"
            path = Path(mesh_file)
            if not path.is_file():
                self.write(mesh_file)

        else:
            self.write(filename + f".mesh.{index:05}.h5")

        # Checkpoint file

        if unique_id:
            checkpoint_file = filename + f"{uw.mpi.unique}.checkpoint.{index:05}.h5"
        else:
            checkpoint_file = filename + f".checkpoint.{index:05}.h5"

        self.dm.setName("uw_mesh")
        viewer = PETSc.ViewerHDF5().create(checkpoint_file, "w", comm=PETSc.COMM_WORLD)

        # Store the parallel-mesh section information for restoring the checkpoint.
        self.dm.sectionView(viewer, self.dm)

        if meshVars is not None:
            for var in meshVars:
                iset, subdm = self.dm.createSubDM(var.field_id)
                subdm.setName(var.clean_name)
                self.dm.globalVectorView(viewer, subdm, var._gvec)
                self.dm.sectionView(viewer, subdm)
                # v._gvec.view(viewer) # would add viz information plus a duplicate of the data

        if swarmVars is not None:
            for svar in swarmVars:
                var = svar._meshVar
                iset, subdm = self.dm.createSubDM(var.field_id)
                subdm.setName(var.clean_name)
                self.dm.globalVectorView(viewer, subdm, var._gvec)
                self.dm.sectionView(viewer, subdm)

        uw.mpi.barrier()  # should not be required
        viewer.destroy()

    @timing.routine_timer_decorator
    def write(self, filename: str, index: Optional[int] = None):
        """
        Save mesh data to the specified hdf5 file.


        Parameters
        ----------
        filename :
            The filename for the mesh checkpoint file.
        index :
            Not yet implemented. An optional index which might
            correspond to the timestep (for example).

        """

        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=PETSc.COMM_WORLD)
        if index:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via
            ## the PETSc xdmf script.
            # viewer.pushTimestepping(viewer)
            # viewer.setTimestep(index)

        viewer(self.dm)
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

                coordinates_type_dict = {
                    "name": self.CoordinateSystemType.name,
                    "value": self.CoordinateSystemType.value,
                }
                g.attrs["coordinate_system_type"] = json.dumps(coordinates_type_dict)

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

    def _test_if_points_in_cells_internal(self, points, cells):
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
        """
        # Internal version - points assumed to already be in model units
        self._mark_faces_inside_and_out()

        cells = cells.reshape(-1)
        assert points.shape[0] == cells.shape[0]

        cStart, cEnd = self.dm.getHeightStratum(0)
        num_cell_faces = self.dm.getConeSize(cStart)

        inside = numpy.ones_like(cells, dtype=bool)
        insiders = numpy.ndarray(shape=(cells.shape[0], num_cell_faces), dtype=bool)

        for f in range(num_cell_faces):
            control_points_o = self.faces_outer_control_points[f, cells]
            control_points_i = self.faces_inner_control_points[f, cells]
            inside = (
                ((control_points_o - points) ** 2).sum(axis=1)
                - ((control_points_i - points) ** 2).sum(axis=1)
            ) > 0

            insiders[:, f] = inside[:]

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

        control_point_kdtree = uw.kdtree.KDTree(numpy.array(control_points_list))
        control_point_sign = numpy.array(control_point_sign_list)

        self.boundary_face_control_points_kdtree = control_point_kdtree
        self.boundary_face_control_points_sign = control_point_sign

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
        in_or_not = self.boundary_face_control_points_sign[closest_control_points_ext] > 0

        ## This choice of distance needs some more thought

        near_boundary = numpy.where(dist2 < 2 * max_radius**2)[0]
        near_boundary_points = model_points[near_boundary]

        in_or_not[near_boundary] = (
            self._get_closest_local_cells_internal(near_boundary_points) != -1
        )

        if strict_validation:
            chosen_ones = numpy.where(in_or_not == True)[0]
            chosen_points = model_points[chosen_ones]
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

    def _get_closest_local_cells_internal(self, coords: numpy.ndarray) -> numpy.ndarray:
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

        inside = self._test_if_points_in_cells_internal(coords, cells)
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
                coords[lost_points], closest_centroids[:, i]
            )
            cells[lost_points[inside]] = closest_centroids[inside, i]

            if np.count_nonzero(cells == -1) == 0:
                break

        return cells

    def test_if_points_in_cells(self, points, cells):
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

        # Call internal implementation
        return self._test_if_points_in_cells_internal(model_points, cells)

    def get_closest_local_cells(self, coords: numpy.ndarray) -> numpy.ndarray:
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
        return self._get_closest_local_cells_internal(model_coords)

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

        domain_centroid = self._centroids.mean(axis=0)
        all_centroids = gather_data(domain_centroid, bcast=True).reshape(-1, self.dim)
        return all_centroids

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

    def get_max_radius(self) -> float:
        """
        This method returns the global maximum distance from any cell centroid to a face.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        import numpy as np

        all_max_radii = uw.utilities.gather_data(np.array((self._radii.max(),)), bcast=True)

        return all_max_radii.max()

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
    def adapt(self, metric_field, verbose=False):
        r"""
        Adapt the mesh discretization based on a metric field.

        This method refines or coarsens the mesh in place, automatically
        transferring all attached MeshVariables, updating Surfaces, and
        marking Solvers for rebuild on their next solve() call.

        Parameters
        ----------
        metric_field : MeshVariable
            A scalar MeshVariable containing target edge lengths (H field).
            Smaller values mean finer mesh, larger values mean coarser.
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

        # Capture current variable data, excluding only the metric field
        # (which becomes invalid after adaptation)
        # All other variables (including surface distance fields) are reinitialized
        old_vars_data = {}
        metric_name = metric_field.name if hasattr(metric_field, 'name') else None
        for var_name, var in self._vars.items():
            if var is not None and var_name != metric_name:
                old_vars_data[var_name] = var

        # Stack boundary labels for adaptation
        adaptivity._dm_stack_bcs(self.dm, self.boundaries, "CombinedBoundaries")

        # Create the metric from the field
        hvec = metric_field._lvec
        metric_vec = self.dm.metricCreateIsotropic(hvec, metric_field.field_id)

        if verbose:
            n_nodes_old = self.dm.getChart()[1] - self.dm.getChart()[0]
            print(f"[{uw.mpi.rank}] Mesh adaptation starting (nodes: ~{n_nodes_old})...", flush=True)

        # Perform the actual mesh adaptation
        new_dm = self.dm.adaptMetric(metric_vec, bdLabel="CombinedBoundaries")

        # Unstack boundary labels on the new dm
        adaptivity._dm_unstack_bcs(new_dm, self.boundaries, "CombinedBoundaries")

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

        # Note: Variable transfer is complex and may hang with large meshes.
        # For now, we skip automatic transfer. Users can reinitialize variables
        # after adaptation using old_var.rbf_interpolate() if needed.
        if verbose and old_vars_data:
            print(f"[{uw.mpi.rank}] Found {len(old_vars_data)} variables. "
                  "Variables will be reset; reinitialize manually if needed.", flush=True)

        # Store old data for potential manual recovery
        old_var_data_backup = {}
        for var_name, old_var in old_vars_data.items():
            try:
                # Back up old data before adaptation
                if old_var._lvec is not None:
                    old_var_data_backup[var_name] = old_var._lvec.array.copy()
            except Exception:
                pass

        # Clean up temp mesh (we created it but won't use it for transfer)
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
                print(f"Mesh update callback - mesh deform")
                coords = array.reshape(-1, array.owner.cdim)
                self._deform_mesh(coords, verbose=True)
                with self._mesh_update_lock:
                    self._mesh_version += 1
                    print(f"Mesh version incremented to {self._mesh_version}")
                return

            self._coords.add_callback(mesh_update_callback)

            # Increment mesh version (marks swarms as stale)
            self._mesh_version += 1
            self._topology_version += 1

            # Rebuild coordinate navigation
            self.nuke_coords_and_rebuild(verbose=False)

        # Reinitialize MeshVariables on the new mesh
        # Note: Variables are reset to zero. Users should reinitialize with data.
        for var_name, old_var in old_vars_data.items():
            try:
                # Destroy old vectors
                if old_var._lvec is not None:
                    old_var._lvec.destroy()
                    old_var._lvec = None
                if old_var._gvec is not None:
                    old_var._gvec.destroy()
                    old_var._gvec = None

                # Invalidate cached data arrays (must be recreated for new shape)
                if hasattr(old_var, '_canonical_data'):
                    old_var._canonical_data = None
                if hasattr(old_var, '_cached_data_array'):
                    old_var._cached_data_array = None

                # Re-setup the variable on the new mesh
                old_var._setup_ds()
                old_var._set_vec(available=True)

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

        # Remove only the metric field from mesh._vars
        # (it was specific to the pre-adaptation mesh and is now invalid)
        # Surface distance variables stay - they're just marked stale and will recompute
        if metric_name and metric_name in self._vars:
            del self._vars[metric_name]

        # Clear caches
        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        if hasattr(self, '_dminterpolation_cache'):
            self._dminterpolation_cache.invalidate_all(reason="mesh_adaptation")

        if verbose:
            print(f"[{uw.mpi.rank}] Mesh adaptation complete", flush=True)

        return


## This is a temporary replacement for the PETSc xdmf generator
## Simplified to allow us to decide how we want to checkpoint


def checkpoint_xdmf(
    filename: str,
    meshUpdates: bool = True,
    meshVars: Optional[list] = [],
    swarmVars: Optional[list] = [],
    index: Optional[int] = 0,
):
    import h5py
    import os

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

    if "viz" in h5 and "topology" in h5["viz"]:
        topoPath = "viz/topology"
        topo = h5["viz"]["topology"]
    else:
        topoPath = "topology"
        topo = h5["topology"]

    vertices = geom["vertices"]
    numVertices = vertices.shape[0]
    spaceDim = vertices.shape[1]
    cells = topo["cells"]
    numCells = cells.shape[0]
    numCorners = cells.shape[1]
    cellDim = topo["cells"].attrs["cell_dim"]

    h5.close()

    # We only use a subset of the possible cell types
    if spaceDim == 2:
        if numCorners == 3:
            topology_type = "Triangle"
        else:
            topology_type = "Quadrilateral"
        geomType = "XY"
    else:
        if numCorners == 4:
            topology_type = "Tetrahedron"
        else:
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
              NumberType="Float" Precision="8"
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

    def get_cell_field_size(h5_filename, mesh_var):
        try:
            with h5py.File(h5_filename, "r") as f:
                size = f[f"cell_fields/{mesh_var.clean_name}_{mesh_var.clean_name}"].shape[0]
            return size
        except:
            with h5py.File(h5_filename, "r") as f:
                size = f[f"fields/{mesh_var.clean_name}"].shape[0]
            return size

    attributes = ""
    for var in meshVars:
        var_filename = filename + f".mesh.{var.clean_name}.{index:05}.h5"

        if var.num_components == 1:
            variable_type = "Scalar"
        else:
            variable_type = "Vector"

        # Determine if data is stored on nodes (vertex_fields) or cells (cell_fields)
        if not getattr(var, "continuous") or getattr(var, "degree") == 0:
            center = "Cell"
            numItems = get_cell_field_size(var_filename, var)
            field_group = "cell_fields"
        else:
            center = "Node"
            numItems = numVertices
            field_group = "vertex_fields"

        var_attribute = f"""
        <Attribute
           Name="{var.clean_name}"
           Type="{variable_type}"
           Center="{center}">
          <DataItem ItemType="HyperSlab"
                Dimensions="1 {numItems} {var.num_components}"
                Type="HyperSlab">
            <DataItem
               Dimensions="3 3"
               Format="XML">
              0 0 0
              1 1 1
              1 {numItems} {var.num_components}
            </DataItem>
            <DataItem
               DataType="Float" Precision="8"
               Dimensions="1 {numItems} {var.num_components}"
               Format="HDF">
              &{var.clean_name+"_Data"};:/{field_group}/{var.clean_name+"_"+var.clean_name}
            </DataItem>
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
