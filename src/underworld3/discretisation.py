from typing import Optional, Tuple, Union
from enum import Enum

import os
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

from underworld3.cython import petsc_discretisation
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
def _from_gmsh(
    filename, comm=None, markVertices=False, useRegions=True, useMultipleTags=True
):
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
        plex_0 = PETSc.DMPlex().createFromFile(
            filename, interpolate=True, comm=PETSc.COMM_SELF
        )

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
    Mesh class for uw - documentation needed
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
        verbose=False,
        *args,
        **kwargs,
    ):
        self.instance = Mesh.mesh_instances
        Mesh.mesh_instances += 1

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
                    print(
                        f"Constructing UW mesh from gmsh {plex_or_meshfile}", flush=True
                    )

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
                self.sf0, self.dm = _from_plexh5(
                    plex_or_meshfile, PETSc.COMM_WORLD, return_sf=True
                )

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
                    coordinate_system_type = uw.discretisation.CoordinateSystemType(
                        coord_type_dict["value"]
                    )
                except KeyError:
                    pass

                f.close()

                # This needs to be done when reading a dm from a checkpoint
                # or building from an imported mesh format

                self.dm.setFromOptions()

            else:
                raise RuntimeError(
                    "Mesh file %s has unknown format '%s'."
                    % (plex_or_meshfile, ext[1:])
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
            all_edges_label.setStratumIS(
                boundaries.Null_Boundary.value, all_edges_IS_dm
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

        try:
            self.isSimplex = self.dm.isSimplex()
        except:
            self.isSimplex = simplex

        self._vars = {}
        self._block_vars = {}

        # a list of equation systems that will
        # need to be rebuilt if the mesh coordinates change

        self._equation_systems_register = []

        self._evaluation_hash = None
        self._evaluation_interpolated_results = None
        self._accessed = False
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

        if (
            self.CoordinateSystem.coordinate_type
            == CoordinateSystemType.CYLINDRICAL2D_NATIVE
            or self.CoordinateSystem.coordinate_type
            == CoordinateSystemType.CYLINDRICAL3D_NATIVE
        ):
            self.vector = uw.maths.vector_calculus_cylindrical(
                mesh=self,
            )
        elif (
            self.CoordinateSystem.coordinate_type
            == CoordinateSystemType.SPHERICAL_NATIVE
        ):
            self.vector = uw.maths.vector_calculus_spherical(
                mesh=self,
            )  ## Not yet complete or tested

        elif (
            self.CoordinateSystem.coordinate_type
            == CoordinateSystemType.SPHERE_SURFACE_NATIVE
        ):
            self.vector = uw.maths.vector_calculus_spherical_surface2D_lonlat(
                mesh=self,
            )

        else:
            self.vector = uw.maths.vector_calculus(mesh=self)

        super().__init__()

    @property
    def dim(self) -> int:
        """
        The mesh dimensionality.
        """
        return self.dm.getDimension()

    @property
    def cdim(self) -> int:
        """
        The mesh dimensionality.
        """
        return self.dm.getCoordinateDim()

    @property
    def element(self) -> dict:
        """
        The element information of the mesh (no mixed meshes in uw3) so this
        applies to every cell of the `mesh dmplex object`
        """

        return self._element

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
            if uw.mpi.rank == 0:
                print(f"\n")
                print(f"Mesh # {self.instance}: {self.name}\n")

                # Only if notebook and serial
                if uw.is_notebook and uw.mpi.size == 1:
                    uw.visualisation.plot_mesh(self, window_size=(600, 400))

                # Total number of cells
                nstart, nend = self.dm.getHeightStratum(0)
                num_cells = nend - nstart

                if uw.mpi.rank == 0:
                    print(f"Number of cells: {num_cells}\n")

                    if len(self.vars) > 0:
                        print(
                            f"| Variable Name       | component | degree |     type        |"
                        )
                        print(
                            f"| ---------------------------------------------------------- |"
                        )
                        for vname in self.vars.keys():
                            v = self.vars[vname]
                            print(
                                f"| {v.clean_name:<20}|{v.num_components:^10} |{v.degree:^7} | {v.vtype.name:^15} |"
                            )

                        print(
                            f"| ---------------------------------------------------------- |"
                        )
                        print("\n", flush=True)
                    else:
                        print(f"No variables are defined on the mesh\n", flush=True)

            ## Boundary information

            if uw.mpi.rank == 0:
                if len(self.boundaries) > 0:
                    print(
                        f"| Boundary Name            | ID    |",
                        flush=True,
                    )
                    print(
                        f"| -------------------------------- |",
                        flush=True,
                    )
                else:
                    print(f"No boundary labels are defined on the mesh\n", flush=True)

            for bd in self.boundaries:
                l = self.dm.getLabel(bd.name)
                if l:
                    i = l.getStratumSize(bd.value)
                else:
                    i = 0

                ii = uw.utilities.gather_data(np.array([i]), dtype="int")

                if uw.mpi.rank == 0:
                    print(
                        f"| {bd.name:<20}     | {bd.value:<5} |",
                        flush=True,
                    )

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            if uw.mpi.rank == 0:
                print(
                    f"| {'All_Boundaries':<20}     | 1001  |",
                    flush=True,
                )

            ## UW_Boundaries:
            l = self.dm.getLabel("UW_Boundaries")
            i = 0
            if l:
                for bd in self.boundaries:
                    i += l.getStratumSize(bd.value)

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            if uw.mpi.rank == 0:
                print(
                    f"| {'UW_Boundaries':<20}     | --    |",
                    flush=True,
                )

            if uw.mpi.rank == 0:
                print(f"| -------------------------------- |")
                print("\n", flush=True)

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
                    print(
                        f"| Variable Name       | component | degree |     type        |"
                    )
                    print(
                        f"| ---------------------------------------------------------- |"
                    )
                    for vname in self.vars.keys():
                        v = self.vars[vname]
                        print(
                            f"| {v.clean_name:<20}|{v.num_components:^10} |{v.degree:^7} | {v.vtype.name:^15} |"
                        )

                    print(
                        f"| ---------------------------------------------------------- |"
                    )
                    print("\n", flush=True)
                else:
                    print(f"No variables are defined on the mesh\n", flush=True)

            ## Boundary information

            if uw.mpi.rank == 0:
                if len(self.boundaries) > 0:
                    print(
                        f"| Boundary Name            | ID    | Min Size | Max Size |",
                        flush=True,
                    )
                    print(
                        f"| ------------------------------------------------------ |",
                        flush=True,
                    )
                else:
                    print(f"No boundary labels are defined on the mesh\n", flush=True)

            for bd in self.boundaries:
                l = self.dm.getLabel(bd.name)
                if l:
                    i = l.getStratumSize(bd.value)
                else:
                    i = 0

                ii = uw.utilities.gather_data(np.array([i]), dtype="int")

                if uw.mpi.rank == 0:
                    print(
                        f"| {bd.name:<20}     | {bd.value:<5} | {ii.min():<8} | {ii.max():<8} |",
                        flush=True,
                    )

            # ## PETSc marked boundaries:
            # l = self.dm.getLabel("All_Boundaries")
            # if l:
            #     i = l.getStratumSize(1001)
            # else:
            #     i = 0

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            if uw.mpi.rank == 0:
                print(
                    f"| {'All_Boundaries':<20}     | 1001  | {ii.min():<8} | {ii.max():<8} |",
                    flush=True,
                )

            ## UW_Boundaries:
            l = self.dm.getLabel("UW_Boundaries")
            i = 0
            if l:
                for bd in self.boundaries:
                    i += l.getStratumSize(bd.value)

            ii = uw.utilities.gather_data(np.array([i]), dtype="int")

            if uw.mpi.rank == 0:
                print(
                    f"| {'UW_Boundaries':<20}     | --    | {ii.min():<8} | {ii.max():<8} |",
                    flush=True,
                )

            if uw.mpi.rank == 0:
                print(f"| ------------------------------------------------------ |")
                print("\n", flush=True)

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

        if uw.mpi.rank == 0:
            print(f"\n")
            print(f"Mesh # {self.instance}: {self.name}\n")

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

        ## Boundary information on each proc

        if uw.mpi.rank == 0:
            if len(self.boundaries) > 0:
                print(f"| Boundary Name            | ID    | Size | Proc ID      |")
                print(f"| ------------------------------------------------------ |")
            else:
                print(f"No boundary labels are defined on the mesh\n")

        ### goes through each processor and gets the label size
        with uw.mpi.call_pattern(pattern="sequential"):
            for bd in self.boundaries:
                l = self.dm.getLabel(bd.name)
                if l:
                    i = l.getStratumSize(bd.value)
                else:
                    i = 0
                print(
                    f"| {bd.name:<20}     | {bd.value:<5} | {i:<8} | {uw.mpi.rank:<8} |"
                )

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

        if verbose and uw.mpi.rank == 0:
            print(
                f"PETScDS - (re) initialised",
                flush=True,
            )

        self._coord_array = {}

        # let's go ahead and do an initial projection from linear (the default)
        # to linear. this really is a nothing operation, but a
        # side effect of this operation is that coordinate DM DMField is
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates.

        options = PETSc.Options()
        options.setValue(
            f"meshproj_{self.mesh_instances}_petscspace_degree", self.degree
        )

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

        if (
            PETSc.Sys.getVersion() <= (3, 20, 5)
            and PETSc.Sys.getVersionInfo()["release"] == True
        ):
            self.dm.projectCoordinates(self.petsc_fe)
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

            with self.access():
                # traverse subdms, taking user generated data in the subdm
                # local vec, pushing it into a global sub vec
                for var, subiset, subdm in zip(self.vars.values(), isets, dms):
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
            raise RuntimeError(
                "Mesh `lvec` needs to be updated using the update_lvec()` method."
            )
        return self._lvec

    def __del__(self):
        if hasattr(self, "_lvec") and self._lvec:
            self._lvec.destroy()

    def deform_mesh(self, new_coords: numpy.ndarray, verbose=False):
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

        # This should not be necessary any more as we now check the
        # coordinates on the DM to see if they have changed (and we rebuild the
        # discretisation as needed)
        #
        # for eq_system in self._equation_systems_register:
        #     eq_system._rebuild_after_mesh_update(verbose)

        return

    def access(self, *writeable_vars: "MeshVariable"):
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
        >>> with someMesh.deform_mesh():
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

        self._accessed = True
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
                            var._data_container[i, j] = var._data_container[
                                i, j
                            ]._replace(
                                data=f"MeshVariable[...].data is only available within mesh.access() context",
                            )

                timing._decrementDepth()
                timing.log_result(time.time() - stime, "Mesh.access", 1)

        return exit_manager(self)

    @property
    def N(self) -> sympy.vector.CoordSys3D:
        """
        The mesh coordinate system.
        """
        return self._N

    @property
    def Gamma_N(self) -> sympy.vector.CoordSys3D:
        """
        The mesh coordinate system.
        """
        return self._Gamma

    @property
    def Gamma(self) -> sympy.vector.CoordSys3D:
        """
        The mesh coordinate system.
        """
        return sympy.Matrix(self._Gamma.base_scalars()[0 : self.cdim]).T

    @property
    def X(self) -> sympy.Matrix:
        return self._CoordinateSystem.X

    @property
    def CoordinateSystem(self) -> CoordinateSystem:
        return self._CoordinateSystem

    @property
    def r(self) -> Tuple[sympy.vector.BaseScalar]:
        """
        The tuple of base scalar objects (N.x,N.y,N.z) for the mesh.
        """
        return self._N.base_scalars()[0 : self.cdim]

    @property
    def rvec(self) -> sympy.vector.Vector:
        """
        The r vector, `r = N.x*N.i + N.y*N.j [+ N.z*N.k]`.
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
        """
        # get flat array
        arr = self.dm.getCoordinatesLocal().array
        return arr.reshape(-1, self.cdim)

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
                save_location = (
                    output_base_name + f".mesh.{var.clean_name}.{index:05}.h5"
                )
                var.write(save_location)

        if swarmVars is not None:
            for svar in swarmVars:
                save_location = (
                    output_base_name + f".proxy.{svar.clean_name}.{index:05}.h5"
                )
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

        uw.mpi.barrier()

        ## Add boundary metadata to the file

        import h5py, json
        import time

        # time.sleep(1)

        # Sequential (no distributed information needs to be saved)

        if uw.mpi.rank == 0:

            f = h5py.File(filename, "a")
            g = f.create_group("metadata")

            boundaries_dict = {i.name: i.value for i in self.boundaries}
            string_repr = json.dumps(boundaries_dict)
            g.attrs["boundaries"] = string_repr

            coordinates_type_dict = {
                "name": self.CoordinateSystemType.name,
                "value": self.CoordinateSystemType.value,
            }
            string_repr = json.dumps(coordinates_type_dict)
            g.attrs["coordinate_system_type"] = string_repr

            f.close()

        uw.mpi.barrier()

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
            self._coord_array[key] = self._get_coords_for_basis(
                var.degree, var.continuous
            )
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
            cell_point_coords = self.data[points - pStart]
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
                control_points_list.append(
                    0.99 * cell_point_coords[i] + 0.01 * cell_centroid
                )
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
        for all faces in a convex polyhedron, then we are inside the cell
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
            cell_point_coords = self.data[points - pStart]

            for face in range(cell_num_faces):

                points = self.dm.getTransitiveClosure(cell_faces[face])[0][
                    -face_num_points:
                ]
                point_coords = self.data[points - pStart]

                face_centroid = point_coords.mean(axis=0)
                cell_centroid = cell_point_coords.mean(axis=0)

                # 2D case
                if self.dim == 2:
                    vector = point_coords[1] - point_coords[0]
                    normal = numpy.array((-vector[1], vector[0]))

                # 3D simplex case (probably also OK for hexes)
                else:
                    normal = numpy.cross(
                        (point_coords[1] - point_coords[0]),
                        (point_coords[2] - point_coords[0]),
                    )

                inward_outward = numpy.sign(normal.dot(face_centroid - cell_centroid))
                normal *= inward_outward / numpy.sqrt(normal.dot(normal))

                outside_control_point = 1e-3 * normal + face_centroid
                inside_control_point = -1e-3 * normal + face_centroid

                mesh_cell_outer_control_points[face, cell, :] = outside_control_point
                mesh_cell_inner_control_points[face, cell, :] = inside_control_point

        self.faces_inner_control_points = mesh_cell_inner_control_points
        self.faces_outer_control_points = mesh_cell_outer_control_points

        return

    def test_if_points_in_cells(self, points, cells):
        """
        Determine if the given points lie in the suggested cells.
        Uses a mesh skeletonization array to determine whether the point is
        with the convex polygon / polyhedron defined by a cell.

        Exact if applied to a linear mesh, approximate otherwise.
        """

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
            point_coords = self.data[points - pStart]
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

                outside_control_point = (
                    1e-8 * normal + 0.8 * point_coords[pt] + 0.2 * face_centroid
                )
                control_points_list.append(outside_control_point)
                control_point_sign_list.append(-1)

                inside_control_point = (
                    -1e-8 * normal + 0.8 * point_coords[pt] + 0.2 * face_centroid
                )
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

        """

        self._mark_local_boundary_faces_inside_and_out()

        max_radius = self.get_max_radius()

        if points.shape[0] == 0:
            return False

        dist2, closest_control_points_ext = (
            self.boundary_face_control_points_kdtree.query(points, k = 1, sqr_dists = True)
        )
        in_or_not = (
            self.boundary_face_control_points_sign[closest_control_points_ext] > 0
        )

        ## This choice of distance needs some more thought

        near_boundary = numpy.where(dist2 < max_radius**2)[0]
        near_boundary_points = points[near_boundary]

        in_or_not[near_boundary] = (
            self.get_closest_local_cells(near_boundary_points) != -1
        )

        if strict_validation:
            chosen_ones = numpy.where(in_or_not == True)[0]
            chosen_points = points[chosen_ones]
            in_or_not[chosen_ones] = self.get_closest_local_cells(chosen_points) != -1

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
            shape (n_coords,dim).

        Returns:
        --------
        closest_cells:
            An array of indices representing the cells closest to the provided
            coordinates. This will be a 1-dimensional array of
            shape (n_coords).
        """
        import numpy as np

        self._build_kd_tree_index()

        if len(coords) > 0:
            dist, closest_points = self._index.query(coords, k = 1, sqr_dists = False)
            if np.any(closest_points > self._index.n):
                raise RuntimeError(
                    "An error was encountered attempting to find the closest cells to the provided coordinates."
                )
        else:
            ### returns an empty array if no coords are on a proc
            closest_points, dist, found = False, False, numpy.array([None])

        return self._indexMap[closest_points]

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
            shape (n_coords,dim).

        Returns:
        --------
        closest_cells:
            An array of indices representing the cells closest to the provided
            coordinates. This will be a 1-dimensional array of
            shape (n_coords).


        """
        import numpy as np

        import numpy as np

        # Create index if required
        self._build_kd_tree_index()

        if len(coords) > 0:
            dist, closest_points = self._index.query(coords, k = 1, sqr_dists = False)
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

        inside = self.test_if_points_in_cells(coords, cells)
        cells[~inside] = -1
        lost_points = np.where(inside == False)[0]

        # Part 2 - try to find the lost points by walking nearby cells

        num_local_cells = self._centroids.shape[0]
        num_testable_neighbours = min(num_local_cells, 50)

        dist2, closest_centroids = self._centroid_index.query(
            coords[lost_points], k=num_testable_neighbours, sqr_dists = False
        )

        # This number is close to the point-point coordination value in 3D unstructured
        # grids (by inspection)

        for i in range(0, num_testable_neighbours):

            inside = self.test_if_points_in_cells(
                coords[lost_points], closest_centroids[:, i]
            )
            cells[lost_points[inside]] = closest_centroids[inside, i]

            if np.count_nonzero(cells == -1) == 0:
                break

        return cells

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
            cell_coords = self.data[cell_points - pStart]

            distsq, _ = centroids_kd_tree.query(cell_coords, k = 1, sqr_dists = True)

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

        all_min_radii = uw.utilities.gather_data(
            np.array((self._radii.min(),)), bcast=True
        )

        return all_min_radii.min()

    def get_max_radius(self) -> float:
        """
        This method returns the global maximum distance from any cell centroid to a face.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        import numpy as np

        all_max_radii = uw.utilities.gather_data(
            np.array((self._radii.max(),)), bcast=True
        )

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

        with self.access(tmp):
            tmp.data[...] = uw.function.evaluate(
                uw_function, tmp.coords, basis
            ).reshape(-1, 1)

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

        with self.access(meshVar):
            meshVar.data[...] = 0.0
            if point_indices is not None:
                meshVar.data[point_indices] = 1.0

        return meshVar


## Here we check the existence of the meshVariable and so on before defining a new one
## (and potentially losing the handle to the old one)


def MeshVariable(
    varname: Union[str, list],
    mesh: "Mesh",
    num_components: Union[int, tuple] = None,
    vtype: Optional["uw.VarType"] = None,
    degree: int = 1,
    continuous: bool = True,
    varsymbol: Union[str, list] = None,
):
    """
    The MeshVariable class generates a variable supported by a finite element mesh and the
    underlying sympy representation that makes it possible to construct expressions that
    depend on the values of the MeshVariable.

    To set / read nodal values, use the numpy interface via the 'data' property.

    Parameters
    ----------
    varname :
        A text name for this variable. Use an R-string if a latex-expression is used
    mesh :
        The supporting underworld mesh.
    num_components :
        The number of components this variable has.
        For example, scalars will have `num_components=1`,
        while a 2d vector would have `num_components=2`.
    vtype :
        Optional. The underworld variable type for this variable.
        If not defined it will be inferred from `num_components`
        if possible.
    degree :
        The polynomial degree for this variable.
    varsymbol:
        Over-ride the varname with a symbolic form for printing etc (latex). Should be an R-string.

    """

    if isinstance(varname, list):
        name = varname[0] + R"+ \dots"
    else:
        name = varname

    ## Smash if already defined (we need to check this BEFORE the old meshVariable object is destroyed)

    import re

    clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    if clean_name in mesh.vars.keys():
        print(f"Variable with name {name} already exists on the mesh - Skipping.")
        return mesh.vars[clean_name]

    if mesh._accessed:
        ## Before adding a new variable, we first snapshot the data from the mesh.dm
        ## (if not accessed, then this will not be necessary and may break)

        mesh.update_lvec()

        old_gvec = mesh.dm.getGlobalVec()
        mesh.dm.localToGlobal(mesh._lvec, old_gvec, addv=False)

    new_meshVariable = _MeshVariable(
        name, mesh, num_components, vtype, degree, continuous, varsymbol
    )

    if mesh._accessed:
        ## Recreate the mesh variable dm and restore the data

        dm0 = mesh.dm
        dm1 = mesh.dm.clone()
        dm0.copyFields(dm1)
        dm1.createDS()

        mdm_is, subdm = dm1.createSubDM(range(0, dm1.getNumFields() - 1))

        mesh._lvec.destroy()
        mesh._lvec = dm1.createLocalVec()
        new_gvec = dm1.getGlobalVec()
        new_gvec_sub = new_gvec.getSubVector(mdm_is)

        # Copy the array data and push to gvec
        new_gvec_sub.array[...] = old_gvec.array[...]
        new_gvec.restoreSubVector(mdm_is, new_gvec_sub)

        # Copy the data to mesh._lvec and delete gvec
        dm1.globalToLocal(new_gvec, mesh._lvec)

        dm1.restoreGlobalVec(new_gvec)
        dm0.restoreGlobalVec(old_gvec)

        # destroy old dm
        dm0.destroy

        # Set new dm on mesh
        mesh.dm = dm1
        mesh.dm_hierarchy[-1] = dm1

    return new_meshVariable


class _MeshVariable(Stateful, uw_object):
    """
    The MeshVariable class generates a variable supported by a finite element mesh and the
    underlying sympy representation that makes it possible to construct expressions that
    depend on the values of the MeshVariable.

    To set / read nodal values, use the numpy interface via the 'data' property.

    Parameters
    ----------
    varname :
        A text name for this variable. Use an R-string if a latex-expression is used
    mesh :
        The supporting underworld mesh.
    num_components :
        The number of components this variable has.
        For example, scalars will have `num_components=1`,
        while a 2d vector would have `num_components=2`.
    vtype :
        Optional. The underworld variable type for this variable.
        If not defined it will be inferred from `num_components`
        if possible.
    degree :
        The polynomial degree for this variable.
    varsymbol:
        Over-ride the varname with a symbolic form for printing etc (latex). Should be an R-string.

    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        varname: Union[str, list],
        mesh: "underworld.mesh.Mesh",
        size: Union[int, tuple],
        vtype: Optional["underworld.VarType"] = None,
        degree: int = 1,
        continuous: bool = True,
        varsymbol: Union[str, list] = None,
    ):
        """
        The MeshVariable class generates a variable supported by a finite element mesh and the
        underlying sympy representation that makes it possible to construct expressions that
        depend on the values of the MeshVariable.

        To set / read nodal values, use the numpy interface via the 'data' property.

        Parameters
        ----------
        varname :
            A text name for this variable. Use an R-string if a latex-expression is used
        mesh :
            The supporting underworld mesh.
        num_components :
            The number of components this variable has.
            For example, scalars will have `num_components=1`,
            while a 2d vector would have `num_components=2`.
        vtype :
            Optional. The underworld variable type for this variable.
            If not defined it will be inferred from `num_components`
            if possible.
        degree :
            The polynomial degree for this variable.
        continuous:
            True for continuous element discretisation across element boundaries.
            False for discontinuous values across element boundaries.
        varsymbol :
            Over-ride the varname with a symbolic form for printing etc (latex). Should be an R-string.
        """

        import re
        import math

        # if varsymbol is None and not isinstance(varname, list):
        #     varsymbol = "{" + repr(varname)[1:-1] + "}"

        if isinstance(varname, list):
            name = varname[0] + " ... "
            symbol = "{" + varname[0] + R"\cdots" + "}"
        else:
            name = varname
            if varsymbol is not None:
                symbol = "{" + varsymbol + "}"
            else:
                symbol = "{" + varname + "}"

        self._lvec = None
        self._gvec = None
        self._data = None

        self._is_accessed = False
        self._available = False

        ## Note sympy needs a unique symbol even across different meshes
        ## or it will get confused when it clones objects. We try this: add
        ## a label to the variable that is not rendered - CHECK this works !!!

        self.name = name
        self.symbol = symbol

        if mesh.instance_number > 1:
            invisible = rf"\hspace{{ {mesh.instance_number/100}pt }}"
            self.symbol = f"{{ {invisible} {symbol} }}"

        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

        # ToDo: Suggest we deprecate this and require it to be set explicitly
        # The tensor types are hard to infer correctly

        if vtype == None:
            if isinstance(size, int) and size == 1:
                vtype = uw.VarType.SCALAR
            elif isinstance(size, int) and size == mesh.dim:
                vtype = uw.VarType.VECTOR
            elif isinstance(size, tuple):
                if size[0] == mesh.dim and size[1] == mesh.dim:
                    vtype = uw.VarType.TENSOR
                else:
                    vtype = uw.VarType.MATRIX
            else:
                raise ValueError(
                    "Unable to infer variable type from `num_components`. Please explicitly set the `vtype` parameter."
                )

        if not isinstance(vtype, uw.VarType):
            raise ValueError(
                "'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`."
            )

        self.vtype = vtype
        self.mesh = mesh
        self.shape = size
        self.degree = degree
        self.continuous = continuous

        # First create the petsc FE object of the
        # correct size / dimension to represent the
        # unknowns when used in computations (for tensors)
        # we will need to pack them correctly as well
        # (e.g. T.sym.reshape(1,len(T.sym))))
        # Symmetric tensors ... a bit more work again

        if vtype == uw.VarType.SCALAR:
            self.shape = (1, 1)
            self.num_components = 1
        elif vtype == uw.VarType.VECTOR:
            self.shape = (1, mesh.dim)
            self.num_components = mesh.dim
        elif vtype == uw.VarType.TENSOR:
            self.num_components = mesh.dim * mesh.dim
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.SYM_TENSOR:
            self.num_components = math.comb(mesh.dim + 1, 2)
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.MATRIX:
            self.num_components = self.shape[0] * self.shape[1]

        self._data_container = numpy.empty(self.shape, dtype=object)

        # create associated sympy function
        from underworld3.function import UnderworldFunction

        if vtype == uw.VarType.SCALAR:
            self._sym = sympy.Matrix.zeros(1, 1)
            self._sym[0] = UnderworldFunction(
                self.symbol,
                self,
                vtype,
                0,
                0,
            )(*self.mesh.r)
            self._ijk = self._sym[0]

        elif vtype == uw.VarType.VECTOR:
            self._sym = sympy.Matrix.zeros(1, mesh.dim)
            for comp in range(mesh.dim):
                self._sym[0, comp] = UnderworldFunction(
                    self.symbol,
                    self,
                    vtype,
                    comp,
                    comp,
                )(*self.mesh.r)

            self._ijk = sympy.vector.matrix_to_vector(self._sym, self.mesh.N)

        elif vtype == uw.VarType.TENSOR:
            self._sym = sympy.Matrix.zeros(mesh.dim, mesh.dim)

            # Matrix form (any number of components)
            for i in range(mesh.dim):
                for j in range(mesh.dim):
                    self._sym[i, j] = UnderworldFunction(
                        self.symbol,
                        self,
                        vtype,
                        (i, j),
                        self._data_layout(i, j),
                    )(*self.mesh.r)

        elif vtype == uw.VarType.SYM_TENSOR:
            self._sym = sympy.Matrix.zeros(mesh.dim, mesh.dim)

            # Matrix form (any number of components)
            for i in range(mesh.dim):
                for j in range(0, mesh.dim):
                    if j >= i:
                        self._sym[i, j] = UnderworldFunction(
                            self.symbol,
                            self,
                            vtype,
                            (i, j),
                            self._data_layout(i, j),
                        )(*self.mesh.r)

                    else:
                        self._sym[i, j] = self._sym[j, i]

        elif vtype == uw.VarType.MATRIX:
            self._sym = sympy.Matrix.zeros(self.shape[0], self.shape[1])

            # Matrix form (any number of components)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self._sym[i, j] = UnderworldFunction(
                        self.symbol,
                        self,
                        vtype,
                        (i, j),
                        self._data_layout(i, j),
                    )(*self.mesh.r)

        # This allows us to define a __getitem__ method
        # to return a view for a given component when
        # the access manager is active

        from collections import namedtuple

        MeshVariable_ij = namedtuple("MeshVariable_ij", ["data", "sym"])

        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                self._data_container[i, j] = MeshVariable_ij(
                    data=f"MeshVariable[...].data is only available within mesh.access() context",
                    sym=self.sym[i, j],
                )

        super().__init__()

        self.mesh.vars[self.clean_name] = self
        self._setup_ds()

        # Setup public view of data
        self._array = self._array_data_structure(self)

        return

    def _object_viewer(self):
        """This will substitute specific information about this object"""
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        # feedback on this instance

        display(
            Markdown(f"**MeshVariable:**"),
            Markdown(
                f"""\
  > symbol:  ${self.symbol}$\n
  > shape:   ${self.shape}$\n
  > degree:  ${self.degree}$\n
  > continuous:  `{self.continuous}`\n
  > type:    `{self.vtype.name}`"""
            ),
            Markdown(f"**FE Data:**"),
            Markdown(
                f"""
  > PETSc field id:  ${self.field_id}$ \n
  > PETSc field name:   `{self.clean_name}` """
            ),
        )

        with self.mesh.access():
            display(self.data),

        return

    def clone(self, name, varsymbol):
        newMeshVariable = MeshVariable(
            varname=name,
            mesh=self.mesh,
            num_components=self.shape,
            vtype=self.vtype,
            degree=self.degree,
            continuous=self.continuous,
            varsymbol=varsymbol,
        )

        return newMeshVariable

    # def __getitem__(self, indices):
    #     if not isinstance(indices, tuple):
    #         if isinstance(indices, int) and self.shape[0] == 1:
    #             i = 0
    #             j = indices
    #         else:
    #             raise IndexError(
    #                 "MeshVariable[i,j] access requires one or two indices "
    #             )
    #     else:
    #         i, j = indices

    #     return self._data_container[i, j]

    def pack(self, data_array):
        """Convert an array in the correct shape for the underlying variable into something that can be loaded into
        the flat storage structure used by PETSc in a numpy assigment (with index broadcasting etc)
        """

        import numpy as np

        shape = self.shape
        storage_size = self._data_layout(-1)
        data_array_2d = np.atleast_2d(data_array)

        with self.mesh.access(self):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ij = self._data_layout(i, j)
                    self.data[:, ij] = data_array_2d[:, i, j]

        return

    def unpack(self, squeeze=True):
        """Return an array in the correct shape for the underlying variable from
        the flat storage structure used by PETSc. By default, use numpy squeeze to remove additional
        dimensions (keep those dimensions to leave all data as 3D array - scalars being shape (1,1), vectors
        being (1,dim) and so on)
        """

        import numpy as np

        shape = self.shape

        with self.mesh.access(self):
            points = self._data.shape[0]
            data_array_3d = np.empty(shape=(points, *shape))

            for i in range(shape[0]):
                for j in range(shape[1]):
                    ij = self._data_layout(i, j)
                    data_array_3d[:, i, j] = self._data[:, ij]

        if squeeze:
            return data_array_3d.squeeze()
        else:
            return data_array_3d

    def rbf_interpolate(
        self, new_coords, meth=0, p=2, verbose=False, nnn=None, rubbish=None
    ):
        # An inverse-distance mapping is quite robust here ... as long
        # as long we take care of the case where some nodes coincide (likely if used mesh2mesh)

        import numpy as np

        if nnn == None:
            if self.mesh.dim == 3:
                nnn = 4
            else:
                nnn = 3

        with self.mesh.access(self):
            D = self.data.copy()

        if verbose and uw.mpi.rank == 0:
            print("Building K-D tree", flush=True)

        mesh_kdt = uw.kdtree.KDTree(self.coords)
        values = mesh_kdt.rbf_interpolator_local(
            new_coords, D, nnn, p=p, verbose=verbose
        )
        del mesh_kdt

        return values

    @timing.routine_timer_decorator
    def save(
        self,
        filename: str,
        name: Optional[str] = None,
        index: Optional[int] = None,
    ):
        """
        Append variable data to the specified mesh hdf5
        data file. The file must already exist.

        Parameters
        ----------
        filename :
            The filename of the mesh checkpoint file. It
            must already exist.
        name :
            Textual name for dataset. In particular, this
            will be used for XDMF generation. If not
            provided, the variable name will be used.
        index :
            Not currently supported. An optional index which
            might correspond to the timestep (for example).
        """

        self._set_vec(available=False)

        viewer = PETSc.ViewerHDF5().create(filename, "a", comm=PETSc.COMM_WORLD)
        if index:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via
            ## the PETSc xdmf script.
            # PetscViewerHDF5PushTimestepping(cviewer)
            # viewer.setTimestep(index)

        if name:
            oldname = self._gvec.getName()
            self._gvec.setName(name)
        viewer(self._gvec)
        if name:
            self._gvec.setName(oldname)

        lvec = self.mesh.dm.getCoordinates()

    # ToDo: rename to vertex_checkpoint (or similar)
    @timing.routine_timer_decorator
    def write(
        self,
        filename: str,
    ):
        """
        Write variable data to the specified mesh hdf5
        data file. The file will be over-written.

        Parameters
        ----------
        filename :
            The filename of the mesh checkpoint file
        """

        self._set_vec(available=False)

        # Variable coordinates - let's put those in the file to
        # make it a standalone "swarm"

        dmold = self.mesh.dm.getCoordinateDM()
        dmold.createDS()
        dmnew = dmold.clone()

        options = PETSc.Options()
        options["coordinterp_petscspace_degree"] = self.degree
        options["coordinterp_petscdualspace_lagrange_continuity"] = self.continuous
        options["coordinterp_petscdualspace_lagrange_node_endpoints"] = False

        dmfe = PETSc.FE().createDefault(
            self.mesh.dim,
            self.mesh.cdim,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            "coordinterp_",
            PETSc.COMM_SELF,
        )

        dmnew.setField(0, dmfe)
        dmnew.createDS()

        lvec = dmnew.getLocalVec()
        gvec = dmnew.getGlobalVec()

        lvec.array[...] = self.coords.reshape(-1)[...]
        dmnew.localToGlobal(lvec, gvec, addv=False)
        gvec.setName("coordinates")

        # Check that this is also synchronised
        # self.mesh.dm.localToGlobal(self._lvec, self._gvec, addv=False)

        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=PETSc.COMM_WORLD)
        viewer(self._gvec)
        viewer(gvec)

        dmnew.restoreGlobalVec(gvec)
        dmnew.restoreLocalVec(lvec)

        uw.mpi.barrier()
        viewer.destroy()
        dmfe.destroy()

        return

    @timing.routine_timer_decorator
    def read_timestep(
        self,
        data_filename,
        data_name,
        index,
        outputPath="",
        verbose=False,
    ):
        """
        Read a mesh variable from an arbitrary vertex-based checkpoint file
        and reconstruct/interpolate the data field accordingly. The data sizes / meshes can be
        different and will be matched using a kd-tree / inverse-distance weighting
        to the new mesh.

        """

        # Fix this to match the write_timestep function

        # mesh.write_timestep( "test", meshUpdates=False, meshVars=[X], outputPath="", index=0)
        # swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath="", index=0)

        output_base_name = os.path.join(outputPath, data_filename)
        data_file = output_base_name + f".mesh.{data_name}.{index:05}.h5"

        # check if data_file exists
        if os.path.isfile(os.path.abspath(data_file)):
            pass
        else:
            raise RuntimeError(f"{os.path.abspath(data_file)} does not exist")

        import h5py
        import numpy as np

        self._set_vec(available=False)

        ## Sub functions that are used to read / interpolate the mesh.
        def field_from_checkpoint(
            data_file=None,
            data_name=None,
        ):
            """Read the mesh data as a swarm-like value"""

            if verbose and uw.mpi.rank == 0:
                print(f"Reading data file {data_file}", flush=True)

            h5f = h5py.File(data_file)
            D = h5f["fields"][data_name][()].reshape(-1, self.shape[1])
            X = h5f["fields"]["coordinates"][()].reshape(-1, self.mesh.dim)

            h5f.close()

            if len(D.shape) == 1:
                D = D.reshape(-1, 1)

            return X, D

        def map_to_vertex_values(X, D, nnn=4, p=2, verbose=False):
            # Map from "swarm" of points to nodal points
            # This is a permutation if we building on the checkpointed
            # mesh file

            mesh_kdt = uw.kdtree.KDTree(X)

            return mesh_kdt.rbf_interpolator_local(self.coords, D, nnn, p, verbose)

        def values_to_mesh_var(mesh_variable, Values):
            mesh = mesh_variable.mesh

            # This should be trivial but there may be problems if
            # the kdtree does not have enough neighbours to allocate
            # values for every point. We handle that here.

            with mesh.access(mesh_variable):
                mesh_variable.data[...] = Values[...]

            return

        ## Read file information

        X, D = field_from_checkpoint(
            data_file,
            data_name,
        )

        remapped_D = map_to_vertex_values(X, D)

        # This is empty at the moment
        values_to_mesh_var(self, remapped_D)

        return

    @timing.routine_timer_decorator
    def load_from_h5_plex_vector(
        self,
        filename: str,
        data_name: Optional[str] = None,
    ):
        if data_name is None:
            data_name = self.clean_name

        with self.mesh.access(self):
            indexset, subdm = self.mesh.dm.createSubDM(self.field_id)

            old_name = self._gvec.getName()
            viewer = PETSc.ViewerHDF5().create(filename, "r", comm=PETSc.COMM_WORLD)

            self._gvec.setName(data_name)
            self._gvec.load(viewer)
            self._gvec.setName(old_name)

            subdm.globalToLocal(self._gvec, self._lvec, addv=False)

            viewer.destroy()

        return

    @property
    def fn(self) -> sympy.Basic:
        """
        The handle to the (i,j,k) spatial view of this variable if it exists (deprecated)
        """
        return self._ijk

    @property
    def ijk(self) -> sympy.Basic:
        """
        The handle to the (i,j,k) spatial view of this variable if it exists
        """
        return self._ijk

    @property
    def sym(self) -> sympy.Basic:
        """
        The handle to the sympy.Matrix view of this variable
        """
        return self._sym

    @property
    def sym_1d(self) -> sympy.Basic:
        """
        The handle to a flattened version of the sympy.Matrix view of this variable.
        Assume components are stored in the same order that sympy iterates entries in
        a matrix except for the symmetric tensor case where we store in a Voigt form
        """

        if self.vtype != uw.VarType.SYM_TENSOR:
            return self._sym.reshape(1, len(self._sym))
        else:
            if self.mesh.dim == 2:
                return sympy.Matrix(
                    [
                        self._sym[0, 0],
                        self._sym[1, 1],
                        self._sym[0, 1],
                    ]
                ).T
            else:
                return sympy.Matrix(
                    [
                        self._sym[0, 0],
                        self._sym[1, 1],
                        self._sym[2, 2],
                        self._sym[0, 1],
                        self._sym[0, 2],
                        self._sym[1, 2],
                    ]
                ).T

    def _data_layout(self, i, j=None):
        # mapping

        if self.vtype == uw.VarType.SCALAR:
            return 0
        if self.vtype == uw.VarType.VECTOR:
            if i < 0 or j < 0:
                return self.mesh.dim
            else:
                if j is None:
                    return i
                elif i == 0:
                    return j
                else:
                    raise IndexError(
                        f"Vectors have shape {self.mesh.dim} or {(1, self.mesh.dim)} "
                    )
        if self.vtype == uw.VarType.TENSOR:
            if self.mesh.dim == 2:
                if i < 0 or j < 0:
                    return 4
                else:
                    return ((0, 1), (2, 3))[i][j]
            else:
                if i < 0 or j < 0:
                    return 9
                else:
                    return ((0, 1, 2), (3, 4, 5), (6, 7, 8))[i][j]

        if self.vtype == uw.VarType.SYM_TENSOR:
            if self.mesh.dim == 2:
                if i < 0 or j < 0:
                    return 3
                else:
                    return ((0, 2), (2, 1))[i][j]
            else:
                if i < 0 or j < 0:
                    return 6
                else:
                    return ((0, 3, 4), (3, 1, 5), (4, 5, 2))[i][j]

        if self.vtype == uw.VarType.MATRIX:
            if i < 0 or j < 0:
                return self.shape[0] * self.shape[1]
            else:
                return i + j * self.shape[0]

    def _setup_ds(self):
        options = PETSc.Options()
        name0 = "VAR"  # self.clean_name ## Filling up the options database
        options.setValue(f"{name0}_petscspace_degree", self.degree)
        options.setValue(f"{name0}_petscdualspace_lagrange_continuity", self.continuous)
        options.setValue(
            f"{name0}_petscdualspace_lagrange_node_endpoints", False
        )  # only active if discontinuous

        dim = self.mesh.dm.getDimension()
        petsc_fe = PETSc.FE().createDefault(
            dim,
            self.num_components,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            name0 + "_",
            PETSc.COMM_SELF,
        )

        self.field_id = self.mesh.dm.getNumFields()
        self.mesh.dm.addField(petsc_fe)
        field, _ = self.mesh.dm.getField(self.field_id)
        field.setName(self.clean_name)
        self.mesh.dm.createDS()

        return

    def _set_vec(self, available):
        if self._lvec == None:
            indexset, subdm = self.mesh.dm.createSubDM(self.field_id)

            self._lvec = subdm.createLocalVector()
            self._lvec.zeroEntries()  # not sure if required, but to be sure.
            self._gvec = subdm.createGlobalVector()
            self._gvec.setName(self.clean_name)  # This is set for checkpointing.
            self._gvec.zeroEntries()

        self._available = available

    def __del__(self):
        if self._lvec:
            self._lvec.destroy()
        if self._gvec:
            self._gvec.destroy()

    @property
    def vec(self) -> PETSc.Vec:
        """
        The corresponding PETSc local vector for this variable.
        """
        if not self._available:
            raise RuntimeError(
                "Vector must be accessed via the mesh `access()` context manager."
            )
        return self._lvec

    @property
    def data(self) -> numpy.ndarray:
        """
        Numpy proxy array to underlying variable data.
        Note that the returned array is a proxy for all the *local* nodal
        data, and is provided as 1d list.

        For both read and write, this array can only be accessed via the
        mesh `access()` context manager.
        """
        if self._data is None:
            raise RuntimeError(
                "Data must be accessed via the mesh `access()` context manager."
            )
        return self._data

    class _array_data_structure(object):
        """This is used to add getitem / setitem to the array property of the variable"""

        def __init__(inner_self, owner):
            inner_self.owner = owner

        # Unreachable
        # def __set__(inner_self, instance, array_value):
        #     inner_self.owner.pack(array_value)

        def __setitem__(inner_self, key, value):
            var_data_copy = inner_self.owner.unpack(squeeze=False)
            var_data_copy[key] = value
            inner_self.owner.pack(var_data_copy)

        def __getitem__(inner_self, key):
            var_data_copy = inner_self.owner.unpack(squeeze=False)
            return var_data_copy[key]

        def __repr__(inner_self):
            var_data_copy = inner_self.owner.unpack(squeeze=False)
            return var_data_copy.__repr__()

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, array_value):
        self.pack(array_value)

    ## ToDo: We should probably deprecate this in favour of using integrals

    def min(self) -> Union[float, tuple]:
        """
        The global variable minimum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.min()
        else:
            return tuple(
                [self._gvec.strideMin(i)[1] for i in range(self.num_components)]
            )

    def max(self) -> Union[float, tuple]:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.max()
        else:
            return tuple(
                [self._gvec.strideMax(i)[1] for i in range(self.num_components)]
            )

    def sum(self) -> Union[float, tuple]:
        """
        The global variable sum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.sum()
        else:
            cpts = []
            for i in range(0, self.num_components):
                cpts.append(self._gvec.strideSum(i))

            return tuple(cpts)

    def norm(self, norm_type) -> Union[float, tuple]:
        """
        The global variable norm value.

        norm_type: type of norm, one of
            - 0: NORM 1 ||v|| = sum_i | v_i |. ||A|| = max_j || v_*j ||
            - 1: NORM 2 ||v|| = sqrt(sum_i |v_i|^2) (vectors only)
            - 3: NORM INFINITY ||v|| = max_i |v_i|. ||A|| = max_i || v_i* ||, maximum row sum
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components > 1 and norm_type == 2:
            raise RuntimeError("Norm 2 is only available for vectors.")

        if self.num_components == 1:
            return self._gvec.norm(norm_type)
        else:
            return tuple(
                [
                    self._gvec.strideNorm(i, norm_type)
                    for i in range(self.num_components)
                ]
            )

    def mean(self) -> Union[float, tuple]:
        """
        The global variable mean value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            vecsize = self._gvec.getSize()
            return self._gvec.sum() / vecsize
        else:
            vecsize = self._gvec.getSize() / self.num_components
            return tuple(
                [self._gvec.strideSum(i) / vecsize for i in range(self.num_components)]
            )

    def stats(self):
        """
        The equivalent of mesh.stats but using the native coordinates for this variable
        Not set up for vector variables so we just skip that for now.

        Returns various norms on the mesh using the native mesh discretisation for this
        variable. It is a wrapper on the various _gvec stats routines for the variable.

          - size
          - mean
          - min
          - max
          - sum
          - L2 norm
          - rms
        """

        if self.num_components > 1:
            raise NotImplementedError(
                "stats not available for multi-component variables"
            )

        #       This uses a private work MeshVariable and the various norms defined there but
        #       could either be simplified to just use petsc vectors, or extended to
        #       compute integrals over the elements which is in line with uw1 and uw2

        from petsc4py.PETSc import NormType

        vsize = self._gvec.getSize()
        vmean = self.mean()
        vmax = self.max()[1]
        vmin = self.min()[1]
        vsum = self.sum()
        vnorm2 = self.norm(NormType.NORM_2)
        vrms = vnorm2 / numpy.sqrt(vsize)

        return vsize, vmean, vmin, vmax, vsum, vnorm2, vrms

    @property
    def coords(self) -> numpy.ndarray:
        """
        The array of variable vertex coordinates.
        """
        return self.mesh._get_coords_for_var(self)

    # vector calculus routines - the advantage of using these inbuilt routines is
    # that they are tied to the appropriate mesh definition.

    def divergence(self):
        try:
            return self.mesh.vector.divergence(self.sym)
        except:
            return None

    def gradient(self):
        try:
            return self.mesh.vector.gradient(self.sym)
        except:
            return None

    def curl(self):
        try:
            return self.mesh.vector.curl(self.sym)
        except:
            return None

    def jacobian(self):
        ## validate if this is a vector ?
        return self.mesh.vector.jacobian(self.sym)


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
        with h5py.File(h5_filename, "r") as f:
            size = f[f"cell_fields/{mesh_var.clean_name}_{mesh_var.clean_name}"].shape[
                0
            ]
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
