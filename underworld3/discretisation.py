# from telnetlib import DM
# from this import d
from typing import Optional, Tuple, Union
import os
from xmlrpc.client import Boolean
import mpi4py
import numpy
import sympy
import sympy.vector
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities import _api_tools
from underworld3.coordinates import CoordinateSystem, CoordinateSystemType
from underworld3.cython import petsc_discretisation


import underworld3.timing as timing
import weakref
from collections import namedtuple


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
        # print(f"Processing gmsh file {filename}")

        plex_0 = PETSc.DMPlex().createFromFile(filename, comm=PETSc.COMM_SELF)

        plex_0.setName("uw_mesh")
        plex_0.markBoundaryFaces("All_Boundaries", 1001)

        viewer = PETSc.ViewerHDF5().create(filename + ".h5", "w", comm=PETSc.COMM_SELF)
        viewer(plex_0)
        viewer.destroy()

        # print(f"Mesh saved to {filename}.h5")

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


class Mesh(_api_tools.Stateful):

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
        boundaries=None,
        distribute=True,
        *args,
        **kwargs,
    ):

        self.instance = Mesh.mesh_instances
        Mesh.mesh_instances += 1

        if distribute:
            comm = PETSc.COMM_WORLD
        else:
            comm = PETSc.COMM_SELF

        if isinstance(plex_or_meshfile, PETSc.DMPlex):
            name = "plexmesh"
            self.dm = plex_or_meshfile
            self.sf0 = None  # Should we build one ?
        else:
            comm = kwargs.get("comm", PETSc.COMM_WORLD)
            name = plex_or_meshfile
            basename, ext = os.path.splitext(plex_or_meshfile)

            # Note: should be able to handle a .geo as well on this pathway
            if ext.lower() == ".msh":
                self.sf0, self.dm = _from_gmsh(
                    plex_or_meshfile,
                    comm,
                    markVertices=markVertices,
                    useRegions=useRegions,
                    useMultipleTags=useMultipleTags,
                )
            elif ext.lower() == ".h5":
                self.sf0, self.dm = _from_plexh5(
                    plex_or_meshfile, PETSc.COMM_SELF, return_sf=True
                )

            else:
                raise RuntimeError(
                    "Mesh file %s has unknown format '%s'."
                    % (plex_or_meshfile, ext[1:])
                )

        self.filename = filename
        self.boundaries = boundaries
        self.refinement_callback = refinement_callback

        self.dm0 = self.dm.clone()
        self.sf1 = None

        if distribute:
            self.sf1 = self.dm.distribute()
        else:
            print("Warning - not distributing mesh !!", flush=True)

        ## This is where we can refine the dm if required, and rebuild

        if isinstance(refinement, int):
            for i in range(refinement):
                self.dm = self.dm.refine()

        # This will be done anyway - the mesh maybe in a
        # partially adapted state

        if callable(refinement_callback):
            refinement_callback(self.dm)

        ## Do we want to save the refined mesh or allow it to be reconstructed
        ## from the coarse mesh ?

        if self.sf1 and self.sf0:
            self.sf = self.sf0.compose(self.sf1)
        else:
            self.sf = self.sf0  # could be None !

        self.dm.setName("uw_mesh")

        # Set sympy constructs. First a generic, symbolic, Cartesian coordinate system
        from sympy.vector import CoordSys3D

        # A unique set of vectors / names for each mesh instance
        self._N = CoordSys3D(f"N")

        self._N.x._latex_form = r"\mathrm{\xi_0}"
        self._N.y._latex_form = r"\mathrm{\xi_1}"
        self._N.z._latex_form = r"\mathrm{\xi_2}"
        self._N.i._latex_form = r"\mathbf{\hat{\mathbf{e}}_0}"
        self._N.j._latex_form = r"\mathbf{\hat{\mathbf{e}}_1}"
        self._N.k._latex_form = r"\mathbf{\hat{\mathbf{e}}_2}"

        # Now add the appropriate coordinate system for the mesh's natural geometry
        # This step will usually over-write the defaults we just defined
        self._CoordinateSystem = CoordinateSystem(self, coordinate_system_type)

        # Tidy some of this printing without changing the
        # underlying vector names (as these are part of the code generation system)

        try:
            self.isSimplex = self.dm.isSimplex()
        except:
            self.isSimplex = simplex

        # Use grid hashing for point location
        options = PETSc.Options()
        options["dm_plex_hash_location"] = None
        self.dm.setFromOptions()

        # self._vars = weakref.WeakValueDictionary()
        self._vars = {}
        self._block_vars = {}

        # a list of equation systems that will
        # need to be rebuilt if the mesh coordinates change

        self._equation_systems_register = []

        self._accessed = False
        self._quadrature = False
        self._stale_lvec = True
        self._lvec = None
        self.petsc_fe = None

        self.degree = degree
        self.qdegree = qdegree

        self.nuke_coords_and_rebuild()

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

    def view(self):
        if uw.mpi.rank == 0:

            print(f"Mesh {self.instance}")

            if len(self.vars) > 0:
                print(f"| Variable Name       | component | degree | type        |")
                print(f"| ------------------------------------------------------ |")
                for vname in self.vars.keys():
                    v = self.vars[vname]
                    print(
                        f"| {v.clean_name:<20}|{v.num_components:^10} |{v.degree:^7} | {v.vtype.name:^9} |"
                    )

                print(
                    f"| ------------------------------------------------------ |",
                    flush=True,
                )

    def nuke_coords_and_rebuild(self):

        # This is a reversion to the old version (3.15 compatible which seems to work in 3.16 too)

        self._coord_array = {}

        # let's go ahead and do an initial projection from linear (the default)
        # to linear. this really is a nothing operation, but a
        # side effect of this operation is that coordinate DM DMField is
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates.

        options = PETSc.Options()
        options.setValue(
            "meshproj_{}_petscspace_degree".format(self.mesh_instances), self.degree
        )

        self.petsc_fe = PETSc.FE().createDefault(
            self.dim,
            self.cdim,
            self.isSimplex,
            self.qdegree,
            "meshproj_{}_".format(self.mesh_instances),
            PETSc.COMM_WORLD,
        )

        ## LM ToDo: check if this is still a valid issue under 3.18.x / 3.19.x
        if self.degree != 1:

            # We have to be careful as a projection onto an equivalent PETScFE can cause problematic
            # issues with petsc that we see in parallel - in which case there is a fallback, pass no
            # PETScFE and let PETSc decide. Note that the petsc4py wrapped version does not allow this
            # (but it should !)

            self.dm.projectCoordinates(self.petsc_fe)

        else:

            uw.cython.petsc_discretisation.petsc_dm_project_coordinates(self.dm)

        # now set copy of this array into dictionary

        arr = self.dm.getCoordinatesLocal().array

        key = (
            self.isSimplex,
            self.degree,
            True,
        )  # True here assumes continuous basis for coordinates ...

        self._coord_array[key] = arr.reshape(-1, self.cdim).copy()

        # invalidate the cell-search k-d tree and the mesh centroid data / rebuild
        self._index = None
        self._radii, self._centroids, self._search_lengths = self._get_mesh_sizes()

        return

    @timing.routine_timer_decorator
    def update_lvec(self):
        """
        This method creates and/or updates the mesh variable local vector.
        If the local vector is already up to date, this method will do nothing.
        """

        if self._stale_lvec:
            if not self._lvec:
                # self.dm.clearDS()
                # self.dm.createDS()
                # create the local vector (memory chunk) and attach to original dm
                self._lvec = self.dm.createLocalVec()
            # push avar arrays into the parent dm array
            a_global = self.dm.getGlobalVec()
            names, isets, dms = self.dm.createFieldDecomposition()

            with self.access():
                # traverse subdms, taking user generated data in the subdm
                # local vec, pushing it into a global sub vec
                for var, subiset, subdm in zip(self.vars.values(), isets, dms):
                    lvec = var.vec
                    subvec = a_global.getSubVector(subiset)
                    subdm.localToGlobal(lvec, subvec, addv=False)
                    a_global.restoreSubVector(subiset, subvec)

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

    def deform_mesh(self, new_coords: numpy.ndarray):
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

        for eq_system in self._equation_systems_register:
            eq_system._rebuild_after_mesh_update()

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

        # This is not needed if we can index into the block variables
        # for block_var in self.block_vars.values():
        #     size = block_var.shape
        #     for i in range(0, size[0]):
        #         for j in range(0, size[1]):
        #             block_var._data[i, j] = block_var._vars[i, j].data

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

                        # subdm.destroy()
                        self.mesh._stale_lvec = True

                    var._data = None
                    var._set_vec(available=False)
                    var._is_accessed = False

                    # This is not needed if we can index into the block variables
                    # for block_var in self.mesh.block_vars.values():
                    #     size = block_var.shape
                    #     for i in range(0, size[0]):
                    #         for j in range(0, size[1]):
                    #             block_var._data[i, j] = None

                for i in range(0, var.shape[0]):
                    for j in range(0, var.shape[1]):
                        # var._data_ij[i, j] = None
                        var._data_container[i, j] = var._data_container[i, j]._replace(
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
    def write_timestep_xdmf(
        self,
        filename: str,
        meshUpdates: bool = True,
        meshVars: Optional[list] = [],
        swarmVars: Optional[list] = [],
        index: Optional[int] = 0,
    ):
        """
        Write the selected mesh, variables and swarm variables (as proxies) for later visualisation.
        An xdmf file is generated and the overall package can then be read by paraview or pyvista.
        Vertex values (on the mesh points) are stored for all variables regardless of their interpolation order"""

        options = PETSc.Options()
        options.setValue("viewer_hdf5_sp_output", True)
        # options.setValue("viewer_hdf5_collective", True)

        # Checkpoint the mesh file itself if required

        if not meshUpdates:
            from pathlib import Path

            mesh_file = filename + ".mesh.0.h5"
            path = Path(mesh_file)
            if not path.is_file():
                self.save(mesh_file)

        else:
            if uw.mpi.rank == 0:
                print("Saving mesh file")
            self.save(filename + f".mesh.{index}.h5")

        if meshVars is not None:
            for var in meshVars:
                save_location = filename + f".{var.clean_name}.{index}.h5"
                var.simple_save(save_location)

        if swarmVars is not None:
            for svar in swarmVars:
                save_location = filename + f".proxy.{svar.clean_name}.{index}.h5"
                svar.simple_save(save_location)

        if uw.mpi.rank == 0:
            checkpoint_xdmf(
                filename,
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
        fname = f"./{outputPath}{'step_'}{index:04d}.h5"
        xfname = f"./{outputPath}{'step_'}{index:04d}.xdmf"
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
                self.save(mesh_file)

        else:
            if uw.mpi.rank == 0:
                print("Saving mesh file")
            self.save(filename + f".mesh.{index}.h5")

        # Checkpoint file

        if unique_id:
            checkpoint_file = filename + f"{uw.mpi.unique}.checkpoint.{index}.h5"
        else:
            checkpoint_file = filename + f".checkpoint.{index}.h5"

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
    def save(self, filename: str, index: Optional[int] = None):
        """
        Save mesh data to the specified hdf5 file.

        Users will generally create this file, and then
        append mesh variable data to it via the variable
        `save` method.

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

    def vtk(self, filename: str):
        """
        Save mesh to the specified file
        """

        viewer = PETSc.Viewer().createVTK(filename, "w", comm=PETSc.COMM_WORLD)
        viewer(self.dm)

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
            PETSc.COMM_WORLD,
        )

        dmnew.setField(0, dmfe)
        dmnew.createDS()

        matInterp, vecScale = dmold.createInterpolation(dmnew)
        coordsOld = self.dm.getCoordinates()
        coordsNewL = dmnew.getLocalVec()
        coordsNewG = matInterp * coordsOld
        dmnew.globalToLocal(coordsNewG, coordsNewL)

        arr = coordsNewL.array
        arrcopy = arr.reshape(-1, self.cdim).copy()

        return arrcopy

    def _build_kd_tree_index(self):

        if hasattr(self, "_index") and self._index is not None:
            return

        ## Bootstrapping - the kd-tree is needed to build the index but
        ## the index is also used in the kd-tree.

        from underworld3.swarm import Swarm, SwarmPICLayout

        # Create a temp swarm which we'll use to populate particles
        # at gauss points. These will then be used as basis for
        # kd-tree indexing back to owning cells.

        tempSwarm = Swarm(self)
        # 4^dim pop is used. This number may need to be considered
        # more carefully, or possibly should be coded to be set dynamically.

        # We can't use our own populate function since this needs THIS kd_tree to exist
        # We will need to use a standard layout instead

        tempSwarm.dm.finalizeFieldRegister()
        tempSwarm.dm.insertPointUsingCellDM(SwarmPICLayout.GAUSS.value, 3)

        with tempSwarm.access():
            # Build index on particle coords
            self._indexCoords = tempSwarm.particle_coordinates.data.copy()
            self._index = uw.kdtree.KDTree(self._indexCoords)
            self._index.build_index()

            # Grab mapping back to cell_ids.
            # Note that this is the numpy array that we eventually return from this
            # method. As such, we take measures to ensure that we use `numpy.int64` here
            # because we cast from this type in  `_function.evaluate` to construct
            # the PETSc cell-sf datasets, and if instead a `numpy.int32` is used it
            # will cause bugs that are difficult to find.

            self._indexMap = numpy.array(
                tempSwarm.particle_cellid.data[:, 0], dtype=numpy.int64
            )

        # Use the "OK" version above to find these lengths
        self._radii, self._centroids, self._search_lengths = self._get_mesh_sizes()

        ## Now, we can use this information to rebuild the index more carefully
        """
        tempSwarm2 = Swarm(self)
        tempSwarm2.populate(fill_param=4)

        with tempSwarm2.access():
            # Build index on particle coords
            self._indexCoords = tempSwarm2.particle_coordinates.data.copy()
            self._index = uw.kdtree.KDTree(self._indexCoords)
            self._index.build_index()

            # Grab mapping back to cell_ids.
            # Note that this is the numpy array that we eventually return from this
            # method. As such, we take measures to ensure that we use `numpy.int64` here
            # because we cast from this type in  `_function.evaluate` to construct
            # the PETSc cell-sf datasets, and if instead a `numpy.int32` is used it
            # will cause bugs that are difficult to find.

            self._indexMap = numpy.array(
                tempSwarm2.particle_cellid.data[:, 0], dtype=numpy.int64
            )

        # update these
        self._radii, self._centroids, self._search_lengths = self._get_mesh_sizes()
        """
        return

    @timing.routine_timer_decorator
    def get_closest_cells(self, coords: numpy.ndarray) -> numpy.ndarray:
        """
        This method uses a kd-tree algorithm to find the closest
        cells to the provided coords. For a regular mesh, this should
        be exactly the owning cell, but if the mesh is deformed, this
        is not guaranteed. Note, the nearest point does may not be all
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

        self._build_kd_tree_index()

        if len(coords) > 0:
            closest_points, dist, found = self._index.find_closest_point(coords)
        else:
            ### returns an empty array if no coords are on a proc
            closest_points, dist, found = False, False, numpy.array([None])

        if found.any() != None:
            if not numpy.allclose(found, True):
                raise RuntimeError(
                    "An error was encountered attempting to find the closest cells to the provided coordinates."
                )

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

        # Create index if required
        self._build_kd_tree_index()

        if len(coords) > 0:
            closest_points, dist, found = self._index.find_closest_point(coords)
        else:
            return -1

        # This is tuned a little bit so that points on a single CPU are never lost

        cells = self._indexMap[closest_points]
        invalid = (
            dist > 2.5 * self._search_lengths[cells]
        )  # 0.25 * self._radii[cells] ** 2
        cells[invalid] = -1

        return cells

    def _get_mesh_sizes(self):
        """
        Obtain the (local) mesh radii and centroids using
        This routine is called when the mesh is built / rebuilt

        """

        # Create index if required
        self._build_kd_tree_index()

        (
            sizes,
            centroids,
        ) = uw.cython.petsc_discretisation.petsc_fvm_get_local_cell_sizes(self)

        import numpy as np

        cStart, cEnd = self.dm.getHeightStratum(0)
        pStart, pEnd = self.dm.getDepthStratum(0)
        cell_length = np.empty_like(sizes)

        for cell in range(cEnd - cStart):

            cell_num_points = self.dm.getConeSize(cell)
            cell_points = self.dm.getTransitiveClosure(cell)[0][-cell_num_points:]
            cell_coords = self.data[cell_points - pStart]

            _, distsq, _ = self._index.find_closest_point(cell_coords)

            cell_length[cell] = distsq.max()

        return sizes, centroids, cell_length

    # ==========

    # Deprecated in favour of _get_mesh_sizes (above)
    def _get_mesh_centroids(self):
        """
        Obtain and cache the (local) mesh centroids using underworld swarm technology.
        This routine is called when the mesh is built / rebuilt

        The global cell number corresponding to a centroid is (supposed to be)
        self.dm.getCellNumbering().array.min() + index

        """

        (
            sizes,
            centroids,
        ) = uw.cython.petsc_discretisation.petsc_fvm_get_local_cell_sizes(self)

        return centroids

    def get_min_radius(self) -> float:
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

    # def get_boundary_subdm(self) -> PETSc.DM:
    #     """
    #     This method returns the boundary subdm that wraps DMPlexCreateSubmesh
    #     """
    #     from underworld3.petsc_discretisation import petsc_create_surface_submesh
    #     return petsc_create_surface_submesh(self, "Boundary", 666, )

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


## Here we check the existence of the meshVariable and so on before defining a new one
## (and potentially losing the handle to the old one)


def MeshVariable(
    varname: Union[str, list],
    mesh: "underworld.mesh.Mesh",
    num_components: Union[int, tuple] = None,
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
        A textual name for this variable.
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
        A symbolic form for printing etc (sympy / latex)

    """

    if isinstance(varname, list):
        name = varname[0] + R"+ \dots"
    else:
        name = varname

    ## Smash if already defined (we should check this BEFORE the old meshVariable object is destroyed)

    import re

    clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    if clean_name in mesh.vars.keys():
        print(f"Variable with name {name} already exists on the mesh - Skipping.")
        return mesh.vars[clean_name]

    if mesh._accessed:
        print(
            "It is not possible to add new variables to a mesh after existing variables have been accessed"
        )
        print(f"Variable {name} has NOT been added to mesh {mesh.instance}")
        return

    return _MeshVariable(
        varname, mesh, num_components, vtype, degree, continuous, varsymbol
    )


class _MeshVariable(_api_tools.Stateful):
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
            A textual name for this variable.

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
            A symbolic form for printing etc (sympy / latex)
        """

        import re
        import math

        if varsymbol is None:
            varsymbol = varname

        if isinstance(varname, list):
            name = varname[0] + R" ... "
            symbol = varsymbol[0] + R"\cdots"
        else:
            name = varname
            symbol = varsymbol

        self._lvec = None
        self._gvec = None
        self._data = None

        self._is_accessed = False
        self._available = False

        self.name = name
        self.symbol = symbol
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

        options = PETSc.Options()
        name0 = "VAR"  # self.clean_name ## Filling up the options database
        options.setValue(f"{name0}_petscspace_degree", degree)
        options.setValue(f"{name0}_petscdualspace_lagrange_continuity", continuous)
        options.setValue(
            f"{name0}_petscdualspace_lagrange_node_endpoints", False
        )  # only active if discontinuous

        dim = self.mesh.dm.getDimension()

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

        # self._data_ij = numpy.empty(self.shape, dtype=object)
        self._data_container = numpy.empty(self.shape, dtype=object)

        self.petsc_fe = PETSc.FE().createDefault(
            dim,
            self.num_components,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            name0 + "_",
            PETSc.COMM_WORLD,
        )

        self.field_id = self.mesh.dm.getNumFields()
        self.mesh.dm.setField(self.field_id, self.petsc_fe)
        field, _ = self.mesh.dm.getField(self.field_id)
        field.setName(self.clean_name)

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
            self._sym[0].mesh = self.mesh
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
                self._sym[0, comp].mesh = self.mesh

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
                    self._sym[i, j].mesh = self.mesh

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

                    self._sym[i, j].mesh = self.mesh

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
                    self._sym[i, j].mesh = self.mesh
                    n += 1

        # Suggest this should be deprecated - seems complicated
        # and liable to produce errors in development as well as use

        # elif (
        #     vtype == uw.VarType.COMPOSITE
        # ):  # This is just to allow full control over the names of the components
        #     self._sym = sympy.Matrix.zeros(1, num_components)
        #     if isinstance(varsymbol, list):
        #         if len(varsymbol) == num_components:
        #             for comp in range(num_components):
        #                 self._sym[0, comp] = UnderworldFunction(
        #                     varsymbol[comp],
        #                     self,
        #                     vtype,
        #                     comp,
        #                     comp,
        #                 )(*self.mesh.r)
        #                 self._sym[0, comp].mesh = self.mesh

        #         else:
        #             raise RuntimeError(
        #                 "Please supply a list of names for all components of this vector"
        #             )
        #     else:
        #         for comp in range(num_components):
        #             self._sym[0, comp] = UnderworldFunction(
        #                 self.symbol, self, vtype, comp, comp
        #             )(*self.mesh.r)

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
        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        return

    def __getitem__(self, indices):

        if not isinstance(indices, tuple):
            if isinstance(indices, int) and self.shape[0] == 1:
                i = 0
                j = indices
            else:
                raise IndexError(
                    "MeshVariable[i,j] access requires one or two indices "
                )
        else:
            i, j = indices

        return self._data_container[i, j]

    # We should be careful - this is an INTERPOLATION
    # that is stable when used for EXTRAPOLATION but
    # not accurate.

    def rbf_interpolate(self, new_coords, verbose=False, nnn=10):

        # An inverse-distance mapping is quite robust here ... as long
        # as long we take care of the case where some nodes coincide (likely if used mesh2mesh)

        import numpy as np

        with self.mesh.access(self):
            D = self.data.copy()

        if verbose and uw.mpi.rank == 0:
            print("Building K-D tree", flush=True)

        mesh_kdt = uw.kdtree.KDTree(self.coords)
        mesh_kdt.build_index()
        return mesh_kdt.rbf_interpolator_local(new_coords, D, nnn, verbose)

        # if verbose and uw.mpi.rank == 0:
        #     print("Building K-D tree ... done", flush=True)

        # # ToDo: This value should be somewhat larger than the ratio of the new_mesh points to the
        # # old mesh points if we have a near-uniform mesh spacing.

        # num_neighbours = nnn

        # closest_n, distance_n = mesh_kdt.find_closest_n_points(
        #     num_neighbours, new_coords
        # )

        # num_local_vertices = new_coords.shape[0]
        # data_size = D.shape[1]

        # Values = np.zeros((num_local_vertices, data_size))
        # Weights = np.zeros((num_local_vertices, 1))

        # if verbose and uw.mpi.rank == 0:
        #     print("Mapping values to vertices ... start", flush=True)

        # epsilon = 1.0e-24
        # for j in range(num_neighbours):
        #     j_distance = epsilon + distance_n[:, j]
        #     j_nearest = closest_n[:, j]
        #     Weights[:, 0] += 1.0 / j_distance[:]

        # epsilon = 1.0e-24
        # for d in range(data_size):
        #     for j in range(num_neighbours):
        #         j_distance = epsilon + distance_n[:, j]
        #         j_nearest = closest_n[:, j]
        #         Values[:, d] += D[j_nearest[:], d] / j_distance[:]

        # Values[...] /= Weights[:]

        # if verbose and uw.mpi.rank == 0:
        #     print("Mapping values to vertices ... done", flush=True)

        return Values

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
    def simple_save(
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
            PETSc.COMM_WORLD,
        )

        dmnew.setField(0, dmfe)
        dmnew.createDS()

        lvec = dmnew.getLocalVec()
        gvec = dmnew.getGlobalVec()

        lvec.array[...] = self.coords.reshape(-1)[...]
        dmnew.localToGlobal(lvec, gvec, addv=False)
        gvec.setName("X")

        # Check that this is also synchronised
        # self.mesh.dm.localToGlobal(self._lvec, self._gvec, addv=False)

        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=PETSc.COMM_WORLD)
        viewer(self._gvec)
        viewer(gvec)

        uw.mpi.barrier()
        viewer.destroy()

        return

    @timing.routine_timer_decorator
    def read_from_vertex_checkpoint(
        self,
        data_file,
        data_name,
        vertex_mesh_file=None,
        # vertex_field=False,
        vertex_field_degree=1,
        verbose=False,
    ):
        """
        Read a mesh variable from an arbitrary vertex-based checkpoint file
        and reconstruct/interpolate the data field accordingly. The data sizes / meshes can be
        different and will be matched using a kd-tree / inverse-distance weighting
        to the new mesh.

        """

        import h5py
        import numpy as np

        self._set_vec(available=False)

        ## Sub functions that are used to read / interpolate the mesh.
        def field_from_checkpoint(
            data_file=None,
            data_name=None,
        ):
            """Read the mesh data as a swarm-like value"""

            h5f = h5py.File(data_file)
            D = h5f["fields"][data_name][()]
            X = h5f["fields"]["X"][()]
            h5f.close()

            if len(D.shape) == 1:
                D = D.reshape(-1, 1)

            return X, D

        # def field_from_vertex_checkpoint(
        #     data_file=None,
        #     data_name=None,
        #     mesh_file=None,
        #     data_degree=1,
        # ):
        #     """Read the mesh data as a swarm-like value"""

        #     # data_field_name = data_name + f"_P{data_degree}"

        #     h5f = h5py.File(data_file)
        #     data_fields = h5f["vertex_fields"].keys()
        #     data_field_name = next(
        #         (field for field in data_fields if data_name in field), None
        #     )

        #     h5f = h5py.File(data_file)
        #     D = h5f["vertex_fields"][data_field_name][()]
        #     h5f.close()

        #     h5f = h5py.File(mesh_file)
        #     X = h5f["geometry"]["vertices"][()]
        #     h5f.close()

        #     if len(D.shape) == 1:
        #         D = D.reshape(-1, 1)

        #     return X, D

        def map_to_vertex_values(X, D, nnn=4, verbose=False):

            # Map from "swarm" of points to nodal points
            # This is a permutation if we building on the checkpointed
            # mesh file

            mesh_kdt = uw.kdtree.KDTree(X)
            mesh_kdt.build_index()

            return mesh_kdt.rbf_interpolator_local(self.coords, D, nnn, verbose)

        def values_to_mesh_var(mesh_variable, Values):

            mesh = mesh_variable.mesh

            # This should be trivial but there may be problems if
            # the kdtree does not have enough neighbours to allocate
            # values for every point. We handle that here.

            with mesh.access(mesh_variable):
                mesh_variable.data[...] = Values[...]

            return

        # if not vertex_field:
        X, D = field_from_checkpoint(
            data_file,
            data_name,
             )
        # else:
        #     X, D = field_from_vertex_checkpoint(
        #         data_file,
        #         data_name,
        #         vertex_mesh_file,
        #         vertex_field_degree,
        #     )

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
            if j is None:
                return i
            elif i == 0:
                return j
            else:
                raise IndexError(
                    f"Vectors have shape {self.mesh.dim} or {(1, self.mesh.dim)} "
                )
        if self.vtype == uw.VarType.TENSOR or self.vtype == uw.VarType.MATRIX:
            if self.mesh.dim == 2:
                return ((0, 1), (2, 3))[i][j]
            else:
                return ((0, 1, 2), (3, 4, 5), (6, 7, 8))[i][j]

        if self.vtype == uw.VarType.SYM_TENSOR:
            if self.mesh.dim == 2:
                return ((0, 2), (2, 1))[i][j]
            else:
                return ((0, 3, 4), (3, 1, 5), (4, 5, 2))[i][j]

    def _set_vec(self, available):

        if self._lvec == None:
            indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
            # subdm = uw.cython.petsc_discretisation.petsc_fe_create_sub_dm(self.mesh.dm, self.field_id)

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
        mesh_filename = filename + ".mesh.0.h5"
    else:
        mesh_filename = filename + f".mesh.{index}.h5"

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
        var_filename = filename + f".{var.clean_name}.{index}.h5"
        header += f"""
<!ENTITY {var.clean_name}_Data "{os.path.basename(var_filename)}">"""

    for var in swarmVars:
        var_filename = filename + f".proxy.{var.clean_name}.{index}.h5"
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

    attributes = ""
    for var in meshVars:
        var_filename = filename + f".{var.clean_name}.{index}.h5"
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
              &{var.clean_name+"_Data"};:/vertex_fields/{var.clean_name+"_P"+str(var.degree)}
            </DataItem>
          </DataItem>
        </Attribute>
    """
        attributes += var_attribute

    for var in swarmVars:
        var_filename = filename + f".proxy.{var.clean_name}.{index}.h5"
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

    xdmf_filename = filename + f".{index}.xdmf"
    with open(xdmf_filename, "w") as fp:
        fp.write(header)
        fp.write(xdmf_start)
        fp.write(attributes)
        fp.write(xdmf_end)

    return
