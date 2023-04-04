from typing import Optional, Tuple
import contextlib

import numpy as np
import petsc4py.PETSc as PETSc
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities import _api_tools
import underworld3.timing as timing

import h5py
import os
import warnings

comm = MPI.COMM_WORLD

from enum import Enum


class SwarmType(Enum):
    DMSWARM_PIC = 1


class SwarmPICLayout(Enum):
    """
    Particle population fill type:

    SwarmPICLayout.REGULAR     defines points on a regular ijk mesh. Supported by simplex cell types only.
    SwarmPICLayout.GAUSS       defines points using an npoint Gauss-Legendre tensor product quadrature rule.
    SwarmPICLayout.SUBDIVISION defines points on the centroid of a sub-divided reference cell.
    """

    REGULAR = 0
    GAUSS = 1
    SUBDIVISION = 2


class SwarmVariable(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        num_components,
        vtype=None,
        dtype=float,
        proxy_degree=1,
        proxy_continuous=True,
        _register=True,
        _proxy=True,
        _nn_proxy=False,
        varsymbol=None,
        rebuild_on_cycle=True,
    ):

        if name in swarm.vars.keys():
            raise ValueError(
                "Variable with name {} already exists on swarm.".format(name)
            )

        import re

        if varsymbol is None:
            varsymbol = name

        self.name = name
        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)
        self.symbol = varsymbol

        self.swarm = swarm
        self.num_components = num_components

        if (dtype == float) or (dtype == "float") or (dtype == np.float64):
            self.dtype = float
            petsc_type = PETSc.ScalarType
        elif (
            (dtype == int)
            or (dtype == "int")
            or (dtype == np.int32)
            or (dtype == np.int64)
        ):
            self.dtype = int
            petsc_type = PETSc.IntType
        else:
            raise TypeError(
                f"Provided dtype={dtype} is not supported. Supported types are 'int' and 'float'."
            )

        if _register:
            self.swarm.dm.registerField(
                self.clean_name, self.num_components, dtype=petsc_type
            )

        self._data = None
        # add to swarms dict

        self.swarm._vars[self.clean_name] = self
        self._is_accessed = False

        # proxy variable
        self._proxy = _proxy
        self._vtype = vtype
        self._proxy_degree = proxy_degree
        self._proxy_continuous = proxy_continuous
        self._nn_proxy = _nn_proxy
        self._create_proxy_variable()

        # recycle swarm
        self._rebuild_on_cycle = rebuild_on_cycle

        self._register = _register

        super().__init__()

        return

    def _create_proxy_variable(self):

        # release if defined
        self._meshVar = None

        if self._proxy:
            self._meshVar = uw.discretisation.MeshVariable(
                "proxy_" + self.clean_name,
                self.swarm._mesh,
                self.num_components,
                self._vtype,
                degree=self._proxy_degree,
                continuous=self._proxy_continuous,
                varsymbol=r"\cal{P}\left(" + self.symbol + r"\right)",
            )

    def _update(self):
        """
        This method updates the proxy mesh variable for the current
        swarm & particle variable state.
        """

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return

        else:
            self._rbf_to_meshVar(self._meshVar)

        return

    # Maybe rbf_interpolate for this one and meshVar is a special case
    def _rbf_to_meshVar(self, meshVar, nnn=None, verbose=False):
        """
        Here is how it works: for each particle, create a distance-weighted average on the node data

        Todo: caching the k-d trees etc for the proxy-mesh-variable nodal points
        Todo: some form of global fall-back for when there are no particles on a processor
        """

        # Mapping to the coordinates of the variable from the
        # particle coords

        if nnn is None:
            nnn = self.swarm.mesh.dim + 1

        if meshVar.mesh != self.swarm.mesh:
            raise RuntimeError("Cannot map a swarm to a different mesh")

        new_coords = meshVar.coords

        Values = self.rbf_interpolate(new_coords, verbose=verbose, nnn=nnn)

        with meshVar.mesh.access(meshVar):
            meshVar.data[...] = Values[...]

        return

    def rbf_interpolate(self, new_coords, verbose=False, nnn=None):

        # An inverse-distance mapping is quite robust here ... as long
        # as long we take care of the case where some nodes coincide (likely if used mesh2mesh)
        # We try to eliminate contributions from recently remeshed particles

        import numpy as np

        if nnn is None:
            nnn = self.swarm.mesh.dim + 1

        with self.swarm.access():
            if self.swarm.recycle_rate > 1:
                not_remeshed = self.swarm._remeshed.data[:, 0] != 0
                D = self.data[not_remeshed].copy()

                kdt = uw.kdtree.KDTree(
                    self.swarm.particle_coordinates.data[not_remeshed, :]
                )
            else:
                D = self.data.copy()
                kdt = uw.kdtree.KDTree(self.swarm.particle_coordinates.data[:, :])

            kdt.build_index()

        return kdt.rbf_interpolator_local(new_coords, D, nnn, verbose)

    # ToDo: I don't think this is used / up to date
    @timing.routine_timer_decorator
    def project_from(self, meshvar):
        # use method found in
        # /tmp/petsc-build/petsc/src/dm/impls/swarm/tests/ex2.c
        # to project from fields to particles

        self.swarm.mesh.dm.clearDS()
        self.swarm.mesh.dm.createDS()

        meshdm = meshvar.mesh.dm
        fields = meshvar.field_id
        _, meshvardm = meshdm.createSubDM(fields)

        ksp = PETSc.KSP().create()
        ksp.setOptionsPrefix("swarm_project_from_")
        options = PETSc.Options()
        options.setValue("swarm_project_from_ksp_type", "lsqr")
        options.setValue("swarm_project_from_ksp_rtol", 1e-17)
        options.setValue("swarm_project_from_pc_type", "none")
        ksp.setFromOptions()

        rhs = meshvardm.getGlobalVec()

        M_p = self.swarm.dm.createMassMatrix(meshvardm)

        # make particle weight vector
        f = self.swarm.createGlobalVectorFromField(self.clean_name)

        # create matrix RHS vector, in this case the FEM field fhat with the coefficients vector #alpha
        M = meshvardm.createMassMatrix(meshvardm)
        with meshvar.mesh.access():
            M.multTranspose(meshvar.vec_global, rhs)

        ksp.setOperators(M_p, M_p)
        ksp.solveTranspose(rhs, f)

        self.swarm.dm.destroyGlobalVectorFromField(self.clean_name)
        meshvardm.restoreGlobalVec(rhs)
        meshvardm.destroy()
        ksp.destroy()
        M.destroy()
        M_p.destroy()

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError(
                "Data must be accessed via the swarm `access()` context manager."
            )
        return self._data

    # @property
    # def fn(self):
    #     return self._meshVar.fn

    @property
    def sym(self):
        return self._meshVar.sym

    @timing.routine_timer_decorator
    def save(
        self,
        filename: int,
        compression: Optional[bool] = False,
        compressionType: Optional[str] = "gzip",
        force_sequential=False,
    ):
        """

        Save the swarm variable to a h5 file.

        Parameters
        ----------
        filename :
            The filename of the swarm variable to save to disk.
        compression :
            Add compression to the h5 files (saves space but increases write times with increasing no. of processors)
        compressionType :
            Type of compression to use, 'gzip' and 'lzf' supported. 'gzip' is default. Compression also needs to be set to 'True'.

        """
        if h5py.h5.get_config().mpi == False and comm.size > 1 and comm.rank == 0:
            warnings.warn(
                "Collective IO not possible as h5py not available in parallel mode. Switching to sequential. This will be slow for models running on multiple processors",
                stacklevel=2,
            )
        if compression == True and comm.rank == 0:
            warnings.warn("Compression may slow down write times", stacklevel=2)
        if filename.endswith(".h5") == False:
            raise RuntimeError("The filename must end with .h5")

        if h5py.h5.get_config().mpi == True and not force_sequential:
            with h5py.File(
                f"{filename[:-3]}.h5", "w", driver="mpio", comm=MPI.COMM_WORLD
            ) as h5f:
                with self.swarm.access(self):
                    if compression == True:
                        h5f.create_dataset(
                            "data", data=self.data[:], compression=compressionType
                        )
                    else:
                        h5f.create_dataset("data", data=self.data[:])
        else:
            with self.swarm.access(self):
                if comm.rank == 0:
                    # print(f'start {self.name} on {comm.rank}')
                    with h5py.File(f"{filename[:-3]}.h5", "w") as h5f:
                        if compression == True:
                            h5f.create_dataset(
                                "data",
                                data=self.data[:],
                                chunks=True,
                                maxshape=(None, self.data.shape[1]),
                                compression=compressionType,
                            )
                        else:
                            h5f.create_dataset(
                                "data",
                                data=self.data[:],
                                chunks=True,
                                maxshape=(None, self.data.shape[1]),
                            )
                    # print(f'finish {self.name} on {comm.rank}')
                comm.barrier()
                for proc in range(1, comm.size):
                    if comm.rank == proc:
                        # print(f'start {self.name} on {comm.rank}')
                        with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                            h5f["data"].resize(
                                (h5f["data"].shape[0] + self.data.shape[0]), axis=0
                            )
                            h5f["data"][-self.data.shape[0] :] = self.data[:]
                        # print(f'finish {self.name} on {comm.rank}')
                    comm.barrier()
                comm.barrier()

        return

    @timing.routine_timer_decorator
    def simple_save(self, filename: str):

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            if uw.mpi.rank == 0:
                print("No proxy mesh variable that can be saved", flush=True)
            return

        self._meshVar.simple_save(filename)

        return

    @timing.routine_timer_decorator
    def load(
        self,
        filename: str,
        swarmFilename: str,
    ):
        ### open up file with coords on all procs and open up data on all procs. May be problematic for large problems.
        with h5py.File(f"{filename}", "r") as h5f_data, h5py.File(
            f"{swarmFilename}", "r"
        ) as h5f_swarm:
            with self.swarm.access(self):
                var_dtype = self.data.dtype
                file_dtype = h5f_data["data"][:].dtype
                file_length = h5f_data["data"][:].shape[0]

                if var_dtype != file_dtype:
                    if comm.rank == 0:
                        warnings.warn(
                            f"{os.path.basename(filename)} dtype ({file_dtype}) does not match {self.name} swarm variable dtype ({var_dtype}) which may result in a loss of data.",
                            stacklevel=2,
                        )

                # First work out which are local points and ignore the rest
                # This might help speed up the load by dropping lots of particles

                all_coords = h5f_swarm["coordinates"][()]
                all_data = h5f_data["data"][()]

                cell = self.swarm.mesh.get_closest_local_cells(all_coords)
                local = np.where(cell >= 0)[0]
                # not_not_local = np.where(cell == -1)[0]

                local_coords = all_coords[local]
                local_data = all_data[local]

                kdt = uw.kdtree.KDTree(local_coords)

                self.data[:] = kdt.rbf_interpolator_local(
                    self.swarm.data, local_data, nnn=1
                )

                #### this produces a shape mismatch, would be quicker not to do it in a loop
                # ind = np.isin(coordinates, self.swarm.data).all(axis=1)
                # # self.data[:] = data[ind]

                # print(f"Looping over coords for swarm load")
                # i = 0

                ### loops through the coords of the swarm to load the data
                # for coord in self.swarm.data:
                #     ind_data = np.isin(h5f_swarm["coordinates"][:], coord).all(axis=1)
                #     ind_swarm = np.isin(self.swarm.data, coord).all(axis=1)
                #     self.data[ind_swarm] = h5f_data["data"][:][ind_data]
                #     i += 1
                #     if i % 1000 == 0:
                #         print(
                #             f"Looping over coords for swarm load ... {i}/{self.swarm.data.shape[0]}"
                #         )

                # print(f"Looping over coords for swarm load ... done")

                ### loops through the coords in the file
                # for i in range(0, file_length):
                #     coord = h5f_swarm["coordinates"][i]
                #     data = h5f_data["data"][i]
                #     ind_swarm = np.isin(self.swarm.data, coord).all(axis=1)

                #     self.data[ind_swarm] = data

        return


class IndexSwarmVariable(SwarmVariable):
    """
    The IndexSwarmVariable is a class for managing material point
    behaviour. The material index variable is rendered into a
    collection of masks each representing the extent of one material
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        indices=1,
        proxy_degree=1,
        proxy_continuous=True,
    ):

        self.indices = indices

        # These are the things we require of the generic swarm variable type
        super().__init__(
            name,
            swarm,
            num_components=1,
            vtype=None,
            dtype=int,
            _proxy=False,
        )

        # The indices variable defines how many level set maps we create as components in the proxy variable

        import sympy

        self._MaskArray = sympy.Matrix.zeros(1, self.indices)
        self._meshLevelSetVars = [None] * self.indices

        for i in range(indices):
            self._meshLevelSetVars[i] = uw.discretisation.MeshVariable(
                name + R"^{[" + str(i) + R"]}",
                self.swarm.mesh,
                num_components=1,
                degree=proxy_degree,
                continuous=proxy_continuous,
            )
            self._MaskArray[0, i] = self._meshLevelSetVars[i].sym[0, 0]

        return

    # This is the sympy vector interface - it's meaningless if these are not spatial arrays
    @property
    def sym(self):
        return self._MaskArray

    def _update(self):
        """
        This method updates the proxy mesh (vector) variable for the index variable on the current swarm locations

        Here is how it works:

            1) for each particle, create a distance-weighted average on the node data
            2) for each index in the set, we create a mask mesh variable by mapping 1.0 wherever the
               index matches and 0.0 where it does not.

        NOTE: If no material is identified with a given nodal value, the default is to material zero

        ## ToDo: This should be revisited to match the updated master copy of _update

        """

        kd = uw.kdtree.KDTree(self._meshLevelSetVars[0].coords)
        kd.build_index()

        for ii in range(self.indices):
            meshVar = self._meshLevelSetVars[ii]

            # 1 - Average particles to nodes with distance weighted average
            with self.swarm.mesh.access(meshVar), self.swarm.access():
                n, d, b = kd.find_closest_point(self.swarm.data)

                node_values = np.zeros((meshVar.data.shape[0],))
                w = np.zeros((meshVar.data.shape[0],))

                for i in range(self.data.shape[0]):
                    if b[i]:
                        node_values[n[i]] += np.isclose(self.data[i], ii) / (
                            1.0e-16 + d[i]
                        )
                        w[n[i]] += 1.0 / (1.0e-16 + d[i])

                node_values[np.where(w > 0.0)[0]] /= w[np.where(w > 0.0)[0]]

            # 2 - set NN vals on mesh var where w == 0.0

            with self.swarm.mesh.access(meshVar), self.swarm.access():
                meshVar.data[...] = node_values[...].reshape(-1, 1)

                # Need to document this assumption, if there is no material found,
                # assume the default material (0). An alternative would be to impose
                # a near-neighbour hunt for a valid material and set that one.

                if ii == 0:
                    meshVar.data[np.where(w == 0.0)] = 1.0
                else:
                    meshVar.data[np.where(w == 0.0)] = 0.0

        return


# @typechecked
class Swarm(_api_tools.Stateful):

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, mesh, recycle_rate=0):

        Swarm.instances += 1

        self._mesh = mesh
        self.dim = mesh.dim
        self.cdim = mesh.cdim
        self.dm = PETSc.DMSwarm().create()
        self.dm.setDimension(self.dim)
        self.dm.setType(SwarmType.DMSWARM_PIC.value)
        self.dm.setCellDM(mesh.dm)
        self._data = None

        # Is the swarm a streak-swarm ?
        self.recycle_rate = recycle_rate
        self.cycle = 0

        # dictionary for variables

        # import weakref (not helpful as garbage collection does not remove the fields from the DM)
        # self._vars = weakref.WeakValueDictionary()
        self._vars = {}

        # add variable to handle particle coords - predefined by DMSwarm, expose to UW
        self._coord_var = SwarmVariable(
            "DMSwarmPIC_coor",
            self,
            self.cdim,
            dtype=float,
            _register=False,
            _proxy=False,
            rebuild_on_cycle=False,
        )

        # add variable to handle particle cell id - predefined by DMSwarm, expose to UW
        self._cellid_var = SwarmVariable(
            "DMSwarm_cellid",
            self,
            1,
            dtype=int,
            _register=False,
            _proxy=False,
            rebuild_on_cycle=False,
        )

        # add variable to hold swarm coordinates during position updates
        self._X0 = uw.swarm.SwarmVariable(
            "DMSwarm_X0",
            self,
            self.cdim,
            dtype=float,
            _register=True,
            _proxy=False,
            rebuild_on_cycle=False,
        )

        # This is for swarm streak management:
        # add variable to hold swarm origins

        if self.recycle_rate > 1:
            # self._Xorig = uw.swarm.SwarmVariable(
            #     "DMSwarm_Xorig",
            #     self,
            #     self.cdim,
            #     dtype=float,
            #     _register=True,
            #     _proxy=False,
            #     rebuild_on_cycle=False,
            # )

            self._remeshed = uw.swarm.SwarmVariable(
                "DMSwarm_remeshed",
                self,
                1,
                dtype=int,
                _register=True,
                _proxy=False,
                rebuild_on_cycle=False,
            )

        self._X0_uninitialised = True
        # self._Xorig_uninitialised = True
        self._index = None
        self._nnmapdict = {}

        super().__init__()

    @property
    def mesh(self):
        return self._mesh

    # The setter needs updating to account for re-distribution of the DM
    # in the general case - see adaptivity.mesh2mesh_swarm()

    # @mesh.setter
    # def mesh(self, new_mesh):
    #     self._mesh = new_mesh
    #     self.dm.setCellDM(new_mesh.dm)

    #     # k-d tree indexing is no longer valid
    #     self._index = None
    #     self._nnmapdict = {}

    #     cellid = self.dm.getField("DMSwarm_cellid")
    #     cellid[:] = 0  # new_mesh.get_closest_cells(coords).reshape(-1)
    #     self.dm.restoreField("DMSwarm_cellid")
    #     self.dm.migrate(remove_sent_points=True)

    #     # Also need to re-proxy the swarm variables on the new mesh !!
    #     for v in self.vars:
    #         var = self.vars[v]
    #         var._create_proxy_variable()
    #         var._update()

    #     return

    @property
    def data(self):
        return self.particle_coordinates.data

    @property
    def particle_coordinates(self):
        return self._coord_var

    @property
    def particle_cellid(self):
        return self._cellid_var

    # @timing.routine_timer_decorator
    # def populate(
    #     self,
    #     fill_param: Optional[int] = 3,
    #     layout: Optional[SwarmPICLayout] = None,
    # ):
    #     (
    #         """
    #     Populate the swarm with particles throughout the domain.

    #     """
    #         + SwarmPICLayout.__doc__
    #         + """

    #     When using SwarmPICLayout.REGULAR,     `fill_param` defines the number of points in each spatial direction.
    #     When using SwarmPICLayout.GAUSS,       `fill_param` defines the number of quadrature points in each spatial direction.
    #     When using SwarmPICLayout.SUBDIVISION, `fill_param` defines the number times the reference cell is sub-divided.

    #     Parameters
    #     ----------
    #     fill_param:
    #         Parameter determining the particle count per cell for the given layout.
    #     layout:
    #         Type of layout to use. Defaults to `SwarmPICLayout.REGULAR` for mesh objects with simplex
    #         type cells, and `SwarmPICLayout.GAUSS` otherwise.

    #     """
    #     )

    #     self.fill_param = fill_param

    #     """
    #     Currently (2021.11.15) supported by PETSc release 3.16.x

    #     When using a DMPLEX the following case are supported:
    #           (i) DMSWARMPIC_LAYOUT_REGULAR: 2D (triangle),
    #          (ii) DMSWARMPIC_LAYOUT_GAUSS: 2D and 3D provided the cell is a tri/tet or a quad/hex,
    #         (iii) DMSWARMPIC_LAYOUT_SUBDIVISION: 2D and 3D for quad/hex and 2D tri.

    #     So this means, simplex mesh in 3D only supports GAUSS - This is based
    #     on the tensor product locations so it is not even in the cells.

    #     """

    @timing.routine_timer_decorator
    def populate(
        self,
        fill_param: Optional[int] = 1,
    ):
        (
            """
        Populate the swarm with particles throughout the domain.

        Parameters
        ----------
        fill_param:
            Parameter determining the particle count per cell (per dimension)
            for the given layout, using the mesh degree.

        cell_search:
            Use k-d tree to locate nearest cells (fails if this swarm is used to build a k-d tree)

        """
        )

        self.fill_param = fill_param

        newp_coords = self.mesh._get_coords_for_basis(fill_param, continuous=False)
        newp_cells = self.mesh.get_closest_local_cells(newp_coords)

        self.dm.finalizeFieldRegister()
        self.dm.addNPoints(newp_coords.shape[0] + 1)

        cellid = self.dm.getField("DMSwarm_cellid")
        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))

        coords[...] = newp_coords[...]
        cellid[:] = newp_cells[:]

        self.dm.restoreField("DMSwarmPIC_coor")
        self.dm.restoreField("DMSwarm_cellid")

        ## Now make a series of copies to allow the swarm cycling to
        ## work correctly (if required)

        if self.recycle_rate > 1:
            with self.access():
                # Actually, this is a mesh-local quantity, so let's just
                # store it on the mesh in an ad_hoc fashion for now

                self.mesh.particle_X_orig = self.particle_coordinates.data.copy()
                self.mesh.particle_CellID_orig = self._cellid_var.data.copy()

            with self.access():
                swarm_orig_size = self.particle_coordinates.data.shape[0]
                all_local_coords = np.vstack(
                    (self.particle_coordinates.data,) * (self.recycle_rate)
                )
                all_local_cells = np.vstack(
                    (self._cellid_var.data,) * (self.recycle_rate)
                )

                swarm_new_size = all_local_coords.data.shape[0]

            self.dm.addNPoints(swarm_new_size - swarm_orig_size)

            cellid = self.dm.getField("DMSwarm_cellid")
            coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))

            coords[...] = (
                all_local_coords[...]
                + (0.5 / (1 + fill_param))
                * (np.random.random(size=all_local_coords.shape) - 0.5)
                * self.mesh._radii[all_local_cells]  # typical cell size
            )
            cellid[:] = all_local_cells[:, 0]

            self.dm.restoreField("DMSwarmPIC_coor")
            self.dm.restoreField("DMSwarm_cellid")

            ## Now set the cycle values

            with self.access(self._remeshed):
                for i in range(0, self.recycle_rate):
                    offset = swarm_orig_size * i
                    self._remeshed.data[offset::, 0] = i

            # with self.access(self._Xorig):
            #     self._Xorig.data[...] = self.data[...]
            #     self._Xorig_uninitialised = False

        return

    ## This is actually an initial population routine.
    ## We can't use this to add particles / manage variables (LM)
    @timing.routine_timer_decorator
    def add_particles_with_coordinates(self, coordinatesArray):
        """
        This method adds particles to the swarm using particle coordinates provided
        using a numpy array.
        Note that particles with coordinates NOT local to the current processor will
        be reject/ignored. Either include an array with all coordinates to all processors
        or an array with the local coordinates.
        Parameters
        ----------
        coordinatesArray : numpy.ndarray
            The numpy array containing the coordinate of the new particles. Array is
            expected to take shape n*dim, where n is the number of new particles, and
            dim is the dimensionality of the swarm's supporting mesh.
        """

        if not isinstance(coordinatesArray, np.ndarray):
            raise TypeError("'coordinateArray' must be provided as a numpy array")
        if not len(coordinatesArray.shape) == 2:
            raise ValueError("The 'coordinateArray' is expected to be two dimensional.")
        if not coordinatesArray.shape[1] == self.mesh.dim:
            #### petsc appears to ignore columns that are greater than the mesh dim, but still worth including
            raise ValueError(
                """The 'coordinateArray' must have shape n*dim, where 'n' is the
                              number of particles to add, and 'dim' is the dimensionality of
                              the supporting mesh ({}).""".format(
                    self.mesh.dim
                )
            )

        cells = self.mesh.get_closest_local_cells(coordinatesArray)

        valid_coordinates = coordinatesArray[cells != -1]
        valid_cells = cells[cells != -1]

        npoints = len(valid_coordinates)
        swarm_size = self.dm.getLocalSize()

        # -1 means no particles have been added yet
        if swarm_size == -1:
            swarm_size = 0
            npoints = npoints + 1

        self.dm.finalizeFieldRegister()
        self.dm.addNPoints(npoints=npoints)

        cellid = self.dm.getField("DMSwarm_cellid")
        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))

        coords[swarm_size::, :] = valid_coordinates[:, :]
        cellid[swarm_size::] = valid_cells[:]

        self.dm.restoreField("DMSwarmPIC_coor")
        self.dm.restoreField("DMSwarm_cellid")

        # Here we update the swarm cycle values as required

        if self.recycle_rate > 1:
            with self.access(self._remeshed):
                # self._Xorig.data[...] = coordinatesArray
                self._remeshed.data[...] = 0

        self.dm.migrate(remove_sent_points=True)

        return

    @timing.routine_timer_decorator
    def save(
        self,
        filename: int,
        compression: Optional[bool] = False,
        compressionType: Optional[str] = "gzip",
        force_sequential=False,
    ):
        """

        Save the swarm coordinates to a h5 file.

        Parameters
        ----------
        filename :
            The filename of the swarm checkpoint file to save to disk.
        compression :
            Add compression to the h5 files (saves space but increases write times with increasing no. of processors)
        compressionType :
            Type of compression to use, 'gzip' and 'lzf' supported. 'gzip' is default. Compression also needs to be set to 'True'.



        """
        if h5py.h5.get_config().mpi == False and comm.size > 1 and comm.rank == 0:
            warnings.warn(
                "Collective IO not possible as h5py not available in parallel mode. Switching to sequential. This will be slow for models running on multiple processors",
                stacklevel=2,
            )
        if filename.endswith(".h5") == False:
            raise RuntimeError("The filename must end with .h5")
        if compression == True and comm.rank == 0:
            warnings.warn("Compression may slow down write times", stacklevel=2)

        if h5py.h5.get_config().mpi == True and not force_sequential:

            # It seems to be a bad idea to mix mpi barriers with the access
            # context manager so the copy-free version of this seems to hang
            # when there are many active cores. This is probably why the parallel
            # h5py write hangs

            with self.access():
                data_copy = self.data[:].copy()

            with h5py.File(
                f"{filename[:-3]}.h5", "w", driver="mpio", comm=MPI.COMM_WORLD
            ) as h5f:
                if compression == True:
                    h5f.create_dataset(
                        "coordinates",
                        data=data_copy[:],
                        compression=compressionType,
                    )
                else:
                    h5f.create_dataset("coordinates", data=data_copy[:])
        else:

            # It seems to be a bad idea to mix mpi barriers with the access
            # context manager so the copy-free version of this seems to hang
            # when there are many active cores

            with self.access():
                data_copy = self.data[:].copy()

            if comm.rank == 0:
                with h5py.File(f"{filename[:-3]}.h5", "w") as h5f:
                    if compression == True:
                        h5f.create_dataset(
                            "coordinates",
                            data=data_copy,
                            chunks=True,
                            maxshape=(None, data_copy.shape[1]),
                            compression=compressionType,
                        )
                    else:
                        h5f.create_dataset(
                            "coordinates",
                            data=data_copy,
                            chunks=True,
                            maxshape=(None, data_copy.shape[1]),
                        )

            comm.barrier()
            for i in range(1, comm.size):
                if comm.rank == i:
                    with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                        h5f["coordinates"].resize(
                            (h5f["coordinates"].shape[0] + data_copy.shape[0]),
                            axis=0,
                        )
                        # passive swarm, zero local particles is not unusual
                        if data_copy.shape[0] > 0:
                            h5f["coordinates"][-data_copy.shape[0] :] = data_copy[:]
                comm.barrier()
            comm.barrier()

        return

    @timing.routine_timer_decorator
    def load(
        self,
        filename: str,
    ):
        ### open up file with coords on all procs
        with h5py.File(f"{filename}", "r") as h5f:
            coordinates = h5f["coordinates"][:]

        #### utilises the UW function for adding a swarm by an array
        self.add_particles_with_coordinates(coordinates)

        return

    @timing.routine_timer_decorator
    def add_variable(
        self, name, num_components=1, dtype=float, proxy_degree=2, _nn_proxy=False
    ):
        return SwarmVariable(
            name,
            self,
            num_components,
            dtype=dtype,
            proxy_degree=proxy_degree,
            _nn_proxy=_nn_proxy,
        )

    @timing.routine_timer_decorator
    def petsc_save_checkpoint(
        self,
        swarmName: str,
        index: int,
        outputPath: Optional[str] = "",
    ):

        """

        Use PETSc to save the swarm and attached data to a .pbin and xdmf file.

        Parameters
        ----------
        swarmName :
            Name of the swarm to save.
        index :
            An index which might correspond to the timestep or output number (for example).
        outputPath :
            Path to save the data. If left empty it will save the data in the current working directory.
        """

        x_swarm_fname = f"{outputPath}{swarmName}_{index:04d}.xmf"
        self.dm.viewXDMF(x_swarm_fname)

    @timing.routine_timer_decorator
    def save_checkpoint(
        self,
        swarmName: str,
        swarmVars: list,
        index: int,
        outputPath: Optional[str] = "",
        time: Optional[int] = None,
        compression: Optional[bool] = False,
        compressionType: Optional[str] = "gzip",
        force_sequential: Optional[bool] = False,
    ):

        """

        Save data to h5 and a corresponding xdmf for visualisation using h5py.

        Parameters
        ----------
        swarmName :
            Name of the swarm to save.
        swarmVars :
            List of swarm objects to save.
        index :
            An index which might correspond to the timestep or output number (for example).
        outputPath :
            Path to save the data. If left empty it will save the data in the current working directory.
        time :
            Attach the time to the generated xdmf.
        compression :
            Whether to compress the h5 files [bool].
        compressionType :
            The type of compression to use. 'gzip' and 'lzf' are the supported types, with 'gzip' as the default.
        """

        if swarmVars != None and not isinstance(swarmVars, list):
            raise RuntimeError("`swarmVars` does not appear to be a list.")

        else:
            ### save the swarm particle location
            self.save(
                filename=f"{outputPath}{swarmName}-{index:04d}.h5",
                compression=compression,
                compressionType=compressionType,
                force_sequential=force_sequential,
            )

        #### Generate a h5 file for each field
        if swarmVars != None:
            for field in swarmVars:
                field.save(
                    filename=f"{outputPath}{field.name}-{index:04d}.h5",
                    compression=compression,
                    compressionType=compressionType,
                    force_sequential=force_sequential,
                )

        if uw.mpi.rank == 0:
            ### only need to combine the h5 files to a single xdmf on one proc
            with open(f"{outputPath}{swarmName}-{index:04d}.xmf", "w") as xdmf:
                # Write the XDMF header
                xdmf.write('<?xml version="1.0" ?>\n')
                xdmf.write(
                    '<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n'
                )
                xdmf.write("<Domain>\n")
                xdmf.write(
                    f'<Grid Name="{swarmName}-{index:04d}" GridType="Uniform">\n'
                )

                if time != None:
                    xdmf.write(f'	<Time Value="{time}" />\n')

                # Write the grid element for the HDF5 dataset
                with h5py.File(f"{outputPath}{swarmName}-{index:04}.h5", "r") as h5f:
                    xdmf.write(
                        f'	<Topology Type="POLYVERTEX" NodesPerElement="{h5f["coordinates"].shape[0]}"> </Topology>\n'
                    )
                    if h5f["coordinates"].shape[1] == 2:
                        xdmf.write('		<Geometry Type="XY">\n')
                    elif h5f["coordinates"].shape[1] == 3:
                        xdmf.write('		<Geometry Type="XYZ">\n')
                    xdmf.write(
                        f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["coordinates"].shape[0]} {h5f["coordinates"].shape[1]}">{os.path.basename(h5f.filename)}:/coordinates</DataItem>\n'
                    )
                    xdmf.write("		</Geometry>\n")

                # Write the attribute element for the field
                if swarmVars != None:
                    for field in swarmVars:
                        with h5py.File(
                            f"{outputPath}{field.name}-{index:04d}.h5", "r"
                        ) as h5f:
                            if h5f["data"].dtype == np.int32:
                                xdmf.write(
                                    f'	<Attribute Type="Scalar" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Int" Precision="4" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )
                            elif h5f["data"].shape[1] == 1:
                                xdmf.write(
                                    f'	<Attribute Type="Scalar" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )
                            elif h5f["data"].shape[1] == 2 or h5f["data"].shape[1] == 3:
                                xdmf.write(
                                    f'	<Attribute Type="Vector" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )
                            else:
                                xdmf.write(
                                    f'	<Attribute Type="Tensor" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )

                            xdmf.write("	</Attribute>\n")
                else:
                    pass

                # Write the XDMF footer
                xdmf.write("</Grid>\n")
                xdmf.write("</Domain>\n")
                xdmf.write("</Xdmf>\n")

    @property
    def vars(self):
        return self._vars

    def access(self, *writeable_vars: SwarmVariable):
        """
        This context manager makes the underlying swarm variables data available to
        the user. The data should be accessed via the variables `data` handle.

        As default, all data is read-only. To enable writeable data, the user should
        specify which variable they wish to modify.

        At the conclusion of the users context managed block, numerous further operations
        will be automatically executed. This includes swarm parallel migration routines
        where the swarm's `particle_coordinates` variable has been modified. The swarm
        variable proxy mesh variables will also be updated for modifed swarm variables.

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

        uw.timing._incrementDepth()
        stime = time.time()

        deaccess_list = []
        for var in self._vars.values():
            # if already accessed within higher level context manager, continue.
            if var._is_accessed == True:
                continue
            # set flag so variable status can be known elsewhere
            var._is_accessed = True
            # add to de-access list to rewind this later
            deaccess_list.append(var)
            # grab numpy object, setting read only if necessary
            var._data = self.dm.getField(var.clean_name).reshape(
                (-1, var.num_components)
            )
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()
        # if particles moving, update swarm state
        if self.particle_coordinates in writeable_vars:
            self._increment()

        # Create a class which specifies the required context
        # manager hooks (`__enter__`, `__exit__`).
        class exit_manager:
            def __init__(self, swarm):
                self.em_swarm = swarm

            def __enter__(self):
                pass

            def __exit__(self, *args):
                for var in self.em_swarm.vars.values():
                    # only de-access variables we have set access for.
                    if var not in deaccess_list:
                        continue
                    # set this back, although possibly not required.
                    if var not in writeable_vars:
                        var._data.flags.writeable = var._old_data_flag
                    var._data = None
                    self.em_swarm.dm.restoreField(var.clean_name)
                    var._is_accessed = False
                # do particle migration if coords changes
                if self.em_swarm.particle_coordinates in writeable_vars:
                    # let's use the mesh index to update the particles owning cells.
                    # note that the `petsc4py` interface is more convenient here as the
                    # `SwarmVariable.data` interface is controlled by the context manager
                    # that we are currently within, and it is therefore too easy to
                    # get things wrong that way.
                    cellid = self.em_swarm.dm.getField("DMSwarm_cellid")
                    coords = self.em_swarm.dm.getField("DMSwarmPIC_coor").reshape(
                        (-1, self.em_swarm.dim)
                    )
                    cellid[:] = self.em_swarm.mesh.get_closest_cells(coords).reshape(-1)
                    self.em_swarm.dm.restoreField("DMSwarmPIC_coor")
                    self.em_swarm.dm.restoreField("DMSwarm_cellid")
                    # now migrate.
                    self.em_swarm.dm.migrate(remove_sent_points=True)
                    # void these things too
                    self.em_swarm._index = None
                    self.em_swarm._nnmapdict = {}
                # do var updates
                for var in self.em_swarm.vars.values():
                    # if swarm migrated, update all.
                    # if var updated, update var.
                    if (self.em_swarm.particle_coordinates in writeable_vars) or (
                        var in writeable_vars
                    ):
                        var._update()

                uw.timing._decrementDepth()
                uw.timing.log_result(time.time() - stime, "Swarm.access", 1)

        return exit_manager(self)

    @timing.routine_timer_decorator
    def _get_map(self, var):
        # generate tree if not avaiable
        if not self._index:
            with self.access():
                self._index = uw.kdtree.KDTree(self.data)

        # get or generate map
        meshvar_coords = var._meshVar.coords
        # we can't use numpy arrays directly as keys in python dicts, so
        # we'll use `xxhash` to generate a hash of array.
        # this shouldn't be an issue performance wise but we should test to be
        # sufficiently confident of this.
        import xxhash

        h = xxhash.xxh64()
        h.update(meshvar_coords)
        digest = h.intdigest()
        if digest not in self._nnmapdict:
            self._nnmapdict[digest] = self._index.find_closest_point(meshvar_coords)[0]
        return self._nnmapdict[digest]

    @timing.routine_timer_decorator
    def advection(
        self,
        V_fn,
        delta_t,
        order=2,
        corrector=False,
        restore_points_to_domain_func=None,
    ):

        # X0 holds the particle location at the start of advection
        # This is needed because the particles may be migrated off-proc
        # during timestepping.

        X0 = self._X0

        # Use current velocity to estimate where the particles would have
        # landed in an implicit step.

        # ? how does this interact with the particle restoration function ?

        V_fn_matrix = self.mesh.vector.to_matrix(V_fn)

        if corrector == True and not self._X0_uninitialised:
            with self.access(self.particle_coordinates):
                v_at_Vpts = np.zeros_like(self.data)

                for d in range(self.dim):
                    v_at_Vpts[:, d] = uw.function.evaluate(
                        V_fn_matrix[d], self.data
                    ).reshape(-1)

                corrected_position = X0.data + delta_t * v_at_Vpts
                if restore_points_to_domain_func is not None:
                    corrected_position = restore_points_to_domain_func(
                        corrected_position
                    )

                updated_current_coords = 0.5 * (corrected_position + self.data)

                # validate_coords to ensure they live within the domain (or there will be trouble)

                if restore_points_to_domain_func is not None:
                    updated_current_coords = restore_points_to_domain_func(
                        updated_current_coords
                    )

                self.data[...] = updated_current_coords[...]

        with self.access(X0):
            X0.data[...] = self.data[...]
            self._X0_uninitialised = False

        # Mid point algorithm (2nd order)
        if order == 2:
            with self.access(self.particle_coordinates):

                v_at_Vpts = np.zeros_like(self.data)

                for d in range(self.dim):
                    v_at_Vpts[:, d] = uw.function.evaluate(
                        V_fn_matrix[d], self.data
                    ).reshape(-1)

                mid_pt_coords = self.data[...] + 0.5 * delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)

                if restore_points_to_domain_func is not None:
                    mid_pt_coords = restore_points_to_domain_func(mid_pt_coords)

                self.data[...] = mid_pt_coords[...]

                ## Let the swarm be updated, and then move the rest of the way

            with self.access(self.particle_coordinates):

                v_at_Vpts = np.zeros_like(self.data)

                for d in range(self.dim):
                    v_at_Vpts[:, d] = uw.function.evaluate(
                        V_fn_matrix[d], self.data
                    ).reshape(-1)

                # if (uw.mpi.rank == 0):
                #     print("Re-launch from X0", flush=True)

                new_coords = X0.data[...] + delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)
                if restore_points_to_domain_func is not None:
                    new_coords = restore_points_to_domain_func(new_coords)

                self.data[...] = new_coords[...]

        # Previous position algorithm (cf above) - we use the previous step as the
        # launch point using the current velocity field. This gives a correction to the previous
        # landing point.

        # assumes X0 is stored from the previous step ... midpoint is needed in the first step

        # forward Euler (1st order)
        else:
            with self.access(self.particle_coordinates):
                for d in range(self.dim):
                    v_at_Vpts[:, d] = uw.function.evaluate(V_fn[d], self.data).reshape(
                        -1
                    )

                new_coords = self.data + delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)

                if restore_points_to_domain_func is not None:
                    new_coords = restore_points_to_domain_func(new_coords)

                self.data[...] = new_coords

        ## Cycling of the swarm is a cheap and cheerful version of population control for particles. It turns the
        ## swarm into a streak-swarm where particles are Lagrangian for a number of steps and then reset to their
        ## original location.

        if self.recycle_rate > 1:
            # Restore particles which have cycle == cycle rate (use >= just in case)

            # Remove remesh points and recreate a new set at the mesh-local
            # locations that we already have stored.

            with self.access(self.particle_coordinates, self._remeshed):
                remeshed = self._remeshed.data[:, 0] == 0
                # This is one way to do it ... we can do this better though
                self.data[remeshed, 0] = 1.0e100

            swarm_size = self.dm.getLocalSize()

            num_remeshed_points = self.mesh.particle_X_orig.shape[0]

            self.dm.addNPoints(num_remeshed_points)

            cellid = self.dm.getField("DMSwarm_cellid")
            coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))
            rmsh = self.dm.getField("DMSwarm_remeshed")

            # print(f"cellid -> {cellid.shape}")
            # print(f"particle coords -> {coords.shape}")
            # print(f"remeshed points  -> {num_remeshed_points}")

            perturbation = (
                (0.75 / self.fill_param)
                * (np.random.random(size=(num_remeshed_points, self.dim)) - 0.5)
                * self.mesh._radii[cellid[swarm_size::]].reshape(-1, 1)
            )

            # print(f"{perturbation}")

            coords[swarm_size::] = self.mesh.particle_X_orig[:, :] + perturbation
            cellid[swarm_size::] = self.mesh.particle_CellID_orig[:, 0]
            rmsh[swarm_size::] = 0

            self.dm.restoreField("DMSwarm_cellid")
            self.dm.restoreField("DMSwarmPIC_coor")
            self.dm.restoreField("DMSwarm_remeshed")

            # when we let this go, the particles may be re-distributed to
            # other processors, and we will need to rebuild the remeshed
            # array before trying to compute / assign values to variables

            for swarmVar in self.vars.values():
                if swarmVar._rebuild_on_cycle:
                    with self.access(swarmVar):

                        if swarmVar.dtype is int:
                            nnn = 1
                        else:
                            nnn = self.mesh.dim + 1  # 3 for triangles, 4 for tets ...

                        interpolated_values = (
                            swarmVar.rbf_interpolate(self.mesh.particle_X_orig, nnn=nnn)
                            # uw.function.evaluate(swarmVar._meshVar.fn, remeshed_coords)
                        ).astype(swarmVar.dtype)

                        swarmVar.data[swarm_size::] = interpolated_values

            with self.access(self._remeshed):
                self._remeshed.data[...] = np.mod(
                    self._remeshed.data[...] - 1, self.recycle_rate
                )

            self.cycle += 1

        return
