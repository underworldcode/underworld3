from typing import Optional, Tuple, Union
import contextlib

import numpy as np
import sympy
import petsc4py.PETSc as PETSc
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities._api_tools import Stateful
from underworld3.utilities._api_tools import uw_object

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


# Note - much of the setup is necessarily the same as the MeshVariable
# and the duplication should be removed.


class SwarmVariable(Stateful, uw_object):
    """
    The SwarmVariable class generates a variable supported by a point cloud or 'swarm' and the
    underlying meshVariable representation that makes it possible to construct expressions that
    depend on the values of the swarmVariable.

    To set / read nodal values, use the numpy interface via the 'data' property.

    Parameters
    ----------
    varname :
        A textual name for this variable.
    swarm :
        The supporting underworld swarm.
    size :
        The shape of a Matrix variable type.
    vtype :
        Semi-Optional. The underworld variable type for this variable.
    proxy_degree :
        The polynomial degree for this variable.
     proxy_continuous :
        The polynomial degree for this variable.
    varsymbol:
        A symbolic form for printing etc (sympy / latex)
    rebuild_on_cycle:
        For cyclic swarm variables — True is the best choice for continuous fields

    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        size=None,  # only needed if MATRIX type
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
        import sympy
        import math

        if varsymbol is None:
            varsymbol = name

        self.name = name
        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)
        self.symbol = varsymbol

        self.swarm = swarm
        self.shape = size

        mesh = swarm.mesh

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

        self.vtype = vtype

        if not isinstance(vtype, uw.VarType):
            raise ValueError(
                "'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`."
            )

        if vtype == uw.VarType.SCALAR:
            self.num_components = 1
            self.shape = (1, 1)
            self.cpt_map = 0
        elif vtype == uw.VarType.VECTOR:
            self.num_components = mesh.dim
            self.shape = (1, mesh.dim)
            self.cpt_map = tuple(range(0, mesh.dim))
        elif vtype == uw.VarType.TENSOR:
            self.num_components = mesh.dim * mesh.dim
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.SYM_TENSOR:
            self.num_components = math.comb(mesh.dim + 1, 2)
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.MATRIX:
            self.num_components = self.shape[0] * self.shape[1]

        self._data_container = np.empty(self.shape, dtype=object)

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

        from collections import namedtuple

        SwarmVariable_ij = namedtuple("SwarmVariable_ij", ["data", "sym"])

        if self._proxy:
            for i in range(0, self.shape[0]):
                for j in range(0, self.shape[1]):
                    self._data_container[i, j] = SwarmVariable_ij(
                        data=f"SwarmVariable[...].data is only available within mesh.access() context",
                        sym=self.sym[i, j],
                    )

        super().__init__()

        return

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            if isinstance(indices, int) and self.shape[0] == 1:
                i = 0
                j = indices
            else:
                raise IndexError(
                    "SwarmVariable[i,j] access requires one or two indices "
                )
        else:
            i, j = indices

        return self._data_container[i, j]

    ## Should be a single master copy
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
        if self.vtype == uw.VarType.TENSOR:
            if self.swarm.mesh.dim == 2:
                return ((0, 1), (2, 3))[i][j]
            else:
                return ((0, 1, 2), (3, 4, 5), (6, 7, 8))[i][j]

        if self.vtype == uw.VarType.SYM_TENSOR:
            if self.swarm.mesh.dim == 2:
                return ((0, 2), (2, 1))[i][j]
            else:
                return ((0, 3, 4), (3, 1, 5), (4, 5, 2))[i][j]

        if self.vtype == uw.VarType.MATRIX:
            return i + j * self.shape[0]

    def _create_proxy_variable(self):
        # release if defined
        self._meshVar = None

        if self._proxy:
            self._meshVar = uw.discretisation.MeshVariable(
                "proxy_" + self.clean_name,
                self.swarm._mesh,
                self.shape,
                self._vtype,
                degree=self._proxy_degree,
                continuous=self._proxy_continuous,
                varsymbol=r"\left<" + self.symbol + r"\right>",
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

    def _rbf_reduce_to_meshVar(self, meshVar, verbose=False):
        """
        This method updates a mesh variable for the current
        swarm & particle variable state by reducing the swarm to
        the nearest point for each particle

        Here is how it works:

            1) for each particle, create a distance-weighted average on the node data
            2) check to see which nodes have zero weight / zero contribution and replace with nearest particle value

        Todo: caching the k-d trees etc for the proxy-mesh-variable nodal points
        Todo: some form of global fall-back for when there are no particles on a processor

        """

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return

        # 1 - Average particles to nodes with distance weighted average

        kd = uw.kdtree.KDTree(meshVar.coords)
        kd.build_index()

        with self.swarm.access():
            n, d, b = kd.find_closest_point(self.swarm.data)

            node_values = np.zeros((meshVar.coords.shape[0], self.num_components))
            w = np.zeros(meshVar.coords.shape[0])

            if not self._nn_proxy:
                for i in range(self.data.shape[0]):
                    if b[i]:
                        node_values[n[i], :] += self.data[i, :] / (1.0e-24 + d[i])
                        w[n[i]] += 1.0 / (1.0e-24 + d[i])

                node_values[np.where(w > 0.0)[0], :] /= w[np.where(w > 0.0)[0]].reshape(
                    -1, 1
                )

        # 2 - set NN vals on mesh var where w == 0.0

        p_nnmap = self.swarm._get_map(self)

        with self.swarm.mesh.access(meshVar), self.swarm.access():
            meshVar.data[...] = node_values[...]
            meshVar.data[np.where(w == 0.0), :] = self.data[
                p_nnmap[np.where(w == 0.0)], :
            ]

        return

    def rbf_interpolate(self, new_coords, verbose=False, nnn=None):
        # An inverse-distance mapping is quite robust here ... as long
        # as long we take care of the case where some nodes coincide (likely if used mesh2mesh)
        # We try to eliminate contributions from recently remeshed particles

        import numpy as np

        with self.swarm.access():
            data_size = self.data.shape

        # What to do if there are no particles
        if data_size[0] <= 1:
            return np.zeros((new_coords.shape[0], data_size[1]))

        if nnn is None:
            nnn = self.swarm.mesh.dim + 1

        if nnn > data_size[0]:
            nnn = data_size[0]

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

            values = kdt.rbf_interpolator_local(new_coords, D, nnn, verbose)

            del kdt

        return values

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError(
                "Data must be accessed via the swarm `access()` context manager."
            )
        return self._data

    @property
    def sym(self):
        return self._meshVar.sym

    @property
    def sym_1d(self):
        return self._meshVar.sym_1d

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

        force_sequential : activate the serial version of hdf5

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
                comm.barrier()
                for proc in range(1, comm.size):
                    if comm.rank == proc:
                        with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                            h5f["data"].resize(
                                (h5f["data"].shape[0] + self.data.shape[0]), axis=0
                            )
                            h5f["data"][-self.data.shape[0] :] = self.data[:]
                    comm.barrier()
                comm.barrier()

        return

    @timing.routine_timer_decorator
    def write_proxy(self, filename: str):
        # if not proxied, nothing to do. return.
        if not self._meshVar:
            if uw.mpi.rank == 0:
                print("No proxy mesh variable that can be saved", flush=True)
            return

        self._meshVar.write(filename)

        return

    @timing.routine_timer_decorator
    def read_timestep(
        self,
        data_filename: str,
        swarmID: str,
        data_name: str,
        index: int,
        outputPath="",
    ):
        # mesh.write_timestep( "test", meshUpdates=False, meshVars=[X], outputPath="", index=0)
        # swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath="", index=0)

        output_base_name = os.path.join(outputPath, data_filename)
        swarmFilename = output_base_name + f".{swarmID}.{index:05}.h5"
        filename = output_base_name + f".{swarmID}.{data_name}.{index:05}.h5"

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
            size=1,
            vtype=None,
            dtype=int,
            _proxy=False,
        )
        """
        vtype = (None,)
        dtype = (float,)
        proxy_degree = (1,)
        proxy_continuous = (True,)
        _register = (True,)
        _proxy = (True,)
        _nn_proxy = (False,)
        varsymbol = (None,)
        rebuild_on_cycle = (True,)
        """
        # The indices variable defines how many "level set" maps we create as components in the proxy variable

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

    @property
    def sym_1d(self):
        return self._MaskArray

    # We could also add a __getitem__ call to access each mask

    def __getitem__(self, index):
        return self.sym[index]

    def createMask(self, funcsList):
        """
        This creates a masked sympy function of swarm variables required for Underworld's solvers
        """

        if not isinstance(funcsList, (tuple, list)):
            raise RuntimeError("Error input for createMask() - wrong type of input")

        if len(funcsList) != self.indices:
            raise RuntimeError("Error input for createMask() - wrong length of input")

        symo = sympy.simplify(0)
        for i in range(self.indices):
            symo += funcsList[i] * self._MaskArray[i]

        return symo

    def viewMask(self, sympy):
        """
        Takes a previously masked sympy function and returns individual sympy objects corresponding to each material
        """

        """ TODO
        output = []
        for i in range( self.indices ):
            tmp = {}
            for j in range( self.indices ):
                if i == j : pass
                tmp

        return output
        """
        pass

    def visMask(self):
        return self.createMask(list(range(self.indices)))

    def view(self):
        """
        Show information on IndexSwarmVariable
        """
        if uw.mpi.rank == 0:
            print(f"IndexSwarmVariable {self}")
            print(f"Numer of indices {self.indices}")

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
class Swarm(Stateful, uw_object):
    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, mesh, recycle_rate=0):
        Swarm.instances += 1

        self.celldm = mesh.dm.clone()

        self._mesh = mesh
        self.dim = mesh.dim
        self.cdim = mesh.cdim
        self.dm = PETSc.DMSwarm().create()
        self.dm.setDimension(self.dim)
        self.dm.setType(SwarmType.DMSWARM_PIC.value)
        self.dm.setCellDM(self.celldm)
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

    @timing.routine_timer_decorator
    def populate_petsc(
        self,
        fill_param: Optional[int] = 3,
        layout: Optional[SwarmPICLayout] = None,
    ):
        """
        Populate the swarm with particles throughout the domain.

        When using SwarmPICLayout.REGULAR,     `fill_param` defines the number of points in each spatial direction.
        When using SwarmPICLayout.GAUSS,       `fill_param` defines the number of quadrature points in each spatial direction.
        When using SwarmPICLayout.SUBDIVISION, `fill_param` defines the number times the reference cell is sub-divided.

        Parameters
        ----------
        fill_param:
            Parameter determining the particle count per cell for the given layout.
        layout:
            Type of layout to use. Defaults to `SwarmPICLayout.REGULAR` for mesh objects with simplex
            type cells, and `SwarmPICLayout.GAUSS` otherwise.

        """

        self.fill_param = fill_param

        """
        Currently (2021.11.15) supported by PETSc release 3.16.x

        When using a DMPLEX the following case are supported:
              (i) DMSWARMPIC_LAYOUT_REGULAR: 2D (triangle),
             (ii) DMSWARMPIC_LAYOUT_GAUSS: 2D and 3D provided the cell is a tri/tet or a quad/hex,
            (iii) DMSWARMPIC_LAYOUT_SUBDIVISION: 2D and 3D for quad/hex and 2D tri.

        So this means, simplex mesh in 3D only supports GAUSS - This is based
        on the tensor product locations so it is not even in the cells.
        """

        if layout == None:
            if self.mesh.isSimplex == True and self.dim == 2 and fill_param > 1:
                layout = SwarmPICLayout.REGULAR
            else:
                layout = SwarmPICLayout.GAUSS

        if not isinstance(layout, SwarmPICLayout):
            raise ValueError("'layout' must be an instance of 'SwarmPICLayout'")

        self.layout = layout
        self.dm.finalizeFieldRegister()

        ## Commenting this out for now.
        ## Code seems to operate fine without it, and the
        ## existing values are wrong. It should be something like
        ## `(elend-elstart)*fill_param^dim` for quads, and around
        ## half that for simplices, depending on layout.
        # elstart,elend = self.mesh.dm.getHeightStratum(0)
        # self.dm.setLocalSizes((elend-elstart) * fill_param, 0)

        self.dm.insertPointUsingCellDM(self.layout.value, fill_param)
        return  # self # LM: Is there any reason to return self ?

    #

    @timing.routine_timer_decorator
    def populate(
        self,
        fill_param: Optional[int] = 1,
    ):
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

        self.fill_param = fill_param

        newp_coords0 = self.mesh._get_coords_for_basis(fill_param, continuous=False)
        newp_cells0 = self.mesh.get_closest_local_cells(newp_coords0)

        valid = newp_cells0 != -1
        newp_coords = newp_coords0[valid]
        newp_cells = newp_cells0[valid]

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

        # cellid = self.dm.getField("DMSwarm_cellid")
        # lost = np.where(cellid == -1)
        # print(f"{uw.mpi.rank} - lost particles: {lost[0].shape} out of {cellid.shape}", flush=True)
        # self.dm.restoreField("DMSwarm_cellid")

        if self.recycle_rate > 1:
            with self.access():
                # This is a mesh-local quantity, so let's just
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
                + (0.33 / (1 + fill_param))
                * (np.random.random(size=all_local_coords.shape) - 0.5)
                * 0.00001
                * self.mesh._search_lengths[all_local_cells]  # typical cell size
            )
            cellid[:] = all_local_cells[:, 0]

            self.dm.restoreField("DMSwarmPIC_coor")
            self.dm.restoreField("DMSwarm_cellid")

            ## Now set the cycle values

            with self.access(self._remeshed):
                for i in range(0, self.recycle_rate):
                    offset = swarm_orig_size * i
                    self._remeshed.data[offset::, 0] = i

        # Validate (eliminate if required)

        # cellid = self.dm.getField("DMSwarm_cellid")
        # lost = np.where(cellid == -1)
        # print(f"{uw.mpi.rank} - lost particles: {lost[0].shape} out of {cellid.shape}", flush=True)
        # self.dm.restoreField("DMSwarm_cellid")

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

            del data_copy

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

            del data_copy

        return

    @timing.routine_timer_decorator
    def read_timestep(
        self,
        base_filename: str,
        swarm_id: str,
        index: int,
        outputPath: Optional[str] = "",
    ):
        output_base_name = os.path.join(outputPath, base_filename)
        swarm_file = output_base_name + f".{swarm_id}.{index:05}.h5"

        ### open up file with coords on all procs
        with h5py.File(f"{swarm_file}", "r") as h5f:
            coordinates = h5f["coordinates"][:]

        #### utilises the UW function for adding a swarm by an array
        self.add_particles_with_coordinates(coordinates)

        return

    @timing.routine_timer_decorator
    def add_variable(
        self,
        name,
        size=1,
        dtype=float,
        proxy_degree=2,
        _nn_proxy=False,
    ):
        return SwarmVariable(
            name,
            self,
            size,
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

        x_swarm_fname = f"{outputPath}{swarmName}_{index:05d}.xmf"
        self.dm.viewXDMF(x_swarm_fname)

    @timing.routine_timer_decorator
    def write_timestep(
        self,
        filename: str,
        swarmname: str,
        index: int,
        swarmVars: Optional[list] = None,
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

        # This will eliminate the issue of whether or not to put path separators in the
        # outputPath. Also does the right thing if outputPath is ""

        import os

        output_base_name = os.path.join(outputPath, filename) + "." + swarmname

        # could also try to coerce this to be a list and raise if it fails (tuple, singleton ... )
        # also ... why the typechecking if this can still happen

        if swarmVars is not None and not isinstance(swarmVars, list):
            raise RuntimeError("`swarmVars` does not appear to be a list.")

        else:
            ### save the swarm particle location
            self.save(
                filename=f"{output_base_name}.{index:05d}.h5",
                compression=compression,
                compressionType=compressionType,
                force_sequential=force_sequential,
            )

        #### Generate a h5 file for each field
        if swarmVars != None:
            for field in swarmVars:
                field.save(
                    filename=f"{output_base_name}.{field.name}.{index:05d}.h5",
                    compression=compression,
                    compressionType=compressionType,
                    force_sequential=force_sequential,
                )

        if uw.mpi.rank == 0:
            ### only need to combine the h5 files to a single xdmf on one proc
            with open(f"{output_base_name}.{index:05d}.xdmf", "w") as xdmf:
                # Write the XDMF header
                xdmf.write('<?xml version="1.0" ?>\n')
                xdmf.write(
                    '<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n'
                )
                xdmf.write("<Domain>\n")
                xdmf.write(
                    f'<Grid Name="{output_base_name}.{index:05d}" GridType="Uniform">\n'
                )

                if time != None:
                    xdmf.write(f'	<Time Value="{time}" />\n')

                # Write the grid element for the HDF5 dataset
                with h5py.File(f"{output_base_name}.{index:05}.h5", "r") as h5f:
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
                            f"{output_base_name}.{field.name}.{index:05d}.h5", "r"
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
            assert var._data is not None
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()

            # make view for each var component
            if var._proxy:
                for i in range(0, var.shape[0]):
                    for j in range(0, var.shape[1]):
                        var._data_container[i, j] = var._data_container[i, j]._replace(
                            data=var.data[:, var._data_layout(i, j)],
                        )

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

                    if var._proxy:
                        for i in range(0, var.shape[0]):
                            for j in range(0, var.shape[1]):
                                # var._data_ij[i, j] = None
                                var._data_container[i, j] = var._data_container[
                                    i, j
                                ]._replace(
                                    data=f"SwarmVariable[...].data is only available within mesh.access() context",
                                )

                uw.timing._decrementDepth()
                uw.timing.log_result(time.time() - stime, "Swarm.access", 1)

        return exit_manager(self)

    ## Better to have one master copy - this one is cut'n'pasted from
    ## the MeshVariable class

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
        if self.vtype == uw.VarType.TENSOR:
            if self.mesh.dim == 2:
                return ((0, 1), (2, 3))[i][j]
            else:
                return ((0, 1, 2), (3, 4, 5), (6, 7, 8))[i][j]

        if self.vtype == uw.VarType.SYM_TENSOR:
            if self.mesh.dim == 2:
                return ((0, 2), (2, 1))[i][j]
            else:
                return ((0, 3, 4), (3, 1, 5), (4, 5, 2))[i][j]

        if self.vtype == uw.VarType.MATRIX:
            return i + j * self.shape[0]

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
        evalf=False,
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

                if evalf:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evalf(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)
                else:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)

                corrected_position = X0.data.copy() + delta_t * v_at_Vpts
                if restore_points_to_domain_func is not None:
                    corrected_position = restore_points_to_domain_func(
                        corrected_position
                    )

                updated_current_coords = 0.5 * (corrected_position + self.data.copy())

                # validate_coords to ensure they live within the domain (or there will be trouble)

                if restore_points_to_domain_func is not None:
                    updated_current_coords = restore_points_to_domain_func(
                        updated_current_coords
                    )

                self.data[...] = updated_current_coords[...]

                del updated_current_coords
                del v_at_Vpts

        with self.access(X0):
            X0.data[...] = self.data[...]
            self._X0_uninitialised = False

        # Mid point algorithm (2nd order)
        if order == 2:
            with self.access(self.particle_coordinates):
                v_at_Vpts = np.zeros_like(self.data)

                if evalf:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evalf(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)
                else:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)

                mid_pt_coords = self.data[...].copy() + 0.5 * delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)

                if restore_points_to_domain_func is not None:
                    mid_pt_coords = restore_points_to_domain_func(mid_pt_coords)

                self.data[...] = mid_pt_coords[...]

                del mid_pt_coords

                ## Let the swarm be updated, and then move the rest of the way

            with self.access(self.particle_coordinates):
                v_at_Vpts = np.zeros_like(self.data)

                if evalf:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evalf(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)
                else:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)

                # if (uw.mpi.rank == 0):
                #     print("Re-launch from X0", flush=True)

                new_coords = X0.data[...].copy() + delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)
                if restore_points_to_domain_func is not None:
                    new_coords = restore_points_to_domain_func(new_coords)

                self.data[...] = new_coords[...]

                del new_coords
                del v_at_Vpts

        # Previous position algorithm (cf above) - we use the previous step as the
        # launch point using the current velocity field. This gives a correction to the previous
        # landing point.

        # assumes X0 is stored from the previous step ... midpoint is needed in the first step

        # forward Euler (1st order)
        else:
            with self.access(self.particle_coordinates):
                v_at_Vpts = np.zeros_like(self.data)

                if evalf:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evalf(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)
                else:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d], self.data
                        ).reshape(-1)

                new_coords = self.data + delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)

                if restore_points_to_domain_func is not None:
                    new_coords = restore_points_to_domain_func(new_coords)

                self.data[...] = new_coords[...].copy()

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

            perturbation = 0.00001 * (
                (0.33 / (1 + self.fill_param))
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
                            #     swarmVar._meshVar.fn, self.mesh.particle_X_orig
                            # )
                        ).astype(swarmVar.dtype)

                        swarmVar.data[swarm_size::] = interpolated_values

            self.dm.migrate(remove_sent_points=True)

            with self.access(self._remeshed):
                self._remeshed.data[...] = np.mod(
                    self._remeshed.data[...] - 1, self.recycle_rate
                )

            self.cycle += 1

        return


class NodalPointSwarm(Swarm):
    r"""Swarm with particles located at the coordinate points of a meshVariable

    The swarmVariable `X0` is defined so that the particles can "snap back" to their original locations
    after they have been moved.

    The purpose of this Swarm is to manage sample points for advection schemes based on upstream sampling
    (method of characteristics etc)"""

    def __init__(
        self,
        trackedVariable: uw.discretisation.MeshVariable,
    ):
        self.trackedVariable = trackedVariable
        self.swarmVariable = None
        mesh = trackedVariable.mesh

        # Set up a standard swarm
        super().__init__(mesh)

        nswarm = self

        meshVar_name = trackedVariable.clean_name
        meshVar_symbol = trackedVariable.symbol

        ks = str(self.instance_number)
        name = f"{meshVar_name}_star"
        symbol = rf"{{ {meshVar_symbol} }}^{{ <*> }}"

        self.swarmVariable = uw.swarm.SwarmVariable(
            name,
            nswarm,
            vtype=trackedVariable.vtype,
            proxy_degree=trackedVariable.degree,
            proxy_continuous=trackedVariable.continuous,
            varsymbol=symbol,
        )

        name = f"ns_X0_{ks}"
        symbol = r"X0^{*^{{[" + ks + "]}}}"
        nX0 = uw.swarm.SwarmVariable(name, nswarm, nswarm.dim, _proxy=False)

        nswarm.dm.finalizeFieldRegister()
        nswarm.dm.addNPoints(
            trackedVariable.coords.shape[0] + 1
        )  # why + 1 ? That's the number of spots actually allocated

        cellid = nswarm.dm.getField("DMSwarm_cellid")
        coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape((-1, nswarm.dim))
        coords[...] = trackedVariable.coords[...]
        cellid[:] = self.mesh.get_closest_cells(coords)

        # Move slightly within the chosen cell to avoid edge effects
        centroid_coords = self.mesh._centroids[cellid]
        shift = 1.0e-3
        coords[...] = (1.0 - shift) * coords[...] + shift * centroid_coords[...]

        nswarm.dm.restoreField("DMSwarmPIC_coor")
        nswarm.dm.restoreField("DMSwarm_cellid")
        nswarm.dm.migrate(remove_sent_points=True)

        with nswarm.access(nX0):
            nX0.data[...] = coords

        self._nswarm = nswarm
        self._X0 = nX0

        return


# class SemiLagrange_D_Dt(uw_object):
#     r"""Nodal-Swarm  Lagrangian History Manager:
#     This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.
#     $$\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$$
#     $$\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$$
#     $$\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$$
#     """

#     @timing.routine_timer_decorator
#     def __init__(
#         self,
#         mesh: uw.discretisation.Mesh,
#         psi_fn: sympy.Function,
#         V_fn: sympy.Function,
#         vtype: uw.VarType,
#         degree: int,
#         continuous: bool,
#         varsymbol: Optional[str] = r"u",
#         verbose: Optional[bool] = False,
#         bcs=[],
#         order=1,
#         smoothing=0.0,
#     ):
#         super().__init__()

#         self.mesh = mesh
#         self.bcs = bcs
#         self.verbose = verbose
#         self.degree = degree

#         # meshVariables are required for:
#         #
#         # u(t) - evaluation of u_fn at the current time
#         # u*(t) - u_* evaluated from

#         # psi is evaluated/stored at `order` timesteps. We can't
#         # be sure if psi is a meshVariable or a function to be evaluated
#         # psi_star is reaching back through each evaluation and has to be a
#         # meshVariable (storage)

#         self._psi_fn = psi_fn
#         self.V_fn = V_fn
#         self.order = order

#         psi_star = []
#         self.psi_star = psi_star

#         for i in range(order):
#             self.psi_star.append(
#                 uw.discretisation.MeshVariable(
#                     f"psi_star_sl_{self.instance_number}_{i}",
#                     self.mesh,
#                     vtype=vtype,
#                     degree=degree,
#                     continuous=continuous,
#                     varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
#                 )
#             )

#         # We just need one swarm since this is inherently a sequential operation
#         nswarm = uw.swarm.NodalPointSwarm(self.psi_star[0])
#         self._nswarm_psi = nswarm

#         # The projection operator for mapping swarm values to the mesh - needs to be different for
#         # each variable type, unfortunately ...

#         if vtype == uw.VarType.SCALAR:
#             self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
#                 self.mesh, self.psi_star[0], verbose=False
#             )
#         elif vtype == uw.VarType.VECTOR:
#             self._psi_star_projection_solver = (
#                 uw.systems.solvers.SNES_Vector_Projection(
#                     self.mesh, self.psi_star[0], verbose=False
#                 )
#             )
#         elif vtype == uw.VarType.SYM_TENSOR or vtype == uw.VarType.TENSOR:
#             self._WorkVar = uw.discretisation.MeshVariable(
#                 f"W_star_slcn_{self.instance_number}",
#                 self.mesh,
#                 vtype=uw.VarType.SCALAR,
#                 degree=degree,
#                 continuous=continuous,
#                 varsymbol=r"W^{*}",
#             )
#             self._psi_star_projection_solver = (
#                 uw.systems.solvers.SNES_Tensor_Projection(
#                     self.mesh, self.psi_star[0], self._WorkVar, verbose=False
#                 )
#             )

#         self._psi_star_projection_solver.uw_function = self.psi_fn
#         self._psi_star_projection_solver.bcs = bcs
#         self._psi_star_projection_solver.smoothing = smoothing

#         return

#     @property
#     def psi_fn(self):
#         return self._psi_fn

#     @psi_fn.setter
#     def psi_fn(self, new_fn):
#         self._psi_fn = new_fn
#         self._psi_star_projection_solver.uw_function = self._psi_fn
#         return

#     def _object_viewer(self):
#         from IPython.display import Latex, Markdown, display

#         super()._object_viewer()

#         ## feedback on this instance
#         # display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
#         # display(
#         #     Latex(
#         #         r"$\quad\Delta t_{\textrm{phys}} = $ "
#         #         + sympy.sympify(self.dt_physical)._repr_latex_()
#         #     )
#         # )
#         display(Latex(rf"$\quad$History steps = {self.order}"))

#     def update(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         self.update_pre_solve(dt, evalf, verbose)
#         return

#     def update_post_solve(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         return

#     def update_pre_solve(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         # if average_over_dt:
#         #     phi = min(1.0, dt / self.dt_physical)
#         # else:
#         #     phi = 1.0

#         if verbose and uw.mpi.rank == 0:
#             print(f"Update {self.psi_fn}", flush=True)

#         ## Progress from the oldest part of the history
#         # 1. Copy the stored values down the chain

#         for i in range(self.order - 1, 0, -1):
#             with self.mesh.access(self.psi_star[i]):
#                 self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

#         # 2. Compute the upstream values

#         # We use the u_star variable as a working value here so we have to work backwards
#         for i in range(self.order - 1, -1, -1):
#             with self._nswarm_psi.access(self._nswarm_psi._X0):
#                 self._nswarm_psi._X0.data[...] = self._nswarm_psi.data[...]

#             # march nodes backwards along characteristics
#             self._nswarm_psi.advection(
#                 self.V_fn,
#                 -dt,
#                 order=2,
#                 corrector=False,
#                 restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
#             )

#             if i == 0:
#                 # Recalculate u_star from u_fn
#                 self._psi_star_projection_solver.uw_function = self.psi_fn
#                 self._psi_star_projection_solver.solve()

#             with self._nswarm_psi.access(self._nswarm_psi.swarmVariable):
#                 for d in range(self.psi_star[i].shape[1]):
#                     self._nswarm_psi.swarmVariable.data[:, d] = uw.function.evaluate(
#                         self.psi_star[i].sym[d], self._nswarm_psi.data
#                     )

#             # restore coords (will call dm.migrate after context manager releases)
#             with self._nswarm_psi.access(self._nswarm_psi.particle_coordinates):
#                 self._nswarm_psi.data[...] = self._nswarm_psi._X0.data[...]

#             # Now project to the mesh using bc's to obtain u_star
#             self._psi_star_projection_solver.uw_function = (
#                 self._nswarm_psi.swarmVariable.sym
#             )
#             self._psi_star_projection_solver.solve()

#             # Copy data from the projection operator if required
#             if i != 0:
#                 with self.mesh.access(self.psi_star[i]):
#                     self.psi_star[i].data[...] = self.psi_star[0].data[...]

#         return

#     def bdf(self, order=None):
#         r"""Backwards differentiation form for calculating DuDt
#         Note that you will need `bdf` / $\delta t$ in computing derivatives"""

#         if order is None:
#             order = self.order
#         else:
#             order = max(1, min(self.order, order))

#         with sympy.core.evaluate(False):
#             if order == 1:
#                 bdf0 = self.psi_fn - self.psi_star[0].sym

#             elif order == 2:
#                 bdf0 = (
#                     3 * self.psi_fn / 2
#                     - 2 * self.psi_star[0].sym
#                     + self.psi_star[1].sym / 2
#                 )

#             elif order == 3:
#                 bdf0 = (
#                     11 * self.psi_fn / 6
#                     - 3 * self.psi_star[0].sym
#                     + 3 * self.psi_star[1].sym / 2
#                     - self.psi_star[2].sym / 3
#                 )

#         return bdf0

#     def adams_moulton_flux(self, order=None):
#         if order is None:
#             order = self.order
#         else:
#             order = max(1, min(self.order, order))

#         with sympy.core.evaluate(False):
#             if order == 1:
#                 am = (self.psi_fn + self.psi_star[0].sym) / 2

#             elif order == 2:
#                 am = (
#                     5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
#                 ) / 12

#             elif order == 3:
#                 am = (
#                     9 * self.psi_fn
#                     + 19 * self.psi_star[0].sym
#                     - 5 * self.psi_star[1].sym
#                     + self.psi_star[2].sym
#                 ) / 24

#         return am


# class Lagrangian_D_Dt(uw_object):
#     r"""Swarm-based Lagrangian History Manager:

#     This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.

#     $\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$

#     $\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$

#     $\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$
#     """

#     instances = 0  # count how many of these there are in order to create unique private mesh variable ids

#     @timing.routine_timer_decorator
#     def __init__(
#         self,
#         mesh: uw.discretisation.Mesh,
#         psi_fn: sympy.Function,
#         V_fn: sympy.Function,
#         vtype: uw.VarType,
#         degree: int,
#         continuous: bool,
#         varsymbol: Optional[str] = r"u",
#         verbose: Optional[bool] = False,
#         bcs=[],
#         order=1,
#         smoothing=0.0,
#         fill_param=3,
#     ):
#         super().__init__()

#         # create a new swarm to manage here
#         dudt_swarm = uw.swarm.Swarm(mesh)

#         self.mesh = mesh
#         self.swarm = dudt_swarm
#         self.psi_fn = psi_fn
#         self.V_fn = V_fn
#         self.verbose = verbose
#         self.order = order

#         psi_star = []
#         self.psi_star = psi_star

#         for i in range(order):
#             print(f"Creating psi_star[{i}]")
#             self.psi_star.append(
#                 uw.swarm.SwarmVariable(
#                     f"psi_star_sw_{self.instance_number}_{i}",
#                     self.swarm,
#                     vtype=vtype,
#                     proxy_degree=degree,
#                     proxy_continuous=continuous,
#                     varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
#                 )
#             )

#         dudt_swarm.populate(fill_param)

#         return

#     def _object_viewer(self):
#         from IPython.display import Latex, Markdown, display

#         super()._object_viewer()

#         ## feedback on this instance
#         display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
#         display(
#             Latex(
#                 r"$\quad\Delta t_{\textrm{phys}} = $ "
#                 + sympy.sympify(self.dt_physical)._repr_latex_()
#             )
#         )
#         display(Latex(rf"$\quad$History steps = {self.order}"))

#     ## Note: We may be able to eliminate this
#     ## The SL updater and the Lag updater have
#     ## different sequencing because of the way they
#     ## update the history. It makes more sense for the
#     ## full Lagrangian swarm to be updated after the solve
#     ## and this means we have to grab the history values first.

#     def update(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         self.update_post_solve(dt, evalf, verbose)
#         return

#     def update_pre_solve(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         return

#     def update_post_solve(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         for h in range(self.order - 1):
#             i = self.order - (h + 1)

#             # copy the information down the chain
#             print(f"Lagrange order = {self.order}")
#             print(f"Lagrange copying {i-1} to {i}")

#             with self.swarm.access(self.psi_star[i]):
#                 self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

#         # Now update the swarm variable

#         if evalf:
#             psi_star_0 = self.psi_star[0]
#             with self.swarm.access(psi_star_0):
#                 for i in range(psi_star_0.shape[0]):
#                     for j in range(psi_star_0.shape[1]):
#                         updated_psi = uw.function.evalf(
#                             self.psi_fn[i, j], self.swarm.data
#                         )
#                         psi_star_0[i, j].data[:] = updated_psi

#         else:
#             psi_star_0 = self.psi_star[0]
#             with self.swarm.access(psi_star_0):
#                 for i in range(psi_star_0.shape[0]):
#                     for j in range(psi_star_0.shape[1]):
#                         updated_psi = uw.function.evaluate(
#                             self.psi_fn[i, j], self.swarm.data
#                         )
#                         psi_star_0[i, j].data[:] = updated_psi

#         # Now update the swarm locations

#         self.swarm.advection(
#             self.V_fn,
#             delta_t=dt,
#             restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
#         )

#     def bdf(self, order=None):
#         r"""Backwards differentiation form for calculating DuDt
#         Note that you will need `bdf` / $\delta t$ in computing derivatives"""

#         if order is None:
#             order = self.order
#         else:
#             order = max(1, min(self.order, order))

#         with sympy.core.evaluate(False):
#             if order <= 1:
#                 bdf0 = self.psi_fn - self.psi_star[0].sym

#             elif order == 2:
#                 bdf0 = (
#                     3 * self.psi_fn / 2
#                     - 2 * self.psi_star[0].sym
#                     + self.psi_star[1].sym / 2
#                 )

#             elif order == 3:
#                 bdf0 = (
#                     11 * self.psi_fn / 6
#                     - 3 * self.psi_star[0].sym
#                     + 3 * self.psi_star[1].sym / 2
#                     - self.psi_star[2].sym / 3
#                 )

#         return bdf0

#     def adams_moulton_flux(self, order=None):
#         if order is None:
#             order = self.order
#         else:
#             order = max(1, min(self.order, order))

#         with sympy.core.evaluate(False):
#             if order == 1:
#                 am = (self.psi_fn + self.psi_star[0].sym) / 2

#             elif order == 2:
#                 am = (
#                     5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
#                 ) / 12

#             elif order == 3:
#                 am = (
#                     9 * self.psi_fn
#                     + 19 * self.psi_star[0].sym
#                     - 5 * self.psi_star[1].sym
#                     + self.psi_star[2].sym
#                 ) / 24

#         return am


# class Lagrangian_Swarm_D_Dt(uw_object):
#     r"""Swarm-based Lagrangian History Manager:
#     This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.

#     $\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$

#     $\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$

#     $\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$
#     """

#     instances = 0  # count how many of these there are in order to create unique private mesh variable ids

#     @timing.routine_timer_decorator
#     def __init__(
#         self,
#         swarm: Swarm,
#         psi_fn: sympy.Function,
#         vtype: uw.VarType,
#         degree: int,
#         continuous: bool,
#         varsymbol: Optional[str] = r"u",
#         verbose: Optional[bool] = False,
#         bcs=[],
#         order=1,
#         smoothing=0.0,
#     ):
#         super().__init__()

#         self.mesh = swarm.mesh
#         self.swarm = swarm
#         self.psi_fn = psi_fn
#         self.verbose = verbose
#         self.order = order

#         psi_star = []
#         self.psi_star = psi_star

#         for i in range(order):
#             print(f"Creating psi_star[{i}]")
#             self.psi_star.append(
#                 uw.swarm.SwarmVariable(
#                     f"psi_star_sw_{self.instance_number}_{i}",
#                     self.swarm,
#                     vtype=vtype,
#                     proxy_degree=degree,
#                     proxy_continuous=continuous,
#                     varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
#                 )
#             )

#         return

#     def _object_viewer(self):
#         from IPython.display import Latex, Markdown, display

#         super()._object_viewer()

#         ## feedback on this instance
#         display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
#         display(
#             Latex(
#                 r"$\quad\Delta t_{\textrm{phys}} = $ "
#                 + sympy.sympify(self.dt_physical)._repr_latex_()
#             )
#         )
#         display(Latex(rf"$\quad$History steps = {self.order}"))

#     ## Note: We may be able to eliminate this
#     ## The SL updater and the Lag updater have
#     ## different sequencing because of the way they
#     ## update the history. It makes more sense for the
#     ## full Lagrangian swarm to be updated after the solve
#     ## and this means we have to grab the history values first.

#     def update(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         self.update_post_solve(dt, evalf, verbose)
#         return

#     def update_pre_solve(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         return

#     def update_post_solve(
#         self,
#         dt: float,
#         evalf: Optional[bool] = False,
#         verbose: Optional[bool] = False,
#     ):
#         for h in range(self.order - 1):
#             i = self.order - (h + 1)

#             # copy the information down the chain
#             print(f"Lagrange order = {self.order}")
#             print(f"Lagrange copying {i-1} to {i}")

#             with self.swarm.access(self.psi_star[i]):
#                 self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

#         # Now update the swarm variable

#         if evalf:
#             psi_star_0 = self.psi_star[0]
#             with self.swarm.access(psi_star_0):
#                 for i in range(psi_star_0.shape[0]):
#                     for j in range(psi_star_0.shape[1]):
#                         updated_psi = uw.function.evalf(
#                             self.psi_fn[i, j], self.swarm.data
#                         )
#                         psi_star_0[i, j].data[:] = updated_psi

#         else:
#             psi_star_0 = self.psi_star[0]
#             with self.swarm.access(psi_star_0):
#                 for i in range(psi_star_0.shape[0]):
#                     for j in range(psi_star_0.shape[1]):
#                         updated_psi = uw.function.evaluate(
#                             self.psi_fn[i, j], self.swarm.data
#                         )
#                         psi_star_0[i, j].data[:] = updated_psi

#     def bdf(self, order=None):
#         r"""Backwards differentiation form for calculating DuDt
#         Note that you will need `bdf` / $\delta t$ in computing derivatives"""

#         if order is None:
#             order = self.order
#         else:
#             order = max(1, min(self.order, order))

#         with sympy.core.evaluate(False):
#             if order <= 1:
#                 bdf0 = self.psi_fn - self.psi_star[0].sym

#             elif order == 2:
#                 bdf0 = (
#                     3 * self.psi_fn / 2
#                     - 2 * self.psi_star[0].sym
#                     + self.psi_star[1].sym / 2
#                 )

#             elif order == 3:
#                 bdf0 = (
#                     11 * self.psi_fn / 6
#                     - 3 * self.psi_star[0].sym
#                     + 3 * self.psi_star[1].sym / 2
#                     - self.psi_star[2].sym / 3
#                 )

#         return bdf0

#     def adams_moulton_flux(self, order=None):
#         if order is None:
#             order = self.order
#         else:
#             order = max(1, min(self.order, order))

#         with sympy.core.evaluate(False):
#             if order == 1:
#                 am = (self.psi_fn + self.psi_star[0].sym) / 2

#             elif order == 2:
#                 am = (
#                     5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
#                 ) / 12

#             elif order == 3:
#                 am = (
#                     9 * self.psi_fn
#                     + 19 * self.psi_star[0].sym
#                     - 5 * self.psi_star[1].sym
#                     + self.psi_star[2].sym
#                 ) / 24

#         return am
