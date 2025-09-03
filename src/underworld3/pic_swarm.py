from posixpath import pardir
import petsc4py.PETSc as PETSc

import numpy as np
import sympy
import h5py
import os
import warnings
from typing import Optional, Tuple

import underworld3 as uw
from underworld3.utilities._api_tools import Stateful
from underworld3.utilities._api_tools import uw_object

import underworld3.timing as timing

comm = uw.mpi.comm

from enum import Enum

# Import necessary classes from swarm module
from .swarm import SwarmType, SwarmVariable


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


class PICSwarm(Stateful, uw_object):
    """
    Particle swarm implementation with automatic mesh-particle interactions.

    The `Swarm` class is Underworld's primary particle management system, built on PETSc's
    DMSWARM_PIC type. It provides automatic particle migration, mesh-particle connectivity,
    and streamlined particle operations for Lagrangian particle tracking and data storage.

    Differences from UW Swarm:
    - **Mesh Integration**: Built-in particle-in-cell (PIC) connectivity with automatic cell tracking
    - **Migration**: Uses the standard PETSc strategy for migration which depends on the DM type. This requires
      calculation of cell-relationships each time the coordinates are updated and particles that are not found will
      be deleted.


    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        The mesh object that defines the computational domain. Particles will be
        automatically associated with mesh cells for efficient spatial operations.
    recycle_rate : int, optional
        Rate at which particles are recycled for streak management. If > 1, enables
        streak particle functionality where particles are duplicated and tracked
        across multiple cycles. Default is 0 (no recycling).
    verbose : bool, optional
        Enable verbose output for debugging and monitoring particle operations.
        Default is False.

    Attributes
    ----------
    mesh : uw.discretisation.Mesh
        Reference to the associated mesh object.
    dim : int
        Spatial dimension of the mesh (2D or 3D).
    cdim : int
        Coordinate dimension of the mesh.
    data : numpy.ndarray
        Direct access to particle coordinate data.
    particle_coordinates : SwarmVariable
        SwarmVariable containing particle coordinate information (auto-created).
    particle_cellid : SwarmVariable
        SwarmVariable containing particle cell ID information (auto-created).
    recycle_rate : int
        Current recycle rate for streak management.
    cycle : int
        Current cycle number for streak particles.

    Methods
    -------
    populate_petsc(fill_param=1)
        Populate swarm using PETSc's built-in particle generation.
    populate(fill_param=1, layout=SwarmPICLayout.GAUSS)
        Populate the swarm with particles using specified layout.
    add_particles_with_coordinates(coords)
        Add new particles at specified coordinate locations.
    add_variable(name, size, dtype=float)
        Add a new variable to track additional particle properties.
    save(filename, meshUnits=1.0, swarmUnits=1.0, units="dimensionless")
        Save swarm data to file.
    read_timestep(filename, step_name, outputPath="./output/")
        Read swarm data from a specific timestep file.
    advection(V_fn, delta_t, evalf=False, corrector=True, restore_points_func=None)
        Advect particles using a velocity field with automatic migration.
    estimate_dt(V_fn, dt_min=1.0e-15, dt_max=1.0)
        Estimate appropriate timestep for particle advection.

    Examples
    --------
    Create a standard swarm with automatic features:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1))
    >>> swarm = uw.swarm.Swarm(mesh=mesh)
    >>> swarm.populate(fill_param=2, layout=uw.swarm.SwarmPICLayout.GAUSS)

    Access automatic coordinate and cell ID fields:

    >>> coords = swarm._particle_coordinates.data
    >>> cell_ids = swarm.particle_cellid.data

    Create a streak swarm with recycling:

    >>> streak_swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=5)
    >>> streak_swarm.populate(fill_param=1)

    Add custom particle data and perform advection:

    >>> temperature = swarm.add_variable("temperature", 1)
    >>> velocity_field = mesh.add_variable("velocity", mesh.dim)
    >>> # ... set up velocity field ...
    >>> swarm.advection(velocity_field.sym, delta_t=0.01)  # Automatic migration

    Notes
    -----
    - Particle migration occurs automatically during advection operations
    - Coordinate and cell ID fields are created and managed automatically at the
      PETSc level


    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, mesh, recycle_rate=0, verbose=False):
        PICSwarm.instances += 1

        self.celldm = mesh.dm.clone()

        self.verbose = verbose
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
        self._index = None
        self._nnmapdict = {}

        super().__init__()

    @property
    def mesh(self):
        return self._mesh

    @property
    def data(self):
        return self._particle_coordinates.data

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
        on the tensor product locations so it is not uniform in the cells.
        """

        if layout == None:
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
        return

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

        if np.any(newp_cells0 > self.mesh._centroids.shape[0]):
            raise RuntimeError("Some new coordinates can't find a owning cell - Error")

        # valid = newp_cells0 != -1
        # newp_coords = newp_coords0[valid]
        # newp_cells = newp_cells0[valid]
        newp_coords = newp_coords0
        newp_cells = newp_cells0

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

                self.mesh.particle_X_orig = self._particle_coordinates.data.copy()
                self.mesh.particle_CellID_orig = self._cellid_var.data.copy()

            with self.access():
                swarm_orig_size = self._particle_coordinates.data.shape[0]
                all_local_coords = np.vstack(
                    (self._particle_coordinates.data,) * (self.recycle_rate)
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

        return

    @timing.routine_timer_decorator
    def add_particles_with_coordinates(self, coordinatesArray) -> int:
        """
        Add particles to the swarm using particle coordinates provided
        using a numpy array.

        Note that particles with coordinates NOT local to the current processor will
        be rejected / ignored.

        Either include an array with all coordinates to all processors
        or an array with the local coordinates.

        Parameters
        ----------
        coordinatesArray : numpy.ndarray
            The numpy array containing the coordinate of the new particles. Array is
            expected to take shape n*dim, where n is the number of new particles, and
            dim is the dimensionality of the swarm's supporting mesh.

        Returns
        --------
        npoints: int
            The number of points added to the local section of the swarm.
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

        return npoints

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

            with h5py.File(f"{filename[:-3]}.h5", "w", driver="mpio", comm=comm) as h5f:
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

        output_base_name = os.path.join(outputPath, filename) + "." + swarmname

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
        if self._particle_coordinates in writeable_vars:
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

                if self.em_swarm._particle_coordinates in writeable_vars:
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

                    # num_lost = np.where(cellid == -1)[0].shape[0]
                    # print(
                    #     f"{uw.mpi.rank} - EM 1: illegal_cells - {num_lost}", flush=True
                    # )

                    # if num_lost != 0:
                    #     print("LOST: ", coords[np.where(cellid == -1)])

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
                    if (self.em_swarm._particle_coordinates in writeable_vars) or (
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
            # self._nnmapdict[digest] = self._index.find_closest_point(meshvar_coords)[0]
            self._nnmapdict[digest] = self._index.query(
                meshvar_coords, k=1, sqr_dists=False
            )[0]
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
        step_limit=True,
    ):

        dt_limit = self.estimate_dt(V_fn)

        if step_limit and dt_limit is not None:
            substeps = int(max(1, round(abs(delta_t) / dt_limit)))
        else:
            substeps = 1

        if uw.mpi.rank == 0 and self.verbose:
            print(f"Substepping {substeps} / {abs(delta_t) / dt_limit}, {delta_t} ")

        # X0 holds the particle location at the start of advection
        # This is needed because the particles may be migrated off-proc
        # during timestepping.

        X0 = self._X0

        V_fn_matrix = self.mesh.vector.to_matrix(V_fn)

        # Use current velocity to estimate where the particles would have
        # landed in an implicit step. WE CANT DO THIS WITH SUB-STEPPING unless
        # We have a lot more information about the previous launch point / timestep
        # Also: how does this interact with the particle restoration function ?

        # if corrector == True and not self._X0_uninitialised:
        #     with self.access(self._particle_coordinates):
        #         v_at_Vpts = np.zeros_like(self.data)

        #         if evalf:
        #             for d in range(self.dim):
        #                 v_at_Vpts[:, d] = uw.function.evalf(
        #                     V_fn_matrix[d], self.data
        #                 ).reshape(-1)
        #         else:
        #             for d in range(self.dim):
        #                 v_at_Vpts[:, d] = uw.function.evaluate(
        #                     V_fn_matrix[d], self.data
        #                 ).reshape(-1)

        #         corrected_position = X0.data.copy() + delta_t * v_at_Vpts
        #         if restore_points_to_domain_func is not None:
        #             corrected_position = restore_points_to_domain_func(
        #                 corrected_position
        #             )

        #         updated_current_coords = 0.5 * (corrected_position + self.data.copy())

        #         # validate_coords to ensure they live within the domain (or there will be trouble)

        #         if restore_points_to_domain_func is not None:
        #             updated_current_coords = restore_points_to_domain_func(
        #                 updated_current_coords
        #             )

        #         self.data[...] = updated_current_coords[...]

        #         del updated_current_coords
        #         del v_at_Vpts

        print(f"{ uw.mpi.rank}: Peace", flush=True)

        # Wrap this whole thing in sub-stepping loop
        for step in range(0, substeps):

            with self.access(X0):
                X0.data[...] = self._particle_coordinates.data[...]

            # Mid point algorithm (2nd order)

            if order == 2:
                with self.access(self._particle_coordinates):
                    v_at_Vpts = np.zeros_like(self._particle_coordinates.data)

                    # if evalf:
                    #     for d in range(self.dim):
                    #         v_at_Vpts[:, d] = uw.function.evalf(
                    #             V_fn_matrix[d], self._particle_coordinates.data
                    #         ).reshape(-1)
                    # else:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d],
                            self._particle_coordinates.data,
                            evalf=evalf,
                        ).reshape(-1)

                    mid_pt_coords = (
                        self._particle_coordinates.data[...]
                        + 0.5 * delta_t * v_at_Vpts / substeps
                    )

                    # validate_coords to ensure they live within the domain (or there will be trouble)

                    if restore_points_to_domain_func is not None:
                        mid_pt_coords = restore_points_to_domain_func(mid_pt_coords)

                    self._particle_coordinates.data[...] = mid_pt_coords[...]

                    del mid_pt_coords

                    ## Let the swarm be updated, and then move the rest of the way

                    v_at_Vpts = np.zeros_like(self.data)

                    # if evalf:
                    #     for d in range(self.dim):
                    #         v_at_Vpts[:, d] = uw.function.evalf(
                    #             V_fn_matrix[d], self._particle_coordinates.data
                    #         ).reshape(-1)
                    # else:
                    #
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d],
                            self._particle_coordinates.data,
                            evalf=evalf,
                        ).reshape(-1)

                    # if (uw.mpi.rank == 0):
                    #     print("Re-launch from X0", flush=True)

                    new_coords = X0.data[...] + delta_t * v_at_Vpts / substeps

                    # validate_coords to ensure they live within the domain (or there will be trouble)
                    if restore_points_to_domain_func is not None:
                        new_coords = restore_points_to_domain_func(new_coords)

                    self._particle_coordinates.data[...] = new_coords[...]

                    del new_coords
                    del v_at_Vpts

            # forward Euler (1st order)
            else:
                with self.access(self._particle_coordinates):
                    v_at_Vpts = np.zeros_like(self.data)

                    # if evalf:
                    #     for d in range(self.dim):
                    #         v_at_Vpts[:, d] = uw.function.evalf(
                    #             V_fn_matrix[d], self.data
                    #         ).reshape(-1)
                    # else:
                    for d in range(self.dim):
                        v_at_Vpts[:, d] = uw.function.evaluate(
                            V_fn_matrix[d],
                            self.data,
                            evalf=evalf,
                        ).reshape(-1)

                    new_coords = self.data + delta_t * v_at_Vpts / substeps

                    # validate_coords to ensure they live within the domain (or there will be trouble)

                    if restore_points_to_domain_func is not None:
                        new_coords = restore_points_to_domain_func(new_coords)

                    self.data[...] = new_coords[...].copy()

        ## End of substepping loop

        ## Cycling of the swarm is a cheap and cheerful version of population control for particles. It turns the
        ## swarm into a streak-swarm where particles are Lagrangian for a number of steps and then reset to their
        ## original location.

        if self.recycle_rate > 1:
            # Restore particles which have cycle == cycle rate (use >= just in case)

            # Remove remesh points and recreate a new set at the mesh-local
            # locations that we already have stored.

            with self.access(self._particle_coordinates, self._remeshed):
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

    @timing.routine_timer_decorator
    def estimate_dt(self, V_fn):
        """
        Calculates an appropriate advective timestep for the given
        mesh and velocity configuration.
        """
        # we'll want to do this on an element by element basis
        # for more general mesh

        # first let's extract a max global velocity magnitude
        import math

        with self.access():
            vel = uw.function.evaluate(V_fn, self._particle_coordinates.data, evalf=True)
            try:
                magvel_squared = vel[:, 0] ** 2 + vel[:, 1] ** 2
                if self.mesh.dim == 3:
                    magvel_squared += vel[:, 2] ** 2

                max_magvel = math.sqrt(magvel_squared.max())

            except (ValueError, IndexError):
                max_magvel = 0.0

        from mpi4py import MPI

        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()

        # The assumption should be that we cross one or two elements (2-4 radii), not more,
        # in a single step (order 2, means one element per half-step or something
        # that we can broadly interpret that way)

        if max_magvel_glob != 0.0:
            return min_dx / max_magvel_glob
        else:
            return None


class NodalPointPICSwarm(PICSwarm):
    r"""Swarm with particles located at the coordinate points of a meshVariable

    The swarmVariable `X0` is defined so that the particles can "snap back" to their original locations
    after they have been moved.

    The purpose of this Swarm is to manage sample points for advection schemes based on upstream sampling
    (method of characteristics etc)"""

    def __init__(
        self,
        trackedVariable: uw.discretisation.MeshVariable,
        verbose=False,
    ):
        self.trackedVariable = trackedVariable
        self.swarmVariable = None

        mesh = trackedVariable.mesh

        # Set up a standard swarm
        super().__init__(mesh, verbose)

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
            _proxy=False,
            # proxy_degree=trackedVariable.degree,
            # proxy_continuous=trackedVariable.continuous,
            varsymbol=symbol,
        )

        # The launch point location
        name = f"ns_X0_{ks}"
        symbol = r"X0^{*^{{[" + ks + "]}}}"
        nX0 = uw.swarm.SwarmVariable(name, nswarm, nswarm.dim, _proxy=False)

        # The launch point index
        name = f"ns_I_{ks}"
        symbol = r"I^{*^{{[" + ks + "]}}}"
        nI0 = uw.swarm.SwarmVariable(name, nswarm, 1, dtype=int, _proxy=False)

        # The launch point processor rank
        name = f"ns_R0_{ks}"
        symbol = r"R0^{*^{{[" + ks + "]}}}"
        nR0 = uw.swarm.SwarmVariable(name, nswarm, 1, dtype=int, _proxy=False)

        nswarm.dm.finalizeFieldRegister()
        nswarm.dm.addNPoints(
            trackedVariable.coords.shape[0] + 1
        )  # why + 1 ? That's the number of spots actually allocated

        cellid = nswarm.dm.getField("DMSwarm_cellid")
        coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape((-1, nswarm.dim))
        coords[...] = trackedVariable.coords[...]
        cellid[:] = self.mesh.get_closest_local_cells(coords)

        # Move slightly within the chosen cell to avoid edge effects
        centroid_coords = self.mesh._centroids[cellid]

        shift = 0.001
        coords[:, :] = (1.0 - shift) * coords[:, :] + shift * centroid_coords[:, :]

        nswarm.dm.restoreField("DMSwarmPIC_coor")
        nswarm.dm.restoreField("DMSwarm_cellid")

        nswarm.dm.migrate(remove_sent_points=True)

        with nswarm.access(nX0, nI0):
            nX0.data[:, :] = coords
            nI0.data[:, 0] = range(0, coords.shape[0])

        self._nswarm = nswarm
        self._nX0 = nX0
        self._nI0 = nI0
        self._nR0 = nR0

        return

    @timing.routine_timer_decorator
    def advection(
        self,
        V_fn,
        delta_t,
        order=2,
        corrector=False,
        restore_points_to_domain_func=None,
        evalf=False,
        step_limit=True,
    ):

        with self.access(self._X0):
            self._X0.data[...] = self._nX0.data[...]

        with self.access(self._nR0):
            self._nR0.data[...] = uw.mpi.rank

        super().advection(
            V_fn,
            delta_t,
            order,
            corrector,
            restore_points_to_domain_func,
            evalf,
            step_limit,
        )

        return