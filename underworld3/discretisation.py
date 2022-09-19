# from telnetlib import DM
# from this import d
from typing import Optional, Tuple, Union
import os
from xmlrpc.client import Boolean
import numpy
import sympy
import sympy.vector
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities import _api_tools

from underworld3.coordinates import CoordinateSystem
import underworld3.timing as timing
import weakref


@PETSc.Log.EventDecorator()
def _from_gmsh(filename, comm=None):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    comm = comm or PETSc.COMM_WORLD
    # Create a read-only PETSc.Viewer
    gmsh_viewer = PETSc.Viewer().create(comm=comm)
    gmsh_viewer.setType("ascii")
    gmsh_viewer.setFileMode("r")
    gmsh_viewer.setFileName(filename)
    gmsh_plex = PETSc.DMPlex().createGmsh(gmsh_viewer, comm=comm)

    # Extract Physical groups from the gmsh file
    import gmsh

    gmsh.initialize()
    gmsh.model.add("Model")
    gmsh.open(filename)

    physical_groups = {}
    for dim, tag in gmsh.model.get_physical_groups():

        name = gmsh.model.get_physical_name(dim, tag)

        physical_groups[name] = tag
        gmsh_plex.createLabel(name)
        label = gmsh_plex.getLabel(name)

        for elem in ["Face Sets"]:
            indexSet = gmsh_plex.getStratumIS(elem, tag)
            if indexSet:
                label.insertIS(indexSet, 1)
            indexSet.destroy()

    gmsh.finalize()

    return gmsh_plex


class Mesh(_api_tools.Stateful):

    mesh_instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self, plex_or_meshfile, degree=1, simplex=True, coordinate_system_type=None, qdegree=2, *args, **kwargs
    ):

        if isinstance(plex_or_meshfile, PETSc.DMPlex):
            name = "plexmesh"
            self.dm = plex_or_meshfile
        else:
            comm = kwargs.get("comm", PETSc.COMM_WORLD)
            name = plex_or_meshfile
            basename, ext = os.path.splitext(plex_or_meshfile)

            if ext.lower() == ".msh":
                self.dm = _from_gmsh(plex_or_meshfile, comm)

            else:
                raise RuntimeError("Mesh file %s has unknown format '%s'." % (plex_or_meshfile, ext[1:]))

        Mesh.mesh_instances += 1

        # Set sympy constructs. First a generic, symbolic, Cartesian coordinate system
        from sympy.vector import CoordSys3D

        self._N = CoordSys3D("N")

        # Now add the appropriate coordinate system for the mesh's natural geometry
        self._CoordinateSystem = CoordinateSystem(self, coordinate_system_type)

        # Tidy some of this printing. Note that this
        # only changes the user interface printing, and
        # switches out for simpler `BaseScalar` representations.
        # For example, we now have "x" instead of "x_N". This makes
        # Jupyter rendered Latex easier to digest, but depending on
        # how we end up using Sympy coordinate systems it may be
        # desirable to bring back the more verbose version.

        self._N.x._latex_form = r"\mathrm{x}"
        self._N.y._latex_form = r"\mathrm{y}"
        self._N.z._latex_form = r"\mathrm{z}"
        self._N.i._latex_form = r"\mathbf{\hat{i}}"
        self._N.j._latex_form = r"\mathbf{\hat{j}}"
        self._N.k._latex_form = r"\mathbf{\hat{k}}"

        try:
            self.isSimplex = self.dm.isSimplex()
        except:
            self.isSimplex = simplex

        # Use grid hashing for point location
        options = PETSc.Options()
        options["dm_plex_hash_location"] = None
        self.dm.setFromOptions()

        self._vars = weakref.WeakValueDictionary()

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

        # A private work array used in the stats routines.
        # This is defined now since we cannot make a new one
        # once the init phase of uw3 is complete.

        self._work_MeshVar = MeshVariable("work_array_1", self, 1, degree=2)

        # This looks a bit strange, but we'd like to
        # put these mesh-dependent vector calculus functions
        # and mesh-based tensor manipulation routines
        # in a bundle to avoid the mesh being required as an argument
        # since this could lead to things going out of sync

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

    def nuke_coords_and_rebuild(self):

        # This is a reversion to the old version (3.15 compatible which seems to work in 3.16 too)

        self._coord_array = {}

        # let's go ahead and do an initial projection from linear (the default)
        # to linear. this really is a nothing operation, but a
        # side effect of this operation is that coordinate DM DMField is
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates.

        ## LM  - I put in the option to specify the default coordinate interpolation degree
        ## LM  - which seems sensible given linear interpolation seems likely to be a problem
        ## LM  - for spherical meshes. However, I am not sure about this because it means that
        ## LM  - the mesh coords and the size of the nodal array are different. This might break
        ## LM  - stuff so I will leave the default at 1

        options = PETSc.Options()
        options.setValue("meshproj_{}_petscspace_degree".format(self.mesh_instances), self.degree)

        self.petsc_fe = PETSc.FE().createDefault(
            self.dim,
            self.cdim,
            self.isSimplex,
            self.qdegree,
            "meshproj_{}_".format(self.mesh_instances),
            PETSc.COMM_WORLD,
        )

        self.dm.clearDS()
        self.dm.createDS()

        # This should replace the cython code
        self.dm.projectCoordinates(self.petsc_fe)

        # cdmfe = self.petsc_fe
        # cdef FE c_fe = cdmfe
        # cdef DM c_dm = self.dm
        # ierr = DMProjectCoordinates( c_dm.dm, c_fe.fe ); CHKERRQ(ierr)

        # now set copy of this array into dictionary

        arr = self.dm.getCoordinatesLocal().array

        key = (self.isSimplex, self.degree, True)  # Assumes continuous basis for coordinates

        self._coord_array[key] = arr.reshape(-1, self.cdim).copy()
        self._get_mesh_centroids()

        # invalidate the cell-search k-d tree and the mesh centroid data
        self._index = None

        return

    def build_quadratures(self, degree):

        return

    @timing.routine_timer_decorator
    def _align_quadratures(self, mesh_var=None, force=False):
        """
        Choose a quadrature that will be used by any of our solvers on
        this mesh. Quadratures are aligned with either:
          - the variable that has the highest degree on the mesh at this point
          - the variable that is provided

        The default quadrature is only updated once unless
        we set `force=True` which might be needed if new variables have
        been added

        """

        # # Find var with the highest degree. We will then configure the integration
        # # to use this variable's quadrature object for all variables.
        # # This needs to be double checked.

        return

        if self._quadrature and not force:
            return

        if mesh_var is None:
            deg = 0
            for key, var in self.vars.items():
                if var.degree >= deg:
                    deg = var.degree
                    var_base = var
        else:
            var = mesh_var

        quad_base = var_base.petsc_fe.getQuadrature()
        self.petsc_fe.setQuadrature(quad_base)

        # Do this now for consistency (it is also done by the solvers)
        for fe in [var.petsc_fe for var in self.vars.values()]:
            fe.setQuadrature(quad_base)

        self._quadrature = True
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
            raise RuntimeError("Mesh `lvec` needs to be updated using the update_lvec()` method.")
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
        r_vec = N.x * N.i + N.y * N.j
        if self.cdim == 3:
            r_vec += N.z * N.k
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
        from underworld3.utilities.petsc_gen_xdmf import generateXdmf

        generateXdmf(filename)

    @property
    def vars(self):
        """
        A list of variables recorded on the mesh.
        """
        return self._vars

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

        # otherwise create and return
        """
        cdmOld = self.dm.getCoordinateDM()
        cdmNew = cdmOld.clone()
        options = PETSc.Options()
        options.setValue("coordinterp_petscspace_degree", var.degree) 
        cdmfe = PETSc.FE().createDefault(self.dim, self.cdim, self.isSimplex, self.qdegree, "coordinterp_", PETSc.COMM_WORLD)
        cdmNew.setField(0,cdmfe)
        cdmNew.createDS()
        (matInterp, vecScale) = cdmOld.createInterpolation(cdmNew)
        vecScale.destroy() # not needed
        coordsOld = self.dm.getCoordinates()
        coordsNewG = cdmNew.getGlobalVec()
        coordsNewL = cdmNew.getLocalVec()

        cdef Mat c_matInterp = matInterp
        cdef Vec c_coordsOld = coordsOld
        cdef Vec c_coordsNewG = coordsNewG
        ierr = MatInterpolate(c_matInterp.mat, c_coordsOld.vec, c_coordsNewG.vec); CHKERRQ(ierr)
        cdmNew.globalToLocal(coordsNewG,coordsNewL)
        """

        ## Cython-free version of the code above

        dmold = self.dm.getCoordinateDM()
        dmnew = dmold.clone()
        options = PETSc.Options()

        options["coordinterp_petscspace_degree"] = var.degree
        options["coordinterp_petscdualspace_lagrange_continuity"] = var.continuous
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
        # reshape and grab copy
        arrcopy = arr.reshape(-1, self.cdim).copy()
        # record into coord array
        self._coord_array[key] = arrcopy

        # clean up (if cython)
        # cdmNew.restoreLocalVec(coordsNewL)
        # cdmNew.restoreGlobalVec(coordsNewG)
        # cdmNew.destroy()
        # cdmfe.destroy()

        return arrcopy

    @timing.routine_timer_decorator
    def get_closest_cells(self, coords: numpy.ndarray) -> numpy.ndarray:
        """
        This method uses a kd-tree algorithm to find the closest
        cells to the provided coords. For a regular mesh, this should
        be exactly the owning cell, but if the mesh is deformed, this
        is not guaranteed.

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
        if not self._index:
            from underworld3.swarm import Swarm, SwarmPICLayout

            # Create a temp swarm which we'll use to populate particles
            # at gauss points. These will then be used as basis for
            # kd-tree indexing back to owning cells.
            tempSwarm = Swarm(self)
            # 4^dim pop is used. This number may need to be considered
            # more carefully, or possibly should be coded to be set dynamically.
            tempSwarm.populate(fill_param=4, layout=SwarmPICLayout.GAUSS)
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
                self._indexMap = numpy.array(tempSwarm.particle_cellid.data[:, 0], dtype=numpy.int64)

        closest_points, dist, found = self._index.find_closest_point(coords)

        if not numpy.allclose(found, True):
            raise RuntimeError(
                "An error was encountered attempting to find the closest cells to the provided coordinates."
            )

        return self._indexMap[closest_points]

    def _get_mesh_centroids(self):
        """
        Obtain and cache the mesh centroids using underworld swarm technology.
        This routine is called when the mesh is built / rebuilt
        """

        from underworld3.swarm import Swarm, SwarmPICLayout

        tempSwarm = Swarm(self)
        tempSwarm.populate(fill_param=1, layout=SwarmPICLayout.GAUSS)

        with tempSwarm.access():
            # Build index on particle coords
            self._centroids = tempSwarm.data.copy()

        # That's it ! we should check that these objects are deleted correctly

        return

    def get_min_radius(self) -> float:
        """
        This method returns the minimum distance from any cell centroid to a face.
        It wraps to the PETSc `DMPlexGetMinRadius` routine.
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

    def stats(self, uw_function):
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

        from petsc4py.PETSc import NormType

        tmp = self._work_MeshVar

        with self.access(tmp):
            tmp.data[...] = uw.function.evaluate(uw_function, tmp.coords).reshape(-1, 1)

        vsize = self._work_MeshVar._gvec.getSize()
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
    num_components: int,
    vtype: Optional["underworld.VarType"] = None,
    degree: int = 1,
    continuous: bool = True,
):

    """
    The MeshVariable class generates a variable supported by a finite element mesh and the
    underlying sympy representation that makes it possible to construct expressions that
    depend on the values of the MeshVariable.

    To set / read nodal values, use the numpy interface via the 'data' property.

    Parameters
    ----------
    name :
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

    """

    if isinstance(varname, list):
        name = varname[0] + R"+ \dots"
    else:
        name = varname

    if mesh._accessed:
        print("It is not possible to add new variables to a mesh after existing variables have been accessed")
        print("Variable {name} has NOT been added")
        return

    ## Smash if already defined (we should check this BEFORE the old meshVariable object is destroyed)

    if name in mesh.vars.keys():
        print(f"Variable with name {name} already exists on the mesh - Skipping.")
        return mesh.vars[name]

    return _MeshVariable(varname, mesh, num_components, vtype, degree, continuous)


class _MeshVariable(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(
        self,
        varname: Union[str, list],
        mesh: "underworld.mesh.Mesh",
        num_components: int,
        vtype: Optional["underworld.VarType"] = None,
        degree: int = 1,
        continuous: bool = True,
    ):
        """
        The MeshVariable class generates a variable supported by a finite element mesh and the
        underlying sympy representation that makes it possible to construct expressions that
        depend on the values of the MeshVariable.

        To set / read nodal values, use the numpy interface via the 'data' property.

        Parameters
        ----------
        name :
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

        """

        if isinstance(varname, list):
            name = varname[0] + R"+ \dots"
        else:
            name = varname

        self._lvec = None
        self._gvec = None
        self._data = None
        self._is_accessed = False
        self._available = False

        self.name = name

        if vtype == None:
            if num_components == 1:
                vtype = uw.VarType.SCALAR
            elif num_components == mesh.dim:
                vtype = uw.VarType.VECTOR
            else:
                raise ValueError(
                    "Unable to infer variable type from `num_components`. Please explicitly set the `vtype` parameter."
                )

        if not isinstance(vtype, uw.VarType):
            raise ValueError("'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`.")

        self.vtype = vtype
        self.mesh = mesh
        self.num_components = num_components
        self.degree = degree
        self.continuous = continuous

        options = PETSc.Options()
        options.setValue(f"{name}_petscspace_degree", degree)
        options.setValue(f"{name}_petscdualspace_lagrange_continuity", continuous)

        self.petsc_fe = PETSc.FE().createDefault(
            self.mesh.dm.getDimension(),
            num_components,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            name + "_",
            PETSc.COMM_WORLD,
        )

        self.field_id = self.mesh.dm.getNumFields()
        self.mesh.dm.setField(self.field_id, self.petsc_fe)

        # create associated sympy function
        from underworld3.function import UnderworldFunction

        if vtype == uw.VarType.SCALAR:
            self._sym = sympy.Matrix.zeros(1, 1)
            self._sym[0] = UnderworldFunction(name, self, vtype)(*self.mesh.r)
            self._ijk = self._sym[0]

        elif vtype == uw.VarType.VECTOR:
            self._sym = sympy.Matrix.zeros(1, num_components)

            # Matrix form (any number of components)
            for comp in range(num_components):
                self._sym[0, comp] = UnderworldFunction(name, self, vtype, comp)(*self.mesh.r)

            # Spatial vector form (2 vectors and 3 vectors according to mesh dim)
            if num_components == mesh.dim:
                self._ijk = sympy.vector.matrix_to_vector(self._sym, self.mesh.N)
                # self.mesh.vector.to_vector(self._sym)

        elif vtype == uw.VarType.COMPOSITE:  # This is just to allow full control over the names of the components
            self._sym = sympy.Matrix.zeros(1, num_components)
            if isinstance(varname, list):
                if len(varname) == num_components:
                    for comp in range(num_components):
                        self._sym[0, comp] = UnderworldFunction(varname[comp], self, vtype, comp)(*self.mesh.r)
                else:
                    raise RuntimeError("Please supply a list of names for all components of this vector")
            else:
                for comp in range(num_components):
                    self._sym[0, comp] = UnderworldFunction(name, self, vtype, comp)(*self.mesh.r)

        super().__init__()

        self.mesh.vars[name] = self

        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        return

    @timing.routine_timer_decorator
    def save(self, filename: str, name: Optional[str] = None, index: Optional[int] = None):
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

    @property
    def fn(self) -> sympy.Basic:
        """
        The handle to the function view of this variable.
        """
        return self._ijk

    @property
    def ijk(self) -> sympy.Basic:
        """
        The handle to the scalar / vector view of this variable.
        """
        return self._ijk

    @property
    def sym(self) -> sympy.Basic:
        """
        The handle to the tensor view of this variable.
        """
        return self._sym

    # @property
    # def f(self) -> sympy.Basic:
    #     """
    #     The handle to the tensor view of this variable.
    #     """
    #    return self._f

    def _set_vec(self, available):
        # cdef DM subdm = PETSc.DM()
        # cdef DM dm = self.mesh.dm
        # cdef PetscInt fields = self.field_id
        # if self._lvec==None:
        # Create a subdm for this variable.
        # This allows us to generate a local vector.
        #    ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)

        # This should achieve the same effect as the cython code above

        if self._lvec == None:
            indexset, subdm = self.mesh.dm.createSubDM(self.field_id)

            self._lvec = subdm.createLocalVector()
            self._lvec.zeroEntries()  # not sure if required, but to be sure.
            self._gvec = subdm.createGlobalVector()
            self._gvec.setName(self.name)  # This is set for checkpointing.
            self._gvec.zeroEntries()

        #    ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)

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
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")
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
            raise RuntimeError("Data must be accessed via the mesh `access()` context manager.")
        return self._data

    def min(self) -> Union[float, tuple]:
        """
        The global variable minimum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.min()
        else:
            return tuple([self._gvec.strideMin(i)[1] for i in range(self.num_components)])

    def max(self) -> Union[float, tuple]:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.max()
        else:
            return tuple([self._gvec.strideMax(i)[1] for i in range(self.num_components)])

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
            return tuple([self._gvec.strideNorm(i, norm_type) for i in range(self.num_components)])

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
            return tuple([self._gvec.strideSum(i) / vecsize for i in range(self.num_components)])

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
            raise NotImplementedError("stats not available for multi-component variables")

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
