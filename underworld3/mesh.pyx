# cython: profile=False

from libc.stdlib cimport malloc, free
from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, FE, PetscFE, Vec, PetscVec, IS, PetscIS, PetscSF, MPI_Comm, PetscObject, Mat, PetscMat, GetCommDefault
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import contextlib
import numpy as np
cimport numpy as np
import sympy
import underworld3 as uw 
from underworld3 import _api_tools
from mpi4py import MPI
import underworld3.timing as timing

ctypedef enum PetscBool:
    PETSC_FALSE
    PETSC_TRUE

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, PetscDM*)
    PetscErrorCode DMPlexComputeGeometryFVM( PetscDM dm, PetscVec *cellgeom, PetscVec *facegeom)
    PetscErrorCode DMPlexGetMinRadius(PetscDM dm, PetscReal *minradius)
    PetscErrorCode VecDestroy(PetscVec *v)
    PetscErrorCode DMDestroy(PetscDM *dm)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMProjectCoordinates(PetscDM dm, PetscFE disc)
    PetscErrorCode MatInterpolate(PetscMat A, PetscVec x, PetscVec y)
    PetscErrorCode DMCompositeGetLocalISs(PetscDM dm,PetscIS **isets)
    PetscErrorCode DMPlexExtrude(PetscDM idm, PetscInt layers, PetscReal height, PetscBool orderHeight, const PetscReal extNormal[], PetscBool interpolate, PetscDM* dm)
    MPI_Comm MPI_COMM_SELF

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")


class MeshVariable(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, name, mesh, num_components, vtype=None, degree=1):

        if mesh._accessed:
            raise RuntimeError("It is not possible to add new variables to a mesh after existing variables have been accessed.")
        if name in mesh.vars.keys():
            raise ValueError("Variable with name {} already exists on mesh.".format(name))
        self.name = name

        if vtype==None:
            if   num_components==1:
                vtype=uw.VarType.SCALAR
            elif num_components==mesh.dim:
                vtype=uw.VarType.VECTOR
            else:
                raise ValueError("Unable to infer variable type from `num_components`. Please explicitly set the `vtype` parameter.")
        if not isinstance(vtype, uw.VarType):
            raise ValueError("'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`.")
        self.vtype = vtype
        self.mesh = mesh
        self.num_components = num_components

        options = PETSc.Options()
        options.setValue(f"{name}_petscspace_degree", degree)
        self.degree = degree

        self.petsc_fe = PETSc.FE().createDefault(self.mesh.dm.getDimension(), num_components, self.mesh.isSimplex, PETSc.DEFAULT, name+"_", PETSc.COMM_WORLD)

        self.field_id = self.mesh.dm.getNumFields()
        self.mesh.dm.setField(self.field_id,self.petsc_fe)

        # create associated sympy function
        from underworld3.function import UnderworldFunction
        if   vtype==uw.VarType.SCALAR:
            self._fn = UnderworldFunction(self,0,name)(*self.mesh.r)
        elif vtype==uw.VarType.VECTOR:
            if num_components!=mesh.dim:
                raise ValueError("For 'VarType.VECTOR' types 'num_components' must equal 'mesh.dim'.")
            from sympy.vector import VectorZero
            self._fn = VectorZero()
            subnames = ["_x","_y","_z"]
            for comp in range(num_components):
                subfn = UnderworldFunction(self,comp,name+subnames[comp])(*self.mesh.r)
                self._fn += subfn*self.mesh.N.base_vectors()[comp]
        super().__init__()

        self.mesh.vars[name] = self

        self._lvec = None
        self._gvec = None
        self._data = None
        self._is_accessed = False

    # def save(self, filename):
    #     viewer = PETSc.Viewer().createHDF5(filename, "w")
    #     viewer(self.petsc_fe)
    #     generateXdmf(filename)


    @property
    def fn(self):
        return self._fn

    def _set_vec(self, available):
        cdef DM subdm = PETSc.DM()
        cdef DM dm = self.mesh.dm
        cdef PetscInt fields = self.field_id
        if self._lvec==None:
            # Create a subdm for this variable.
            # This allows us to generate a local vectors.
            ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)
            self._lvec  = subdm.createLocalVector()
            self._lvec.zeroEntries()  # not sure if required, but to be sure. 
            self._gvec  = subdm.createGlobalVector()
            self._lvec.zeroEntries()
            ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)
        self._available = available

    def __del__(self):
        if self._lvec:
            self._lvec.destroy()
        if self._gvec:
            self._gvec.destroy()

    @property
    def vec(self):
        if not self._available:
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")
        return self._lvec

    @property
    def vec_global(self):
        if not self._available:
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")
        return self._gvec

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError("Data must be accessed via the mesh `access()` context manager.")
        return self._data

    def min(self):
        """
        Global min.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set as of yet.")
        return self._gvec.min()

    def max(self):
        """
        Global max.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set as of yet.")
        return self._gvec.max()

    @property
    def coords(self):
        """
        Returns the array of variable vertex coordinates. 
        """
        return self.mesh._get_coords_for_var(self)


class _MeshBase(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, simplex, *args,**kwargs):
        self.isSimplex = simplex
        # create boundary sets
        for val in self.boundary:
            boundary_set = self.dm.getStratumIS("marker",val.value)        # get the set
            self.dm.createLabel(str(val).encode('utf8'))               # create the label
            boundary_label = self.dm.getLabel(str(val).encode('utf8')) # get label
            if boundary_set:
                boundary_label.insertIS(boundary_set, 1) # add set to label with value 1


        # set sympy constructs
        from sympy.vector import CoordSys3D
        self._N = CoordSys3D("N")
        self._r = self._N.base_scalars()[0:self.dim]

        # dictionary for variables
        import weakref
        self._vars = weakref.WeakValueDictionary()

        # let's calculate the minradius (min cell size)
        cdef PetscVec cellgeom = NULL
        cdef PetscVec facegeom = NULL
        cdef DM dm = self.dm
        DMPlexComputeGeometryFVM(dm.dm,&cellgeom,&facegeom)
        cdef double minradius
        DMPlexGetMinRadius(dm.dm,&minradius)
        self.min_radius = minradius
        VecDestroy(&cellgeom)
        VecDestroy(&facegeom)

        self._accessed = False
        self._stale_lvec = True
        self._lvec = None

        # dictionary for variable coordinate arrays
        self._coord_array = {}
        # let's go ahead and do an initial projection from linear (the default) 
        # to linear. this really is a nothing operation, but a 
        # side effect of this operation is that coordinate DM DMField is 
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates. 
        options = PETSc.Options()
        options.setValue("meshproj_petscspace_degree", 1) 
        cdmfe = PETSc.FE().createDefault(self.dim, self.dim, self.isSimplex, 1,"meshproj_", PETSc.COMM_WORLD)
        cdef FE c_fe = cdmfe
        cdef DM c_dm = self.dm
        ierr = DMProjectCoordinates( c_dm.dm, c_fe.fe ); CHKERRQ(ierr)
        # now set copy of linear array into dictionary
        arr = self.dm.getCoordinatesLocal().array
        self._coord_array[(self.isSimplex,1)] = arr.reshape(-1, self.dim).copy()

        super().__init__()

    @timing.routine_timer_decorator
    def update_lvec(self):
        """
        This method creates and/or updates the mesh variable local vector. 
        If the local vector is already up to date, this method will do nothing.
        """
        cdef DM dm = self.dm
        if self._stale_lvec:
            if not self._lvec:
                self.dm.clearDS()
                self.dm.createDS()
                # create the local vector (memory chunk) and attach to original dm
                self._lvec = self.dm.createLocalVec()
            # push avar arrays into the parent dm array
            a_global = self.dm.getGlobalVec()
            names, isets, dms = self.dm.createFieldDecomposition()
            with self.access():
                # traverse subdms, taking user generated data in the subdm
                # local vec, pushing it into a global sub vec 
                for var,subiset,subdm in zip(self.vars.values(),isets,dms):
                    lvec = var.vec
                    subvec = a_global.getSubVector(subiset)
                    subdm.localToGlobal(lvec,subvec, addv=False)
                    a_global.restoreSubVector(subiset,subvec)

            self.dm.globalToLocal(a_global,self._lvec)
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
        if self._lvec:
            self._lvec.destroy()

    def access(self, *writeable_vars:MeshVariable):
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
        >>> someMesh = uw.mesh.FeMesh_Cartesian()
        >>> with someMesh.deform_mesh():
        ...     someMesh.data[0] = [0.1,0.1]
        >>> someMesh.data[0]
        array([ 0.1,  0.1])
        """

        import time
        uw.timing._incrementDepth()
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
            var._data = var.vec.array.reshape( -1, var.num_components )
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()

        class exit_manager:
            def __init__(self,mesh): self.mesh = mesh
            def __enter__(self): pass
            def __exit__(self,*args):
                cdef DM subdm = PETSc.DM()
                cdef DM dm = self.mesh.dm
                cdef PetscInt fields
                cdef Vec vec = PETSc.Vec()
                for var in self.mesh.vars.values():
                    # only de-access variables we have set access for.
                    if var not in deaccess_list:
                        continue
                    # set this back, although possibly not required.
                    if var not in writeable_vars:
                        var._data.flags.writeable = var._old_data_flag
                    # perform sync for any modified vars.
                    if var in writeable_vars:
                        fields = var.field_id
                        ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)
                        subdm.localToGlobal(var.vec,var.vec_global, addv=False)
                        subdm.globalToLocal(var.vec_global,var.vec, addv=False)
                        ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)
                        self.mesh._stale_lvec = True
                    var._data = None
                    var._set_vec(available=False)
                    var._is_accessed = False
                uw.timing._decrementDepth()
                uw.timing.log_result(time.time()-stime, "Mesh.access",1)
        return exit_manager(self)


    @property
    def N(self):
        return self._N

    @property
    def r(self):
        return self._r

    @property
    def data(self):
        # get flat array
        arr = self.dm.getCoordinatesLocal().array
        return arr.reshape(-1, self.dim)

    @property
    def dim(self):
        """ Number of dimensions of the mesh """
        return self.dm.getDimension()

    @timing.routine_timer_decorator
    def save(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, "w")
        viewer(self.dm)
        generateXdmf(filename)

    # def add_mesh_variable(self):
    #     return

    @property
    def vars(self):
        return self._vars

    def _get_coords_for_var(self, var):
        """
        This function returns the vertex array for the 
        provided variable. If the array does not already exist, 
        it is first created and then returned.
        """
        key = (self.isSimplex,var.degree) 
        # if array already created, return. 
        if key in self._coord_array:
            return self._coord_array[key]
        # otherwise create and return
        cdmOld = self.dm.getCoordinateDM()
        cdmNew = cdmOld.clone()
        options = PETSc.Options()
        options.setValue("coordinterp_petscspace_degree", var.degree) 
        cdmfe = PETSc.FE().createDefault(self.dim, self.dim, self.isSimplex, var.degree, "coordinterp_", PETSc.COMM_WORLD)
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
        arr = coordsNewL.array
        # reshape and grab copy
        arrcopy = arr.reshape(-1,self.dim).copy()
        # record into coord array
        self._coord_array[key] = arrcopy
        # clean up
        cdmNew.restoreLocalVec(coordsNewL)
        cdmNew.restoreGlobalVec(coordsNewG)
        cdmNew.destroy()
        cdmfe.destroy()
        # return
        return arrcopy

class Mesh(_MeshBase):
    @timing.routine_timer_decorator
    def __init__(self, 
                elementRes=(16, 16), 
                minCoords=None,
                maxCoords=None,
                simplex=False,
                interpolate=False):

        options = PETSc.Options()
        options["dm_plex_separate_marker"] = None
        if "dm_plex_hash_location" in options: del options["dm_plex_hash_location"]
        if "dm_plex_hash_box_nijk" in options: del options["dm_plex_hash_box_nijk"]
        if len(elementRes)==2:
            options["dm_plex_hash_location"] = None
            options["dm_plex_hash_box_nijk"] = max(elementRes)
        self.elementRes = elementRes
        if minCoords==None : minCoords=len(elementRes)*(0.,)
        self.minCoords = minCoords
        if maxCoords==None : maxCoords=len(elementRes)*(1.,)
        self.maxCoords = maxCoords
        self.dm = PETSc.DMPlex().createBoxMesh(
            elementRes, 
            lower=minCoords, 
            upper=maxCoords,
            simplex=simplex)
        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        # bcs
        from enum import Enum        
        if len(elementRes) == 2:
            class Boundary2D(Enum):
                BOTTOM = 1
                RIGHT  = 2
                TOP    = 3
                LEFT   = 4
            self.boundary = Boundary2D
        else:
            class Boundary3D(Enum):
                BOTTOM = 1
                TOP    = 2
                FRONT  = 3
                BACK   = 4
                RIGHT  = 5
                LEFT   = 6
            self.boundary = Boundary3D

        # self.dm.view()

        super().__init__(simplex=simplex)


class Spherical(_MeshBase):
    @timing.routine_timer_decorator
    def __init__(self, 
                refinements=4, 
                radius=1.):

        self.refinements = refinements
        self.radius = radius

        options = PETSc.Options()
        options.setValue("bd_dm_refine", self.refinements)

        cdef DM dm = PETSc.DMPlex()
        cdef MPI_Comm ccomm = GetCommDefault()
        cdef PetscInt cdim = 3
        cdef PetscReal cradius = self.radius
        DMPlexCreateBallMesh(ccomm, cdim, cradius, &dm.dm)
        self.dm = dm


        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        from enum import Enum        
        class Boundary(Enum):
            OUTER = 1
        
        self.boundary = Boundary

        self.dm.view()        
        
        super().__init__(simplex=True)
            

class CubedSphere(_MeshBase):
    @timing.routine_timer_decorator
    def __init__(self, 
                refinements=1, 
                inner_radius=0.5, 
                outer_radius=1.,
                nlayers=4):

        self.refinements = refinements
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        options = PETSc.Options()
        options.setValue("bd_dm_refine", self.refinements)

        from . import mesh_utils
        cells, coords = mesh_utils._cubedsphere_cells_and_coords(inner_radius, refinements)
        from mpi4py import MPI
        cdef DM cubedsphere_dm = mesh_utils._from_cell_list(2, cells, coords, MPI.COMM_WORLD)

        cdef DM dm = PETSc.DMPlex()
        ierr = DMPlexExtrude(cubedsphere_dm.dm, nlayers, -(inner_radius-outer_radius), PETSC_FALSE, NULL, PETSC_TRUE, &dm.dm); CHKERRQ(ierr)
        self.dm = dm
        self.dm.markBoundaryFaces("1",1)

        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        from enum import Enum        
        class Boundary(Enum):
            OUTER = 1
            INNER = 2
        
        self.boundary = Boundary
        
        self.dm.view()        
        
        super().__init__(simplex=False)

class ExtrudeBox(_MeshBase):
    @timing.routine_timer_decorator
    def __init__(self, 
                elementRes=(4, 4, 4), 
                minCoords=None,
                maxCoords=None):

        # options = PETSc.Options()
        # options["dm_plex_separate_marker"] = None
        # options["dm_plex_hash_location"] = None
        self.elementRes = elementRes
        if minCoords==None : minCoords=len(elementRes)*(0.,)
        self.minCoords = minCoords
        if maxCoords==None : maxCoords=len(elementRes)*(1.,)
        self.maxCoords = maxCoords
        self.dm_2d = PETSc.DMPlex().createBoxMesh(
            elementRes[0:2], 
            lower=minCoords[0:2], 
            upper=maxCoords[0:2],
            simplex=False)

        cdef DM dm_2d_c = self.dm_2d
        cdef DM dm = PETSc.DMPlex()
        ierr = DMPlexExtrude(dm_2d_c.dm, elementRes[2], maxCoords[2]-minCoords[2], PETSC_FALSE, NULL, PETSC_TRUE, &dm.dm); CHKERRQ(ierr)
        self.dm = dm

        # for ind,val in enumerate(self.boundary):
        #     boundary_set = self.dm.getStratumIS("marker",ind+1)        # get the set
        #     self.dm.createLabel(str(val).encode('utf8'))               # create the label
        #     boundary_label = self.dm.getLabel(str(val).encode('utf8')) # get label
        #     if boundary_set:
        #         boundary_label.insertIS(boundary_set, 1) # add set to label with value 1

        self.dm.markBoundaryFaces("marker",1)

        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        from enum import Enum        
        class Boundary(Enum):
            OUTER = 1
        
        self.boundary = Boundary
        
        self.dm.view()        
        
        super().__init__(simplex=False)


class CylinderMesh(_MeshBase):
    @timing.routine_timer_decorator
    def __init__(self, 
                refinements=4, 
                inner_radius=0.5, outer_radius=1., ncells=16, nlayers=4):

        self.refinements = refinements
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        options = PETSc.Options()
        # options.setValue("bd_dm_refine", self.refinements)

        """Generated a 1D mesh of the circle, immersed in 2D.

        :arg ncells: number of cells the circle should be
            divided into (min 3)
        :kwarg radius: (optional) radius of the circle to approximate
            (defaults to 1).
        :kwarg comm: Optional communicator to build the mesh on (defaults to
            COMM_WORLD).
        """
        if ncells < 3:
            raise ValueError("CircleManifoldMesh must have at least three cells")

        vertices = inner_radius*np.column_stack((np.cos(np.arange(ncells, dtype=np.double)*(2*np.pi/ncells)),
                                        np.sin(np.arange(ncells, dtype=np.double)*(2*np.pi/ncells))))

        cells = np.column_stack((np.arange(0, ncells, dtype=np.int32),
                                np.roll(np.arange(0, ncells, dtype=np.int32), -1)))

        from mpi4py import MPI
        from . import mesh_utils
        # cdef DM circle_dm = mesh_utils._from_cell_list(1, cells, vertices, MPI.COMM_WORLD)

        cdef DM circle_dm = PETSc.DMPlex().createBoxMesh(
            (2,)*2, 
            lower=(0.,)*2, 
            upper=(1.,)*2,
            simplex=True, interpolate=True)

        cdef DM dm = PETSc.DMPlex()
        ierr = DMPlexExtrude(circle_dm.dm, nlayers, -(inner_radius-outer_radius), PETSC_TRUE, NULL, PETSC_TRUE, &dm.dm); CHKERRQ(ierr)
        self.dm = dm

        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        from enum import Enum        
        class Boundary(Enum):
            OUTER = 1
        
        self.boundary = Boundary
        
        self.dm.view()        
        
        super().__init__(simplex=False)
