from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, FE, PetscFE, Vec, PetscVec, IS, PetscIS, PetscDM, PetscSF, MPI_Comm, PetscObject
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
from petsc4py.PETSc cimport GetCommDefault, GetComm
from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import contextlib
import numpy as np
cimport numpy as np
import sympy

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, PetscDM*)
    PetscErrorCode DMPlexComputeGeometryFVM( PetscDM dm, PetscVec *cellgeom, PetscVec *facegeom)
    PetscErrorCode DMPlexGetMinRadius(PetscDM dm, PetscReal *minradius)
    PetscErrorCode VecDestroy(PetscVec *v)
    PetscErrorCode DMFieldEvaluate(void *field, PetscVec points, int datatype, void *B, void *D, void *H)
    PetscErrorCode DMGetField(PetscDM dm, PetscInt f, void *label, void *field)
    PetscErrorCode DMInterpolationCreate(MPI_Comm comm, void *ipInfo)
    PetscErrorCode DMInterpolationSetDim(void *ipInfo, PetscInt dim)
    PetscErrorCode DMInterpolationSetDof(void *ipInfo, PetscInt dof)
    PetscErrorCode DMInterpolationAddPoints(void *ipInfo, PetscInt n, PetscReal points[])
    PetscErrorCode DMInterpolationSetUp(void *ipInfo, PetscDM dm, int petscbool)
    PetscErrorCode DMInterpolationEvaluate(void *ipInfo, PetscDM dm, PetscVec vec, PetscVec out)
    PetscErrorCode DMInterpolationDestroy(void *ipInfo)
    PetscErrorCode DMInterpolationGetVector(void* ipInfo, PetscVec *v)
    PetscErrorCode DMInterpolationRestoreVector(void* ipInfo, PetscVec *v)
    PetscErrorCode DMDestroy(PetscDM *dm)
    # PetscErrorCode CHKERRQ(PetscErrorCode ierr)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMCreateLocalVector(PetscDM dm, PetscVec *vec)
    MPI_Comm MPI_COMM_SELF

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")



# class MeshVariable(sympy.Function):
#     def __new__(cls, mesh, *args, **kwargs):
#         # call the sympy __new__ method without args/kwargs, as unexpected args 
#         # trip up an exception.  
#         obj = super(MeshVariable, cls).__new__(cls, mesh.N.base_vectors()[0:mesh.dim])
#         return obj
    
#     @classmethod
#     def eval(cls, x):
#         return None

#     def _ccode(self, printer):
#         # return f"{type(self).__name__}({', '.join(printer._print(arg) for arg in self.args)})_bobobo"
#         return f"petsc_u[0]"

from enum import Enum
class VarType(Enum):
    SCALAR=1
    VECTOR=2
    OTHER=3  # add as required 

class MeshVariable:
    class MeshVariableFn(sympy.Function):
        _printstr = None 
        _header = None
        def _ccode(self, printer):
            # This is just a copy paste from the analytic solutions implementation. 
            # For usage within element assembly we generally won't need to use this as
            # PETSc passes in evaluated variables via the (Benny-And-The) Jets 
            # function. 
            # However, if we wish to use a variable in a completely independent
            # system, we may still need to provide an implementation here. We may also
            # need to provide a code printed implementation for evaluation from within 
            # Python via `evaluate()` (or equivalent). Alternatively, it might be sufficent
            # to simply used sympy `evalf` methods, so that overloading type behaviours (+/-/*/etc)
            # are handled by sympy within its evaluation tree, with c-level implementations 
            # (such as `DMFieldEvaluate`) accessed at the nodes of the tree. 
            raise RuntimeError("Not implemented.")
            printer.headers.add(self._header)
            param_str = ""
            for arg in self.args:
                param_str += printer._print(arg) + ","
            param_str = param_str[:-1]  # drop final comma
            if not self._printstr:
                raise RuntimeError("Trying to print unprintable function.")
            return self._printstr.format(param_str)

    def __init__(self, mesh, num_components, name, vtype, degree=1):
        if mesh._accessed:
            raise RuntimeError("It is not possible to add new variables to a mesh after existing variables have been accessed.")
        if name in mesh.vars.keys():
            raise ValueError("Variable with name {} already exists on mesh.".format(name))
        if not isinstance(vtype, VarType):
            raise ValueError("'vtype' must be an instance of 'Variable_Type', for example `uw.mesh.VarType.SCALAR`.")
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
        if   vtype==VarType.SCALAR:
            self._fn = sympy.Function(name)(*self.mesh.r)
        elif vtype==VarType.VECTOR:
            if num_components!=mesh.dim:
                raise ValueError("For 'VarType.VECTOR' types 'num_components' must equal 'mesh.dim'.")
            from sympy.vector import VectorZero
            self._fn = VectorZero()
            subnames = ["_x","_y","_z"]
            for comp in range(num_components):
                subfn = sympy.Function(name+subnames[comp])(*self.mesh.r)
                self._fn += subfn*self.mesh.N.base_vectors()[comp]
        super().__init__()

        self.mesh.vars[name] = self

        self.__lvec = None
        self._data = None
        self._is_accessed = False

    # def save(self, filename):
    #     viewer = PETSc.Viewer().createHDF5(filename, "w")
    #     viewer(self.petsc_fe)
    #     generateXdmf(filename)

    def evaluate(self, np.ndarray coords):
        if coords.shape[1] not in [2,3]:
            raise ValueError("Provided `coords` must be 2 or 3 dimensional array.")
        if coords.dtype != np.double:
            raise ValueError("Provided `coords` must be an array of doubles.")

        # Create interpolation object.
        # Use MPI_COMM_SELF as following uw2 paradigm, interpolations will be local.
        # TODO: Investigate whether it makes sense to default to global operations here.
        cdef void* ipInfo
        cdef PetscErrorCode ierr
        import time
        global now_time 
        now_time = time.time()
        def delta_time():
            global now_time
            old_now_time = now_time
            now_time = time.time()
            return now_time - old_now_time
        ierr = DMInterpolationCreate(MPI_COMM_SELF, &ipInfo); CHKERRQ(ierr)
        ierr = DMInterpolationSetDim(ipInfo, self.mesh.dim); CHKERRQ(ierr)
        ierr = DMInterpolationSetDof(ipInfo, self.num_components); CHKERRQ(ierr)
        print(f"eval 1 {delta_time()}")

        # Add interpolation points
        # Get c-pointer to data buffer
        cdef double* coords_buff = <double*> coords.data
        ierr = DMInterpolationAddPoints(ipInfo, coords.shape[0], coords_buff); CHKERRQ(ierr)
        print(f"eval 2 {delta_time()}")

        # Setup interpolation
        cdef DM dm = self.mesh.dm
        cdef PetscDM subdm
        cdef PetscInt fields = self.field_id
        DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm)
        ierr = DMInterpolationSetUp(ipInfo, subdm, 0); CHKERRQ(ierr)
        print(f"eval 3 {delta_time()}")

        cdef np.ndarray outarray = np.empty([len(coords), self.num_components], dtype=np.double)
        # Vector for output
        cdef Vec outvec = PETSc.Vec().createWithArray(outarray,comm=PETSc.COMM_SELF)
        # Execute interpolation.
        cdef Vec pyfieldvec
        print(f"eval 4 {delta_time()}")
        with self.mesh.access():
            pyfieldvec = self.vec
            ierr = DMInterpolationEvaluate(ipInfo, subdm, pyfieldvec.vec, outvec.vec);CHKERRQ(ierr)
        print(f"eval 5 {delta_time()}")
        ierr = DMDestroy(&subdm);CHKERRQ(ierr)
        ierr = DMInterpolationDestroy(&ipInfo);CHKERRQ(ierr)

        return outarray


    @property
    def fn(self):
        return self._fn

    def _set_vec(self, available):
        cdef DM subdm = PETSc.DM()
        cdef DM dm = self.mesh.dm
        cdef PetscInt fields = self.field_id
        if self.__lvec==None:
            # Create a subdm for this variable.
            # This allows us to generate a local vectors.
            ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)
            self.__lvec  = subdm.createLocalVector()
            self.__gvec  = subdm.createGlobalVector()
            ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)
        self._available = available

    @property
    def vec(self):
        if not self._available:
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")
        return self.__lvec

    @property
    def vec_global(self):
        if not self._available:
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")
        return self.__gvec

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError("Data must be accessed via the mesh `access()` context manager.")
        return self._data

    def min(self):
        """
        Global min.
        """
        return self.__gvec.min()

    def max(self):
        """
        Global max.
        """
        return self.__gvec.max()

class _MeshBase():
    def __init__(self, simplex, *args,**kwargs):
        self.isSimplex = simplex
        # create boundary sets
        for ind,val in enumerate(self.boundary):
            boundary_set = self.dm.getStratumIS("marker",ind+1)        # get the set
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

    def getLocalVariableVec(self) -> PETSc.Vec:
        """
        Returns a local Petsc temporary vector containing a 
        flattened representation of all the mesh variables.

        This is temporary vector which should be returned using
        `restoreLocalVariableVec()`.

        Alternatively, the user might use the context managed 
        version `getLocalVariableVecManaged()`.
        """
        # create the local vector (memory chunk) and attach to original dm
        a_local = self.dm.getLocalVec()
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

        self.dm.globalToLocal(a_global,a_local)
        self.dm.restoreGlobalVec(a_global)
        self._a_local = a_local
        return a_local

    def restoreLocalVariableVec(self):
        """
        Restores vector obtained using `getLocalVariableVec()`
        """
        self.dm.restoreLocalVec(self._a_local)

    @contextlib.contextmanager
    def getLocalVariableVecManaged(self) -> PETSc.Vec:
        """
        Context managed version of `getLocalVariableVec()`.
        """
        try:
            yield self.getLocalVariableVec()
        except:
            raise
        finally:
            self.restoreLocalVariableVec()


    @contextlib.contextmanager
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

        cdef DM subdm = PETSc.DM()
        cdef DM dm = self.dm
        cdef PetscInt fields
        cdef Vec vec = PETSc.Vec()

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
            var._data = var.vec.array.reshape( (-1, var.num_components) )
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False

        try:
            yield
        except:
            raise
        finally:
            for var in self.vars.values():
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
                    ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)
                var._data = None
                var._set_vec(available=False)
                var._is_accessed = False


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
        # get number of nodes
        nnodes = len(arr)/self.dim
        # round & cast to int to ensure correct value
        nnodes = int(round(nnodes))
        return arr.reshape((nnodes, self.dim))

    @property
    def dim(self):
        """ Number of dimensions of the mesh """
        return self.dm.getDimension()

    def save(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, "w")
        viewer(self.dm)
        generateXdmf(filename)

    def add_mesh_variable(self):
        return

    @property
    def vars(self):
        return self._vars

class Mesh(_MeshBase):
    def __init__(self, 
                elementRes=(16, 16), 
                minCoords=None,
                maxCoords=None,
                simplex=False,
                interpolate=False):

        options = PETSc.Options()
        options["dm_plex_separate_marker"] = None
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

        self.dm.view()

        super().__init__(simplex=simplex)


class Spherical(_MeshBase):
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
            
