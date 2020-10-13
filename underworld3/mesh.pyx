from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec, PetscIS, PetscDM, PetscSF, MPI_Comm
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
from petsc4py.PETSc cimport GetCommDefault, GetComm
from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import numpy as np
import sympy

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMPlexSetMigrationSF( PetscDM, PetscSF )
    PetscErrorCode DMPlexGetMigrationSF( PetscDM, PetscSF*)
    PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, PetscDM*)


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
        self._avars = weakref.WeakValueDictionary()

        # add the auxiliary dm
        self.aux_dm = self.dm.clone()

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
    @property
    def avars(self):
        return self._avars


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

    def __init__(self, mesh, num_components, name, vtype, unknown=False):
        if name in mesh.vars.keys():
            raise ValueError("Variable with name {} already exists on mesh.".format(name))
        if not isinstance(vtype, VarType):
            raise ValueError("'vtype' must be an instance of 'Variable_Type', for example `uw.mesh.VarType.SCALAR`.")
        self.vtype = vtype
        self.unknown = unknown
        self.mesh = mesh
        self.num_components = num_components

        # choose to put variable on the dm or auxiliary dm
        odm = mesh.dm if unknown else mesh.aux_dm

        self.petsc_fe = PETSc.FE().createDefault(odm.getDimension(), num_components, self.mesh.isSimplex, PETSc.DEFAULT, name+"_", PETSc.COMM_WORLD)

        self.field_id = odm.getNumFields()
        odm.setField(self.field_id,self.petsc_fe)
        
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
        # now add to mesh list
        if unknown:
            self.mesh.vars[name] = self
        else:
            self.mesh.avars[name] = self
    # def save(self, filename):
    #     viewer = PETSc.Viewer().createHDF5(filename, "w")
    #     viewer(self.petsc_fe)
    #     generateXdmf(filename)


    @property
    def fn(self):
        return self._fn

#     def getLocalData(self):
#         # create a subdm for this variable. 
#         # this allows us to extract corresponding arrays.
#         cdef DM subdm = PETSc.DMPlex()
#         cdef PetscInt fields = self.field_id
#         cdef PetscIS fis = NULL #PETSc.IS().create()
#         cdef DM dm 
#         # is this conditional assingment costly?
#         if self.unknown:
#             dm = self.mesh.dm
#         else:
#             dm = self.mesh.aux_dm
# 
#         DMCreateSubDM(dm.dm, 1, &fields, &fis, &subdm.dm)
#         vec = subdm.createLocalVector()
#         return vec
