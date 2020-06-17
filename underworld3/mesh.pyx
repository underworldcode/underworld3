from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import numpy as np
import sympy as sym

class Mesh():

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
        self.isSimplex = simplex
        self.plex = PETSc.DMPlex().createBoxMesh(
            elementRes, 
            lower=minCoords, 
            upper=maxCoords,
            simplex=simplex)
        part = self.plex.getPartitioner()
        part.setFromOptions()
        self.plex.distribute()
        self.plex.setFromOptions()

        # from sympy import MatrixSymbol
        # self._x = MatrixSymbol('x', m=1, n=self.dim)

        from sympy.vector import CoordSys3D
        self._N = CoordSys3D("N")
        import weakref
        self._vars = weakref.WeakValueDictionary()

        # sort bcs
        from enum import Enum
        class Boundary2D(Enum):
            BOTTOM = 1
            RIGHT  = 2
            TOP    = 3
            LEFT   = 4
        class Boundary3D(Enum):
            BOTTOM = 1
            TOP    = 2
            FRONT  = 3
            BACK   = 4
            RIGHT  = 5
            LEFT   = 6
        
        if len(elementRes) == 2:
            self.boundary = Boundary2D
        else:
            self.boundary = Boundary3D

        for ind,val in enumerate(self.boundary):
            boundary_set = self.plex.getStratumIS("marker",ind+1)        # get the set
            self.plex.createLabel(str(val).encode('utf8'))               # create the label
            boundary_label = self.plex.getLabel(str(val).encode('utf8')) # get label
            if boundary_set:
                boundary_label.insertIS(boundary_set, 1) # add set to label with value 1

        self.plex.view()

    @property
    def N(self):
        return self._N

    @property
    def data(self):
        nnodes = np.prod([val + 1 for val in self.elementRes])
        return self.plex.getCoordinates().array.reshape((nnodes, self.dim))

    @property
    def dim(self):
        """ Number of dimensions of the mesh """
        return self.plex.getDimension()

    def save(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, "w")
        viewer(self.plex)
        generateXdmf(filename)

    def add_mesh_variable(self):
        return

    @property
    def vars(self):
        return self._vars

# class MeshVariable(sym.Function):
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

    def __init__(self, mesh, num_components, name, vtype, isSimplex=False):
        if name in mesh.vars.keys():
            raise ValueError("Variable with name {} already exists on mesh.".format(name))
        if not isinstance(vtype, VarType):
            raise ValueError("'vtype' must be an instance of 'Variable_Type', for example `uw.mesh.VarType.SCALAR`.")
        self.vtype = vtype
        self.mesh = mesh
        self.num_components = num_components
        self.petsc_fe = PETSc.FE().createDefault(mesh.plex.getDimension(), num_components, isSimplex, PETSc.DEFAULT, name+"_", PETSc.COMM_WORLD)
        self.field_id = mesh.plex.getNumFields()
        mesh.plex.setField(self.field_id,self.petsc_fe)
        # create associated sympy function
        if   vtype==VarType.SCALAR:
            self._fn = sym.Function(name)(*self.mesh.N.base_scalars()[0:mesh.dim])
        elif vtype==VarType.VECTOR:
            if num_components!=mesh.dim:
                raise ValueError("For 'VarType.VECTOR' types 'num_components' must equal 'mesh.dim'.")
            from sympy.vector import VectorZero
            self._fn = VectorZero()
            subnames = ["_x","_y","_z"]
            for comp in range(num_components):
                subfn = sym.Function(name+subnames[comp])(*self.mesh.N.base_scalars()[0:mesh.dim])
                self._fn += subfn*self.mesh.N.base_vectors()[comp]
        super().__init__()
        # now add to mesh list
        self.mesh.vars[name] = self

    @property
    def fn(self):
        return self._fn

    