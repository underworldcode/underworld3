from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import numpy as np
import sympy as sym

class Mesh():

    def __init__(self, elementRes=(16, 16), minCoords=(0., 0.),
                 maxCoords=(1.0, 1.0), simplex=False):
        options = PETSc.Options()
        options.setValue("dm_plex_separate_marker", None)
        self.elementRes = elementRes
        self.minCoords = minCoords
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

class MeshVariable:
    def __init__(self, mesh, num_components, name, isSimplex=False):
        if name in mesh.vars.keys():
            raise ValueError("Variable with name {} already exists on mesh.".format(name))
        self.mesh = mesh
        self.num_components = num_components
        self.petsc_fe = PETSc.FE().createDefault(mesh.plex.getDimension(), num_components, isSimplex, PETSc.DEFAULT, name+"_", PETSc.COMM_WORLD)
        self.field_id = mesh.plex.getNumFields()
        mesh.plex.setField(self.field_id,self.petsc_fe)
        # create associated sympy undefined function
        self._fn = sym.Function(name)(*self.mesh.N.base_scalars()[0:mesh.dim])
        super().__init__()
        # now add to mesh list
        self.mesh.vars[name] = self

    @property
    def fn(self):
        return self._fn

    