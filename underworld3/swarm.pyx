from petsc4py.PETSc cimport MPI_Comm
from petsc4py.PETSc cimport GetCommDefault, GetComm
import petsc4py.PETSc as PETSc
from .petsc_gen_xdmf import generateXdmf
from mpi4py import MPI

comm = MPI.COMM_WORLD

from enum import Enum
class SwarmType(Enum):
    DMSWARM_PIC = 1

class SwarmPICLayout(Enum):
    DMSWARMPIC_LAYOUT_GAUSS = 1

class VarType(Enum):
    SCALAR=1
    VECTOR=2
    OTHER=3  # add as required 


class SwarmVariable:

    def __init__(self, swarm, name, num_components, dtype=PETSc.ScalarType):

        if name in swarm.vars.keys():
            raise ValueError("Variable with name {} already exists on swarm.".format(name))
    
        self.name = name
        self.swarm = swarm
        self.num_components = num_components
        self.dtype = dtype
        self.swarm.registerField(self.name, self.num_components, dtype=self.dtype)

    def data(self):
        data = self.swarm.getField(self.name)
        return data
        
    @property
    def fn(self):
        return self._fn


class Swarm(PETSc.DMSwarm):

    def __init__(self, mesh):
        
        self.mesh = mesh
        self.dim = mesh.dim
        self.dm = Swarm.create(self)
        self.dm.setDimension(self.dim)
        self.dm.setType(SwarmType.DMSWARM_PIC.value)
        self.dm.setCellDM(mesh.dm)

        # dictionary for variables
        import weakref
        self._vars = weakref.WeakValueDictionary()

    def populate(self, ppcell=25, layout=SwarmPICLayout.DMSWARMPIC_LAYOUT_GAUSS):
        
        self.ppcell = ppcell
        
        if not isinstance(layout, SwarmPICLayout):
            raise ValueError("'layout' must be an instance of 'SwarmPICLayout'")
        
        self.layout = layout
        
        elements_counts = self.mesh.elementRes[0] * self.mesh.elementRes[1]
        self.dm.finalizeFieldRegister()
        self.dm.setLocalSizes(elements_counts * ppcell, 0)
        self.dm.insertPointUsingCellDM(self.layout.value, ppcell)
        return self

    def add_variable(self, name, num_components=1, dtype=PETSc.ScalarType):
        var = SwarmVariable(self, name, num_components, dtype)
        return var

    def save(self, filename):
        self.dm.viewXDMF(filename)
    
    def particle_coordinates(self):
        data = self.getField("DMSwarmPIC_coor")
        self.restoreField("DMSwarmPIC_coor")
        return data.reshape((-1, self.dim))

    @property
    def vars(self):
        return self._vars

