from petsc4py.PETSc cimport MPI_Comm
from petsc4py.PETSc cimport GetCommDefault, GetComm
import petsc4py.PETSc as PETSc
from .petsc_gen_xdmf import generateXdmf
from mpi4py import MPI

comm = MPI.COMM_WORLD


class Swarm(PETSc.DMSwarm):
    def __init__(self, mesh, ppcell=25):
        self.dm = Swarm.create(self)
        elements_counts = mesh.elementRes[0] * mesh.elementRes[1]
        self.dm.setDimension(mesh.dim)
        self.dm.setType(1)
        self.dm.setCellDM(mesh.dm)

        self.dm.finalizeFieldRegister()

        self.dm.setLocalSizes(elements_counts * ppcell, 0)
        self.dm.insertPointUsingCellDM(1, ppcell)

    def save(self, filename):
        self.dm.viewXDMF(filename)
