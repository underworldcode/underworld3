from mpi4py import MPI  # for initialising MPI
import petsc4py as _petsc4py
import sys

_petsc4py.init(sys.argv)

from petsc4py import PETSc

# pop the default petsc Signal handler to let petsc errors appear in python
# unclear if this is the appropriate way see discussion
# https://gitlab.com/petsc/petsc/-/issues/1066

PETSc.Sys.popErrorHandler()


from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


print(f"{rank} - All done", flush=True)
