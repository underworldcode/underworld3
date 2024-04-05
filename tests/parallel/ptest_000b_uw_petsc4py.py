import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dm = PETSc.DM().create()

print(f"{rank} - All done", flush=True)
