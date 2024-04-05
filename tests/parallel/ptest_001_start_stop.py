import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy


mesh1 = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.1)

print(f"{uw.mpi.rank} - define continuous variable", flush=True)
C1 = uw.discretisation.MeshVariable(r"C_1", mesh1, 1, degree=1, continuous=True)
print(f"{uw.mpi.rank} - define continuous variable", flush=True)
C2 = uw.discretisation.MeshVariable(r"C_2", mesh1, 1, degree=2, continuous=True)
print(f"{uw.mpi.rank} - define continuous variable", flush=True)
C3 = uw.discretisation.MeshVariable(r"C_3", mesh1, 1, degree=3, continuous=True)

# This always seems to fail in parallel

print(f"{uw.mpi.rank} - define dis-continuous (dC0) variable", flush=True)
dC0 = uw.discretisation.MeshVariable(r"dC_0", mesh1, 1, degree=0, continuous=False)
print(f"{uw.mpi.rank} - define dis-continuous (dC1) variable", flush=True)
dC1 = uw.discretisation.MeshVariable(r"dC_1", mesh1, 1, degree=1, continuous=False)
print(f"{uw.mpi.rank} - define dis-continuous (dC2) variable", flush=True)
dC2 = uw.discretisation.MeshVariable(r"dC_2", mesh1, 1, degree=2, continuous=False)

print(f"{uw.mpi.rank} - All done", flush=True)
