import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--discontinuous", action="store_true")
args = parser.parse_args()

# from underworld3.cython import petsc_discretisation

mesh1 = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.1)
x, y = mesh1.X

# Continuous function
print(f"{uw.mpi.rank} - define continuous variables", flush=True)
s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)
s_soln = uw.discretisation.MeshVariable("S", mesh1, 1, degree=1)

# second mesh variable
print(f"{uw.mpi.rank} - define 2nd variable", flush=True)
s_values = uw.discretisation.MeshVariable("S2", mesh1, 1, degree=2, continuous=True)


# Projection operation
print(f"{uw.mpi.rank} - build projections", flush=True)
scalar_projection = uw.systems.Projection(mesh1, s_soln, verbose=True)
print(f"{uw.mpi.rank} - build projections ... done", flush=True)

scalar_projection.uw_function = s_values.sym[0]
scalar_projection.smoothing = 1.0e-6

print(f"{uw.mpi.rank} - check values ...", flush=True)

# S2 coordinates
with mesh1.access(s_values):
    print(f"{uw.mpi.rank} ", s_values.coords[0:10], flush=True)

# Values on S2
print(f"{uw.mpi.rank} - set values", flush=True)
with mesh1.access(s_values):
    s_values.data[:, 0] = uw.function.evaluate(s_fn, s_values.coords)

print(f"{uw.mpi.rank} - solve projection", flush=True)
scalar_projection.solve()

print(f"Finalised")
