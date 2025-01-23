# TODO: Currently error with extreame memory usage only on arm64-macos.
# with errors like:
# ERROR: SCOTCH_dgraphMapInit: internal error

import underworld3 as uw
import numpy as np
import sympy

import argparse

from underworld3.systems.solvers import SNES_Darcy

parser = argparse.ArgumentParser()
parser.add_argument("--discontinuous", action="store_true")
args = parser.parse_args()

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
with mesh1.access():
    print(f"{uw.mpi.rank} ", s_values.coords[0:10], flush=True)

# Values on S2
# print(f"{uw.mpi.rank} - set values", flush=True)
with mesh1.access(s_values):
    print(s_values.data[0:10,0])
    s_values.data[:, 0] = 1.0 # uw.function.evalf(sympy.sympify(1), s_values.coords)

print(f"{uw.mpi.rank} - solve projection", flush=True)
mesh1.dm.view()

scalar_projection.solve()

print(f"{uw.mpi.rank} - solve projection", flush=True)
mesh1.dm.view()


print(f"Finalised")
