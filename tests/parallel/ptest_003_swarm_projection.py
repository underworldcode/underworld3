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

# Continuous function 1
print(f"{uw.mpi.rank} - define continuous variables", flush=True)
s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)
s_soln = uw.discretisation.MeshVariable("T", mesh1, 1, degree=2)

# Swarm
print(f"{uw.mpi.rank} - define swarm / swarm variable", flush=True)
swarm = uw.swarm.Swarm(mesh=mesh1)
sw_values = uw.swarm.SwarmVariable("Ss", swarm, 1, proxy_degree=1, proxy_continuous=True)
swarm.populate(fill_param=3)

# Projection operation
print(f"{uw.mpi.rank} - build projections", flush=True)
scalar_projection = uw.systems.Projection(mesh1, s_soln)
scalar_projection.uw_function = sw_values.sym[0]
scalar_projection.smoothing = 1.0e-6

# Values on swarm
print(f"{uw.mpi.rank} - define swarm values", flush=True)
with swarm.access(sw_values):
    sw_values.data[:, 0] = uw.function.evaluate(s_fn, swarm.data)

print(f"{uw.mpi.rank} - solve projection", flush=True)
scalar_projection.solve()

print(f"Finalised")
