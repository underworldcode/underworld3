# # Rayleigh Taylor in a Disc
#
# Demonstrating the use of the `sympy.tensor` interface to solvers and mesh variables.
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

render = True

# +
import meshio

mesh = uw.util_mesh.Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.05)

mesh.dm.view()   

# +
import sympy

# Some useful coordinate stuff 

x = mesh.N.x
y = mesh.N.y
# -
v_soln = uw.mesh.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    mesh, 1, degree=1 )


from sympy.tensor.array.expressions import conv_matrix_to_array
from sympy.tensor.array.expressions import conv_array_to_matrix

VX = sympy.derive_by_array(v_soln.f, mesh.X).reshape(v_soln.f.shape[1], mesh.X.shape[1]).tomatrix().T

VX



swarm     = uw.swarm.Swarm(mesh=meshbox)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=4)


v = sympy.Matrix.zeros(3,1)
v[0] = 1




swarm.populate(fill_param=5)
