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
# +
# but we have this on the mesh

mesh.X
# -

v_soln = uw.mesh.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    mesh, 1, degree=1 )


# +
# This is the way to get back to a matrix after diff by array for two row vectors (yuck)
# We might just try to wrap this so it give back sane types.
# Diff by array blows up a 1x3 by 1x3 into a 1,3,1,3 tensor rather than a 3x3 matrix 
# and the 1 indices cannot be automatically removed 

VX = sympy.derive_by_array(v_soln.f, mesh.X).reshape(v_soln.f.shape[1], mesh.X.shape[1]).tomatrix().T
# -




