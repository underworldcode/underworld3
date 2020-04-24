# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw


# %%
mesh = uw.Mesh()


# %%
from underworld3 import poisson
# Create Poisson object
poisson = poisson.Poisson(mesh)


# %%
# Set some things
poisson.k = 1. 
poisson.h = 0.
poisson.add_dirichlet_bc( 1., 1 )  # index 1 is bottom boundary
poisson.add_dirichlet_bc( 0., 3 )  # index 3 is top boundary


# %%
# Solve time
poisson.solve()
soln = poisson.u_local


# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
if not np.allclose((1. - mesh.data[:,1]),soln.array):
    raise RuntimeError("Unexpected values encountered.")


# %%
# Now let's construct something a little more complex.
# First get the coord system off the mesh/dm.
N = mesh.N


# %%
# Create some function using one of the base scalars N.x/N.y/N.z
import sympy
k = sympy.exp(-N.y)


# %%
# View
k


# %%
# Don't forget to set the diffusivity
poisson.k = k


# %%
poisson.solve()


# %%
# Simply confirm different results
if np.allclose(soln.array, poisson.u_local.array):
    raise RuntimeError("Unexpected values encountered.")

