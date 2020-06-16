# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.poisson import Poisson
import numpy as np

options = PETSc.Options()
# options["snes_monitor_short"] = True
# options["snes_converged_reason"] = True
options["pc_type"]  = "svd"
# options["snes_type"]  = "fas"
# options["ksp_rtol"] =  1.0e-10
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
options["snes_view"]=None
options["ksp_rtol"] = 1.0e-10
# options["ksp_monitor_short"] = None
# options["snes_rtol"] = 1.0e-10


# %%
mesh = uw.Mesh()


# %%
# Create Poisson object
poisson = Poisson(mesh)


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


# %%
# nonlinear example
mesh = uw.Mesh( elementRes=(8, 8), simplex=False )
poisson = Poisson(mesh, degree=1)


# %%
u = poisson.u.fn


# %%
from sympy.vector import gradient
nabla_u = gradient(u)
poisson.k = 0.5*(nabla_u.dot(nabla_u))
poisson.k


# %%
N = mesh.N
abs_r2 = (N.x**2 + N.y**2)
poisson.h = 16*abs_r2
poisson.h


# %%
poisson.add_dirichlet_bc(abs_r2, [1,] )
poisson.add_dirichlet_bc(abs_r2, [2,] )
poisson.add_dirichlet_bc(abs_r2, [3,] )
poisson.add_dirichlet_bc(abs_r2, [4,] )


# %%
poisson._setup_terms()
soln = poisson.u_local


# %%
poisson.solve(setup=False)
soln = poisson.u_local


# %%
exact = mesh.data[:,0]**2 + mesh.data[:,1]**2
l2 = np.linalg.norm(exact-soln.array[:])
print("L2 = {}".format(l2))
if not np.allclose(soln.array,exact,rtol=7.e-2):
    raise RuntimeError("Unexpected values encountered.")

