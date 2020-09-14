# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.poisson import Poisson
import numpy as np

options = PETSc.Options()
options["pc_type"]  = "svd"

options["ksp_rtol"] = 1.0e-7
# options["ksp_monitor_short"] = None

# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
options["snes_rtol"] = 1.0e-7

# %%
# nonlinear example
mesh = uw.Mesh( elementRes=(8, 8), simplex=False )
poisson = Poisson(mesh, degree=1)
bnds = mesh.boundary


# %%
u = poisson.u.fn


# %%
from sympy.vector import gradient
nabla_u = gradient(u)
poisson.k = 0.5*(nabla_u.dot(nabla_u))
poisson.k


# %%
N = mesh.N
abs_r2 = (N.x**1 + N.y**1)
poisson.h = 16*abs_r2

poisson.add_dirichlet_bc(abs_r2, [bnds.TOP,bnds.BOTTOM,bnds.LEFT,bnds.RIGHT] )

# %%
poisson.solve()
soln_1 = poisson.u_local

# %%
aux_var = poisson.createAux(num_components=1, degree=1, isSimplex=mesh.isSimplex)
# example of setting the auxiliary field by numpy array, a.k.a by hand
# fancy petsc "compose" way to get the aux petsc vector directly
lVec = poisson.mesh.dm.query("A")

lVec.array[:] = 16*(mesh.data[:,0]**1 + mesh.data[:,1]**1)

# %%
poisson.h = aux_var.fn


# %%
### TODO: fix below. 
# The aux_var can't be used for the dirichlet conditions, it's something 
# to do with the (aux_var) petsc vector values being available inside 
# the boundary condition callback of PETSC

# poisson.add_dirichlet_bc(aux_var.fn, [bnds.TOP,bnds.BOTTOM,bnds.LEFT,bnds.RIGHT] )

# %%
poisson.solve()
soln_2 = poisson.u_local

# %%
if not np.allclose(soln_1,soln_2):
    raise RuntimeError("Unexpected results")
