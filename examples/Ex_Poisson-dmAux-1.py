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
mesh = uw.Mesh(elementRes=(10,10), minCoords=(-2.2,-.4))
bnds = mesh.boundary

# %%
# Create Poisson object
poisson = Poisson(mesh)

# %%
# Model parameters
T1 = -1.0   # top surface temperature
T0 = 7.0   # bottom surface temperature
k = 3.     # diffusivity
h = 10.     # heat production, source term
y1 = mesh.maxCoords[1]
y0 = mesh.minCoords[1]

# %%
aux_var = poisson.createAux(num_components=1, isSimplex=mesh.isSimplex)
# example of setting the auxiliary field by numpy array, a.k.a by hand
# fancy petsc "compose" way to get the aux petsc vector directly
lVec = poisson.mesh.dm.query("A")
lVec.array[:] = k

# %%
# Set some things
poisson.k = aux_var.fn 
poisson.h = -h
poisson.add_dirichlet_bc( T0, bnds.BOTTOM )  
poisson.add_dirichlet_bc( T1, bnds.TOP )  

# %%
# Solve time
poisson.solve()
soln = poisson.u_local


# %%
# analytic solution definitions
def analyticTemperature(y, h, k, c0, c1):
     return -h/(2.*k)*y**2 + c0*y + c1

# arbitrary constant given the 2 dirichlet conditions
c0 = (T1-T0+h/(2*k)*(y1**2-y0**2)) / (y1-y0)
c1 = T1 + h/(2*k)*y1**2 - c0*y1

# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
if not np.allclose(analyticTemperature(mesh.data[:,1], h, k, c0, c1),soln.array):
    raise RuntimeError("Unexpected values encountered.")
