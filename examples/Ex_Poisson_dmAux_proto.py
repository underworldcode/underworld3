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
mesh = uw.Mesh(elementRes=(9,9), minCoords=(-2.2,-.4))
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
# variable description list - (name, number_components, degree)
# avar_want = [ ("diff", 1, 1) ]

avar_want = [ ("diff", 1, 1),("vel", 2, 1)]

avar = {}

# %%
for aux in avar_want:
    (name, nc, degree) = aux
    options.setValue(name+"_petscspace_degree", degree)
    
    if nc == 1:
        varType = uw.mesh.VarType.SCALAR
    else:
        varType = uw.mesh.VarType.VECTOR

    avar[name] = uw.MeshVariable(mesh, nc, name, varType)

# %%
# create the local vector (memory chunk) and attach to original dm
mesh.aux_dm.createDS()
a_local = mesh.aux_dm.createLocalVector()
mesh.dm.compose("A", a_local)

# %%
for var in mesh.avars.values():
    print(var.fn)

# %%
# a means to index into the a_local vector - not sure how to interpret it 
auxds = mesh.aux_dm.getDS()
field_components = auxds.getComponents()
auxds.getDimensions(), field_components, auxds.getTotalComponents()

# %%
a_local.array.reshape(-1,auxds.getTotalComponents()).shape

# %%
# example of setting the auxiliary field by numpy array, a.k.a by hand
# fancy petsc "compose" way to get the aux petsc vector directly
lVec = poisson.mesh.dm.query("A")
lVec.array[:] = k # just set every aux dof to k

# %%
# Set some things
poisson.k = avar['diff'].fn 
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
