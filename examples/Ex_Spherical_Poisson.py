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
mesh = uw.Spherical(refinements=1)
bnds = mesh.boundary

# %%
# Create Poisson object
poisson = Poisson(mesh)

# %%
# Set some things
poisson.k = 1. 
poisson.h = 0.
poisson.add_dirichlet_bc( mesh.N.x, bnds.OUTER )  

# %%
# Solve time
poisson.solve()
soln = poisson.u_local

# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
if not np.allclose(mesh.data[:,0],soln.array):
    raise RuntimeError("Unexpected values encountered.")

# %%
#Underworld3 plotting prototype using lavavu
import plot
# %%

#Create viewer
resolution=(1000,800)

fig = plot.Plot(rulers=True)
fig.nodes(mesh, pointsize=5, pointtype="sphere")
fig.display(resolution)
# %%

fig = plot.Plot(rulers=True)
fig.edges(mesh)
fig.display(resolution)
# %%

fig = plot.Plot(rulers=True)
fig.faces(mesh, values=soln.array, colourmap="diverge")
fig.colourbar(align="right", size=(0.865,10), position=26, outline=False)
fig.display(resolution)
# %%
mesh.save("mesh.h5")
