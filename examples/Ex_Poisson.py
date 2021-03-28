# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
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
mesh = uw.mesh.Mesh()
bnds = mesh.boundary

# %%
# Create Poisson object
poisson = Poisson(mesh)

# %%
# Set some things
poisson.k = 1.
poisson.h = 0.
poisson.add_dirichlet_bc( 1., bnds.BOTTOM )  
poisson.add_dirichlet_bc( 0., bnds.TOP )  

# %%
# Solve time
poisson.solve()

# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
with mesh.access():
    if not np.allclose((1. - mesh.data[:,1]),poisson.u.data[:,0]):
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
with mesh.access():
    orig_soln = poisson.u.data.copy()
poisson.solve()

# %%
# Simply confirm different results
with mesh.access():
    if np.allclose(poisson.u.data, orig_soln):
        raise RuntimeError("Unexpected values encountered.")

# %%
# nonlinear example
mesh = uw.mesh.Mesh( elementRes=(8, 8), simplex=False )
poisson = Poisson(mesh, degree=1)

# %%
u = poisson.u

# %%
from underworld3.function import gradient
nabla_u = gradient(u)
poisson.k = 0.5*(nabla_u.dot(nabla_u))
poisson.k

# %%
N = mesh.N
abs_r2 = (N.x**2 + N.y**2)
poisson.h = 16*abs_r2
poisson.h

# %%
poisson.add_dirichlet_bc(abs_r2, [bnds.TOP,bnds.BOTTOM,bnds.LEFT,bnds.RIGHT] )

# %%
poisson.solve()

# %%
with mesh.access():
    exact = mesh.data[:,0]**2 + mesh.data[:,1]**2
    l2 = np.linalg.norm(exact-poisson.u.data[:])
    print("L2 = {}".format(l2))
    if not np.allclose(poisson.u.data[:,0],exact[:],rtol=7.e-2):
        raise RuntimeError("Unexpected values encountered.")
