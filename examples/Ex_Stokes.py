# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.stokes import Stokes
import numpy as np

options = PETSc.Options()
# options["help"] = None

# options["pc_type"]  = "svd"

options["ksp_rtol"] =  1.0e-6
options["ksp_monitor_short"] = None

# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_rtol"] = 1.0e-2  # set this low to force single SNES it. 
# options["snes_max_it"] = 1

# %%
n_els = 64
mesh = uw.Mesh(elementRes=(n_els,n_els))
# %%
v_degree = 1
stokes = Stokes(mesh, u_degree=v_degree, p_degree=v_degree-1 )

# %%
# Set some things
import sympy
from sympy import Piecewise
N = mesh.N
eta_0 = 1.
x_c   = 0.5
f_0   = -1.
stokes.viscosity = 1. 
stokes.bodyforce = Piecewise((f_0, N.x>x_c,), \
                            (  0.,    True) )*N.j
# free slip.  
# note with petsc we always need to provide a vector of correct cardinality. 
bnds = mesh.boundary
stokes.add_dirichlet_bc( (0.,0.), [bnds.TOP,  bnds.BOTTOM], 1 )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( (0.,0.), [bnds.LEFT, bnds.RIGHT],  0 )  # left/right: components, function, markers

# %%
# Solve time
stokes.solve()

# %%
import underworld as uw2
solC = uw2.function.analytic.SolC()

# %%
vel_soln_analytic = solC.fn_velocity.evaluate(mesh.data)

# %%
soln = stokes.u_local.array



# %%
n_nodes = n_els*v_degree+1
vel_soln = soln[(n_els*n_els)::].reshape((n_nodes*n_nodes,2)) 

# %%
from numpy import linalg as LA
print("Diff norm = {}".format(LA.norm(vel_soln - vel_soln_analytic)))

# %%
if not np.allclose(vel_soln, vel_soln_analytic,rtol=1.e-2):
    raise RuntimeError("Solve did not produce expected result.")
