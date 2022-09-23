# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank

options = PETSc.Options()
# options["help"] = None

# options["pc_type"]  = "svd"

options["ksp_rtol"] =  1.0e-8
options["snes_converged_reason"] = None
options["snes_monitor"] = None
# options["snes_monitor_short"] = None
# options["snes_view"]=None
options["snes_test_jacobian"] = None
options["snes_rtol"] = 1.0e-7
# options["snes_max_it"] = 1
# options["snes_linesearch_monitor"] = None


# %%
n_els = 32
v_degree = 1
mesh = uw.util_mesh.StructuredQuadBox(elementRes=(n_els,n_els), 
                                      minCoords=(0.0,0.0),
                                      maxCoords=(1.0,1.0))

# %%
# NL problem 
# Create solution functions
from underworld3.function.analytic import AnalyticSolNL_velocity, AnalyticSolNL_bodyforce, AnalyticSolNL_viscosity 
r = mesh.r
eta0 = 1.
n = 1
r0 = 1.5
params = (eta0, n, r0) 
sol_bf   = AnalyticSolNL_bodyforce( *params, *r )
sol_vel  = AnalyticSolNL_velocity(  *params, *r )
sol_visc = AnalyticSolNL_viscosity( *params, *r )

# %%
stokes = Stokes(mesh, u_degree=v_degree, p_degree=v_degree-1 )
stokes.add_dirichlet_bc( sol_vel, ["left", "right"],  [0,1] )  # left/right: function, markers, components
stokes.add_dirichlet_bc( sol_vel, ["top", "bottom"], [ 1, ] )  # top/bottom: function, markers, components


stokes.petsc_options["ksp_rtol"] =  1.0e-6
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None
# stokes.petsc_options["snes_view"]=None
# stokes.petsc_options["snes_test_jacobian"] = None
stokes.petsc_options["snes_rtol"] = 1.0e-5
# stokes.petsc_options["snes_max_it"] = 1
# stokes.petsc_options["snes_linesearch_monitor"] = None


# %%
stokes.bodyforce = sol_bf
# do linear first to get reasonable starting place
stokes.viscosity = 1.
stokes.solve()
# %%
stokes._u_f0

# %%
# get strainrate
sr = stokes.strainrate
# not sure if the following is needed as div_u should be zero
sr -= (stokes.div_u/mesh.dim)*sympy.eye(mesh.dim)
# second invariant of strain rate
inv2 = sr[0,0]**2 + sr[0,1]**2 + sr[1,0]**2 + sr[1,1]**2
inv2 = 1/2*inv2
inv2 = sympy.sqrt(inv2)
alpha_by_two = 2/r0 - 2
stokes.viscosity = 2*eta0*inv2**alpha_by_two
stokes.penalty = 1.0
# stokes._Ppre_fn = 0.01 + 1.0 / (0.01 + stokes.viscosity)
stokes.solve(zero_init_guess=False)

vdiff = stokes.u.fn - sol_vel
vdiff_dot_vdiff = uw.maths.Integral(mesh, vdiff.dot(vdiff)).evaluate()
v_dot_v = uw.maths.Integral(mesh, stokes.u.fn.dot(stokes.u.fn)).evaluate()

import math
rel_rms_diff = math.sqrt(vdiff_dot_vdiff/v_dot_v)
if rank==0: print(f"RMS diff = {rel_rms_diff}")

if not np.allclose(rel_rms_diff, 0.00109, rtol=1.e-2):
    raise RuntimeError("Solve did not produce expected result.")

# %%
stokes._uu_g3[0,1]

# %%
