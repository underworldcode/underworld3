# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy

options = PETSc.Options()
# options["help"] = None

# options["pc_type"]  = "svd"

options["ksp_rtol"] =  1.0e-8
# options["ksp_monitor_short"] = None
# options["ksp_monitor_true_residual"] = None
# options["ksp_converged_reason"] = None

# options["snes_type"]  = "qn"
# options["snes_type"]  = "nrichardson"
options["snes_converged_reason"] = None
options["snes_monitor"] = None
# options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_rtol"] = 1.0e-7
# options["snes_max_it"] = 1
# options["snes_linesearch_monitor"] = None


# %%
n_els = 64
v_degree = 1
mesh = uw.mesh.Mesh(elementRes=(n_els,n_els))

# %%
# NL problem 
# Create solution functions
from underworld3.analytic import AnalyticSolNL_velocity, AnalyticSolNL_bodyforce, AnalyticSolNL_viscosity 
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
bnds = mesh.boundary
stokes.add_dirichlet_bc( sol_vel, [bnds.LEFT, bnds.RIGHT],  [0,1] )  # left/right: function, markers, components
stokes.add_dirichlet_bc( sol_vel, [bnds.TOP,  bnds.BOTTOM], [1, ] )  # top/bottom: function, markers, components

# %%
stokes.bodyforce = sol_bf
# do linear first to get reasonable starting place
print("Linear solve")
stokes.viscosity = 1.
stokes.solve()
# %%
print("Non Linear solve")
# get strainrate
sr = stokes.strainrate
# not sure if the following is needed as div_u should be zero
sr -= (stokes.div_u/mesh.dim)*sympy.eye(mesh.dim)
# second invariant of strain rate
inv2 = sr[0,0]**2 + sr[0,1]**2 + sr[1,0]**2 + sr[1,1]**2
inv2 = 1/2*inv2
# note we need to switch to sympy here
inv2 = sympy.sqrt(inv2.sfn)
alpha_by_two = 2/r0 - 2
stokes.viscosity = 2*eta0*inv2**alpha_by_two
stokes.solve(zero_init_guess=False)

# %%
with mesh.access():
    vel_soln_ana = stokes.u.data.copy()
    # %%
    for index, coord in enumerate(mesh.data):
        # interface to this is still yuck... 
        vel_soln_ana[index] = sol_vel.evalf(subs={mesh.N.x:coord[0], mesh.N.y:coord[1]}).to_matrix(mesh.N)[0:2]
    from numpy import linalg as LA
    l2diff = LA.norm(stokes.u.data - vel_soln_ana)
    print("Diff norm = {}".format(l2diff))
    if not np.allclose(l2diff, 0.0367,rtol=1.e-2):
        raise RuntimeError("Solve did not produce expected result.")
    if not np.allclose(stokes.u.data, vel_soln_ana, rtol=1.e-2):
        raise RuntimeError("Solve did not produce expected result.")

