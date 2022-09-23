# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np

options = PETSc.Options()
# options["help"] = None

# options["pc_type"]  = "svd"

options["ksp_rtol"] = 1.0e-6
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
mesh = uw.discretisation.Spherical(refinements=4)
# %%
v_degree = 1
stokes = Stokes(mesh, u_degree=v_degree, p_degree=v_degree - 1)

# %%
# Set some things
import sympy
from sympy import Piecewise

# get spherical coord system
P = mesh.P
r = P.r  # radial direction
t = P.t  # theta direction
p = P.p  # phi direction

eta_0 = 1.0
r_c = 0.5
t_c = sympy.pi / 4
p_c = sympy.pi / 8

f_0 = -1.0
stokes.viscosity = 1.0
stokes.bodyforce = (
    Piecewise(
        (
            f_0,
            (r < r_c) & (sympy.Abs(t) < t_c) & (sympy.Abs(p) < p_c),
        ),
        (0.0, True),
    )
    * P.R
)
# free slip.
# note with petsc we always need to provide a vector of correct cardinality.
bnds = mesh.boundary
stokes.add_dirichlet_bc((0.0, 0.0), bnds.OUTER, [0, 1])  # function, boundaries, component list

# %%
# Solve time
stokes.solve()

# # %%
# import underworld as uw2
# solC = uw2.function.analytic.SolC()

# # %%
# vel_soln_analytic = solC.fn_velocity.evaluate(mesh.data).flatten()

# # %%
# vel_soln  = stokes.u_local.array
# pres_soln = stokes.p_local.array
# # %%
# from numpy import linalg as LA
# print("Diff norm = {}".format(LA.norm(vel_soln - vel_soln_analytic)))

# # %%
# if not np.allclose(vel_soln, vel_soln_analytic,rtol=1.e-2):
#     raise RuntimeError("Solve did not produce expected result.")
