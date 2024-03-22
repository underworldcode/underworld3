# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Non-linear Stokes benchmark

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

# +
# options = PETSc.Options()
# options["help"] = None

# +
# options["ksp_rtol"] = 1.0e-8
# options["snes_converged_reason"] = None
# options["snes_monitor"] = None
# # options["snes_monitor_short"] = None
# # options["snes_view"]=None
# options["snes_test_jacobian"] = None
# options["snes_rtol"] = 1.0e-7
# # options["snes_max_it"] = 1
# # options["snes_linesearch_monitor"] = None
# -


# %%
n_els = 32
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1/n_els
)

# %%
# NL problem
# Create solution functions
from underworld3.function.analytic import (
    AnalyticSolNL_velocity,
    AnalyticSolNL_bodyforce,
    AnalyticSolNL_viscosity,
)

x, y = mesh.X

r = mesh.r
eta0 = 1.0
n = 1
r0 = 1.5
params = (eta0, n, r0)
sol_bf_ijk = AnalyticSolNL_bodyforce(*params, *r)
sol_vel_ijk = AnalyticSolNL_velocity(*params, *r)

sol_bf = mesh.vector.to_matrix(sol_bf_ijk)
sol_vel = mesh.vector.to_matrix(sol_vel_ijk)
sol_visc = AnalyticSolNL_viscosity(*params, *r)

# debug - are problems just because there is no analytic solution module on mac
# The solNL case is a MMS force term (complicated) designed to produce a specific
# velocity field. This is a placeholder that just lets the non-linear problem run.

# +
# sol_vel = sympy.Matrix([0, 0])
# sol_bf = sympy.Matrix([0, sympy.cos(3 * sympy.pi * x) * sympy.cos(3 * sympy.pi * y)])
# sol_visc = 1
# -

# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

# +
# BC

stokes.add_dirichlet_bc(sol_vel, "Top", [0, 1])  
stokes.add_dirichlet_bc(sol_vel, "Bottom", [0, 1])
stokes.add_dirichlet_bc(sol_vel, "Left", [0, 1])
stokes.add_dirichlet_bc(sol_vel, "Right", [0, 1])
# -

stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_rtol"] = 1.0e-5


stokes._setup_pointwise_functions(verbose=True)
stokes._setup_discretisation(verbose=True)
stokes.dm.ds.view()

# +
# %%
# do linear first to get reasonable starting place
stokes.bodyforce = sol_bf
stokes.solve()
# %%
# get strainrate
sr = stokes.strainrate
# not sure if the following is needed as div_u should be zero
# sr -= (stokes.div_u / mesh.dim) * sympy.eye(mesh.dim)
# second invariant of strain rate

inv2 = sr[0, 0] ** 2 + sr[0, 1] ** 2 + sr[1, 0] ** 2 + sr[1, 1] ** 2
inv2 = 1 / 2 * inv2
inv2 = sympy.sqrt(inv2)
alpha_by_two = 2 / r0 - 2
# -
viscosity = 2 * eta0 * inv2**alpha_by_two

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity

stokes.penalty = 1.0
stokes.solve(zero_init_guess=False)

sol_vel.T

# %%
vdiff = stokes.u.sym - sol_vel.T
vdiff_dot_vdiff = uw.maths.Integral(mesh, vdiff.dot(vdiff)).evaluate()
v_dot_v = uw.maths.Integral(mesh, stokes.u.fn.dot(stokes.u.fn)).evaluate()

import math

rel_rms_diff = math.sqrt(vdiff_dot_vdiff / v_dot_v)
if rank == 0:
    print(f"RMS diff = {rel_rms_diff}")

if not np.allclose(rel_rms_diff, 0.00109, rtol=1.0e-2):
    raise RuntimeError("Solve did not produce expected result.")



