# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Cylindrical Stokes (Cartesian formulation)
#
#

# +
import os

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = False

# +
# Define the problem size

#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 3

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass

r_o = 1.0
r_i = 0.5

if problem_size <= 1:
    res = 0.2
elif problem_size == 2:
    res = 0.1
elif problem_size == 3:
    res = 0.05
elif problem_size == 4:
    res = 0.025
elif problem_size == 5:
    res = 0.01
elif problem_size >= 6:
    res = 0.005

# +
from underworld3 import timing

timing.reset()
timing.start()
# -

meshball = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res)


meshball.dm.view()

# +
# Test that the second one is skipped

v_soln = uw.discretisation.MeshVariable(r"\mathbf{u}", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable(r"p_c", meshball, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=3)
maskr = uw.discretisation.MeshVariable("r", meshball, 1, degree=1)


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = meshball.CoordinateSystem.xR[0]
radius_fn = maskr.sym[0]
unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = 1  # radius_fn / r_o

# Some useful coordinate stuff

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR

Rayleigh = 1.0e5

hw = 2000.0 / res
surface_fn = sympy.exp(-((radius_fn - r_o) ** 2) * hw)
base_fn = sympy.exp(-((radius_fn - r_i) ** 2) * hw)


# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
    meshball.dim
)
stokes.constitutive_model.Parameters.viscosity = 1

# There is a null space if there are no fixed bcs, so we'll do this:

if not free_slip_upper:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))

stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# -


pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

t_init = sympy.cos(3 * th)

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(
        t_init, coords=t_soln.coords, coord_sys=meshball.N
    )
    # print(t_soln.data.min(), t_soln.data.max())

with meshball.access(maskr):
    maskr.data[:, 0] = uw.function.evaluate(
        r, coords=maskr.coords, coord_sys=meshball.N
    )

# +
# I = uw.maths.Integral(meshball, surface_fn)
# s_norm = I.evaluate()
# print(s_norm)

# I.fn = base_fn
# b_norm = I.evaluate()
# print(b_norm)
# +
buoyancy_force = Rayleigh * gravity_fn * t_init
if free_slip_upper:
    buoyancy_force -= 1.0e6 * v_soln.sym.dot(unit_rvec) * surface_fn # / s_norm
    buoyancy_force -= 1.0e6 * v_soln.sym.dot(unit_rvec) * base_fn # / b_norm

stokes.bodyforce = unit_rvec * buoyancy_force

# This may help the solvers - penalty in the preconditioner
stokes.saddle_preconditioner = 1.0

stokes.petsc_options["ksp_monitor"] = None
stokes.tolerance = 1.0e-4


# +
# stokes.petsc_options.getAll()
# -

stokes.solve(zero_init_guess=True)

# Pressure at mesh nodes
pressure_solver.solve()

# +
timing.print_table(display_fraction=0.999)

if uw.mpi.rank==0:
    print("", flush=True)
