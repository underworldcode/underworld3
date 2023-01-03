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

# # Rayleigh-Taylor in the sphere
#
#

# +
# Enable timing (before uw imports)

import os
os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

render = True


# +
lightIndex = 0
denseIndex = 1

viscosityRatio = 0.1

r_layer = 0.7
r_o = 1.0
r_i = 0.5


# +
# Define the problem size
#      1  - ultra low res for automatic checking
#      2  - low res problem to play with this notebook
#      3  - medium resolution (be prepared to wait)
#      4  - highest resolution for parallel tests
#      5+ - v. high resolution (parallel only)

problem_size = 1

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get('UW_TESTING_LEVEL')
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass
    
if problem_size <= 1: 
    cell_size = 0.30
elif problem_size == 2: 
    cell_size = 0.15
elif problem_size == 3: 
    cell_size = 0.05
elif problem_size == 4: 
    cell_size = 0.03
elif problem_size == 5: 
    cell_size = 0.02
elif problem_size >= 6: 
    cell_size = 0.01
    
res = cell_size
# -

Rayleigh = 1.0e6 / (r_o - r_i) ** 3
offset = 0.5 * res

# +
from underworld3 import timing

timing.reset()
timing.start()
# -

mesh = uw.meshing.SphericalShell(
    radiusInner=r_i, radiusOuter=r_o, cellSize=res, qdegree=2
)



v_soln = uw.discretisation.MeshVariable(r"U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(r"P", mesh, 1, degree=1)
meshr = uw.discretisation.MeshVariable(r"r", mesh, 1, degree=1)


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable(r"\cal{L}", swarm, proxy_degree=1, num_components=1)
swarm.populate(fill_param=2)


with swarm.access(material):
    r = np.sqrt(
        swarm.particle_coordinates.data[:, 0] ** 2
        + swarm.particle_coordinates.data[:, 1] ** 2
        + (swarm.particle_coordinates.data[:, 2] - offset) ** 2
    )

    material.data[:, 0] = r - r_layer

# +

# Some useful coordinate stuff

x, y, z = mesh.CoordinateSystem.X
ra, l1, l2 = mesh.CoordinateSystem.xR

hw = 1000.0 / res
surface_fn_a = sympy.exp(-(((ra - r_o) / r_o) ** 2) * hw)
surface_fn = sympy.exp(-(((meshr.sym[0] - r_o) / r_o) ** 2) * hw)

base_fn_a = sympy.exp(-(((ra - r_i) / r_o) ** 2) * hw)
base_fn = sympy.exp(-(((meshr.sym[0] - r_i) / r_o) ** 2) * hw)


# +

density = sympy.Piecewise((0.0, material.sym[0] < 0.0), (1.0, True))
display(density)

viscosity = sympy.Piecewise((1.0, material.sym[0] < 0.0), (viscosityRatio, True))
display(viscosity)


# +
stokes = uw.systems.Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

stokes.tolerance = 1.0e-4 
stokes.petsc_options["ksp_monitor"] = None

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity

# buoyancy (magnitude)
buoyancy = Rayleigh * density # * (1 - surface_fn) * (1 - base_fn)

unit_vec_r = mesh.CoordinateSystem.unit_e_0

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty_upper = v_soln.sym.dot(unit_vec_r) * unit_vec_r * surface_fn
free_slip_penalty_lower = v_soln.sym.dot(unit_vec_r) * unit_vec_r * base_fn

stokes.bodyforce = -unit_vec_r * buoyancy
stokes.bodyforce -= 1000000 * (free_slip_penalty_upper + free_slip_penalty_lower)

stokes.saddle_preconditioner = 1 / viscosity

# -

with mesh.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2 + z**2), mesh.data, mesh.N
    )  # cf radius_fn which is 0->1


stokes.solve(zero_init_guess=True)

t_step = 0

# +
# Update in time

expt_name = "output/swarm_rt_sph"

for step in range(0, 3):

    stokes.solve(zero_init_guess=False)
    delta_t = 2.0 * stokes.estimate_dt()

    # update swarm / swarm variables

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))

    # advect swarm
    swarm.advection(v_soln.sym, delta_t)

    t_step += 1

# -


savefile = "output/swarm_rt.h5".format(step)
mesh.save(savefile)
v_soln.save(savefile)
# mesh.generate_xdmf(savefile)

timing.print_table(display_fraction=0.999)

