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

# # Stokes flow in a Spherical Domain
#
# Timing example for the archetype uw3 problem: Stokes flow in a sphere with free slip boundaries
#

# ## Computational script in python

# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
from petsc4py import PETSc
import mpi4py

import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)


# +
# Define the problem size
#      1  - ultra low res for automatic checking
#      2  - low res problem to play with this notebook
#      3  - medium resolution (be prepared to wait)
#      4  - highest resolution for parallel tests
#      5+ - v. high resolution (parallel only)

problem_size = uw.options.getInt("problem_size", default=2)


# +
visuals = 1
output_dir = "output"

# Some gmsh issues, so we'll use a pre-built one
r_o = 1.0
r_i = 0.5

Rayleigh = 1.0e6  # Doesn't actually matter to the solution pattern,

# + tags=[]
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
elif problem_size == 6:
    cell_size = 0.01
elif problem_size == 7:
    cell_size = 0.005

res = cell_size

expt_name = f"Stokes_Sphere_free_slip_{cell_size}"
# -


timing.reset()
timing.start()

# +
from pathlib import Path
from underworld3.coordinates import CoordinateSystemType

mesh_cache_file = f".meshes/uw_spherical_shell_ro{r_o}_ri{r_i}_csize{res}.msh.h5"
path = Path(mesh_cache_file)

if path.is_file():
    if uw.mpi.rank == 0:
        print(f"Re-using mesh: {mesh_cache_file}", flush=True)

    meshball = uw.discretisation.Mesh(
        mesh_cache_file,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        qdegree=2,
    )
else:
    meshball = uw.meshing.SphericalShell(
        radiusInner=r_i,
        radiusOuter=r_o,
        cellSize=cell_size,
        qdegree=2,
    )

meshball.dm.view()
# -

v_soln = uw.discretisation.MeshVariable(
    r"u", meshball, meshball.dim, degree=2, vtype=uw.VarType.VECTOR
)
p_soln = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)
meshr = uw.discretisation.MeshVariable(r"r", meshball, 1, degree=1)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x, y, z = meshball.CoordinateSystem.N
ra, l1, l2 = meshball.CoordinateSystem.xR

hw = 1000.0 / res
surface_fn_a = sympy.exp(-(((ra - r_o) / r_o) ** 2) * hw)
surface_fn = sympy.exp(-(((meshr.sym[0] - r_o) / r_o) ** 2) * hw)

base_fn_a = sympy.exp(-(((ra - r_i) / r_o) ** 2) * hw)
base_fn = sympy.exp(-(((meshr.sym[0] - r_i) / r_o) ** 2) * hw)

## Buoyancy (T) field

t_forcing_fn = 1.0 * (
    sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)


# +
# Create NS object

stokes = uw.systems.Stokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

stokes.tolerance = 1.0e-4
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_max_it"] = 1  # Only for timing examples
stokes.penalty = 0.1

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.Parameters.viscosity = 1

# thermal buoyancy force
buoyancy_force = Rayleigh * gravity_fn * t_forcing_fn * (1 - surface_fn) * (1 - base_fn)

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty_upper = v_soln.sym.dot(unit_rvec) * unit_rvec * surface_fn
free_slip_penalty_lower = v_soln.sym.dot(unit_rvec) * unit_rvec * base_fn

stokes.bodyforce = unit_rvec * buoyancy_force
stokes.bodyforce -= 100000 * (free_slip_penalty_upper + free_slip_penalty_lower)

stokes.saddle_preconditioner = 1.0

# Velocity boundary conditions
# stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Upper", (0,1,2))
# stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Lower", (0,1,2))

# +
with meshball.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2 + z**2), meshball.data, meshball.N
    )  # cf radius_fn which is 0->1

with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(
        t_forcing_fn, t_soln.coords, meshball.N
    ).reshape(-1, 1)
# -

stokes._setup_terms()
stokes.solve(zero_init_guess=True)

timing.print_table(display_fraction=0.999)


meshball.write_checkpoint(
    f"output/{expt_name}", meshUpdates=False, meshVars=[p_soln, v_soln]
)


# savefile = "output/{}_ts_{}.h5".format(expt_name, 0)
# meshball.save(savefile)
# v_soln.save(savefile)
# p_soln.save(savefile)
# meshball.generate_xdmf(savefile)
