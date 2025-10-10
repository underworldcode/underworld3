# %% [markdown]
"""
# 🔬 MoresiSolomatov Convection Cartesian

**PHYSICS:** convection  
**DIFFICULTY:** intermediate  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# # Rayleigh-Bénard Convection in an flat layer
#
#

# +
import underworld3 as uw

import os
import numpy as np
import sympy

from underworld3.systems import Stokes
from underworld3 import function
import mpi4py



# +
# The problem setup

# mesh parameters

width = 1
visc_expt = 4.5
ra_expt=7
resolution=15
expt_desc="tau_y_ii"
restart_step=550

# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_resolution=0.1 etc

ra_expt = uw.options.getReal("ra_expt", default=ra_expt)
visc_expt = uw.options.getReal("visc_expt", default=visc_expt)
width = uw.options.getInt("width", default=width)
resolution = uw.options.getInt("resolution", default=resolution)
resolution_in = uw.options.getInt("resolution_in", default=-1)
max_steps = uw.options.getInt("max_steps", default=201)
restart_step = uw.options.getInt("restart_step", default=restart_step)
expt_desc = uw.options.getString("expt_description", default=expt_desc)

if expt_desc != "":
    expt_desc += "_"

uw.pprint(f"Restarting from step {restart_step}")

if resolution_in == -1:
    resolution_in = resolution

# How that works

rayleigh_number = uw.function.expression(
    r"\textrm{Ra}", pow(10, ra_expt), "Rayleigh number"  # / (r_o-r_i)**3 )

old_expt_name = f"{expt_desc}Ra1e{ra_expt}_visc{visc_expt}_res{resolution_in}"
expt_name = f"{expt_desc}Ra1e{ra_expt}_visc{visc_expt}_res{resolution}"
output_dir = os.path.join("output", f"cartesian_{width}x1", f"Ra1e{ra_expt}")

os.makedirs(output_dir, exist_ok=True)


# +
## Set up the mesh geometry / discretisation

meshbox = uw.meshing.UnstructuredSimplexBox(
    cellSize=1 / resolution,
    minCoords=(0.0, 0.0),
    maxCoords=(width, 1.0),
    degree=1,
    qdegree=3,
    regular=False)

x, y = meshbox.CoordinateSystem.X
x_vector = meshbox.CoordinateSystem.unit_e_0
y_vector = meshbox.CoordinateSystem.unit_e_1

# -

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
eta_soln =  uw.discretisation.MeshVariable("\eta_n", meshbox, 1, degree=1)

# +
# Create solver to solver the momentum equation (Stokes flow)

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln)

C = uw.function.expression("C", sympy.log(sympy.Pow(10, sympy.sympify(visc_expt))))
visc_fn = uw.function.expression(
    r"\eta",
    sympy.exp(-C.sym * t_soln.sym[0]) * sympy.Pow(10, sympy.sympify(visc_expt)),
    "1")

stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn

stokes.tolerance = 1e-6
stokes.penalty = 0.0

# penalty = max(1000000, 10*rayleigh_number.sym)

# Prevent flow crossing the boundaries

stokes.add_essential_bc((None, 0.0), "Top")
stokes.add_essential_bc((None, 0.0), "Bottom")
stokes.add_essential_bc((0.0, None), "Left")
stokes.add_essential_bc((0.0, None), "Right")

stokes.bodyforce = y_vector * rayleigh_number * t_soln.sym[0]

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")


# +
# Create solver for the energy equation (Advection-Diffusion of temperature)

adv_diff = uw.systems.AdvDiffusion(
    meshbox,
    u_Field=t_soln,
    V_fn=v_soln,
    order=1,
    verbose=False)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1

## Boundary conditions for this solver

adv_diff.add_dirichlet_bc(+1.0, "Bottom")
adv_diff.add_dirichlet_bc(-0.0, "Top")
# -

eta_solver = uw.systems.Projection(meshbox, eta_soln)
eta_solver.uw_function = stokes.constitutive_model.viscosity
eta_solver.smoothing = 0.0

# +
# The advection / diffusion equation is an initial value problem
# We set this up with an approximation to the ultimate boundary
# layer structure (you need to provide delta, the TBL thickness)
#
# Add some perturbation and try to offset this on the different boundary
# layers to avoid too much symmetry

delta = 0.1
aveT = 0.5 - 0.5 * (sympy.tanh(2 * y / delta) - sympy.tanh(2 * (1 - y) / delta))

init_t = (
    0.0 * sympy.cos(0.5 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    + 0.02 * sympy.cos(10.0 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    + aveT
)

with meshbox.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(init_t, t_soln.coords).reshape(-1, 1)

if restart_step != -1:
    print(f"Reading step {restart_step} at resolution {resolution_in}")
    t_soln.read_timestep(
        data_filename=f"{old_expt_name}",
        data_name="T",
        index=restart_step,
        outputPath=output_dir)

# +
# linear solve first

stokes.constitutive_model.Parameters.yield_stress = sympy.oo
stokes.solve()
eta_solver.solve()



# +
# now add non-linear effects 

stokes.constitutive_model.Parameters.yield_stress = 1e7 + 1e7 * (1-y) 
stokes.constitutive_model.Parameters.shear_viscosity_min = 1.0
stokes.solve(zero_init_guess=False)

uw.pprint("NL Solve 1")

stokes.constitutive_model.Parameters.yield_stress = 1e5 + 1e7 * (1-y) 
stokes.constitutive_model.Parameters.shear_viscosity_min = 1.0
stokes.solve(zero_init_guess=False)

uw.pprint("NL Solve 2")

eta_solver.solve()


# +
if restart_step == -1:
    timestep = 0
else:
    timestep = restart_step

elapsed_time = 0.0

# +
# Convection model / update in time

output = os.path.join(output_dir, expt_name)

for step in range(0, max_steps):

    stokes.solve(zero_init_guess=False)
    eta_solver.solve()

    delta_t = 2.0 * adv_diff.estimate_dt()
    delta_ta = stokes.estimate_dt()

    adv_diff.solve(timestep=delta_t)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print(
            f"Timestep {timestep}, dt {delta_t:.4e}, dta {delta_ta:.4e}, t {elapsed_time:.4e} "
        )

    meshbox.write_timestep(
        filename=f"{expt_name}",
        index=timestep,
        outputPath=output_dir,
        meshVars=[v_soln, p_soln, t_soln, eta_soln])

    timestep += 1
    elapsed_time += delta_t
# +



# -



