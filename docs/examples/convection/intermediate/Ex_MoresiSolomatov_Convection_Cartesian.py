# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Moresi-Solomatov Stagnant Lid Convection

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Temperature-dependent viscosity convection with viscoplastic rheology following
Moresi & Solomatov (1995). This creates stagnant lid convection where a cold,
high-viscosity lid forms at the surface.

## Key Concepts

- **Stagnant lid**: High-viscosity lid at surface due to T-dependent viscosity
- **Viscoplastic rheology**: Combined viscous and plastic behavior
- **Yield stress**: Depth-dependent yield strength
- **Restart capability**: Can continue from previous timestep

## Mathematical Formulation

Temperature-dependent viscosity:
$$\\eta_T = \\eta_0 \\exp(-C \\cdot T)$$

Where C controls the viscosity contrast across the temperature range.

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_ra_expt`: log10(Rayleigh number)
- `uw_visc_expt`: log10(viscosity contrast)
- `uw_width`: Domain aspect ratio
- `uw_max_steps`: Maximum time steps
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import underworld3 as uw
import numpy as np
import sympy
import os

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_MoresiSolomatov_Convection_Cartesian.py -uw_ra_expt 7
python Ex_MoresiSolomatov_Convection_Cartesian.py -uw_visc_expt 5
python Ex_MoresiSolomatov_Convection_Cartesian.py -uw_resolution 30
```
"""

# %%
params = uw.Params(
    uw_resolution = 15,              # Mesh resolution
    uw_ra_expt = 7,                  # log10(Rayleigh number)
    uw_visc_expt = 4.5,              # log10(viscosity contrast)
    uw_width = 1,                    # Domain width (aspect ratio)
    uw_max_steps = 201,              # Maximum time steps
    uw_restart_step = -1,            # Restart from step (-1 = fresh start)
)

# Derived parameters
rayleigh_number = 10 ** params.uw_ra_expt
visc_contrast = 10 ** params.uw_visc_expt

expt_name = f"Ra1e{params.uw_ra_expt}_visc{params.uw_visc_expt}_res{params.uw_resolution}"
output_dir = os.path.join("output", f"cartesian_{params.uw_width}x1", f"Ra1e{params.uw_ra_expt}")

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
    cellSize=1 / params.uw_resolution,
    minCoords=(0.0, 0.0),
    maxCoords=(params.uw_width, 1.0),
    degree=1,
    qdegree=3,
    regular=False,
)

x, y = meshbox.CoordinateSystem.X
y_vector = meshbox.CoordinateSystem.unit_e_1

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
eta_soln = uw.discretisation.MeshVariable("eta_n", meshbox, 1, degree=1)

# %% [markdown]
"""
## Stokes Solver with Viscoplastic Rheology

Temperature-dependent viscosity with yield stress.
"""

# %%
stokes = uw.systems.Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

# Temperature-dependent viscosity
C = sympy.log(visc_contrast)
visc_fn = sympy.exp(-C * t_soln.sym[0]) * visc_contrast

stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn

stokes.tolerance = 1e-6
stokes.penalty = 0.0

# Free-slip boundary conditions
stokes.add_essential_bc((None, 0.0), "Top")
stokes.add_essential_bc((None, 0.0), "Bottom")
stokes.add_essential_bc((0.0, None), "Left")
stokes.add_essential_bc((0.0, None), "Right")

# Buoyancy force
stokes.bodyforce = y_vector * rayleigh_number * t_soln.sym[0]

# Solver options
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
"""
## Advection-Diffusion Solver
"""

# %%
adv_diff = uw.systems.AdvDiffusion(
    meshbox,
    u_Field=t_soln,
    V_fn=v_soln,
    order=1,
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1

# Temperature boundary conditions
adv_diff.add_dirichlet_bc(+1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

# %% [markdown]
"""
## Viscosity Projection
"""

# %%
eta_solver = uw.systems.Projection(meshbox, eta_soln)
eta_solver.uw_function = stokes.constitutive_model.viscosity
eta_solver.smoothing = 0.0

# %% [markdown]
"""
## Initial Conditions

Thermal boundary layer structure with small perturbation.
"""

# %%
delta = 0.1
aveT = 0.5 - 0.5 * (sympy.tanh(2 * y / delta) - sympy.tanh(2 * (1 - y) / delta))

init_t = (
    0.02 * sympy.cos(10.0 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    + aveT
)

with meshbox.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(init_t, t_soln.coords).reshape(-1, 1)

# Handle restart if specified
if params.uw_restart_step != -1:
    print(f"Reading step {params.uw_restart_step}")
    t_soln.read_timestep(
        data_filename=expt_name,
        data_name="T",
        index=params.uw_restart_step,
        outputPath=output_dir,
    )

# %% [markdown]
"""
## Initial Solve (Linear)
"""

# %%
stokes.constitutive_model.Parameters.yield_stress = sympy.oo
stokes.solve()
eta_solver.solve()

uw.pprint("Linear solve complete")

# %% [markdown]
"""
## Enable Nonlinear Yielding

Depth-dependent yield stress.
"""

# %%
stokes.constitutive_model.Parameters.yield_stress = 1e7 + 1e7 * (1 - y)
stokes.constitutive_model.Parameters.shear_viscosity_min = 1.0
stokes.solve(zero_init_guess=False)
uw.pprint("NL Solve 1")

stokes.constitutive_model.Parameters.yield_stress = 1e5 + 1e7 * (1 - y)
stokes.solve(zero_init_guess=False)
uw.pprint("NL Solve 2")

eta_solver.solve()

# %% [markdown]
"""
## Time Evolution
"""

# %%
if params.uw_restart_step == -1:
    timestep = 0
else:
    timestep = params.uw_restart_step

elapsed_time = 0.0

for step in range(0, params.uw_max_steps):
    stokes.solve(zero_init_guess=False)
    eta_solver.solve()

    delta_t = 2.0 * adv_diff.estimate_dt()
    delta_ta = stokes.estimate_dt()

    adv_diff.solve(timestep=delta_t)

    # Stats
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print(f"Timestep {timestep}, dt {delta_t:.4e}, dta {delta_ta:.4e}, t {elapsed_time:.4e}")

    # Save output
    meshbox.write_timestep(
        filename=expt_name,
        index=timestep,
        outputPath=output_dir,
        meshVars=[v_soln, p_soln, t_soln, eta_soln],
    )

    timestep += 1
    elapsed_time += delta_t

# %%
print(f"Simulation complete: {expt_name}")
