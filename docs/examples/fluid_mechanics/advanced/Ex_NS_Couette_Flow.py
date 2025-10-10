# %% [markdown]
"""
# 🎓 NS Couette Flow

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** advanced  
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
# %% [markdown]
# # Navier Stokes benchmark: Couette flow
# By: Juan Carlos Graciosa 
# 

# %%
import os

import petsc4py
import underworld3 as uw

import numpy as np
import sympy
import argparse
import pickle

#parser = argparse.ArgumentParser()
#parser.add_argument('-i', "--idx", type=int, required=True)
#parser.add_argument('-p', "--prev", type=int, required=True) # set to 0 if no prev_res, 1 if there is
#args = parser.parse_args()

#idx = args.idx
#prev = args.prev

idx = 0
prev = 0

# %%
resolution = 16
refinement = 0
save_every = 200
maxsteps   = 1
Cmax       = 1      # target Courant number

order = 1           # solver order
tol = 1e-12         # solver tolerance (sets atol and rtol)

mesh_type = "Pirr" # or Preg, Pirr, Quad
qdeg = 3
Vdeg = 3
Pdeg = Vdeg - 1
Pcont = False

# %%
expt_name = f"Couette-res{resolution}-order{order}-{mesh_type}"

outfile = f"{expt_name}_run{idx}"
outdir = f"./{expt_name}"

# %%
if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    infile = f"{expt_name}_run{prev_idx}"

if uw.mpi.rank == 0 and uw.mpi.size > 1:
    os.makedirs(f"{outdir}", exist_ok=True)

# %%
width   = 4.
height  = 1.
vel     = 1.

fluid_rho   = 1.
kin_visc    = 1.
dyn_visc    = fluid_rho * kin_visc

# %%
minX, maxX = -0.5 * width, 0.5 * width
minY, maxY = -0.5 * height, 0.5 * height

uw.pprint("min X, max X:", minX, maxX)
    print("min Y, max Y:", minY, maxY)
    print("kinematic viscosity: ", kin_visc)
    print("fluid density: ", fluid_rho)
    print("dynamic viscosity: ", kin_visc * fluid_rho)

# %%
# cell size calculation
if mesh_type == "Preg":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = 1 / resolution, qdegree = qdeg, regular = True)
elif mesh_type == "Pirr":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = 1 / resolution, qdegree = qdeg, regular = False)
elif mesh_type == "Quad":
    meshbox = uw.meshing.StructuredQuadBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), elementRes = (width * resolution, height * resolution), qdegree = qdeg, regular = False)

# %%
meshbox.dm.view()

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree = Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree = Pdeg, continuous = Pcont)

# %%
if infile is None:
    pass
else:
    uw.pprint(f"Reading: {infile}")

    v_soln.read_timestep(data_filename = infile, data_name = "U", index = maxsteps, outputPath = outdir)
    p_soln.read_timestep(data_filename = infile, data_name = "P", index = maxsteps, outputPath = outdir)

# %%
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

navier_stokes = uw.systems.NavierStokesSLCN(
    meshbox,
    velocityField = v_soln,
    pressureField = p_soln,
    rho = fluid_rho,
    verbose = False,
    order = order)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Constant visc
navier_stokes.constitutive_model.Parameters.viscosity = dyn_visc

navier_stokes.penalty = 0
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc((vel, 0.0), "Top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
# left and right are open

navier_stokes.tolerance = tol

# %%
# navier_stokes.petsc_options["snes_monitor"] = None
# navier_stokes.petsc_options["snes_converged_reason"] = None
# navier_stokes.petsc_options["snes_monitor_short"] = None
# navier_stokes.petsc_options["ksp_monitor"] = None

# navier_stokes.petsc_options["snes_type"] = "newtonls"
# navier_stokes.petsc_options["ksp_type"] = "fgmres"

# navier_stokes.petsc_options["snes_max_it"] = 50
# navier_stokes.petsc_options["ksp_max_it"] = 50

navier_stokes.petsc_options["snes_monitor"] = None
navier_stokes.petsc_options["ksp_monitor"] = None

navier_stokes.petsc_options["snes_type"] = "newtonls"
navier_stokes.petsc_options["ksp_type"] = "fgmres"

navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %%
# set the timestep
# for now, set it to be constant
delta_x = meshbox.get_min_radius()
max_vel = vel

delta_t = Cmax*delta_x/max_vel

uw.pprint(f"Min radius: {delta_x}")
    print("Timestep used:", delta_t)

# %%
ts = 0
timeVal =  np.zeros(maxsteps + 1) * np.nan      # time values
elapsed_time = 0.0

# %%
for step in range(0, maxsteps):

    uw.pprint(f"Timestep: {step}")

    navier_stokes.solve(timestep = delta_t, zero_init_guess=True)

    elapsed_time += delta_t
    timeVal[step] = elapsed_time

    uw.pprint("Timestep {}, t {}, dt {}".format(ts, elapsed_time, delta_t))
    
    if ts % save_every == 0 and ts > 0:
        meshbox.write_timestep(
            outfile,
            meshUpdates=True,
            meshVars=[p_soln, v_soln],
            outputPath=outdir,
            index =ts)

        with open(outdir + f"/{outfile}.pkl", "wb") as f:
            pickle.dump([timeVal], f)

    # update timestep
    ts += 1

# save after all iterations
meshbox.write_timestep(
    outfile,
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=outdir,
    index =maxsteps)

with open(outdir + f"/{outfile}.pkl", "wb") as f:
    pickle.dump([timeVal], f)


