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
# Stokes Sinker - Multiple Materials

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

This is the notorious "Stokes sinker" problem in which we have a dense and
"rigid" (highly viscous) blob sinking in a low-viscosity fluid. This
combination of high velocity and low strain rate is challenging for
iterative solvers and there is a limit to the viscosity jump that can be
introduced before the solvers fail to converge.

## Key Concepts

- **IndexSwarmVariable**: Automatically generates masks for discrete level
  values (integers representing material indices)
- **Material masks**: Orthogonal in the sense that M^i * M^j = 0 if i != j,
  and complete: sum(M^i) = 1 at all points
- **Penalty method**: For free-slip conditions on Stokes equations

## Parameters

- `uw_problem_size`: Controls mesh resolution (1=ultra low, 6=benchmark)
- `uw_viscosity_contrast`: Viscosity ratio between sinker and background
- `uw_density_contrast`: Density ratio between sinker and background
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Sinker.py -uw_problem_size 4
python Ex_Stokes_Sinker.py -uw_viscosity_contrast 1e4
```
"""

# %%
# Problem parameters - editable here or via command line
params = uw.Params(
    uw_problem_size = 2,              # 1-6: resolution level
    uw_viscosity_contrast = 1.0e6,    # Viscosity ratio (sinker/background)
    uw_density_contrast = 10.0,       # Density ratio (sinker/background)
    uw_sphere_radius = 0.1,           # Sinker radius
    uw_sphere_x = 0.0,                # Sinker center x
    uw_sphere_y = 0.7,                # Sinker center y
    uw_n_steps = 15,                  # Number of time steps
)

# Map problem_size to mesh resolution
resolution_map = {1: 8, 2: 16, 3: 32, 4: 48, 5: 64, 6: 128}
res = resolution_map.get(params.uw_problem_size, 16)

os.environ["UW_TIMING_ENABLE"] = "1"

# Create output directory
if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)

# %% [markdown]
"""
## Material Properties
"""

# %%
# Sphere geometry
sphereRadius = params.uw_sphere_radius
sphereCentre = (params.uw_sphere_x, params.uw_sphere_y)

# Material indices
materialLightIndex = 0
materialHeavyIndex = 1

# Viscosities
viscBG = 1.0
viscSphere = params.uw_viscosity_contrast

# Densities
densityBG = 1.0
densitySphere = params.uw_density_contrast

expt_name = f"output/sinker_eta{viscSphere:.0e}_rho{densitySphere:.0f}_res{res}"

# Tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1] - sphereRadius

# %% [markdown]
"""
## Mesh Generation
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
)

# %% [markdown]
"""
## Stokes Solver Setup
"""

# %%
stokes = uw.systems.Stokes(mesh)

v = stokes.Unknowns.u
p = stokes.Unknowns.p

# Penalty method for incompressibility
stokes.penalty = 1.0

# Boundary conditions: free-slip (zero normal velocity)
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

# %% [markdown]
"""
## Swarm and Material Definition
"""

# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_continuous=False, proxy_degree=1
)
swarm.populate(fill_param=4)

# Define the sinker blob
blob = np.array([[sphereCentre[0], sphereCentre[1], sphereRadius, 1]])

with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(blob.shape[0]):
        cx, cy, r, m = blob[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# %%
# Tracer particle for tracking sinker position
tracer = np.zeros(shape=(1, 2))
tracer[:, 0], tracer[:, 1] = x_pos, y_pos

# Material-dependent properties using masks
density = densityBG * material.sym[0] + densitySphere * material.sym[1]
viscosity = viscBG * material.sym[0] + viscSphere * material.sym[1]

# %% [markdown]
"""
## Constitutive Model
"""

# %%
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * density])

# %% [markdown]
"""
## Visualization Setup
"""

# %%
render = True

if uw.mpi.size == 1:
    import pyvista as pv

    pl = pv.Plotter(notebook=True)
    pl.camera.position = (1.1, 1.5, 0.0)
    pl.camera.focal_point = (0.2, 0.3, 0.3)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(1.4)

def plot_T_mesh(filename):
    if not render or uw.mpi.size != 1:
        return

    import pyvista as pv
    import underworld3.visualisation

    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
    point_cloud = underworld3.visualisation.swarm_to_pv_cloud(swarm)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl.clear()
    pl.add_mesh(pvmesh, "Black", "wireframe")
    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.5,
    )
    pl.screenshot(
        filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False
    )

# %% [markdown]
"""
## Solver Configuration
"""

# %%
snes_rtol = 1.0e-6
stokes.tolerance = snes_rtol
stokes.petsc_options["ksp_monitor"] = None

# %% [markdown]
"""
## Time Stepping
"""

# %%
nstep = params.uw_n_steps
step = 0
time = 0.0

tSinker = np.zeros(nstep)
ySinker = np.zeros(nstep)

# %%
# Initial solve with timing
from underworld3 import timing

timing.reset()
timing.start()
stokes.solve(zero_init_guess=True)
timing.print_table()

# %%
while step < nstep:
    # Track sinker position
    ymin = tracer[:, 1].min()
    ySinker[step] = ymin
    tSinker[step] = time

    # Estimate timestep
    dt = stokes.estimate_dt()
    uw.pprint(f"dt = {dt}")

    # Advect swarm
    swarm.advection(stokes.u.sym, dt, corrector=True)

    # Solve Stokes
    stokes.solve(zero_init_guess=False)

    # Output
    if uw.mpi.size == 1:
        print(f"Step: {str(step).rjust(3)}, time: {time:6.2f}, tracer:  {ymin:6.2f}")
        plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    mesh.write_timestep("stokesSinker", meshUpdates=False, meshVars=[p, v], index=step)

    step += 1
    time += dt

# %% [markdown]
"""
## Results Analysis
"""

# %%
if uw.mpi.size == 1:
    uw.pprint("Initial position: t = {0:.3f}, y = {1:.3f}".format(tSinker[0], ySinker[0]))
    uw.pprint("Final position:   t = {0:.3f}, y = {1:.3f}".format(tSinker[nstep - 1], ySinker[nstep - 1]))

    import matplotlib.pyplot as pyplot

    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tSinker, ySinker)
    ax.set_xlabel("Time")
    ax.set_ylabel("Sinker position")

# %% [markdown]
"""
## Final Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation

    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.X.coords)

    swarm_points = underworld3.visualisation.swarm_to_pv_cloud(swarm)
    swarm_points.point_data["M"] = uw.visualisation.scalar_fn_to_pv_points(
        swarm_points, material.visMask()
    )

    velocity_points = underworld3.visualisation.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(
        velocity_points, v.sym
    )

    pvstream = pvmesh.streamlines_from_source(
        swarm_points,
        vectors="V",
        integration_direction="both",
        max_steps=10,
        surface_streamlines=True,
        max_step_length=0.05,
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, "Black", "wireframe")
    streamlines = pl.add_mesh(pvstream, opacity=0.25)
    streamlines.SetVisibility(False)

    pl.add_mesh(
        swarm_points,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="M",
        use_transparency=False,
        point_size=2.0,
        opacity=0.5,
        show_scalar_bar=False,
    )

    arrows = pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=3.0,
        opacity=0.33,
        show_scalar_bar=False,
    )

    def toggle_streamlines(flag):
        streamlines.SetVisibility(flag)

    def toggle_arrows(flag):
        arrows.SetVisibility(flag)

    pl.add_checkbox_button_widget(toggle_streamlines, value=False, size=10, position=(10, 20))
    pl.add_checkbox_button_widget(toggle_arrows, value=False, size=10, position=(30, 20))

    pl.show(cpos="xy")
