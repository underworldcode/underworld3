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
# Rayleigh-Taylor Instability - Swarm Materials

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

Rayleigh-Taylor instability simulated using swarm-based material tracking.
A dense layer overlies a light layer, creating gravitational instability.
Uses IndexSwarmVariable for discrete material tracking with automatic mask
generation.

## Key Concepts

- **IndexSwarmVariable**: Discrete material indices on swarm with automatic masks
- **Material masks**: Orthogonal masks where M^i * M^j = 0 for i != j
- **Rayleigh-Taylor instability**: Dense fluid sinking into less dense fluid
- **van Keken benchmark**: Standard RT setup from van Keken et al. (1997)
- **Swarm advection**: Lagrangian tracking of material interfaces

## Mathematical Formulation

Initial interface perturbation:
$$y = y_0 + A \cos(k x)$$

where $k = 2\pi / \lambda$ is the wavenumber.

## Parameters

- `uw_cell_size`: Mesh cell size
- `uw_particle_fill`: Swarm particle fill parameter
- `uw_viscosity_ratio`: Viscosity contrast between materials
- `uw_n_steps`: Number of time steps
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Swarm_RT_Cartesian.py -uw_cell_size 0.02
python Ex_Stokes_Swarm_RT_Cartesian.py -uw_viscosity_ratio 10.0
python Ex_Stokes_Swarm_RT_Cartesian.py -uw_n_steps 250
```
"""

# %%
params = uw.Params(
    uw_cell_size = 1.0 / 32,       # Mesh cell size
    uw_particle_fill = 7,           # Swarm fill parameter
    uw_viscosity_ratio = 1.0,       # Viscosity contrast between materials
    uw_n_steps = 2,                 # Number of time steps (set higher for full run)
    uw_max_dt = 10.0,               # Maximum time step
    uw_amplitude = 0.02,            # Initial perturbation amplitude
    uw_offset = 0.2,                # Interface offset from bottom
)

render = True

# %% [markdown]
"""
## Physical Constants

Following van Keken et al. (1997) benchmark setup.
"""

# %%
lightIndex = 0
denseIndex = 1

boxLength = 0.9142
boxHeight = 1.0
amplitude = params.uw_amplitude
offset = params.uw_offset
model_end_time = 300.0

# Material perturbation from van Keken et al. 1997
wavelength = 2.0 * boxLength
k = 2.0 * np.pi / wavelength

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(boxLength, boxHeight),
    cellSize=params.uw_cell_size,
    regular=False,
    qdegree=2,
)

x, y = meshbox.CoordinateSystem.X

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
m_cont = uw.discretisation.MeshVariable("M_c", meshbox, 1, degree=1, continuous=True)

# %% [markdown]
"""
## Swarm and Material Index

The IndexSwarmVariable automatically generates orthogonal masks for each material.
"""

# %%
swarm = uw.swarm.Swarm(mesh=meshbox)
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_degree=1, proxy_continuous=False
)
swarm.populate(fill_param=int(params.uw_particle_fill))

# %% [markdown]
"""
## Initial Material Distribution

Set the interface with cosine perturbation.
"""

# %%
with swarm.access(material):
    material.data[...] = 0

with swarm.access(material):
    perturbation = offset + amplitude * np.cos(
        k * swarm._particle_coordinates.data[:, 0]
    )
    material.data[:, 0] = np.where(
        perturbation > swarm._particle_coordinates.data[:, 1], lightIndex, denseIndex
    )

# %% [markdown]
"""
## Material Properties

Density and viscosity defined per material using masks.
"""

# %%
mat_density = np.array([0, 1])  # lightIndex, denseIndex
density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([params.uw_viscosity_ratio, 1])
viscosity = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = uw.systems.Stokes(meshbox, velocityField=v_soln, pressureField=p_soln)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity

stokes.bodyforce = sympy.Matrix([0, -density])
stokes.saddle_preconditioner = 1.0 / viscosity

# Free-slip boundary conditions
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

stokes.rtol = 1.0e-3  # Rough solution is sufficient

# %% [markdown]
"""
## Material Projection

Project discrete material to continuous field for visualization.
"""

# %%
m_solver = uw.systems.Projection(meshbox, m_cont)
m_solver.uw_function = material.sym[1]
m_solver.smoothing = 1.0e-3
m_solver.solve()

print("Projection solve complete", flush=True)

# %% [markdown]
"""
## Initial Solve
"""

# %%
stokes.solve(zero_init_guess=True)

# %% [markdown]
"""
## Visualization of Initial State
"""

# %%
if uw.mpi.size == 1 and render:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
    pvmesh.point_data["visc"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(viscosity))
    pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, m_cont.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    # Point sources at cell centres
    cpoints = np.zeros((meshbox._centroids[::4].shape[0], 3))
    cpoints[:, 0] = meshbox._centroids[::4, 0]
    cpoints[:, 1] = meshbox._centroids[::4, 1]
    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="forward",
        compute_vorticity=False,
        max_steps=25,
        surface_streamlines=True,
    )

    spoints = vis.swarm_to_pv_cloud(swarm)
    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvstream, opacity=1.0)
    pl.add_mesh(
        pvmesh,
        cmap="Blues_r",
        edge_color="Gray",
        show_edges=True,
        scalars="M",
        opacity=0.75,
    )
    pl.add_points(
        spoint_cloud,
        cmap="Reds_r",
        scalars="M",
        render_points_as_spheres=True,
        point_size=3,
        opacity=0.5,
    )

    pl.show(cpos="xy")

# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
        pvmesh.point_data["visc"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(viscosity))
        pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, m_cont.sym)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

        # Point sources at cell centres
        subsample = 3
        cpoints = np.zeros((meshbox._centroids[::subsample].shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[::subsample, 0]
        cpoints[:, 1] = meshbox._centroids[::subsample, 1]
        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
            cpoint_cloud,
            vectors="V",
            integrator_type=45,
            integration_direction="forward",
            compute_vorticity=False,
            max_steps=25,
            surface_streamlines=True,
        )

        spoints = vis.swarm_to_pv_cloud(swarm)
        spoint_cloud = pv.PolyData(spoints)

        with swarm.access():
            spoint_cloud.point_data["M"] = material.data[...]

        pl = pv.Plotter()

        pl.add_mesh(pvstream, opacity=1)
        pl.add_mesh(
            pvmesh,
            cmap="Blues_r",
            edge_color="Gray",
            show_edges=True,
            scalars="M",
            opacity=0.75,
        )

        pl.add_points(
            spoint_cloud,
            cmap="Reds_r",
            scalars="M",
            render_points_as_spheres=True,
            point_size=3,
            opacity=0.3,
        )

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("V")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1250, 1250),
            return_img=False,
        )

        pv.close_all()

        return


# %% [markdown]
"""
## Time Evolution
"""

# %%
t_step = 0
expt_name = "output/swarm_rt"

for step in range(0, int(params.uw_n_steps)):
    stokes.solve(zero_init_guess=False)
    m_solver.solve(zero_init_guess=False)
    delta_t = min(params.uw_max_dt, stokes.estimate_dt())

    uw.pprint(f"Timestep {t_step}, dt {delta_t:.4f}")

    # Advect swarm
    swarm.advection(v_soln.sym, delta_t)

    if t_step % 5 == 0:
        plot_mesh(filename=f"{expt_name}_step_{t_step}")

        # Checkpoints
        meshbox.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, m_cont],
            outputPath="output",
            index=t_step,
        )

    t_step += 1

# %%
print(f"Rayleigh-Taylor example complete: {t_step} steps")
