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
# Spherical Stokes Visualization

**PHYSICS:** utilities
**DIFFICULTY:** intermediate

## Description

Visualization utility for loading and displaying spherical Stokes model
checkpoint data. Loads velocity and temperature fields from HDF5 checkpoints
and renders with PyVista including streamlines and particle swarms.

## Key Concepts

- **Checkpoint loading**: Reading mesh variables and swarm data from HDF5
- **PyVista visualization**: 3D rendering with clipping, streamlines
- **Spherical shell rendering**: Clipping sphere to expose interior

## Parameters

- `uw_resolution`: Mesh cell size (default: 0.1)
- `uw_radius_o`: Outer shell radius (default: 1.0)
- `uw_radius_i`: Inner shell radius (default: 0.05)
- `uw_checkpoint_dir`: Directory containing checkpoint files
- `uw_checkpoint_base`: Base filename for checkpoints
- `uw_step`: Timestep to visualize (default: 210)

## Usage

```bash
python Ex_SphericalStokes_Visualise.py -uw_checkpoint_dir /path/to/data -uw_step 100
```

## Notes

This is a visualization utility that requires pre-existing checkpoint files.
Adjust the checkpoint_dir and checkpoint_base parameters to match your data.
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
# Fix trame async issue
import nest_asyncio
nest_asyncio.apply()

import petsc4py
import underworld3 as uw
import numpy as np

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_SphericalStokes_Visualise.py -uw_resolution 0.05
python Ex_SphericalStokes_Visualise.py -uw_checkpoint_dir /path/to/checkpoints
```
"""

# %%
params = uw.Params(
    uw_resolution = 0.1,                          # Mesh cell size
    uw_radius_o = 1.0,                            # Outer radius
    uw_radius_i = 0.05,                           # Inner radius
    uw_step = 210,                                # Timestep to visualize
    uw_checkpoint_dir = "./output",               # Checkpoint directory
    uw_checkpoint_base = "free_slip_sphere",      # Checkpoint base filename
)

res = params.uw_resolution
r_o = params.uw_radius_o
r_i = params.uw_radius_i
step = int(params.uw_step)
checkpoint_dir = str(params.uw_checkpoint_dir)
checkpoint_base = str(params.uw_checkpoint_base)

# %% [markdown]
"""
## Create Mesh and Variables
"""

# %%
meshball = uw.meshing.SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=res,
    qdegree=2,
)

swarm = uw.swarm.Swarm(mesh=meshball)
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)

# %% [markdown]
"""
## Load Checkpoint Data
"""

# %%
print(f"Loading checkpoint from {checkpoint_dir}/{checkpoint_base}, step {step}", flush=True)

# Load swarm data
swarm_file = f"{checkpoint_dir}/{checkpoint_base}.passive_swarm.{step}.h5"
print(f"Loading swarm from: {swarm_file}", flush=True)
swarm.load(swarm_file)

# Load mesh variables
v_soln.read_timestep(checkpoint_base, "u", 0, outputPath=checkpoint_dir)
t_soln.read_timestep(checkpoint_base, "deltaT", 0, outputPath=checkpoint_dir)

# %% [markdown]
"""
## Visualization
"""

# %%
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # Point sources at cell centres for streamlines
    skip = 250
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    points[:, 2] = meshball._centroids[::skip, 2]  # Fixed: was incorrectly using y coordinate
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=2.0,
    )

    with swarm.access():
        points = swarm.data.copy()
        r2 = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
        point_cloud = pv.PolyData(points[r2 < 0.98**2])

    sphere = pv.Sphere(radius=0.85, center=(0.0, 0.0, 0.0))
    clipped = pvmesh.clip_surface(sphere)

    pl = pv.Plotter(window_size=[1000, 1000])

    pl.camera_position = [(2.1, -4.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_mesh(pvstream, opacity=0.4)
    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.1)

    pl.add_points(point_cloud, color="White", point_size=3.0, opacity=0.25)

    pl.remove_scalar_bar("T")
    try:
        pl.remove_scalar_bar("mag")
    except KeyError:
        pass
    try:
        pl.remove_scalar_bar("V-normed")
    except KeyError:
        pass
    try:
        pl.remove_scalar_bar("V")
    except KeyError:
        pass

    pl.screenshot(filename="sphere_visualization.png", window_size=(1000, 1000), return_img=False)
    pl.show()

# %%
print(f"Visualization complete for step {step}")
