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
# Linear Diffusion - Hot Pipe Benchmark

**PHYSICS:** heat_transfer
**DIFFICULTY:** intermediate

## Description

Linear diffusion of a rectangular temperature pulse ("hot pipe") using the
advection-diffusion solver with zero velocity. Compares 2D UW3 solution
against a 1D numerical reference solution.

## Key Concepts

- **SLCN advection-diffusion**: Semi-Lagrangian Crank-Nicolson scheme
- **Diffusion-only mode**: V = 0, pure thermal diffusion
- **1D benchmark**: Compare against numerical 1D solution
- **Time stepping**: Stability constraint from diffusivity

## Mathematical Formulation

Heat equation:
$$\\frac{\\partial T}{\\partial t} = \\kappa \\nabla^2 T$$

Timestep estimate:
$$\\Delta t = \\frac{(\\Delta x)^2}{\\kappa}$$

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_diffusivity`: Thermal diffusivity (k)
- `uw_pipe_thickness`: Width of initial hot region
- `uw_n_steps`: Number of time steps
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI
import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_DiffusionSLCN_Cartesian.py -uw_resolution 64
python Ex_DiffusionSLCN_Cartesian.py -uw_pipe_thickness 0.2
python Ex_DiffusionSLCN_Cartesian.py -uw_n_steps 50
```
"""

# %%
params = uw.Params(
    uw_resolution = 32,          # Mesh resolution
    uw_diffusivity = 1.0,        # Thermal diffusivity
    uw_pipe_thickness = 0.4,     # Width of initial hot region
    uw_t_min = 0.5,              # Background temperature
    uw_t_max = 1.0,              # Hot pipe temperature
    uw_n_steps = 1,              # Number of time steps
)

# Domain bounds
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

# Scaling parameters (for dimensional analysis if needed)
k0 = 1e-6   # m^2/s (diffusivity)
l0 = 1e5    # 100 km in m (length of box)
time_scale = l0**2 / k0  # s
time_scale_Myr = time_scale / (60 * 60 * 24 * 365.25 * 1e6)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
res = int(params.uw_resolution)

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(res, res),
    minCoords=(xmin, ymin),
    maxCoords=(xmax, ymax),
)

# %% [markdown]
"""
## Variables
"""

# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

# %% [markdown]
"""
## Advection-Diffusion Solver

Using SLCN scheme with zero velocity for pure diffusion.
"""

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=v,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = params.uw_diffusivity

# %% [markdown]
"""
## Boundary Conditions

Fixed temperature on top and bottom walls.
"""

# %%
adv_diff.add_dirichlet_bc(params.uw_t_min, "Bottom", 0)
adv_diff.add_dirichlet_bc(params.uw_t_min, "Top", 0)

# %% [markdown]
"""
## Initial Conditions

Hot rectangular region in center of domain.
"""

# %%
# Use variable coords for coordinate bounds and conditions
y_coords = T.coords[:, 1]
maxY = y_coords.max()
minY = y_coords.min()

T.data[...] = params.uw_t_min

# Center the pipe vertically
pipe_position = ((maxY - minY) - params.uw_pipe_thickness) / 2.0

# Set hot region
T.data[
    (y_coords >= (minY + pipe_position))
    & (y_coords <= (maxY - pipe_position))
] = params.uw_t_max

# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_fig():
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(mesh)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)

        velocity_points = vis.meshVariable_to_pv_cloud(v)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

        pl = pv.Plotter(window_size=(750, 750))

        pl.add_mesh(pvmesh, "Black", "wireframe")

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.95,
        )

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=5.0, opacity=0.5)

        pl.show(cpos="xy")


plot_fig()

# %% [markdown]
"""
## 1D Reference Solution

Numerical 1D diffusion for comparison.
"""

# %%
# Sample points for vertical profile
sample_y = np.arange(
    mesh.X.coords[:, 1].min(), mesh.X.coords[:, 1].max(), mesh.get_min_radius()
)
sample_x = np.zeros_like(sample_y)  # center of box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y

# Initial profile
t0 = uw.function.evaluate(adv_diff.u.fn, sample_points)

# %% [markdown]
"""
## Time Step Estimate
"""

# %%
dt = mesh.get_min_radius() ** 2 / params.uw_diffusivity
print(f"Time step: {dt:.6f} (dimensionless)")
print(f"Time step: {dt * time_scale_Myr:.4f} Myr (scaled)")


# %% [markdown]
"""
## 1D Diffusion Reference
"""

# %%
def diffusion_1D(sample_points, temp_profile, k, model_dt):
    """1D numerical diffusion solver for benchmark comparison."""
    x = sample_points
    T = temp_profile.copy()

    dx = sample_points[1] - sample_points[0]
    dt_stable = 0.5 * (dx**2 / k)

    # Time stepping
    total_time = model_dt
    time_1d = min(model_dt, dt_stable)
    nts = math.ceil(total_time / time_1d)
    final_dt = total_time / nts

    for i in range(nts):
        qT = -k * np.diff(T) / dx
        dTdt = -np.diff(qT) / dx
        T[1:-1] += dTdt * final_dt

    return T


# Get initial temperature profile
temp_data = uw.function.evaluate(adv_diff.u.fn, sample_points)

# %% [markdown]
"""
## Time Integration Loop
"""

# %%
step = 0
time = 0.0
nsteps = int(params.uw_n_steps)

if uw.mpi.size == 1:
    plt.figure(figsize=(9, 3))
    plt.plot(t0, sample_points[:, 1], ls=":", label="Initial")

while step < nsteps:
    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {time:6.5f}")

    # UW3 profile
    t1 = uw.function.evalf(adv_diff.u.sym[0], sample_points)

    if uw.mpi.size == 1 and step % 10 == 0:
        plt.figure()
        plt.plot(t1, sample_points[:, 1], ls="-", c="red", label="2D UW3 model")
        plt.plot(temp_data, sample_points[:, 1], ls=":", c="k", label="1D reference")
        plt.legend()
        plt.xlabel("Temperature")
        plt.ylabel("Y position")
        plt.title(f"Step {step}")
        plt.show()

    # 1D reference diffusion
    temp_data = diffusion_1D(
        sample_points=sample_points[:, 1],
        temp_profile=temp_data,
        k=params.uw_diffusivity,
        model_dt=dt,
    )

    # UW3 solve
    adv_diff.solve(timestep=dt)

    step += 1
    time += dt

# %%
print(f"Diffusion SLCN example complete: {nsteps} steps")
