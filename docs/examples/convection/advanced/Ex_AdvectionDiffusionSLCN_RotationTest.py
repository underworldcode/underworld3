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
# Semi-Lagrangian Advection-Diffusion Rotation Test

**PHYSICS:** convection
**DIFFICULTY:** advanced

## Description

Tests the Semi-Lagrangian Crank-Nicolson (SLCN) advection-diffusion solver
using rigid body rotation in an annulus. A Gaussian temperature anomaly
is advected by a prescribed velocity field while undergoing diffusion.

## Key Concepts

- **Semi-Lagrangian method**: Traces characteristics backward in time
- **Rigid body rotation**: v_theta = constant, v_r = 0
- **Annulus geometry**: Tests solver on curved domain

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_diffusivity`: Thermal diffusivity
- `uw_n_steps`: Number of advection steps
- `uw_dt`: Time step size
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import os
os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import VarType
from underworld3 import timing
import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_AdvectionDiffusionSLCN_RotationTest.py -uw_diffusivity 0.001
python Ex_AdvectionDiffusionSLCN_RotationTest.py -uw_n_steps 20
```
"""

# %%
params = uw.Params(
    uw_cell_size = 0.2,          # Mesh cell size
    uw_refinement = 1,           # Mesh refinement levels
    uw_diffusivity = 0.01,       # Thermal diffusivity
    uw_n_steps = 10,             # Number of time steps
    uw_dt = 0.05,                # Time step size
    uw_radius_outer = 1.0,       # Outer radius
    uw_radius_inner = 0.5,       # Inner radius
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshball = uw.meshing.Annulus(
    radiusOuter=params.uw_radius_outer,
    radiusInner=params.uw_radius_inner,
    cellSize=params.uw_cell_size,
    refinement=params.uw_refinement,
    qdegree=3,
)

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3, varsymbol=r"T_{0}")

# %% [markdown]
"""
## Prescribed Velocity Field

Rigid body rotation: v_theta = r * theta_dot, v_r = 0
One complete revolution in time t=1.0
"""

# %%
radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)

x, y = meshball.X
r, th = meshball.CoordinateSystem.xR

# Rigid body rotation: one revolution in time 1.0
theta_dot = 2.0 * np.pi
v_x = -r * theta_dot * sympy.sin(th)
v_y = r * theta_dot * sympy.cos(th)

with meshball.access(v_soln):
    v_soln.data[:, 0] = uw.function.evaluate(v_x, v_soln.coords)
    v_soln.data[:, 1] = uw.function.evaluate(v_y, v_soln.coords)

# %% [markdown]
"""
## Advection-Diffusion Solver
"""

# %%
k = params.uw_diffusivity
r_i = params.uw_radius_inner
r_o = params.uw_radius_outer

adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    order=2,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel(adv_diff.Unknowns)
adv_diff.constitutive_model.Parameters.diffusivity = k

# %% [markdown]
"""
## Initial Conditions

Gaussian temperature anomaly near the top of the annulus.
"""

# %%
abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = sympy.exp(-30.0 * (meshball.N.x**2 + (meshball.N.y - 0.75) ** 2))

adv_diff.add_dirichlet_bc(0.0, "Lower")
adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]

# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        points = vis.meshVariable_to_pv_cloud(t_soln)
        points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)

        point_cloud = pv.PolyData(points)

        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

        pl = pv.Plotter(window_size=(1000, 750))

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.0001, opacity=0.75)

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.66,
        )

        pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )


# %% [markdown]
"""
## Initial Solve Test
"""

# %%
timing.reset()
timing.start()

delta_t = 0.001
adv_diff.solve(timestep=delta_t, verbose=False, _force_setup=False)

# %% [markdown]
"""
## Visualization of Initial State
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)

    point_cloud = pv.PolyData(points)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01, opacity=0.75)
    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7,
        opacity=0.66,
    )
    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.remove_scalar_bar("mag")

    pl.show()

# %% [markdown]
"""
## Time Evolution
"""

# %%
expt_name = "rotation_test_slcn"
delta_t = params.uw_dt

plot_T_mesh(filename="{}_step_{}".format(expt_name, 0))

for step in range(0, params.uw_n_steps):
    adv_diff.solve(timestep=delta_t, verbose=False)

    uw.pprint("Timestep {}, dt {}".format(step, delta_t))

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

# %% [markdown]
"""
## Final State
"""

# %%
t_soln.stats()

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    points.point_data["dT"] = (
        vis.scalar_fn_to_pv_points(points, t_soln.sym)
        - vis.scalar_fn_to_pv_points(points, t_0.sym)
    )

    point_cloud = pv.PolyData(points)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.0001, opacity=0.75)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        scalars="T",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.remove_scalar_bar("mag")

    pl.show()

# %%
uw.timing.print_table()
