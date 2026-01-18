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
# Advection-Diffusion 1D Block Test

**PHYSICS:** convection
**DIFFICULTY:** advanced

## Description

Benchmark test for the advection-diffusion solver. A rectangular temperature
pulse is advected horizontally while diffusing. Results are compared against
the analytical solution.

## Key Concepts

- **Semi-Lagrangian method**: SLCN advection scheme
- **Analytical benchmark**: Error function solution for diffusing block
- **Phase error**: Small velocity differences from analytical solution
- **Mesh comparison**: Quads vs triangles performance

## Analytical Solution

$$T(x,t) = \\frac{1}{2}\\left[\\text{erf}\\left(\\frac{x_0 + \\delta/2 - x + v(t+t_0)}{2\\sqrt{\\kappa(t+t_0)}}\\right) + \\text{erf}\\left(\\frac{-x_0 + \\delta/2 + x - v(t+t_0)}{2\\sqrt{\\kappa(t+t_0)}}\\right)\\right]$$

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_kappa`: Thermal diffusivity
- `uw_velocity`: Advection velocity
- `uw_simplex`: Use simplex (True) or quad (False) mesh
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import numpy as np
import sympy
import os
from scipy import special

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_AdvectionDiffusion_1dBlock.py -uw_resolution 32
python Ex_AdvectionDiffusion_1dBlock.py -uw_kappa 0.1
python Ex_AdvectionDiffusion_1dBlock.py -uw_simplex 0
```
"""

# %%
params = uw.Params(
    uw_resolution = 16,              # Mesh resolution
    uw_kappa = 1.0,                  # Thermal diffusivity
    uw_velocity = 1000.0,            # Advection velocity
    uw_t_degree = 3,                 # Temperature field degree
    uw_v_degree = 2,                 # Velocity field degree
    uw_simplex = 1,                  # Use simplex mesh (1=True, 0=False)
    uw_init_t = 0.0001,              # Initial time offset
    uw_dt = 0.0006,                  # Time step
    uw_centre = 0.1,                 # Block centre
    uw_width = 0.2,                  # Block width
)

# Convert simplex to boolean
use_simplex = bool(params.uw_simplex)

# Output directory
outputPath = './output/adv_diff-hot_pipe/'
if uw.mpi.rank == 0:
    os.makedirs(outputPath, exist_ok=True)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
xmin, xmax = 0, 1
ymin, ymax = 0, 0.2

if use_simplex:
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(xmin, ymin),
        maxCoords=(xmax, ymax),
        cellSize=(ymax - ymin) / params.uw_resolution,
        regular=False,
        qdegree=max(params.uw_t_degree, params.uw_v_degree),
    )
else:
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(int(params.uw_resolution) * 5, int(params.uw_resolution)),
        minCoords=(xmin, ymin),
        maxCoords=(xmax, ymax),
        qdegree=max(params.uw_t_degree, params.uw_v_degree),
    )

x, y = mesh.X

# %% [markdown]
"""
## Analytical Solution

The solution is derived from the diffusion of a step function applied to
the leading and trailing edges of the temperature block.
"""

# %%
# Symbolic variables for analytical solution
x0 = sympy.symbols(r"{x_0}")
t0 = sympy.symbols(r"{t_0}")
delta = sympy.symbols(r"{\delta}")
ks = sympy.symbols(r"\kappa")
ts = sympy.symbols("t")
vs = sympy.symbols("v")

# Analytical solution (error function form)
Ts = (
    sympy.erf((x0 + delta / 2 - x + (vs * (ts + t0))) / (2 * sympy.sqrt(ks * (ts + t0))))
    + sympy.erf((-x0 + delta / 2 + x - ((ts + t0) * vs)) / (2 * sympy.sqrt(ks * (ts + t0))))
) / 2


def build_analytic_fn_at_t(time):
    """Build analytical solution at given time."""
    fn = Ts.subs({
        vs: params.uw_velocity,
        ts: time,
        ks: params.uw_kappa,
        delta: params.uw_width,
        x0: params.uw_centre,
        t0: params.uw_init_t,
    })
    return fn


# Initial and final analytical solutions
Ts0 = build_analytic_fn_at_t(time=0.0)
TsVKT = build_analytic_fn_at_t(time=params.uw_dt)

# %% [markdown]
"""
## Variables and Solver
"""

# %%
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=params.uw_t_degree)

# Velocity field (constant horizontal advection)
v = sympy.Matrix([params.uw_velocity, 0])

# Advection-diffusion solver
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=v,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = params.uw_kappa

# Boundary conditions
adv_diff.add_dirichlet_bc(0.0, "Left")
adv_diff.add_dirichlet_bc(0.0, "Right")

# %% [markdown]
"""
## Initial Conditions
"""

# %%
T.data[:, 0] = uw.function.evalf(Ts0, T.coords)

# %% [markdown]
"""
## Time Integration
"""

# %%
# Estimate number of substeps
dt_estimate = adv_diff.estimate_dt()[1]
steps = int(params.uw_dt // (12 * dt_estimate))
steps = max(steps, 1)

print(f"Running {steps} substeps, dt_estimate = {dt_estimate:.2e}")

adv_diff.petsc_options["snes_monitor_short"] = None

model_time = 0.0
for step in range(0, steps):
    adv_diff.solve(timestep=params.uw_dt / steps, zero_init_guess=False)
    model_time += params.uw_dt / steps
    print(f"Timestep: {step + 1}/{steps}, model time {model_time:.6f}")

# %% [markdown]
"""
## Visualization and Error Analysis
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, sympy.Matrix([params.uw_velocity, 0]).T)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)
    pvmesh.point_data["Ta"] = vis.scalar_fn_to_pv_points(pvmesh, Ts0)
    pvmesh.point_data["dT"] = pvmesh.point_data["T"] - pvmesh.point_data["Ta"]

    T_points = vis.meshVariable_to_pv_cloud(T)
    T_points.point_data["T"] = vis.scalar_fn_to_pv_points(T_points, T.sym)
    T_points.point_data["Ta"] = vis.scalar_fn_to_pv_points(T_points, TsVKT)
    T_points.point_data["T0"] = vis.scalar_fn_to_pv_points(T_points, Ts0)
    T_points.point_data["dT"] = T_points.point_data["T"] - T_points.point_data["Ta"]

    # Initial state mesh
    pvmesh2 = vis.mesh_to_pv_mesh(mesh)
    pvmesh2.point_data["T0"] = vis.scalar_fn_to_pv_points(pvmesh2, Ts0)
    pvmesh2.points[:, 1] += 0.3

    # Analytical final state mesh
    pvmesh3 = vis.mesh_to_pv_mesh(mesh)
    pvmesh3.point_data["Ta"] = vis.scalar_fn_to_pv_points(pvmesh3, TsVKT)
    pvmesh3.points[:, 1] -= 0.3

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh2,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T0",
        use_transparency=False,
        show_scalar_bar=False,
        opacity=1,
    )

    pl.add_mesh(
        pvmesh3,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Ta",
        use_transparency=False,
        show_scalar_bar=False,
        opacity=1,
    )

    pl.add_points(
        T_points,
        color="White",
        scalars="dT",
        cmap="coolwarm",
        point_size=5.0,
        opacity=0.5,
    )

    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.00003, opacity=0.5, show_scalar_bar=False)

    pl.show(cpos="xy")

# %%
# Report maximum error
if uw.mpi.size == 1:
    max_error = T_points.point_data["dT"].max()
    print(f"Maximum error (numerical - analytical): {max_error:.6f}")
