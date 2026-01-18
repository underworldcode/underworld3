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
# Channel Flow (Poiseuille Flow)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** basic

## Description

Pressure-driven flow between parallel plates with analytical solution available.
This is the classic Poiseuille flow problem used to validate Stokes solvers.

## Key Concepts

- **Poiseuille flow**: Parabolic velocity profile in pressure-driven channel flow
- **No-slip boundaries**: Zero velocity at walls
- **Body force**: Pressure gradient represented as driving force
- **Analytical validation**: Compare against known solution

## Analytical Solution

Parabolic velocity profile:
$$v(y) = \\frac{\\Delta P}{2 \\mu L} y(H-y)$$

Maximum velocity at channel center:
$$v_{max} = \\frac{\\Delta P H^2}{8 \\mu L}$$

## Parameters

- `uw_length`: Channel length
- `uw_height`: Channel height
- `uw_resolution_x`: Horizontal mesh resolution
- `uw_resolution_y`: Vertical mesh resolution
- `uw_viscosity`: Dynamic viscosity
- `uw_pressure_gradient`: Driving pressure gradient
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import underworld3 as uw
import numpy as np
import sympy as sp

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python channel_flow.py -uw_resolution_x 80
python channel_flow.py -uw_height 2.0
python channel_flow.py -uw_viscosity 0.5
```
"""

# %%
params = uw.Params(
    uw_length = 2.0,             # Channel length
    uw_height = 1.0,             # Channel height
    uw_resolution_x = 40,        # Horizontal resolution
    uw_resolution_y = 20,        # Vertical resolution
    uw_viscosity = 1.0,          # Dynamic viscosity
    uw_pressure_gradient = 1.0,  # Driving pressure gradient
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(params.uw_resolution_x), int(params.uw_resolution_y)),
    minCoords=(0.0, 0.0),
    maxCoords=(params.uw_length, params.uw_height),
    qdegree=2
)

# %% [markdown]
"""
## Variables
"""

# %%
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %% [markdown]
"""
## Stokes Solver

Set up the Stokes system with a body force representing the pressure gradient.
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = params.uw_viscosity

# Add pressure gradient as body force
stokes.bodyforce = sp.Matrix([params.uw_pressure_gradient, 0])

# %% [markdown]
"""
## Boundary Conditions

No-slip on top and bottom walls.
"""

# %%
stokes.add_essential_bc([0.0, 0.0], "Top", [0, 1])
stokes.add_essential_bc([0.0, 0.0], "Bottom", [0, 1])

# %% [markdown]
"""
## Solve and Validate
"""

# %%
stokes.solve()

# Check against analytical solution
if uw.mpi.size == 1:
    H = params.uw_height
    mu = params.uw_viscosity
    dP = params.uw_pressure_gradient

    v_max_analytical = (dP * H**2) / (8 * mu)

    print(f"Analytical max velocity: {v_max_analytical:.4f}")
    print(f"Channel flow solved successfully!")

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, velocity.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, velocity.sym.dot(velocity.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, pressure.sym)

    pl = pv.Plotter(window_size=(1000, 400))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    velocity_points = vis.meshVariable_to_pv_cloud(velocity)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, velocity.sym)
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.1, opacity=0.75)

    pl.show(cpos="xy")

# %%
print(f"Channel flow example complete")
