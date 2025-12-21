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
# Basic Rayleigh-Benard Convection

**PHYSICS:** convection
**DIFFICULTY:** basic
**RUNTIME:** < 3 minutes

## Description

Thermal convection in a heated layer - fundamental geophysics problem.
This example demonstrates the onset of convection when the Rayleigh
number exceeds the critical value (~1708 for this geometry).

## Key Concepts

- **Rayleigh number**: Controls convection vigor (Ra > ~1708 for onset)
- **Coupled system**: Temperature drives buoyancy, flow advects heat
- **Boussinesq approximation**: Density varies linearly with temperature
- **Free-slip sides**: Allows convection cells to develop naturally

## Parameters

- `uw_resolution`: Mesh resolution (elements per unit height)
- `uw_rayleigh`: Rayleigh number
- `uw_aspect_ratio`: Domain width/height ratio
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
python rayleigh_benard.py -uw_resolution 64
python rayleigh_benard.py -uw_rayleigh 1e5
```
"""

# %%
params = uw.Params(
    uw_resolution = 32,           # Mesh resolution (elements per unit height)
    uw_rayleigh = 1.0e4,          # Rayleigh number
    uw_aspect_ratio = 2.0,        # Width/height ratio
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(params.uw_resolution * params.uw_aspect_ratio), params.uw_resolution),
    minCoords=(0.0, 0.0),
    maxCoords=(params.uw_aspect_ratio, 1.0),
    qdegree=2,
)

# %% [markdown]
"""
## Variables
"""

# %%
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# %% [markdown]
"""
## Stokes Solver

Buoyancy force from Boussinesq approximation:
$$f_y = -Ra \cdot T$$
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

# Buoyancy force (Boussinesq approximation)
stokes.bodyforce = sp.Matrix([0, -params.uw_rayleigh * temperature.sym[0]])

# %% [markdown]
"""
## Thermal Solver
"""

# %%
thermal = uw.systems.Poisson(mesh, u_Field=temperature)
thermal.constitutive_model = uw.constitutive_models.DiffusionModel
thermal.constitutive_model.Parameters.diffusivity = 1.0

# %% [markdown]
"""
## Boundary Conditions

- Velocity: free-slip on sides, no-slip on top/bottom
- Temperature: T=1 at bottom (hot), T=0 at top (cold)
"""

# %%
# Velocity BCs
stokes.add_essential_bc([0.0], "Left", [0])
stokes.add_essential_bc([0.0], "Right", [0])
stokes.add_essential_bc([0.0, 0.0], "Top", [0, 1])
stokes.add_essential_bc([0.0, 0.0], "Bottom", [0, 1])

# Temperature BCs
thermal.add_essential_bc([1.0], "Bottom")
thermal.add_essential_bc([0.0], "Top")

# %% [markdown]
"""
## Initial Conditions

Linear temperature profile with small perturbation to trigger convection.
"""

# %%
with mesh.access(temperature):
    temperature.array[:] = 1.0 - mesh.data[:, 1]  # Linear profile
    # Add small perturbation to trigger convection
    temperature.array[:] += 0.01 * np.sin(np.pi * mesh.data[:, 0] / params.uw_aspect_ratio)

# %% [markdown]
"""
## Solve
"""

# %%
thermal.solve()
stokes.solve()

print(f"Rayleigh number: {params.uw_rayleigh}")
print(f"Convection system solved!")

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, temperature.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, velocity.sym)

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
    )

    pl.show(cpos="xy")
