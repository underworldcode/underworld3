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
# Elastic Beam Deformation

**PHYSICS:** solid_mechanics
**DIFFICULTY:** basic
**RUNTIME:** < 2 minutes

## Description

Simple elastic beam under applied loading - fundamental solid mechanics problem.
The beam is fixed at one end and loaded at the other, demonstrating linear
elastic deformation.

## Key Concepts

- **Linear elasticity**: Stress proportional to strain
- **Cantilever beam**: Fixed at one end, free at other
- **Displacement field**: Vector field of material displacement
- **Applied loading**: Force boundary condition

## Parameters

- `uw_beam_length`: Length of the beam
- `uw_beam_height`: Height (thickness) of the beam
- `uw_youngs_modulus`: Young's modulus (stiffness)
- `uw_poissons_ratio`: Poisson's ratio
- `uw_applied_load`: Force applied at free end
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
python elastic_beam.py -uw_youngs_modulus 1e6
python elastic_beam.py -uw_applied_load 10
```
"""

# %%
params = uw.Params(
    uw_beam_length = 2.0,           # Beam length
    uw_beam_height = 0.5,           # Beam height
    uw_youngs_modulus = 1.0e3,      # Young's modulus
    uw_poissons_ratio = 0.3,        # Poisson's ratio
    uw_applied_load = 1.0,          # Applied force
    uw_resolution_x = 40,           # Elements in x direction
    uw_resolution_y = 10,           # Elements in y direction
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(params.uw_resolution_x, params.uw_resolution_y),
    minCoords=(0.0, 0.0),
    maxCoords=(params.uw_beam_length, params.uw_beam_height),
    qdegree=2,
)

# %% [markdown]
"""
## Variables
"""

# %%
displacement = uw.discretisation.MeshVariable("u", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %% [markdown]
"""
## Stokes Solver

Note: For elastic problems in steady-state, we use the Stokes solver
with appropriate constitutive model. The "velocity" field represents
displacement for static problems.
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=displacement, pressureField=pressure)

# Simplified elastic model (viscous analogy for static problem)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = params.uw_youngs_modulus / 2

# %% [markdown]
"""
## Boundary Conditions

- Left end: Fixed (zero displacement)
- Right end: Applied load in x-direction
"""

# %%
# Fixed left end
stokes.add_essential_bc([0.0, 0.0], "Left", [0, 1])

# Applied load on right end
stokes.add_natural_bc([params.uw_applied_load, 0.0], "Right")

# %% [markdown]
"""
## Solve
"""

# %%
stokes.solve()

print(f"Elastic beam: L={params.uw_beam_length}, H={params.uw_beam_height}")
print(f"Material: E={params.uw_youngs_modulus}, nu={params.uw_poissons_ratio}")
print(f"Applied load: {params.uw_applied_load}")
print(f"Elastic deformation solved!")

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["u"] = vis.vector_fn_to_pv_points(pvmesh, displacement.sym)

    # Compute displacement magnitude
    u_mag = np.sqrt(pvmesh.point_data["u"][:, 0]**2 + pvmesh.point_data["u"][:, 1]**2)
    pvmesh.point_data["u_mag"] = u_mag

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_mesh(
        pvmesh,
        cmap="viridis",
        edge_color="Black",
        show_edges=True,
        scalars="u_mag",
        scalar_bar_args={"title": "Displacement Magnitude"},
    )

    # Add displacement arrows
    pl.add_arrows(
        pvmesh.points,
        pvmesh.point_data["u"],
        mag=0.5,
        color="red",
    )

    pl.show(cpos="xy")

# %%
displacement.stats()
