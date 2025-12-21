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
# Groundwater Flow with Topography

**PHYSICS:** porous_flow
**DIFFICULTY:** intermediate

## Description

Flow driven by gravity and topography in a deformed mesh domain. This example
demonstrates:
- Mesh deformation to represent surface topography
- Steady-state Darcy flow with depth-dependent permeability
- Streamline visualization of groundwater flow

## Key Concepts

- **Darcy's Law**: Flow rate proportional to permeability and pressure gradient
- **Depth-dependent permeability**: k = (y + 0.01)^pw, simulating compaction
- **Mesh deformation**: Surface varies as 1 + 0.2*x/4 + 0.04*cos(2*pi*x)*y

## Parameters

- `uw_cell_size`: Mesh resolution (default 0.05)
- `uw_permeability_power`: Exponent for depth-dependent permeability
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

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Darcy_Cartesian.py -uw_cell_size 0.025
python Ex_Darcy_Cartesian.py -uw_permeability_power 3
```
"""

# %%
params = uw.Params(
    uw_cell_size = 0.05,          # Mesh cell size
    uw_permeability_power = 2,    # Exponent for depth-dependent permeability
    uw_domain_width = 4.0,        # Domain width
    uw_domain_height = 1.0,       # Domain height (before deformation)
)

# %% [markdown]
"""
## Mesh Generation and Deformation
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(params.uw_domain_width, params.uw_domain_height),
    cellSize=params.uw_cell_size,
    qdegree=3,
)

p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1, continuous=False)

# %%
# Mesh deformation to represent topography
x, y = mesh.X

h_fn = 1.0 + x * 0.2 / 4 + 0.04 * sympy.cos(2.0 * np.pi * x) * y

new_coords = mesh.data.copy()
new_coords[:, 1] = uw.function.evaluate(h_fn * y, mesh.data, mesh.N)

mesh._deform_mesh(new_coords=new_coords)

# %% [markdown]
"""
## Visualization of Deformed Mesh
"""

# %%
if uw.mpi.size == 1 and uw.is_notebook:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )

    pl.show(cpos="xy", jupyter_backend="html")

# %% [markdown]
"""
## Darcy Flow Setup
"""

# %%
darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=p_soln, v_Field=v_soln)
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.petsc_options.delValue("ksp_monitor")

# Depth-dependent permeability
k = (y + 0.01)
pw = params.uw_permeability_power
darcy.constitutive_model.Parameters.permeability = k**pw

# Source term and gravity direction
darcy.f = 0.0
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T

# Boundary conditions
darcy.add_dirichlet_bc(0.0, "Top")  # Zero pressure at surface

# Velocity projection settings
darcy._v_projector.smoothing = 0.0

# %% [markdown]
"""
## Solve
"""

# %%
darcy.petsc_options.setValue("snes_monitor", None)
darcy.solve(verbose=False)

p_soln.stats()

# %% [markdown]
"""
## Visualization with Streamlines
"""

# %%
if uw.mpi.size == 1 and uw.is_notebook:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["dP"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0] - (h_fn - y))
    pvmesh.point_data["S"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(v_soln.sym.dot(v_soln.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # Point sources at cell centres for streamlines
    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points[::3])

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        max_steps=1000,
        max_time=0.2,
        initial_step_length=0.001,
        max_step_length=0.01,
    )

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.5, opacity=0.5)
    pl.add_mesh(pvstream, line_width=1.0)
    pl.show(cpos="xy", jupyter_backend="html")

# %% [markdown]
"""
## Metrics
"""

# %%
_, _, _, max_p, _, _, _ = p_soln.stats()
print("Max pressure: {:4f}".format(max_p))
