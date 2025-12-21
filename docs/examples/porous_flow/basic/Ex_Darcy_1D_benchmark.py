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
# Darcy Flow 1D Benchmark

**PHYSICS:** porous_flow
**DIFFICULTY:** basic

## Description

One-dimensional Darcy flow through a layered domain with different
permeabilities. This example:
- Validates the Darcy solver against an analytical solution
- Demonstrates layered permeability with a sharp interface
- Shows the effect of gravity on pressure distribution

## Analytical Solution

For 1D flow through two layers with permeabilities k1 and k2:
- Pressure is linear within each layer
- Flux is continuous across the interface
- Pressure gradient is discontinuous (inversely proportional to k)

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_permeability_ratio`: k1/k2 ratio
- `uw_max_pressure`: Boundary pressure at bottom
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
from sympy import Piecewise

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Darcy_1D_benchmark.py -uw_cell_size 0.02
python Ex_Darcy_1D_benchmark.py -uw_permeability_ratio 1e-5
```
"""

# %%
params = uw.Params(
    uw_cell_size = 1 / 25,          # Mesh cell size
    uw_permeability_ratio = 1.0e-4, # k2/k1 ratio (k1=1, k2=this value)
    uw_max_pressure = 0.5,          # Pressure at bottom boundary
    uw_interface_y = -0.26,         # Y-coordinate of interface
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=params.uw_cell_size,
    qdegree=3,
)

x = mesh.N.x
y = mesh.N.y

# %% [markdown]
"""
## Darcy Solver Setup
"""

# %%
darcy = uw.systems.SteadyStateDarcy(mesh)

p_soln = darcy.Unknowns.u
v_soln = darcy.v

darcy.petsc_options["snes_rtol"] = 1.0e-6
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

# %%
# Clone for comparison (no gravity case)
p_soln_0 = p_soln.clone("P_no_g", r"{p_\textrm{(no g)}}")
v_soln_0 = v_soln.clone("V_no_g", r"{v_\textrm{(no g)}}")

# %% [markdown]
"""
## Material Properties

Two-layer permeability structure with a piecewise function.
"""

# %%
interfaceY = params.uw_interface_y
max_pressure = params.uw_max_pressure

k1 = 1.0
k2 = params.uw_permeability_ratio

# Piecewise permeability function
kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))

darcy.constitutive_model.Parameters.permeability = kFunc
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0]).T  # No gravity initially
darcy.f = 0.0

# Boundary conditions
darcy.add_dirichlet_bc(0.0, "Top")
darcy.add_dirichlet_bc(-1.0 * minY * max_pressure, "Bottom")

# %% [markdown]
"""
## Solve Without Gravity
"""

# %%
darcy.solve()

with mesh.access(p_soln_0, v_soln_0):
    p_soln_0.data[...] = p_soln.data[...]
    v_soln_0.data[...] = v_soln.data[...]

# %% [markdown]
"""
## Solve With Gravity
"""

# %%
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T
darcy.solve()

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["K"] = vis.scalar_fn_to_pv_points(pvmesh, kFunc)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # Point sources for streamlines
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
        max_time=0.1,
        initial_step_length=0.001,
        max_step_length=0.01,
    )

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_mesh(pvstream, line_width=1.0)

    pl.show(cpos="xy")

# %% [markdown]
"""
## Analytical Comparison
"""

# %%
# Set up interpolation coordinates
ycoords = np.linspace(minY + 0.001 * (maxY - minY), maxY - 0.001 * (maxY - minY), 100)
xcoords = np.full_like(ycoords, -0.5)
xy_coords = np.column_stack([xcoords, ycoords])

pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords)
pressure_interp_0 = uw.function.evaluate(p_soln_0.sym[0], xy_coords)

# %%
# Analytical solution
La = -1.0 * interfaceY
Lb = 1.0 + interfaceY
dP = max_pressure

# With gravity
S = 1
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [
        lambda ycoords: -Pa * ycoords / La,
        lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
    ],
)

# Without gravity
S = 0
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic_noG = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [
        lambda ycoords: -Pa * ycoords / La,
        lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
    ],
)

# %% [markdown]
"""
## Plot Comparison
"""

# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111, xlabel="Pressure", ylabel="Depth")
ax1.plot(pressure_interp, ycoords, linewidth=3, label="Numerical solution")
ax1.plot(pressure_interp_0, ycoords, linewidth=3, label="Numerical solution (no G)")
ax1.plot(
    pressure_analytic, ycoords, linewidth=3, linestyle="--", label="Analytic solution"
)
ax1.plot(
    pressure_analytic_noG,
    ycoords,
    linewidth=3,
    linestyle="--",
    label="Analytic (no gravity)",
)
ax1.grid("on")
ax1.legend()

# %% [markdown]
"""
## Validation
"""

# %%
assert np.allclose(pressure_analytic, pressure_interp, atol=0.01), \
    "Numerical solution does not match analytical solution"

print("Validation passed: numerical solution matches analytical solution")
