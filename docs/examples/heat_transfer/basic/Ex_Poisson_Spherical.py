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
# Steady State Diffusion in Hollow Geometries

**PHYSICS:** heat_transfer
**DIFFICULTY:** basic

## Description

This example solves the steady-state diffusion equation (Poisson problem)
in hollow geometries:
- 2D annulus (hollow disk)
- 3D spherical shell

The solution is validated against the analytical solution for radial
heat conduction.

## Analytical Solution

For radial heat conduction with constant diffusivity:
- 2D: T(r) = A * ln(r) + B
- 3D: T(r) = A / r + B

Where A and B are determined by boundary conditions.

## Parameters

- `uw_problem_size`: Controls mesh resolution (1=coarse, 4=fine)
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np
import sympy
import os

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Poisson_Spherical.py -uw_problem_size 3
```
"""

# %%
# Problem parameters - editable here or via command line
params = uw.Params(
    uw_problem_size = 1,     # 1-4: resolution level
    uw_diffusivity = 1.0,    # Thermal diffusivity
    uw_temp_inner = 2.0,     # Temperature at inner boundary
    uw_temp_outer = 1.0,     # Temperature at outer boundary
    uw_radius_inner = 0.5,   # Inner radius
    uw_radius_outer = 1.0,   # Outer radius
)

# Material properties
k = params.uw_diffusivity
f = 0.0  # Source term (zero for pure conduction)
t_i = params.uw_temp_inner
t_o = params.uw_temp_outer
r_i = params.uw_radius_inner
r_o = params.uw_radius_outer

# %% [markdown]
"""
## 2D Annulus Problem
"""

# %%
from underworld3.meshing import Annulus

# Map problem_size to cell size
cell_size_map = {1: 0.05, 2: 0.02, 3: 0.01, 4: 0.0033}
cell_size = cell_size_map.get(params.uw_problem_size, 0.05)

mesh = Annulus(radiusInner=r_i, radiusOuter=r_o, cellSize=cell_size)

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# %%
mesh.dm.view()

# %% [markdown]
"""
## Poisson Solver Setup (2D)
"""

# %%
poisson = Poisson(mesh, u_Field=t_soln)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = k
poisson.f = f

poisson.petsc_options["snes_rtol"] = 1.0e-6
poisson.petsc_options.delValue("ksp_monitor")
poisson.petsc_options.delValue("ksp_rtol")

# %%
poisson.add_dirichlet_bc(t_i, "Lower", 0)
poisson.add_dirichlet_bc(t_o, "Upper", 0)

# %%
poisson.solve()

# %% [markdown]
"""
## Validation (2D)

Compare numerical solution to analytical:
T(r) = A * ln(r) + B
"""

# %%
import math

A = (t_i - t_o) / (sympy.log(r_i) - math.log(r_o))
B = t_o - A * sympy.log(r_o)
sol = A * sympy.log(sympy.sqrt(mesh.N.x**2 + mesh.N.y**2)) + B

mesh_analytic_soln = uw.function.evaluate(sol, mesh.X.coords, mesh.N)
mesh_numerical_soln = uw.function.evaluate(t_soln.fn, mesh.X.coords, mesh.N)

if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.01):
    raise RuntimeError("2D validation failed: numerical solution differs from analytical.")

print("2D validation passed!")

# %% [markdown]
"""
## Non-linear Extension (2D)

Test with temperature-dependent diffusivity and source term.
"""

# %%
poisson.constitutive_model.Parameters.diffusivity = 1.0 + 0.1 * poisson.u.fn**1.5
poisson.f = 0.01 * poisson.u.sym[0] ** 0.5
poisson.solve(zero_init_guess=False)

# %% [markdown]
"""
## Visualization (2D)
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = mesh_analytic_soln
    pvmesh.point_data["T2"] = mesh_numerical_soln
    pvmesh.point_data["DT"] = mesh_analytic_soln - mesh_numerical_soln

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="DT",
        use_transparency=False,
        opacity=0.5,
    )

    pl.camera_position = "xy"
    pl.show(cpos="xy")

# %%
expt_name = "Poisson-Annulus"
outdir = "output"
os.makedirs(f"{outdir}", exist_ok=True)

mesh.write_timestep(
    expt_name, meshUpdates=True, meshVars=[t_soln], outputPath=outdir, index=0
)

# %% [markdown]
"""
## 3D Spherical Shell Problem
"""

# %%
from underworld3.meshing import SphericalShell

# 3D uses coarser mesh
cell_size_3d_map = {1: 0.3, 2: 0.15, 3: 0.05, 4: 0.02}
cell_size_3d = cell_size_3d_map.get(params.uw_problem_size, 0.3)

mesh_3d = SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=cell_size_3d,
    refinement=1,
)

t_soln_3d = uw.discretisation.MeshVariable("T", mesh_3d, 1, degree=2)

# %%
mesh_3d.dm.view()

# %% [markdown]
"""
## Poisson Solver Setup (3D)
"""

# %%
poisson_3d = Poisson(mesh_3d, u_Field=t_soln_3d)
poisson_3d.constitutive_model = uw.constitutive_models.DiffusionModel
poisson_3d.constitutive_model.Parameters.diffusivity = k
poisson_3d.f = f

poisson_3d.petsc_options["snes_rtol"] = 1.0e-6
poisson_3d.petsc_options.delValue("ksp_rtol")

poisson_3d.add_dirichlet_bc(t_i, "Lower", 0)
poisson_3d.add_dirichlet_bc(t_o, "Upper", 0)

# %%
poisson_3d.solve()

# %% [markdown]
"""
## Validation (3D)

Compare numerical solution to analytical:
T(r) = A / r + B
"""

# %%
A_3d = (t_i - t_o) / (1 / r_i - 1 / r_o)
B_3d = t_o - A_3d / r_o
sol_3d = A_3d / (sympy.sqrt(mesh_3d.N.x**2 + mesh_3d.N.y**2 + mesh_3d.N.z**2)) + B_3d

with mesh_3d.access():
    mesh_analytic_soln_3d = uw.function.evaluate(sol_3d, mesh_3d.data, mesh_3d.N)
    mesh_numerical_soln_3d = uw.function.evaluate(t_soln_3d.fn, mesh_3d.data, mesh_3d.N)

if not np.allclose(mesh_analytic_soln_3d, mesh_numerical_soln_3d, rtol=0.1):
    raise RuntimeError("3D validation failed: numerical solution differs from analytical.")

print("3D validation passed!")

# %% [markdown]
"""
## Visualization (3D)
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh_3d)
    pvmesh.point_data["T"] = mesh_analytic_soln_3d
    pvmesh.point_data["T2"] = mesh_numerical_soln_3d
    pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"]

    clipped = pvmesh.clip(origin=(0.001, 0.0, 0.0), normal=(1, 0, 0), invert=True)

    pl = pv.Plotter()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="T2",
        use_transparency=False,
        opacity=1.0,
    )

    pl.camera_position = "xy"
    pl.show(cpos="xy")

# %%
expt_name = "Poisson-Sphere"
outdir = "output"
os.makedirs(f"{outdir}", exist_ok=True)

mesh_3d.write_timestep(
    expt_name, meshUpdates=True, meshVars=[t_soln_3d], outputPath=outdir, index=0
)
