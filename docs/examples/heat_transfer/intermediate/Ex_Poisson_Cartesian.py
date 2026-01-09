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
# Poisson Equation in Cartesian Coordinates

**PHYSICS:** heat_transfer
**DIFFICULTY:** intermediate

## Description

Solve the Poisson equation with constant and spatially-varying diffusivity.
Demonstrates linear and nonlinear diffusion with analytical validation.

## Key Concepts

- **Poisson equation**: Laplacian with source term
- **Variable diffusivity**: Spatially varying thermal conductivity
- **Nonlinear diffusion**: Diffusivity depends on gradient
- **Gradient recovery**: Project derivatives to mesh variables
- **Multigrid solver**: Efficient preconditioner for elliptic problems

## Mathematical Formulation

Linear Poisson equation:
$$\\nabla \\cdot (k \\nabla \\phi) = f$$

Nonlinear diffusivity example:
$$k = 5 + \\frac{|\\nabla \\phi|^2}{2}$$

## Parameters

- `uw_cell_size`: Base mesh cell size
- `uw_refinement`: Mesh refinement level
- `uw_diffusivity`: Base diffusivity value
- `uw_tolerance`: Solver tolerance
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

from petsc4py import PETSc
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
from underworld3 import timing
import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Poisson_Cartesian.py -uw_cell_size 0.1
python Ex_Poisson_Cartesian.py -uw_refinement 3
```
"""

# %%
params = uw.Params(
    uw_cell_size = 0.25,       # Base mesh cell size
    uw_refinement = 4,         # Mesh refinement level
    uw_diffusivity = 1.0,      # Base diffusivity
    uw_tolerance = 1.0e-6,     # Solver tolerance
    uw_use_regular = 0,        # Use regular mesh (1) or unstructured (0)
)

use_regular = bool(params.uw_use_regular)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
if use_regular:
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=params.uw_cell_size,
        regular=True,
        refinement=int(params.uw_refinement),
    )
else:
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=params.uw_cell_size,
        refinement=int(params.uw_refinement),
    )

# %% [markdown]
"""
## Variables
"""

# %%
phi = uw.discretisation.MeshVariable("Phi", mesh, 1, degree=2, varsymbol=r"\phi")
scalar = uw.discretisation.MeshVariable(
    "Theta", mesh, 1, degree=1, continuous=False, varsymbol=r"\Theta"
)

# %% [markdown]
"""
## Linear Poisson Solver

Constant diffusivity with Dirichlet boundary conditions.
"""

# %%
poisson = uw.systems.Poisson(mesh, u_Field=phi)

# Constitutive law (diffusivity)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = params.uw_diffusivity

# %%
# Set source term and boundary conditions
poisson.f = 0.0
poisson.add_dirichlet_bc(1.0, "Bottom", components=0)
poisson.add_dirichlet_bc(0.0, "Top", components=0)

poisson.tolerance = params.uw_tolerance
poisson.petsc_options["snes_type"] = "newtonls"
poisson.petsc_options["ksp_type"] = "fgmres"

poisson.petsc_options["snes_monitor"] = None
poisson.petsc_options["ksp_monitor"] = None
poisson.petsc_options.setValue("pc_type", "mg")
poisson.petsc_options.setValue("pc_mg_type", "kaskade")
poisson.petsc_options["mg_levels_ksp_type"] = "fgmres"
poisson.petsc_options["mg_levels_ksp_max_it"] = 100
poisson.petsc_options["mg_levels_ksp_converged_maxits"] = None
poisson.petsc_options["mg_coarse_pc_type"] = "svd"

# %%
timing.reset()
timing.start()

# Solve
poisson.solve()

# %% [markdown]
"""
## Validate Against Analytical Solution

For this configuration, the solution is a linear function: phi = 1 - y
"""

# %%
mesh_numerical_soln = uw.function.evalf(poisson.u.fn, mesh.X.coords)
mesh_analytic_soln = uw.function.evalf(1.0 - mesh.N.y, mesh.X.coords)

max_error = np.abs(mesh_analytic_soln - mesh_numerical_soln).max()
print(f"Maximum error: {max_error:.2e}")

if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.0001):
    print("Warning: Numerical solution differs from analytical")

# %% [markdown]
"""
## Variable Diffusivity

Gaussian-shaped diffusivity centered in domain.
"""

# %%
x, y = mesh.X
x0 = y0 = sympy.Rational(1, 2)
k_variable = sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2))

poisson.constitutive_model.Parameters.diffusivity = k_variable

orig_soln = poisson.u.data.copy()

# %%
poisson.solve(zero_init_guess=True, _force_setup=True)

print(poisson.Unknowns.u.stats())

# Confirm results changed
if np.allclose(poisson.u.data, orig_soln, rtol=0.001):
    raise RuntimeError("Values did not change with variable diffusivity!")

# %% [markdown]
"""
## Nonlinear Diffusion

Diffusivity that depends on the solution gradient.
"""

# %%
# RHS term for new problem
abs_r2 = x**2 + y**2
poisson.f = -16 * abs_r2
poisson.add_dirichlet_bc([abs_r2], "Bottom")
poisson.add_dirichlet_bc([abs_r2], "Top")
poisson.add_dirichlet_bc([abs_r2], "Right")
poisson.add_dirichlet_bc([abs_r2], "Left")

# Linear solve first
poisson.constitutive_model.Parameters.diffusivity = 1
poisson.solve()

# Nonlinear diffusivity: k = 5 + |grad(phi)|^2 / 2
grad_phi = mesh.vector.gradient(phi.sym)
k_nonlinear = 5 + (grad_phi.dot(grad_phi)) / 2
poisson.constitutive_model.Parameters.diffusivity = k_nonlinear

# Use initial guess from linear solve
poisson.solve(zero_init_guess=False)

# %% [markdown]
"""
## Gradient Recovery

Project derivatives to a mesh variable for visualization.
"""

# %%
projection = uw.systems.Projection(mesh, scalar)
projection.uw_function = sympy.diff(phi.sym[0], mesh.X[1])
projection.smoothing = 1.0e-4

projection.solve()

print(f"Phi stats: {phi.stats()}")
print(f"dPhi/dy stats: {scalar.stats()}")

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, phi.sym)
    pvmesh.point_data["dTdy"] = vis.scalar_fn_to_pv_points(pvmesh, scalar.sym)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.75,
    )

    pl.camera_position = "xy"
    pl.show(cpos="xy")

# %%
timing.print_table()
print("Poisson Cartesian example complete")
