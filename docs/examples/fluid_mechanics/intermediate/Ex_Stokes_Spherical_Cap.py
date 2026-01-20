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
# Stokes Flow in a Spherical Cap (Regional)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

Stokes flow in a regional spherical domain (spherical cap) with free-slip
boundary conditions on all faces. This demonstrates:
- RegionalSphericalBox mesh generator for regional models
- Free-slip conditions on curved and planar boundaries
- Buoyancy-driven flow in a regional context

## Mathematical Formulation

The non-dimensional Stokes equations with Boussinesq approximation:

$$
\\nabla^2 u - \\nabla p = Ra T' \\hat{g}
$$

Where Ra is the Rayleigh number controlling the vigor of convection.

## Parameters

- `uw_problem_size`: Controls mesh resolution (1-6)
- `uw_grid_refinement`: Additional refinement levels
- `uw_simplex`: Use simplex (True) or hex (False) elements
- `uw_rayleigh`: Rayleigh number
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
from underworld3 import timing
import numpy as np
import sympy

# Create output directory
if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Spherical_Cap.py -uw_problem_size 3
python Ex_Stokes_Spherical_Cap.py -uw_rayleigh 1e7
```
"""

# %%
params = uw.Params(
    uw_problem_size = 4,           # 1-6: resolution level
    uw_grid_refinement = 0,        # Additional refinement levels
    uw_simplex = True,             # Use simplex elements
    uw_rayleigh = 1.0e6,           # Rayleigh number
    uw_radius_outer = 1.0,         # Outer radius
    uw_radius_inner = 0.547,       # Inner radius
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
r_o = params.uw_radius_outer
r_i = params.uw_radius_inner
grid_refinement = params.uw_grid_refinement
grid_simplex = params.uw_simplex
Rayleigh = params.uw_rayleigh

# Map problem_size to element count
els_map = {1: 3, 2: 6, 3: 12, 4: 24, 5: 48, 6: 96}
els = els_map.get(params.uw_problem_size, 24)
cell_size = 1 / els

expt_name = f"Stokes_Spherical_Cap_free_slip_{els}"

timing.reset()
timing.start()

# %%
meshball = uw.meshing.RegionalSphericalBox(
    radiusInner=r_i,
    radiusOuter=r_o,
    numElements=els,
    refinement=grid_refinement,
    qdegree=2,
    simplex=grid_simplex,
)

meshball.dm.view()

# %% [markdown]
"""
## Stokes Solver Setup
"""

# %%
stokes = uw.systems.Stokes(meshball, verbose=False)

v_soln = stokes.Unknowns.u
p_soln = stokes.Unknowns.p

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1
stokes.penalty = 1.0

# %% [markdown]
"""
## Coordinate System and Buoyancy
"""

# %%
x, y, z = meshball.CoordinateSystem.N
ra, l1, l2 = meshball.CoordinateSystem.R

# Radial functions
radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
unit_rvec = meshball.X / radius_fn
gravity_fn = radius_fn

# %% [markdown]
"""
## Temperature Forcing

Gaussian blobs of buoyancy at three locations.
"""

# %%
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)

t_forcing_fn = 1.0 * (
    sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z + 0.8) ** 2))
)

with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(
        t_forcing_fn, t_soln.coords, meshball.N
    ).reshape(-1, 1)

# %% [markdown]
"""
## Solver Configuration
"""

# %%
stokes.tolerance = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None

# %% [markdown]
"""
## Boundary Conditions

Free-slip on all boundaries (Upper, Lower, North, East, South, West).
"""

# %%
Gamma = meshball.Gamma
bc_penalty = 10000

stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "Upper")
stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "Lower")
stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "North")
stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "East")
stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "South")
stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "West")

stokes.bodyforce = unit_rvec * Rayleigh * gravity_fn * t_forcing_fn

# %% [markdown]
"""
## Solve
"""

# %%
timing.reset()
timing.start()

stokes.solve(zero_init_guess=True)

timing.print_table()

# %% [markdown]
"""
## Output
"""

# %%
outdir = "output"

meshball.write_timestep(
    expt_name,
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=outdir,
    index=0,
)

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    clipped = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=(0.0, 1, 0), invert=True)

    # Streamlines
    skip = 10
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    points[:, 2] = meshball._centroids[::skip, 2]

    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="forward",
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=0.25,
        max_steps=1000,
    )

    pl = pv.Plotter(window_size=[1000, 750])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        show_scalar_bar=False,
        opacity=1,
    )

    pl.add_mesh(pvstream)

    arrows = pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        show_scalar_bar=False,
        mag=10 / Rayleigh,
    )

    pl.show(cpos="xy")
