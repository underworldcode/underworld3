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
# Rayleigh-Taylor Instability in Spherical Shell

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

Rayleigh-Taylor instability simulated in a 3D spherical shell geometry using
a level-set based material tracking approach. A dense layer overlies a light
layer, with the interface tracked using signed distance functions.

## Key Concepts

- **Level-set tracking**: Signed distance to interface for two materials
- **Spherical shell geometry**: 3D shell with inner and outer radii
- **Free-slip boundaries**: Penalty-based natural BCs on shell boundaries
- **Swarm advection**: Lagrangian tracking of material interfaces
- **Checkpointing**: PETSc-based save/restore

## Mathematical Formulation

Level-set function:
$$\\phi = r - r_{layer}$$

where $\\phi < 0$ is light material and $\\phi > 0$ is dense material.

## Parameters

- `uw_cell_size`: Mesh cell size
- `uw_particle_fill`: Swarm particle fill parameter
- `uw_viscosity_ratio`: Viscosity contrast between materials
- `uw_n_steps`: Number of time steps
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

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import timing

import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Swarm_RT_Spherical.py -uw_cell_size 0.1
python Ex_Stokes_Swarm_RT_Spherical.py -uw_viscosity_ratio 10.0
python Ex_Stokes_Swarm_RT_Spherical.py -uw_n_steps 50
```
"""

# %%
params = uw.Params(
    uw_cell_size = 0.1,            # Mesh cell size
    uw_particle_fill = 5,          # Swarm fill parameter
    uw_viscosity_ratio = 1.0,      # Viscosity contrast between materials
    uw_n_steps = 10,               # Number of time steps
    uw_r_outer = 1.0,              # Outer radius
    uw_r_inner = 0.54,             # Inner radius
    uw_r_layer = 0.7,              # Layer interface radius
    uw_rayleigh_base = 1.0e6,      # Base Rayleigh number
)

render = True

# %% [markdown]
"""
## Physical Constants
"""

# %%
lightIndex = 0
denseIndex = 1

viscosityRatio = params.uw_viscosity_ratio

r_layer = params.uw_r_layer
r_o = params.uw_r_outer
r_i = params.uw_r_inner

res = 0.25

# Scale Rayleigh number by shell thickness
Rayleigh = params.uw_rayleigh_base / (r_o - r_i) ** 3

offset = 0.5 * res

# %% [markdown]
"""
## Mesh Generation
"""

# %%
mesh = uw.meshing.SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=res,
    qdegree=2,
)

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
meshr = uw.discretisation.MeshVariable("r", mesh, 1, degree=1)

# %% [markdown]
"""
## Swarm and Material Level-Set

Using signed distance to interface for smooth material tracking.
"""

# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable(r"\cal{L}", swarm, proxy_degree=1, size=1)
swarm.populate(fill_param=2)

# %% [markdown]
"""
## Initial Material Distribution

Level-set based on distance to spherical interface.
"""

# %%
with swarm.access(material):
    r = np.sqrt(
        swarm._particle_coordinates.data[:, 0] ** 2
        + swarm._particle_coordinates.data[:, 1] ** 2
        + (swarm._particle_coordinates.data[:, 2] - offset) ** 2
    )

    material.data[:, 0] = r - r_layer

# %% [markdown]
"""
## Coordinate System
"""

# %%
x, y, z = mesh.CoordinateSystem.X
ra, l1, l2 = mesh.CoordinateSystem.xR

# %% [markdown]
"""
## Material Properties

Density and viscosity defined by level-set sign.
"""

# %%
density = sympy.Piecewise((0.0, material.sym[0] < 0.0), (1.0, True))
viscosity = sympy.Piecewise((1.0, material.sym[0] < 0.0), (1.0, True))

with swarm.access():
    print(f"Material range: {material.data.max():.3f} to {material.data.min():.3f}")

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = uw.systems.Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

stokes.petsc_options["snes_rtol"] = 1.0e-4
stokes.petsc_options["snes_rtol"] = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity

# Free-slip on shell boundaries via penalty
Gamma = mesh.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) * Gamma, "Upper")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) * Gamma, "Lower")

# Buoyancy
unit_vec_r = mesh.CoordinateSystem.X / mesh.CoordinateSystem.xR[0]
stokes.bodyforce = -unit_vec_r * Rayleigh * density

stokes.saddle_preconditioner = 1 / viscosity

# %% [markdown]
"""
## Mesh Radius Variable
"""

# %%
with mesh.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2 + z**2), mesh.data, mesh.N
    )

# %% [markdown]
"""
## Initial Solve
"""

# %%
timing.reset()
timing.start()

stokes.solve(zero_init_guess=True)

timing.print_table()

# %% [markdown]
"""
## Visualization of Initial State
"""

# %%
if uw.mpi.size == 1 and render:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
    pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym)
    V_data = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["V"] = 10.0 * V_data / V_data.max()

    # Point sources at cell centres
    subsample = 2
    cpoints = np.zeros((mesh._centroids[::subsample, 0].shape[0], 3))
    cpoints[:, 0] = mesh._centroids[::subsample, 0]
    cpoints[:, 1] = mesh._centroids[::subsample, 1]
    cpoints[:, 2] = mesh._centroids[::subsample, 2]

    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        compute_vorticity=False,
        surface_streamlines=False,
    )

    spoints = vis.swarm_to_pv_cloud(swarm)
    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    contours = pvmesh.contour(isosurfaces=[0.0], scalars="M")

    pl = pv.Plotter(window_size=(1000, 1000))

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.5)
    pl.add_mesh(pvstream, opacity=1.0, cmap="RdGy_r")

    pl.add_points(
        spoint_cloud,
        cmap="Reds_r",
        scalars="M",
        render_points_as_spheres=True,
        point_size=10,
        opacity=0.3,
    )

    pl.show(cpos="xz")


# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_mesh(filename):
    if uw.mpi.size != 1:
        return

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, density)
    pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym)
    V_data = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["V"] = 10.0 * V_data / V_data.max()
    print(f"Vscale {V_data.max()}")

    # Point sources at cell centres
    cpoints = np.zeros((mesh._centroids[::2].shape[0], 3))
    cpoints[:, 0] = mesh._centroids[::2, 0]
    cpoints[:, 1] = mesh._centroids[::2, 1]
    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        compute_vorticity=False,
        surface_streamlines=False,
    )

    spoints = vis.swarm_to_pv_cloud(swarm)
    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    contours = pvmesh.contour(isosurfaces=[0.0], scalars="M")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Gray", "wireframe")
    pl.add_mesh(pvstream, opacity=0.33)
    pl.add_mesh(contours, opacity=0.75, color="Yellow")

    pl.remove_scalar_bar("V")

    pl.camera_position = "xz"
    pl.screenshot(
        filename="{}.png".format(filename),
        window_size=(1000, 1000),
        return_img=False,
    )

    pv.close_all()

    return


# %% [markdown]
"""
## Time Evolution
"""

# %%
t_step = 0
expt_name = "output/swarm_rt_sph"

for step in range(0, int(params.uw_n_steps)):
    stokes.solve(zero_init_guess=False)
    delta_t = 2.0 * stokes.estimate_dt()

    uw.pprint(f"Timestep {t_step}, dt {delta_t:.4f}")

    # Advect swarm
    swarm.advection(v_soln.sym, delta_t)

    if t_step < 10 or t_step % 5 == 0:
        mesh.petsc_save_checkpoint(
            index=t_step, meshVars=[v_soln], outputPath="./output/"
        )

    t_step += 1

# %%
print(f"Spherical RT example complete: {t_step} steps")
