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
# Stokes Flow Around Circular Obstruction

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

Stokes flow in a channel with a circular obstruction using a free-slip boundary
condition on the inclusion surface. This is a classic benchmark for flow around
rigid bodies.

## Key Concepts

- **Flow obstruction**: Circular inclusion in channel flow
- **Natural boundary condition**: Free-slip on curved surface
- **pygmsh meshing**: Custom mesh with circular hole
- **Passive swarm**: Lagrangian particle tracking
- **Parabolic inflow**: Poiseuille velocity profile at inlet

## Mathematical Formulation

Inflow boundary condition (parabolic profile):
$$V_b = \\frac{4 U_0 y (H - y)}{H^2}$$

Where H = 0.41 is the channel height and U_0 is the maximum velocity.

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_refinement`: Mesh refinement level
- `uw_u0`: Inflow velocity scale
- `uw_radius`: Obstruction radius
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
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy
import pygmsh
from enum import Enum

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Flow_Obstruction.py -uw_resolution 30
python Ex_Stokes_Flow_Obstruction.py -uw_radius 0.1
python Ex_Stokes_Flow_Obstruction.py -uw_u0 2.0
```
"""

# %%
params = uw.Params(
    uw_resolution = 20,          # Mesh resolution
    uw_refinement = 0,           # Mesh refinement level
    uw_u0 = 1.0,                 # Inflow velocity scale
    uw_width = 2.2,              # Channel width
    uw_height = 0.41,            # Channel height
    uw_radius = 0.05,            # Obstruction radius
    uw_centre_x = 0.2,           # Obstruction center x
    uw_centre_y = 0.2,           # Obstruction center y
)

# Derived values
outdir = f"output/output_res_{int(params.uw_resolution)}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# %% [markdown]
"""
## Mesh Generation

Create a channel mesh with a circular hole using pygmsh.
"""

# %%
class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left = 4
    inclusion = 5
    All_Boundaries = 1001

# Mesh parameters
csize = 1.0 / params.uw_resolution
csize_circle = 0.25 * csize
res = csize_circle

width = params.uw_width
height = params.uw_height
radius = params.uw_radius
centre = (params.uw_centre_x, params.uw_centre_y)


def pipemesh_mesh_refinement_callback(dm):
    """Snap boundary nodes to exact circular boundary."""
    r_p = radius

    c2 = dm.getCoordinatesLocal()
    coords = c2.array.reshape(-1, 2) - centre

    R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2).reshape(-1, 1)

    pipeIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        dm, "inclusion"
    )

    coords[pipeIndices] *= r_p / R[pipeIndices]
    coords = coords + centre

    c2.array[...] = coords.reshape(-1)
    dm.setCoordinatesLocal(c2)

    return


def pipemesh_return_coords_to_bounds(coords):
    """Restore inflow samples to inflow points."""
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = 0.0001
    return coords


# %%
if uw.mpi.rank == 0:
    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (centre[0], centre[1], 0.0),
            radius,
            make_surface=False,
            mesh_size=csize_circle,
        )
        domain = geom.add_rectangle(
            xmin=0.0,
            ymin=0.0,
            xmax=width,
            ymax=height,
            z=0,
            holes=[inclusion],
            mesh_size=csize,
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label=boundaries.bottom.name)
        geom.add_physical(domain.surface.curve_loop.curves[1], label=boundaries.right.name)
        geom.add_physical(domain.surface.curve_loop.curves[2], label=boundaries.top.name)
        geom.add_physical(domain.surface.curve_loop.curves[3], label=boundaries.left.name)
        geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry(f".meshes/ns_pipe_flow_{int(params.uw_resolution)}.msh")

# %%
pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{int(params.uw_resolution)}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=int(params.uw_refinement),
    refinement_callback=pipemesh_mesh_refinement_callback,
    return_coords_to_bounds=pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3,
)

pipemesh.view()

# %% [markdown]
"""
## Coordinate Functions
"""

# %%
x = pipemesh.N.x
y = pipemesh.N.y

# Relative to inclusion center
r = sympy.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
th = sympy.atan2(y - centre[1], x - centre[0])

# Unit radial vector for inclusion
inclusion_rvec = sympy.Matrix((x - centre[0], y - centre[1]))
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# Parabolic inflow profile
Vb = (4.0 * params.uw_u0 * y * (height - y)) / height**2

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
r_inc = uw.discretisation.MeshVariable("Rinc", pipemesh, 1, degree=1, continuous=True)

# %% [markdown]
"""
## Passive Swarm

Create a swarm for Lagrangian particle tracking.
"""

# %%
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(fill_param=1)

# Add new points at the inflow
npoints = 100
passive_swarm.dm.addNPoints(npoints)
with passive_swarm.access(passive_swarm._particle_coordinates):
    for i in range(npoints):
        passive_swarm._particle_coordinates.data[-1 : -(npoints + 1) : -1, :] = np.array(
            [0.01, 0.195] + 0.01 * np.random.random((npoints, 2))
        )

# %% [markdown]
"""
## Stokes Solver

Constant viscosity flow with free-slip on the inclusion.
"""

# %%
stokes = uw.systems.Stokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Constant viscosity
stokes.penalty = 100
stokes.bodyforce = sympy.Matrix([0, 0])

# Distance function for inclusion surface
hw = 1000.0 / res
with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evalf(r, pipemesh.data, pipemesh.N)

surface_defn = sympy.exp(-(((r_inc.sym[0] - radius) / radius) ** 2) * hw)

# %% [markdown]
"""
## Boundary Conditions
"""

# %%
# No-slip on walls
stokes.add_dirichlet_bc((0.0, 0.0), "top")
stokes.add_dirichlet_bc((0.0, 0.0), "bottom")

# Parabolic inflow
stokes.add_dirichlet_bc((Vb, 0.0), "left")

# Free-slip on inclusion (natural BC)
stokes.add_natural_bc(
    1000 * v_soln.sym.dot(inclusion_unit_rvec) * inclusion_unit_rvec, "inclusion"
)

stokes.tolerance = 1.0e-4

# %% [markdown]
"""
## Solver Configuration
"""

# %%
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
"""
## Solve
"""

# %%
timing.reset()
timing.start()

stokes.solve(verbose=False)

timing.print_table(display_fraction=0.999)

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # Point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.025 / params.uw_u0,
        opacity=0.75,
    )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False,
    )

    pl.add_mesh(pvstream, show_scalar_bar=False)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=2,
        opacity=0.25,
    )

    pl.show()

# %%
print(f"Flow obstruction example complete: resolution {int(params.uw_resolution)}")
