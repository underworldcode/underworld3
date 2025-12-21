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
# Stokes Flow Through Explicit Grain Pack

**PHYSICS:** porous_flow
**DIFFICULTY:** advanced

## Description

Stokes flow through a 2D channel containing explicit circular grains (obstacles).
Simulates pore-scale flow in a porous medium with randomly sized grains arranged
in a regular pattern. Passive tracers are advected to visualize flow paths.

## Key Concepts

- **Explicit porous medium**: Individual grains as no-slip boundaries
- **gmsh mesh generation**: Complex geometry with many circular holes
- **Passive tracers**: Swarm advection for breakthrough curve analysis
- **Porosity calculation**: Area-based porosity from mesh
- **Pore-scale velocity**: Local flow around grains

## Mathematical Formulation

Stokes flow through pore space:
$$\\nabla \\cdot \\mathbf{u} = 0$$
$$-\\nabla p + \\mu \\nabla^2 \\mathbf{u} = 0$$

with no-slip on grain surfaces.

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_refinement`: Mesh refinement level
- `uw_max_steps`: Maximum time steps
- `uw_restart_step`: Restart from checkpoint (-1 = fresh start)
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Explicit_Flow_Grains.py -uw_resolution 20
python Ex_Explicit_Flow_Grains.py -uw_max_steps 500
```
"""

# %%
params = uw.Params(
    uw_resolution = 14,           # Mesh resolution
    uw_refinement = 0,            # Mesh refinement levels
    uw_max_steps = 201,           # Maximum time steps
    uw_restart_step = -1,         # Restart from step (-1 = fresh start)
    uw_width = 4.0,               # Domain width
    uw_height = 1.0,              # Domain height
    uw_rows = 5,                  # Number of grain rows
    uw_radius_base = 0.075,       # Base grain radius
    uw_radius_variation = 0.075,  # Radius random variation
    uw_u0 = 1.0,                  # Inlet velocity
)

resolution = int(params.uw_resolution)
refinement = int(params.uw_refinement)
expt_name = "Expt_Grains"

# %% [markdown]
"""
## Output Directory
"""

# %%
outdir = f"output/output_res_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# %% [markdown]
"""
## Grain Pack Parameters
"""

# %%
width = params.uw_width
height = params.uw_height
rows = int(params.uw_rows)
columns = int((width - 1) * rows)
radius_0 = params.uw_radius_base
variation = params.uw_radius_variation
U0 = params.uw_u0

csize = 1.0 / resolution
csize_circle = 0.66 * csize
res = csize_circle

# %% [markdown]
"""
## Mesh Generation with gmsh

Create grain pack with random radius variation.
"""

# %%
import pygmsh
from enum import Enum


class boundaries(Enum):
    bottom = 1
    right = 2
    left = 3
    top = 4
    inclusion = 5
    All_Boundaries = 1001


def pipemesh_return_coords_to_bounds(coords):
    """Restore inflow samples to inflow points."""
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = 0.0001
    return coords


if uw.mpi.rank == 0:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("Domain")

    inclusions = []
    inclusion_curves = []

    # Repeatable random numbers
    rrand = np.random.default_rng(66666)

    dy = 1.0 / (rows + 0.5)
    dx = dy * 1.2

    for row in range(0, rows):
        for col in range(0, columns):
            y = dy * (row + 0.75)
            x = 0.25 + dx * col + (row % 2) * 0.5 * dx
            r = radius_0 + variation * (rrand.random() - 0.5)

            i_points = [
                gmsh.model.occ.add_point(x, y, 0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x, y + r, 0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x - r, y, 0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x, y - r, 0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x + r, y, 0.0, meshSize=csize_circle),
            ]

            i_quarter_circles = [
                gmsh.model.occ.add_circle_arc(i_points[1], i_points[0], i_points[2]),
                gmsh.model.occ.add_circle_arc(i_points[2], i_points[0], i_points[3]),
                gmsh.model.occ.add_circle_arc(i_points[3], i_points[0], i_points[4]),
                gmsh.model.occ.add_circle_arc(i_points[4], i_points[0], i_points[1]),
            ]

            inclusion_loop = gmsh.model.occ.add_curve_loop(i_quarter_circles)
            inclusion = gmsh.model.occ.add_plane_surface([inclusion_loop])

            inclusions.append((2, inclusion))
            inclusion_curves.extend(i_quarter_circles)

            gmsh.model.occ.synchronize()

    # Domain corners
    corner_points = []
    corner_points.append(gmsh.model.occ.add_point(0.0, 0.0, 0.0, csize))
    corner_points.append(gmsh.model.occ.add_point(width, 0.0, 0.0, csize))
    corner_points.append(gmsh.model.occ.add_point(width, 1.0, 0.0, csize))
    corner_points.append(gmsh.model.occ.add_point(0.0, 1.0, 0.0, csize))

    bottom = gmsh.model.occ.add_line(corner_points[0], corner_points[1])
    right = gmsh.model.occ.add_line(corner_points[1], corner_points[2])
    top = gmsh.model.occ.add_line(corner_points[2], corner_points[3])
    left = gmsh.model.occ.add_line(corner_points[3], corner_points[0])

    domain_loop = gmsh.model.occ.add_curve_loop((bottom, right, top, left))
    gmsh.model.occ.add_plane_surface([domain_loop])

    gmsh.model.occ.synchronize()

    # Save bounding boxes for boundary identification after cut
    brtl_bboxes = [
        gmsh.model.get_bounding_box(1, bottom),
        gmsh.model.get_bounding_box(1, right),
        gmsh.model.get_bounding_box(1, top),
        gmsh.model.get_bounding_box(1, left),
    ]

    brtl_indices = [bottom, right, top, left]

    domain_cut, index = gmsh.model.occ.cut([(2, domain_loop)], inclusions)
    domain = domain_cut[0]
    gmsh.model.occ.synchronize()

    # Match boundaries after cut operation
    brtl_map = [
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1, bottom)),
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1, right)),
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1, top)),
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1, left)),
    ]

    new_bottom = brtl_indices[brtl_map.index(0)]
    new_right = brtl_indices[brtl_map.index(1)]
    new_top = brtl_indices[brtl_map.index(2)]
    new_left = brtl_indices[brtl_map.index(3)]

    gmsh.model.addPhysicalGroup(1, [new_bottom], boundaries.bottom.value, name=boundaries.bottom.name)
    gmsh.model.addPhysicalGroup(1, [new_right], boundaries.right.value, name=boundaries.right.name)
    gmsh.model.addPhysicalGroup(1, [new_top], boundaries.top.value, name=boundaries.top.name)
    gmsh.model.addPhysicalGroup(1, [new_left], boundaries.left.value, name=boundaries.left.name)
    gmsh.model.addPhysicalGroup(1, inclusion_curves, boundaries.inclusion.value, name=boundaries.inclusion.name)
    gmsh.model.addPhysicalGroup(2, [domain[1]], 666666, "Elements")

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=2)
    gmsh.write(f".meshes/ns_pipe_flow_{resolution}.msh")
    gmsh.finalize()

# %%
pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=None,
    return_coords_to_bounds=pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3,
)

pipemesh.dm.view()

x = pipemesh.N.x
y = pipemesh.N.y

# %% [markdown]
"""
## Visualization of Mesh
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)

    pl = pv.Plotter(window_size=(800, 250))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
    )

    pl.camera.position = (2.0, 0.5, 3)
    pl.camera.focal_point = (2.0, 0.5, 0.0)

    pl.show()

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
vorticity = uw.discretisation.MeshVariable("omega", pipemesh, 1, degree=1)

# %% [markdown]
"""
## Passive Tracers for Breakthrough Curve
"""

# %%
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(fill_param=0)

# Add new points at the inflow
new_points = 5000
new_coords = np.zeros((new_points, 2))
new_coords[:, 0] = 0.1
new_coords[:, 1] = np.linspace(0, 1.0, new_points)
passive_swarm.add_particles_with_coordinates(new_coords)

# Remove original swarm particles
with passive_swarm.access(passive_swarm._particle_coordinates):
    XY = passive_swarm._particle_coordinates.data
    XY[XY[:, 0] > 0.12] = 5.0

# %% [markdown]
"""
## Vorticity Projection
"""

# %%
nodal_vorticity_from_v = uw.systems.Projection(pipemesh, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(pipemesh.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3
nodal_vorticity_from_v.petsc_options.delValue("ksp_monitor")

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = uw.systems.Stokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

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

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

stokes.tolerance = 0.00001

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

stokes.penalty = 10
stokes.bodyforce = sympy.Matrix([0, 0])

# Boundary conditions: no-slip on grains and walls
stokes.add_dirichlet_bc((0.0, 0.0), "inclusion")
stokes.add_dirichlet_bc((0.0, 0.0), "top")
stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
stokes.add_dirichlet_bc((U0, 0.0), "left")

# %% [markdown]
"""
## Initial Solve
"""

# %%
stokes.solve(zero_init_guess=True)

continuous_pressure_projection = uw.systems.Projection(pipemesh, p_cont)
continuous_pressure_projection.uw_function = p_soln.sym[0]
continuous_pressure_projection.solve()

# %% [markdown]
"""
## Save Initial State
"""

# %%
import shutil

os.makedirs(expt_name, exist_ok=True)

pipemesh.write_timestep(
    "ExplicitGrains",
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=expt_name,
    index=0,
)

# %% [markdown]
"""
## Porosity and Average Velocity
"""

# %%
I = uw.maths.Integral(mesh=pipemesh, fn=1.0)
area = I.evaluate()
porosity = area / width

I.fn = v_soln.sym[0]
ave_velocity = I.evaluate() / area

uw.pprint(f"Porosity: {porosity:.4f}")
uw.pprint(f"Average velocity: {ave_velocity:.4f}")
uw.pprint(f"1/porosity: {1/porosity:.4f}")

# %% [markdown]
"""
## Tracer Advection
"""

# %%
time = 0
steps = 0
num_finishing = []

dt = 2 * stokes.estimate_dt()
max_time = 2.5

for step in range(0, int(max_time / dt)):
    passive_swarm.advection(v_soln.sym, dt)
    uw.pprint(f"{steps:04d} - t = {time:0.4f} - particles {passive_swarm.dm.getLocalSize()}")

    # Remove particles that exit the domain
    with passive_swarm.access(passive_swarm._particle_coordinates):
        p_no = passive_swarm.dm.getLocalSize()
        XY = passive_swarm._particle_coordinates.data
        XY[XY[:, 0] > 0.95 * width] = width + 1

    p_no_1 = passive_swarm.dm.getLocalSize()
    num_finishing.append(p_no - p_no_1)

    if steps % 50 == 0:
        passive_swarm.write_timestep(
            "Explicit_Grains",
            "passive_swarm",
            swarmVars=None,
            outputPath=expt_name,
            index=steps,
            force_sequential=True,
        )

    steps += 1
    time += dt

# %% [markdown]
"""
## Save Breakthrough Data
"""

# %%
with open(f"{expt_name}/Particle_numbers.txt", mode="w") as fp:
    for i, num in enumerate(num_finishing):
        print(i, num, file=fp)

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
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["Pc"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # Point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="forward",
        surface_streamlines=True,
        max_steps=100,
    )

    pl = pv.Plotter(window_size=(1500, 750))

    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.01 / U0,
        opacity=0.25,
        show_scalar_bar=False,
    )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="Pc",
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False,
    )

    pl.add_mesh(pvstream)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=4,
        opacity=1.0,
        show_scalar_bar=False,
    )

    pl.camera.position = (2.0, 0.5, 3)
    pl.camera.focal_point = (2.0, 0.5, 0.0)

    pl.show()

# %%
print(f"Explicit grain flow example complete: {steps} steps")
