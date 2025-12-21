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
# Stokes Flow Around Spherical Inclusion (3D)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

3D Stokes flow around a spherical inclusion in a channel (pipe flow).
The inclusion has a free-slip boundary condition implemented via penalty.
Uses gmsh for mesh generation with local refinement around the obstacle.

## Key Concepts

- **3D Stokes flow**: Full 3D momentum equation
- **gmsh mesh generation**: 3D channel with spherical hole
- **Free-slip on inclusion**: Penalty-based natural BC
- **Local refinement**: Finer mesh near obstacle
- **Parallel scaling**: Performance tests with different MPI ranks

## Parameters

- `uw_resolution`: Base mesh resolution
- `uw_refinement`: Mesh refinement level
- `uw_u0`: Inlet velocity magnitude
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

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Flow_Obstruction-3D.py -uw_resolution 5
python Ex_Stokes_Flow_Obstruction-3D.py -uw_refinement 1
```
"""

# %%
params = uw.Params(
    uw_resolution = 5,            # Base mesh resolution
    uw_refinement = 0,            # Mesh refinement levels
    uw_u0 = 1.0,                  # Inlet velocity magnitude
    uw_length = 2.5,              # Channel length
    uw_width = 0.41,              # Channel width
    uw_height = 0.41,             # Channel height
    uw_radius = 0.05,             # Inclusion radius
)

resolution = int(params.uw_resolution)
refinement = int(params.uw_refinement)
U0 = params.uw_u0

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
## 3D Mesh Generation with gmsh

Channel with cylindrical hole for the spherical inclusion.
"""

# %%
from enum import Enum


class boundaries(Enum):
    inlet = 50
    outlet = 60
    walls = 90
    inclusion = 99
    All_Boundaries = 1001


L = params.uw_length
B = params.uw_width
H = params.uw_height
r = params.uw_radius
centre = (0.5, 0.0, 0.2)

if uw.mpi.rank == 0:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    gmsh.model.add("DFG 3D")

    channel = gmsh.model.occ.addBox(0, 0, 0, L, B, H)
    cylinder = gmsh.model.occ.addCylinder(0.5, 0, 0.2, 0, B, 0, r)
    fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)])

    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    assert volumes == fluid[0]
    fluid_marker = 11

    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid_Volume")

    surfaces = gmsh.model.occ.getEntities(dim=2)
    walls = []
    obstacles = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0, B / 2, H / 2]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundaries.inlet.value)
            inlet = surface[1]
            gmsh.model.setPhysicalName(surface[0], boundaries.inlet.value, boundaries.inlet.name)
        elif np.allclose(com, [L, B / 2, H / 2]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundaries.outlet.value)
            gmsh.model.setPhysicalName(surface[0], boundaries.outlet.value, boundaries.outlet.name)
        elif (
            np.isclose(com[2], 0)
            or np.isclose(com[1], B)
            or np.isclose(com[2], H)
            or np.isclose(com[1], 0)
        ):
            walls.append(surface[1])
        else:
            obstacles.append(surface[1])

    gmsh.model.addPhysicalGroup(2, walls, boundaries.walls.value)
    gmsh.model.setPhysicalName(2, boundaries.walls.value, boundaries.walls.name)
    gmsh.model.addPhysicalGroup(2, obstacles, boundaries.inclusion.value)
    gmsh.model.setPhysicalName(2, boundaries.inclusion.value, boundaries.inclusion.name)

    # Mesh refinement near inclusion
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)

    mesh_resolution = r / 10
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", mesh_resolution)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * mesh_resolution)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

    inlet_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
    inlet_thre = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
    gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * mesh_resolution)
    gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * mesh_resolution)
    gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write("mesh3D.msh")
    gmsh.finalize()

# %%
pipemesh = uw.discretisation.Mesh(
    "mesh3D.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    boundaries=boundaries,
    qdegree=3,
)

pipemesh.view()

# %% [markdown]
"""
## Coordinate System
"""

# %%
x = pipemesh.N.x
y = pipemesh.N.y
z = pipemesh.N.z

# Relative to the centre of the cylinder
r_cyl = sympy.sqrt((x - centre[0]) ** 2 + (z - centre[2]) ** 2)

# Unit radial vector for inclusion
inclusion_rvec = sympy.Matrix((x - centre[0], 0, z - centre[2]))
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

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
## Stokes Solver
"""

# %%
stokes = uw.systems.Stokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

stokes.penalty = 10
stokes.bodyforce = sympy.Matrix([0, 0, 0])

with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evalf(r_cyl, pipemesh.data)

# Boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "walls")
stokes.add_dirichlet_bc((U0, 0.0, 0.0), "inlet")
stokes.add_dirichlet_bc((None, 0.0, None), "inclusion")

# Free-slip on inclusion via penalty
stokes.add_natural_bc(1000 * v_soln.sym.dot(inclusion_unit_rvec) * inclusion_unit_rvec, "inclusion")
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

# Performance notes (Mac M4, resolution=5):
# np1 -> 582s, np4 -> 193s, np8 -> 123s, np10 -> 115s
# np12 -> 117s (full fast cores), np16 -> 160s (all cores)

# %% [markdown]
"""
## Save Results
"""

# %%
pipemesh.write_timestep(
    f"StokesInclusion3D_np{uw.mpi.size}",
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath="output",
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

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # Point sources at cell centres
    skip = 7
    points = np.zeros((pipemesh._centroids[::skip].shape[0], 3))
    points[:, 0] = pipemesh._centroids[::skip, 0]
    points[:, 1] = pipemesh._centroids[::skip, 1]
    points[:, 2] = pipemesh._centroids[::skip, 2]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=0.75,
        show_scalar_bar=False,
    )

    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.01 / U0, opacity=0.75)

    pl.show()

# %%
print(f"3D Stokes inclusion example complete: resolution {resolution}, refinement {refinement}")
