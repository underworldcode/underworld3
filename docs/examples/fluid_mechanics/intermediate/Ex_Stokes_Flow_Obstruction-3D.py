# %% [markdown]
"""
# 🔬 Stokes Flow Obstruction-3D

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** intermediate  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Stokes test: flow around a spherical inclusion (3D) with a free-slip bc
#

import nest_asyncio
nest_asyncio.apply()


# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy


# +
# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_model_resolution=0.1 etc

resolution = 5
refinement = 0

resolution = uw.options.getInt("model_resolution", default=resolution)
refinement = uw.options.getInt("model_refinement", default=refinement)

U0 = 1

# -

outdir = f"output/output_res_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# +
# https://jsdokken.com/src/tutorial_gmsh.html

from enum import Enum

class boundaries(Enum):
    inlet = 50
    outlet = 60
    walls = 90
    inclusion = 99
    All_Boundaries = 1001 

L, B, H, r = 2.5, 0.41, 0.41, 0.05  
centre = (0.5, 0.0, 0.2)

if uw.mpi.rank == 0:

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    
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
    
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)
    
    resolution = r / 10
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)
    
    inlet_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
    inlet_thre = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
    gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * resolution)
    gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * resolution)
    gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)
    
    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    gmsh.write("mesh3D.msh")
    gmsh.finalize()
    


pipemesh = uw.discretisation.Mesh(
    # f".meshes/ns_pipe_flow_{resolution}.msh",
    "mesh3D.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    # refinement_callback=pipemesh_mesh_refinement_callback,
    # return_coords_to_bounds= pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3)

pipemesh.view()

# +
# Some useful coordinate stuff

x = pipemesh.N.x
y = pipemesh.N.y
z = pipemesh.N.z

# relative to the centre of the cylinder
r = sympy.sqrt((x - centre[0]) ** 2 + (z - centre[2]) ** 2)

# need a unit_r_vec equivalent

inclusion_rvec = sympy.Matrix((x - centre[0], 0, z-centre[2]))
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# Boundary condition as specified in the diagram

Vb = U0
# -

v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
r_inc = uw.discretisation.MeshVariable("Rinc", pipemesh, 1, degree=1, continuous=True)

# +
# passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
# passive_swarm.populate(
#     fill_param=1,
# )

# # add new points at the inflow
# npoints = 100
# passive_swarm.dm.addNPoints(npoints)
# with passive_swarm.access(passive_swarm._particle_coordinates):
#     for i in range(npoints):
#         passive_swarm._particle_coordinates.data[-1 : -(npoints + 1) : -1, :] = np.array(
#             [0.01, 0.195] + 0.01 * np.random.random((npoints, 2))
#         )

# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

stokes = uw.systems.Stokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Constant visc

stokes.penalty = 10
stokes.bodyforce = sympy.Matrix([0, 0, 0])

with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evalf(r, pipemesh.data)

# surface_defn = sympy.exp(-(((r_inc.sym[0] - radius) / radius) ** 2) * hw)

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "walls")
stokes.add_dirichlet_bc((Vb, 0.0, 0.0), "inlet")
stokes.add_dirichlet_bc((None, 0.0, None), "inclusion")

stokes.add_natural_bc(1000*v_soln.sym.dot(inclusion_unit_rvec)*inclusion_unit_rvec, "inclusion")
stokes.tolerance = 1.0e-4
# -


stokes.view()

# +
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

# # # mg, multiplicative - very robust ... similar to gamg, additive

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# +
timing.reset()
timing.start()

stokes.solve(
    verbose=False)  # Stokes-like initial flow

timing.print_table(display_fraction=0.999)

# Note: resolution = 5
# Mac M4
# np1 -> 582s
# np4 -> 193s
# np8 -> 123s
# np10 -> 115s
# np12 -> 117s (this is the full number of fast cores)
# np16 -> 160s (this is the full number of cores, including slower ones)


# +
## Write the results

pipemesh.write_timestep(
    f"StokesInclusion3D_np{uw.mpi.size}",
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath="output",
    index=0)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres
    skip=7
    points = np.zeros((pipemesh._centroids[::skip].shape[0], 3))
    points[:, 0] = pipemesh._centroids[::skip, 0]
    points[:, 1] = pipemesh._centroids[::skip, 1]
    points[:, 2] = pipemesh._centroids[::skip, 2]
    point_cloud = pv.PolyData(points)

    # passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    # points = vis.swarm_to_pv_cloud(passive_swarm)
    # point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_arrows(velocity_points.points, 
    #               velocity_points.point_data["V"], 
    #               mag=0.025 / U0, 
    #               opacity=0.75)

    # pl.add_points(
    #     point_cloud,
    #     color="Black",
    #     render_points_as_spheres=False,
    #     point_size=5,
    #     opacity=0.66,
    # )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=0.75,
        show_scalar_bar=False)
    
    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.01 / U0, opacity=0.75)
    # pl.add_mesh(pvstream, show_scalar_bar=False)

    # pl.add_points(
    #     passive_swarm_points,
    #     color="Black",
    #     render_points_as_spheres=True,
    #     point_size=2,
    #     opacity=0.25,
    # )

    pl.show()
# -
0/0 # breakpoint

# +
# Optional - compare the solution to other decompositions

# +
v_soln.read_timestep(
    "StokesInclusion3D_np16",
    "U",
    0,
    outputPath="output",
    verbose=True)

p_soln.read_timestep(
    "StokesInclusion3D_np16",
    "P",
    0,
    outputPath="output",
    verbose=True)


# -


