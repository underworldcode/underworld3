# %% [markdown]
"""
# 🎓 Explicit Flow Grains

**PHYSICS:** porous_flow  
**DIFFICULTY:** advanced  
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
#     formats: py:light,ipynb
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

# # Navier Stokes test: flow around a circular inclusion (2D)
#
# No slip conditions
#
# Note ...
#
#
#

# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import sympy

# import psutil
# pid = os.getpid()
# python_process = psutil.Process(pid)
# print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# +
# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_resolution=0.1 etc

resolution = uw.options.getInt("model_resolution", default=20)
refinement = uw.options.getInt("model_refinement", default=0)
model = uw.options.getInt("model_number", default=1)
maxsteps = uw.options.getInt("max_steps", default=201)
restart_step = uw.options.getInt("restart_step", default=-1)
# -

outdir = f"output/output_res_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# +
width = 4.0
height = 1.0
resolution = 14
expt_name = "Expt_3"

csize = 1.0 / resolution
csize_circle = 0.66 * csize
res = csize_circle

width = 4.0
height = 1.0

rows = 5
columns = int((width-1)*rows)
radius_0 = 0.075
variation = 0.075

U0 = 1.0
# -



# +
## Pure gmsh version

import pygmsh
from enum import Enum

## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh

class boundaries(Enum):
    bottom = 1
    right = 2
    left  = 3
    top = 4
    inclusion = 5
    All_Boundaries = 1001 

# Mesh a 2D pipe with circular holes

## Restore inflow samples to inflow points
def pipemesh_return_coords_to_bounds(coords):
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
    
    dy = 1.0/(rows+0.5)
    dx = dy*1.2
    
    for row in range(0,rows):
        for col in range(0,columns):
    
            y = dy*(row+0.75) 
            x = 0.25 + dx * col + ( row%2 ) * 0.5 * dx
            r = radius_0  +  variation * (rrand.random()-0.5)
    
            i_points = [
                gmsh.model.occ.add_point(x,y,0.0,   meshSize=csize_circle),
                gmsh.model.occ.add_point(x,y+r,0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x-r,y,0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x,y-r,0.0, meshSize=csize_circle),
                gmsh.model.occ.add_point(x+r,y,0.0, meshSize=csize_circle)
            ]
            
            i_quarter_circles = [
                gmsh.model.occ.add_circle_arc(i_points[1], i_points[0], i_points[2]),
                gmsh.model.occ.add_circle_arc(i_points[2], i_points[0], i_points[3]),
                gmsh.model.occ.add_circle_arc(i_points[3], i_points[0], i_points[4]),
                gmsh.model.occ.add_circle_arc(i_points[4], i_points[0], i_points[1]),
            ]
           
            inclusion_loop = gmsh.model.occ.add_curve_loop(i_quarter_circles)
            inclusion = gmsh.model.occ.add_plane_surface([inclusion_loop])            
    
            inclusions.append((2,inclusion))
            inclusion_curves.append(i_quarter_circles[0])
            inclusion_curves.append(i_quarter_circles[1])
            inclusion_curves.append(i_quarter_circles[2])
            inclusion_curves.append(i_quarter_circles[3])
    
            gmsh.model.occ.synchronize()
    
    corner_points = []
    corner_points.append(gmsh.model.occ.add_point(0.0, 0.0, 0.0,  csize))
    corner_points.append(gmsh.model.occ.add_point(width, 0.0, 0.0, csize))
    corner_points.append(gmsh.model.occ.add_point(width, 1.0, 0.0,  csize))
    corner_points.append(gmsh.model.occ.add_point(0.0, 1.0, 0.0, csize))
    
    bottom = gmsh.model.occ.add_line(corner_points[0], corner_points[1])
    right = gmsh.model.occ.add_line(corner_points[1], corner_points[2])
    top = gmsh.model.occ.add_line(corner_points[2], corner_points[3])
    left =  gmsh.model.occ.add_line(corner_points[3], corner_points[0])
    
    # gmsh.model.occ.synchronize()
    
    domain_loop = gmsh.model.occ.add_curve_loop((bottom, right, top, left))
    gmsh.model.occ.add_plane_surface([domain_loop])
    
    gmsh.model.occ.synchronize()
    
    # The ordering of the boundaries is scrambled in the 
    # occ.cut stage, save the bb and match the boundaries afterwards.
    
    brtl_bboxes = [ 
               gmsh.model.get_bounding_box(1,bottom),
               gmsh.model.get_bounding_box(1,right),
               gmsh.model.get_bounding_box(1,top),
               gmsh.model.get_bounding_box(1,left) 
            ]
    
    brtl_indices = [bottom, right, top, left]
     
    domain_cut, index = gmsh.model.occ.cut([(2,domain_loop)], inclusions)
    domain = domain_cut[0]
    gmsh.model.occ.synchronize()

    ## There is surely a better way !
  
    brtl_indices = [bottom, right, top, left]
    brtl_map = [
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,bottom)), 
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,right)),
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,top)), 
        brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,left))
    ]
    
    new_bottom = brtl_indices[brtl_map.index(0)]
    new_right  = brtl_indices[brtl_map.index(1)]
    new_top    = brtl_indices[brtl_map.index(2)]
    new_left   = brtl_indices[brtl_map.index(3)]
      
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

pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=None,
    return_coords_to_bounds= pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3)

pipemesh.dm.view()

# Some useful coordinate stuff

x = pipemesh.N.x
y = pipemesh.N.y



# +
# check the mesh if in a notebook / serial

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
        opacity=1.0)

    pl.camera.position = (2.0, 0.5, 3)
    pl.camera.focal_point=(2.0,0.5,0.0)

    pl.show(jupyter_backend='html')
# -



v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
vorticity = uw.discretisation.MeshVariable("omega", pipemesh, 1, degree=1)



# +
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(
    fill_param=0)

# add new points at the inflow
new_points = 5000
new_coords = np.zeros((new_points,2))
new_coords[:,0] = 0.1
new_coords[:,1] = np.linspace(0, 1.0, new_points)
passive_swarm.add_particles_with_coordinates(new_coords)    

## Blast away all the original swarm particles

with passive_swarm.access(passive_swarm._particle_coordinates):
    XY = passive_swarm._particle_coordinates.data
    XY[XY[:,0] > 0.12] = 5.0


# -
nodal_vorticity_from_v = uw.systems.Projection(pipemesh, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(pipemesh.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3
nodal_vorticity_from_v.petsc_options.delValue("ksp_monitor")

# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

stokes = uw.systems.Stokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False)

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

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

stokes.tolerance = 0.00001


stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Constant visc

stokes.penalty = 10
stokes.bodyforce = sympy.Matrix([0, 0])


# Velocity boundary conditions

stokes.add_dirichlet_bc(
    (0.0, 0.0),
    "inclusion")

# Gamma = pipemesh.Gamma
# GammaNorm = uw.function.expression(r"|\Gamma|", sympy.sqrt(Gamma.dot(Gamma)), "Scaling for surface normals")
# GammaN = Gamma / GammaNorm
# stokes.add_natural_bc(100000 * v_soln.sym.dot(GammaN) * GammaN, "inclusion")

stokes.add_dirichlet_bc((0.0, 0.0), "top")
stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
stokes.add_dirichlet_bc((U0, 0.0), "left")
# -


stokes.solve(zero_init_guess=True)

continuous_pressure_projection = uw.systems.Projection(pipemesh, p_cont)
continuous_pressure_projection.uw_function = p_soln.sym[0]
continuous_pressure_projection.solve()

# +
## Write out this file and data 
import os, shutil

os.makedirs(expt_name, exist_ok=True)
shutil.copy("Ex_Explicit_Flow_Grains.py", expt_name)


pipemesh.write_timestep(
    "ExplicitGrains",
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=expt_name,
    index=0)
# +
I = uw.maths.Integral(mesh=pipemesh, fn=1.0)
area = I.evaluate()
porosity = area / 4 

I.fn = v_soln.sym[0]
ave_velocity = I.evaluate() / area
# -

ave_velocity

1 / porosity


0/0

time=0
steps = 0
num_finishing = []

# +
dt = 2 * stokes.estimate_dt()

for step in range(0, int(2.5/dt)):
    
    passive_swarm.advection(v_soln.sym, dt)
    print(f"{steps:04d} - t = {time:0.4f} - particles {passive_swarm.dm.getLocalSize()}")

    with passive_swarm.access(passive_swarm._particle_coordinates):
        p_no = passive_swarm.dm.getLocalSize()
        XY = passive_swarm._particle_coordinates.data
        XY[XY[:,0] > 0.95 * width] = width + 1
        
    p_no_1 = passive_swarm.dm.getLocalSize()
    num_finishing.append(p_no - p_no_1)

    if steps%50 == 0:
        passive_swarm.write_timestep(
            "Explicit_Grains",
            "passive_swarm",
            swarmVars=None,
            outputPath=expt_name,
            index=steps,
            force_sequential=True)

    
    steps += 1
    time += dt

# -

with open("Particle_numbers.txt",  mode="w") as fp:
    for i, num in enumerate(num_finishing):
        print(i, num, file=fp)

# +
# check the mesh if in a notebook / serial

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

    # point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", 
        surface_streamlines=True, max_steps=100
    )

    points = vis.swarm_to_pv_cloud(passive_swarm)
    point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(1500, 750))

    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["V"], 
                  mag=0.01 / U0, opacity=0.25, show_scalar_bar=False)


    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="Pc",
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False)
    
    pl.add_mesh(pvstream)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=4,
        opacity=1.0,
        show_scalar_bar=False)
    
    pl.camera.position = (2.0, 0.5, 3)
    pl.camera.focal_point=(2.0,0.5,0.0)

    pl.show(jupyter_backend="html")
# -

pl.screenshot(window_size=(2000,500), filename=f"{expt_name}/ExplicitGrains.{step}.png")
pass


