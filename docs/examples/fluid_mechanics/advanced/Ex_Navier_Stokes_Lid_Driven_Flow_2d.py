# %% [markdown]
"""
# 🎓 Navier Stokes Lid Driven Flow 2d

**PHYSICS:** fluid_mechanics  
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
# %%
import os

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy

idx = 0
prev = 0

# %% [markdown]
# # Lid-driven cavity flow
# By: Juan Carlos Graciosa
# - A simple example problem for Navier-Stokes
# - Has option for finer mesh resolution at the borders
# - Still needs some fine-tuning to reach benchmark values from literature

# %%
Re_num = 100
resolution = 16
wall_res_factor = 0.5 # controls finer resolution close to walls
maxsteps = 1
save_every = 5
tol = 1e-8

# lid driven flow model parameters
vel   = 1.0    # top boundary horizontal velocity
dt_ns = 0.01  # time step - constant for now

# mesh and solver controls
refinement = 0
qdeg = 3
Vdeg = 3
Pdeg = Vdeg - 1
Pcont = False
ns_order = 1

# control flags
show_vis = True       # set to True if we want to display visualizations
gen_mesh = True       # if we want to generate a mesh with finer resolution close to the walls
save_output = False   # save mesh and output fields

# %%
outfile = f"NS_LDF_run{idx}"
outdir = f"./NS_LDF_res{resolution}_Re{Re_num}"

# %%
if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    infile = f"NS_LDF_run{prev_idx}"

if uw.mpi.rank == 0 and if save_output:
    os.makedirs(outdir, exist_ok=True)

# %%
# dimensional quantities
width = 1.
height = 1.
fluid_rho = 1.

# %%
# cell size calculation
csize = height / resolution
csize_walls = wall_res_factor * csize
res = csize_walls

# %%
import gmsh
from enum import Enum

# create mesh with finer cells at the walls
class boundaries(Enum):
    Bottom = 1
    Right = 2
    Top = 3
    Left  = 4
    All_Boundaries = 1001

# mesh with boundary refinement
if uw.mpi.rank == 0 and infile is None and gen_mesh:
    gmsh.initialize()
    gmsh.model.add("box_fine_bdndry")

    # points in domain
    p1 = gmsh.model.geo.addPoint(0.      , 0.    , 0., csize)
    p2 = gmsh.model.geo.addPoint(width   , 0.    , 0., csize)
    p3 = gmsh.model.geo.addPoint(width   , height, 0., csize)
    p4 = gmsh.model.geo.addPoint(0.      , height, 0., csize)

    l1 = gmsh.model.geo.addLine(p1, p2, tag = boundaries.Bottom.value)
    l2 = gmsh.model.geo.addLine(p2, p3, tag = boundaries.Right.value )
    l3 = gmsh.model.geo.addLine(p3, p4, tag = boundaries.Top.value   )
    l4 = gmsh.model.geo.addLine(p4, p1, tag = boundaries.Left.value  )

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # do refinement of boundary elements
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [l1, l2, l3, l4])
    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)

    # threshold field
    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", csize_walls)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", csize)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 0.3)

    # background mesh
    bg_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(bg_field, "FieldsList", [thresh_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

    # set these to zero
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Delaunay algorithm is better for complex mesh size fields
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.geo.synchronize()

    # add physical groups for boundaries
    gmsh.model.add_physical_group(1, [l1], l1)
    gmsh.model.set_physical_name(1, l1, boundaries.Bottom.name)
    gmsh.model.add_physical_group(1, [l2], l2)
    gmsh.model.set_physical_name(1, l2, boundaries.Right.name)
    gmsh.model.add_physical_group(1, [l3], l3)
    gmsh.model.set_physical_name(1, l3, boundaries.Top.name)
    gmsh.model.add_physical_group(1, [l4], l4)
    gmsh.model.set_physical_name(1, l4, boundaries.Left.name)

    gmsh.model.add_physical_group(2, [surface], 99999)
    gmsh.model.set_physical_name(2, 99999, "Elements")

    gmsh.model.mesh.generate(2)
    gmsh.write(f".meshes/graded_mesh_{resolution}.msh")
    gmsh.finalize()


def box_return_coords_to_bounds(coords):

    x00s = coords[:, 0] < 0
    x01s = coords[:, 0] > width
    x10s = coords[:, 1] < 0
    x11s = coords[:, 1] > height

    coords[x00s, 0] = 0
    coords[x01s, 0] = width
    coords[x10s, 1] = 0
    coords[x11s, 1] = height

    return coords

# %%
if gen_mesh:
    meshbox = uw.discretisation.Mesh(
        f".meshes/graded_mesh_{resolution}.msh",
        markVertices = True,
        useMultipleTags = True,
        useRegions = True,
        refinement = refinement,
        refinement_callback = None,
        return_coords_to_bounds = box_return_coords_to_bounds,
        boundaries = boundaries,
        qdegree = qdeg)
else:
    meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords = (0.0, 0.0),
                                                maxCoords = (width, height),
                                                cellSize = 1.0 / resolution,
                                                regular = False,
                                                qdegree = qdeg
                                            )


# %%
meshbox.dm.view()

# %%
# calculate courant number
courant = vel * dt_ns / meshbox.get_min_radius()

print("Courant number: ", courant)

# %%
if uw.mpi.size == 1 and show_vis and uw.is_notebook:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False)

    pl.show(cpos="xy")

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree = Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree = Pdeg, continuous = Pcont)

# %%
# passive_swarm = uw.swarm.Swarm(mesh=pipemesh)

if infile is None:
    pass
else:
    v_soln.read_timestep(data_filename = infile, data_name = "U", index = maxsteps, outputPath = outdir)
    p_soln.read_timestep(data_filename = infile, data_name = "P", index = maxsteps, outputPath = outdir)

# %%
navier_stokes = uw.systems.NavierStokesSLCN(
    meshbox,
    velocityField = v_soln,
    pressureField = p_soln,
    rho = fluid_rho,
    verbose = True,
    order=ns_order)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Constant visc
navier_stokes.constitutive_model.Parameters.viscosity = 1./Re_num

navier_stokes.penalty = 0
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc((vel, 0.0), "Top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Left")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Right")

navier_stokes.tolerance = tol

# %%
# PETSc solver parameters

navier_stokes.petsc_options["snes_monitor"] = None
navier_stokes.petsc_options["ksp_monitor"] = None

navier_stokes.petsc_options["snes_type"] = "newtonls"
navier_stokes.petsc_options["ksp_type"] = "fgmres"

navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %%
ts = 0
elapsed_time = 0.0
timeVal =  np.zeros(maxsteps)*np.nan      # time values

# %%
for step in range(0, maxsteps):

    delta_t = dt_ns

    navier_stokes.solve(timestep=delta_t, zero_init_guess = True)

    elapsed_time += delta_t
    timeVal[step] = elapsed_time

    uw.pprint("Timestep {}, t {}, dt {}".format(ts, elapsed_time, delta_t))

    ts += 1

# %%
if save_output:
    meshbox.write_timestep(
        outfile,
        meshUpdates = True,
        meshVars = [p_soln, v_soln],
        outputPath = outdir,
        index = ts)

# %%
if uw.mpi.size == 1 and show_vis and uw.is_notebook:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pvmesh.point_data["V"]      = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["P"]      = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points                 = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    sargs = dict(title = "Pressure", vertical = False, font_family = "arial", position_x=0.2, position_y = 0.05)

    pl = pv.Plotter(window_size=(1000, 750), notebook = True, off_screen = True)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1,
        line_width = 0.0,
        scalar_bar_args = sargs
    )

    pl.add_arrows(
        velocity_points.points[::2],
        velocity_points.point_data["V"][::2],
        mag=0.2,
        color="k")

    pl.camera_position = "xy"
    # pl.screenshot(
    #     filename=f"{outdir}/{fname}",
    #     window_size=(2560, 1280),
    #     return_img=False,
    # )
    pl.show()


# %%



