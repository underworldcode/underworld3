# %% [markdown]
"""
# 🔬 Stokes Flow Internal BC

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

# # Stokes test: flow around a circular inclusion (2D) with a free-slip bc
#
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

resolution = 20
refinement = 0

resolution = uw.options.getInt("model_resolution", default=resolution)
refinement = uw.options.getInt("model_refinement", default=refinement)

U0 = 1

# -

outdir = f"output/output_res_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# +
import pygmsh
from enum import Enum

## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh

class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left  = 4
    inclusion = 5
    All_Boundaries = 1001 

# Mesh a 2D pipe with a circular hole

csize = 1.0 / resolution
csize_circle = 0.25 * csize
res = csize_circle

width = 2.2
height = 0.41
radius = 0.05
centre = (0.2, 0.2)

def pipemesh_mesh_refinement_callback(dm):
    r_p = radius

    # print(f"Refinement callback - spherical", flush=True)

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

## Restore inflow samples to inflow points
def pipemesh_return_coords_to_bounds(coords):
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = 0.0001

    return coords

  

if uw.mpi.rank == 0:
    # Generate local mesh on boss process

    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (centre[0], centre[1], 0.0),
            radius,
            make_surface=True,
            mesh_size=csize_circle)
        domain = geom.add_rectangle(
            xmin=0.0,
            ymin=0.0,
            xmax=width,
            ymax=height,
            z=0,
            # holes=[inclusion],
            mesh_size=csize)

        geom.synchronize()

        for cloop in inclusion.curve_loop.curves:
            geom.in_surface(cloop, domain)

        geom.add_physical(domain.surface.curve_loop.curves[0], label=boundaries.bottom.name)
        geom.add_physical(domain.surface.curve_loop.curves[1], label=boundaries.right.name)
        geom.add_physical(domain.surface.curve_loop.curves[2], label=boundaries.top.name)
        geom.add_physical(domain.surface.curve_loop.curves[3], label=boundaries.left.name)
        geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry(f".meshes/ns_pipe_flow_{resolution}.msh")

pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=pipemesh_mesh_refinement_callback,
    return_coords_to_bounds= pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3)

pipemesh.view()


# Some useful coordinate stuff

x = pipemesh.N.x
y = pipemesh.N.y

# relative to the centre of the inclusion
r = sympy.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2)
th = sympy.atan2(y - 0.2, x - 0.2)

# need a unit_r_vec equivalent

inclusion_rvec = sympy.Matrix((x - centre[0], y-centre[1]))
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# Boundary condition as specified in the diagram

Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41**2
# -


v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
r_inc = uw.discretisation.MeshVariable("Rinc", pipemesh, 1, degree=1, continuous=True)

# +
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(
    fill_param=1)

# add new points at the inflow
npoints = 100
passive_swarm.dm.addNPoints(npoints)
with passive_swarm.access(passive_swarm._particle_coordinates):
    for i in range(npoints):
        passive_swarm._particle_coordinates.data[-1 : -(npoints + 1) : -1, :] = np.array(
            [0.01, 0.195] + 0.01 * np.random.random((npoints, 2))
        )

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

stokes.penalty = 100
stokes.bodyforce = sympy.Matrix([0, 0])

hw = 1000.0 / res
with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evalf(r, pipemesh.data, pipemesh.N)

surface_defn = sympy.exp(-(((r_inc.sym[0] - radius) / radius) ** 2) * hw)

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0), "top")
stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
stokes.add_dirichlet_bc((Vb, 0.0), "left")

stokes.add_natural_bc(1000*v_soln.sym.dot(inclusion_unit_rvec)*inclusion_unit_rvec, "inclusion")
# -


stokes.tolerance = 1.0e-4


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

# +
I = uw.maths.Integral(pipemesh, v_soln.sym.dot(v_soln.sym))
Vmag = np.sqrt(I.evaluate())

uw.pprint(0, f"Velocity magnitude: {Vmag}")

I.fn = p_soln.sym[0]**2
Pmag = np.sqrt(I.evaluate())

uw.pprint(0, f"Pressure magnitude: {Pmag}")


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
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    points = vis.swarm_to_pv_cloud(passive_swarm)
    point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["V"], 
                  mag=0.025 / U0, 
                  opacity=0.75)

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
        opacity=1.0,
        show_scalar_bar=False)
    
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025 / U0, opacity=0.75)
    pl.add_mesh(pvstream, show_scalar_bar=False)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=2,
        opacity=0.25)

    pl.show()
# -



