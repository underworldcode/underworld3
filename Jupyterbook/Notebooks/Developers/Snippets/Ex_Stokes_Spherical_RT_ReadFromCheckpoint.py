# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Rayleigh-Taylor in the sphere (from checkpoint)
#
# Read in a checkpointed RT (level set) problem and validate that it works. We need to use the identical mesh and variables need to be of the same order as the checkpoint. Swarm values are reconstructed from the proxy field. 
#

# +
# Enable timing (before uw imports)

import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

render = True


# +
lightIndex = 0
denseIndex = 1

r_layer = 0.7
r_o = 1.0
r_i = 0.5

# +
# Sample data is for a spherical mesh 
# Not sure if gmsh is deterministic but the mesh is too large to put in github 
# (hint - rerun the sample data to be smaller !)

cell_size = 0.05
res = cell_size


# +
from underworld3 import timing

timing.reset()
timing.start()

# +
from pathlib import Path
from underworld3.coordinates import CoordinateSystemType

mesh_cache_file = f"./SampleData/Stokes_Sphere_RT_0.1_1.0s.mesh.0.h5"
path = Path(mesh_cache_file)

if path.is_file():
    if uw.mpi.rank == 0:
        print(f"Re-using mesh: {mesh_cache_file}", flush=True)

    mesh = uw.discretisation.Mesh(
        mesh_cache_file,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        qdegree=2,
    )
    
    mesh.dm.view()
    
else:
    print("No valid mesh file found")




# +
v_soln = uw.discretisation.MeshVariable(r"U", mesh, mesh.dim, degree=2)
v_soln1 = uw.discretisation.MeshVariable(r"U1", mesh, mesh.dim, degree=1)

p_soln = uw.discretisation.MeshVariable(r"P", mesh, 1, degree=1)
levelSet = uw.discretisation.MeshVariable(r"L", mesh, 1, degree=2)
levelSet1 = uw.discretisation.MeshVariable(r"L1", mesh, 1, degree=1)
# -


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable(r"\cal{L}", swarm, proxy_degree=2, num_components=1)
swarm.populate(fill_param=1)

# +
if uw.mpi.rank == 0:
    print(f"Read mesh checkpoints", flush=True)

v_soln.load("./SampleData/Stokes_Sphere_RT_0.1_1.0s.U.5.h5")
p_soln.load("./SampleData/Stokes_Sphere_RT_0.1_1.0s.P.5.h5")
levelSet.load("./SampleData/Stokes_Sphere_RT_0.1_1.0s.proxy.calL.5.h5", data_name="calL")

if uw.mpi.rank == 0:
    print(f"Read mesh checkpoints ... done", flush=True)


# +
#[-0.24505581821483266, 0.34464701946023985]
# -

with mesh.access():
    print(levelSet.data.min(), levelSet.data.max())

project_to_mesh = uw.systems.Projection(mesh, levelSet1, solver_name="meshmapper")
project_to_mesh.uw_function = levelSet.sym[0]
project_to_mesh.smoothing = 1.0e-3
project_to_mesh.add_dirichlet_bc(levelSet.sym[0], "Upper", 0)
project_to_mesh.add_dirichlet_bc(levelSet.sym[0], "Lower", 0)
project_to_mesh.solve()

with mesh.access():
    print(levelSet1.data.min(), levelSet1.data.max())

levelSet.simple_save("testLS")

LSarray = uw.function.evaluate(levelSet.sym[0], mesh.data, mesh.N)

project_v_to_mesh = uw.systems.Vector_Projection(mesh, v_soln1, solver_name="meshmapper_v")
project_v_to_mesh.uw_function = v_soln.sym
project_v_to_mesh.smoothing = 1.0e-3
project_v_to_mesh.solve()

# +
# This takes forever ... it's needed for an actual restart but not for visualisation

# if uw.mpi.rank == 0:
#     print(f"Set level-set values on swarm ...", flush=True)

# with swarm.access(material):
#     material.data[:, 0] = uw.function.evaluate(levelSet.sym[0], swarm.particle_coordinates.data)

# if uw.mpi.rank == 0:
#     print(f"Set level-set values on swarm ... done", flush=True)

# +
# if level_set_checkpoint == "":
#     with swarm.access(material):
#         r = np.sqrt(
#             swarm.particle_coordinates.data[:, 0] ** 2
# + swarm.particle_coordinates.data[:, 1] ** 2



# + (swarm.particle_coordinates.data[:, 2] - offset) ** 2
#         )

#         material.data[:, 0] = r - r_layer
        
# else:
#     path = Path(level_set_checkpoint)
#     if path.is_file():
#         if uw.mpi.rank == 0:
#             print(f"Found checkpoint file: {level_set_checkpoint}", flush=True)
            
#         levelSet.load(level_set_checkpoint, data_name="calL")
        
#         with swarm.access(material):
#             material.data[:, 0] = uw.function.evaluate(levelSet.sym[0], swarm.particle_coordinates.data)
    
# -

import h5py
h5 = h5py.File("./SampleData/Stokes_Sphere_RT_0.1_1.0s.proxy.calL.5.h5", "r")
h5ls = h5py.File("testLS")

h5['fields']['calL']

h5ls['vertex_fields']['L_P2']

LS_h5_array = h5['vertex_fields']['calL_P2'][()]
LS_h5_array1 = h5ls['vertex_fields']['L_P2'][()]

LSarray

LS_h5_array

mesh.dm.globalVectorLoad()

with mesh.access():
    LSarray1 = levelSet1.data[:,0].copy() 

LSarray1



# +
# check the solution

if uw.mpi.size == 1 and render:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [1.0, 1.0, 1.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    # pv.start_xvfb()

    mesh.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    with mesh.access():
        pvmesh.point_data["M"] = levelSet1.data[:]
        # pvmesh.point_data["M"] = LS_h5_array - LSarray
        pvmesh.point_data["V"] = v_soln1.data[:,:]
        
    vmax = pvmesh.point_data["V"].max()
    pvmesh.point_data["V"] /= vmax

    # point sources at cell centres

    subsample = 10

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

    contours = pvmesh.contour(isosurfaces=[0.0], scalars="M")

    pl = pv.Plotter(window_size=(1000, 1000))

    pl.add_mesh(pvmesh, "Gray", "wireframe")
    # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

    # pl.add_mesh(pvstream, opacity=1.0)
    # pl.add_mesh(pvmesh, cmap="Blues_r", edge_color="Gray", show_edges=True, scalars="rho", opacity=0.25)

    pl.add_mesh(contours, opacity=0.75, color="Yellow")

    # pl.add_points(spoint_cloud, cmap="Reds_r", scalars="M", render_points_as_spheres=True, point_size=2, opacity=0.3)
    # pl.add_points(pdata)

    pl.show(cpos="xz")
# -


