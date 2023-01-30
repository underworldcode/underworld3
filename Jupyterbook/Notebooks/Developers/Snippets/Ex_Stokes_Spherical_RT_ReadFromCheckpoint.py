# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Rayleigh-Taylor in the sphere (from checkpoint)
#
# Read in a checkpointed RT (level set) problem and validate that it works.  Swarm values have to be reconstructed from the proxy field (slowly !)
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
# -

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
levelSet = uw.discretisation.MeshVariable(r"L", mesh, 1, degree=1)

# -


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable(r"\cal{L}", swarm, proxy_degree=2, num_components=1)
swarm.populate(fill_param=1)

# +
# if uw.mpi.rank == 0:
#     print(f"Read mesh checkpoints", flush=True)

# v_soln.load("./SampleData/Stokes_Sphere_RT_0.1_1.0s.U.5.h5")
# p_soln.load("./SampleData/Stokes_Sphere_RT_0.1_1.0s.P.5.h5")
# levelSet.load("./SampleData/Stokes_Sphere_RT_0.1_1.0s.proxy.calL.5.h5", data_name="calL")

# if uw.mpi.rank == 0:
#     print(f"Read mesh checkpoints ... done", flush=True)
# -


# ls SampleData/

# +
mesh_file = mesh_cache_file
v_file = "./SampleData/Stokes_Sphere_RT_0.1_1.0s.U.5.h5"
p_file = "./SampleData/Stokes_Sphere_RT_0.1_1.0s.P.5.h5"
l_file = "./SampleData/Stokes_Sphere_RT_0.1_1.0s.proxy.calL.5.h5"

# Here's a high resolution version:

mesh_file = "/Users/lmoresi/+Simulations/SphericalRT_eta01_eta1/Stokes_Sphere_RT_0.03_0.1s.mesh.0.h5"
v_file = "/Users/lmoresi/+Simulations/SphericalRT_eta01_eta1/Stokes_Sphere_RT_0.03_0.1s.U.90.h5"
p_file = "/Users/lmoresi/+Simulations/SphericalRT_eta01_eta1/Stokes_Sphere_RT_0.03_0.1s.P.90.h5"
l_file = "/Users/lmoresi/+Simulations/SphericalRT_eta01_eta1/Stokes_Sphere_RT_0.03_0.1s.proxy.calL.90.h5"
# -

print(uw.utilities.h5_scan(v_file))
print(uw.utilities.h5_scan(p_file))
print(uw.utilities.h5_scan(l_file))

# +
# uw.utilities.h5_scan(mesh_cache_file)

# v_soln1.load_from_vertex_checkpoint(mesh_file, v_file, 'U', 2, verbose=False)
# p_soln.load_from_vertex_checkpoint(mesh_file, p_file, 'P', 1, verbose=False)
# levelSet.load_from_vertex_checkpoint(mesh_file, l_file, 'calL', 1, verbose=False)


v_soln1.load_from_vertex_checkpoint(v_file, "U", 
                               vertex_mesh_file=mesh_file,
                               vertex_field=True, 
                               vertex_field_degree=2)

p_soln.load_from_vertex_checkpoint(p_file, "P", 
                               vertex_mesh_file=mesh_file,
                               vertex_field=True, 
                               vertex_field_degree=1)

levelSet.load_from_vertex_checkpoint(l_file, "calL", 
                               vertex_mesh_file=mesh_file,
                               vertex_field=True, 
                               vertex_field_degree=1)

# -



with mesh.access():
    print(levelSet.data.min(), levelSet.data.max())

mesh.data.shape

timing.print_table(display_fraction=0.999)


# + tags=[]
# check the solution

if uw.mpi.size == 1 and render:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [1.0, 1.0, 1.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    # pv.start_xvfb()

    mesh.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    with mesh.access():
        pvmesh.point_data["M"] = levelSet.data[:]
        # pvmesh.point_data["M"] = LS_h5_array - LSarray
        pvmesh.point_data["V"] = v_soln1.data[:,:]
        
    vmax = pvmesh.point_data["V"].max()
    pvmesh.point_data["V"] /= vmax

    # point sources at cell centres

    subsample = 50

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
    pl.add_mesh(pvstream, opacity=1.0)
    pl.add_mesh(contours, opacity=1.0, color="Yellow")

    pl.show(cpos="xz")
# -



