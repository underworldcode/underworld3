# %% [markdown]
"""
# 🔬 Navier Stokes Visualise Disc

**PHYSICS:** utilities  
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
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# # Navier Stokes test: flow in an disk / annulus with a moving boundary (2D)
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

# + language="sh"
#
# ls -tr /Users/lmoresi/+Simulations/NS_benchmarks/NS_Disc | tail -5
# #ls -tr /Users/lmoresi/+Underworld/underworld3/JupyterBook/Notebooks/Examples-NavierStokes/output_res_033/*mesh*h5 | tail -10
#
# -


# ls -tr /Users/lmoresi/+Simulations/NS_benchmarks/NS_Disc/Rho3e3 | tail -5

# +
## Reading the checkpoints back in ... 

step = 135

checkpoint_dir = "/Users/lmoresi/+Simulations/NS_benchmarks/NS_Disc/Rho3e3/"
# checkpoint_dir = "/Users/lmoresi/+Underworld/underworld3/JupyterBook/Notebooks/Examples-NavierStokes//Users/lmoresi/+Simulations/NS_benchmarks/NS_BMK_DvDt_std"

checkpoint_base = "Cylinder_NS_rho_3000.0_30_dt0.1"
base_filename = os.path.join(checkpoint_dir, checkpoint_base)

# +
# mesh = uw.discretisation.Mesh(f"{base_filename}.mesh.{step:05d}.h5")
mesh = uw.discretisation.Mesh(f"{base_filename}.mesh.00000.h5")

v_soln_ckpt = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln_ckpt = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
vorticity_ckpt = uw.discretisation.MeshVariable("omega", mesh, 1, degree=1)

passive_swarm_ckpt = uw.swarm.Swarm(mesh)



# +
v_soln_ckpt.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
p_soln_ckpt.read_timestep(checkpoint_base, "P", step, outputPath=checkpoint_dir)
vorticity_ckpt.read_timestep(checkpoint_base, "omega", step, outputPath=checkpoint_dir)

# This one is just the individual points
passive_swarm_ckpt.read_timestep(checkpoint_base, "passive_swarm", step, outputPath=checkpoint_dir)
# -


passive_swarm_ckpt.dm.getLocalSize()

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln_ckpt.sym[0])
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity_ckpt.sym)                                       
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_ckpt.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_soln_ckpt.sym.dot(v_soln_ckpt.sym)))

    pvmesh.point_data["Omag"] = np.abs(pvmesh.point_data["Omega"])
    pvmesh.point_data["Omag"] /= pvmesh.point_data["Omag"].max()
    pvmesh.point_data["Omag"] = 0.2 + 0.8 * pvmesh.point_data["Omag"]**0.25
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln_ckpt)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_ckpt.sym)
    
    x,y = mesh.X

    # swarm points
    points = vis.swarm_to_pv_cloud(passive_swarm_ckpt)
    swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres
    skip = 25
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=0.5)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(
        velocity_points.points, 
        velocity_points.point_data["V"], mag=0.01, 
        opacity=0.25, show_scalar_bar=False)

    pl.add_mesh(pvmesh,'Grey', 'wireframe', opacity=0.25)
    pl.add_mesh(pvstream, opacity=0.5, show_scalar_bar=False)

    pl.add_mesh(
        pvmesh,
        cmap="RdBu_r",
        edge_color="Black",
        color="White",
        show_edges=False,
        scalars="Omega",
        opacity="Omag",
        clim=[-250,250],
        show_scalar_bar=False)



    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3, opacity=0.2
                )


    pl.camera.SetPosition(0.0, 0.0, 4.7)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
            
    pl.show(jupyter_backend="client")

    # pl.close()
# -




0/0

# +
import glob
steps = []
U_files = glob.glob(f"{checkpoint_dir}/{checkpoint_base}.mesh.U*h5")
for Uf in U_files:
    steps.append(int(Uf.split('.U.')[1].split('.')[0]))
steps.sort()

print(steps)


# +
# Override output range
# steps = range(500,600,5)

# +
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis
    
    pl = pv.Plotter(window_size=(1000, 750))

for step in steps:
    # check the mesh if in a notebook / serial
    
    v_soln_ckpt.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir, verbose=True)
    p_soln_ckpt.read_timestep(checkpoint_base, "P", step, outputPath=checkpoint_dir, verbose=True)
    vorticity_ckpt.read_timestep(checkpoint_base, "omega", step, outputPath=checkpoint_dir, verbose=True)

# This one is just the individual points
    passive_swarm_ckpt = uw.swarm.Swarm(mesh)
    passive_swarm_ckpt.read_timestep(checkpoint_base, "passive_swarm", step, outputPath=checkpoint_dir)

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln_ckpt.sym)
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity_ckpt.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_ckpt.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_soln_ckpt.sym.dot(v_soln_ckpt.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln_ckpt)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_ckpt.sym)
        
    x,y = mesh.X

    # swarm points
    points = vis.swarm_to_pv_cloud(passive_swarm_ckpt)
    swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres
    skip = 10
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=0.5)

    # pl.add_arrows(
    #     velocity_points.points, 
    #     velocity_points.point_data["V"], 
    #     mag=0.01, opacity=0.25, 
    #     show_scalar_bar=False)


    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3, opacity=0.25
                )

    pl.add_mesh(pvmesh,'Grey', 'wireframe', opacity=0.25)
    
    pl.add_mesh(
        pvmesh,
        cmap="bwr",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        clim=[-250,250],
        use_transparency=False,
        opacity=0.75,
        show_scalar_bar=False)
    
    pl.add_mesh(pvstream, opacity=0.5, show_scalar_bar=False)
    
    pl.camera.SetPosition(0.0, 0.0, 4.7)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
         
    pl.screenshot(
            filename=f"{base_filename}.{step}.png",
            window_size=(1600, 1600),
            return_img=False)
    
    pl.clear()
# +
# # ! mkdir /Users/lmoresi/+Simulations/NS_benchmarks/NS_Disc/rho1e4/Cylinder_NS_rho_10000.0_50_dt0.1_images
# # ! cp /Users/lmoresi/+Simulations/NS_benchmarks/NS_Disc/rho1e4/Cylinder_NS_rho_10000.0_50_dt0.1*png /Users/lmoresi/+Simulations/NS_benchmarks/NS_Disc/rho1e4/Cylinder_NS_rho_10000.0_50_dt0.1_images


# -






