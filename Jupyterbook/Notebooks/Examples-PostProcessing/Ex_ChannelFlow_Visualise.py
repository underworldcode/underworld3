# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Visualise Channel Flow model 


import nest_asyncio
nest_asyncio.apply()
import os

import petsc4py
import underworld3 as uw
import numpy as np
import sympy






# %%
# ls -trl ../Examples-StokesFlow/output/ChannelFlow3D

# %%
checkpoint_dir = "../Examples-StokesFlow/output/ChannelFlow3D"
checkpoint_base = f"WigglyBottom_20"
meshfile = os.path.join(checkpoint_dir, checkpoint_base) + ".mesh.00000.h5"

step = 0

# %%
uw.utilities.h5_scan("../Examples-StokesFlow/output/ChannelFlow3D/WigglyBottom_20.mesh.P1.00000.h5")

# %%
terrain_mesh = uw.discretisation.Mesh(meshfile)

v_soln_ckpt = uw.discretisation.MeshVariable("U1", terrain_mesh, terrain_mesh.dim, degree=2)
p_soln_ckpt = uw.discretisation.MeshVariable("P1", terrain_mesh, 1, degree=1)

v_soln_ckpt.read_timestep(checkpoint_base, "U1", step, outputPath=checkpoint_dir)
p_soln_ckpt.read_timestep(checkpoint_base, "P1", step, outputPath=checkpoint_dir)


# %%
## Visualise the mesh

# OR
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    v = v_soln_ckpt
    p = p_soln_ckpt

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(terrain_mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

    clipped = pvmesh.clip(origin=(0.0, 0.0, -0.09), normal=(0.0, 0, 1), invert=True)
    clipped.point_data["V"] = vis.vector_fn_to_pv_points(clipped, v.sym)

    clipped2 = pvmesh.clip(origin=(0.0, 0.0, -0.05), normal=(0.0, 0, 1), invert=True)
    clipped2.point_data["V"] = vis.vector_fn_to_pv_points(clipped2, v.sym)
    
    clipped3 = pvmesh.clip(origin=(0.0, 0.0, 0.4), normal=(0.0, 0, 1), invert=False)
    clipped3.point_data["V"] = vis.vector_fn_to_pv_points(clipped3, v.sym)


    skip = 10
    points = np.zeros((terrain_mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = terrain_mesh._centroids[::skip, 0]
    points[:, 1] = terrain_mesh._centroids[::skip, 1]
    points[:, 2] = terrain_mesh._centroids[::skip, 2]

    point_cloud = pv.PolyData(points[np.logical_and(points[:, 0] < 2.0, points[:, 0] > 0.0)]  )

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.1,
        max_time=0.5,
        max_steps=1000
    )

    point_cloud2 = pv.PolyData(points[np.logical_and(points[:, 2] < 0.5, points[:, 2] > 0.45)]  )

    pvstream2 = pvmesh.streamlines_from_source(
        point_cloud2, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=0.5,
        max_steps=1000
    )

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(pvmesh,'Grey', 'wireframe', opacity=0.1)
    pl.add_mesh(clipped,'Blue', show_edges=False, opacity=0.25)
    # pl.add_mesh(pvmesh, 'white', show_edges=True, opacity=0.5)

    #pl.add_mesh(pvstream)
    pl.add_mesh(pvstream2)


    arrows = pl.add_arrows(clipped2.points, clipped2.point_data["V"], 
                           show_scalar_bar = False, opacity=1,
                           mag=100, )
    
    # arrows = pl.add_arrows(clipped3.points, clipped3.point_data["V"], 
    #                        show_scalar_bar = False, opacity=1,
    #                        mag=33, )


    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    
    pl.show(cpos="xy")




# %%
