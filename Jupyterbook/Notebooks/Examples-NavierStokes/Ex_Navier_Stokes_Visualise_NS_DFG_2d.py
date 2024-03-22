# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# # Navier Stokes test: flow around a circular inclusion (2D)
#
# http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
#
# No slip conditions
#
# ![](http://www.mathematik.tu-dortmund.de/~featflow/media/dfg_bench1_2d/geometry.png)
#
# Note ...
#
# In this benchmark, I have scaled $\rho = 1000$ and $\nu = 1.0$ as otherwise it fails to converge. This occurs because we are locked into a range of $\Delta t$ by the flow velocity (and accurate particle transport), and by the assumption that $\dot{\epsilon}$ is computed in the Eulerian form. The Crank-Nicholson scheme still has some timestep requirements associated with diffusivity (viscosity in this case) and this may be what I am seeing.
#
# Velocity is the same, but pressure scales by 1000. This should encourage us to implement scaling / units.
#
# Model 4 is not one of the benchmarks, but just turns up the Re parameter to see if the mesh can resolve higher values than 100
#
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

# ls -trl ~/+Simulations/NS_benchmarks/Re100_dt0.01_proj0_tau2_pr2_rbf | tail

# +
## Reading the checkpoints back in ... 

step = 230
basename = f"/Users/lmoresi/+Simulations/NS_benchmarks/Re250_res001/NS_test_Re_250_0.01"
mesh_filename = f"{basename}.mesh.50.h5"
mesh_filename

# +
mesh = uw.discretisation.Mesh(mesh_filename)

v_soln_ckpt = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln_ckpt = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
vorticity_ckpt = uw.discretisation.MeshVariable("omega", mesh, 1, degree=1)

passive_swarm_ckpt = uw.swarm.Swarm(mesh, recycle_rate=0)


# +
v_soln_ckpt.read_from_vertex_checkpoint(f"{basename}.U.{step}.h5", "U")
p_soln_ckpt.read_from_vertex_checkpoint(f"{basename}.P.{step}.h5", "P")
vorticity_ckpt.read_from_vertex_checkpoint(f"{basename}.omega.{step}.h5", "omega")

# passive_swarm_ckpt.load(f"output/NS_test_Re_250_{res}.passive_swarm.{step}.h5")

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_ckpt.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln_ckpt.sym)
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity_ckpt.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln_ckpt)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_ckpt.sym)

    # swarm points

#     with passive_swarm_ckpt.access():
#         points = np.zeros((passive_swarm_ckpt.data.shape[0], 3))
#         points[:, 0] = passive_swarm_ckpt.data[:, 0]
#         points[:, 1] = passive_swarm_ckpt.data[:, 1]

#         swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres

    skip = 10
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="forward",
        max_time=0.5,
    )

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.001, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        use_transparency=False,
        opacity=1.0,
    )

    # pl.add_points(swarm_point_cloud, color="Black",
    #               render_points_as_spheres=True,
    #               point_size=5, opacity=0.66
    #             )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    pl.add_mesh(pvstream, opacity=0.33)

    # pl.remove_scalar_bar("S")
    # pl.remove_scalar_bar("mag")

    pl.show()
# -

