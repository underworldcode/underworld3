# ## Visualise circular stokes model (flow etc)

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()
import os

import petsc4py
import underworld3 as uw
import numpy as np

# ls -trl ../Examples-Convection/output/*Disc*h5 | tail

# +
checkpoint_dir = "../Examples-Convection/output"
checkpoint_base = f"Disc_Ra1e7_H1_deleta_1000.0"
meshfile = os.path.join(checkpoint_dir, checkpoint_base) + ".mesh.00000.h5"

step = 50


# +
discmesh = uw.discretisation.Mesh(meshfile)

# swarm = uw.swarm.Swarm(mesh=discmesh)
v_soln = uw.discretisation.MeshVariable("U", discmesh, discmesh.dim, degree=2)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", discmesh, 1, degree=2)

# -

v_soln.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
t_soln.read_timestep(checkpoint_base, "T", step, outputPath=checkpoint_dir)


# +
import mpi4py
import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(discmesh)
pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

# point sources at cell centres
skip = 1
points = np.zeros((discmesh._centroids[::skip].shape[0], 3))
points[:, 0] = discmesh._centroids[::skip, 0]
points[:, 1] = discmesh._centroids[::skip, 1]
point_cloud = pv.PolyData(points)

pvstream = pvmesh.streamlines_from_source(
    point_cloud,
    vectors="V",
    integration_direction="both",
    max_time=0.2,
)

pl = pv.Plotter(window_size=[1000, 1000])

pl.add_mesh(
    pvmesh,
    cmap="coolwarm",
    edge_color="Grey",
    show_edges=False,
    scalars="T",
    use_transparency=False,
    opacity=1.0,
)

pl.add_mesh(pvstream, opacity=0.4, show_scalar_bar=False)
# pl.add_mesh(pvmesh, "Black", "wireframe",  opacity=0.1)

pl.add_points(point_cloud, color="White", point_size=3.0, opacity=0.25)

pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01, show_scalar_bar=True)

    
# pl.remove_scalar_bar("V")

imagefile = os.path.join(checkpoint_dir, checkpoint_base) + f"{step}.png"

pl.screenshot(filename=imagefile, window_size=(1000, 1000), return_img=False)
# OR
pl.show()
# -






