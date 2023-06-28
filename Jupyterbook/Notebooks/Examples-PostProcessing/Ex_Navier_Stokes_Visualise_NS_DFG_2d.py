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


# + [markdown] tags=[]
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
# -

import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

# + language="sh"
#
# # ls -tr /Users/lmoresi/+Simulations/NS_benchmarks/NS_BMK_r025_dt005_phi05_no_evalf | tail -10
# ls -tr /Users/lmoresi/+Underworld/underworld3/JupyterBook/Notebooks/Examples-NavierStokes/output_res_033/*mesh*h5 | tail -10
#
#


# +
## Reading the checkpoints back in ... 

step = 150

# checkpoint_dir = "/Users/lmoresi/+Simulations/NS_benchmarks/NS_BMK_r025_dt005_phi05_no_evalf"
checkpoint_dir = "/Users/lmoresi/+Underworld/underworld3/JupyterBook/Notebooks/Examples-NavierStokes/output_res_033"

checkpoint_base = "NS_benchmark_DFG2d_2iii_0.05"
base_filename = os.path.join(checkpoint_dir, checkpoint_base)

# +
mesh = uw.discretisation.Mesh(f"{base_filename}.mesh.00000.h5")

v_soln_ckpt = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln_ckpt = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
vorticity_ckpt = uw.discretisation.MeshVariable("omega", mesh, 1, degree=1)

passive_swarm_ckpt = uw.swarm.Swarm(mesh)
active_swarm_ckpt = uw.swarm.Swarm(mesh)


# +
v_soln_ckpt.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
p_soln_ckpt.read_timestep(checkpoint_base, "P", step, outputPath=checkpoint_dir)
vorticity_ckpt.read_timestep(checkpoint_base, "omega", step, outputPath=checkpoint_dir)

# This one is just the individual points
passive_swarm_ckpt.read_timestep(checkpoint_base, "passive_swarm", step, outputPath=checkpoint_dir)


# + tags=[]
# check the mesh if in a notebook / serial

import pyvista as pv

pv.global_theme.background = "white"
pv.global_theme.window_size = [1250, 1250]
pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0]
pv.global_theme.camera['position'] = [0.0, 0.0, 1.0]

if uw.mpi.size == 1:
    
    import numpy as np
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v_soln_ckpt.data.copy()

    with mesh.access():            
        pvmesh.point_data["P"] = uw.function.evalf(p_soln_ckpt.sym[0], mesh.data)
        pvmesh.point_data["Omega"] = uw.function.evalf(vorticity_ckpt.sym[0], mesh.data)
        pvmesh.point_data["Vmag"] = uw.function.evalf(
            sympy.sqrt(v_soln_ckpt.sym.dot(v_soln_ckpt.sym)), mesh.data
        )
        
    x,y = mesh.X
    U0 = 1.5
    Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41**2
        
    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0] = uw.function.evalf(v_soln_ckpt[0].sym, mesh.data)
    v_vectors[:, 1] = uw.function.evalf(v_soln_ckpt[1].sym, mesh.data)
    pvmesh.point_data["V"] = v_vectors

    arrow_loc = np.zeros((v_soln_ckpt.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln_ckpt.coords[...]

    arrow_length = np.zeros((v_soln_ckpt.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # swarm points

    with passive_swarm_ckpt.access():
        points = np.zeros((passive_swarm_ckpt.data.shape[0], 3))
        points[:, 0] = passive_swarm_ckpt.data[:, 0]
        points[:, 1] = passive_swarm_ckpt.data[:, 1]

        swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres

    skip = 5
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=0.5,
    )

    pl = pv.Plotter()
    pl.add_arrows(arrow_loc, arrow_length, mag=0.02, opacity=0.5)

    pl.add_mesh(
        pvmesh,
        cmap="bwr",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3, opacity=0.2
                )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.25)
    pl.add_mesh(pvstream, opacity=0.5)


    pl.remove_scalar_bar("Omega")
    pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")
        

    pl.camera.SetPosition(1.15, 0.2, 1.7)
    pl.camera.SetFocalPoint(1.15, 0.2, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
        
        
    pl.screenshot(
            filename=f"{base_filename}.{step}.png",
            window_size=(2560, 1280),
            return_img=False,
        )
    
    pl.show()


# +
import glob
steps = []
U_files = glob.glob(f"{checkpoint_dir}/{checkpoint_base}.mesh.U*h5")
for Uf in U_files:
    steps.append(int(Uf.split('.U.')[1].split('.')[0]))
steps.sort()

print(steps)
# -


0/0

# +
if uw.mpi.size == 1:
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0]
    pv.global_theme.camera['position'] = [0.0, 0.0, 1.0]
    
    pl = pv.Plotter()

for step in steps:
    # check the mesh if in a notebook / serial
    
    v_soln_ckpt.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
    p_soln_ckpt.read_timestep(checkpoint_base, "P", step, outputPath=checkpoint_dir)
    vorticity_ckpt.read_timestep(checkpoint_base, "omega", step, outputPath=checkpoint_dir)

# This one is just the individual points
    passive_swarm_ckpt = uw.swarm.Swarm(mesh)
    passive_swarm_ckpt.read_timestep(checkpoint_base, "passive_swarm", step, outputPath=checkpoint_dir)

    import numpy as np
    import vtk

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v_soln_ckpt.data.copy()

    with mesh.access():            
        pvmesh.point_data["P"] = p_soln_ckpt.rbf_interpolate(mesh.data)
        pvmesh.point_data["Omega"] = vorticity_ckpt.rbf_interpolate(mesh.data)
        
    x,y = mesh.X
    U0 = 1.5
    Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41**2

    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0] = uw.function.evalf(v_soln_ckpt[0].sym, mesh.data)
    v_vectors[:, 1] = uw.function.evalf(v_soln_ckpt[1].sym, mesh.data)
    pvmesh.point_data["V"] = v_vectors

    arrow_loc = np.zeros((v_soln_ckpt.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln_ckpt.coords[...]

    arrow_length = np.zeros((v_soln_ckpt.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # swarm points

    with passive_swarm_ckpt.access():
        points = np.zeros((passive_swarm_ckpt.data.shape[0], 3))
        points[:, 0] = passive_swarm_ckpt.data[:, 0]
        points[:, 1] = passive_swarm_ckpt.data[:, 1]

        swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres

    skip = 15
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=0.5,
    )

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.02, opacity=0.5)



    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3, opacity=0.5
                )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.05)
    
    pl.add_mesh(
        pvmesh,
        cmap="bwr",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        use_transparency=False,
        opacity=0.75,
    )
    
    pl.add_mesh(pvstream, opacity=0.5)


    pl.remove_scalar_bar("Omega")
    # pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")
    
    pl.camera.SetPosition(0.95, 0.2, 1.5)
    pl.camera.SetFocalPoint(0.95, 0.2, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
        
    pl.screenshot(
            filename=f"{base_filename}.{step}.png",
            window_size=(2500, 1000),
            return_img=False,
        )
    
    pl.clear()
# -


0/0
