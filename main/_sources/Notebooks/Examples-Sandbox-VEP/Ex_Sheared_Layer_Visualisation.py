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

# # Validate constitutive models
#
# Simple shear with material defined by particle swarm (based on inclusion model), position, pressure, strain rate etc. Check the implmentation of the Jacobians using various non-linear terms.
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
import underworld3 as uw
import numpy as np

import pyvista as pv
import vtk

pv.global_theme.background = "white"
pv.global_theme.window_size = [1250, 500]
pv.global_theme.anti_aliasing = "ssaa"
pv.global_theme.jupyter_backend = "trame"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 20.0]


# +
step = 1 #50

# basename = "/Users/lmoresi/+Simulations/ShearTest/ShearTestHP_InclusionMu05/shear_band_sw_nonp_0.5"
basename = "output/shear_band_sw_nonp_0.5"


# + language="sh"
# ls -trl "/Users/lmoresi/+Simulations/ShearTest/ShearTestHP_InclusionMu05/" | tail

# +
# # ls /Users/lmoresi/+Simulations/ShearTest/ShearMu05_res0.01

# +
# Simplified (not periodic) mesh that covers the 
# same area as the original (periodic) mesh

mesh1 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.5,-0.5),
    maxCoords=(+1.5,+0.5),
    cellSize=0.02,
)

swarm = uw.swarm.Swarm(mesh=mesh1)


# +
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)

strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
strain_rate_inv2_p = uw.discretisation.MeshVariable("eps_p", mesh1, 1, degree=1, varsymbol=r"\dot\varepsilon_p")
strain_p = uw.discretisation.MeshVariable("p_strain", mesh1, 1, degree=2, varsymbol=r"varepsilon_p")

strain = uw.swarm.SwarmVariable(
    "Strain", swarm, size=1, 
    proxy_degree=1, proxy_continuous=False, 
    varsymbol=r"\varepsilon", dtype=float,
)

# dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=2)
# yield_stress = uw.discretisation.MeshVariable("tau_y", mesh1, 1, degree=1)

swarm.load(f"{basename}.swarm.{step}.h5")
strain.load(f"{basename}.strain.{step}.h5", swarmFilename=f"{basename}.swarm.{step}.h5")

# -


v_soln.read_from_vertex_checkpoint(f"{basename}.U.{step}.h5", "U")
p_soln.read_from_vertex_checkpoint(f"{basename}.P.{step}.h5", "P")
strain_rate_inv2_p.read_from_vertex_checkpoint(f"{basename}.eps_p.{step}.h5", "eps_p")
strain_p.read_from_vertex_checkpoint(f"{basename}.proxy.Strain.{step}.h5", "proxy_Strain")

# +
mesh1.vtk("tmp_shear_inclusion.vtk")
pvmesh = pv.read("tmp_shear_inclusion.vtk")

pvpoints = pvmesh.points[:, 0:2]
usol = v_soln.rbf_interpolate(pvpoints)


pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
pvmesh.point_data["Edotp"] = strain_rate_inv2_p.rbf_interpolate(pvpoints)
pvmesh.point_data["Strn"] = strain_p.rbf_interpolate(pvpoints)
pvmesh.point_data["Strn2"] = strain._meshVar.rbf_interpolate(pvpoints)

# Velocity arrows

v_vectors = np.zeros_like(pvmesh.points)
v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)

# Points (swarm)

with swarm.access():
    plot_points = np.where(strain.data > 0.0)
    strain_data = strain.data.copy()
    
    points = np.zeros((swarm.data[plot_points].shape[0], 3))
    points[:, 0] = swarm.data[plot_points[0],0]
    points[:, 1] = swarm.data[plot_points[0],1]
    point_cloud = pv.PolyData(points)
    point_cloud.point_data["strain"] = strain.data[plot_points]

pl = pv.Plotter(window_size=(500, 500))

# pl.add_arrows(pvmesh.points, v_vectors, mag=0.05, opacity=0.25)
# pl.camera_position = "xy"


pl.add_mesh(
    pvmesh,
    cmap="Blues",
    edge_color="Grey",
    show_edges=False,
    # clim=[0.1,0.5],
    scalars="Edotp",
    use_transparency=False,
    opacity=0.75,
)

pl.add_points(point_cloud, colormap="Oranges", 
              scalars="strain", 
              # clim=[0.0,0.001],
              point_size=5.0,
              opacity=0.5)

pl.camera.SetPosition(0.0, 0.0, 3.0)
pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
pl.camera.SetClippingRange(1.0, 8.0)

# pl.camera.SetPosition(0.75, 0.2, 1.5)
# pl.camera.SetFocalPoint(0.75, 0.2, 0.0)


pl.screenshot(
            filename=f"{basename}.{step}.png",
            window_size=(2560, 1280),
            return_img=False,
        )
    

pl.show()
# -





