# ## Visualise circular stokes model (flow etc)

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()
import os

import petsc4py
import underworld3 as uw
import numpy as np
import sympy

# ls -trl ../Examples-Convection/output/Disc_Ra1e7_H1_deleta_1000.0*T* | tail

# ls -trl /Users/lmoresi/+Simulations/InnerCore/ConvectionDisk | tail

###### checkpoint_dir = "../Examples-Convection/output"
checkpoint_dir = "/Users/lmoresi/+Simulations/InnerCore/ConvectionDisk"
checkpoint_base = f"Disc_Ra1e7_H1_deleta_1000.0"
meshfile = os.path.join(checkpoint_dir, checkpoint_base) + ".mesh.05100.h5"


# +
discmesh = uw.discretisation.Mesh(meshfile, 
                                  coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D)

x = discmesh.N.x
y = discmesh.N.y

r = sympy.sqrt(x**2 + y**2)  # cf radius_fn which is 0->1
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# swarm = uw.swarm.Swarm(mesh=discmesh)
v_soln = uw.discretisation.MeshVariable("U", discmesh, discmesh.dim, degree=2)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", discmesh, 1, degree=2)
flux = uw.discretisation.MeshVariable(r"dTdz", discmesh, 1, degree=2)

# +
# v_soln.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
# t_soln.read_timestep(checkpoint_base, "T", step, outputPath=checkpoint_dir)
# -


steps = range(200,6650,10)

# +
import mpi4py
import pyvista as pv
import underworld3.visualisation as vis

pl = pv.Plotter(window_size=[1000, 1000])

for step in steps:

    try:
        v_soln.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
        t_soln.read_timestep(checkpoint_base, "T", step, outputPath=checkpoint_dir)
    except:
        continue

    pl.clear()

    pvmesh = vis.mesh_to_pv_mesh(discmesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    
    # point sources at cell centres
    skip = 3
    points = np.zeros((discmesh._centroids[::skip].shape[0], 3))
    points[:, 0] = discmesh._centroids[::skip, 0]
    points[:, 1] = discmesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)
    
    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=1,
        surface_streamlines=True,
    )
    
    
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
    pl.add_mesh(pvmesh, "Black", "wireframe",  opacity=0.1)
    
    pl.add_points(point_cloud, color="White", point_size=3.0, opacity=0.25)
    
    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.001, show_scalar_bar=True)
    
        
    # pl.remove_scalar_bar("V")
    
    imagefile = os.path.join(checkpoint_dir, checkpoint_base) + f"{step}.png"
    
    pl.screenshot(filename=imagefile, window_size=(1000, 1000), return_img=False)



# -
pl.show()

uw.systems.Stokes.view()



# +
## Calculate heat flux, evaluate at surface — proxy for boundary layer thickness

# +
flux_solver = uw.systems.Projection(discmesh, flux)

# Conductive flux only !
radial_flux = -discmesh.vector.gradient(t_soln.sym[0]).dot(discmesh.CoordinateSystem.unit_e_0)
radial_flux *= sympy.exp(-100*(r-1)**2)

flux_solver.uw_function = radial_flux
flux_solver.smoothing = 1.0e-3
flux_solver.solve()

# +
import mpi4py
import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(discmesh)
pvmesh.point_data["dTdz"] = vis.scalar_fn_to_pv_points(pvmesh, flux.sym[0])
pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
pvmesh.point_data["V"] -= pvmesh.point_data["V"].mean()


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
    max_time=0.5,
)

pl = pv.Plotter(window_size=[1000, 1000])

pl.add_mesh(
    pvmesh,
    cmap="coolwarm",
    edge_color="Grey",
    show_edges=True,
    scalars="dTdz",
    use_transparency=False,
    opacity=1.0,
)

# pl.add_mesh(pvstream, opacity=0.4, show_scalar_bar=False)
# pl.add_mesh(pvmesh, "Black", "wireframe",  opacity=0.1)

pl.add_points(point_cloud, color="White", point_size=3.0, opacity=0.25)

pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.002, show_scalar_bar=True)

    
# pl.remove_scalar_bar("V")

imagefile = os.path.join(checkpoint_dir, checkpoint_base) + f"{step}.png"

pl.screenshot(filename=imagefile, window_size=(1000, 1000), return_img=False)
# OR

# -


