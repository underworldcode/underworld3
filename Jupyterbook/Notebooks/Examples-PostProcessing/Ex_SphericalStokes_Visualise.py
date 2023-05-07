# ## Visualise spherical stokes model (velocity, particles etc)

# +
import petsc4py
import underworld3 as uw
import numpy as np

import pyvista as pv
import vtk

pv.global_theme.background = "white"
pv.global_theme.window_size = [1250, 1250]
pv.global_theme.anti_aliasing = "ssaa"
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 20.0]
# -

# ls -tr /Users/lmoresi/+Simulations/InnerCore/

# +
basename = f"/Users/lmoresi/+Simulations/InnerCore/outputs_free_slip_fk1e-2_ViscGrad0_iic100_QTemp_mr/free_slip_sphere.h5"
# basename = f"/Users/lmoresi/+Simulations/InnerCore/outputs_free_slip_fk1e-2_ViscGrad0_iic0_QTemp_mr/free_slip_sphere.h5"

step = 210

res = uw.options.getReal("resolution", default=0.1)
r_o = uw.options.getReal("radius_o", default=1.0)
r_i = uw.options.getReal("radius_i", default=0.05)



# +
# # ls -ltr ~/+Simulations/InnerCore/outputs_free_slip_fk1e-2_ViscGrad0_iic100_QTemp_mr | tail

# +
meshball = uw.meshing.SphericalShell(
    radiusInner=r_i,
    radiusOuter=r_o,
    cellSize=res,
    qdegree=2,
)

swarm = uw.swarm.Swarm(mesh=meshball)
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)

# -

print(f"Read swarm data", flush=True)
swarm.load(f"{basename}.passive_swarm.{step}.h5")

v_soln.read_from_vertex_checkpoint(f"{basename}.u.0.h5", "u")
t_soln.read_from_vertex_checkpoint(f"{basename}.DeltaT.0.h5", "DeltaT")

# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "fxaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshball.vtk("tmp_meshball.vtk")
    pvmesh = pv.read("tmp_meshball.vtk")

    pvmesh.point_data["T"] = t_soln.rbf_interpolate(meshball.data)
    pvmesh.point_data["V"] = v_soln.rbf_interpolate(meshball.data)
    
    
    # point sources at cell centres

    skip = 250
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    points[:, 2] = meshball._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        # max_time=2.0,
    )

    with swarm.access():
        points = swarm.data.copy()
        r2 = points[:,0]**2 + points[:,1]**2 + points[:,2]**2 
        point_cloud = pv.PolyData(points[r2<0.98**2])
        # point_cloud.point_data["strain"] = strain.data[:,0]

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[...] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[...] = v_soln.rbf_interpolate(v_soln.coords)

    sphere = pv.Sphere(radius=0.85, center=(0.0, 0.0, 0.0))
    clipped = pvmesh.clip_surface(sphere)
    
    # clipped = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=True)
# -

    pl = pv.Plotter(window_size=[1000, 1000])
    # pl.add_axes()
    
    pl.camera_position = [(2.1,-4.0,0.0), (0.0,0.0,0.0), (0.0,0.0,1.0)]
    # pl.camera.
    # pl.camera.azimuth = -65
    # pl.camera.distance = 10.0
    
 #    pl.camera_position = [(0.00036144256591796875, -0.00045242905616760254, 6.692800318757354),
 # (0.00036144256591796875, -0.00045242905616760254, 0.00010478496551513672),
 # (0.0, 1.0, 0.0)]

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_mesh(pvstream, opacity=0.4)
    pl.add_mesh(pvmesh, "Black", "wireframe",  opacity=0.1)
 
    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)
    
    pl.add_points(point_cloud, color="White", point_size=3.0, opacity=0.25)

    # pl.add_arrows(arrow_loc, arrow_length, mag=20)
    
    pl.remove_scalar_bar("T")
    try:
        pl.remove_scalar_bar("mag")
    except KeyError:
        pass
    try:
        pl.remove_scalar_bar("V-normed")
    except KeyError:
        pass
    try:
        pl.remove_scalar_bar("V")
    except KeyError:
        pass
       
        
    # pl.remove_scalar_bar("V")

    pl.screenshot(filename="sphere_iic0.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show()

# + language="sh"
#
# open sphere_iic0.png
# -


