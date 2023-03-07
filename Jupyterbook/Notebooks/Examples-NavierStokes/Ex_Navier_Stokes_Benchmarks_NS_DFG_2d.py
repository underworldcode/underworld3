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

import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

# +
# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_resolution=0.1 etc 

resolution = uw.options.getReal("model_resolution", default=0.1)
model = uw.options.getInt("model_number", default=1)

# +
if model == 1:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_1_{resolution}"
    
elif model == 2:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_1_ss_{resolution}"

elif model == 3:
    U0 = 1.5
    expt_name = f"NS_benchmark_DFG2d_2iii_{resolution}"

elif model == 4:
    U0 = 3.75
    expt_name = f"NS_test_Re_250_{resolution}"
    
elif model == 5:
    U0 = 15
    expt_name = f"NS_test_Re_1000_{resolution}"
# -

os.makedirs(".meshes", exist_ok=True)

# +
import pygmsh

# Mesh a 2D pipe with a circular hole

csize = resolution
csize_circle = 0.5 * csize
res = csize_circle

width = 2.2
height = 0.41
radius = 0.05


if uw.mpi.rank == 0:

    # Generate local mesh on boss process

    with pygmsh.geo.Geometry() as geom:

        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (0.2, 0.2, 0.0), radius, make_surface=False, mesh_size=csize_circle
        )
        domain = geom.add_rectangle(
            xmin=0.0,
            ymin=0.0,
            xmax=width,
            ymax=height,
            z=0,
            holes=[inclusion],
            mesh_size=csize,
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
        geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
        geom.add_physical(domain.surface.curve_loop.curves[2], label="top")
        geom.add_physical(domain.surface.curve_loop.curves[3], label="left")
        geom.add_physical(inclusion.curve_loop.curves, label="inclusion")
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry(f".meshes/ns_pipe_flow_{resolution}.msh")

pipemesh = uw.discretisation.Mesh(f".meshes/ns_pipe_flow_{resolution}.msh", 
                                  markVertices=True, 
                                  useMultipleTags=True, 
                                  useRegions=True,
                                  qdegree=3)
pipemesh.dm.view()


# radius_fn = sympy.sqrt(pipemesh.rvec.dot(pipemesh.rvec)) # normalise by outer radius if not 1.0
# unit_rvec = pipemesh.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff

x = pipemesh.N.x
y = pipemesh.N.y

# relative to the centre of the inclusion
r = sympy.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2)
th = sympy.atan2(y - 0.2, x - 0.2)

# need a unit_r_vec equivalent

inclusion_rvec = pipemesh.rvec - 1.0 * pipemesh.N.i - 0.5 * pipemesh.N.j
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# Boundary condition as specified in the diagram

Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41**2

# -

v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
vs_soln = uw.discretisation.MeshVariable("Us", pipemesh, pipemesh.dim, degree=2)
v_soln_1 = uw.discretisation.MeshVariable("U_1", pipemesh, pipemesh.dim, degree=2)
v_stokes = uw.discretisation.MeshVariable("U_0", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1)
vorticity = uw.discretisation.MeshVariable("omega", pipemesh, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", pipemesh, 1, degree=1)


# +
swarm = uw.swarm.Swarm(mesh=pipemesh)
v_star = uw.swarm.SwarmVariable("Vs", swarm, pipemesh.dim, proxy_degree=3, proxy_continuous=True)
remeshed = uw.swarm.SwarmVariable("Vw", swarm, 1, proxy_degree=1, dtype="int")
X_0 = uw.swarm.SwarmVariable("X0", swarm, pipemesh.dim, _proxy=False)

swarm.populate(fill_param=5)

# +
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(
    fill_param=3,
)

with passive_swarm.access(passive_swarm.particle_coordinates):
    passive_swarm.particle_coordinates.data[:, 0] /= 2.0 * width
    passive_swarm.particle_coordinates.data[:, 1] = 0.2


# +
# Create NS object

navier_stokes = uw.systems.NavierStokesSwarm(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    velocityStar_fn=v_star.sym,
    rho=1.0,
    theta=0.5,
    verbose=False,
    projection=False,
    solver_name="navier_stokes",
)

navier_stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(pipemesh.dim)
navier_stokes.constitutive_model.Parameters.viscosity = 1
# -


if model == 2:  # Steady state !
    # remove the d/dt term ... replace the time dependence with the
    # steady state advective transport term
    # to lean towards steady state solutions

    navier_stokes.UF0 = -(
        navier_stokes.rho * (v_soln.sym - v_soln_1.sym) / navier_stokes.delta_t
    )


nodal_vorticity_from_v = uw.systems.Projection(pipemesh, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(pipemesh.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3
nodal_vorticity_from_v.petsc_options.delValue("ksp_monitor")

# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

navier_stokes.rho = 1000.0
navier_stokes.theta = 0.5
navier_stokes.penalty = 0.0
navier_stokes.bodyforce = sympy.Matrix([0,0])

navier_stokes.saddle_preconditioner = 1.0 / navier_stokes.constitutive_model.Parameters.viscosity

hw = 1000.0 / res
with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evaluate(r, pipemesh.data, pipemesh.N)

surface_defn = sympy.exp(-(((r_inc.fn - radius) / radius) ** 2) * hw)

# Velocity boundary conditions

navier_stokes.add_dirichlet_bc((0.0, 0.0), "inclusion", (0, 1))
navier_stokes.add_dirichlet_bc((0.0, 0.0), "top", (0, 1))
navier_stokes.add_dirichlet_bc((0.0, 0.0), "bottom", (0, 1))
navier_stokes.add_dirichlet_bc((Vb, 0.0), "left", (0, 1))

# -


navier_stokes._setup_terms()

# + tags=[]
navier_stokes.solve(timestep=10.0)  # Stokes-like initial flow
nodal_vorticity_from_v.solve()

# +
with pipemesh.access(v_stokes, v_soln, v_soln_1):
    v_stokes.data[...] = v_soln.data[...]
    v_soln_1.data[...] = v_soln.data[...]

with swarm.access(v_star, remeshed, X_0):
    v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)
    X_0.data[...] = swarm.data[...]
    remeshed.data[...] = 0
# -


swarm.advection(v_soln.fn, delta_t=navier_stokes.estimate_dt(), corrector=False)

# + tags=[]
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0]
    pv.global_theme.camera['position'] = [0.0, 0.0, 1.0]

    pvmesh = pv.read(f".meshes/ns_pipe_flow_{resolution}.msh")

    #     points = np.zeros((t_soln.coords.shape[0],3))
    #     points[:,0] = t_soln.coords[:,0]
    #     points[:,1] = t_soln.coords[:,1]

    #     point_cloud = pv.PolyData(points)

    with pipemesh.access():
        # usol = navier_stokes._u_star_projector.u.data.copy()
        usol = v_soln.data.copy()

    with pipemesh.access():
        pvmesh.point_data["Vmag"] = uw.function.evaluate(
            sympy.sqrt(v_soln.fn.dot(v_soln.fn)), pipemesh.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, pipemesh.data)
        pvmesh.point_data["dVy"] = uw.function.evaluate(
            (v_soln.fn - v_stokes.fn).dot(pipemesh.N.j), pipemesh.data
        )

    v_vectors = np.zeros((pipemesh.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, pipemesh.data)
    pvmesh.point_data["V"] = v_vectors

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.025 / U0, opacity=0.75)

    # pl.add_points(point_cloud, cmap="coolwarm",
    #               render_points_as_spheres=False,
    #               point_size=10, opacity=0.66
    #             )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    # pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("mag")

    pl.show()
# +
if uw.mpi.size == 1:

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    # pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0]
    # pv.global_theme.camera['position'] = [0.0, 0.0, 2.0]

    pl = pv.Plotter()


def plot_V_mesh(filename):

    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import vtk

        ## Plotting into existing pl (memory leak in pyvista)
        pl.clear()
        
        pvmesh = pv.read(f".meshes/ns_pipe_flow_{resolution}.msh")

        with passive_swarm.access():
            points = np.zeros((passive_swarm.data.shape[0], 3))
            points[:, 0] = passive_swarm.data[:, 0]
            points[:, 1] = passive_swarm.data[:, 1]

        point_cloud = pv.PolyData(points)

        points = np.zeros((pipemesh._centroids.shape[0], 3))
        points[:, 0] = pipemesh._centroids[:, 0]
        points[:, 1] = pipemesh._centroids[:, 1]

        c_point_cloud = pv.PolyData(points)

        with pipemesh.access():
            pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, pipemesh.data)
            pvmesh.point_data["dVy"] = uw.function.evaluate(
                (v_soln.fn - v_stokes.fn).dot(pipemesh.N.j), pipemesh.data
            )
            pvmesh.point_data["Omega"] = uw.function.evaluate(
                vorticity.fn, pipemesh.data
            )

        with pipemesh.access():
            usol = v_soln.data.copy()

        v_vectors = np.zeros((pipemesh.data.shape[0], 3))
        v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, pipemesh.data)
        pvmesh.point_data["V"] = v_vectors

        arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
        arrow_loc[:, 0:2] = v_soln.coords[...]

        arrow_length = np.zeros((v_soln.coords.shape[0], 3))
        arrow_length[:, 0:2] = usol[...]

        pl.add_arrows(arrow_loc, arrow_length, mag=0.033 / U0, opacity=0.5)

        pvstream = pvmesh.streamlines_from_source(
            c_point_cloud, vectors="V", integration_direction="both", max_time=0.25
        )

        # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="Omega",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_mesh(pvstream)

        pl.add_points(
            point_cloud,
            color="Black",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.5,
        )

        pl.camera.SetPosition(0.75, 0.2, 1.5)
        pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
        pl.camera.SetClippingRange(1.0, 8.0)

        pl.remove_scalar_bar("Omega")
        pl.remove_scalar_bar("mag")
        pl.remove_scalar_bar("V")
        
        # pl.camera_position = "xz"
        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 1280),
            return_img=False,
        )


# -

ts = 0
dt_ns = 1.0e-3
swarm_loop = 10

for step in range(0, 250):
    delta_t_swarm = 5.0 * navier_stokes.estimate_dt()
    delta_t = min(delta_t_swarm, dt_ns)

    phi = delta_t / dt_ns

    navier_stokes.solve(timestep=dt_ns, 
                        zero_init_guess=False)
    
    with pipemesh.access(v_soln_1):
        v_soln_1.data[...] = 0.5 * v_soln_1.data[...] + 0.5 * v_soln.data[...]

    with swarm.access(v_star):
        v_star.data[...] = (
            phi * uw.function.evaluate(v_soln.fn, swarm.data)
            + (1.0 - phi) * v_star.data
        )
        

    # update passive swarm

    passive_swarm.advection(v_soln.fn, delta_t, corrector=False)

    npoints = 10
    passive_swarm.dm.addNPoints(npoints)
    with passive_swarm.access(passive_swarm.particle_coordinates):
        for i in range(npoints):
            passive_swarm.particle_coordinates.data[
                -1 : -(npoints + 1) : -1, :
            ] = np.array([0.0, 0.195] + 0.01 * np.random.random((npoints, 2)))

    # update integration swarm
    swarm.advection(v_soln.fn, delta_t, corrector=False)

    # Restore a subset of points to start
    offset_idx = step % swarm_loop

    with swarm.access(swarm.particle_coordinates, remeshed):
        remeshed.data[...] = 0
        remeshed.data[offset_idx::swarm_loop, :] = 1
        swarm.data[offset_idx::swarm_loop, :] = X_0.data[offset_idx::swarm_loop, :]

    # re-calculate v history for remeshed particles
    # Note, they may have moved procs after the access manager closed
    # so we re-index

    with swarm.access(v_star, remeshed):
        idx = np.where(remeshed.data == 1)[0]
        v_star.data[idx] = uw.function.evaluate(v_soln.fn, swarm.data[idx])

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}, phi {}".format(ts, delta_t, phi))

    if ts % 1 == 0:
        nodal_vorticity_from_v.solve()
        plot_V_mesh(filename="output/{}_step_{}".format(expt_name, ts))
        
        
        savefile = f"output/{expt_name}"
        pipemesh.write_checkpoint(savefile, 
                                  meshUpdates=False, 
                                  meshVars=[p_soln,v_soln], 
                                  index=ts)

        
    ts += 1


# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    
    pl.close()

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    # pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0]
    # pv.global_theme.camera['position'] = [0.0, 0.0, 1.0]

    pvmesh = pv.read(f".meshes/ns_pipe_flow_{resolution}.msh")

    #     points = np.zeros((t_soln.coords.shape[0],3))
    #     points[:,0] = t_soln.coords[:,0]
    #     points[:,1] = t_soln.coords[:,1]

    #     point_cloud = pv.PolyData(points)

    with pipemesh.access():
        usol = v_soln.data.copy()

    with pipemesh.access():
        pvmesh.point_data["Vmag"] = uw.function.evaluate(
            sympy.sqrt(v_soln.fn.dot(v_soln.fn)), pipemesh.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, pipemesh.data)
        pvmesh.point_data["dVy"] = uw.function.evaluate(
            (v_soln.fn - v_stokes.fn).dot(pipemesh.N.j), pipemesh.data
        )
        pvmesh.point_data["Omega"] = uw.function.evaluate(vorticity.fn, pipemesh.data)

    v_vectors = np.zeros((pipemesh.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, pipemesh.data)
    pvmesh.point_data["V"] = v_vectors

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # swarm points

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]

        swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres

    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        surface_streamlines=True,
        max_time=0.5,
    )

    pl = pv.Plotter()

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.033/U0, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    # pl.add_points(swarm_point_cloud, color="Black",
    #               render_points_as_spheres=True,
    #               point_size=0.5, opacity=0.66
    #             )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    # pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("S")
    # pl.remove_scalar_bar("mag")

    pl.show()
# +
## Pressure difference at front/rear of the cylinder should be 117

p = uw.function.evaluate(p_soln.fn, np.array([(0.15, 0.2), (0.25, 0.2)]))
p[0] - p[1]
