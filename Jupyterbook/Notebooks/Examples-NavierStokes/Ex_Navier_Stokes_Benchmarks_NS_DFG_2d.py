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

# +
import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

import psutil
pid = os.getpid()
python_process = psutil.Process(pid)
print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# +
# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_resolution=0.1 etc 

resolution = uw.options.getReal("model_resolution", default=0.066)
model = uw.options.getInt("model_number", default=3)
maxsteps = uw.options.getInt("max_steps", default=500)
restart_step = uw.options.getInt("restart_step", default=-1)

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

# +
outdir = "output_res_025"

os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

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


print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# +
v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
vs_soln = uw.discretisation.MeshVariable("Us", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1)
vorticity = uw.discretisation.MeshVariable("omega", pipemesh, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", pipemesh, 1, degree=1)

# Nodal values of deviatoric stress (symmetric tensor)
work   = uw.discretisation.MeshVariable("W", pipemesh, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=False)
St = uw.discretisation.MeshVariable(r"Stress", pipemesh, (2,2), vtype=uw.VarType.SYM_TENSOR, degree=1, 
                                         continuous=False, varsymbol=r"{\tau}")
# -


uw.maths.tensor.rank2_to_voigt(St.sym, dim=2)

# +
swarm = uw.swarm.Swarm(mesh=pipemesh, recycle_rate=20)
v_star = uw.swarm.SwarmVariable("Vs", swarm, pipemesh.dim, 
                            proxy_degree=2, proxy_continuous=True, varsymbol=r"{v^{*}}")

v_star_star = uw.swarm.SwarmVariable("Vss", swarm, pipemesh.dim, 
                            proxy_degree=2, proxy_continuous=True, varsymbol=r"{v^{**}}")

v_star_2 = uw.swarm.SwarmVariable("Vs2", swarm, pipemesh.dim, 
                            proxy_degree=2, proxy_continuous=True, varsymbol=r"{v^{*2}}")


stress_star_p = uw.swarm.SwarmVariable(r"stress^{*}", swarm,
                                     (2,2), vtype=uw.VarType.SYM_TENSOR, 
                                     proxy_continuous=True,
                                     proxy_degree=2,
                                     varsymbol=r"{\sigma^{*}_{p}}",)


swarm.populate(fill_param=3)
# -

stress_star_p.sym

# + tags=[]
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(
    fill_param=3,
)

with passive_swarm.access(passive_swarm.particle_coordinates):
    passive_swarm.particle_coordinates.data[:, 0] /= 2.0 * width
    passive_swarm.particle_coordinates.data[:, 1] = 0.2
# -


print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# +
# Create NS object

navier_stokes = uw.systems.NavierStokesSwarm(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    velocityStar=v_star.sym,
    velocityStarStar=v_star_star.sym,
    # stressStar=stress_star_p.sym,
    # stressStarStar=stress_star_star_p.sym,
    rho=1.0,
    verbose=False,
    projection=True,
    solver_name="navier_stokes",
)

navier_stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(pipemesh.dim)
navier_stokes.constitutive_model.Parameters.viscosity = 1
# -


navier_stokes.constitutive_model

# +
# stress computation and projection to particles

stress_projection = uw.systems.Tensor_Projection(pipemesh, tensor_Field=St, scalar_Field=work  )
stress_projection.uw_function = navier_stokes.stress_deviator_1d
stress_projection.solve()

with swarm.access(stress_star_p), pipemesh.access():
    stress_star_p.data[...] = St.rbf_interpolate(swarm.particle_coordinates.data)

# -

navier_stokes._setup_terms()

navier_stokes._u_f1

print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

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
navier_stokes.penalty = 0.1
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
navier_stokes.add_dirichlet_bc((Vb, 0.0),  "left", (0, 1))

# -


navier_stokes._setup_terms()
navier_stokes.tolerance = 1.0e-4

print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# + tags=[]
navier_stokes.solve(timestep=10.0)  # Stokes-like initial flow
nodal_vorticity_from_v.solve()
# -

with swarm.access(v_star, v_star_star, v_star_2):
    v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)    
    v_star_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)
    v_star_2.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)

print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# + tags=[]
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0]
    pv.global_theme.camera['position'] = [0.0, 0.0, 1.0]

    pvmesh = pv.read(f".meshes/ns_pipe_flow_{resolution}.msh")


    with pipemesh.access():
        # usol = navier_stokes._u_star_projector.u.data.copy()
        usol = v_soln.data.copy()

    with pipemesh.access():
        pvmesh.point_data["Vmag"] = uw.function.evaluate(
            sympy.sqrt(v_soln.fn.dot(v_soln.fn)), pipemesh.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, pipemesh.data)


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
    
    with swarm.access():
        spoints = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        spoints[:, 0] = swarm.particle_coordinates.data[:, 0]
        spoints[:, 1] = swarm.particle_coordinates.data[:, 1]
        spoint_cloud = pv.PolyData(spoints)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.025 / U0, opacity=0.75)

    pl.add_points(spoint_cloud,color="Black",
                  render_points_as_spheres=False,
                  point_size=5, opacity=0.66
                )

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
    pv.global_theme.anti_aliasing = "msaa"
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
        
        
        with swarm.access():
            spoints = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
            spoints[:, 0] = swarm.particle_coordinates.data[:, 0]
            spoints[:, 1] = swarm.particle_coordinates.data[:, 1]
            spoint_cloud = pv.PolyData(spoints)

        with pipemesh.access():
            pvmesh.point_data["P"] = p_soln.rbf_interpolate(pipemesh.data)
            pvmesh.point_data["Omega"] = vorticity.rbf_interpolate(pipemesh.data)

        with pipemesh.access():
            usol = v_soln.data.copy()

        v_vectors = np.zeros((pipemesh.data.shape[0], 3))
        v_vectors[:, 0:2] = v_soln.rbf_interpolate(pipemesh.data)
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
        
        # pl.add_points(
        #     spoint_cloud,
        #     color="Grey",
        #     render_points_as_spheres=True,
        #     point_size=3,
        #     opacity=0.5,
        # )


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
dt_ns = 1.0e-2


navier_stokes._setup_terms()
navier_stokes._u_f0

print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

for step in range(0, maxsteps):
    delta_t_swarm = 2.0 * navier_stokes.estimate_dt()
    dt_ns = 1.0 * delta_t_swarm
    delta_t = min(delta_t_swarm, dt_ns)
    phi = min(1.0, delta_t / dt_ns)

    
    print(f"Memory usage [1] = {python_process.memory_info().rss//1000000} Mb", flush=True)

    navier_stokes.solve(timestep=delta_t, 
                        zero_init_guess=False)
    
    print(f"Memory usage [2] = {python_process.memory_info().rss//1000000} Mb", flush=True)
    
    stress_projection.solve()
    
    print(f"Memory usage [2.1] = {python_process.memory_info().rss//1000000} Mb", flush=True)

    
    with swarm.access(stress_star_p), pipemesh.access():
        
        print(f"Memory usage [2.1.1] = {python_process.memory_info().rss//1000000} Mb", flush=True)

        stress_star_p.data[:,0] = (
            uw.function.evaluate(St.sym_1d[0], swarm.data) )
        
        print(f"Memory usage [2.1.2] = {python_process.memory_info().rss//1000000} Mb", flush=True)

        
        stress_star_p.data[:,1] = (
            uw.function.evaluate(St.sym_1d[1], swarm.data) )
        stress_star_p.data[:,2] = (
            uw.function.evaluate(St.sym_1d[2], swarm.data) )
        
    print(f"Memory usage [2.5] = {python_process.memory_info().rss//1000000} Mb", flush=True)

        
        
    with swarm.access(v_star, v_star_star, v_star_2):
        v_star_star.data[...] = v_star.data[...]
        v_star.data[:,0] = (
            phi * uw.function.evaluate(v_soln[0].sym, swarm.data) +
            (1-phi) * v_star.data[:,0]
        )
        v_star.data[:,1] = (
            phi * uw.function.evaluate(v_soln[1].sym, swarm.data) +
            (1-phi) * v_star.data[:,1]
        )
        # v_star_2.data[:,:] = 4 * v_star.data[:,:] - v_star_star.data[:,:]
      
    
    print(f"Memory usage [3] = {python_process.memory_info().rss//1000000} Mb", flush=True)

    # update passive swarm
    passive_swarm.advection(v_soln.fn, delta_t, corrector=False)
    
    print(f"Memory usage [4] = {python_process.memory_info().rss//1000000} Mb", flush=True)


    npoints = 20
    passive_swarm.dm.addNPoints(npoints)
    with passive_swarm.access(passive_swarm.particle_coordinates):
        for i in range(npoints):
            passive_swarm.particle_coordinates.data[
                -1 : -(npoints + 1) : -1, :
            ] = np.array([0.0, 0.195] + 0.01 * np.random.random((npoints, 2)))
            
    print(f"Memory usage [5] = {python_process.memory_info().rss//1000000} Mb", flush=True)

    # update integration swarm
    swarm.advection(v_soln.fn, delta_t, corrector=False)
    
    print(f"Memory usage [6] = {python_process.memory_info().rss//1000000} Mb", flush=True)


    ps_num = passive_swarm.dm.getSize()
    
    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}, dt_s {} phi {}".format(ts, delta_t, delta_t_swarm, phi))

    if ts % 5 == 0:
        nodal_vorticity_from_v.solve()
        plot_V_mesh(filename="{}/{}_step_{}".format(outdir,expt_name, ts))
        
        
        savefile = f"{outdir}/{expt_name}"
        
        pipemesh.write_timestep_xdmf(savefile, 
                                     meshUpdates=True,
                                     meshVars=[p_soln,v_soln,vorticity], 
                                     index=ts)
        
        passive_swarm.save(filename=f"{savefile}.passive_swarm.{ts}.h5")
        
    print(f"Memory usage [7] = {python_process.memory_info().rss//1000000} Mb", flush=True)
   
        
    if uw.mpi.rank == 0:
        print(f"------", flush=True)

        
    ts += 1


navier_stokes.u_star_fn

with swarm.access():
    print(v_star_star._meshVar.rbf_interpolate(swarm.particle_coordinates.data))
    print(v_star._meshVar.rbf_interpolate(swarm.particle_coordinates.data))

with swarm.access():
    print(uw.function.evaluate(v_star._meshVar.sym[0], swarm.particle_coordinates.data))





python_process.memory_info()


memoryUse


