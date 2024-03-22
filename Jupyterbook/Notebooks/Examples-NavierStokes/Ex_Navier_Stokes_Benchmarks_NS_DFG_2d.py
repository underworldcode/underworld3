# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
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

resolution = uw.options.getReal("model_resolution", default=16)
refinement = uw.options.getInt("model_refinement", default=0)
model = uw.options.getInt("model_number", default=3)
maxsteps = uw.options.getInt("max_steps", default=1000)
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
# -

outdir = f"output_swarm_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# +
import pygmsh

# Mesh a 2D pipe with a circular hole

csize = 1.0 / resolution
csize_circle = 0.5 * csize
res = csize_circle

width = 2.2
height = 0.41
radius = 0.05

from enum import Enum

## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh

class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left  = 4
    inclusion = 5
    All_Boundaries = 1001 

def pipemesh_mesh_refinement_callback(dm):
    r_p = radius

    # print(f"Refinement callback - spherical", flush=True)

    c2 = dm.getCoordinatesLocal()
    coords = c2.array.reshape(-1, 2) - centre

    R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2).reshape(-1, 1)

    pipeIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        dm, "inclusion"
    )

    coords[pipeIndices] *= r_p / R[pipeIndices]
    coords = coords + centre

    c2.array[...] = coords.reshape(-1)
    dm.setCoordinatesLocal(c2)

    return

## Restore inflow samples to inflow points
def pipemesh_return_coords_to_bounds(coords):
    lefty_troublemakers = coords[:, 0] < 0.0
    far_right = coords[:, 0] > 2.2
    too_low = coords[:, 1] < 0.0
    too_high = coords[:, 1] > 0.41

    coords[lefty_troublemakers, 0] = 0.0001
    coords[far_right, 0] = 2.2 - 0.0001

    coords[too_low, 1] = 0.0001
    coords[too_high, 1] = 0.41 - 0.0001

    return coords

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

pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=pipemesh_mesh_refinement_callback,
    return_coords_to_bounds= pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3,
)

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
rho = uw.discretisation.MeshVariable("rho", pipemesh, 1, degree=1, varsymbol=r"{\rho}")

# Nodal values of deviatoric stress (symmetric tensor)
work = uw.discretisation.MeshVariable(
    "W", pipemesh, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=False
)
St = uw.discretisation.MeshVariable(
    r"Stress",
    pipemesh,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=1,
    continuous=False,
    varsymbol=r"{\tau}",
)


# +
swarm = uw.swarm.Swarm(mesh=pipemesh, recycle_rate=6)

DvDt = uw.systems.ddt.Lagrangian_Swarm(
    swarm, 
    v_soln.sym, 
    uw.VarType.VECTOR,
    degree=2,
    order=2,
    verbose=False,
    continuous=True,
)


swarm.populate(fill_param=4)

# -




# +
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(
    fill_param=1,
)

# add new points at the inflow
npoints = 50
passive_swarm.dm.addNPoints(npoints)
with passive_swarm.access(passive_swarm.particle_coordinates):
    for i in range(npoints):
        passive_swarm.particle_coordinates.data[-1 : -(npoints + 1) : -1, :] = np.array(
            [0.0, 0.195] + 0.01 * np.random.random((npoints, 2))
        )

# -
nodal_vorticity_from_v = uw.systems.Projection(pipemesh, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(pipemesh.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3
nodal_vorticity_from_v.petsc_options.delValue("ksp_monitor")

# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

navier_stokes = uw.systems.NavierStokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    DuDt=DvDt,
    rho=1000.0,
    verbose=False,
    solver_name="navier_stokes",
    order=2,
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Constant visc

navier_stokes.rho = 1000.0
navier_stokes.penalty = 0.1
navier_stokes.bodyforce = sympy.Matrix([0, 0])

hw = 1000.0 / res
with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evalf(r, pipemesh.data, pipemesh.N)

surface_defn = sympy.exp(-(((r_inc.fn - radius) / radius) ** 2) * hw)

# Velocity boundary conditions

navier_stokes.add_dirichlet_bc(
    (0.0, 0.0),
    "inclusion",
)
navier_stokes.add_dirichlet_bc((0.0, 0.0), "top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
navier_stokes.add_dirichlet_bc((Vb, 0.0), "left")

navier_stokes.tolerance = 1.0e-3
navier_stokes.delta_t = 10.0  # stokes-like at the beginning


if model == 2:  # Steady state !
    # remove the d/dt term ... replace the time dependence with the
    # steady state advective transport term
    # to lean towards steady state solutions

    navier_stokes.UF0 = -(
        navier_stokes.rho * (v_soln.sym - v_soln_1.sym) / navier_stokes.delta_t
    )

# -
navier_stokes.Unknowns.DuDt.update_pre_solve(0.0)

# +
   
navier_stokes.solve(timestep=10.0, verbose=False)  # Stokes-like initial flow
nodal_vorticity_from_v.solve()


# +
# stress_projection = uw.systems.Tensor_Projection(
#     pipemesh, tensor_Field=St, scalar_Field=work
# )
# stress_projection.uw_function = navier_stokes.stress_deviator
# stress_projection.solve()

# with swarm.access(stress_star_p), pipemesh.access():
#     stress_star_p.data[:, 0] = uw.function.evaluate(
#         St.sym_1d[0], swarm.particle_coordinates.data
#     )
#     stress_star_p.data[:, 1] = uw.function.evaluate(
#         St.sym_1d[1], swarm.particle_coordinates.data
#     )
#     stress_star_p.data[:, 2] = uw.function.evaluate(
#         St.sym_1d[2], swarm.particle_coordinates.data
#     )


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres

    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)
    active_swarm_points = uw.visualisation.swarm_to_pv_cloud(swarm)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025 / U0, opacity=0.75)

    pl.add_points(
        active_swarm_points,
        color="Black",
        render_points_as_spheres=False,
        point_size=3,
        opacity=0.66,
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
    pl = pv.Plotter(window_size=(1000, 750))


def plot_V_mesh(filename):

    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis
    
        pvmesh = vis.mesh_to_pv_mesh(pipemesh)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
        pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
        pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    
        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    
    
        # point sources at cell centres for streamlines
    
        points = np.zeros((pipemesh._centroids.shape[0], 3))
        points[:, 0] = pipemesh._centroids[:, 0]
        points[:, 1] = pipemesh._centroids[:, 1]
        point_cloud = pv.PolyData(points)

        passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)
        active_swarm_points = uw.visualisation.swarm_to_pv_cloud(swarm)
    
        pvstream = pvmesh.streamlines_from_source(
            point_cloud, vectors="V", integration_direction="forward", max_steps=10, 
        )

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="Omega",
            use_transparency=False,
            opacity=1.0,
            show_scalar_bar=False,
        )

        
        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], 
                      mag=0.025 / U0, opacity=0.75, 
                      show_scalar_bar=False)
        
        pl.add_mesh(pvstream, show_scalar_bar=False)

        pl.add_points(
            passive_swarm_points,
            color="Black",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.5,
        )
    

        pl.add_points(
            active_swarm_points,
            color="DarkGreen",
            render_points_as_spheres=True,
            point_size=2,
            opacity=0.25,
        )
    

        
        pl.camera.SetPosition(0.75, 0.2, 1.5)
        pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
        pl.camera.SetClippingRange(1.0, 8.0)
    
    
        # pl.camera_position = "xz"
        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 1280),
            return_img=False,
        )
    
        pl.clear()
    
# -



# +
ts = 0
elapsed_time = 0.0
dt_ns = 0.01
delta_t_cfl = 5 * navier_stokes.estimate_dt()
delta_t = min(delta_t_cfl, dt_ns)

navier_stokes.DuDt.update(delta_t)
# -


for step in range(0, 251): #1500
    delta_t_cfl = 5 * navier_stokes.estimate_dt()

    if step % 10 == 0:
        delta_t = min(delta_t_cfl, dt_ns)

    navier_stokes.solve(timestep=dt_ns, zero_init_guess=False)

    # update passive swarm
    passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    # update material point swarm
    swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    
    # add new points at the inflow
    npoints = 200
    passive_swarm.dm.addNPoints(npoints)
    with passive_swarm.access(passive_swarm.particle_coordinates):
        for i in range(npoints):
            passive_swarm.particle_coordinates.data[
                -1 : -(npoints + 1) : -1, :
            ] = np.array([0.0, 0.195] + 0.01 * np.random.random((npoints, 2)))

    if uw.mpi.rank == 0:
        print("Timestep {}, t {}, dt {}, dt_s {}".format(ts, elapsed_time, delta_t, delta_t_cfl))

    if ts % 10 == 0:
        nodal_vorticity_from_v.solve()
        plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

        pipemesh.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, vorticity, St],
            outputPath=outdir,
            index=ts,
        )

        passive_swarm.write_timestep(
            expt_name,
            "passive_swarm",
            swarmVars=None,
            outputPath=outdir,
            index=ts,
            force_sequential=True,
        )

    elapsed_time += delta_t
    ts += 1

# +
# check the mesh if in a notebook / serial

if 1 and uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    ustar = navier_stokes.Unknowns.DuDt.psi_star[0].sym
    pvmesh.point_data["Vs"] = vis.scalar_fn_to_pv_points(pvmesh, ustar.dot(ustar))

    # point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)
    active_swarm_points = uw.visualisation.swarm_to_pv_cloud(swarm)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    points = vis.swarm_to_pv_cloud(passive_swarm)
    point_cloud = pv.PolyData(points)

    pl0 = pv.Plotter(window_size=(1000, 750))


    pl0.add_points(
        passive_swarm_points,
        color="DarkGreen",
        render_points_as_spheres=False,
        point_size=3,
        opacity=0.66,
    )

    pl0.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Omega",
        use_transparency=False,
        opacity=1.0,
    )
    
    pl0.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.03 / U0, opacity=0.75)
    pl0.add_mesh(pvstream)

    pl0.add_points(
        active_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=3,
        opacity=0.5,
    )

    pl0.show()
# -




