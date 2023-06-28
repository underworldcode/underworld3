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

# # Multiple materials - Linear stokes sinker
#
# This is the notorious "Stokes sinker" problem in which we have a dense and "rigid" (highly viscous) blob sinking in a low-viscosity fluid. This combination of high velocity and low strain rate is challenging for iterative solvers and there is a limit to the viscosity jujmp that can be introduced before the solvers fail to converge.
#
# ![Sinker image with streamlines](images/SinkerSolution.png)
#
# We introduce the notion of an `IndexSwarmVariable` which automatically generates masks for a swarm
# variable that consists of discrete level values (integers).
#
# For a variable $M$, the mask variables are $\left\{ M^0, M^1 \ldots M^{N-1} \right\}$ where $N$ is the number of indices (e.g. material types) on the variable. This value *must be defined in advance*.
#
# The masks are orthogonal in the sense that $M^i * M^j = 0$ if $i \ne j$, and they are complete in the sense that $\sum_i M^i = 1$ at all points.
#
# The masks are implemented as continuous mesh variables (the user can specify the interpolation order) and so they are also differentiable (once).

# +
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import os

os.environ["UW_TIMING_ENABLE"] = "1"

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)



# +
# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 2

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass
# -

sys = PETSc.Sys()
sys.pushErrorHandler("traceback")


if problem_size <= 1:
    res = 8
elif problem_size == 2:
    res = 16
elif problem_size == 3:
    res = 32
elif problem_size == 4:
    res = 48
elif problem_size == 5:
    res = 64
elif problem_size >= 6:
    res = 128


# + tags=[]
# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0.0, 0.7)
# -

# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1

# Set constants for the viscosity and density of the sinker.
viscBG = 1.0
viscSphere = 1.0e6

expt_name = f"output/stinker_eta{viscSphere}_rho10_res{res}"

densityBG = 1.0
densitySphere = 10.0

# location of tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1] - sphereRadius

nsteps = 0

swarmGPC = 2

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
)

# ## Create Stokes object

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(v)


stokes.add_dirichlet_bc(
    (0.0), ["Top", "Bottom"], [1]
)  # top/bottom: components, function, markers
stokes.add_dirichlet_bc(
    (0.0), ["Left", "Right"], [0]
)  # left/right: components, function, markers


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_continuous=False, proxy_degree=1
)
swarm.populate(fill_param=4)

# +
# print("Coarsening", flush=True)
# dmh = mesh.dm.coarsenHierarchy(3)
# print("Coarsening ... complete", flush=True)
# -

blob = np.array([[sphereCentre[0], sphereCentre[1], sphereRadius, 1]])


with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(blob.shape[0]):
        cx, cy, r, m = blob[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# %%
### add tracer for sinker velocity
tracer = np.zeros(shape=(1, 2))
tracer[:, 0], tracer[:, 1] = x_pos, y_pos

# %%
mat_density = np.array([densityBG, densitySphere])

density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([viscBG, viscSphere])

viscosityMat = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

# viscosity = sympy.Max( sympy.Min(viscosityMat, eta_max), eta_min)
viscosity = viscBG * material.sym[0] + viscSphere * material.sym[1]

render = True

import numpy as np
import pyvista as pv
import vtk

pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 750]
pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

pl = pv.Plotter(notebook=True)


def plot_T_mesh(filename):

    if not render:
        return

    import numpy as np
    import pyvista as pv
    import vtk

    mesh.vtk("tmpMsh.vtk")
    pvmesh = pv.read("tmpMsh.vtk")

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        points[:, 2] = 0.0

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    ## Plotting into existing pl (memory leak in panel code)
    pl.clear()

    pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.5,
    )

    pl.screenshot(
        filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False
    )


stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * density])
stokes.penalty = 1.0


# +
# stokes.petsc_options.view()

snes_rtol = 1.0e-6
stokes.tolerance = snes_rtol

# stokes.petsc_options["snes_converged_reason"] = None
# stokes.petsc_options["ksp_type"] = "gmres"
# stokes.petsc_options["ksp_rtol"] = 1.0e-9
# stokes.petsc_options["ksp_atol"] = 1.0e-12
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-8
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-8
# stokes.petsc_options["snes_atol"] = 0.1 * snes_rtol # by inspection
stokes.petsc_options["ksp_monitor"] = None


# -


nstep = 10

step = 0
time = 0.0
nprint = 0.0

# %%
tSinker = np.zeros(nstep)
ySinker = np.zeros(nstep)

# +
from underworld3 import timing

timing.reset()
timing.start()
stokes._setup_terms()
stokes.solve(zero_init_guess=True)
timing.print_table()
# -

while step < nstep:
    ### Get the position of the sinking ball
    ymin = tracer[:, 1].min()
    ySinker[step] = ymin
    tSinker[step] = time


    ### estimate dt
    dt = stokes.estimate_dt()
    if uw.mpi.rank == 0:
        print(f"dt = {dt}", flush=True)

    ## This way should be a bit safer in parallel where particles can move
    ## processors in the middle of the calculation if you are not careful
    ## PS - the function.evaluate needs fixing to take sympy.Matrix functions

    swarm.advection(stokes.u.sym, dt, corrector=True)
    
    ### solve stokes
    stokes.solve(zero_init_guess=False)

    ### print some stuff
    if uw.mpi.size == 1:
        print(f"Step: {str(step).rjust(3)}, time: {time:6.2f}, tracer:  {ymin:6.2f}")
        plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    mesh.write_timestep(
        "stokesSinker", meshUpdates=False, meshVars=[p, v], index=step
    )

    step += 1
    time += dt


# %%
if uw.mpi.rank == 0:
    print("Initial position: t = {0:.3f}, y = {1:.3f}".format(tSinker[0], ySinker[0]))
    print(
        "Final position:   t = {0:.3f}, y = {1:.3f}".format(
            tSinker[nsteps - 1], ySinker[nsteps - 1]
        )
    )

    import matplotlib.pyplot as pyplot

    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tSinker, ySinker)
    ax.set_xlabel("Time")
    ax.set_ylabel("Sinker position")

# ## check if that worked

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    # mesh.vtk("ignore_periodic_mesh.vtk")
    mesh.vtk("tmpMsh.vtk")

    pvmesh = pv.read("tmpMsh.vtk")

    # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

    with mesh.access():
        vsol = v.data.copy()

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        points[:, 2] = 0.0

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v.fn, mesh.data)
    pvmesh.point_data["V"] = v_vectors

    arrow_loc = np.zeros((v.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v.coords[...]

    arrow_length = np.zeros((v.coords.shape[0], 3))
    arrow_length[:, 0:2] = vsol[...]

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_steps=10,
        surface_streamlines=True,
        max_step_length=0.33,
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, "Black", "wireframe")

    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="rho",
    #                 use_transparency=False, opacity=0.95)

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
    #               use_transparency=False, opacity=0.5)

    pl.add_mesh(pvstream, opacity=0.5)

    pl.add_mesh(
        point_cloud,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="M",
        use_transparency=False,
        point_size=2.0,
        opacity=0.75,
    )

    # pl.add_arrows(arrow_loc, arrow_length, mag=2.0, opacity=0.33)
    # pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.remove_scalar_bar("M")
    pl.remove_scalar_bar("V")
    # pl.screenshot(filename="SinkerSolution_hr.png", window_size=(4000, 2000))

    pl.show(cpos="xy")


