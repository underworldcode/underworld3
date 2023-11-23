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

# # Multiple materials - Linear stokes sinker
#
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

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

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
# -


uw

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


# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0.0, 0.7)

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

# +
stokes = uw.systems.Stokes(mesh)

v = stokes.Unknowns.u
p = stokes.Unknowns.p

# Set some options
stokes.penalty = 1.0

# Set some bcs
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")
# -


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_continuous=False, proxy_degree=1
)
swarm.populate(fill_param=4)

blob = np.array([[sphereCentre[0], sphereCentre[1], sphereRadius, 1]])


with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(blob.shape[0]):
        cx, cy, r, m = blob[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# %%
tracer = np.zeros(shape=(1, 2))
tracer[:, 0], tracer[:, 1] = x_pos, y_pos

density = densityBG * material.sym[0] + densitySphere * material.sym[1]
viscosity = viscBG * material.sym[0] + viscSphere * material.sym[1]


# +
# viscosity = sympy.Max( sympy.Min(viscosityMat, eta_max), eta_min)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * density])

# -

render = True




# +
import pyvista as pv
pl = pv.Plotter(notebook=True)
pl.camera.position = (1.1, 1.5, 0.0)
pl.camera.focal_point = (0.2, 0.3, 0.3)
pl.camera.up = (0.0, 1.0, 0.0)
pl.camera.zoom(1.4)

def plot_T_mesh(filename):
    if not render:
        return

    import numpy as np
    import pyvista as pv
    import underworld3.visualisation

    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
    point_cloud = underworld3.visualisation.swarm_to_pv_cloud(swarm)

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


nstep = 15

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

    mesh.write_timestep("stokesSinker", meshUpdates=False, meshVars=[p, v], index=step)

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

# +
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation

pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
pvmesh.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v.sym)
pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

swarm_points = underworld3.visualisation.swarm_to_pv_cloud(swarm)
swarm_points.point_data["M"] = uw.visualisation.scalar_fn_to_pv_points(swarm_points, material.visMask())

velocity_points = underworld3.visualisation.meshVariable_to_pv_cloud(v)
velocity_points.point_data["X"] = uw.visualisation.coords_to_pv_coords(v.coords)
velocity_points.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v.sym)
# -
# ## check if that worked

if uw.mpi.size == 1:
    
    import numpy as np
    import pyvista as pv
    import underworld3 as uw
    import underworld3.visualisation

    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

    swarm_points = underworld3.visualisation.swarm_to_pv_cloud(swarm)
    swarm_points.point_data["M"] = uw.visualisation.scalar_fn_to_pv_points(swarm_points, material.visMask())
    
    velocity_points = underworld3.visualisation.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v.sym)

    pvstream = pvmesh.streamlines_from_source(
        swarm_points,
        vectors="V",
        integration_direction="both",
        max_steps=10,
        surface_streamlines=True,
        max_step_length=0.05,
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, "Black", "wireframe")

    streamlines = pl.add_mesh(pvstream, opacity=0.25)
    streamlines.SetVisibility(False)


    pl.add_mesh(
        swarm_points,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="M",
        use_transparency=False,
        point_size=2.0,
        opacity=0.5,
        show_scalar_bar=False,
    )

    pl.add_mesh(
        velocity_points,

    )

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=3.0, opacity=0.33, show_scalar_bar=False)


    ## Widgets

    def toggle_streamlines(flag):
        streamlines.SetVisibility(flag)
        
    def toggle_arrows(flag):
        arrows.SetVisibility(flag)

    pl.add_checkbox_button_widget(toggle_streamlines, value=False, size = 10, position = (10, 20))
    pl.add_checkbox_button_widget(toggle_arrows, value=False, size = 10, position = (30, 20))



    # pl.screenshot(filename="SinkerSolution_hr.png", window_size=(4000, 2000))



    

    pl.show(cpos="xy")

velocity_points.point_data["V"]

uw.function.evalf(v.sym[0], velocity_points.points[:,0:2])

velocity_points.point_data["V"]

# +

velocity_points.points.min()
# -

