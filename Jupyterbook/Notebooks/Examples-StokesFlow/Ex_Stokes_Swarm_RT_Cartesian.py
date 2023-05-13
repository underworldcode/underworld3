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

# # Rayleigh Taylor - swarm materials
#
# We introduce the notion of an `IndexSwarmVariable` which automatically generates masks for a swarm
# variable that consists of discrete level values (integers).
#
# For a variable $M$, the mask variables are $\left\{ M^0, M^1, M^2 \ldots M^{N-1} \right\}$ where $N$ is the number of indices (e.g. material types) on the variable. This value *must be defined in advance*.
#
# The masks are orthogonal in the sense that $M^i * M^j = 0$ if $i \ne j$, and they are complete in the sense that $\sum_i M^i = 1$ at all points.
#
# The masks are implemented as continuous mesh variables (the user can specify the interpolation order) and so they are also differentiable (once).
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy
render = True

cell_size = uw.options.getReal("mesh_cell_size", default=1.0/32)
particle_fill = uw.options.getInt("particle_fill", default=7)
viscosity_ratio = uw.options.getReal("rt_viscosity_ratio", default=1.0)



# +
lightIndex = 0
denseIndex = 1

boxLength = 0.9142
boxHeight = 1.0
viscosityRatio = viscosity_ratio
amplitude = 0.02
offset = 0.2
model_end_time = 300.0

# material perturbation from van Keken et al. 1997
wavelength = 2.0 * boxLength
k = 2.0 * np.pi / wavelength
# -

meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(boxLength, boxHeight),
    cellSize=cell_size,
    regular=False,
    qdegree=2,
)


# +
import sympy

# Some useful coordinate stuff

x, y = meshbox.CoordinateSystem.X

# -

v_soln = uw.discretisation.MeshVariable(r"U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(r"P", meshbox, 1, degree=1)
m_cont = uw.discretisation.MeshVariable(r"M_c", meshbox, 1, degree=1, continuous=True)


swarm = uw.swarm.Swarm(mesh=meshbox)
material = uw.swarm.IndexSwarmVariable(
    r"M", swarm, indices=2, proxy_degree=1, proxy_continuous=False
)
swarm.populate(fill_param=particle_fill)


# +
with swarm.access(material):
    material.data[...] = 0

with swarm.access(material):
    perturbation = offset + amplitude * np.cos(
        k * swarm.particle_coordinates.data[:, 0]
    )
    material.data[:, 0] = np.where(
        perturbation > swarm.particle_coordinates.data[:, 1], lightIndex, denseIndex
    )

material.sym
# -


print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)


X = meshbox.CoordinateSystem.X

mat_density = np.array([0, 1])  # lightIndex, denseIndex
density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([viscosityRatio, 1])
viscosity = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

# +
# Create Stokes object

stokes = uw.systems.Stokes(
    meshbox, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

# Set some things
import sympy
from sympy import Piecewise

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity

stokes.bodyforce = sympy.Matrix([0, -density])
stokes.saddle_preconditioner = 1.0 / viscosity

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Bottom", "Top"], 1
)  # top/bottom: components, function, markers
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Left", "Right"], 0
)  # left/right: components, function, markers
# -


stokes.rtol = 1.0e-3 # rough solution is all that's needed

m_solver = uw.systems.Projection(meshbox, m_cont)
m_solver.uw_function = material.sym[1]
m_solver.smoothing = 1.0e-3
m_solver.solve()

stokes.solve(zero_init_guess=True)

# +
# check the solution


if uw.mpi.size == 1 and render:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    meshbox.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    # check the solution

if uw.mpi.size == 1 and render:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    meshbox.vtk("tmp_box.vtk")
    pvmesh = pv.read("tmp_box.vtk")

    pvmesh.point_data["rho"] = uw.function.evaluate(density, meshbox.data)
    pvmesh.point_data["visc"] = uw.function.evaluate(sympy.log(viscosity), meshbox.data)
    pvmesh.point_data["M"] = uw.function.evaluate(m_cont.sym[0], meshbox.data)

    velocity = np.zeros((meshbox.data.shape[0], 3))
    velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
    velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

    pvmesh.point_data["V"] = velocity

    # point sources at cell centres

    cpoints = np.zeros((meshbox._centroids[::4].shape[0], 3))
    cpoints[:, 0] = meshbox._centroids[::4, 0]
    cpoints[:, 1] = meshbox._centroids[::4, 1]
    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="forward",
        compute_vorticity=False,
        max_steps=25,
        surface_streamlines=True,
    )

    with swarm.access():
        spoints = np.zeros((swarm.data.shape[0], 3))
        spoints[:, 0] = swarm.data[:, 0]
        spoints[:, 1] = swarm.data[:, 1]
        spoints[:, 2] = 0.0

    spoint_cloud = pv.PolyData(spoints)

    with swarm.access():
        spoint_cloud.point_data["M"] = material.data[...]

    pl = pv.Plotter(window_size=(500, 500))

    pl.add_mesh(pvstream, opacity=1.0)
    pl.add_mesh(
        pvmesh,
        cmap="Blues_r",
        edge_color="Gray",
        show_edges=True,
        scalars="M",
        opacity=0.75,
    )
    pl.add_points(
        spoint_cloud,
        cmap="Reds_r",
        scalars="M",
        render_points_as_spheres=True,
        point_size=3,
        opacity=0.5,
    )

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# +
import numpy as np
import pyvista as pv
import vtk

if uw.mpi.size == 1:

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "pythreejs"
    pv.global_theme.smooth_shading = False
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]


    pl = pv.Plotter()


def plot_mesh(filename):

    if uw.mpi.size == 1:

        meshbox.vtk("tmp_box.vtk")
        pvmesh = pv.read("tmp_box.vtk")

        pvmesh.point_data["rho"] = uw.function.evaluate(density, meshbox.data)
        pvmesh.point_data["visc"] = uw.function.evaluate(
            sympy.log(viscosity), meshbox.data
        )
        pvmesh.point_data["M"] = uw.function.evaluate(m_cont.sym[0], meshbox.data)

        velocity = np.zeros((meshbox.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
        velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

        pvmesh.point_data["V"] = velocity

        # point sources at cell centres
        subsample = 3
        cpoints = np.zeros((meshbox._centroids[::subsample].shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[::subsample, 0]
        cpoints[:, 1] = meshbox._centroids[::subsample, 1]
        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
            cpoint_cloud,
            vectors="V",
            integrator_type=45,
            integration_direction="forward",
            compute_vorticity=False,
            max_steps=25,
            surface_streamlines=True,
        )

        with swarm.access():
            spoints = np.zeros((swarm.data.shape[0], 3))
            spoints[:, 0] = swarm.data[:, 0]
            spoints[:, 1] = swarm.data[:, 1]
            spoints[:, 2] = 0.0

        spoint_cloud = pv.PolyData(spoints)

        with swarm.access():
            spoint_cloud.point_data["M"] = material.data[...]

        pl.clear()

        # pl.add_mesh(pvmesh, "Gray",  "wireframe")
        # pl.add_arrows(arrow_loc, velocity_field, mag=0.2/vmag, opacity=0.5)

        pl.add_mesh(pvstream, opacity=1)
        pl.add_mesh(
            pvmesh,
            cmap="Blues_r",
            edge_color="Gray",
            show_edges=True,
            scalars="M",
            opacity=0.75,
        )

        pl.add_points(
            spoint_cloud,
            cmap="Reds_r",
            scalars="M",
            render_points_as_spheres=True,
            point_size=3,
            opacity=0.3,
        )

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("V")
        # pl.remove_scalar_bar("rho")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1250, 1250),
            return_img=False,
        )

        return


# -

t_step = 0

# +
# Update in time

expt_name = "output/swarm_rt"

for step in range(0, 200):

    stokes.solve(zero_init_guess=False)
    m_solver.solve(zero_init_guess=False)
    delta_t = min(10.0, stokes.estimate_dt())

    # update swarm / swarm variables

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))

    # advect swarm
    swarm.advection(v_soln.sym, delta_t)

    if t_step % 5 == 0:
        plot_mesh(filename="{}_step_{}".format(expt_name, t_step))

        # "Checkpoints"
        savefile = f"output/swarm_rt_xy"    
        meshbox.write_checkpoint(savefile, 
                              meshUpdates=False, 
                              meshVars=[p_soln,v_soln, m_cont], 
                              index=step)


    t_step += 1
