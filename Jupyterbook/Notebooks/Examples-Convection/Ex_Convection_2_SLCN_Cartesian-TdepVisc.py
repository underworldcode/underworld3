# # Constant viscosity convection, Cartesian domain (benchmark)
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
#
# The next step is to add particles at node points and sample back along the streamlines to find values of the T field at a previous time.
#
# (Note, we keep all the pieces from previous increments of this problem to ensure that we don't break something along the way)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

# -


meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0, regular=True, qdegree=3
)


meshbox.dm.view()

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, -5.0]
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")

    pl = pv.Plotter(notebook=True)
    pl.add_mesh(pvmesh, edge_color="Gray", show_edges=True)
    pl.show(cpos="xy")

    pv.close_all()
# -


v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)


# +
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
# stokes.petsc_options.delValue("ksp_monitor")

stokes.petsc_options["snes_rtol"] = 1.0e-5
# stokes.petsc_options["snes_test_jacobian"] = None

# T dependent visc
delta_eta = 10**4
viscosity = delta_eta * sympy.exp(-sympy.log(delta_eta) * t_soln.sym[0])
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=viscosity)

stokes.penalty = 0.0

stokes.saddle_preconditioner = 1.0 / viscosity

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

# Some useful coordinate stuff

x, y = meshbox.X


# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.material_properties = adv_diff.constitutive_model.Parameters(diffusivity=k)

adv_diff.theta = 0.5


# +
# Define T boundary conditions via a sympy function

import sympy

init_t = 0.01 * sympy.sin(5.0 * x) * sympy.sin(np.pi * y) + (1.0 - y)

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# -


buoyancy_force = 1.0e6 * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

with meshbox.access(v_soln, p_soln):
    v_soln.data[:, :] *= 0.95 + 0.1 * np.random.random(size=v_soln.data.shape)
    p_soln.data[:, :] *= 0.95 + 0.1 * np.random.random(size=p_soln.data.shape)

# check the stokes solve is set up and that it converges
stokes.solve(zero_init_guess=True)


# Check the diffusion part of the solve converges
adv_diff.solve(timestep=0.1 * stokes.estimate_dt())

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")

    velocity = np.zeros((meshbox.data.shape[0], 3))
    velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
    velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

    pvmesh.point_data["V"] = velocity * 3

    points = np.zeros((t_soln.coords.shape[0], 3))
    points[:, 0] = t_soln.coords[:, 0]
    points[:, 1] = t_soln.coords[:, 1]

    point_cloud = pv.PolyData(points)

    with meshbox.access():
        point_cloud.point_data["Tp"] = t_soln.data.copy()

    # point sources at cell centres

    cpoints = np.zeros((meshbox._centroids.shape[0] // 4, 3))
    cpoints[:, 0] = meshbox._centroids[::4, 0]
    cpoints[:, 1] = meshbox._centroids[::4, 1]

    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=2,
        integration_direction="forward",
        compute_vorticity=False,
        max_steps=1000,
        surface_streamlines=True,
    )

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Gray', 'wireframe')

    # pl.add_mesh(
    #     pvmesh, cmap="coolwarm", edge_color="Black",
    #     show_edges=True, scalars="T", use_transparency=False, opacity=0.5,
    # )

    pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=True, point_size=10, opacity=0.33)

    pl.add_mesh(pvstream, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
    pvmesh.clear_data()
    pvmesh.clear_point_data()


# -


pvmesh.clear_point_data()


adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5

# +

pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 750]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

meshbox.vtk("tmp_box_mesh.vtk")


def plot_T_mesh(filename):

    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = "white"
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "panel"
        pv.global_theme.smooth_shading = True
        pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
        pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

        meshbox.vtk("tmp_box_mesh.vtk")
        pvmesh = pv.read("tmp_box_mesh.vtk")

        velocity = np.zeros((meshbox.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
        velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

        pvmesh.point_data["V"] = velocity / 333
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)

        # point sources at cell centres

        cpoints = np.zeros((meshbox._centroids.shape[0] // 4, 3))
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

        points = np.zeros((t_soln.coords.shape[0], 3))
        points[:, 0] = t_soln.coords[:, 0]
        points[:, 1] = t_soln.coords[:, 1]

        point_cloud = pv.PolyData(points)

        with meshbox.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        pl = pv.Plotter()

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.5)

        pl.add_mesh(pvstream, opacity=0.4)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("V")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False)
        # pl.show()
        pl.close()

        pvmesh.clear_data()
        pvmesh.clear_point_data()

        pv.close_all()


# -

t_step = 0

# +
# Convection model / update in time

##
## There is a strange interaction here between the solvers if the zero_guess is
## set to False
##

expt_name = "output/Ra1e6_eta1e4"

for step in range(0, 1000):

    # # perturb the old solution for solver convergence reasons
    # with meshbox.access(v_soln, p_soln):
    #     v_soln.data[:,:] *= (0.95+0.1*np.random.random(size=v_soln.data.shape))
    #     p_soln.data[:,:] *= (0.95+0.1*np.random.random(size=p_soln.data.shape))

    stokes.solve(zero_init_guess=True)
    delta_t = 5.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
    #         print(tstats)

    if t_step % 5 == 0:
        plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1

# savefile = "{}_ts_{}.h5".format(expt_name,step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)

# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)


# +


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    pv.start_xvfb()

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")

    points = np.zeros((t_soln.coords.shape[0], 3))
    points[:, 0] = t_soln.coords[:, 0]
    points[:, 1] = t_soln.coords[:, 1]

    point_cloud = pv.PolyData(points)

    with meshbox.access():
        point_cloud.point_data["T"] = t_soln.data.copy()

    with meshbox.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=True, point_size=7.5, opacity=0.25)

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")
