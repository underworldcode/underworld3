# # Non-linear viscosity convection, Cartesian domain (benchmark)
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
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 24.0, qdegree=3
)

# +
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)

visc = uw.discretisation.MeshVariable(r"\eta(\dot\varepsilon)", meshbox, 1, degree=1)


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

# Linear visc
delta_eta = 1.0e6

viscosity_L = delta_eta * sympy.exp(-sympy.log(delta_eta) * t_soln.sym[0])
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity_L

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity
stokes.penalty = 0.0

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))
# -


stokes.constitutive_model.solver = stokes

isinstance(Stokes, uw.systems.SNES_SaddlePoint)

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

adv_diff = uw.systems.AdvDiffusion(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.theta = 0.5


# +
# Create a scalar function calculator that we can use to obtain viscosity etc

scalar_projection = uw.systems.Projection(meshbox, visc)
scalar_projection.uw_function = 0.1 + 10.0 / (1.0 + stokes._Einv2)
scalar_projection.smoothing = 1.0e-6


# +
expt_name = "output/Ra1e6_NL"

Rayleigh = 1.0e6
buoyancy_force = Rayleigh * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# +
# Define T boundary conditions via a sympy function

import sympy

init_t = 0.9 * (0.05 * sympy.cos(sympy.pi * x) + sympy.cos(0.5 * np.pi * y)) + 0.05

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# -


# Linear problem for initial solution of velocity field
stokes.solve()

# +
# Now make the viscosity non-linear

viscosity_NL = viscosity_L * (0.1 + 10.0 / (1.0 + stokes._Einv2))

stokes.constitutive_model.Parameters.viscosity = viscosity_NL
stokes.saddle_preconditioner = 1 / viscosity_NL
# -

stokes.saddle_preconditioner

stokes.solve(zero_init_guess=False)

# Check the diffusion part of the solve converges
adv_diff.solve(timestep=0.01 * stokes.estimate_dt())


# Compute viscosity field
scalar_projection.solve()

with meshbox.access():
    print(visc.min(), visc.max())

# +
# check the mesh if in a notebook / serial
import pyvista as pv

pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 750]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

pl = pv.Plotter()

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")

    velocity = np.zeros((meshbox.data.shape[0], 3))
    velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
    velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

    pvmesh.point_data["V"] = 10.0 * velocity / velocity.max()
    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)
    pvmesh.point_data["eta"] = uw.function.evaluate(visc.fn, meshbox.data)

    # point sources at cell centres

    cpoints = np.zeros((meshbox._centroids[::2].shape[0], 3))
    cpoints[:, 0] = meshbox._centroids[::2, 0]
    cpoints[:, 1] = meshbox._centroids[::2, 1]
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

    ## PLOTTING

    pl.clear()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Gray",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.5,
    )

    pl.add_mesh(
        pvmesh,
        cmap="Greys",
        show_edges=False,
        scalars="eta",
        use_transparency=False,
        opacity=0.25,
    )

    # pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.5)

    pl.add_mesh(pvstream, opacity=0.4)

    pl.remove_scalar_bar("T")
    pl.remove_scalar_bar("V")
    pl.remove_scalar_bar("eta")

    # pl.screenshot(filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False)
    pl.show()


# +
import pyvista as pv


pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 750]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "pythreejs"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

pl = pv.Plotter()


def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        import numpy as np
        import vtk

        meshbox.vtk("tmp_box_mesh.vtk")
        pvmesh = pv.read("tmp_box_mesh.vtk")

        velocity = np.zeros((meshbox.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
        velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

        pvmesh.point_data["V"] = 10.0 * velocity / velocity.max()
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)
        pvmesh.point_data["eta"] = uw.function.evaluate(visc.fn, meshbox.data)

        # point sources at cell centres

        cpoints = np.zeros((meshbox._centroids[::2].shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[::2, 0]
        cpoints[:, 1] = meshbox._centroids[::2, 1]
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

        ## PLOTTING

        pl.clear()

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_mesh(
            pvmesh,
            cmap="Greys",
            show_edges=False,
            scalars="eta",
            use_transparency=False,
            opacity=0.25,
        )

        # pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.5)

        pl.add_mesh(pvstream, opacity=0.4)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("V")
        pl.remove_scalar_bar("eta")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )


# +
# Convection model / update in time


for step in range(0, 250):
    stokes.solve(zero_init_guess=False)
    delta_t = 5.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    # savefile = "{}_ts_{}.h5".format(expt_name,step)
    # meshbox.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshbox.generate_xdmf(savefile)

pass

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
    pv.global_theme.jupyter_backend = "pythreejs"
    pv.global_theme.smooth_shading = True

    pv.start_xvfb()

    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

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

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7.5,
        opacity=0.25,
    )

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")
