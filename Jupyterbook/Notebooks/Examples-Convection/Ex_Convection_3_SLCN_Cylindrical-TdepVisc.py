# # Temperature-dependent viscosity convection, Cylindrical domain (benchmark)
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
#
# The next step is to add particles at node points and sample back along the streamlines to find values of the T field at a previous time.
#
# This has a free-slip lower boundary and a fixed upper boundary (simplifies the null space, and the lid is stagnant anyway)

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy


# +
# Parameters

r_o = 1.0
r_i = 0.5
res = 1 / 24

Rayleigh = 1.0e6 / (r_o - r_i) ** 3

log10_delta_eta = 4


# +
# Visualisation

import pyvista as pv

pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 250]
pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.jupyter_backend = "trame"
pv.global_theme.smooth_shading = True


# +
meshdisc = uw.meshing.Annulus(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=float(res),
    qdegree=3,
)

#
meshdisc.vtk("tmp_ann_mesh.vtk")

# -


v_soln = uw.discretisation.MeshVariable("U", meshdisc, meshdisc.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshdisc, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable("T", meshdisc, 1, degree=3)
meshr = uw.discretisation.MeshVariable(r"r", meshdisc, 1, degree=1)


# +
radius_fn = sympy.sqrt(
    meshdisc.rvec.dot(meshdisc.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshdisc.X / (radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x, y = meshdisc.CoordinateSystem.X
ra, th = meshdisc.CoordinateSystem.xR

hw = 1000.0 / res
surface_fn_a = sympy.exp(-(((ra - r_o) / r_o) ** 2) * hw)
surface_fn = sympy.exp(-(((meshr.sym[0] - r_o) / r_o) ** 2) * hw)

base_fn_a = sympy.exp(-(((ra - r_i) / r_o) ** 2) * hw)
base_fn = sympy.exp(-(((meshr.sym[0] - r_i) / r_o) ** 2) * hw)

free_slip_penalty_upper = v_soln.sym.dot(unit_rvec) * unit_rvec * surface_fn
free_slip_penalty_lower = v_soln.sym.dot(unit_rvec) * unit_rvec * base_fn

# +
# Create Stokes object

stokes = Stokes(
    meshdisc,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
# stokes.petsc_options.delValue("ksp_monitor")
# stokes.petsc_options["snes_test_jacobian"] = None

# T dependent visc
delta_eta = 10**log10_delta_eta

stokes.petsc_options["snes_rtol"] = 1 / delta_eta

viscosity = delta_eta * sympy.exp(-sympy.log(delta_eta) * t_soln.sym[0])
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.penalty = 0.0

stokes.saddle_preconditioner = 1.0 / viscosity

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
# stokes.add_dirichlet_bc((0.0,0.0), "Lower", (0,1))

# Buoyancy force RHS plus free slip surface enforcement
buoyancy_force = Rayleigh * t_soln.sym[0] * unit_rvec * (1.0 - base_fn)
penalty_terms = 10000000 * free_slip_penalty_lower

stokes.bodyforce = buoyancy_force - penalty_terms


# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshdisc,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5


# +
# Define T boundary conditions via a sympy function

import sympy

init_t = 0.9 + 0.05 * (sympy.cos(sympy.pi * th / 2)) * sympy.cos(
    0.5 * np.pi * (ra - r_i) / (r_o - r_i)
)

adv_diff.add_dirichlet_bc(1.0, "Lower")
adv_diff.add_dirichlet_bc(0.0, "Upper")


# +
with meshdisc.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(init_t, t_soln.coords, meshdisc.N).reshape(
        -1, 1
    )

with meshdisc.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(
        sympy.sqrt(x**2 + y**2), meshdisc.data, meshdisc.N
    )  # cf radius_fn which is 0->1
# -

# check the stokes solve is set up and that it converges
stokes.solve(zero_init_guess=True)


# Check the diffusion part of the solve converges
adv_diff.solve(timestep=0.1 * stokes.estimate_dt())

# +
# adv_diff

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh =  vis.mesh_to_pv_mesh(meshdisc)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["Tp"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    point_cloud = pv.PolyData(points)

    # point sources at cell centres
    subsample = 2
    cpoints = np.zeros((meshdisc._centroids[::subsample, 0].shape[0], 3))
    cpoints[:, 0] = meshdisc._centroids[::subsample, 0]
    cpoints[:, 1] = meshdisc._centroids[::subsample, 1]
    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
                                                cpoint_cloud,
                                                vectors="V",
                                                integrator_type=2,
                                                integration_direction="forward",
                                                compute_vorticity=False,
                                                max_steps=100,
                                                surface_streamlines=True,
                                            )

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh, "Gray", "wireframe")

    # pl.add_mesh(
    #     pvmesh, cmap="coolwarm", edge_color="Black",
    #     show_edges=True, scalars="T", use_transparency=False, opacity=0.5,
    # )

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=3,
        opacity=0.33,
    )

    pl.add_mesh(pvstream, opacity=0.5)
    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=1.0e-4)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshdisc)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

        # point sources at cell centres
        cpoints = np.zeros((meshdisc._centroids.shape[0], 3))
        cpoints[:, 0] = meshdisc._centroids[:, 0]
        cpoints[:, 1] = meshdisc._centroids[:, 1]
        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
                                                    cpoint_cloud,
                                                    vectors="V",
                                                    integrator_type=45,
                                                    integration_direction="forward",
                                                    compute_vorticity=False,
                                                    max_steps=100,
                                                    surface_streamlines=True,
                                                )


        points = vis.meshVariable_to_pv_cloud(t_soln)
        points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
        point_cloud = pv.PolyData(points)
        
        pl = pv.Plotter(window_size=(750, 750))

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.5,
        )

        pl.add_mesh(pvstream, opacity=0.4)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("V")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )
        # pl.show()
        pl.close()

        pvmesh.clear_data()
        pvmesh.clear_point_data()

        pv.close_all()

t_step = 0

# +
# Convection model / update in time

##
## There is a strange interaction here between the solvers if the zero_guess is
## set to False
##

expt_name = f"output/Ra1e6_cyl_eta1e{log10_delta_eta}"

for step in range(0, 1000):
    stokes.solve(zero_init_guess=True)
    delta_t = 2.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=True)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
    #         print(tstats)

    if t_step % 5 == 0:
        plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1

# savefile = "{}_ts_{}.h5".format(expt_name,step)
# meshdisc.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshdisc.generate_xdmf(savefile)

# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshdisc.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshdisc.generate_xdmf(savefile)


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshdisc)

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1e-4, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7,
        opacity=0.25,
    )

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")


