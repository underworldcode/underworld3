# # Constant viscosity convection, Cartesian domain (benchmark)
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
#
# The next step is to add particles at node points and sample back along the streamlines to find values of the T field at a previous time.
#
# (Note, we keep all the pieces from previous increments of this problem to ensure that we don't break something along the way)

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
# -

meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 32.0,
    regular=False,
    qdegree=3,
)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    # pv.start_xvfb()

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")
# -

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
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

# Constant viscosity

viscosity = 1
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.tolerance = 1.0e-3


# free slip.
# note with petsc we always need to provide a vector of correct cardinality.

stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")


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
    V_fn=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5


# +
# Define T boundary conditions via a sympy function

import sympy

init_t = 0.01 * sympy.sin(5.0 * x) * sympy.sin(np.pi * y) + (1.0 - y)

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords, meshbox.N).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# -


buoyancy_force = 1.0e6 * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# check the stokes solve is set up and that it converges
stokes.solve(zero_init_guess=True)

# Check the diffusion part of the solve converges
adv_diff.solve(timestep=0.1 * stokes.estimate_dt(), zero_init_guess=True)

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)/10
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    points = vis.meshVariable_to_pv_cloud(t_soln)
    point_cloud = pv.PolyData(points)
    point_cloud.point_data["Tp"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)

    # points = np.zeros((t_soln.coords.shape[0], 3))
    # points[:, 0] = t_soln.coords[:, 0]
    # points[:, 1] = t_soln.coords[:, 1]

    # point_cloud = pv.PolyData(points)

    # with meshbox.access():
    #     point_cloud.point_data["Tp"] = t_soln.data.copy()

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

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_mesh(pvmesh,'Gray', 'wireframe')

    # pl.add_mesh(
    #     pvmesh, cmap="coolwarm", edge_color="Black",
    #     show_edges=True, scalars="T", use_transparency=False, opacity=0.5,
    # )

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=10,
        opacity=0.33,
    )

    pl.add_mesh(pvstream, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
    pvmesh.clear_data()
    pvmesh.clear_point_data()


# -


pvmesh.clear_point_data()


def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)/333
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

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

        points = vis.meshVariable_to_pv_cloud(t_soln)
        points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
        point_cloud = pv.PolyData(points)

        pl = pv.Plotter(window_size=(1000, 750))

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

expt_name = "output/Ra1e6"

for step in range(0, 50):
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

# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh =  vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, stokes.u.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=7.5, opacity=0.25)

    pl.add_mesh(pvmesh, cmap="coolwarm", scalars="T", opacity=0.75)

    pl.show(cpos="xy")


