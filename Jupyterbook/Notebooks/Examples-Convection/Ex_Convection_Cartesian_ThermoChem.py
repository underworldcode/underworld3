# # Thermochemical convection
#
# We have a thermal convection (advection-diffusion) problem and a material-swarm mediated density variation
#
#

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

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()
# -

meshbox = uw.meshing.UnstructuredSimplexBox(
                                            minCoords=(0.0, 0.0),
                                            maxCoords=(1.0, 1.0),
                                            cellSize=1.0 / 24.0,
                                            regular=False,
                                            )
meshbox.dm.view()

# check the mesh if in a notebook / serial
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)


swarm = uw.swarm.Swarm(mesh=meshbox)
Mat = uw.swarm.SwarmVariable("Material", swarm, 1, proxy_degree=3)
X0 = uw.swarm.SwarmVariable("X0", swarm, meshbox.dim, _proxy=False)
swarm.populate(fill_param=5)


# +
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
    verbose=False,
)

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Constant visc
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))

# -


mMat = uw.discretisation.MeshVariable("mMat", meshbox, 1, degree=2)
projector = uw.systems.solvers.SNES_Projection(meshbox, mMat)
projector.smoothing = 1.0e-3

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

# Some useful coordinate stuff

x = meshbox.N.x
y = meshbox.N.y


# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0
r_i = 0.5
r_o = 1.0

adv_diff = uw.systems.AdvDiffusion(
                                    meshbox,
                                    u_Field=t_soln,
                                    V_fn=v_soln,
                                    solver_name="adv_diff",
                                    order=3,
                                    verbose=False,
                                )

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")


# +
# Define T boundary / initial conditions via a sympy function

import sympy

init_t = 0.01 * sympy.sin(5.0 * x) * sympy.sin(np.pi * y) + (1.0 - y)

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# -


with swarm.access(Mat):
    Mat.data[:, 0] = 0.5 + 0.5 * np.tanh(100.0 * (swarm.data[:, 1] - 0.25))

projector.uw_function = Mat.sym
projector.solve()

# +
expt_name = "output/Ra1e6_Rc5e5"

# # +ve Rc means heavy chemical component,
# -ve Rc means light chemical component

# Here we are using the projected mMat field but we
# can switch this out for the particle field
# to show the equivalence

buoyancy_force = 1.0e6 * t_soln.fn + 5.0e5 * mMat.fn
stokes.bodyforce = meshbox.N.j * buoyancy_force

# check the stokes solve is set up and that it converges
stokes.solve()
# -

# Check the diffusion part of the solve converges
adv_diff.solve(timestep=0.01 * stokes.estimate_dt())


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.5,
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1.0e-4, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
        pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, mMat.sym)
        
        tpoints = vis.meshVariable_to_pv_cloud(t_soln)
        tpoints.point_data["T"] = vis.scalar_fn_to_pv_points(tpoints, t_soln.sym)
        tpoint_cloud = pv.PolyData(tpoints)

        spoints = vis.swarm_to_pv_cloud(swarm)
        swarm_point_cloud = pv.PolyData(spoints)
        with swarm.access():
            swarm_point_cloud.point_data["M"] = Mat.data.copy()

        velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)


        pl = pv.Plotter(window_size=(750, 750))

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00001, opacity=0.75)

        # pl.add_points(point_cloud, cmap="gray",
        #               render_points_as_spheres=False,
        #               point_size=10, opacity=0.5
        #             )

        pl.add_points(
            swarm_point_cloud,
            cmap="RdYlBu",
            render_points_as_spheres=True,
            point_size=7.5,
            opacity=1.0,
        )

        pl.add_mesh(
            pvmesh,
            cmap="gray",
            edge_color="Black",
            show_edges=True,
            scalars="M",
            use_transparency=False,
            opacity=0.5,
        )

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("mag")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )
        # pl.show()


# +
# Convection model / update in time

for step in range(0, 50):
    stokes.solve(zero_init_guess=False)
    delta_t = 3.0e-5  # 5.0*stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # update swarm locations using v_soln

    swarm.advection(v_soln.fn, delta_t, order=2, corrector=True)
    # projector.solve(zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
    #         print(tstats)

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    # savefile = "{}_ts_{}.h5".format(expt_name,step)
    # meshbox.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshbox.generate_xdmf(savefile)
# -


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, mMat.sym)

    tpoints = vis.meshVariable_to_pv_cloud(t_soln)
    tpoints.point_data["T"] = vis.scalar_fn_to_pv_points(tpoints, t_soln.sym)
    tpoints.point_data["M"] = vis.scalar_fn_to_pv_points(tpoints, mMat.sym)
    tpoint_cloud = pv.PolyData(tpoints)

    spoints = vis.swarm_to_pv_cloud(swarm)
    swarm_point_cloud = pv.PolyData(spoints)
    with swarm.access():
        swarm_point_cloud.point_data["M"] = Mat.data.copy()

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)


    pl = pv.Plotter(window_size=(750, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.75e-5, opacity=0.75)


    pl.add_points(
        swarm_point_cloud,
        cmap="RdYlBu",
        render_points_as_spheres=True,
        point_size=3.0,
        opacity=1.0,
    )

    pl.add_mesh(
        pvmesh,
        cmap="gray",
        edge_color="Black",
        show_edges=True,
        scalars="M",
        use_transparency=False,
        opacity=0.5,
    )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    pl.show(cpos="xy")




