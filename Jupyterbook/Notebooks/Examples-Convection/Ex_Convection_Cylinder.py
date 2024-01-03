# # Stokes in a disc with adv_diff to solve T and back-in-time sampling with particles
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
#
# The next step is to add particles at node points and sample back along the streamlines to find values of the T field at a previous time.
#
# (Note, we keep all the pieces from previous increments of this problem to ensure that we don't break something along the way)

# +
# to fix trame issue
# import nest_asyncio
# nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
# -

meshball = uw.meshing.Annulus(
    radiusInner=0.5, radiusOuter=1.0, cellSize=0.1, degree=1, qdegree=3
)


# check the mesh if in a notebook / serial
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )

    pl.show()

v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)


swarm = uw.swarm.Swarm(mesh=meshball)
T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1)
X1 = uw.swarm.SwarmVariable("Xminus1", swarm, 2)
swarm.populate(fill_param=3)






# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0


# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Constant visc
stokes.viscosity = 1.0

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(
    meshball.X.dot(meshball.X)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (1.0e-10 + radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.X[0]
y = meshball.X[1]

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0
r_i = 0.5
r_o = 1.0

adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1

adv_diff.theta = 0.5
# adv_diff.f = t_soln.fn / delta_t - t_star.fn / delta_t


# +
# Define T boundary conditions via a sympy function

import sympy

abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.01 * sympy.sin(15.0 * th) * sympy.sin(np.pi * (r - r_i) / (r_o - r_i)) + (
    r_o - r
) / (r_o - r_i)

adv_diff.add_dirichlet_bc(1.0, "Lower")
adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# +
buoyancy_force = 1.0e6 * t_soln.sym[0] / (0.5) ** 3
stokes.bodyforce = unit_rvec * buoyancy_force

# check the stokes solve converges
stokes.solve()

# +
# Check the diffusion part of the solve converges
adv_diff.petsc_options["ksp_monitor"] = None
adv_diff.petsc_options["monitor"] = None

adv_diff.solve(timestep=0.00001 * stokes.estimate_dt())


# +
# diff = uw.systems.Poisson(meshball, u_Field=t_soln, solver_name="diff_only")

# diff.constitutive_model = uw.constitutive_models.DiffusionModel(meshball.dim)
# diff.constitutive_model.material_properties = adv_diff.constitutive_model.Parameters(diffusivity=1)
# diff.solve()


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    # pvmesh.point_data["Ts"] = vis.scalar_fn_to_pv_points(pvmesh, adv_diff._u_star.sym)
    # pvmesh.point_data["dT"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym) - vis.scalar_fn_to_pv_points(pvmesh, adv_diff._u_star.sym)

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

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.0005)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# +
# pvmesh.point_data["Ts"].min()
# -

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 1

# check the mesh if in a notebook / serial
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym) - vis.scalar_fn_to_pv_points(pvmesh, t_0.sym)

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

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.025)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

        tpoints = vis.meshVariable_to_pv_cloud(t_soln)
        tpoints.point_data["T"] = vis.scalar_fn_to_pv_points(tpoints, t_soln.sym)
        tpoint_cloud = pv.PolyData(tpoints)

        velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

        pl = pv.Plotter(window_size=(750, 750))

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00002, opacity=0.75)

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.66,
        )

        pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )
        # pl.show()


# +
# Convection model / update in time

expt_name = "output/Cylinder_Ra1e6i"

for step in range(0, 50):
    stokes.solve()
    delta_t = 5.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
    #         print(tstats)

    #     plot_T_mesh(filename="{}_step_{}".format(expt_name,step))

    meshball.petsc_save_checkpoint(index=step, 
                                   meshVars=[v_soln, t_soln], 
                                   outputPath=expt_name)



# +
# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)
# -


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    point_cloud = pv.PolyData(points)

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00005, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7.5,
        opacity=0.75,
    )

    pl.add_mesh(pvmesh, scalars="T", cmap="coolwarm", opacity=1)

    pl.show(cpos="xy")


