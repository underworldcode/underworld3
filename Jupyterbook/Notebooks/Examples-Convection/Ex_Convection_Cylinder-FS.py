# # Stokes in an annulus with adv_diff to solve T and back-in-time sampling with particles
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
from underworld3 import function

import numpy as np

# +
import os

rayleigh=1.0e5

output_dir = os.path.join("output","Cylinder_FS_Ra1e5_p")
expt_name = "Cylinder_FS"

os.makedirs(output_dir, exist_ok=True  )


viz = True
# -

meshball = uw.meshing.Annulus(
    radiusInner=0.5, radiusOuter=1.0, cellSize=0.05, qdegree=3
)


# check the mesh if in a notebook / serial
if viz and uw.mpi.size == 1:
    
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
#### Create Stokes object

stokes = uw.systems.Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0

stokes.petsc_options.delValue("ksp_monitor")
stokes.petsc_options.delValue("snes_monitor")

# Constant visc
stokes.viscosity = 1.0

# Velocity boundary conditions
stokes.add_essential_bc((0.0, 0.0), "Lower", (0, 1))


# +
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(
    meshball.X.dot(meshball.X)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.X[0]
y = meshball.X[1]

r = meshball.CoordinateSystem.R[0]
th = meshball.CoordinateSystem.R[1]

# +
# Create adv_diff object

# Set some things
k = 1.0
h = 0.0
r_i = 0.5
r_o = 1.0

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1

adv_diff.tolerance=1.0e-4



# +
# Define T boundary conditions via a sympy function

import sympy

abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.01 * sympy.sin(5.0 * th) * sympy.sin(np.pi * (r - r_i) / (r_o - r_i)) + (
    r_o - r
) / (r_o - r_i)

adv_diff.add_dirichlet_bc(1.0, "Lower")
adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# +
buoyancy_force = rayleigh * t_soln.sym[0] / (0.5) ** 3
stokes.bodyforce = unit_rvec * buoyancy_force

stokes.tolerance = 1.0e-3
stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)

stokes.add_essential_bc([0.0, 0.0], "Lower")  # no slip on the base
stokes.add_natural_bc(
    10000 * unit_rvec.dot(v_soln.sym) * unit_rvec.T, "Upper"
)

stokes.solve(verbose=False, zero_init_guess=True, picard=1 )


# +
# Check the diffusion part of the solve converges
# adv_diff.petsc_options["ksp_monitor"] = None
adv_diff.petsc_options["snes_monitor"] = None

adv_diff.solve(verbose=False, timestep=0.5 * stokes.estimate_dt())
# -


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

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1000.0/rayleigh, opacity=0.75)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


def plot_T_mesh(filename):
    if viz and uw.mpi.size == 1:
        
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

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=10.0/rayleigh, opacity=0.75)

        pl.add_points(
            tpoint_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.66,
        )

        pl.add_mesh(pvmesh, scalars="T", cmap="coolwarm", show_edges=True, opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )
        # pl.show()


ts = 0

# +
# Convection model / update in time


for step in range(0, 51):

    stokes.solve(verbose=False, zero_init_guess=False, picard=0)

    delta_t = adv_diff.estimate_dt(v_factor=2.0)
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    stats = t_soln.stats()
    stats_star = adv_diff.DuDt.psi_star[0].stats()
    
    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(ts, delta_t))
        print(stats)
        print(stats_star)

    if step%5 == 0:
        plot_T_mesh(filename="{}_step_{}".format(os.path.join(output_dir, expt_name),ts))

    if step%10 == 0:

        meshball.write_timestep(
                expt_name,
                meshUpdates=True,
                meshVars=[p_soln, v_soln, t_soln],
                outputPath=output_dir,
                index=ts,
            )

    ts += 1



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


