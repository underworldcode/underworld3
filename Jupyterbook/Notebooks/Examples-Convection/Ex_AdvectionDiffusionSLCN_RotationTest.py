# # Field (SemiLagrange) Advection solver test
#
# Shear flow driven by a pre-defined, rigid body rotation in a disc or by the boundary conditions
#

# +
import petsc4py
from petsc4py import PETSc

import os
import nest_asyncio
nest_asyncio.apply()

os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import VarType
from underworld3 import timing

import numpy as np
import sympy

options = PETSc.Options()

# options.getAll()
# -
meshball = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=1, qdegree=3
)


# +
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3, varsymbol=r"T_{0}")

# Create a temperature structure / buoyancy force

import sympy

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)

# Some useful coordinate stuff

x, y = meshball.X
r, th = meshball.CoordinateSystem.xR

# Rigid body rotation v_theta = constant, v_r = 0.0

theta_dot = 2.0 * np.pi  # i.e one revolution in time 1.0
v_x = -r * theta_dot * sympy.sin(th)
v_y = r * theta_dot * sympy.cos(th)

with meshball.access(v_soln):
    v_soln.data[:, 0] = uw.function.evaluate(v_x, v_soln.coords)
    v_soln.data[:, 1] = uw.function.evaluate(v_y, v_soln.coords)

# +
# swarm  = uw.swarm.Swarm(mesh=meshball)
# T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1)
# X1 = uw.swarm.SwarmVariable("Xminus1", swarm, 2)
# swarm.populate(fill_param=3)


# +
# Create adv_diff object

# Set some things
k = 0.01
h = 0.1
t_i = 2.0
t_o = 1.0
r_i = 0.5
r_o = 1.0
delta_t = 1.0
# -


adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
    order=2,
)


adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel(adv_diff.Unknowns)
adv_diff.constitutive_model.Parameters.diffusivity = k


# +
# Define T boundary conditions via a sympy function

import sympy

abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = sympy.exp(-30.0 * (meshball.N.x**2 + (meshball.N.y - 0.75) ** 2))

adv_diff.add_dirichlet_bc(0.0, "Lower")
adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]


# +
# # We can over-ride the swarm-particle update routine since we can integrate
# # the velocity field by hand.

# with adv_diff._nswarm.access():
#     coords0 = adv_diff._nswarm.data.copy()

# delta_t = 0.000

# n_x = uw.function.evaluate(r * sympy.cos(th - delta_t * theta_dot), coords0)
# n_y = uw.function.evaluate(r * sympy.sin(th - delta_t * theta_dot), coords0)

# coords = np.empty_like(coords0)
# coords[:, 0] = n_x
# coords[:, 1] = n_y

# # delta_t will be baked in when this is defined ... so re-define it
# adv_diff.solve(timestep=delta_t) # , coords=coords)
# -
def plot_T_mesh(filename):
    if uw.mpi.size == 1:
    
        import pyvista as pv
        import underworld3.visualisation as vis
        
        pvmesh = vis.mesh_to_pv_mesh(meshball)
        points = vis.meshVariable_to_pv_cloud(t_soln)
        points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    
        point_cloud = pv.PolyData(points)
    
        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
        
        pl = pv.Plotter(window_size=(1000, 750))

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.0001, opacity=0.75)

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
    
    pl.show()


t_soln2 = uw.discretisation.MeshVariable(
    "U2", meshball, vtype=uw.VarType.SCALAR, degree=2
)

adv_diff.estimate_dt()


adv_diff.DuDt._psi_star_projection_solver._constitutive_model

# +
timing.reset()
timing.start()

delta_t = 0.001

adv_diff.solve(timestep=delta_t, verbose=False, _force_setup=False)

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)

    point_cloud = pv.PolyData(points)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    
    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01, opacity=0.75)
    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7,
        opacity=0.66,
    )
    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    # pl.remove_scalar_bar("T")
    pl.remove_scalar_bar("mag")

    pl.show()

# +
# Advection/diffusion model / update in time

expt_name = "rotation_test_slcn"

delta_t = 0.05

plot_T_mesh(filename="{}_step_{}".format(expt_name, 0))

for step in range(0, 10):
    # delta_t will be baked in when this is defined ... so re-define it
    adv_diff.solve(timestep=delta_t, verbose=False)

    # stats then loop

    # tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
        # print(tstats)

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    # savefile = "output_conv/convection_cylinder_{}_iter.h5".format(step)
    # meshball.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshball.generate_xdmf(savefile)
# -


t_soln.stats()

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    
    pvmesh = vis.mesh_to_pv_mesh(meshball)
    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    points.point_data["dT"] = vis.scalar_fn_to_pv_points(points, t_soln.sym) - vis.scalar_fn_to_pv_points(points, t_0.sym)

    point_cloud = pv.PolyData(points)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    
    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.0001, opacity=0.75)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        scalars="T",  # clim=[-0.2,0.2],
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    # pl.remove_scalar_bar("T")
    pl.remove_scalar_bar("mag")

    pl.show()

# +
# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)
# -
uw.timing.print_table()


#
