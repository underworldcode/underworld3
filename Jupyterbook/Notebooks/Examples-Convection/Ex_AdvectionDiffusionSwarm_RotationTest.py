# # Swarm Advection solver test - shear flow driven by a pre-defined, rigid body rotation in a disc
#
# This example uses the Swarm advection approach rather than SLCN

# +
import petsc4py
from petsc4py import PETSc

import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

# import os
# os.environ["SYMPY_USE_CACHE"]="no"

# options.getAll()
# +
import meshio

meshball = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=1, qdegree=3
)
x, y = meshball.X
# -


v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_soln_dt = uw.discretisation.MeshVariable("Tdt", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)


DTdt = uw.systems.Lagrangian_DDt(
        meshball,
        psi_fn = t_soln.sym,
        V_fn = v_soln.sym,
        vtype = uw.VarType.SCALAR,
        degree = 1,
        order = 1,
        continuous=True,
        varsymbol=r'T_s',
        fill_param=3,
)


# check that the swarm variable works  as a continuous field as well
DTdt.psi_star[0].sym.jacobian(meshball.X)

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


# +
adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn = v_soln,
    DuDt = DTdt,
    solver_name="adv_diff_swarms",  # not needed if coords is provided
    order=1,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k


# +
# Create a density structure / bu()oyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

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
# Validation - small timestep

# delta_t = 0.01
# adv_diff.solve(timestep=delta_t)
# -


def plot_T_mesh(filename):

    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import underworld3.visualisation as vis
        
        pvmesh = vis.mesh_to_pv_mesh(meshball)
        swarm_points = vis.swarm_to_pv_cloud(adv_diff.DuDt.swarm)
        tsoln_points = vis.meshVariable_to_pv_cloud(t_soln)
            
        swarm_points.point_data["T"] = vis.scalar_fn_to_pv_points(swarm_points,adv_diff.DuDt.psi_fn)
        
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh,t_soln.sym)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh,v_soln.sym)

        pl = pv.Plotter(window_size=(1000, 750))

        pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.0001, opacity=0.75)

        # pl.add_points(
        #     swarm_points,
        #     cmap="coolwarm",
        #     render_points_as_spheres=False,
        #     point_size=20,
        #     opacity=0.66,
        # )
    
        pl.add_mesh(pvmesh, cmap="coolwarm", opacity=0.75)
    
        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )

    # pl.show()


# +
with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]




# +
adv_diff.DuDt.update(dt=0.05)

# # Update the swarm locations
# swarm.advection(
#     v_soln.sym,
#     delta_t=0.05,
#     corrector=False,
#     restore_points_to_domain_func=meshball.return_coords_to_bounds,
# )  

# +
# Advection/diffusion model / update in time

delta_t = 0.05
expt_name = "output/rotation_test_k_001"

plot_T_mesh(filename="{}_step_{}".format(expt_name, 0))

for step in range(0, 10):

    adv_diff.solve(timestep=delta_t, verbose=False)

    tstats = t_soln.stats()
    print("psi*", adv_diff.DuDt.psi_star[0]._meshVar.stats())

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
        print(tstats)

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    # savefile = "output_conv/convection_cylinder_{}_iter.h5".format(step)
    # meshball.save(savefile)
    # v_soln.save(savefile)
    # t_soln.save(savefile)
    # meshball.generate_xdmf(savefile)


# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    swarm_points = vis.swarm_to_pv_cloud(adv_diff.DuDt.swarm)
    tsoln_points = vis.meshVariable_to_pv_cloud(t_soln)
        
    swarm_points.point_data["Ts"] = vis.scalar_fn_to_pv_points(swarm_points, adv_diff.DuDt.psi_star[0].sym[0] )

    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh,t_soln.sym)
    pvmesh.point_data["Ts"] = vis.scalar_fn_to_pv_points(pvmesh,adv_diff.DuDt.psi_star[0].sym[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh,v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.02, opacity=0.75)

    pl.add_points(
        swarm_points,
        cmap="coolwarm",
        render_points_as_spheres=False,
        scalars="Ts",
        point_size=10,
        opacity=0.66,
    )

    # pl.add_mesh(pvmesh, cmap="coolwarm", opacity=0.75, scalars="T")

    # pl.remove_scalar_bar("T")
    # pl.remove_scalar_bar("mag")

    pl.show()

# +
# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)
# -

DTdt.psi_fn


