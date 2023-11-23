# # Convection in a disc with internal heating and rigid or free boundaries
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

# +
Free_Slip = True
Rayleigh = 1.0e5
H_int = 1
res = 0.1 # 0.05
r_o = 1.0
r_i = 0.0

expt_name = "Disc_Ra1e5_H1"
# -

meshball = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i,
                              cellSize=res, centre=False, degree=1, )

# +
# ===

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y
# z = meshball.N.z

r = sympy.sqrt(x**2 + y**2)  # cf radius_fn which is 0->1
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)
# -


# check the mesh if in a notebook / serial
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pl.show()

v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)
r_mesh = uw.discretisation.MeshVariable("r", meshball, 1, degree=1)


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)


# +
# Create Stokes object
import sympy

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                order=2, solver_name="stokes", verbose=True)

# Constant viscosity
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0


# Inexact Jacobian may be OK.
stokes.petsc_options["snes_rtol"] = 1.0e-3
stokes.petsc_options["ksp_rtol"] = 1.0e-3

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Velocity boundary conditions

if Free_Slip:

    hw = 1000.0 / res
    surface_fn = sympy.exp(-(((r - r_o) / r_o) ** 2) * hw)

#     o_mask_fn = 0.5 - 0.5 * sympy.tanh(5000.0*(r-rm_o))
#     i_mask_fn = 0.5 - 0.5 * sympy.tanh(5000.0*(r-rm_i))
#     surface_fn = o_mask_fn - i_mask_fn

else:
    surface_fn = 0.0
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))


# +
# Advection / diffusion mesh restore function
# Which probably should be a feature of the mesh type ...


def points_in_disc(coords):
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2).reshape(-1, 1)
    outside = np.where(r > 1.0)
    coords[outside] *= 0.999 / r[outside]
    return coords


# +
# Create adv_diff object

# Set some things
k = 1.0
h = 10
r_i = 0.5
r_o = 1.0

adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
    order=3,
    restore_points_func=points_in_disc,
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k

# adv_diff.k = k
adv_diff.f = h
adv_diff.theta = 0.5
adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 1


# +
# Define T boundary conditions via a sympy function

import sympy

abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.01 * sympy.sin(15.0 * th) * sympy.sin(np.pi * (r - r_i) / (r_o - r_i)) + (r_o - r) / (r_o - r_i)

adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# -
with meshball.access(r_mesh):
    r_mesh.data[:, 0] = uw.function.evaluate(r, meshball.data)

# +
buoyancy_force = gravity_fn * Rayleigh * t_soln.fn
buoyancy_force -= Rayleigh * 100000.0 * v_soln.fn.dot(unit_rvec) * surface_fn

stokes.bodyforce = unit_rvec * buoyancy_force

# check the stokes solve converges
stokes.solve()

# +
# Check the diffusion part of the solve converges
# adv_diff.solve(timestep=0.01*stokes.estimate_dt())
# -
# check the mesh if in a notebook / serial
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["S"] = vis.scalar_fn_to_pv_points(pvmesh, surface_fn)

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S", use_transparency=False, opacity=0.5
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=10000 / Rayleigh)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
def plot_T_mesh(filename):

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


        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=10 / Rayleigh)

        pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.66)

        pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False)
        # pl.show()

# +
# Convection model / update in time

expt_name = "output/{}".format(expt_name)

for step in range(0, 2): #1000

    stokes.solve()

    delta_t = 5.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
        print(tstats)

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))


#    savefile = "{}_ts_{}.h5".format(expt_name,step)
#    meshball.save(savefile)
#     v_soln.save(savefile)
#     t_soln.save(savefile)
#     meshball.generate_xdmf(savefile)

# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)


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

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.001, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=True, point_size=7.5, opacity=0.25)

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")


