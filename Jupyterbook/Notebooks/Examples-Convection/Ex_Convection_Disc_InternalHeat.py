# # Convection in a disc with internal heating and rigid or free boundaries
#
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import os

import numpy as np
import sympy

import petsc4py
from petsc4py import PETSc

# +
## Command line parameters use -uw_resolution 0.1, for example

res = uw.options.getReal("resolution", default=0.1)
Free_Slip = uw.options.getBool("free_slip", default=True)
restart_step = uw.options.getInt("restart_step", default=0)
max_steps = uw.options.getInt("max_steps", default=1)
delta_eta = uw.options.getReal("delta_eta", default=1000.0)

viz = True

# -

uw.options.view()

# +
Rayleigh = 1.0e7
H_int = 1.0
k = 1.0
resI = res * 3
r_o = 1.0
r_i = 0.0


# For now, assume restart is from same location !
expt_name = f"Disc_Ra1e7_H1_deleta_{delta_eta}"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True  )

# -

meshball = uw.meshing.AnnulusWithSpokes(radiusOuter=r_o, radiusInner=r_i,
                                            cellSizeOuter=res,
                                            cellSizeInner=resI,
                                           qdegree=3, )


meshball.dm.view()

# +

radius_fn = sympy.sqrt(meshball.X.dot(meshball.X)) # normalise by r_o if required
unit_rvec = meshball.X / radius_fn
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)  # cf radius_fn which is 0->1
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# -


# check the mesh if in a notebook / serial
if viz and uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pl.show()



v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)
r_mesh = uw.discretisation.MeshVariable("r", meshball, 1, degree=1)
kappa = uw.discretisation.MeshVariable("kappa", meshball, 1, degree=3, varsymbol=r"\kappa")

# +
## F-K viscosity function

C = sympy.log(delta_eta)
viscosity_fn = delta_eta * sympy.exp(-C * 0)


# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                solver_name="stokes", 
                verbose=False)

# Constant viscosity
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn

# Set solve options here (or remove default values
stokes.tolerance = 1.0e-6
# stokes.petsc_options.delValue("ksp_monitor")

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)

# Velocity boundary conditions

if Free_Slip:    
    GammaN = meshball.Gamma  # boundary_normals["Upper"].value
    # bc = sympy.Piecewise((1.0, r > 0.99 * r_o), (0.0, True))
    stokes.add_natural_bc(
        1.0e6 * GammaN.dot(v_soln.sym) * GammaN.T, "Upper"
    )

else:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper")

# -


meshball.Gamma

# +
# Create adv_diff object

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
    verbose=False,
    order=2,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel

## Flux limiting diffusivity (stabilizing term)

Tgrad = meshball.vector.gradient(t_soln.sym)
Tslope = sympy.sqrt(Tgrad.dot(Tgrad))
Tslope_max = 25

k_lim = (Tslope/Tslope_max) 
k_eff = k * sympy.Max(1, k_lim)

adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.f = H_int



# +
## Projection to compute the diffusivity

calculate_diffusivity = uw.systems.Projection(meshball, u_Field=kappa)
calculate_diffusivity.uw_function = k_eff


# +
# Define T boundary conditions via a sympy function

import sympy

abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.25 + 0.25 * sympy.sin(7.0 * th) * sympy.sin(np.pi * (r - r_i) / (r_o - r_i)) + 0.0 * (r_o - r) / (r_o - r_i)

adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evalf(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
# +
# If restart, then pull T from there

if restart_step != 0:
    t_soln.read_timestep(expt_name, "T", restart_step, outputPath=output_dir, verbose=True)

# -

with meshball.access(r_mesh):
    r_mesh.data[:, 0] = uw.function.evalf(r, meshball.data)

stokes.bodyforce = unit_rvec * gravity_fn * Rayleigh * t_soln.fn
stokes.solve(verbose=False)

# +
# Check the diffusion part of the solve converges

dt = 0.00001
adv_diff.solve(timestep=dt)
adv_diff.constitutive_model.Parameters.diffusivity = k_eff
adv_diff.solve(timestep=dt, zero_init_guess=False)

# -

calculate_diffusivity.solve()

# +
# check the mesh if in a notebook / serial
if viz and uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym[0])
    pvmesh.point_data["K"] = vis.scalar_fn_to_pv_points(pvmesh, kappa.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", 
        show_edges=True, scalars="T", 
        use_transparency=False, opacity=1.0,
        # clim=[0,1],
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
    
def plot_T_mesh(filename):

    if viz and uw.mpi.size == 1:

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


        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=50 / Rayleigh)
        pl.add_mesh(pvmesh, cmap="coolwarm", 
                    show_edges=True,
                    scalars="T", opacity=0.75)

        pl.add_points(point_cloud, cmap="coolwarm", 
                      render_points_as_spheres=False, 
                      # clim=[0,1],
                      point_size=10, opacity=0.66)


        # pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False)
        # pl.show()
# -

ts = restart_step

# +
# Convection model / update in time

delta_t = 5.0e-5

for step in range(0, max_steps ): #

    stokes.solve(verbose=False, zero_init_guess=False)

    calculate_diffusivity.solve()

    if step%10 == 0:
        delta_t = adv_diff.estimate_dt(v_factor=2.0, diffusivity=kappa.sym[0])
        
    adv_diff.solve(timestep=delta_t, zero_init_guess=False )

    # stats, dt (all collective) print if rank 0, then loop
    tstats = t_soln.stats()
    Tgrad_stats = kappa.stats()
    dt_estimate =  adv_diff.estimate_dt(v_factor=2.0, diffusivity=kappa.sym[0])

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {} ({})".format(ts, delta_t, dt_estimate), flush=True)
        # print(tstats)
        # print("-----")
        # print(Tgrad_stats)
        # print("=====\n")

        # print(tstats_star)

    if ts % 10 == 0:
        plot_T_mesh(filename="output/{}_step_{}".format(expt_name, ts))

        meshball.write_timestep(
                expt_name,
                meshUpdates=True,
                meshVars=[p_soln, v_soln, t_soln],
                outputPath=output_dir,
                index=ts,
            )

    ts += 1

# -


if viz and uw.mpi.size == 1:

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

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(point_cloud, cmap="coolwarm", 
                  render_points_as_spheres=True, 
                  point_size=7.5, opacity=0.25)

    pl.add_mesh(pvmesh, cmap="coolwarm", scalars="T", opacity=0.75)

    pl.show(cpos="xy")




