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
import sympy

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()

# +
Free_Slip = True
Rayleigh = 1.0e5
H_int = 1.0
k = 1.0
res = 0.05
resI = 0.15
r_o = 1.0
r_i = 0.0

expt_name = "Disc_Ra1e5_H1_ii"
# -

meshball = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i,
                              cellSize=res, cellSizeInner=resI, centre=False, qdegree=3, )

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
kappa = uw.discretisation.MeshVariable("kappa", meshball, 1, degree=3, varsymbol=r"\kappa")



# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                solver_name="stokes", verbose=True)

# Constant viscosity
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

stokes.tolerance = 1.0e-3

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# Velocity boundary conditions

if Free_Slip:

    stokes.add_natural_bc(
        10000 * unit_rvec.dot(v_soln.sym) * unit_rvec.T, "Upper"
    )

else:
    surface_fn = 0.0
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))


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
Tslope_max = 15

k_lim = (Tslope/Tslope_max) 
k_eff = k * sympy.Max(1, k_lim)

adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.f = H_int
adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 1




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
tstats = t_soln.stats()
print(tstats)

Tgrad_stats = kappa.stats()

print(Tgrad_stats)

# -

with meshball.access(r_mesh):
    r_mesh.data[:, 0] = uw.function.evalf(r, meshball.data)

# +

stokes.bodyforce = unit_rvec * gravity_fn * Rayleigh * t_soln.fn
stokes.solve()
# -

# Check the diffusion part of the solve converges
adv_diff.solve(timestep=0.1*stokes.estimate_dt())
adv_diff.constitutive_model.Parameters.diffusivity = k_eff
adv_diff.solve(timestep=0.1*stokes.estimate_dt(), zero_init_guess=False)

calculate_diffusivity.solve()

# +
# check the mesh if in a notebook / serial
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", 
        show_edges=True, scalars="T", 
        use_transparency=False, opacity=0.5,
        clim=[0,1],
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=50 / Rayleigh)
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

ts = 0

# +
# Convection model / update in time

for step in range(0, 1000): #1000

    stokes.solve()

    calculate_diffusivity.solve()
    
    delta_t = adv_diff.estimate_dt(v_factor=2.0, diffusivity=kappa.sym[0])
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()
    # tstats_star = adv_diff.DuDt.psi_star[0].stats()

    Tgrad_stats = kappa.stats()

    with meshball.access():    
        print(f"Flux: min = {adv_diff.Unknowns.DFDt.psi_star[0].data.min()}", 
              f" max = {adv_diff.Unknowns.DFDt.psi_star[0].data.max()}", )

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(ts, delta_t))
        print(tstats)
        print("-----")
        print(Tgrad_stats)
        print("=====\n")

        # print(tstats_star)

    if ts % 10 == 0:
        plot_T_mesh(filename="output/{}_step_{}".format(expt_name, ts))


#    savefile = "{}_ts_{}.h5".format(expt_name,step)
#    meshball.save(savefile)
#     v_soln.save(savefile)
#     t_soln.save(savefile)
#     meshball.generate_xdmf(savefile)

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

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.001, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(point_cloud, cmap="coolwarm", 
                  render_points_as_spheres=True, 
                  point_size=7.5, opacity=0.25)

    pl.add_mesh(pvmesh, cmap="coolwarm", scalars="T", opacity=0.75)

    pl.show(cpos="xy")


