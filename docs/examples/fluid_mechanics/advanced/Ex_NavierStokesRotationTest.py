# %% [markdown]
"""
# 🎓 NavierStokesRotationTest

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Navier Stokes test: boundary driven ring with step change in boundary conditions
#
# This should develop a boundary layer with sqrt(t) growth rate

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()



# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.systems import NavierStokesSLCN
from underworld3 import function


import numpy as np
import sympy

# +
# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_resolution=0.1 etc

resolution = uw.options.getInt("model_resolution", default=10)
refinement = uw.options.getInt("model_refinement", default=0)
maxsteps = uw.options.getInt("max_steps", default=25)
restart_step = uw.options.getInt("restart_step", default=-1)
rho = uw.options.getReal("rho", default=1000)

outdir="output"

uw.pprint(f"restart: {restart_step}")
    print(f"resolution: {resolution}")
# -

meshball = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.0, cellSize=1/resolution, qdegree=3
)

meshball.view()

# +
# Define some functions on the mesh

import sympy

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# Rigid body rotation v_theta = constant, v_r = 0.0

theta_dot = 2.0 * np.pi  # i.e one revolution in time 1.0
v_x = -1.0 * r * theta_dot * sympy.sin(th) * y # to make a convergent / divergent bc
v_y = r * theta_dot * sympy.cos(th) * y
# -

v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
vorticity = uw.discretisation.MeshVariable(
    "\omega", meshball, 1, degree=1, continuous=True
)

navier_stokes = uw.systems.NavierStokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    rho=rho,
    order=2)

# +
navier_stokes.petsc_options["snes_monitor"] = None
navier_stokes.petsc_options["ksp_monitor"] = None

navier_stokes.petsc_options["snes_type"] = "newtonls"
navier_stokes.petsc_options["ksp_type"] = "fgmres"

navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# mg, multiplicative - very robust ... similar to gamg, additive

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -


nodal_vorticity_from_v = uw.systems.Projection(meshball, vorticity)
nodal_vorticity_from_v.uw_function = meshball.vector.curl(v_soln.sym)
nodal_vorticity_from_v.smoothing = 0.0


passive_swarm = uw.swarm.Swarm(mesh=meshball)
passive_swarm.populate(
    fill_param=3)

# +
# Constant visc

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
navier_stokes.constitutive_model.Parameters.viscosity = 1.0

# Constant visc

navier_stokes.penalty = 0.1
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions
# navier_stokes.add_dirichlet_bc((v_x, v_y), "Upper")
# navier_stokes.add_dirichlet_bc((0.0, 0.0), "Lower")

# Try this one:


# upper_mask = meshball.meshVariable_mask_from_label("UW_Boundaries", meshball.boundaries.Upper.value )
# lower_mask = meshball.meshVariable_mask_from_label("UW_Boundaries", meshball.boundaries.Lower.value )

# vbc_xn = 1000 * ((v_soln.sym[0] - v_x) * upper_mask.sym[0] + v_soln.sym[0] * lower_mask.sym[0])
# vbc_yn = 1000 * ((v_soln.sym[1] - v_y) * upper_mask.sym[0] + v_soln.sym[1] * lower_mask.sym[0])

# vbc_x = v_x * upper_mask.sym[0] + 0 * lower_mask.sym[0]
# vbc_y = v_y * upper_mask.sym[0] + 0 * lower_mask.sym[0]

# navier_stokes.add_natural_bc( (vbc_xn, vbc_yn), "All_Boundaries")

# navier_stokes.add_natural_bc( (1000 * (v_soln.sym[0] - v_x), 1000* ( v_soln.sym[1] - v_y)), "Upper")
# navier_stokes.add_natural_bc( (1000 * v_x, 1000*v_y), "Lower")


# navier_stokes.add_natural_bc( (vbc_xn, vbc_yn), "Upper")
# navier_stokes.add_natural_bc( (vbc_xn, vbc_yn), "Lower")

navier_stokes.add_dirichlet_bc((v_x, v_y), "Upper")
# navier_stokes.add_dirichlet_bc((0.0, 0.0), "Lower")

expt_name = f"Cylinder_NS_rho_{navier_stokes.rho}_{resolution}"

# -


navier_stokes.DuDt.bdf(1) 


navier_stokes.rho * navier_stokes.DuDt.bdf(1) 



0/0

navier_stokes.delta_t_physical = 0.1
navier_stokes.solve(timestep=0.1, verbose=False, evalf=True, order=1)
# navier_stokes.rho = rho

navier_stokes.estimate_dt()

if restart_step > 0:
    uw.pprint(f"Reading step {restart_step}")

    passive_swarm = uw.swarm.Swarm(mesh=meshball)
    passive_swarm.read_timestep(
        expt_name, "passive_swarm", restart_step, outputPath=outdir
    )
    
    v_soln.read_timestep(expt_name, "U", restart_step, outputPath=outdir)
    p_soln.read_timestep(expt_name, "P", restart_step, outputPath=outdir)
    
    # Flux history variable might be a good idea
# +
nodal_vorticity_from_v.solve()

# check the mesh if in a notebook / serial
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)


    # point sources at cell centres
    points = np.zeros((meshball._centroids.shape[0], 3))
    points[:, 0] = meshball._centroids[:, 0]
    points[:, 1] = meshball._centroids[:, 1]
    centroid_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        centroid_cloud,
        vectors="V",
        integration_direction="both",
        surface_streamlines=True,
        max_time=0.25)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, cmap="RdBu", scalars="Omega", opacity=0.5, show_edges=True)
    pl.add_mesh(pvstream, opacity=0.33)
    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=2.0e-2,
        opacity=0.75)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=3,
        opacity=0.5)

    pl.camera.SetPosition(0.75, 0.2, 1.5)
    pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    # pl.remove_scalar_bar("Omega")
    pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")

    pl.show(jupyter_backend="client")


def plot_V_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
        pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
            velocity_points, v_soln.sym
        )

        passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

        # point sources at cell centres
        points = np.zeros((meshball._centroids.shape[0], 3))
        points[:, 0] = meshball._centroids[:, 0]
        points[:, 1] = meshball._centroids[:, 1]
        centroid_cloud = pv.PolyData(points)

        pvstream = pvmesh.streamlines_from_source(
            centroid_cloud,
            vectors="V",
            integration_direction="both",
            surface_streamlines=True,
            max_time=0.25)

        pl = pv.Plotter()

        pl.add_arrows(
            velocity_points.points,
            velocity_points.point_data["V"],
            mag=0.01,
            opacity=0.75)
        
        pl.add_points(
            passive_swarm_points,
            color="Black",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.5)

        # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="Omega",
            use_transparency=False,
            opacity=0.5)

        pl.add_mesh(
            pvmesh,
            cmap="RdBu",
            scalars="Omega",
            opacity=0.1,  # clim=[0.0, 20.0]
        )

        pl.add_mesh(pvstream, opacity=0.33)

        scale_bar_items = list(pl.scalar_bars.keys())

        for scalar in scale_bar_items:
            pl.remove_scalar_bar(scalar)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 2560),
            return_img=False)

        # pl.show()


# -
if restart_step > 0:
    ts = restart_step
else:
    ts = 0


# +
# Time evolution model / update in time
navier_stokes.delta_t_physical = 0.1

delta_t = 0.1 #  5.0 * navier_stokes.estimate_dt()

for step in range(0, maxsteps+1):  # 250
    
    # if step%10 == 0:
    #     delta_t = 5.0 * navier_stokes.estimate_dt()

    navier_stokes.solve(timestep=delta_t, zero_init_guess=False, evalf=True)    
    passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    nodal_vorticity_from_v.solve()

    uw.pprint("Timestep {}, dt {}".format(ts, delta_t), flush=True)

    if ts % 5 == 0:
        plot_V_mesh(filename=f"{outdir}/{expt_name}_step_{ts}")
        
        meshball.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, vorticity],
            outputPath=outdir,
            index=ts)

        passive_swarm.write_timestep(
            expt_name,
            "passive_swarm",
            swarmVars=None,
            outputPath=outdir,
            index=ts,
            force_sequential=True)

    ts += 1
# -



# # ! open .

navier_stokes._u_f0


