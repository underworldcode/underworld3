# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Navier-Stokes Rotation Test

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Boundary-driven rotating flow in an annular domain with step change in
boundary conditions. Tests the Navier-Stokes solver with inertial effects.
The boundary layer develops with sqrt(t) growth rate.

## Key Concepts

- **Navier-Stokes equations**: Full momentum equation with inertia
- **Rotating flow**: Boundary-driven rigid body rotation
- **Boundary layer development**: Transient spin-up dynamics
- **Passive tracers**: Swarm advection for flow visualization
- **Restart capability**: Continue from previous checkpoint

## Mathematical Formulation

Navier-Stokes equations:
$$\\rho \\left( \\frac{\\partial \\mathbf{u}}{\\partial t} + \\mathbf{u} \\cdot \\nabla \\mathbf{u} \\right) = -\\nabla p + \\mu \\nabla^2 \\mathbf{u}$$

Rotating boundary condition:
$$v_\\theta = \\dot{\\theta} r$$

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_refinement`: Mesh refinement level
- `uw_rho`: Fluid density (controls Reynolds number)
- `uw_max_steps`: Maximum time steps
- `uw_restart_step`: Restart from checkpoint (-1 = fresh start)
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.systems import NavierStokesSLCN
from underworld3 import function

import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_NavierStokesRotationTest.py -uw_resolution 20
python Ex_NavierStokesRotationTest.py -uw_rho 100
python Ex_NavierStokesRotationTest.py -uw_max_steps 50
```
"""

# %%
params = uw.Params(
    uw_resolution = 10,           # Mesh resolution
    uw_refinement = 0,            # Mesh refinement levels
    uw_max_steps = 25,            # Maximum time steps
    uw_restart_step = -1,         # Restart from step (-1 = fresh start)
    uw_rho = 1000,                # Fluid density
    uw_viscosity = 1.0,           # Fluid viscosity
    uw_dt = 0.1,                  # Time step
)

outdir = "output"

uw.pprint(f"restart: {params.uw_restart_step}")
uw.pprint(f"resolution: {params.uw_resolution}")

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshball = uw.meshing.Annulus(
    radiusOuter=1.0,
    radiusInner=0.0,
    cellSize=1 / int(params.uw_resolution),
    qdegree=3,
)

meshball.view()

# %% [markdown]
"""
## Coordinate System and Boundary Conditions
"""

# %%
radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
unit_rvec = meshball.rvec / (1.0e-10 + radius_fn)

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# Rigid body rotation: v_theta = constant, v_r = 0.0
theta_dot = 2.0 * np.pi  # One revolution in time 1.0
v_x = -1.0 * r * theta_dot * sympy.sin(th) * y  # Convergent/divergent BC
v_y = r * theta_dot * sympy.cos(th) * y

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
vorticity = uw.discretisation.MeshVariable(
    r"\omega", meshball, 1, degree=1, continuous=True
)

# %% [markdown]
"""
## Navier-Stokes Solver
"""

# %%
navier_stokes = uw.systems.NavierStokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    rho=params.uw_rho,
    order=2,
)

# %% [markdown]
"""
## Solver Configuration
"""

# %%
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

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
"""
## Vorticity Projection
"""

# %%
nodal_vorticity_from_v = uw.systems.Projection(meshball, vorticity)
nodal_vorticity_from_v.uw_function = meshball.vector.curl(v_soln.sym)
nodal_vorticity_from_v.smoothing = 0.0

# %% [markdown]
"""
## Passive Swarm for Visualization
"""

# %%
passive_swarm = uw.swarm.Swarm(mesh=meshball)
passive_swarm.populate(fill_param=3)

# %% [markdown]
"""
## Constitutive Model and Boundary Conditions
"""

# %%
navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
navier_stokes.constitutive_model.Parameters.viscosity = params.uw_viscosity

navier_stokes.penalty = 0.1
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Rotating boundary condition on outer boundary
navier_stokes.add_dirichlet_bc((v_x, v_y), "Upper")

expt_name = f"Cylinder_NS_rho_{navier_stokes.rho}_{int(params.uw_resolution)}"

# %% [markdown]
"""
## Initial Solve
"""

# %%
navier_stokes.delta_t_physical = params.uw_dt
navier_stokes.solve(timestep=params.uw_dt, verbose=False, evalf=True, order=1)

uw.pprint(f"Estimated dt: {navier_stokes.estimate_dt()}")

# %% [markdown]
"""
## Restart from Checkpoint
"""

# %%
if params.uw_restart_step > 0:
    uw.pprint(f"Reading step {params.uw_restart_step}")

    passive_swarm = uw.swarm.Swarm(mesh=meshball)
    passive_swarm.read_timestep(
        expt_name, "passive_swarm", int(params.uw_restart_step), outputPath=outdir
    )

    v_soln.read_timestep(expt_name, "U", int(params.uw_restart_step), outputPath=outdir)
    p_soln.read_timestep(expt_name, "P", int(params.uw_restart_step), outputPath=outdir)

# %% [markdown]
"""
## Visualization of Initial State
"""

# %%
nodal_vorticity_from_v.solve()

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

    # Point sources at cell centres
    points = np.zeros((meshball._centroids.shape[0], 3))
    points[:, 0] = meshball._centroids[:, 0]
    points[:, 1] = meshball._centroids[:, 1]
    centroid_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        centroid_cloud,
        vectors="V",
        integration_direction="both",
        surface_streamlines=True,
        max_time=0.25,
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, cmap="RdBu", scalars="Omega", opacity=0.5, show_edges=True)
    pl.add_mesh(pvstream, opacity=0.33)
    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=2.0e-2,
        opacity=0.75,
    )

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=3,
        opacity=0.5,
    )

    pl.camera.SetPosition(0.75, 0.2, 1.5)
    pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")

    pl.show()


# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_V_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
        pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

        passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

        # Point sources at cell centres
        points = np.zeros((meshball._centroids.shape[0], 3))
        points[:, 0] = meshball._centroids[:, 0]
        points[:, 1] = meshball._centroids[:, 1]
        centroid_cloud = pv.PolyData(points)

        pvstream = pvmesh.streamlines_from_source(
            centroid_cloud,
            vectors="V",
            integration_direction="both",
            surface_streamlines=True,
            max_time=0.25,
        )

        pl = pv.Plotter()

        pl.add_arrows(
            velocity_points.points,
            velocity_points.point_data["V"],
            mag=0.01,
            opacity=0.75,
        )

        pl.add_points(
            passive_swarm_points,
            color="Black",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.5,
        )

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="Omega",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_mesh(
            pvmesh,
            cmap="RdBu",
            scalars="Omega",
            opacity=0.1,
        )

        pl.add_mesh(pvstream, opacity=0.33)

        scale_bar_items = list(pl.scalar_bars.keys())
        for scalar in scale_bar_items:
            pl.remove_scalar_bar(scalar)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 2560),
            return_img=False,
        )

        pv.close_all()


# %% [markdown]
"""
## Time Evolution
"""

# %%
if params.uw_restart_step > 0:
    ts = int(params.uw_restart_step)
else:
    ts = 0

navier_stokes.delta_t_physical = params.uw_dt
delta_t = params.uw_dt

for step in range(0, int(params.uw_max_steps) + 1):
    navier_stokes.solve(timestep=delta_t, zero_init_guess=False, evalf=True)
    passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    nodal_vorticity_from_v.solve()

    uw.pprint(f"Timestep {ts}, dt {delta_t:.4e}", flush=True)

    if ts % 5 == 0:
        plot_V_mesh(filename=f"{outdir}/{expt_name}_step_{ts}")

        meshball.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, vorticity],
            outputPath=outdir,
            index=ts,
        )

        passive_swarm.write_timestep(
            expt_name,
            "passive_swarm",
            swarmVars=None,
            outputPath=outdir,
            index=ts,
            force_sequential=True,
        )

    ts += 1

# %%
print(f"Navier-Stokes rotation test complete: {ts} steps")
