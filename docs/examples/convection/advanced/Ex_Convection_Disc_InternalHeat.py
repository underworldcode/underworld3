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
# Convection in a Disc with Internal Heating

**PHYSICS:** convection
**DIFFICULTY:** advanced

## Description

Thermal convection in a disc (2D annulus with zero inner radius) driven by
internal heating. Supports both free-slip and no-slip boundary conditions
on the outer boundary. Uses an annulus mesh with spokes for improved
element quality.

## Key Concepts

- **Internal heating**: Volumetric heat source driving convection
- **Disc geometry**: AnnulusWithSpokes mesh with r_inner = 0
- **Boundary conditions**: Free-slip via penalty or no-slip
- **Temperature-dependent viscosity**: Frank-Kamenetskii rheology
- **Flux-limited diffusivity**: Stabilization for steep gradients
- **Restart capability**: Continue from previous checkpoint

## Mathematical Formulation

Governing equations:
$$\\nabla \\cdot \\mathbf{u} = 0$$
$$-\\nabla p + \\nabla \\cdot (\\eta \\nabla \\mathbf{u}) + Ra T \\hat{r} = 0$$
$$\\frac{\\partial T}{\\partial t} + \\mathbf{u} \\cdot \\nabla T = \\kappa \\nabla^2 T + H$$

## Parameters

- `uw_resolution`: Mesh cell size
- `uw_free_slip`: Use free-slip (True) or no-slip (False) on outer boundary
- `uw_delta_eta`: Viscosity contrast
- `uw_max_steps`: Maximum time steps
- `uw_restart_step`: Restart from checkpoint (0 = fresh start)
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
from underworld3 import function
import os

import numpy as np
import sympy

import petsc4py
from petsc4py import PETSc

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Convection_Disc_InternalHeat.py -uw_resolution 0.05
python Ex_Convection_Disc_InternalHeat.py -uw_free_slip 0
python Ex_Convection_Disc_InternalHeat.py -uw_max_steps 500
python Ex_Convection_Disc_InternalHeat.py -uw_delta_eta 100
```
"""

# %%
params = uw.Params(
    uw_resolution = 0.1,          # Mesh cell size
    uw_free_slip = 1,             # Free-slip BC (1=True, 0=False)
    uw_restart_step = 0,          # Restart from step (0 = fresh start)
    uw_max_steps = 1,             # Maximum time steps
    uw_delta_eta = 1000.0,        # Viscosity contrast
    uw_rayleigh = 1.0e7,          # Rayleigh number
    uw_h_int = 1.0,               # Internal heating rate
)

# Convert to boolean
Free_Slip = bool(params.uw_free_slip)
viz = True

uw.options.view()

# %% [markdown]
"""
## Physical Parameters
"""

# %%
Rayleigh = params.uw_rayleigh
H_int = params.uw_h_int
k = 1.0
resI = params.uw_resolution * 3
r_o = 1.0
r_i = 0.0

# Output directory
expt_name = f"Disc_Ra1e7_H1_deleta_{params.uw_delta_eta}"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
"""
## Mesh Generation

AnnulusWithSpokes provides good element quality near the center.
"""

# %%
meshball = uw.meshing.AnnulusWithSpokes(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSizeOuter=params.uw_resolution,
    cellSizeInner=resI,
    qdegree=3,
)

meshball.dm.view()

# %% [markdown]
"""
## Coordinate System
"""

# %%
radius_fn = sympy.sqrt(meshball.X.dot(meshball.X))
unit_rvec = meshball.X / radius_fn
gravity_fn = radius_fn

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# %% [markdown]
"""
## Visualization of Mesh
"""

# %%
if viz and uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )

    pl.show()

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)
r_mesh = uw.discretisation.MeshVariable("r", meshball, 1, degree=1)
kappa = uw.discretisation.MeshVariable("kappa", meshball, 1, degree=3, varsymbol=r"\kappa")

# %% [markdown]
"""
## Viscosity Function

Frank-Kamenetskii temperature-dependent viscosity.
"""

# %%
C = sympy.log(params.uw_delta_eta)
viscosity_fn = params.uw_delta_eta * sympy.exp(-C * 0)

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = Stokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn

stokes.tolerance = 1.0e-6

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)

# Boundary conditions
if Free_Slip:
    GammaN = meshball.Gamma
    stokes.add_natural_bc(1.0e6 * GammaN.dot(v_soln.sym) * GammaN.T, "Upper")
else:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper")

# %% [markdown]
"""
## Advection-Diffusion Solver

With flux-limited diffusivity for steep temperature gradients.
"""

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    verbose=False,
    order=2,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel

# Flux limiting diffusivity (stabilizing term)
Tgrad = meshball.vector.gradient(t_soln.sym)
Tslope = sympy.sqrt(Tgrad.dot(Tgrad))
Tslope_max = 25

k_lim = Tslope / Tslope_max
k_eff = k * sympy.Max(1, k_lim)

adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.f = H_int

# %% [markdown]
"""
## Diffusivity Projection
"""

# %%
calculate_diffusivity = uw.systems.Projection(meshball, u_Field=kappa)
calculate_diffusivity.uw_function = k_eff

# %% [markdown]
"""
## Initial Conditions
"""

# %%
abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = (
    0.25
    + 0.25 * sympy.sin(7.0 * th) * sympy.sin(np.pi * (r - r_i) / (r_o - r_i))
    + 0.0 * (r_o - r) / (r_o - r_i)
)

adv_diff.add_dirichlet_bc(0.0, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evalf(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]

# %% [markdown]
"""
## Restart from Checkpoint
"""

# %%
if params.uw_restart_step != 0:
    print(f"Reading step {params.uw_restart_step}")
    t_soln.read_timestep(
        expt_name, "T", int(params.uw_restart_step), outputPath=output_dir, verbose=True
    )

# %%
with meshball.access(r_mesh):
    r_mesh.data[:, 0] = uw.function.evalf(r, meshball.data)

# %% [markdown]
"""
## Initial Solve
"""

# %%
stokes.bodyforce = unit_rvec * gravity_fn * Rayleigh * t_soln.fn
stokes.solve(verbose=False)

# %%
# Test diffusion solve
dt = 0.00001
adv_diff.solve(timestep=dt)
adv_diff.constitutive_model.Parameters.diffusivity = k_eff
adv_diff.solve(timestep=dt, zero_init_guess=False)

calculate_diffusivity.solve()

# %% [markdown]
"""
## Visualization of Initial State
"""

# %%
if viz and uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym[0])
    pvmesh.point_data["K"] = vis.scalar_fn_to_pv_points(pvmesh, kappa.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01)

    pl.show(cpos="xy")


# %% [markdown]
"""
## Visualization Function
"""

# %%
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
        pl.add_mesh(pvmesh, cmap="coolwarm", show_edges=True, scalars="T", opacity=0.75)

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.66,
        )

        pl.remove_scalar_bar("mag")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )

        pv.close_all()


# %% [markdown]
"""
## Time Evolution
"""

# %%
ts = int(params.uw_restart_step)
delta_t = 5.0e-5

for step in range(0, int(params.uw_max_steps)):
    stokes.solve(verbose=False, zero_init_guess=False)

    calculate_diffusivity.solve()

    if step % 10 == 0:
        delta_t = adv_diff.estimate_dt(v_factor=2.0, diffusivity=kappa.sym[0])

    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # Statistics
    tstats = t_soln.stats()
    Tgrad_stats = kappa.stats()
    dt_estimate = adv_diff.estimate_dt(v_factor=2.0, diffusivity=kappa.sym[0])

    uw.pprint(f"Timestep {ts}, dt {delta_t:.2e} ({dt_estimate:.2e})", flush=True)

    if ts % 10 == 0:
        plot_T_mesh(filename=f"output/{expt_name}_step_{ts}")

        meshball.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, t_soln],
            outputPath=output_dir,
            index=ts,
        )

    ts += 1

# %% [markdown]
"""
## Final Visualization
"""

# %%
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

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7.5,
        opacity=0.25,
    )

    pl.add_mesh(pvmesh, cmap="coolwarm", scalars="T", opacity=0.75)

    pl.show(cpos="xy")

# %%
print(f"Disc convection example complete: {ts} steps")
