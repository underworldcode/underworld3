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
# Nonlinear Viscosity Convection

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Thermal convection with strain-rate dependent (nonlinear) viscosity. The viscosity
depends on both temperature and strain rate, creating more complex flow patterns.

## Key Concepts

- **Nonlinear rheology**: Viscosity depends on strain rate
- **Strain-rate softening**: Higher strain rates reduce effective viscosity
- **Coupled T and strain rate**: Both control the flow
- **Iterative solve**: Newton method for nonlinear system

## Mathematical Formulation

Combined viscosity law:
$$\\eta(T, \\dot\\varepsilon) = \\eta_T(T) \\cdot \\eta_{NL}(\\dot\\varepsilon)$$

Where:
- Temperature dependence: $\\eta_T = \\Delta\\eta \\cdot \\exp(-\\ln(\\Delta\\eta) \\cdot T)$
- Strain-rate dependence: $\\eta_{NL} = 0.1 + 10/(1 + \\dot\\varepsilon_{II})$

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_rayleigh`: Rayleigh number
- `uw_log_viscosity_contrast`: log10(temperature viscosity ratio)
- `uw_n_steps`: Number of time steps
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Convection_4_SLCN_Cartesian-NL.py -uw_rayleigh 1e7
python Ex_Convection_4_SLCN_Cartesian-NL.py -uw_n_steps 500
```
"""

# %%
params = uw.Params(
    uw_cell_size = 1.0 / 24.0,           # Mesh cell size
    uw_rayleigh = 1.0e6,                 # Rayleigh number
    uw_log_viscosity_contrast = 6,       # log10(delta_eta) for temperature
    uw_diffusivity = 1.0,                # Thermal diffusivity
    uw_n_steps = 250,                    # Number of time steps
    uw_dt_factor = 5.0,                  # Time step multiplier
)

# Derived parameters
delta_eta = 10 ** params.uw_log_viscosity_contrast

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=params.uw_cell_size,
    qdegree=3,
)

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)
visc = uw.discretisation.MeshVariable(r"\eta", meshbox, 1, degree=1)

x, y = meshbox.X

# %% [markdown]
"""
## Stokes Solver with Temperature-Dependent Viscosity

Initial setup with linear (temperature-only) viscosity for first solve.
"""

# %%
stokes = uw.systems.Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

# Temperature-dependent viscosity (Arrhenius-type)
viscosity_L = delta_eta * sympy.exp(-sympy.log(delta_eta) * t_soln.sym[0])

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity_L

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity
stokes.penalty = 0.0

# Set solver reference for constitutive model
stokes.constitutive_model.solver = stokes

# %% [markdown]
"""
## Boundary Conditions

Free-slip on all walls.
"""

# %%
stokes.add_dirichlet_bc((0.0), "Left", (0))
stokes.add_dirichlet_bc((0.0), "Right", (0))
stokes.add_dirichlet_bc((0.0), "Top", (1))
stokes.add_dirichlet_bc((0.0), "Bottom", (1))

# Buoyancy force
buoyancy_force = params.uw_rayleigh * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %% [markdown]
"""
## Advection-Diffusion Solver
"""

# %%
adv_diff = uw.systems.AdvDiffusion(
    meshbox,
    u_Field=t_soln,
    V_fn=v_soln,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = params.uw_diffusivity
adv_diff.theta = 0.5

# Temperature boundary conditions
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

# %% [markdown]
"""
## Viscosity Projection

Project the effective viscosity onto a mesh variable for visualization.
"""

# %%
scalar_projection = uw.systems.Projection(meshbox, visc)
scalar_projection.uw_function = 0.1 + 10.0 / (1.0 + stokes.Unknowns.Einv2)
scalar_projection.smoothing = 1.0e-6

# %% [markdown]
"""
## Initial Conditions
"""

# %%
init_t = 0.9 * (0.05 * sympy.cos(sympy.pi * x) + sympy.cos(0.5 * np.pi * y)) + 0.05

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]

# %% [markdown]
"""
## Initial Solve (Linear Viscosity)
"""

# %%
stokes.solve()

# %% [markdown]
"""
## Enable Nonlinear Viscosity

Now add strain-rate dependence to the viscosity:
$$\\eta_{NL} = \\eta_T \\cdot (0.1 + 10/(1 + \\dot\\varepsilon_{II}))$$
"""

# %%
viscosity_NL = viscosity_L * (0.1 + 10.0 / (1.0 + stokes.Unknowns.Einv2))

stokes.constitutive_model.Parameters.viscosity = viscosity_NL
stokes.saddle_preconditioner = 1 / viscosity_NL

stokes.solve(zero_init_guess=False)

# Check the diffusion solve
adv_diff.solve(timestep=0.01 * stokes.estimate_dt())

# Compute viscosity field
scalar_projection.solve()

with meshbox.access():
    print(f"Viscosity range: [{visc.min():.2e}, {visc.max():.2e}]")

# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["V"] = 10.0 * vis.vector_fn_to_pv_points(pvmesh, v_soln.sym) / vis.vector_fn_to_pv_points(pvmesh, v_soln.sym).max()
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
        pvmesh.point_data["eta"] = vis.scalar_fn_to_pv_points(pvmesh, visc.sym)

        # Point sources for streamlines
        cpoints = np.zeros((meshbox._centroids[::2].shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[::2, 0]
        cpoints[:, 1] = meshbox._centroids[::2, 1]
        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
            cpoint_cloud,
            vectors="V",
            integrator_type=45,
            integration_direction="forward",
            compute_vorticity=False,
            max_steps=25,
            surface_streamlines=True,
        )

        pl = pv.Plotter(window_size=(750, 750))

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_mesh(
            pvmesh,
            cmap="Greys",
            show_edges=False,
            scalars="eta",
            use_transparency=False,
            opacity=0.25,
        )

        pl.add_mesh(pvstream, opacity=0.4)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("V")
        pl.remove_scalar_bar("eta")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )


# %% [markdown]
"""
## Time Evolution
"""

# %%
expt_name = "output/Ra1e6_NL"

for step in range(0, params.uw_n_steps):
    stokes.solve(zero_init_guess=False)
    delta_t = params.uw_dt_factor * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # Stats
    tstats = t_soln.stats()

    uw.pprint("Timestep {}, dt {:.2e}".format(step, delta_t))

    plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

# %% [markdown]
"""
## Final Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    points = vis.meshVariable_to_pv_cloud(t_soln)
    points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
    point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00002, opacity=0.75)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7.5,
        opacity=0.25,
    )

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")

# %%
print(f"Final temperature stats: {t_soln.stats()}")
