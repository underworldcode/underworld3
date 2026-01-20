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
# Convection with Yield Stress

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Thermal convection with a yield stress rheology. The viscosity depends on
both temperature and strain rate, with plastic yielding when stress exceeds
the yield strength. No strain softening is included.

## Key Concepts

- **Yield stress**: Stress limit before plastic flow
- **Piecewise viscosity**: Sympy.Piecewise for differentiable rheology
- **Depth-dependent yield**: tau_Y increases with depth (lithostatic pressure)
- **Newton convergence**: Differentiable formulation enables Jacobians

## Mathematical Formulation

Viscosity law with yielding:
$$\\eta = \\begin{cases}
\\eta_T & \\text{if } 2\\eta_T\\dot\\varepsilon_{II} < \\tau_Y \\\\
\\tau_Y / (2\\dot\\varepsilon_{II}) & \\text{otherwise}
\\end{cases}$$

Where $\\tau_Y = \\tau_0 (1 + 100(1-y))$ increases with depth.

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_rayleigh`: Rayleigh number
- `uw_yield_stress_base`: Base yield stress
- `uw_depth_factor`: Depth dependence of yield stress
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
python Ex_Convection_5_SLCN_Cartesian-Yield.py -uw_yield_stress_base 5e4
python Ex_Convection_5_SLCN_Cartesian-Yield.py -uw_rayleigh 1e7
```
"""

# %%
params = uw.Params(
    uw_cell_size = 1.0 / 24.0,           # Mesh cell size
    uw_rayleigh = 1.0e6,                 # Rayleigh number
    uw_log_viscosity_contrast = 6,       # log10(delta_eta) for temperature
    uw_yield_stress_base = 1.0e5,        # Base yield stress
    uw_depth_factor = 100.0,             # Depth dependence factor
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
visc = uw.discretisation.MeshVariable(r"\eta", meshbox, 1, degree=2)
tau_inv = uw.discretisation.MeshVariable(r"|\tau|", meshbox, 1, degree=2)

x, y = meshbox.X

# %% [markdown]
"""
## Stokes Solver

Initial setup with temperature-dependent viscosity (linear case).
"""

# %%
stokes = uw.systems.Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

stokes.petsc_options["ksp_monitor"] = None

# Temperature-dependent viscosity (Arrhenius-type)
viscosity_L = delta_eta * sympy.exp(-sympy.log(delta_eta) * t_soln.sym[0])

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.0

# Boundary conditions (free-slip)
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
## Projection Solvers

For visualizing viscosity and stress fields.
"""

# %%
viscosity_evaluation = uw.systems.Projection(meshbox, visc)
viscosity_evaluation.uw_function = viscosity_L
viscosity_evaluation.smoothing = 1.0e-3

stress_inv_evaluation = uw.systems.Projection(meshbox, tau_inv)
stress_inv_evaluation.uw_function = 2.0 * stokes.constitutive_model.Parameters.viscosity * stokes.Unknowns.Einv2
stress_inv_evaluation.smoothing = 1.0e-3

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
## Enable Yield Stress Rheology

Depth-dependent yield stress:
$$\\tau_Y = \\tau_0 (1 + \\alpha(1-y))$$

Piecewise viscosity enables differentiable Jacobians.
"""

# %%
tau_Y = params.uw_yield_stress_base * (1 + params.uw_depth_factor * (1 - y))

viscosity_NL = sympy.Piecewise(
    (viscosity_L, 2 * viscosity_L * stokes.Unknowns.Einv2 < tau_Y),
    (tau_Y / (2 * stokes.Unknowns.Einv2), True),
)

stokes.constitutive_model.Parameters.viscosity = viscosity_NL
stokes.saddle_preconditioner = 1 / viscosity_NL

stokes.solve(zero_init_guess=False)
adv_diff.solve(timestep=0.01 * stokes.estimate_dt())

# Compute fields for visualization
viscosity_evaluation.uw_function = viscosity_NL
viscosity_evaluation.solve()

stress_inv_evaluation.uw_function = 2.0 * viscosity_NL * stokes.Unknowns.Einv2
stress_inv_evaluation.solve()

with meshbox.access():
    print(f"Viscosity range: [{visc.min():.2e}, {visc.max():.2e}]")
    print(f"Stress range: [{tau_inv.min():.2e}, {tau_inv.max():.2e}]")

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
        pvmesh.point_data["tau"] = vis.scalar_fn_to_pv_points(pvmesh, tau_inv.sym)

        # Point sources for streamlines
        subsample = 10
        cpoints = np.zeros((meshbox._centroids[::subsample].shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[::subsample, 0]
        cpoints[:, 1] = meshbox._centroids[::subsample, 1]
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
            opacity=0.75,
        )

        pl.add_mesh(
            pvmesh,
            cmap="Greys",
            show_edges=False,
            scalars="eta",
            use_transparency=False,
            opacity="geom",
        )

        pl.add_mesh(pvstream, opacity=0.5)

        for key in pvmesh.point_data.keys():
            try:
                pl.remove_scalar_bar(key)
            except KeyError:
                pass

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
expt_name = "output/Ra1e6_TauY"

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
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

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
