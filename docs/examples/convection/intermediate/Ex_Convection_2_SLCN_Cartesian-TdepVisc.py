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
# Temperature-Dependent Viscosity Convection

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Thermal convection with strongly temperature-dependent viscosity. The viscosity
varies by several orders of magnitude from hot (low viscosity) to cold (high
viscosity), creating a stagnant lid at the surface.

## Key Concepts

- **Arrhenius viscosity**: Exponential temperature dependence
- **Stagnant lid**: Cold, high-viscosity lid at surface
- **Nonlinear solve**: Newton iteration for viscosity-dependent equations
- **Viscosity contrast**: delta_eta = 10^6 between hot and cold

## Mathematical Formulation

Viscosity law:
$$\\eta(T) = \\eta_0 \\cdot \\exp(-\\ln(\\Delta\\eta) \\cdot T)$$

Where T is normalized temperature (0 at top, 1 at bottom).

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_rayleigh`: Rayleigh number
- `uw_log_viscosity_contrast`: log10(eta_max/eta_min)
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
python Ex_Convection_2_SLCN_Cartesian-TdepVisc.py -uw_log_viscosity_contrast 4
python Ex_Convection_2_SLCN_Cartesian-TdepVisc.py -uw_rayleigh 1e7
```
"""

# %%
params = uw.Params(
    uw_cell_size = 1.0 / 32.0,           # Mesh cell size
    uw_rayleigh = 1.0e6,                 # Rayleigh number
    uw_log_viscosity_contrast = 6,       # log10(delta_eta)
    uw_diffusivity = 1.0,                # Thermal diffusivity
    uw_n_steps = 1000,                   # Number of time steps
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
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)

x, y = meshbox.X

# %% [markdown]
"""
## Stokes Solver with Temperature-Dependent Viscosity

The viscosity follows an Arrhenius-type law:
$$\\eta(T) = \\Delta\\eta \\cdot \\exp(-\\ln(\\Delta\\eta) \\cdot T)$$

This gives eta = delta_eta at T=0 (cold) and eta = 1 at T=1 (hot).
"""

# %%
stokes = uw.systems.Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

# Temperature-dependent viscosity (Arrhenius-type)
viscosity = delta_eta * sympy.exp(-sympy.log(delta_eta) * t_soln.sym[0])

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.penalty = 0.0

# Preconditioner for nonlinear solve
stokes.saddle_preconditioner = 1.0 / viscosity

# Solver tolerances adjusted for high viscosity contrast
stokes.petsc_options["snes_rtol"] = 1.0 / delta_eta
stokes.petsc_options["snes_atol"] = 0.01

# %% [markdown]
"""
## Boundary Conditions

Free-slip on all walls: normal velocity = 0, tangential stress = 0.
"""

# %%
stokes.add_dirichlet_bc((0.0), "Top", (1))
stokes.add_dirichlet_bc((0.0), "Bottom", (1))
stokes.add_dirichlet_bc((0.0), "Left", (0))
stokes.add_dirichlet_bc((0.0), "Right", (0))

# Buoyancy force
buoyancy_force = params.uw_rayleigh * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %% [markdown]
"""
## Advection-Diffusion Solver
"""

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
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
## Initial Conditions

Cosine perturbation to break symmetry and start convection.
"""

# %%
init_t = 0.9 * (0.05 * sympy.cos(sympy.pi * x) + sympy.cos(0.5 * np.pi * y)) + 0.05

with meshbox.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
    print(f"Initial T range: [{t_0.data.min():.3f}, {t_0.data.max():.3f}]")

# %% [markdown]
"""
## Initial Solve
"""

# %%
stokes.solve(zero_init_guess=True)
adv_diff.solve(timestep=0.1 * stokes.estimate_dt())

print(f"Ra = {params.uw_rayleigh}, delta_eta = 10^{params.uw_log_viscosity_contrast}")

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
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym) / 333
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

        # Point sources for streamlines
        cpoints = np.zeros((meshbox._centroids[::4, 0].shape[0], 3))
        cpoints[:, 0] = meshbox._centroids[::4, 0]
        cpoints[:, 1] = meshbox._centroids[::4, 1]
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

        points = vis.meshVariable_to_pv_cloud(t_soln)
        points.point_data["T"] = vis.scalar_fn_to_pv_points(points, t_soln.sym)
        point_cloud = pv.PolyData(points)

        pl = pv.Plotter(window_size=(1000, 750))

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.5,
        )

        pl.add_mesh(pvstream, opacity=0.4)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("V")

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )

        pvmesh.clear_data()
        pvmesh.clear_point_data()


# %% [markdown]
"""
## Time Evolution
"""

# %%
expt_name = f"output/Ra1e6_eta1e{params.uw_log_viscosity_contrast}"

for step in range(0, params.uw_n_steps):
    stokes.solve(zero_init_guess=False)
    delta_t = params.uw_dt_factor * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=True)

    # Stats
    tstats = t_soln.stats()

    uw.pprint("Timestep {}, dt {:.2e}".format(step, delta_t))

    if step % 5 == 0:
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

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00001, opacity=0.75)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7,
        opacity=0.25,
    )

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")

# %%
print(f"Final temperature stats: {t_soln.stats()}")
