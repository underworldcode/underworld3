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
# Constant Viscosity Convection (Cartesian)

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Thermal convection with constant viscosity in a Cartesian domain. This example
couples a Stokes solver with an advection-diffusion solver to simulate
buoyancy-driven flow.

## Key Concepts

- **Coupled solvers**: Stokes and advection-diffusion work together
- **Semi-Lagrangian**: SLCN method for advection with diffusion
- **Rayleigh number**: Ra = 10^6 controls convection vigor
- **Free-slip boundaries**: Velocity constrained normal to walls

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_rayleigh`: Rayleigh number (buoyancy force scaling)
- `uw_n_steps`: Number of time steps to evolve
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
python Ex_Convection_1_SLCN_Cartesian.py -uw_cell_size 0.04
python Ex_Convection_1_SLCN_Cartesian.py -uw_rayleigh 1e7
```
"""

# %%
params = uw.Params(
    uw_cell_size = 1.0 / 12.0,      # Mesh cell size
    uw_rayleigh = 1.0e6,            # Rayleigh number
    uw_diffusivity = 1.0,           # Thermal diffusivity
    uw_n_steps = 100,               # Number of time steps
    uw_dt_factor = 2.0,             # Time step multiplier
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=params.uw_cell_size,
    regular=False,
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

x, y = meshbox.X

# %% [markdown]
"""
## Stokes Solver

Constant viscosity with free-slip boundary conditions on all walls.
"""

# %%
stokes = uw.systems.Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.tolerance = 1.0e-3

# Free-slip boundary conditions
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

# Buoyancy force
buoyancy_force = params.uw_rayleigh * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %% [markdown]
"""
## Advection-Diffusion Solver

Semi-Lagrangian Crank-Nicolson (SLCN) scheme for temperature evolution.
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

Sinusoidal perturbation on linear temperature profile.
"""

# %%
init_t = sympy.sin(5 * sympy.pi * x) * sympy.sin(sympy.pi * y) / 5 + (1.0 - y)

t_0.array[...] = uw.function.evaluate(init_t, t_0.coords)
t_soln.array[...] = t_0.array[...]

# %% [markdown]
"""
## Initial Solve
"""

# %%
stokes.solve(zero_init_guess=True)
adv_diff.solve(timestep=2 * stokes.estimate_dt(), zero_init_guess=True)

print(f"Ra = {params.uw_rayleigh}, initial dt = {stokes.estimate_dt():.2e}")

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

        pv_mesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
        pv_mesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_mesh_t, t_soln.sym)

        # Point sources at cell centres for streamlines
        cpoints = np.zeros((meshbox._centroids[::4].shape[0], 3))
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

        pl = pv.Plotter(window_size=(1000, 750))

        pl.add_mesh(
            pv_mesh_t,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=False,
            scalars="T",
            use_transparency=False,
            opacity=1,
        )

        pl.add_mesh(
            pv_mesh_t.copy(),
            style="wireframe",
            color="Black",
            use_transparency=False,
            opacity=0.1,
        )

        pl.add_mesh(pvstream, opacity=0.666)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1280, 1280),
            return_img=False,
        )
        pl.close()
        pv.close_all()


# %% [markdown]
"""
## Time Evolution
"""

# %%
expt_name = "output/Ra1e6"

for step in range(0, params.uw_n_steps):
    stokes.solve(zero_init_guess=False)
    delta_t = params.uw_dt_factor * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

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
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, stokes.u.sym)

    pv_mesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pv_mesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_mesh_t, t_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pv_mesh_t,
        cmap="coolwarm",
        edge_color="Gray",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        opacity=1,
    )

    pl.add_mesh(
        pv_mesh_t.copy(),
        style="wireframe",
        color="Black",
        use_transparency=False,
        opacity=0.05,
    )

    pl.show(cpos="xy")

# %%
print(f"Final temperature stats: {t_soln.stats()}")
