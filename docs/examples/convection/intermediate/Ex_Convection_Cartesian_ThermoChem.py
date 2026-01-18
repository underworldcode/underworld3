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
# Thermochemical Convection

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Coupled thermal convection with a material-swarm mediated density variation.
This example demonstrates:
- Advection-diffusion for temperature
- Swarm-based material tracking
- Coupled buoyancy from temperature and composition

## Key Concepts

- **Thermal Rayleigh number (Ra)**: Controls vigor of thermal convection
- **Chemical Rayleigh number (Rc)**: Controls chemical buoyancy contribution
- **Buoyancy**: Combined thermal and chemical density variations drive flow

## Parameters

- `uw_rayleigh_thermal`: Thermal Rayleigh number (default 1e6)
- `uw_rayleigh_chemical`: Chemical Rayleigh number (default 5e5)
- `uw_cell_size`: Mesh resolution (default 1/24)
- `uw_n_steps`: Number of time steps (default 50)
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
import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Convection_Cartesian_ThermoChem.py -uw_rayleigh_thermal 1e7
python Ex_Convection_Cartesian_ThermoChem.py -uw_n_steps 100
```
"""

# %%
params = uw.Params(
    uw_rayleigh_thermal = 1.0e6,    # Thermal Rayleigh number
    uw_rayleigh_chemical = 5.0e5,   # Chemical Rayleigh number (+ve = heavy)
    uw_cell_size = 1.0 / 24.0,      # Mesh cell size
    uw_diffusivity = 1.0,           # Thermal diffusivity
    uw_n_steps = 50,                # Number of time steps
    uw_dt = 3.0e-5,                 # Time step size
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
)
meshbox.dm.view()

# %% [markdown]
"""
## Visualization of Mesh
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)
    pl.show(cpos="xy")

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3)

# %%
swarm = uw.swarm.Swarm(mesh=meshbox)
Mat = uw.swarm.SwarmVariable("Material", swarm, 1, proxy_degree=3)
X0 = uw.swarm.SwarmVariable("X0", swarm, meshbox.dim, _proxy=False)
swarm.populate(fill_param=5)

# %% [markdown]
"""
## Stokes Solver Setup
"""

# %%
stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

stokes.petsc_options.delValue("ksp_monitor")

# Constant viscosity
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1

# Velocity boundary conditions: free-slip
stokes.add_dirichlet_bc((0.0), "Left", (0))
stokes.add_dirichlet_bc((0.0), "Right", (0))
stokes.add_dirichlet_bc((0.0), "Top", (1))
stokes.add_dirichlet_bc((0.0), "Bottom", (1))

# %% [markdown]
"""
## Material Projection
"""

# %%
mMat = uw.discretisation.MeshVariable("mMat", meshbox, 1, degree=2)
projector = uw.systems.solvers.SNES_Projection(meshbox, mMat)
projector.smoothing = 1.0e-3

# %%
x = meshbox.N.x
y = meshbox.N.y

# %% [markdown]
"""
## Advection-Diffusion Setup
"""

# %%
k = params.uw_diffusivity

adv_diff = uw.systems.AdvDiffusion(
    meshbox,
    u_Field=t_soln,
    V_fn=v_soln,
    order=3,
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

# %% [markdown]
"""
## Initial Conditions
"""

# %%
# Temperature: linear profile with perturbation
init_t = 0.01 * sympy.sin(5.0 * x) * sympy.sin(np.pi * y) + (1.0 - y)

# TODO: Consider uw.synchronised_array_update() for multi-variable assignment
t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
t_soln.data[...] = t_0.data[...]

# %%
# Material: step function at y=0.25
Mat.data[:, 0] = 0.5 + 0.5 * np.tanh(100.0 * (swarm.data[:, 1] - 0.25))

projector.uw_function = Mat.sym
projector.solve()

# %% [markdown]
"""
## Buoyancy Force

Combined thermal and chemical buoyancy:
- Positive Rc means heavy chemical component
- Negative Rc means light chemical component
"""

# %%
expt_name = "output/Ra{:.0e}_Rc{:.0e}".format(
    params.uw_rayleigh_thermal, params.uw_rayleigh_chemical
)

buoyancy_force = params.uw_rayleigh_thermal * t_soln.fn + params.uw_rayleigh_chemical * mMat.fn
stokes.bodyforce = meshbox.N.j * buoyancy_force

# Initial solve
stokes.solve()

# %%
# Check the diffusion part converges
adv_diff.solve(timestep=0.01 * stokes.estimate_dt())

# %% [markdown]
"""
## Visualization Setup
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
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
        opacity=0.5,
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1.0e-4, opacity=0.5)

    pl.show(cpos="xy")


# %%
def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshbox)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
        pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, mMat.sym)

        spoints = vis.swarm_to_pv_cloud(swarm)
        swarm_point_cloud = pv.PolyData(spoints)
        swarm_point_cloud.point_data["M"] = Mat.data.copy()

        velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

        pl = pv.Plotter(window_size=(750, 750))

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00001, opacity=0.75)

        pl.add_points(
            swarm_point_cloud,
            cmap="RdYlBu",
            render_points_as_spheres=True,
            point_size=7.5,
            opacity=1.0,
        )

        pl.add_mesh(
            pvmesh,
            cmap="gray",
            edge_color="Black",
            show_edges=True,
            scalars="M",
            use_transparency=False,
            opacity=0.5,
        )

        pl.remove_scalar_bar("M")
        pl.remove_scalar_bar("mag")

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
for step in range(0, params.uw_n_steps):
    stokes.solve(zero_init_guess=False)
    delta_t = params.uw_dt
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # Update swarm locations
    swarm.advection(v_soln.fn, delta_t, order=2, corrector=True)

    # Stats
    tstats = t_soln.stats()

    uw.pprint("Timestep {}, dt {}".format(step, delta_t))

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
    pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, mMat.sym)

    tpoints = vis.meshVariable_to_pv_cloud(t_soln)
    tpoints.point_data["T"] = vis.scalar_fn_to_pv_points(tpoints, t_soln.sym)
    tpoints.point_data["M"] = vis.scalar_fn_to_pv_points(tpoints, mMat.sym)

    spoints = vis.swarm_to_pv_cloud(swarm)
    swarm_point_cloud = pv.PolyData(spoints)
    swarm_point_cloud.point_data["M"] = Mat.data.copy()

    velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.75e-5, opacity=0.75)

    pl.add_points(
        swarm_point_cloud,
        cmap="RdYlBu",
        render_points_as_spheres=True,
        point_size=3.0,
        opacity=1.0,
    )

    pl.add_mesh(
        pvmesh,
        cmap="gray",
        edge_color="Black",
        show_edges=True,
        scalars="M",
        use_transparency=False,
        opacity=0.5,
    )

    pl.show(cpos="xy")
