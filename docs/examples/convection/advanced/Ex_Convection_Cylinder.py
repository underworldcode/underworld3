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
# Convection in a Cylindrical Annulus

**PHYSICS:** convection
**DIFFICULTY:** advanced

## Description

Thermal convection in a cylindrical (2D annulus) domain. This demonstrates
convection in a geometry relevant to planetary mantles, with radial gravity
and curved boundaries.

## Key Concepts

- **Annulus geometry**: Inner and outer radii define the domain
- **Radial gravity**: Gravity points toward the center
- **Curved boundaries**: No-slip on inner and outer surfaces
- **Swarm tracking**: Particles for Lagrangian tracking

## Parameters

- `uw_cell_size`: Mesh resolution
- `uw_rayleigh`: Rayleigh number
- `uw_radius_inner`: Inner radius
- `uw_radius_outer`: Outer radius
- `uw_n_steps`: Number of time steps
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Convection_Cylinder.py -uw_rayleigh 1e7
python Ex_Convection_Cylinder.py -uw_cell_size 0.05
```
"""

# %%
params = uw.Params(
    uw_cell_size = 0.1,              # Mesh cell size
    uw_rayleigh = 1.0e6,             # Rayleigh number
    uw_radius_inner = 0.5,           # Inner radius
    uw_radius_outer = 1.0,           # Outer radius
    uw_diffusivity = 1.0,            # Thermal diffusivity
    uw_n_steps = 50,                 # Number of time steps
    uw_dt_factor = 5.0,              # Time step multiplier
)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
meshball = uw.meshing.Annulus(
    radiusInner=params.uw_radius_inner,
    radiusOuter=params.uw_radius_outer,
    cellSize=params.uw_cell_size,
    degree=1,
    qdegree=3,
)

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)

# %% [markdown]
"""
## Swarm for Lagrangian Tracking
"""

# %%
swarm = uw.swarm.Swarm(mesh=meshball)
T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1)
X1 = uw.swarm.SwarmVariable("Xminus1", swarm, 2)
swarm.populate(fill_param=3)

# %% [markdown]
"""
## Coordinate System

Set up radial coordinates and gravity pointing toward center.
"""

# %%
x = meshball.X[0]
y = meshball.X[1]

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

# Radial unit vector (points outward)
radius_fn = sympy.sqrt(meshball.X.dot(meshball.X))
unit_rvec = meshball.X / (1.0e-10 + radius_fn)

# Gravity increases linearly with radius
gravity_fn = radius_fn

r_i = params.uw_radius_inner
r_o = params.uw_radius_outer

# %% [markdown]
"""
## Stokes Solver

Constant viscosity with radial buoyancy force.
"""

# %%
stokes = uw.systems.Stokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0

stokes.petsc_options.delValue("ksp_monitor")

# No-slip on inner and outer boundaries
stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# Radial buoyancy force
buoyancy_force = params.uw_rayleigh * t_soln.sym[0] / (r_i**3)
stokes.bodyforce = unit_rvec * buoyancy_force

# %% [markdown]
"""
## Advection-Diffusion Solver
"""

# %%
adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = params.uw_diffusivity
adv_diff.theta = 0.5

adv_diff.petsc_options["ksp_monitor"] = None

# Temperature boundary conditions: hot inner, cold outer
adv_diff.add_dirichlet_bc(1.0, "Lower")
adv_diff.add_dirichlet_bc(0.0, "Upper")

# %% [markdown]
"""
## Initial Conditions

Small sinusoidal perturbation on conductive profile.
"""

# %%
abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.01 * sympy.sin(15.0 * th) * sympy.sin(np.pi * (r - r_i) / (r_o - r_i)) + (r_o - r) / (r_o - r_i)

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]

# %% [markdown]
"""
## Initial Solve
"""

# %%
stokes.solve()
adv_diff.solve(timestep=0.00001 * stokes.estimate_dt())

print(f"Initial dt estimate: {stokes.estimate_dt():.2e}")

# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_T_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(meshball)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

        tpoints = vis.meshVariable_to_pv_cloud(t_soln)
        tpoints.point_data["T"] = vis.scalar_fn_to_pv_points(tpoints, t_soln.sym)
        point_cloud = pv.PolyData(tpoints)

        velocity_points = vis.meshVariable_to_pv_cloud(stokes.u)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, stokes.u.sym)

        pl = pv.Plotter(window_size=(750, 750))

        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00002, opacity=0.75)

        pl.add_points(
            point_cloud,
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=10,
            opacity=0.66,
        )

        pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

        pl.remove_scalar_bar("T")
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
expt_name = "output/Cylinder_Ra1e6"

for step in range(0, params.uw_n_steps):
    stokes.solve()
    delta_t = params.uw_dt_factor * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t)

    # Stats
    tstats = t_soln.stats()

    uw.pprint("Timestep {}, dt {:.2e}".format(step, delta_t))

    # Save checkpoint
    meshball.petsc_save_checkpoint(
        index=step,
        meshVars=[v_soln, t_soln],
        outputPath=expt_name,
    )

# %% [markdown]
"""
## Final Visualization
"""

# %%
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

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.00005, opacity=0.75)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=7.5,
        opacity=0.75,
    )

    pl.add_mesh(pvmesh, scalars="T", cmap="coolwarm", opacity=1)

    pl.show(cpos="xy")

# %%
print(f"Final temperature stats: {t_soln.stats()}")
