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
# Viscous Fingering in Porous Media

**PHYSICS:** porous_flow
**DIFFICULTY:** advanced

## Description

Viscous fingering instability in porous media flow. When a less viscous
fluid displaces a more viscous fluid, the interface becomes unstable
and develops finger-like patterns.

Based on Darcy flow with advection of two fluids with varying viscosity.

## Key Concepts

- **Darcy flow**: Pressure-driven flow through porous media
- **Mobility ratio**: Ratio of viscosities controls instability
- **Swarm advection**: Tracking material interface with particles

## Physical Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Domain size | x,y | 10 | m |
| Permeability | k | 1e-13 | m^2 |
| Porosity | phi | 0.1 | - |
| Diffusivity | kappa | 1e-9 | m^2/s |
| Viscosity (solvent) | mu_s | 1.33e-4 | Pa s |
| Viscosity (oil) | mu_o | 20*mu_s | Pa s |

## References

- Homsy, G.M. (1987). Viscous Fingering in Porous Media. Ann. Rev. Fluid Mech.
- Simpson (2017). Practical Finite Element Modeling in Earth Science using Matlab

## Parameters

- `uw_n_elements`: Number of mesh elements per side
- `uw_n_steps`: Number of time steps
- `uw_viscosity_ratio`: mu_oil / mu_solvent ratio
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
import matplotlib.pyplot as plt
import os

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_viscousFingering.py -uw_n_elements 50
python Ex_viscousFingering.py -uw_viscosity_ratio 30
```
"""

# %%
params = uw.Params(
    uw_n_elements = 25,           # Elements per side
    uw_n_steps = 20,              # Number of time steps
    uw_viscosity_ratio = 20.0,    # mu_oil / mu_solvent
    uw_porosity = 0.1,            # Porosity
    uw_permeability = 1.0e-13,    # Permeability (m^2)
    uw_diffusivity = 1.0e-9,      # Diffusivity (m^2/s)
)

# Output directory
outputDir = "./output/viscousFingering_example/"
if uw.mpi.rank == 0:
    os.makedirs(outputDir, exist_ok=True)

# %% [markdown]
"""
## Unit Scaling
"""

# %%
u = uw.scaling.units
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# Physical parameters
refLength = 10  # Domain size in meters
eta = 1.33e-4   # Solvent viscosity (Pa s)
kappa = params.uw_diffusivity
perm = params.uw_permeability
porosity = params.uw_porosity

refTime = perm / kappa
refViscosity = eta * u.pascal * u.second

KL = 1.0 * u.millimetre
KT = 1300 * u.kelvin
Kt = 0.01 * u.year
KM = refViscosity * KL * Kt

scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"] = KM
scaling_coefficients["[temperature]"] = KT

# %% [markdown]
"""
## Mesh Generation
"""

# %%
minX, maxX = 0, nd(10 * u.meter)
minY, maxY = 0, nd(10 * u.meter)

elements = params.uw_n_elements

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=maxY / elements,
    qdegree=5,
)

# Visualization mesh (finer)
vizmesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=0.5 * maxY / elements,
    qdegree=1,
)

# %% [markdown]
"""
## Variables
"""

# %%
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
mat = uw.discretisation.MeshVariable("mat", mesh, 1, degree=3, continuous=True)

x = mesh.N.x
y = mesh.N.y

# %% [markdown]
"""
## Mesh Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )

    pl.show(cpos="xy")

# %% [markdown]
"""
## Darcy Solver Setup
"""

# %%
darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=p_soln, v_Field=v_soln)
darcy.petsc_options.delValue("ksp_monitor")
darcy.petsc_options["snes_rtol"] = 1.0e-6
darcy.constitutive_model = uw.constitutive_models.DiffusionModel

# %% [markdown]
"""
## Swarm for Material Tracking
"""

# %%
swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=5)

material = swarm.add_variable(name="M", size=1, proxy_degree=mat.degree)
conc = swarm.add_variable(name="C", size=1, proxy_degree=mat.degree)

swarm.populate(fill_param=4)

# %% [markdown]
"""
## Advection-Diffusion Solver
"""

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh=mesh,
    u_Field=mat,
    V_fn=v_soln,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel

# %% [markdown]
"""
## Initial Material Distribution

Random perturbation along the interface to trigger instability.
"""

# %%
np.random.seed(100)

x0 = nd(2.5 * u.meter)
dx = max(mesh.get_min_radius(), nd(0.1 * u.meter))

# Perturbations at interface
fluctuation = nd(0.01 * u.meter) * np.cos(
    mat.coords[:, 1] / nd(0.5 * u.meter) * np.pi
)
fluctuation += nd(0.01 * u.meter) * np.cos(
    mat.coords[:, 1] / nd(2.0 * u.meter) * np.pi
)
fluctuation += nd(0.05 * u.meter) * np.random.random(size=mat.coords.shape[0])

mat.data[...] = 0
mat.data[mat.coords[:, 0] + fluctuation < x0] = 1

# Initialize swarm material
with swarm.access(material):
    material.data[:, 0] = uw.function.evalf(mat.sym, swarm._particle_coordinates.data)

# %% [markdown]
"""
## Visualization of Initial Material
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["mat"] = vis.scalar_fn_to_pv_points(pvmesh, mat.sym)

    points = vis.swarm_to_pv_cloud(swarm)
    point_cloud = pv.PolyData(points)

    with swarm.access(material):
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=True,
        point_size=10,
        opacity=0.33,
    )

    pl.show(cpos="xy")

# %% [markdown]
"""
## Viscosity Model

Quarter-power mixing rule for viscosity:
$$\\mu_c = \\left( \\frac{c}{\\mu_o^{1/4}} + \\frac{1-c}{\\mu_s^{1/4}} \\right)^{-4}$$
"""

# %%
eta_s = nd(1.33e-4 * u.pascal * u.second)
eta_o = params.uw_viscosity_ratio * eta_s

# Quarter-power mixing rule
eta_fn = (material.sym[0] / eta_s**0.25 + (1 - material.sym[0]) / eta_o**0.25) ** (-4)

# %% [markdown]
"""
## Material Properties
"""

# %%
nd_perm = nd(perm * u.meter**2)

diffusivity_fn = nd_perm / eta_fn

darcy.constitutive_model.Parameters.diffusivity = diffusivity_fn
adv_diff.constitutive_model.Parameters.diffusivity = nd(1e-9 * u.meter**2 / u.second)

# %% [markdown]
"""
## Boundary Conditions
"""

# %%
p0_nd = nd(0.1e6 * u.pascal)

# Darcy solver BCs
darcy.f = 0.0
darcy.s = sympy.Matrix([0, 0]).T

darcy.add_dirichlet_bc(p0_nd, "Left")
darcy.add_dirichlet_bc(0.0, "Right")

# Advection-diffusion BCs
adv_diff.add_dirichlet_bc(1.0, "Left")
adv_diff.add_dirichlet_bc(0.0, "Right")

# %% [markdown]
"""
## Initial Solve
"""

# %%
darcy.solve()

time = 0
step = 0

# %% [markdown]
"""
## Time Evolution
"""

# %%
finish_time = 0.01 * u.year

for iteration in range(0, params.uw_n_steps):
    if uw.mpi.rank == 0:
        print(f"\n\nstep: {step}, time: {dim(time, u.year)}")

    if step % 5 == 0:
        mesh.write_timestep(
            "viscousFinger",
            meshUpdates=False,
            meshVars=[p_soln, v_soln, mat],
            outputPath=outputDir,
            index=step,
        )

    # Solve Darcy flow
    darcy.solve(zero_init_guess=True)

    dt = ndim(0.0002 * u.year)

    # Advect swarm (only in x-direction, vertical velocity negligible)
    swarm.advection(
        V_fn=v_soln.sym * sympy.Matrix.diag(1 / porosity, 1 / sympy.sympify(1000000000)),
        delta_t=dt,
        order=2,
        evalf=True,
    )

    # Compute Vrms
    I = uw.maths.Integral(mesh, sympy.sqrt(v_soln.sym.dot(v_soln.sym)))
    Vrms = I.evaluate()
    I.fn = 1.0
    Vrms /= I.evaluate()

    if uw.mpi.rank == 0:
        print(f"V_rms = {Vrms} ... delta t = {dt}.  dL = {Vrms * dt}")

    step += 1
    time += dt

    if time > nd(finish_time):
        break

# %% [markdown]
"""
## Final Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(vizmesh)
    pvmesh.point_data["mat"] = vis.scalar_fn_to_pv_points(pvmesh, material.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = (
        vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
        / vis.vector_fn_to_pv_points(velocity_points, v_soln.sym).max()
    )

    points = vis.swarm_to_pv_cloud(swarm)
    point_cloud = pv.PolyData(points)

    with swarm.access(material):
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        style="surface",
        cmap="coolwarm",
        edge_color="Grey",
        scalars="P",
        show_edges=False,
        use_transparency=False,
        opacity=1,
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1250, opacity=1)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=2,
        opacity=0.66,
    )

    pl.show(cpos="xy")

# %%
mat.stats()
