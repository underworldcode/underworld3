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
# Shear Band Plasticity - Pure Shear

**PHYSICS:** solid_mechanics
**DIFFICULTY:** advanced

## Description

Flow and shear banding around a circular inclusion in pure shear. Based on
Masuda & Mizuno (1995) for deflection of pure shear viscous flow around a
rigid spherical body.

## Key Concepts

- **Pure shear deformation**: Extensional flow field
- **Plastic yielding**: Drucker-Prager type yield criterion
- **Shear band localization**: Strain rate concentration
- **Rigid inclusion**: Free-slip on circular boundary
- **pygmsh meshing**: Custom mesh with circular hole

## References

Masuda, T., & Mizuno, N. (1995). Deflection of pure shear viscous flow around
a rigid spherical body. Journal of Structural Geology, 17(11), 1615-1620.

## Parameters

- `uw_cell_size`: Background mesh cell size
- `uw_cell_size_circle`: Cell size near inclusion
- `uw_radius`: Inclusion radius
- `uw_cohesion`: Yield stress cohesion (C)
- `uw_friction`: Friction coefficient (mu)
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import numpy as np
import petsc4py
import underworld3 as uw
import sympy
import pygmsh
from enum import Enum

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Shear_Band_Plasticity_PS.py -uw_cohesion 2.0
python Ex_Shear_Band_Plasticity_PS.py -uw_friction 0.5
python Ex_Shear_Band_Plasticity_PS.py -uw_cell_size 0.05
```
"""

# %%
params = uw.Params(
    uw_cell_size = 0.075,            # Background cell size
    uw_cell_size_circle = 0.025,     # Cell size near inclusion
    uw_width = 1.0,                  # Domain half-width
    uw_height = 1.0,                 # Domain half-height
    uw_radius = 0.2,                 # Inclusion radius
    uw_cohesion = 2.5,               # Yield cohesion C
    uw_friction = 0.25,              # Friction coefficient mu
    uw_n_iterations = 1,             # Number of nonlinear iterations
)

expt_name = "PS_ShearBand"

# %% [markdown]
"""
## Mesh Generation

Create mesh with circular inclusion using pygmsh.
"""

# %%
class boundaries_2D(Enum):
    bottom = 1
    top = 3
    right = 2
    left = 4
    inclusion = 5


if uw.mpi.rank == 0:
    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = params.uw_cell_size

        inclusion = geom.add_circle(
            (0.0, 0.0, 0.0),
            params.uw_radius,
            make_surface=False,
            mesh_size=params.uw_cell_size_circle,
        )
        domain = geom.add_rectangle(
            xmin=-params.uw_width,
            ymin=-params.uw_height,
            xmax=params.uw_width,
            ymax=params.uw_height,
            z=0,
            holes=[inclusion],
            mesh_size=params.uw_cell_size,
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
        geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
        geom.add_physical(domain.surface.curve_loop.curves[2], label="top")
        geom.add_physical(domain.surface.curve_loop.curves[3], label="left")
        geom.add_physical(inclusion.curve_loop.curves, label="inclusion")
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry("tmp_ps_shear_inclusion.msh")

# %%
mesh1 = uw.discretisation.Mesh(
    "tmp_ps_shear_inclusion.msh",
    markVertices=True,
    useRegions=True,
    simplex=True,
    boundaries=boundaries_2D,
)
mesh1.dm.view()

# %% [markdown]
"""
## Coordinate Functions
"""

# %%
x, y = mesh1.X

# Relative to inclusion center
r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y, x)

# Unit radial vector for inclusion
inclusion_rvec = mesh1.X
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)
inclusion_unit_rvec = mesh1.vector.to_matrix(inclusion_unit_rvec)

# Pure shear velocity field
vx_ps = mesh1.N.x
vy_ps = -mesh1.N.y

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
t_soln = uw.discretisation.MeshVariable("T", mesh1, 1, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1)

vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.penalty = 0.0

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None

# %% [markdown]
"""
## Initial Velocity Field
"""

# %%
v_soln.array[...] = uw.function.evaluate(
    sympy.Matrix(((vx_ps, vy_ps))), v_soln.coords
)

# %% [markdown]
"""
## Projection Solvers

For computing strain rate, stress, and viscosity fields.
"""

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh1, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2)
S = stokes.stress_deviator
nodal_tau_inv2.uw_function = (
    sympy.simplify(sympy.sqrt(((S**2).trace()) / 2)) - p_soln.sym[0]
)
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

# %% [markdown]
"""
## Boundary Conditions

Pure shear on all boundaries, free-slip on inclusion.
"""

# %%
res = params.uw_cell_size_circle
hw = 1000.0 / res
surface_defn_fn = sympy.exp(-(((r - params.uw_radius) / params.uw_radius) ** 2) * hw)

stokes.bodyforce = mesh1.CoordinateSystem.unit_e_0 * 1.0e-5

# Natural BC for free-slip on inclusion
Gamma = mesh1.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) * Gamma, "inclusion")

# Pure shear velocity on all boundaries
stokes.add_dirichlet_bc((vx_ps, vy_ps), "top")
stokes.add_dirichlet_bc((vx_ps, vy_ps), "bottom")
stokes.add_dirichlet_bc((vx_ps, vy_ps), "left")
stokes.add_dirichlet_bc((vx_ps, vy_ps), "right")

stokes.penalty = 0.1

# %% [markdown]
"""
## Linear Solve
"""

# %%
stokes.solve(zero_init_guess=False)

# %% [markdown]
"""
## Nonlinear Plastic Rheology

Drucker-Prager type yield criterion:
$$\\tau_y = C + \\mu \\cdot p$$
"""

# %%
for i in range(int(params.uw_n_iterations)):
    mu = params.uw_friction
    C = params.uw_cohesion

    print(f"Iteration {i+1}: mu = {mu}, C = {C}")

    tau_y = sympy.Max(C + mu * stokes.p.sym[0], 0.1)
    viscosity = sympy.Min(tau_y / (2 * stokes.Unknowns.Einv2 + 0.01), 1.0)

    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
    stokes.solve(zero_init_guess=False)

# %% [markdown]
"""
## Post-Processing
"""

# %%
nodal_tau_inv2.uw_function = (
    stokes.constitutive_model.Parameters.shear_viscosity_0 * stokes.Unknowns.Einv2
)
nodal_tau_inv2.solve()

nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_inv2.sym)
    pvmesh.point_data["Visc"] = vis.scalar_fn_to_pv_points(pvmesh, node_viscosity.sym)
    pvmesh.point_data["Str"] = vis.scalar_fn_to_pv_points(pvmesh, dev_stress_inv2.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(
        pvmesh, v_soln.sym.dot(v_soln.sym)
    )

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )

    # Point sources at cell centres
    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="both", max_steps=100
    )

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_arrows(
        velocity_points.points, velocity_points.point_data["V"], mag=0.1, opacity=0.25
    )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Edot",
        use_transparency=False,
        opacity=1.0,
    )

    pl.show()

# %%
print(f"Shear band plasticity (PS) example complete: {expt_name}")
