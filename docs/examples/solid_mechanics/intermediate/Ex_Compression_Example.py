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
# Compression with Plastic Yielding

**PHYSICS:** solid_mechanics
**DIFFICULTY:** intermediate

## Description

Compression of a domain with a rigid circular inclusion and Drucker-Prager
plasticity. This demonstrates shear band development around a stress
concentrator.

## Key Concepts

- **Drucker-Prager plasticity**: Pressure-dependent yield criterion
- **Strain localization**: Shear bands form at high strain rate regions
- **Rigid inclusion**: Stress concentrator triggers localization
- **Nonlinear rheology**: Viscosity depends on yield stress and strain rate

## Mathematical Formulation

Drucker-Prager yield criterion:
$$\\tau_y = C + \\mu \\cdot p$$

Effective viscosity:
$$\\eta = \\frac{\\tau_y}{2 \\dot\\varepsilon_{II}}$$

## Parameters

- `uw_cohesion`: Cohesion (yield stress at zero pressure)
- `uw_friction`: Friction coefficient (pressure dependence)
- `uw_resolution`: Mesh resolution
- `uw_inclusion_radius`: Radius of rigid inclusion
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
import gmsh
from enum import Enum

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Compression_Example.py -uw_cohesion 0.05
python Ex_Compression_Example.py -uw_friction 0.5
```
"""

# %%
params = uw.Params(
    uw_cohesion = 0.1,              # Cohesion C
    uw_friction = 0.1,              # Friction coefficient mu
    uw_resolution = 0.33,           # Base mesh cell size
    uw_resolution_inclusion = 0.02, # Cell size near inclusion
    uw_width = 2.0,                 # Domain half-width
    uw_height = 1.0,                # Domain height
    uw_inclusion_radius = 0.25,     # Inclusion radius
)

# Derived name for output
expt_name = f"Compression_C{params.uw_cohesion}_mu{params.uw_friction}"

# %% [markdown]
"""
## Mesh Generation

Create a mesh with a semicircular inclusion at the bottom, refined near
the inclusion surface.
"""

# %%
class boundaries(Enum):
    Left = 1
    Right = 2
    Top = 3
    FlatBottom = 4
    Hump = 5
    Elements = 6
    All_Boundaries = 1001


if uw.mpi.rank == 0:
    gmsh.initialize()
    gmsh.model.add("Compression")
    gmsh.model.geo.characteristic_length_max = params.uw_resolution

    # Geometry parameters
    width = params.uw_width
    height = params.uw_height
    radius = params.uw_inclusion_radius
    csize = params.uw_resolution
    csize_inc = params.uw_resolution_inclusion

    # Points for inclusion
    c0 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, csize_inc)
    cr1 = gmsh.model.geo.add_point(-radius, 0.0, 0.0, csize_inc)
    cr2 = gmsh.model.geo.add_point(0.0, radius, 0.0, csize_inc)
    cr3 = gmsh.model.geo.add_point(+radius, 0.0, 0.0, csize_inc)

    # Corner points
    cp1 = gmsh.model.geo.add_point(-width, 0.0, 0.0, csize)
    cp2 = gmsh.model.geo.add_point(+width, 0.0, 0.0, csize)
    cp3 = gmsh.model.geo.add_point(+width, height, 0.0, csize)
    cp4 = gmsh.model.geo.add_point(-width, height, 0.0, csize)

    # Lines
    l1 = gmsh.model.geo.add_line(cr3, cp2)
    l2 = gmsh.model.geo.add_line(cp2, cp3)
    l3 = gmsh.model.geo.add_line(cp3, cp4)
    l4 = gmsh.model.geo.add_line(cp4, cp1)
    l5 = gmsh.model.geo.add_line(cp1, cr1)

    # Semicircular arc for inclusion
    l6 = gmsh.model.geo.add_circle_arc(cr1, c0, cr2)
    l7 = gmsh.model.geo.add_circle_arc(cr2, c0, cr3)

    # Create surface
    cl1 = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4, l5, l6, l7])
    surf1 = gmsh.model.geo.add_plane_surface([cl1])

    gmsh.model.geo.synchronize()

    # Physical groups for boundaries
    gmsh.model.add_physical_group(1, [l4], -1, name="Left")
    gmsh.model.add_physical_group(1, [l2], -1, name="Right")
    gmsh.model.add_physical_group(1, [l3], -1, name="Top")
    gmsh.model.add_physical_group(1, [l1, l5], -1, name="FlatBottom")
    gmsh.model.add_physical_group(1, [l6, l7], -1, name="Hump")
    gmsh.model.add_physical_group(2, [surf1], -1, name="Elements")

    gmsh.model.mesh.generate(2)
    gmsh.write("tmp_compression.msh")
    gmsh.finalize()

# %% [markdown]
"""
## Load Mesh
"""

# %%
mesh1 = uw.discretisation.Mesh(
    "tmp_compression.msh",
    boundaries=boundaries,
    useRegions=True,
    simplex=True,
)

mesh1.dm.view()

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)

x, y = mesh1.X

# %% [markdown]
"""
## Stokes Solver with Plasticity
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
stokes.penalty = 0.1

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_atol"] = 1.0e-4
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "cg"
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "mg"
stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "mg"

# %% [markdown]
"""
## Projection Solvers

For computing and visualizing strain rate, stress, and viscosity fields.
"""

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh1, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 0.0
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2)
S = stokes.stress_deviator
nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace()) / 2)) - p_soln.sym[0]
nodal_tau_inv2.smoothing = 0.0
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

# %% [markdown]
"""
## Boundary Conditions

- Left/Right: Horizontal velocity boundary conditions (compression)
- Bottom (flat): No vertical motion
- Hump (inclusion): Fixed (rigid inclusion)
"""

# %%
# Body force (gravity)
stokes.bodyforce = -10 * mesh1.CoordinateSystem.unit_j

# Velocity BCs
stokes.add_dirichlet_bc((+1.0, 0.0), "Left")
stokes.add_dirichlet_bc((-1.0, 0.0), "Right")
stokes.add_dirichlet_bc((None, 0.0), "FlatBottom")
stokes.add_dirichlet_bc((0.0, 0.0), "Hump")

# %% [markdown]
"""
## Initial Linear Solve
"""

# %%
stokes.solve(zero_init_guess=False)

# %% [markdown]
"""
## Enable Plastic Rheology

Drucker-Prager yield criterion:
$$\\tau_y = C + \\mu \\cdot p$$
"""

# %%
mu = params.uw_friction
C = params.uw_cohesion

# Yield stress (pressure-dependent)
tau_y = sympy.Max(C + mu * stokes.p.sym[0], 0.0001)

# Effective viscosity from yield stress
viscosity = 1.0 / (2 * stokes.Unknowns.Einv2 / tau_y)

stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.solve(zero_init_guess=False)

print(f"Plastic solve complete: C={C}, mu={mu}")

# %% [markdown]
"""
## Post-Processing
"""

# %%
nodal_tau_inv2.uw_function = 2 * stokes.constitutive_model.Parameters.viscosity * stokes.Unknowns.Einv2
nodal_tau_inv2.solve()

nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()

# Save output
mesh1.petsc_save_checkpoint(
    index=0,
    meshVars=[v_soln, p_soln, dev_stress_inv2, strain_rate_inv2, node_viscosity],
    outputPath="./output/",
)

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

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.1,
        opacity=0.75,
    )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Str",
        use_transparency=False,
        opacity=1.0,
    )

    pl.show()

# %%
print(f"Compression example complete: {expt_name}")
