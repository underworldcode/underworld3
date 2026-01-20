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
# Viscoelastic Shearing with Inclusion

**PHYSICS:** solid_mechanics
**DIFFICULTY:** advanced

## Description

Simple shear of a viscoelastic material containing a circular inclusion.
Demonstrates the VE_Stokes solver with elastic stress accumulation and
the ViscoElasticPlasticFlowModel constitutive model.

## Key Concepts

- **VE_Stokes solver**: Viscoelastic extension of Stokes equations
- **Elastic stress**: Accumulation over timescale dt_elastic
- **ViscoElasticPlasticFlowModel**: Combined viscous, elastic, plastic rheology
- **Stress tensor projection**: Full stress tensor on mesh
- **Circular inclusion**: Rigid body embedded in deforming matrix

## Mathematical Formulation

Maxwell viscoelastic rheology:
$$\\dot{\\varepsilon} = \\frac{\\dot{\\sigma}}{2G} + \\frac{\\sigma}{2\\eta}$$

where G is shear modulus and eta is viscosity.

## Parameters

- `uw_resolution`: Mesh cell size
- `uw_mu`: Shear modulus parameter
- `uw_max_steps`: Maximum time steps
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import os

os.environ["UW_TIMING_ENABLE"] = "1"

import numpy as np
import petsc4py
import sympy
import underworld3 as uw
from underworld3 import timing

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Sheared_Layer_Elastic.py -uw_resolution 0.05
python Ex_Sheared_Layer_Elastic.py -uw_mu 1.0
python Ex_Sheared_Layer_Elastic.py -uw_max_steps 200
```
"""

# %%
params = uw.Params(
    uw_resolution = 0.05,         # Mesh cell size
    uw_mu = 0.5,                  # Shear modulus
    uw_max_steps = 100,           # Maximum time steps
    uw_width = 3.0,               # Domain width
    uw_height = 1.0,              # Domain height
    uw_radius = 0.1,              # Inclusion radius
    uw_shear_viscosity = 1.0,     # Background viscosity
    uw_shear_modulus = 10.0,      # Shear modulus
)

# Time observation scale
observation_timescale = 0.0033

# %% [markdown]
"""
## Mesh Generation with pygmsh

Domain with circular inclusion (rigid body).
"""

# %%
from enum import Enum
import pygmsh

csize = 0.075
csize_circle = 0.025
res = csize_circle

width = params.uw_width
height = params.uw_height
radius = params.uw_radius


class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left = 4
    inclusion = 5
    All_Boundaries = 1001


if uw.mpi.rank == 0:
    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (0.0, 0.0, 0.0), radius, make_surface=False, mesh_size=csize_circle
        )
        domain = geom.add_rectangle(
            xmin=-width / 2,
            ymin=-height / 2,
            xmax=width / 2,
            ymax=height / 2,
            z=0,
            holes=[inclusion],
            mesh_size=csize,
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label=boundaries.bottom.name)
        geom.add_physical(domain.surface.curve_loop.curves[1], label=boundaries.right.name)
        geom.add_physical(domain.surface.curve_loop.curves[2], label=boundaries.top.name)
        geom.add_physical(domain.surface.curve_loop.curves[3], label=boundaries.left.name)
        geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry("tmp_shear_inclusion.msh")


def mesh_return_coords_to_bounds(coords):
    """Restore particles that escape domain bounds."""
    lefty_troublemakers = coords[:, 0] < -width / 2
    righty_troublemakers = coords[:, 0] > width / 2
    coords[lefty_troublemakers, 0] = -width / 2 + 0.0001
    coords[righty_troublemakers, 0] = width / 2 - 0.0001

    return coords


# %%
mesh1 = uw.discretisation.Mesh(
    "tmp_shear_inclusion.msh",
    markVertices=True,
    useRegions=True,
    refinement=0,
    return_coords_to_bounds=mesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3,
)

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(
    "P", mesh1, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=True
)
Stress = uw.discretisation.MeshVariable(
    "Stress",
    mesh1,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=2,
    continuous=True,
    varsymbol=r"{\sigma}",
)
work = uw.discretisation.MeshVariable(
    "W", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, continuous=True
)
strain_rate_inv2 = uw.discretisation.MeshVariable(
    "eps_dot", mesh1, 1, degree=2, varsymbol=r"{\dot\varepsilon}"
)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=2)

x, y = mesh1.X

# %% [markdown]
"""
## VE_Stokes Solver with Viscoelastic Rheology
"""

# %%
stokes = uw.systems.VE_Stokes(
    mesh1, velocityField=v_soln, pressureField=p_soln, verbose=False, order=1
)

stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None

# Viscoelastic-plastic constitutive model
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel

stokes.constitutive_model.Parameters.shear_viscosity_0 = params.uw_shear_viscosity
stokes.constitutive_model.Parameters.shear_modulus = params.uw_shear_modulus
stokes.constitutive_model.Parameters.dt_elastic = sympy.sympify(observation_timescale)

# %% [markdown]
"""
## Stress Projection
"""

# %%
sigma_projector = uw.systems.Tensor_Projection(
    mesh1, tensor_Field=Stress, scalar_Field=work
)
sigma_projector.uw_function = stokes.stress

# %% [markdown]
"""
## Strain Rate and Stress Projections
"""

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh1, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2)
nodal_tau_inv2.uw_function = (
    2 * stokes.constitutive_model.viscosity * stokes.Unknowns.Einv2
)
nodal_tau_inv2.smoothing = 1.0e-3

# %% [markdown]
"""
## Boundary Conditions

Shearing motion applied at top and bottom boundaries.
Inclusion is fixed (rigid body).
"""

# %%
stokes.penalty = 1.0
stokes.tolerance = 1.0e-4

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0), "inclusion")
stokes.add_dirichlet_bc((1.0, 0.0), "top")
stokes.add_dirichlet_bc((-1.0, 0.0), "bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "right")

# %% [markdown]
"""
## Initial Solve
"""

# %%
nodal_strain_rate_inv2.solve()
sigma_projector.uw_function = stokes.stress_deviator
sigma_projector.solve()

timing.reset()
timing.start()

stokes.solve(zero_init_guess=False, evalf=False)
timing.print_table(display_fraction=1)
print(f"Max velocity: {stokes.Unknowns.u.max()}, Max pressure: {stokes.Unknowns.p.max()}")

# %% [markdown]
"""
## Update Projections
"""

# %%
stokes.solve(zero_init_guess=False, verbose=False, evalf=False)
timing.print_table(display_fraction=1)
print(f"Max velocity: {stokes.Unknowns.u.max()}, Max pressure: {stokes.Unknowns.p.max()}")

nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.solve()

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = (
    stokes.constitutive_model.viscosity * 2 * stokes.Unknowns.Einv2
)
nodal_tau_inv2.solve()

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0])
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_inv2.sym[0])
    pvmesh.point_data["Strs"] = vis.scalar_fn_to_pv_points(pvmesh, dev_stress_inv2.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.1,
        opacity=1,
        show_scalar_bar=False,
    )

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        scalars="Strs",
        use_transparency=False,
        opacity=0.5,
    )

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()

# %% [markdown]
"""
## Time Evolution
"""

# %%
ts = 0
time = 0.0
delta_t = stokes.delta_t
maxsteps = int(params.uw_max_steps)

expt_name = f"shear_band_ve_{params.uw_mu}"

for step in range(0, maxsteps):
    stokes.solve(zero_init_guess=False, evalf=False)

    nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
    nodal_strain_rate_inv2.solve()

    S = stokes.stress_deviator
    nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace()) / 2))
    nodal_tau_inv2.solve()

    uw.pprint(f"Stress Inv II - {dev_stress_inv2.mean()}")

    mesh1.write_timestep(
        expt_name,
        meshUpdates=False,
        meshVars=[p_soln, v_soln],
        outputPath="output",
        index=ts,
    )

    uw.pprint(f"Timestep {step}, dt {delta_t:.4e}")

    ts += 1
    time += delta_t

# %% [markdown]
"""
## Final Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0])
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_inv2.sym[0])
    pvmesh.point_data["Strs"] = vis.scalar_fn_to_pv_points(pvmesh, dev_stress_inv2.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.1,
        opacity=1,
        show_scalar_bar=False,
    )

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        scalars="Strs",
        use_transparency=False,
        opacity=0.5,
    )

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()

# %%
print(f"Viscoelastic shearing example complete: {ts} steps")
