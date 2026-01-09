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
# Sheared Layer Test (Constitutive Model Validation)

**PHYSICS:** solid_mechanics
**DIFFICULTY:** advanced
**STATUS:** Development/Testing

## Description

Validation of constitutive models through simple shear testing. Uses a particle
swarm to define materials and track strain accumulation. Tests implementation
of Jacobians using various non-linear terms including viscoelasticity and
plasticity.

## Key Concepts

- **Simple shear**: Velocity boundary conditions driving shear flow
- **Material tracking**: Particle swarm for material properties
- **Strain softening**: Yield stress evolution with accumulated strain
- **Viscoelastic constitutive models**: ViscoElasticPlasticFlowModel

## Parameters

- `uw_resolution`: Mesh resolution (default: 0.033)
- `uw_mu`: Friction coefficient for pressure-dependent yield (default: 0.5)
- `uw_max_steps`: Maximum timesteps (default: 500)

## Notes

This is a development/testing notebook. Contains debug breakpoints to allow
incremental execution during development.
"""

# %% [markdown]
"""
## Setup
"""

# %%
# Fix trame async issue
import nest_asyncio
nest_asyncio.apply()

import os
os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import pyvista as pv
import vtk

from underworld3 import timing

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Sheared_Layer_Test.py -uw_resolution 0.05
python Ex_Sheared_Layer_Test.py -uw_mu 0.3
```
"""

# %%
params = uw.Params(
    uw_resolution = 0.033,        # Mesh resolution
    uw_mu = 0.5,                  # Friction coefficient
    uw_max_steps = 500,           # Maximum timesteps
)

resolution = params.uw_resolution
mu = params.uw_mu
maxsteps = int(params.uw_max_steps)

# %% [markdown]
"""
## Mesh Generation

Create a 2D shear box mesh with optional circular inclusion.
"""

# %%
csize = resolution
csize_circle = resolution * 0.5
res = csize
cellSize = csize

width = 3.0
height = 1.0
radius = 0.0  # Set > 0 for inclusion

eta1 = 1000
eta2 = 1

from enum import Enum


class boundaries(Enum):
    Bottom = 1
    Right = 3
    Top = 2
    Left = 4
    Inclusion = 5
    All_Boundaries = 1001


if uw.mpi.rank == 0:

    import gmsh

    gmsh.initialize()
    gmsh.model.add("Periodic x")

    xmin, ymin = -width / 2, -height / 2
    xmax, ymax = +width / 2, +height / 2

    p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
    p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
    p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
    p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

    l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries.Bottom.value)
    l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries.Right.value)
    l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries.Top.value)
    l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries.Left.value)

    loops = []
    if radius > 0.0:
        p5 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=csize_circle)
        p6 = gmsh.model.geo.add_point(+radius, 0.0, 0.0, meshSize=csize_circle)
        p7 = gmsh.model.geo.add_point(-radius, 0.0, 0.0, meshSize=csize_circle)

        c1 = gmsh.model.geo.add_circle_arc(p6, p5, p7)
        c2 = gmsh.model.geo.add_circle_arc(p7, p5, p6)

        cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=55)
        loops = [cl1] + loops

    cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
    loops = [cl] + loops

    surface = gmsh.model.geo.add_plane_surface(loops, tag=99999)

    gmsh.model.geo.synchronize()

    # Add Physical groups
    for bd in boundaries:
        if bd.value < 1000:  # Skip All_Boundaries
            gmsh.model.add_physical_group(1, [bd.value], bd.value)
            gmsh.model.set_physical_name(1, bd.value, bd.name)

    if radius > 0.0:
        gmsh.model.addPhysicalGroup(1, [c1, c2], 55)
        gmsh.model.setPhysicalName(1, 55, "Inclusion")

    gmsh.model.addPhysicalGroup(2, [surface], surface)
    gmsh.model.setPhysicalName(2, surface, "Elements")

    gmsh.model.mesh.generate(2)
    gmsh.write("tmp_shear_inclusion.msh")
    gmsh.finalize()

# %%
mesh1 = uw.discretisation.Mesh(
    "tmp_shear_inclusion.msh",
    simplex=True,
    markVertices=True,
    useRegions=True,
    boundaries=boundaries,
)

mesh1.view()

# %%
bmask = mesh1.meshVariable_mask_from_label("Bottom", 1)
tmask = mesh1.meshVariable_mask_from_label("UW_Boundaries", mesh1.boundaries.Top.value)

# %%
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["MB"] = vis.scalar_fn_to_pv_points(pvmesh, bmask.sym)
    pvmesh.point_data["MT"] = vis.scalar_fn_to_pv_points(pvmesh, tmask.sym)

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        scalars="MT",
        use_transparency=False,
        opacity=0.5,
    )

    pl.show()

# %% [markdown]
"""
## WIP: Swarm and Variables

The following sections are work-in-progress. Debug breakpoint below
allows mesh visualization before continuing.
"""

# %%
# Development pause - mesh is set up, swarm/solve follows
uw.pause("Mesh visualization complete", explanation="Run next cell to continue with swarm setup")

# %% [markdown]
"""
## Swarm and Material Variables
"""

# %%
swarm = uw.swarm.Swarm(mesh=mesh1, recycle_rate=5)

material = uw.swarm.SwarmVariable(
    "M",
    swarm,
    size=1,
    proxy_continuous=True,
    proxy_degree=2,
    dtype=int,
)

# Strain tracked on particles
strain = uw.swarm.SwarmVariable(
    "Strain",
    swarm,
    size=1,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{\varepsilon_{p}}",
    dtype=float,
)

stress_dt = uw.swarm.SwarmVariable(
    r"Stress_p",
    swarm,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    varsymbol=r"{\sigma^{*}_{p}}",
)

swarm.populate(fill_param=2)

# %% [markdown]
"""
## Mesh Variables
"""

# %%
# Coordinate system
x, y = mesh1.X

# Relative to centre for inclusion
r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y, x)

inclusion_rvec = mesh1.X
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)
inclusion_unit_rvec = mesh1.vector.to_matrix(inclusion_unit_rvec)

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)
work = uw.discretisation.MeshVariable(r"W", mesh1, 1, degree=1, continuous=False)
Stress = uw.discretisation.MeshVariable(
    r"{\sigma}",
    mesh1,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=1,
    continuous=False,
    varsymbol=r"{\sigma}",
)

vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=2)
strain_rate_inv2_p = uw.discretisation.MeshVariable(
    "eps_p", mesh1, 1, degree=2, varsymbol=r"\dot\varepsilon_p"
)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=2)
yield_stress = uw.discretisation.MeshVariable("tau_y", mesh1, 1, degree=1)

node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)

mesh1.view()

# %% [markdown]
"""
## WIP: Stokes Setup

Debug breakpoint for variable inspection.
"""

# %%
uw.pause("Variables defined", explanation="Run next cell to continue with Stokes setup")

# %% [markdown]
"""
## Initialize Strain
"""

# %%
# TODO: Consider uw.synchronised_array_update() for multi-variable assignment
XX = swarm._particle_coordinates.data[:, 0]
YY = swarm._particle_coordinates.data[:, 1]
mask = (1.0 - (YY * 2) ** 8) * (1 - (2 * XX / 3) ** 6)
material.data[(XX**2 + YY**2 < 0.01), 0] = 1
strain.data[:, 0] = 0.0

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=True,
)

eta1 = 1000
eta2 = 1

viscosity_L = sympy.Piecewise((eta2, material.sym[0] > 0.5), (eta1, True))

# %%
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.bg_viscosity = viscosity_L

stokes.constitutive_model.Parameters.shear_modulus = 1.0
stokes.constitutive_model.Parameters.dt_elastic = 0.1

# %%
sigma_projector = uw.systems.Tensor_Projection(mesh1, tensor_Field=Stress, scalar_Field=work)
sigma_projector.uw_function = stokes.stress_1d

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh1, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2)
nodal_tau_inv2.uw_function = 2 * stokes.constitutive_model.viscosity * stokes.Unknowns.Einv2
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

yield_stress_calc = uw.systems.Projection(mesh1, yield_stress)
yield_stress_calc.uw_function = 0.0
yield_stress_calc.smoothing = 1.0e-3
yield_stress_calc.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

# %% [markdown]
"""
## Boundary Conditions
"""

# %%
stokes.penalty = 1.0
stokes.bodyforce = -0.00000001 * mesh1.CoordinateSystem.unit_e_1.T

stokes.tolerance = 1.0e-4

if radius > 0.0:
    stokes.add_dirichlet_bc((0.0, 0.0), "Inclusion")

stokes.add_dirichlet_bc((1.0, 0.0), "Top")
stokes.add_dirichlet_bc((-1.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")

# %% [markdown]
"""
## WIP: Plasticity and Timestepping

Debug breakpoint before adding plasticity.
"""

# %%
uw.pause("Stokes setup complete", explanation="Run next cell to continue with plasticity")

# %% [markdown]
"""
## Add Plasticity
"""

# %%
eps_ref = sympy.sympify(1)
scale = sympy.sympify(25)
C0 = 2500
Cinf = 500

C = 2 * (y * 2) ** 2 + (C0 - Cinf) * (1 - sympy.tanh((strain.sym[0] / eps_ref - 1) * scale)) / 2 + Cinf

stokes.constitutive_model.Parameters.yield_stress = C + mu * p_soln.sym[0]
stokes.constitutive_model.Parameters.edot_II_fn = stokes.Unknowns.Einv2
stokes.constitutive_model.Parameters.min_viscosity = 0.1
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.viscosity

# %% [markdown]
"""
## Solve
"""

# %%
mesh1.dm.view()

stokes._setup_pointwise_functions()
stokes._setup_discretisation(verbose=True)
stokes._setup_solver()

stokes.solve()

# %%
sigma_projector.solve()

print(f"Stress[0,0] max: {Stress[0, 0].data.max()}")
print(f"Stress[1,1] max: {Stress[1, 1].data.max()}")
print(f"Stress[0,1] max: {Stress[0, 1].data.max()}")

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints) ** 2

    pl = pv.Plotter(window_size=(500, 500))

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=0.5,
    )

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()

# %%
print("Sheared layer test complete")
