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
# Stokes Benchmark SolCx

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

The SolCx benchmark tests the Stokes solver with a sharp viscosity contrast.
A vertical step in viscosity at x = 0.5 creates a challenging test case for
iterative solvers. Compares Dirichlet and natural boundary conditions.

## Key Concepts

- **Viscosity jump**: Step function viscosity contrast (10^6)
- **Benchmark validation**: Comparison with analytical solution (if UW2 available)
- **Natural vs Dirichlet BCs**: Free-slip implemented different ways
- **Piecewise functions**: Using sympy.Piecewise for sharp interfaces
- **Multigrid preconditioning**: Essential for high viscosity contrast

## Mathematical Formulation

Viscosity step function:
$$\\eta(x) = \\begin{cases} 10^6 & x > 0.5 \\\\ 1 & x \\le 0.5 \\end{cases}$$

Buoyancy forcing:
$$f_y = -\\cos(\\pi x) \\sin(2\\pi y)$$

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_refinement`: Mesh refinement level
- `uw_viscosity_contrast`: log10 of viscosity contrast
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import petsc4py
from petsc4py import PETSc

import nest_asyncio
nest_asyncio.apply()

import os
os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import timing

import numpy as np
import sympy
from sympy import Piecewise

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Cartesian_SolC.py -uw_resolution 8
python Ex_Stokes_Cartesian_SolC.py -uw_viscosity_contrast 4
```
"""

# %%
params = uw.Params(
    uw_resolution = 4,              # Base mesh resolution
    uw_refinement = 2,              # Mesh refinement levels
    uw_viscosity_contrast = 6,      # log10 of viscosity contrast
    uw_use_simplex = 1,             # Use simplex mesh (1) or quad (0)
    uw_penalty = 100,               # Stokes penalty parameter
)

# Derived parameters
eta_ratio = 10 ** params.uw_viscosity_contrast
use_simplex = bool(params.uw_use_simplex)

# %% [markdown]
"""
## Mesh Generation
"""

# %%
n_els = int(params.uw_resolution)
refinement = int(params.uw_refinement)

if use_simplex:
    mesh = uw.meshing.UnstructuredSimplexBox(
        regular=True,
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1 / n_els,
        qdegree=3,
        refinement=refinement,
    )
else:
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(n_els, n_els),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        qdegree=3,
        refinement=refinement,
    )

x, y = mesh.X

# %% [markdown]
"""
## Variables
"""

# %%
v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR, degree=3, varsymbol=r"{v}")
p = uw.discretisation.MeshVariable(
    "P", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=False, varsymbol=r"{p}"
)

# Clone for storing different solutions
v0 = v.clone("v0", r"{v_0}")
v1 = v0.clone("v1", r"{v_1}")

# %% [markdown]
"""
## Stokes Solver - Initial Setup
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=False)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

# %% [markdown]
"""
## Test Case 1: Simple Piecewise Body Force

Verify solver with a simple step function forcing.
"""

# %%
eta_0 = 1
x_c = sympy.Rational(1, 2)
f_0 = 1

stokes.penalty = params.uw_penalty
stokes.bodyforce = sympy.Matrix(
    [
        0,
        Piecewise(
            (f_0, x > x_c),
            (0.0, True),
        ),
    ]
)

# Free-slip boundary conditions (Dirichlet form)
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

stokes.tolerance = 1e-6

# %% [markdown]
"""
## Solver Configuration
"""

# %%
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %%
stokes.solve()

# %% [markdown]
"""
## SolCx Benchmark Configuration

Step viscosity at x = 0.5 with harmonic forcing.
"""

# %%
stokes.bodyforce = sympy.Matrix(
    [0, -sympy.cos(sympy.pi * x) * sympy.sin(2 * sympy.pi * y)]
)

viscosity_fn = sympy.Piecewise(
    (eta_ratio, x > x_c),
    (1.0, True),
)

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn

# %%
timing.reset()
timing.start()
stokes.solve(zero_init_guess=True)
timing.print_table(display_fraction=0.999)

# Save solution with Dirichlet BCs
v0.data[...] = v.data[...]

# %% [markdown]
"""
## Alternative: Natural Boundary Conditions

Compare with free-slip using natural (Neumann) BCs.
"""

# %%
stokes._reset()
stokes.tolerance = 1.0e-6

# Free-slip via penalty on normal velocity
stokes.add_natural_bc([0.0, 1e6 * v.sym[1]], "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

timing.reset()
timing.start()
stokes.solve()
timing.print_table(display_fraction=0.999)

v1.data[...] = v.data[...]

# %% [markdown]
"""
## Alternative: Using Mesh Gamma for Normal Vector
"""

# %%
stokes._reset()
stokes.tolerance = 1.0e-6

Gamma = mesh.Gamma
stokes.add_natural_bc(1e6 * Gamma.dot(v.sym) * Gamma, "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

timing.reset()
timing.start()
stokes.solve()
timing.print_table(display_fraction=0.999)

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v0.sym.dot(v0.sym))
    pvmesh.point_data["Visc"] = vis.scalar_fn_to_pv_points(
        pvmesh, stokes.constitutive_model.Parameters.shear_viscosity_0
    )

    pvmesh.point_data["V0"] = vis.vector_fn_to_pv_points(
        pvmesh, v0.sym * stokes.constitutive_model.viscosity
    )
    pvmesh.point_data["V1"] = vis.vector_fn_to_pv_points(
        pvmesh, v1.sym * stokes.constitutive_model.viscosity
    )
    pvmesh.point_data["V2"] = vis.vector_fn_to_pv_points(
        pvmesh, v.sym * stokes.constitutive_model.viscosity
    )
    pvmesh.point_data["dV"] = pvmesh.point_data["V1"] - pvmesh.point_data["V0"]

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=100.0,
        opacity=1,
        show_scalar_bar=False,
    )

    pl.show(cpos="xy")

# %% [markdown]
"""
## Validation Against UW2 (if available)
"""

# %%
try:
    import underworld as uw2

    solC = uw2.function.analytic.SolC()
    vel_soln_analytic = solC.fn_velocity.evaluate(mesh.X.coords)
    from mpi4py import MPI
    from numpy import linalg as LA

    comm = MPI.COMM_WORLD

    num = function.evaluate(v.fn, mesh.X.coords)
    if comm.rank == 0:
        print(f"Velocity difference norm: {LA.norm(v.data - vel_soln_analytic):.6e}")
    comm.barrier()
except ImportError:
    import warnings

    warnings.warn("Unable to validate against UW2 analytical solution (UW2 not available).")

# %%
print(f"SolCx benchmark complete: resolution {n_els}, refinement {refinement}")
