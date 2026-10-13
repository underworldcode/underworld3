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
# Stokes Benchmark SolC

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

SolC is the *isoviscous* benchmark: a dense column occupying half the box drives
flow through a fluid of uniform viscosity. The difficulty is the discontinuous
forcing, not a material contrast — the pressure has a kink at the column edge
that a discretisation has to resolve.

(For the sharp **viscosity** contrast, see SolCx. This file was previously
titled and documented as SolCx while solving SolC, which is worth knowing if
you are comparing old output.)

## Key Concepts

- **Discontinuous body force**: buoyancy steps at x = 0.5, viscosity is uniform
- **Benchmark validation**: against `uw.analytic.SolC`, in-tree and validated
- **Truncated series**: the exact solution is a Fourier sum, `modes` terms of it
- **Free slip on all walls**, with a pressure null space

## Mathematical Formulation

Uniform viscosity $\\eta = 1$, with buoyancy stepping at the column edge. Both
the forcing and the exact velocity come from `uw.analytic.SolC`, so they cannot
disagree about sign or side — see the note in the validation section below.

## Parameters

- `uw_resolution`: Mesh resolution
- `uw_refinement`: Mesh refinement level
- `uw_modes`: Fourier modes in the SolC analytic solution
- `uw_viscosity_contrast`: log10 contrast, used by the SolCx section
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
    uw_use_simplex = 1,             # Use simplex mesh (1) or quad (0)
    uw_penalty = 100,               # Stokes penalty parameter
    uw_modes = 40,                  # Fourier modes in the SolC analytic solution
    uw_viscosity_contrast = 6,      # log10 contrast, for the SolCx section below
)

# Derived parameters
use_simplex = bool(params.uw_use_simplex)
eta_ratio = 10 ** params.uw_viscosity_contrast

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
x_c = sympy.Rational(1, 2)

# The exact solution supplies the forcing as well as the answer. Writing the
# body force out by hand here is how this file came to solve a mirrored,
# sign-flipped problem from the one it compared against: SolC's buoyancy is
# negative on x < x_c, and the Piecewise previously used was +1 on x > x_c.
# Nobody noticed, because the comparison sat behind an `import underworld` that
# always failed.
solC = uw.analytic.SolC(mesh, x_c=x_c, modes=int(params.uw_modes))

stokes.penalty = params.uw_penalty
stokes.constitutive_model.Parameters.shear_viscosity_0 = solC.fn_viscosity
stokes.bodyforce = solC.fn_bodyforce

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
## Validation against the analytic solution

This has to happen **here**, not at the end of the file. Everything below
reconfigures the same solver for other experiments, and `v` then holds those
answers rather than this one — which is exactly how the old check came to
compare a SolC analytic solution against a SolCx solve with penalty boundary
conditions.
"""

# %%
# `error` is a global reduction, so this is the same number on any rank count.
solC_velocity_error = solC.error("velocity", v)
solC_pressure_error = solC.error("pressure", p)

uw.pprint(f"SolC relative velocity error: {solC_velocity_error:.6e}")
uw.pprint(f"SolC relative pressure error: {solC_pressure_error:.6e}")

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
timing.print_table()  # see #499: display_fraction was removed from the API

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
timing.print_table()  # see #499: display_fraction was removed from the API

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
timing.print_table()  # see #499: display_fraction was removed from the API

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

    # Only when there is somewhere to show it. Guarded on mpi.size alone, this
    # blocks a script run forever waiting on a window that never opens — which
    # is why running this file to completion was not something anyone had done.
    if uw.is_notebook:
        pl.show(cpos="xy")

# %%
uw.pprint(
    f"Complete: resolution {n_els}, refinement {refinement}, "
    f"modes {int(params.uw_modes)}"
)
uw.pprint(f"  SolC velocity error (validated above): {solC_velocity_error:.6e}")
uw.pprint(
    "  The SolCx and natural-BC solves that follow it are BC experiments, "
    "not benchmarks — nothing here compares them against an exact solution."
)
