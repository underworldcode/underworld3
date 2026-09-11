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
# Materials on a swarm

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate
**PURPOSE:** demonstration

## Description

Two viscosity layers, 1 and 1000, carried by particles and driven from the
top. The model names its materials and says where they are; it never writes
a level set, a mask, or a blend.

With the interface on mesh edges the exact velocity is piecewise linear and
lies in the P2 velocity space, so the only error in the solve is how the
material is represented. Read at the integration points (the default), where
each point takes the material of its nearest particle, the problem solves to
2e-7. Run with `-uw_proxy_location nodes` to watch the nodal level set smear
the interface across a cell and leave an L2 error of 8e-2.
"""

# %%
import numpy as np
import sympy

import underworld3 as uw

params = uw.Params(
    uw_proxy_location="integration_points",   # or "nodes", or "cells"
    uw_proxy_sampling="nearest",              # or "share"
    uw_cell_size=0.1,
    uw_eta_top=1000.0,
    uw_interface=0.5,
    uw_fill_param=3,
)

mesh = uw.meshing.UnstructuredSimplexBox(
    cellSize=params.uw_cell_size, qdegree=2, regular=True
)
v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %% [markdown]
"""
## The materials

`MaterialSwarm` is a `Swarm` that carries materials. Declare them with the
property names the constitutive model knows, then say where each one is: a
symbolic condition on the mesh coordinates, a boolean array over the
particles, or a callable. Every `add` must come before the first read, which
is what allocates the particles and the level sets.
"""

# %%
materials = uw.swarm.MaterialSwarm(
    mesh,
    fill_param=params.uw_fill_param,
    proxy_location=params.uw_proxy_location,
    proxy_sampling=(
        params.uw_proxy_sampling
        if params.uw_proxy_location == "integration_points"
        else None
    ),
)
materials.add("lower", shear_viscosity_0=1.0, density=3300)
materials.add("upper", shear_viscosity_0=params.uw_eta_top, density=3400)

materials["upper"] = mesh.X[1] > params.uw_interface

# %% [markdown]
"""
## The solve

`stokes.materials = materials` sets every parameter the constitutive model
recognises — here `shear_viscosity_0`. `density` is not a viscous-model
parameter, so it stays a blended symbol for the model script to use; this
problem is driven by the boundary, so it goes unused.
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.materials = materials

stokes.add_dirichlet_bc((1.0, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
stokes.tolerance = 1e-8
stokes.solve()

# %% [markdown]
"""
## Against the exact layered Couette profile

The error is a global integral rather than a nodal norm, so the number is the
same however the mesh is partitioned.
"""

# %%
h, eta_top = params.uw_interface, params.uw_eta_top
gradient = 1.0 / (h + (1.0 - h) / eta_top)
y = mesh.X[1]
exact = sympy.Piecewise(
    (gradient * y, y < h),
    (gradient * h + gradient / eta_top * (y - h), True),
)

error = uw.maths.Integral(mesh, (v.sym[0] - exact) ** 2).evaluate() ** 0.5
viscosity = uw.maths.Integral(mesh, materials.shear_viscosity_0).evaluate()
exact_viscosity = 1.0 * h + eta_top * (1.0 - h)

uw.pprint(
    f"proxy_location={params.uw_proxy_location} "
    f"sampling={params.uw_proxy_sampling} fill={params.uw_fill_param}: "
    f"assembled int(eta) {viscosity:.4f} (exact {exact_viscosity:.4f}) | "
    f"velocity L2 {error:.3e}"
)

# %% [markdown]
"""
## What the weak form sees

The upper-material mask along a line crossing the interface. At the
integration points it is a step in the right place; the nodal level set ramps
across a whole cell.
"""

# %%
line = np.column_stack(
    [np.full(201, 0.5), np.linspace(max(0.0, h - 0.25), min(1.0, h + 0.25), 201)]
)
upper = np.asarray(uw.function.evaluate(materials["upper"].mask, line)).reshape(-1)
uw.pprint(f"mask along the line: {upper.min():+.4f} .. {upper.max():+.4f}")
