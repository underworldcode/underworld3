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
top. The model names its materials and says where they are; it never
writes a level set, a mask, or a blend.

With the interface on mesh edges the exact velocity is piecewise linear and
lies in the P2 velocity space, so the only error in the solve is how the
material is represented. Read at the integration points (the default),
where each point takes the material of its nearest particle, the problem
solves to 2e-7. Run with `PROXY=nodes` to see the nodal level set smear the
interface across a cell and leave an L2 velocity error of 8e-2.
"""

# %%
import os

import numpy as np
import sympy

import underworld3 as uw

PROXY = os.environ.get("PROXY", "integration_points")
SAMPLING = os.environ.get("SAMPLING", "nearest")
CELL = float(os.environ.get("CELL", "0.1"))
ETA_TOP = float(os.environ.get("ETA_TOP", "1000"))
H = float(os.environ.get("H", "0.5"))
FILL = int(os.environ.get("FILL", "3"))
OUT = os.environ.get("OUT", "runs")

mesh = uw.meshing.UnstructuredSimplexBox(cellSize=CELL, qdegree=2, regular=True)
v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %% [markdown]
"""
## The materials

`MaterialSwarm` is a `Swarm` that carries materials. Declare them with the
property names the constitutive model knows, then paint the regions: a
symbolic condition on the mesh coordinates, a boolean array over the
particles, or a callable. The swarm is populated the first time anything
reads it, so every `add` can come first.
"""

# %%
materials = uw.swarm.MaterialSwarm(
    mesh, fill_param=FILL, proxy_location=PROXY,
    proxy_sampling=SAMPLING if PROXY == "integration_points" else None,
)
materials.add("lower", shear_viscosity_0=1.0, density=3300)
materials.add("upper", shear_viscosity_0=ETA_TOP, density=3400)

materials["upper"] = mesh.X[1] > H

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
"""

# %%
A = 1.0 / (H + (1.0 - H) / ETA_TOP)
Xv = np.asarray(v.coords)
vx_exact = np.where(Xv[:, 1] < H, A * Xv[:, 1],
                    A * H + A / ETA_TOP * (Xv[:, 1] - H))
err = np.abs(np.asarray(v.data[:, 0]) - vx_exact)

viscosity = materials.shear_viscosity_0
assembled = float(uw.maths.Integral(mesh, viscosity).evaluate())
exact_int = 1.0 * H + ETA_TOP * (1.0 - H)

# the material as the weak form sees it, across the interface
ys = np.linspace(max(0.0, H - 0.25), min(1.0, H + 0.25), 201)
line = np.c_[np.full_like(ys, 0.5), ys]
upper = np.asarray(uw.function.evaluate(materials["upper"].mask, line)).reshape(-1)

os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/matindex_{PROXY}.npz", ys=ys, upper=upper, vy=Xv[:, 1],
         vx=np.asarray(v.data[:, 0]), vx_exact=vx_exact,
         l2=np.sqrt(np.mean(err ** 2)), linf=err.max(), assembled=assembled)

print(f"RESULT proxy={PROXY} sampling={SAMPLING} fill={FILL}: "
      f"assembled int(eta) {assembled:.4f} (exact {exact_int:.4f}) | "
      f"vx L2 {np.sqrt(np.mean(err**2)):.3e} Linf {err.max():.3e} | "
      f"mask range {upper.min():+.4f}..{upper.max():+.4f}", flush=True)
