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
# A material index read at the integration points

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate
**PURPOSE:** demonstration

## Description

Two viscosity layers, 1 and 1000, carried by particles as an
`IndexSwarmVariable` and driven from the top. With the interface on
mesh edges the exact velocity is piecewise linear and lies in the P2
velocity space, so the only error in the solve is how the material is
represented.

Run with `PROXY=nodes`, `PROXY=integration_points` and `PROXY=cells`.
The nodal level set smears the interface across a cell and leaves an
L2 velocity error of 8e-2; read at the integration points, where each
point takes the material of its nearest particle, the same problem
solves to 2e-7.
"""

# %%
import os

import numpy as np
import sympy

import underworld3 as uw

PROXY = os.environ.get("PROXY", "nodes")
CELL = float(os.environ.get("CELL", "0.1"))
ETA_TOP = float(os.environ.get("ETA_TOP", "1000"))
H = float(os.environ.get("H", "0.5"))
FILL = int(os.environ.get("FILL", "3"))
DEG = int(os.environ.get("DEG", "1"))
OUT = os.environ.get("OUT", "runs")

mesh = uw.meshing.UnstructuredSimplexBox(cellSize=CELL, qdegree=2, regular=True)
v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

swarm = uw.swarm.Swarm(mesh)
mat = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=DEG, proxy_location=PROXY)
swarm.populate(fill_param=FILL)
X = np.asarray(swarm._particle_coordinates.data)
with uw.synchronised_array_update():
    mat.data[:, 0] = (X[:, 1] > H).astype(int)      # 0 below the interface, 1 above

eta = mat.createMask([1.0, ETA_TOP])

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
stokes.add_dirichlet_bc((1.0, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
stokes.tolerance = 1e-8
stokes.solve()

# Exact layered Couette profile
A = 1.0 / (H + (1.0 - H) / ETA_TOP)
Xv = np.asarray(v.coords)
vx_exact = np.where(Xv[:, 1] < H, A * Xv[:, 1], A * H + A / ETA_TOP * (Xv[:, 1] - H))
err = np.abs(np.asarray(v.data[:, 0]) - vx_exact)
assembled = float(uw.maths.Integral(mesh, eta).evaluate())
exact_int = 1.0 * H + ETA_TOP * (1.0 - H)

# The material as the weak form sees it, on a fine sampling line across the interface
ys = np.linspace(max(0.0, H - 0.25), min(1.0, H + 0.25), 201)
line = np.c_[np.full_like(ys, 0.5), ys]
upper = np.asarray(uw.function.evaluate(mat.sym[1], line)).reshape(-1)

os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/matindex_{PROXY}.npz", ys=ys, upper=upper, vy=Xv[:, 1],
         vx=np.asarray(v.data[:, 0]), vx_exact=vx_exact, l2=np.sqrt(np.mean(err ** 2)),
         linf=err.max(), assembled=assembled)
print(f"RESULT proxy={PROXY} deg={DEG} fill={FILL}: assembled int(eta) {assembled:.4f} "
      f"(exact {exact_int:.4f}) | vx L2 {np.sqrt(np.mean(err**2)):.3e} Linf {err.max():.3e} "
      f"| mask range {upper.min():+.4f}..{upper.max():+.4f}", flush=True)
