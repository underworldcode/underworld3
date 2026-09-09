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
# Swarm population control in an extending box

**PHYSICS:** utilities
**DIFFICULTY:** intermediate
**PURPOSE:** demonstration

## Description

Pure-shear extension `v = (x, -y)` on a fixed mesh: the side walls are
outflow and the top and bottom are inflow. Nothing enters through an
inflow boundary unless we put it there, so the cells along the top and
bottom starve and the marker layer they carry breaks up.

Setting `swarm.population_control` refills the starved cells at the end
of every `advection()`. Run it with `POPCTL=0` and `POPCTL=1` and
compare: without control the box drains to 13% of its particles and
most cells are empty; with control no cell is ever empty.

The velocity is prescribed, so what this measures is the swarm
machinery alone, with no solve in the way.
"""

# %%
import os

import numpy as np
import sympy

import underworld3 as uw

POPCTL = os.environ.get("POPCTL", "1") == "1"
CELL = float(os.environ.get("CELL", "0.08"))
STEPS = int(os.environ.get("STEPS", "40"))
DT = float(os.environ.get("DT", "0.05"))
FILL = int(os.environ.get("FILL", "3"))
OUT = os.environ.get("OUT", "runs")

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=CELL, qdegree=2
)
x, y = mesh.X
V = sympy.Matrix([[x, -y]])                      # incompressible pure shear
mesh.return_coords_to_bounds = None              # a true outflow: particles leave

swarm = uw.swarm.Swarm(mesh)
M = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1,
                                proxy_location="integration_points")
swarm.populate(fill_param=FILL)
X0 = np.asarray(swarm._particle_coordinates.data)
with uw.synchronised_array_update():
    M.data[:, 0] = (np.abs(X0[:, 1]) < 0.2).astype(int)   # a central layer

if POPCTL:
    # Refill starved cells. A new particle takes the material of its nearest
    # neighbour: repopulate() does that for any INTEGER variable without being
    # asked, because the average of two material labels is not a label.
    swarm.population_control = dict(min_per_cell=6)

c0, c1 = mesh.dm.getHeightStratum(0)
ncells = c1 - c0


def census():
    P = np.asarray(swarm._particle_coordinates.data)
    cells = np.asarray(mesh._robust_owning_cells(P))
    return np.bincount(cells[cells >= 0], minlength=ncells)


hist = []
frames = {}
for step in range(STEPS + 1):
    n = census()
    hist.append((step * DT, swarm.local_size, int((n == 0).sum()), int((n < 6).sum()), int(n.min())))
    if step in (0, STEPS // 2, STEPS):
        P = np.asarray(swarm._particle_coordinates.data)
        frames[step] = dict(P=P.copy(), M=np.asarray(M.data[:, 0]).copy(), n=n.copy())
    if step < STEPS:
        swarm.advection(V, DT, order=2)

os.makedirs(OUT, exist_ok=True)
tag = "popctl" if POPCTL else "nopopctl"
np.savez(f"{OUT}/repop_{tag}.npz", hist=np.array(hist),
         **{f"P{k}": v["P"] for k, v in frames.items()},
         **{f"M{k}": v["M"] for k, v in frames.items()},
         **{f"n{k}": v["n"] for k, v in frames.items()},
         steps=np.array(sorted(frames)), cell=CELL, dt=DT)

h = np.array(hist)
print(f"RESULT {tag}: cells {ncells}, start {int(h[0,1])} particles "
      f"-> end {int(h[-1,1])} | empty cells {int(h[0,2])} -> {int(h[-1,2])} | "
      f"cells below 6 {int(h[0,3])} -> {int(h[-1,3])} | min per cell {int(h[-1,4])}", flush=True)
