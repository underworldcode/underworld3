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
# Population control in an extending box

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate
**PURPOSE:** demonstration

## Description

Pure-shear extension $\\mathbf{v} = (x, -y)$ on a fixed mesh. The side walls
are outflow and the top and bottom are inflow, and nothing arrives through an
inflow boundary unless you put it there — so the cells along the top and
bottom starve.

What that costs depends on how the material is read. This example measures it
on the area of a marker layer, a global integral of the layer's own material
mask, so the number is the same however the mesh is partitioned.

Incompressible pure shear thins the layer by $e^{-t}$, so after $t = 2$ the
area should be $e^{-2} = 13.5\\%$ of where it started. Anything else is the
material representation failing, not physics.

Run with `-uw_population_control 0` to switch the refilling off, and with
`-uw_proxy_location integration_points` to see how much the choice of
mapping matters.
"""

# %%
import math

import sympy

import underworld3 as uw

params = uw.Params(
    uw_population_control=1,
    uw_proxy_location="cells",       # or "integration_points"
    uw_cell_size=0.08,
    uw_steps=40,
    uw_dt=0.05,
    uw_fill_param=3,
    uw_min_per_cell=6,
)

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5),
    cellSize=params.uw_cell_size, qdegree=2,
)
x, y = mesh.X
velocity = sympy.Matrix([[x, -y]])          # incompressible pure shear

# A true outflow: particles that leave the box are deleted rather than
# clamped onto the wall.
mesh.return_coords_to_bounds = None

# %% [markdown]
"""
## A marker layer on a material swarm

`proxy_location="cells"` fits a polynomial to the particles **of each cell**,
so a cell that runs out of particles has nothing to fit. That is the mapping
population control exists for.
"""

# %%
materials = uw.swarm.MaterialSwarm(
    mesh, fill_param=params.uw_fill_param,
    proxy_location=params.uw_proxy_location,
)
materials.add("matrix", density=1.0)
materials.add("layer", density=1.0)
materials["layer"] = sympy.Abs(y) < 0.2

if params.uw_population_control:
    # Refill starved cells at the end of every advection. A new particle takes
    # the material of its nearest neighbour: repopulate() does that for any
    # INTEGER variable without being asked, because the average of two
    # material labels is not a label.
    materials.population_control = dict(min_per_cell=params.uw_min_per_cell)

layer_area = uw.maths.Integral(mesh, materials["layer"].mask)

# %% [markdown]
"""
## Extend the box
"""

# %%
initial_area = layer_area.evaluate()

for _ in range(params.uw_steps):
    materials.advection(velocity, params.uw_dt, order=2)

elapsed = params.uw_steps * params.uw_dt
final_area = layer_area.evaluate()
expected = initial_area * math.exp(-elapsed)

uw.pprint(
    f"proxy_location={params.uw_proxy_location} "
    f"population_control={bool(params.uw_population_control)}: "
    f"layer area {initial_area:.4f} -> {final_area:.4f} at t={elapsed:g} "
    f"(exact thinning gives {expected:.4f}, "
    f"so this is {final_area / expected:.2f}x the right answer)"
)

# %% [markdown]
"""
## What the numbers say

Measured on this rig, `cellSize=0.08`, 40 steps:

| `proxy_location` | population control | layer area | vs exact |
|---|---|---|---|
| `"cells"` | on | 0.1125 | 1.03x |
| `"cells"` | off | 0.5505 | **5.03x** |
| `"integration_points"` | on | 0.1169 | 1.06x |
| `"integration_points"` | off | 0.1175 | 1.07x |

Starvation wrecks the per-cell fit, which needs particles *in that cell*. The
nearest-particle mapping degrades gracefully by comparison — an empty cell
still finds a plausible particle nearby — so with `"integration_points"` the
field-level answer barely moves and the cost of losing particles shows up
only in the swarm's own bookkeeping.

Population control is cheap and always safe; this is where it is *necessary*.
"""
