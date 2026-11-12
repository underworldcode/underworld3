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
# Level set in the LeVeque swirling flow: SUPG against SLCN

**PHYSICS:** convection
**DIFFICULTY:** advanced

## Description

The swirling deformation flow of LeVeque (1996), the standard stress test
for interface transport: a circle is stretched into a thin spiral filament
for half a period, then the flow reverses exactly and the circle should
come back. Any irreversible error, whether interpolation loss in a
trace-back or stabilisation diffusion, shows up as a failure to recover
the initial shape.

The same conservative level set is carried by the two transport solvers,
Eulerian SUPG and semi-Lagrangian, under the same velocity and timestep,
each with its own reinitialisation and mass correction. The script reports
the shape error against the frozen initial field, the enclosed volume, and
the wall time of each.

The stream function is

$$\psi(x, y, t) = \frac{1}{\pi}\sin^2(\pi x)\,\sin^2(\pi y)\,\cos(\pi t / T)$$

with period `T`: 2 (LeVeque's own value, a gentle round trip) or 8
(Enright et al. 2002, filaments thinner than the mesh).

Contributed by NengLu (issue #657); converted to the repository's script
conventions.

## Parameters

- `uw_res`: cells across the unit square
- `uw_period`: reversal period `T`
- `uw_courant`: timestep as a multiple of the cell-crossing time
"""

# %%
import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3.systems import level_set

# %%
params = uw.Params(
    uw_res=64,
    uw_period=2.0,
    uw_courant=0.5,
    uw_reini_frequency=5,
    uw_outdir="output/levelset_leveque",
)

# %% [markdown]
"""
## Mesh and the time-dependent velocity
"""

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(params.uw_res, params.uw_res), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
x, y = mesh.X

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)

stream = (1 / sympy.pi) * sympy.sin(sympy.pi * x) ** 2 * sympy.sin(sympy.pi * y) ** 2
u_x, u_y = -sympy.diff(stream, y), sympy.diff(stream, x)


def set_velocity(t):
    modulation = float(np.cos(np.pi * t / params.uw_period))
    v.array[:, 0, 0] = modulation * uw.function.evaluate(u_x, v.coords).reshape(-1)
    v.array[:, 0, 1] = modulation * uw.function.evaluate(u_y, v.coords).reshape(-1)


# %% [markdown]
"""
## Two level sets, one initial circle, one per solver
"""

# %%
radius, centre = 0.15, (0.5, 0.75)
angles = np.linspace(0.0, 2.0 * np.pi, 91)
circle = np.column_stack((centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)))

solvers = {}
for name in ("supg", "slcn"):
    psi = uw.discretisation.MeshVariable(f"psi_{name}", mesh, 1, degree=2)
    eps = level_set.interface_thickness(mesh, psi, scale=0.35)
    level_set.initialise_psi(psi, eps, interface_geometry="polygon", interface_coordinates=circle)
    psi0 = uw.discretisation.MeshVariable(f"psi0_{name}", mesh, 1, degree=2)
    psi0.array[...] = psi.array[...]
    solver = uw.systems.LevelSetSolver(
        psi, velocity=v.sym, epsilon=eps, advection=name,
        reini_steps=1, reini_frequency=params.uw_reini_frequency)
    solvers[name] = dict(psi=psi, psi0=psi0, solver=solver, wall=0.0)


def shape_error(psi, psi0):
    return float(np.sqrt(max(uw.maths.Integral(mesh, (psi.sym[0] - psi0.sym[0]) ** 2).evaluate(), 0.0)))


# %% [markdown]
"""
## Time loop

Both solvers take the same step, chosen as a multiple of the cell-crossing
time so the comparison is at equal Courant number.
"""

# %%
dt = params.uw_courant / params.uw_res
n_steps = int(np.round(params.uw_period / dt))
dt = params.uw_period / n_steps
report_every = max(1, n_steps // 16)
initial_area = np.pi * radius ** 2

t = 0.0
for step in range(n_steps):
    set_velocity(t)
    for name, s in solvers.items():
        t0 = time.perf_counter()
        s["solver"].solve(dt)
        s["wall"] += time.perf_counter() - t0
    t += dt
    if step % report_every == 0 or step == n_steps - 1:
        for name, s in solvers.items():
            volume = s["solver"].interface_volume()
            uw.pprint(f"t = {t:6.3f}  {name}: volume drift {100 * (volume - initial_area) / initial_area:+.3f}%  "
                      f"shape error {shape_error(s['psi'], s['psi0']):.3e}  wall {s['wall']:.1f} s")

# %% [markdown]
"""
## Round trip

At `t = T` the flow has returned the fluid to where it started; the shape
error measures what the transport did not undo.
"""

# %%
for name, s in solvers.items():
    uw.pprint(f"{name}: round-trip shape error {shape_error(s['psi'], s['psi0']):.3e}, "
              f"total wall {s['wall']:.1f} s")

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pl = pv.Plotter(window_size=(900, 450), shape=(1, 2))
    for i, (name, s) in enumerate(solvers.items()):
        pvmesh = vis.mesh_to_pv_mesh(mesh)
        pvmesh.point_data["psi"] = vis.scalar_fn_to_pv_points(pvmesh, s["psi"].sym)
        pl.subplot(0, i)
        pl.add_mesh(pvmesh, scalars="psi", cmap="RdBu_r", clim=(0, 1), show_edges=False)
        pl.add_mesh(pvmesh.contour([0.5], scalars="psi"), color="black", line_width=2)
        pl.add_text(name, font_size=10)
    os.makedirs(params.uw_outdir, exist_ok=True)
    pl.show(cpos="xy", screenshot=os.path.join(params.uw_outdir, "leveque_round_trip.png"))
