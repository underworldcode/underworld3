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
# Eulerian SUPG Advection-Diffusion Rotation Test

**PHYSICS:** convection
**DIFFICULTY:** advanced

## Description

A Gaussian anomaly carried round the origin by rigid rotation, solved with
the fully implicit Eulerian solver `uw.systems.AdvDiffusionSUPG`. The exact
solution is known at every time (`uw.analytic.RotatingGaussian`), so the
error is measured directly rather than inferred from a picture.

The scheme is stable at any cell Courant number; what limits the timestep
is how far the anomaly moves per step relative to its own width. Try
`-uw_courant 4` to see the accuracy fall off as `dt**2` while the solve
stays perfectly stable, and `-uw_order 2` to see the second-order scheme.

## Key Concepts

- **Implicit Eulerian transport**: no trace-back, no departure points; the
  timestep is a runtime constant of the compiled kernels.
- **SUPG stabilisation**: the streamline-upwind test-function perturbation
  written as a flux, so PETSc needs no modified test space.
- **Drop-in for SLCN**: the same constructor, `order`, `theta`, `estimate_dt`
  and `solve`; change the class name and nothing else.

## Parameters

- `uw_res`: cells across the box
- `uw_courant`: timestep as a multiple of the cell-crossing time
- `uw_order`, `uw_theta`: the time scheme, with the semi-Lagrangian solver's meaning
- `uw_diffusivity`: thermal diffusivity (0 is pure advection)
"""

# %%
import numpy as np
import sympy
import underworld3 as uw

# %% [markdown]
"""
## Configurable Parameters

Override from the command line:

```bash
python Ex_AdvectionDiffusionSUPG_RotationTest.py -uw_courant 4 -uw_order 2
```
"""

# %%
params = uw.Params(
    uw_res=32,
    uw_courant=1.0,
    uw_order=1,          # 1 with theta 0.5 is Crank-Nicolson; 2 with theta 1.0 is BDF2
    uw_theta=0.5,
    uw_diffusivity=0.0,
    uw_sigma=0.12,
)

# %% [markdown]
"""
## Mesh, exact solution and the transported field
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=2.0 / params.uw_res, qdegree=3)
x, y = mesh.X

exact = uw.analytic.RotatingGaussian(
    mesh, sigma=params.uw_sigma, centre_radius=0.5, omega=1.0,
    diffusivity=params.uw_diffusivity)

T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
T.array[:, 0, 0] = uw.function.evaluate(exact.at(0.0), T.coords).reshape(-1)

# Rigid rotation about the origin, one revolution in 2 pi
velocity = sympy.Matrix([[-y, x]])

# %% [markdown]
"""
## The solver

Diffusivity is set on the constitutive model, as for every scalar solver. The
walls carry T = 0, which is exact to rounding a few sigma from the orbit.
"""

# %%
adv_diff = uw.systems.AdvDiffusionSUPG(
    mesh, T, velocity, order=params.uw_order, theta=params.uw_theta)
adv_diff.constitutive_model.Parameters.diffusivity = params.uw_diffusivity
for boundary in ("Left", "Right", "Top", "Bottom"):
    adv_diff.add_dirichlet_bc(0.0, boundary)

# %% [markdown]
"""
## Time loop

`estimate_dt` returns the cell-crossing time. It is a resolution guide, not a
stability limit, so the timestep is a chosen multiple of it. For a multistep
scheme the exact history is planted so the first step already runs at full
order.
"""

# %%
period = float(exact.period)
dt_cell = float(adv_diff.estimate_dt())
n_steps = int(np.ceil(period / (params.uw_courant * dt_cell)))
dt = period / n_steps

if params.uw_order > 1:
    history = [uw.function.evaluate(exact.at(-k * dt), T.coords).reshape(-1, 1, 1)
               for k in range(params.uw_order)]
    adv_diff.DuDt.set_initial_history(history, dt=dt)

t = 0.0
for step in range(n_steps):
    adv_diff.solve(timestep=dt)
    t += dt
    if step % max(1, n_steps // 4) == 0 or step == n_steps - 1:
        err = exact.error(exact.at(t), T, norm="integral")
        uw.pprint(f"step {step:4d}  t = {t:6.3f}  relative L2 error = {err:.3e}")

# %% [markdown]
"""
## Result

After one revolution the field should match its initial state. At a Courant
number of one half the round-trip error is below one per cent on this mesh;
it grows as `dt**2` from there.
"""

# %%
round_trip = exact.error(exact.at(t), T, norm="integral")
uw.pprint(f"round-trip relative L2 error: {round_trip:.3e}  "
          f"(min {float(T.array.min()):.3f}, max {float(T.array.max()):.3f})")

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)
    pvmesh.point_data["T_exact"] = vis.scalar_fn_to_pv_points(pvmesh, exact.at(t))
    pvmesh.point_data["error"] = pvmesh.point_data["T"] - pvmesh.point_data["T_exact"]

    pl = pv.Plotter(window_size=(900, 450), shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(pvmesh, scalars="T", cmap="RdBu_r", clim=(0, 1), show_edges=False)
    pl.subplot(0, 1)
    pl.add_mesh(pvmesh, scalars="error", cmap="RdBu_r", show_edges=False)
    pl.show(cpos="xy")
