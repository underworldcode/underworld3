# %% [markdown]
"""
# Navier-Stokes Lid-Driven Cavity (Re=100)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced
**RUNTIME:** ~10 minutes (order=1), ~10 minutes (order=2)

## Description

Classic lid-driven cavity benchmark at Re=100, validated against Ghia et al. (1982)
reference data. Compares BDF order-1 and order-2 time integration using the
Semi-Lagrangian Navier-Stokes solver.

## Key Concepts

- Navier-Stokes equations with Semi-Lagrangian advection
- BDF time integration (order 1 vs order 2)
- Quantitative validation against published benchmark data
- Centreline velocity profile extraction

## Reference

Ghia, Ghia & Shin (1982), "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", J. Comp. Physics 48, 387-411.
"""

# %% [markdown]
"""
## Parameters
"""

# %%
RE = 100.0          # PARAM: Reynolds number
CELLSIZE = 0.04     # PARAM: mesh element size
NSTEPS = 200        # PARAM: number of time steps
DT = 0.05           # PARAM: time step size

# %%
import numpy as np
import sympy

import underworld3 as uw
from underworld3.systems import NavierStokes

# Ghia et al. (1982) reference data: u-velocity along vertical centreline
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                    0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                    0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                    -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                    0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

# %% [markdown]
"""
## Solver Setup
"""

# %%
def run_cavity(order):
    """Run lid-driven cavity to steady state and return centreline velocity."""

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=CELLSIZE, qdegree=3)

    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                        vtype=uw.VarType.SCALAR)

    ns = NavierStokes(mesh, velocityField=v, pressureField=p, rho=1.0, order=order)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.viscosity = 1.0 / RE
    ns.saddle_preconditioner = 1.0
    ns.bodyforce = sympy.Matrix([0.0, 0.0])
    ns.tolerance = 1.0e-4
    ns.petsc_options["ksp_type"] = "fgmres"

    # Boundary conditions: moving lid on top, no-slip elsewhere
    ns.add_essential_bc(sympy.Matrix([1.0, 0.0]), "Top")
    ns.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    ns.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Left")
    ns.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Right")

    # Time stepping
    for step in range(NSTEPS):
        ns.solve(timestep=DT, verbose=False)
        if (step + 1) % 50 == 0:
            uw.pprint(0, f"  order={order}, step {step+1}/{NSTEPS}")

    # Extract u-velocity along vertical centreline at x=0.5
    from scipy.interpolate import griddata

    coords = v.coords[:, :2]
    sample_pts = np.column_stack([np.full_like(GHIA_Y, 0.5), GHIA_Y])
    u_centreline = griddata(coords, v.data[:, 0], sample_pts, method="cubic").flatten()
    rms = float(np.sqrt(np.mean((u_centreline - GHIA_U) ** 2)))

    return u_centreline, rms


# %% [markdown]
"""
## Run Benchmark
"""

# %%
results = {}
for order in [1, 2]:
    uw.pprint(0, f"\n=== BDF order={order} ===")
    u, rms = run_cavity(order)
    results[order] = (u, rms)
    uw.pprint(0, f"  RMS vs Ghia: {rms:.6f}")

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1 and uw.is_notebook():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    ax.plot(GHIA_U, GHIA_Y, "ko", markersize=8, label="Ghia et al. (1982)", zorder=10)
    ax.plot(results[1][0], GHIA_Y, "b-", linewidth=2,
            label=f"order=1 (RMS={results[1][1]:.4f})")
    ax.plot(results[2][0], GHIA_Y, "r--", linewidth=2,
            label=f"order=2 (RMS={results[2][1]:.4f})")
    ax.set_xlabel("u-velocity at x=0.5")
    ax.set_ylabel("y")
    ax.set_title("Lid-driven cavity Re=100\nGhia et al. (1982) validation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("sl_cavity_benchmark.png", dpi=150)
    plt.show()
