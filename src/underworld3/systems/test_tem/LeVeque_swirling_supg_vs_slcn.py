#!/usr/bin/env python
# coding: utf-8
"""
LeVeque (1996) swirling deformation-flow benchmark
====================================================

Runs the SAME conservative level-set (CLS) advection problem through TWO
independent solvers on the SAME mesh, under the SAME analytic velocity
field, and compares them head-to-head:

  * ``level_set_SUPG.LevelSetSolver``  -- Crank-Nicolson + SUPG
    (Brooks & Hughes 1982), a hand-built implicit weak-form solve on
    ``uw.systems.SNES_Scalar`` (see ``SUPGAdvection``).
  * ``level_set_SLCN.LevelSetSolver``  -- UW3's canned
    ``AdvDiffusionSLCN`` (i.e. ``SNES_AdvectionDiffusion`` +
    ``SemiLagrangian``), the "old"/built-in solver.

Benchmark
---------
LeVeque, R.J. (1996), "High-resolution conservative algorithms for
advection in incompressible flow," SIAM J. Numer. Anal. 33(2):627-665,
introduced the "swirling deformation flow" velocity field derived from
the stream function

    psi(x,y,t) = (1/pi) sin^2(pi*x) sin^2(pi*y) cos(pi*t/T),

giving

    u = -dpsi/dy = -sin^2(pi*x) sin(2*pi*y) cos(pi*t/T)
    v =  dpsi/dx =  sin^2(pi*y) sin(2*pi*x) cos(pi*t/T).

This is the SAME formula used in the earlier ``SingleVortex_*`` scripts
(it is the standard "single vortex" test of Bell, Colella & Glaz (1989)
/ Enright et al. (2002), who use exactly this LeVeque stream function
with T=8 -- the two names refer to the same benchmark in the level-set
literature). The cos(pi*t/T) modulation makes the flow time-reversing:
the swirl runs "forward" for t in [0, T/2), stretching/spiralling the
interface into thin filaments, then EXACTLY reverses, so at t=T the
interface should return to its initial shape and position. That
round-trip is what makes this such a discriminating test -- any
irreversible numerical diffusion (interpolation smoothing in a
semi-Lagrangian trace-back, or over-diffusive stabilisation) shows up
directly as a FAILURE to recover the sharp initial shape, not just as a
transient blur that self-heals.

Diagnostics recorded for each solver, at every save interval:

  * ``interface_volume``      -- ∫phi dΩ (mass-conservation drift).
  * shape (L2) error          -- sqrt(∫(phi - phi_0)^2 dΩ), phi_0 the
                                  FROZEN initial field; large mid-run
                                  (filaments under-resolved / no longer
                                  matching phi_0's position) but should
                                  return close to its t=0 value (~0) at
                                  t=T if the round-trip is well resolved.
  * wall-clock time per `solve(dt)` call (advection + reinitialisation +
    mass correction together, i.e. the full per-step user-facing cost).

Output: a comparison plot (volume drift, shape error, cumulative
wall-clock, all vs model time) plus periodic XDMF/HDF5 checkpoints for
each solver in separate folders for Paraview inspection, and a short
printed summary table at t=T.

Usage
-----
    python LeVeque_swirling_supg_vs_slcn.py [--xres 64] [--T 8.0] [--severity]

``--severity`` is a shortcut for a shorter reversal period (T=2, the
value LeVeque's own paper favours) which reverses BEFORE the filaments
have thinned as much -- a gentler round-trip, useful as a quick sanity
check before committing to the full T=8 filament-resolution stress test.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import sympy
from mpi4py import MPI

import underworld3 as uw

import underworld3.systems.level_set_SLCN as ls_slcn
import underworld3.systems.level_set_SUPG as ls_supg


# =============================================================================
# CLI / problem setup
# =============================================================================

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--xres", type=int, default=64, help="mesh resolution (square)")
parser.add_argument("--T", type=float, default=8.0,
                     help="reversal period T (LeVeque's own paper uses 2; "
                          "Enright et al. 2002 use 8 for a much more severe "
                          "filament-stretching stress test -- default here)")
parser.add_argument("--severity", action="store_true",
                     help="shortcut for --T 2.0 (gentler round-trip)")
parser.add_argument("--save-dtime", type=float, default=None,
                     help="model-time interval between diagnostics/checkpoints "
                          "(default: T/64)")
parser.add_argument("--outdir", type=str, default="op_LeVeque_swirling_supg_vs_slcn/")
args = parser.parse_args()

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
xres = yres = args.xres

T_reversal = 2.0 if args.severity else args.T
dt_set = 0.5 / xres                      # same CFL-based dt convention as SingleVortex_*
save_dtime = args.save_dtime if args.save_dtime is not None else T_reversal / 64.0
save_every = max(1, int(np.round(save_dtime / dt_set)))
max_steps = int(np.round(save_every * (T_reversal / save_dtime))) + 1

outputPath = args.outdir
if uw.mpi.rank == 0:
    for sub in ("supg", "slcn"):
        p = os.path.join(outputPath, sub)
        os.makedirs(p, exist_ok=True)
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
    print(f"LeVeque swirling deformation flow: xres={xres}, T={T_reversal}, "
          f"dt={dt_set:.5g}, max_steps={max_steps}, save_every={save_every}")


# =============================================================================
# Mesh + shared velocity field (identical for both solvers -> a fair, purely
# solver-attributable comparison -- neither solver ever sees a different u)
# =============================================================================

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(xres, yres), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2, continuous=True)
timeField = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)

x, y = mesh.N.x, mesh.N.y


def make_velocity_expr(t_val: float):
    """LeVeque (1996) swirling deformation-flow velocity at model time t_val,
    from stream function psi = (1/pi) sin^2(pi x) sin^2(pi y) cos(pi t/T)."""
    stream = (1 / sympy.pi) * sympy.sin(sympy.pi * x) ** 2 * sympy.sin(sympy.pi * y) ** 2
    u_ = -sympy.diff(stream, y)
    v_ = sympy.diff(stream, x)
    modulation = sympy.cos(sympy.pi * t_val / T_reversal)
    return sympy.Matrix([[u_ * modulation, v_ * modulation]])


def update_velocity(t_val: float):
    v_expr = make_velocity_expr(t_val)
    with mesh.access(v):
        v.data[:, 0] = uw.function.evaluate(v_expr[0, 0], v.coords)[:, 0, 0]
        v.data[:, 1] = uw.function.evaluate(v_expr[0, 1], v.coords)[:, 0, 0]


# =============================================================================
# Two independent level-set fields, SAME initial geometry, one per solver
# =============================================================================

radius = 0.15
centre = [0.5, 0.75]
num_points = 91
angles = np.linspace(0, 2 * np.pi, num_points)
x0 = radius * np.cos(angles) + centre[0]
y0 = radius * np.sin(angles) + centre[1]
interface_coords = np.ascontiguousarray(np.array([x0, y0]).T)
polygon = np.vstack((interface_coords, interface_coords[0, :]))

psi_supg = uw.discretisation.MeshVariable(r"\psi_{supg}", mesh, 1, degree=2, continuous=True)
psi_slcn = uw.discretisation.MeshVariable(r"\psi_{slcn}", mesh, 1, degree=2, continuous=True)

eps_supg = ls_supg.interface_thickness(mesh, psi_supg, scale=0.35)
eps_slcn = ls_slcn.interface_thickness(mesh, psi_slcn, scale=0.35)

ls_supg.initialise_psi(psi_supg, eps_supg, interface_geometry="polygon",
                        interface_coordinates=polygon)
ls_slcn.initialise_psi(psi_slcn, eps_slcn, interface_geometry="polygon",
                        interface_coordinates=polygon)

# Frozen t=0 snapshots for the round-trip shape-error metric.
psi0_supg = uw.discretisation.MeshVariable(r"\psi^0_{supg}", mesh, 1, degree=2, continuous=True)
psi0_slcn = uw.discretisation.MeshVariable(r"\psi^0_{slcn}", mesh, 1, degree=2, continuous=True)
with mesh.access(psi0_supg, psi0_slcn):
    psi0_supg.data[:, 0] = psi_supg.data[:, 0]
    psi0_slcn.data[:, 0] = psi_slcn.data[:, 0]

solver_supg = ls_supg.LevelSetSolver(
    psi_supg, velocity=v.sym, epsilon=eps_supg, reini_steps=1, reini_frequency=5)
solver_slcn = ls_slcn.LevelSetSolver(
    psi_slcn, velocity=v.sym, epsilon=eps_slcn, reini_steps=1, reini_frequency=5)

initial_area = np.pi * radius ** 2


def shape_error(psi, psi0):
    """sqrt(integral((psi - psi0)^2) dOmega) -- 0 for a perfect round-trip."""
    integ = uw.maths.Integral(mesh, (psi.sym[0, 0] - psi0.sym[0, 0]) ** 2)
    return float(np.sqrt(max(integ.evaluate(), 0.0)))


# =============================================================================
# Time loop -- both solvers advanced by the SAME dt, under the SAME v, at
# the SAME model times, so any divergence between them is attributable
# purely to the advection scheme.
# =============================================================================

history = {
    "t": [],
    "vol_supg": [], "vol_slcn": [],
    "err_supg": [], "err_slcn": [],
    "cumtime_supg": [], "cumtime_slcn": [],
}
walltime_supg = 0.0
walltime_slcn = 0.0

step, time_now, dt = 0, 0.0, 0.0

while step < max_steps:
    if uw.mpi.rank == 0:
        msg = (f"Step: {step:5d}  Model Time: {time_now:7.4f}  dt: {dt:7.4f}  "
               f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        sys.stdout.write(msg)
        sys.stdout.flush()

    update_velocity(time_now)

    if step % save_every == 0:
        vol_supg = solver_supg.interface_volume()
        vol_slcn = solver_slcn.interface_volume()
        err_supg = shape_error(psi_supg, psi0_supg)
        err_slcn = shape_error(psi_slcn, psi0_slcn)

        history["t"].append(time_now)
        history["vol_supg"].append(vol_supg)
        history["vol_slcn"].append(vol_slcn)
        history["err_supg"].append(err_supg)
        history["err_slcn"].append(err_slcn)
        history["cumtime_supg"].append(walltime_supg)
        history["cumtime_slcn"].append(walltime_slcn)

        if uw.mpi.rank == 0:
            print(f"    SUPG: volume={vol_supg:.6f} "
                  f"(drift={100*(vol_supg-initial_area)/initial_area:+.3f}%)  "
                  f"shape_err={err_supg:.5e}  cum_wall={walltime_supg:.2f}s")
            print(f"    SLCN: volume={vol_slcn:.6f} "
                  f"(drift={100*(vol_slcn-initial_area)/initial_area:+.3f}%)  "
                  f"shape_err={err_slcn:.5e}  cum_wall={walltime_slcn:.2f}s")

        timeField.data[:, 0] = time_now
        mesh.write_timestep("mesh", meshUpdates=False, meshVars=[v, psi_supg, timeField],
                             outputPath=os.path.join(outputPath, "supg"), index=step)
        mesh.write_timestep("mesh", meshUpdates=False, meshVars=[v, psi_slcn, timeField],
                             outputPath=os.path.join(outputPath, "slcn"), index=step)

    dt = dt_set

    t0 = time.perf_counter()
    solver_supg.solve(dt=dt)
    dt_wall = time.perf_counter() - t0
    walltime_supg += (dt_wall if uw.mpi.comm.size == 1
                       else uw.mpi.comm.allreduce(dt_wall, op=MPI.MAX))

    t0 = time.perf_counter()
    solver_slcn.solve(dt=dt)
    dt_wall = time.perf_counter() - t0
    walltime_slcn += (dt_wall if uw.mpi.comm.size == 1
                       else uw.mpi.comm.allreduce(dt_wall, op=MPI.MAX))

    step += 1
    time_now += dt


# =============================================================================
# Final round-trip summary (flow has returned to t=0 configuration)
# =============================================================================

final_vol_supg = solver_supg.interface_volume()
final_vol_slcn = solver_slcn.interface_volume()
final_err_supg = shape_error(psi_supg, psi0_supg)
final_err_slcn = shape_error(psi_slcn, psi0_slcn)

if uw.mpi.rank == 0:
    print("\n" + "=" * 70)
    print(f"LeVeque swirling deformation flow -- round-trip summary at t={time_now:.4f}")
    print("=" * 70)
    print(f"{'':14s}{'volume drift %':>16s}{'shape L2 error':>18s}{'total wall (s)':>18s}")
    print(f"{'SUPG':14s}{100*(final_vol_supg-initial_area)/initial_area:16.4f}"
          f"{final_err_supg:18.5e}{walltime_supg:18.2f}")
    print(f"{'SLCN (old)':14s}{100*(final_vol_slcn-initial_area)/initial_area:16.4f}"
          f"{final_err_slcn:18.5e}{walltime_slcn:18.2f}")
    print("=" * 70)
    print("Lower shape L2 error at t=T = better round-trip shape recovery "
          "(less irreversible numerical diffusion). Lower |volume drift| = "
          "better mass conservation. Lower total wall time = faster.")


# =============================================================================
# Comparison plot
# =============================================================================

if uw.mpi.rank == 0:
    t_arr = np.array(history["t"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(t_arr, 100 * (np.array(history["vol_supg"]) - initial_area) / initial_area,
            label="SUPG", lw=2)
    ax.plot(t_arr, 100 * (np.array(history["vol_slcn"]) - initial_area) / initial_area,
            label="SLCN (old)", lw=2, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("model time")
    ax.set_ylabel("volume drift (%)")
    ax.set_title("Mass conservation")
    ax.legend()

    ax = axes[1]
    ax.semilogy(t_arr, np.maximum(history["err_supg"], 1e-16), label="SUPG", lw=2)
    ax.semilogy(t_arr, np.maximum(history["err_slcn"], 1e-16), label="SLCN (old)",
                lw=2, ls="--")
    ax.axvline(T_reversal / 2, color="gray", lw=0.7, ls=":", label="flow reversal (T/2)")
    ax.set_xlabel("model time")
    ax.set_ylabel(r"shape error  $\|\phi-\phi_0\|_2$")
    ax.set_title("Round-trip shape recovery\n(should dip back down near t=T)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_arr, history["cumtime_supg"], label="SUPG", lw=2)
    ax.plot(t_arr, history["cumtime_slcn"], label="SLCN (old)", lw=2, ls="--")
    ax.set_xlabel("model time")
    ax.set_ylabel("cumulative wall time (s)")
    ax.set_title("Performance")
    ax.legend()

    fig.suptitle(f"LeVeque (1996) swirling deformation flow -- SUPG vs SLCN "
                 f"(xres={xres}, T={T_reversal})")
    fig.tight_layout()
    fig_path = os.path.join(outputPath, "supg_vs_slcn_comparison.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nComparison plot written to {fig_path}")
