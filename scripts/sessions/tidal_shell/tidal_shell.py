"""Global tidal response of a thin (visco)elastic shell on a sphere.

The shell floats on an ocean: no basal drag, and its radial motion is the tide's,
w = d/dt [h2 V / g], prescribed as the normal velocity of the sheet (the datum of
the tangency constraint). The sheet solves for the tangential velocity under the
plane-stress membrane closure; the strain rate carries the curvature term w/R P.

Non-dimensional: R = 1, orbital period 1, tidal radial amplitude A (the h2 V/g
scale), shear modulus 1, Maxwell time De (= eta / mu) in periods. Eccentricity
tide, synchronous rotation, degree 2 (Wahr et al. 2009 pattern):

    W(x, t) = A [ -3/2 P20 cos(nt) + 3/4 ( 3 (x^2 - y^2) cos(nt) + 8 x y sin(nt) ) ],
    P20 = (3 z^2 - 1) / 2,  n = 2 pi,

with the sub-planet point on +x and the spin axis +z. Closed form for the viscous
limit on a uniform membrane: a degree-2 radial pattern W Y2 gives the tangential
velocity (3/11) W grad_s Y2, which is checked every step (--check).

Parallel-safe: every diagnostic is a collective reduction; the maps are a
collective evaluation on a longitude-latitude grid drawn on rank 0.
"""
import argparse
import json
import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities.rotated_bc import SHEET_TANGENCY

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--h", type=float, default=0.15, help="cell size on the unit sphere")
ap.add_argument("--A", type=float, default=0.01, help="radial tidal amplitude (h2 V/g scale)")
ap.add_argument("--De", type=float, default=0.0, help="Maxwell time in periods; 0: viscous sheet")
ap.add_argument("--eta", type=float, default=1.0, help="viscosity of the viscous sheet (De = 0)")
ap.add_argument("--L", type=float, default=0.08, help="shell thickness / R (Enceladus ~ 20/252)")
ap.add_argument("--periods", type=float, default=2.0)
ap.add_argument("--steps_per_period", type=int, default=32)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--check", type=int, default=1, help="compare with the membrane closed form each step")
ap.add_argument("--fig_every", type=int, default=8)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

comm = uw.mpi.comm
rank0 = uw.mpi.rank == 0
os.makedirs(args.out, exist_ok=True)


def log(*a):
    if rank0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- mesh and sheet
sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=args.h, degree=1, qdegree=5)
x, y, z = sphere.X
sheet = uw.systems.ThinSheet(sphere)
if args.De > 0:
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate="jaumann")
    sheet.constitutive_model.Parameters.shear_modulus = 1.0
    sheet.constitutive_model.Parameters.shear_viscosity_0 = args.De      # Maxwell time De periods
else:
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = args.eta
sheet.thickness = args.L
sheet.tolerance = args.tol

n_orb = 2 * sympy.pi
P20 = (3 * z ** 2 - 1) / 2
pattern_c = -sympy.Rational(3, 2) * P20 + sympy.Rational(9, 4) * (x ** 2 - y ** 2)   # multiplies cos(nt)
pattern_s = 6 * x * y                                                                 # multiplies sin(nt)


def radial_velocity(t):
    """w = dW/dt at time t (a sympy expression of position)."""
    c, s = float(np.cos(2 * np.pi * t)), float(np.sin(2 * np.pi * t))
    return args.A * 2 * np.pi * (-pattern_c * s + pattern_s * c)


# closed form of the tangential velocity for the viscous membrane: (3/11) grad_s of the
# degree-2 radial pattern (both patterns are degree-2 harmonics on the unit sphere)
X = sympy.Matrix([x, y, z])
rhat = X / sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
P = sympy.eye(3) - rhat * rhat.T


def grad_s(f):
    return P * sympy.Matrix([f.diff(v) for v in (x, y, z)])


v_c = sympy.Rational(3, 11) * grad_s(pattern_c)
v_s = sympy.Rational(3, 11) * grad_s(pattern_s)

# collective diagnostics: node arrays reduced over ranks
coords = np.asarray(sheet.u.coords)
rh = coords / np.linalg.norm(coords, axis=1)[:, None]
from mpi4py import MPI      # reductions only; the solver handles the rest

# the map grid (collective evaluation, drawn on rank 0)
lon = np.linspace(-np.pi, np.pi, 181)
lat = np.linspace(-np.pi / 2, np.pi / 2, 91)
LON, LAT = np.meshgrid(lon, lat)
grid = np.column_stack([(np.cos(LAT) * np.cos(LON)).ravel(), (np.cos(LAT) * np.sin(LON)).ravel(), np.sin(LAT).ravel()])


def diagnostics(t):
    u = np.asarray(sheet.u.array[:, 0, :])
    un = (u * rh).sum(axis=1)
    ut = u - un[:, None] * rh
    w_ex = np.asarray(uw.function.evaluate(radial_velocity(t), coords)).ravel() if coords.shape[0] else np.zeros(0)
    c, s = np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)
    v_ex = np.asarray(uw.function.evaluate(args.A * 2 * np.pi * (-v_c * s + v_s * c), coords)).reshape(-1, 3) if coords.shape[0] else np.zeros((0, 3))
    loc = np.array([float((ut ** 2).sum()), float(((ut - v_ex) ** 2).sum()), float((v_ex ** 2).sum()),
                    float(np.abs(un - w_ex).max()) if un.size else 0.0, float(coords.shape[0])])
    tot = np.zeros_like(loc)
    comm.Allreduce(loc, tot, op=MPI.SUM)
    nrm = comm.allreduce(loc[3], op=MPI.MAX)
    n = max(tot[4], 1.0)
    return {"vt_rms": float(np.sqrt(tot[0] / n)), "vex_rms": float(np.sqrt(tot[2] / n)),
            "rel_err_membrane": float(np.sqrt(tot[1] / max(tot[2], 1e-300))), "normal_datum_err": float(nrm)}


def figure(step, t):
    stress = sheet.stress
    # in-plane principal stresses from the 3x3 sheet stress on the grid (collective evaluation)
    S = np.asarray(uw.function.evaluate(stress, grid)).reshape(-1, 3, 3)
    U = np.asarray(uw.function.evaluate(sheet.u.sym, grid)).reshape(-1, 3)
    if not rank0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = np.linalg.eigvalsh(0.5 * (S + np.swapaxes(S, 1, 2)))     # one eigenvalue ~ 0 (normal)
    ev_sorted = np.sort(ev, axis=1)
    s_max = ev_sorted[:, 2].reshape(LAT.shape)
    s_min = ev_sorted[:, 0].reshape(LAT.shape)
    un = (U * grid).sum(axis=1)
    ut = np.linalg.norm(U - un[:, None] * grid, axis=1).reshape(LAT.shape)
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), subplot_kw={"projection": "mollweide"})
    for ax, fld, ttl, cmap in ((axes[0], un.reshape(LAT.shape), "radial velocity w (prescribed)", "RdBu_r"),
                               (axes[1], ut, "tangential speed", "viridis"),
                               (axes[2], s_max, "largest in-plane principal stress", "RdBu_r")):
        lim = np.abs(fld).max() + 1e-30
        pc = ax.pcolormesh(LON, LAT, fld, cmap=cmap, shading="auto",
                           **({"vmin": -lim, "vmax": lim} if cmap == "RdBu_r" else {}))
        ax.set_title(ttl, fontsize=10, loc="left"); ax.grid(alpha=0.3)
        fig.colorbar(pc, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05)
    fig.suptitle(f"tidal shell: step {step}, t = {t:.3f} periods, De = {args.De}, A = {args.A}", fontsize=10)
    fig.savefig(os.path.join(args.out, f"tide_{step:04d}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- time loop
dt = 1.0 / args.steps_per_period
nsteps = int(round(args.periods * args.steps_per_period))
records = []
t = 0.0
log(f"sphere h {args.h}, {comm.allreduce(int(sphere.dm.getHeightStratum(0)[1]), op=MPI.SUM)} cells; "
    f"De {args.De}, A {args.A}, dt {dt}, {nsteps} steps")
for k in range(nsteps):
    t0 = time.perf_counter()
    t_mid = t + dt                                   # implicit: the datum at the end of the step
    sheet.normal_velocity = radial_velocity(t_mid)
    if args.De > 0:
        sheet.solve(timestep=dt, zero_init_guess=(k == 0), verbose=args.verbose)
    else:
        sheet.solve(zero_init_guess=(k == 0), verbose=args.verbose)
    t = t_mid
    info = sheet._rotated_freeslip_info or {}
    rec = {"step": k, "t": t, "dt": dt, "converged": bool(info.get("converged")),
           "nonlinear": info.get("nonlinear_iterations"), "t_step": round(time.perf_counter() - t0, 1)}
    if args.check:
        rec.update(diagnostics(t))
    records.append(rec)
    log("STEP", json.dumps(rec))
    if k % args.fig_every == 0 or k == nsteps - 1:
        figure(k, t)
        if rank0:
            json.dump(records, open(os.path.join(args.out, "records.json"), "w"), indent=1)
log("DONE")
