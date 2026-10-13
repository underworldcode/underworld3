"""A Zhong & Gurnis (1995) analogue on the annulus: temperature-dependent
convection with a dipping trench fault that moves with the overriding plate,
surface velocity and dynamic topography read every step.

The base annulus (static, balanced, never cut) carries the temperature and
the advection-diffusion. The add-on finest is the base's finest with the
trench placed as the structured ladder (one element either side of the
spine, outcropping through its cap); it is rebuilt whenever the trench has
moved by a rung. Stokes is solved on the add-on with rotated free-slip on
both walls (exact radial normal), the spine split and in frictionless
contact (``--arm split``, the ladder's cap and last rungs painted TI to
bridge to the surface) or the whole band painted TI (``--arm ti``), and
custom FMG on the base's own levels. Velocity goes back to the base by
evaluation for the semi-Lagrangian step.

python zg95_annulus.py --out ~/+Simulations/zg95/run1 --steps 50 [--arm split|ti]
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import sympy
import underworld3 as uw
from scipy.special import erf
from mpi4py import MPI
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

comm = uw.mpi.comm


def gmax(x):
    return float(comm.allreduce(float(x), op=MPI.MAX))


def gmin(x):
    return float(comm.allreduce(float(x), op=MPI.MIN))


def gather_arrays(*arrs):
    """Concatenate rank-local 1-D arrays onto every rank (small surface data)."""
    out = []
    for a in arrs:
        parts = comm.allgather(np.asarray(a))
        out.append(np.concatenate(parts) if parts else np.asarray(a))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/+Simulations/zg95/run1"))
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--spinup", type=int, default=0, help="steps on the base without the fault first")
ap.add_argument("--ra", type=float, default=1e6)
ap.add_argument("--deta", type=float, default=1e3)
ap.add_argument("--eta_w", type=float, default=1e-3, help="TI weak-plane viscosity ratio")
ap.add_argument("--H", type=float, default=0.16, help="coarsest base cell")
ap.add_argument("--ref", type=int, default=3, help="base halvings (finest base cell H / 2**ref)")
ap.add_argument("--S", type=float, default=0.005, help="ladder spacing")
ap.add_argument("--dip", type=float, default=45.0)
ap.add_argument("--depth", type=float, default=0.15)
ap.add_argument("--arm", choices=["split", "ti"], default="split")
ap.add_argument("--fac", type=int, default=0, help="band patch smoother on the finest")
ap.add_argument("--dt_mult", type=float, default=2.0)
ap.add_argument("--dt_max", type=float, default=1e-3)
ap.add_argument("--restart", type=int, default=-1, help="restart T from checkpoint index in --out (skips the spin-up)")
ap.add_argument("--ckpt_every", type=int, default=25)
ap.add_argument("--fig_every", type=int, default=5)
ap.add_argument("--theta0", type=float, default=0.0, help="initial trench position (rad)")
ap.add_argument("--migrate", type=int, default=1, help="advect the trench with the overriding plate")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
RI, RO = 0.55, 1.0
S, W = args.S, 2.0 * args.S
DIP = np.deg2rad(args.dip)
rank0 = uw.mpi.rank == 0


def log(*a):
    if rank0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- base
base = uw.meshing.Annulus(radiusOuter=RO, radiusInner=RI, cellSize=args.H, refinement=args.ref,
                          qdegree=2)
tail = base._coarse_level_meshes()[:-1] + [base]
T_b = uw.discretisation.MeshVariable("T_b", base, 1, degree=2)
V_b = uw.discretisation.MeshVariable("V_b", base, 2, degree=2)
xb = np.asarray(T_b.coords)
rb = np.hypot(xb[:, 0], xb[:, 1])
thb = np.arctan2(xb[:, 1], xb[:, 0])
z = (RO - rb) / (RO - RI)
T_int = 0.6
T0 = (T_int * erf(z / 0.1) + (1.0 - T_int) * (1.0 - erf((1.0 - z) / 0.1))
      - 0.04 * np.cos(3 * thb) * np.sin(np.pi * z))     # downwelling under the trench at theta 0
T_b.array[:, 0, 0] = np.clip(T0, 0.0, 1.0)
if args.restart >= 0:
    T_b.read_timestep("zg95", "T_b", args.restart, outputPath=args.out)
    log(f"restarted T from checkpoint {args.restart}")
log(f"base: {base.dm.getHeightStratum(0)[1]} cells, {len(tail)} levels; finest cell {args.H / 2**args.ref}")

adv = uw.systems.AdvDiffusionSLCN(base, u_Field=T_b, V_fn=V_b.sym, order=1, monotone_mode="clamp")
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.f = 0.0
adv.add_dirichlet_bc(0.0, "Upper")
adv.add_dirichlet_bc(1.0, "Lower")
unit_r_b = base.CoordinateSystem.unit_e_0


def eta_of(Tsym):
    return sympy.exp(sympy.log(args.deta) * (1 - Tsym))          # 1 at T = 1, deta at T = 0


def clamp_radii(x):
    r = np.hypot(x[:, 0], x[:, 1])
    f = np.clip(r, RI + 1e-10, RO - 1e-10) / np.maximum(r, 1e-14)
    return x * f[:, None]


# ----------------------------------------------------------------------------- add-on
class AddOn:
    """The trench at angle theta_t placed into a copy of the base's finest, with
    its Stokes solver."""

    def __init__(self, theta_t, tag):
        t0 = time.perf_counter()
        t_hat = np.array([np.cos(theta_t), np.sin(theta_t)])
        s_hat = np.array([-np.sin(theta_t), np.cos(theta_t)])
        d_hat = -np.cos(DIP) * t_hat + np.sin(DIP) * s_hat            # down-dip toward +theta
        self.n_hat = np.array([-d_hat[1], d_hat[0]])
        top = t_hat * (RO + 0.02)
        L = (args.depth + 0.02) / np.cos(DIP)
        n_s = int(round(L / S))
        spine = np.array([top + d_hat * (L * i / n_s) for i in range(n_s + 1)])
        r_sp = np.hypot(spine[:, 0], spine[:, 1])
        self.trace = spine[r_sp <= RO - 3.0 * S]
        self.theta_t = theta_t
        self.mesh, self.info = place_fault_ribbon_2d(
            base, [("Trench", self.trace)], W, spines=[("Trench", spine)], margin_rings=1,
            clearance=0.7, split=(args.arm == "split"), mesher="ladder")
        mesh = self.mesh
        self.band = np.asarray(self.info["band"], dtype=bool)
        self.T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
        self.v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, 2, degree=2)
        self.p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1, continuous=True)
        self.h = uw.discretisation.MeshVariable(f"h_{tag}", mesh, 1, degree=1)
        self.zone = uw.discretisation.MeshVariable(f"zone_{tag}", mesh, 1, degree=0, continuous=False)
        dm = mesh.dm
        vS, vE = dm.getDepthStratum(0)
        cS, cE = dm.getHeightStratum(0)
        X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
        if args.arm == "split":
            pairs = (getattr(mesh, "_fault_point_pairs", None) or {}).get("Trench", {})
            pp = np.array([int(q) for q in pairs] + [int(q) for q in pairs.values()], dtype=np.int64)
            pv = pp[(pp >= vS) & (pp < vE)] - vS
            self.r_tip = gmax(np.hypot(X[pv, 0], X[pv, 1]).max() if pv.size else 0.0)
            cent = np.array([X[[int(q) - vS for q in dm.getTransitiveClosure(c)[0] if vS <= q < vE]].mean(axis=0)
                             for c in range(cS, cE)])
            self.bridge = self.band & (np.hypot(cent[:, 0], cent[:, 1]) > self.r_tip - 0.5 * W)
        else:
            self.r_tip = None
            self.bridge = self.band.copy()
        self.zone.array[:, 0, 0] = self.bridge.astype(float)
        s = uw.systems.Stokes(mesh, velocityField=self.v, pressureField=self.p)
        eta0 = eta_of(self.T.sym[0])
        s.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = eta0
        s.constitutive_model.Parameters.shear_viscosity_1 = eta0 * (1 - (1 - args.eta_w) * self.zone.sym[0])
        s.constitutive_model.Parameters.director = sympy.Matrix([[float(self.n_hat[0]), float(self.n_hat[1])]])
        unit_r = mesh.CoordinateSystem.unit_e_0
        s.bodyforce = args.ra * self.T.sym[0] * unit_r
        s.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
        s.add_rotated_freeslip_bc(0, "Lower", normal=unit_r)
        if args.arm == "split":
            s.add_fault_bc(0, boundary="Trench")
        s.petsc_use_pressure_nullspace = True
        s.tolerance = 1e-5
        set_custom_fmg(s, tail, field_id=0, fac_zone=self.band if args.fac else None)
        self.stokes = s
        self.t_build = time.perf_counter() - t0
        log(f"  add-on {tag}: {mesh.dm.getHeightStratum(0)[1]} cells, band {int(self.band.sum())}, "
            f"bridge {int(self.bridge.sum())}, trench at {np.degrees(theta_t):.3f} deg, built in {self.t_build:.1f} s")

    def solve(self):
        self.T.array[:, 0, 0] = np.asarray(uw.function.global_evaluate(
            T_b.sym[0], clamp_radii(np.asarray(self.T.coords)))).ravel()
        t0 = time.perf_counter()
        self.stokes.solve()
        info = self.stokes._rotated_freeslip_info
        self.t_solve = time.perf_counter() - t0
        self.stokes.dynamic_topography("Upper", self.h, buoyancy_scale=args.ra)
        rec = {"vel_its": info.get("vel_its_last"), "pres_its": info.get("pres_its_last"),
               "converged": bool(info.get("converged")), "t_solve": round(self.t_solve, 1)}
        if args.arm == "split":
            from underworld3.utilities.fault_contact import fault_slip
            _s, Vs, leak = fault_slip(self.stokes, "Trench", info)
            rec["slip_max"] = gmax(np.abs(Vs).max() if len(Vs) else 0.0)
            rec["leak"] = gmax(np.abs(leak).max() if len(leak) else 0.0)
        return rec

    def surface(self):
        vc = np.asarray(self.v.coords)
        on = np.hypot(vc[:, 0], vc[:, 1]) > RO - 1e-6
        th = np.arctan2(vc[on, 1], vc[on, 0])
        vx = self.v.array[on, 0, :]
        vt = -vx[:, 0] * np.sin(th) + vx[:, 1] * np.cos(th)
        hc = np.asarray(self.h.coords)
        hon = np.hypot(hc[:, 0], hc[:, 1]) > RO - 1e-6
        thh = np.arctan2(hc[hon, 1], hc[hon, 0])
        th, vt, thh, hh = gather_arrays(th, vt, thh, self.h.array[hon, 0, 0])
        o = np.argsort(th)
        oh = np.argsort(thh)
        return th[o], vt[o], thh[oh], hh[oh]


# ----------------------------------------------------------------------------- base-only Stokes (spin-up)
def base_stokes(tag):
    v = uw.discretisation.MeshVariable(f"Ub_{tag}", base, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"Pb_{tag}", base, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(base, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = eta_of(T_b.sym[0])
    s.bodyforce = args.ra * T_b.sym[0] * unit_r_b
    s.add_rotated_freeslip_bc(0, "Upper", normal=unit_r_b)
    s.add_rotated_freeslip_bc(0, "Lower", normal=unit_r_b)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1e-5
    set_custom_fmg(s, tail[:-1], field_id=0)
    return s, v


# ----------------------------------------------------------------------------- figures
def figure(step, t, addon, rec):
    surf = addon.surface() if addon is not None else None      # collective gather
    if not rank0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    if uw.mpi.size == 1:
        dm = base.dm
        vS, vE = dm.getDepthStratum(0)
        cS, cE = dm.getHeightStratum(0)
        X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
        cells = np.array([[int(q) - vS for q in dm.getTransitiveClosure(c)[0] if vS <= q < vE] for c in range(cS, cE)])
        Tv = uw.function.evaluate(T_b.sym[0], clamp_radii(X)).ravel()
        tc = ax.tripcolor(mtri.Triangulation(X[:, 0], X[:, 1], cells), Tv, shading="gouraud", cmap="RdBu_r",
                          vmin=0, vmax=1)
        fig.colorbar(tc, ax=ax, shrink=0.8)
    if addon is not None:
        tr = addon.trace
        ax.plot(tr[:, 0], tr[:, 1], color="k", lw=1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {step}, t = {t:.5f}: T on the base (serial render); trench (black)", fontsize=9, loc="left")
    if addon is not None:
        th, vt, thh, hh = surf
        ax = axes[1]
        ax.plot(np.degrees(th), vt, lw=1)
        ax.axvline(np.degrees(addon.theta_t), color="#c0392b", lw=0.8, ls="--")
        ax.set_xlabel("theta (deg)"); ax.set_ylabel("surface tangential velocity")
        ax.set_title(f"surface velocity; velocity its {rec.get('vel_its')}, slip {rec.get('slip_max', 0):.2f}",
                     fontsize=9, loc="left")
        ax.grid(alpha=0.3)
        ax = axes[2]
        ax.plot(np.degrees(thh), hh, lw=1)
        ax.axvline(np.degrees(addon.theta_t), color="#c0392b", lw=0.8, ls="--")
        ax.set_xlabel("theta (deg)"); ax.set_ylabel("dynamic topography (-(s_nn - mean)/Ra)")
        ax.set_title("topography from the free-slip reaction", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"step_{step:04d}.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- run
records = []
t = 0.0
step = 0


def checkpoint(step, t):
    base.write_timestep("zg95", meshUpdates=False, meshVars=[T_b, V_b], index=step, outputPath=args.out)
    if rank0:
        json.dump({"step": step, "t": t, "theta_t": float(globals().get("theta_t", args.theta0))},
                  open(os.path.join(args.out, f"state_{step:04d}.json"), "w"))
        json.dump(records, open(os.path.join(args.out, "records.json"), "w"), indent=1)


if args.spinup > 0 and args.restart < 0:
    s_b, v_b = base_stokes("spin")
    for k in range(args.spinup):
        t0 = time.perf_counter()
        s_b.solve()
        info = s_b._rotated_freeslip_info
        V_b.array[...] = v_b.array
        dt = min(args.dt_mult * float(adv.estimate_dt()), args.dt_max)
        adv.solve(timestep=dt)
        t += dt
        vmax = gmax(np.linalg.norm(V_b.array[:, 0, :], axis=1).max() if V_b.array.shape[0] else 0.0)
        rec = {"step": step, "phase": "spinup", "t": t, "dt": dt, "vel_its": info.get("vel_its_last"),
               "vmax": vmax, "t_step": round(time.perf_counter() - t0, 1)}
        records.append(rec)
        log("SPIN", json.dumps(rec))
        if step % args.fig_every == 0:
            figure(step, t, None, rec)
        if step % args.ckpt_every == 0 or k == args.spinup - 1:
            checkpoint(step, t)
        step += 1
    del s_b, v_b
    gc.collect()
if args.restart >= 0:
    st = json.load(open(os.path.join(args.out, f"state_{args.restart:04d}.json")))
    t, step = float(st["t"]), int(st["step"]) + 1
    args.theta0 = float(st.get("theta_t", args.theta0))

theta_t = args.theta0
addon = None
n_built = 0
for k in range(args.steps):
    t0 = time.perf_counter()
    if addon is None or abs(theta_t - addon.theta_t) * RO > S:
        if addon is not None:
            del addon
            gc.collect()
        addon = AddOn(theta_t, f"a{n_built}")
        n_built += 1
    rec = addon.solve()
    th, vt, thh, hh = addon.surface()
    # the trench moves with the overriding plate: the surface just on the +theta side
    j = int(np.argmin(np.abs(th - (addon.theta_t + 0.03))))
    v_trench = float(vt[j])
    # velocity back onto the base for the advection step
    Vb = np.asarray(uw.function.global_evaluate(addon.v.sym, clamp_radii(np.asarray(V_b.coords)))).reshape(-1, 2)
    V_b.array[:, 0, :] = Vb
    dt = min(args.dt_mult * float(adv.estimate_dt()), args.dt_max)
    adv.solve(timestep=dt)
    t += dt
    if args.migrate:
        theta_t += v_trench * dt / RO
    vmax = gmax(np.linalg.norm(V_b.array[:, 0, :], axis=1).max() if V_b.array.shape[0] else 0.0)
    rec.update({"step": step, "phase": "fault", "t": t, "dt": dt, "vmax": vmax, "v_trench": v_trench,
                "theta_t": float(theta_t), "theta_built": float(addon.theta_t), "n_built": n_built,
                "topo_min": float(hh.min()), "topo_max": float(hh.max()),
                "np": uw.mpi.size,
                "t_build": round(addon.t_build, 1), "t_step": round(time.perf_counter() - t0, 1)})
    records.append(rec)
    log("STEP", json.dumps(rec))
    if step % args.fig_every == 0 or k == args.steps - 1:
        figure(step, t, addon, rec)
        if rank0:
            json.dump(records, open(os.path.join(args.out, "records.json"), "w"), indent=1)
            np.savez(os.path.join(args.out, f"surface_{step:04d}.npz"), theta=th, vt=vt, theta_h=thh, h=hh,
                     trace=addon.trace)
    if step % args.ckpt_every == 0 or k == args.steps - 1:
        checkpoint(step, t)
    step += 1
if rank0:
    json.dump(records, open(os.path.join(args.out, "records.json"), "w"), indent=1)
log("DONE")
