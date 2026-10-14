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
ap.add_argument("--heating", type=float, default=0.0, help="internal heating rate (source term)")
ap.add_argument("--yield0", type=float, default=0.0, help="yield stress at the surface (0: no yielding)")
ap.add_argument("--yield_depth", type=float, default=0.0, help="yield stress gradient with depth")
ap.add_argument("--eta_min", type=float, default=1.0, help="viscosity floor under yielding")
ap.add_argument("--T_int", type=float, default=0.6, help="initial interior temperature")
ap.add_argument("--tol", type=float, default=1e-5, help="Stokes solve tolerance (loosen for a spin-up)")
ap.add_argument("--yield_soft", type=float, default=0.1, help="yield-corner softness delta of the smooth yield law")
ap.add_argument("--picard", type=int, default=3, help="Picard warm-up iterations before Newton in the rotated nonlinear loop")
ap.add_argument("--eta_ref", type=float, default=1.0, help="temperature at which the viscosity is 1 (1: the base; T_int: the interior)")
ap.add_argument("--advect", choices=["supg", "slcn"], default="supg",
                help="temperature transport: Eulerian SUPG (its own step estimate) or semi-Lagrangian (dt_mult x Courant)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
if os.environ.get("ZG95_NOINODE"):
    from petsc4py import PETSc
    PETSc.Options()["mat_no_inode"] = True
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
Y_b = uw.discretisation.MeshVariable("Y_b", base, 1, degree=1)     # eta_eff / eta(T): 1 unyielded
xb = np.asarray(T_b.coords)
rb = np.hypot(xb[:, 0], xb[:, 1])
thb = np.arctan2(xb[:, 1], xb[:, 0])
z = (RO - rb) / (RO - RI)
T_int = args.T_int
T0 = (T_int * erf(z / 0.1) + (1.0 - T_int) * (1.0 - erf((1.0 - z) / 0.1))
      - 0.04 * np.cos(3 * thb) * np.sin(np.pi * z))     # downwelling under the trench at theta 0
T_b.array[:, 0, 0] = np.clip(T0, 0.0, 1.0)
if args.restart >= 0:
    T_b.read_timestep("zg95", "T_b", args.restart, outputPath=args.out)
    log(f"restarted T from checkpoint {args.restart}")
log(f"base: {base.dm.getHeightStratum(0)[1]} cells, {len(tail)} levels; finest cell {args.H / 2**args.ref}")

if args.advect == "supg":
    adv = uw.systems.AdvDiffusion(base, T_b, V_b.sym, order=1)
else:
    adv = uw.systems.AdvDiffusionSLCN(base, u_Field=T_b, V_fn=V_b.sym, order=1, monotone_mode="clamp")
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.f = args.heating
adv.add_dirichlet_bc(0.0, "Upper")
adv.add_dirichlet_bc(1.0, "Lower")
unit_r_b = base.CoordinateSystem.unit_e_0


def eta_of(Tsym):
    return sympy.exp(sympy.log(args.deta) * (args.eta_ref - Tsym))   # 1 at T = eta_ref, deta at eta_ref - 1


def yield_stress_on(mesh):
    """tau_y = yield0 + yield_depth * depth, depth measured from the outer wall"""
    x, y = mesh.X
    depth = RO - sympy.sqrt(x ** 2 + y ** 2)
    return args.yield0 + args.yield_depth * depth


def set_rheology(solver, mesh, Tsym, zone_sym=None):
    """Temperature-dependent viscosity, with yielding when ``--yield0`` is set.
    ``zone_sym`` (0/1 per cell) marks the weak bridge: under the TI model it is the
    weak plane's footprint; under yielding (the TI model has no yield stress
    without elasticity) it is an isotropic weak zone, viscosity and yield stress
    both reduced by ``--eta_w``."""
    eta0 = eta_of(Tsym)
    weak = 1 if zone_sym is None else (1 - (1 - args.eta_w) * zone_sym)
    if args.yield0 > 0:
        solver.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
        cm = solver.constitutive_model
        for attr in ("yield_softness", "_yield_softness"):
            if hasattr(cm, attr):
                try:
                    setattr(cm, attr, float(args.yield_soft))
                    break
                except Exception:
                    continue
        P = solver.constitutive_model.Parameters
        P.shear_viscosity_0 = eta0 * weak
        P.shear_viscosity_min = args.eta_min
        P.yield_stress = yield_stress_on(mesh) * weak
        return
    if zone_sym is None:
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        solver.constitutive_model.Parameters.shear_viscosity_0 = eta0
        return
    solver.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    P = solver.constitutive_model.Parameters
    P.shear_viscosity_0 = eta0
    P.shear_viscosity_1 = eta0 * weak
    return P                                   # the caller sets the director


def next_dt():
    """SUPG: the solver's own estimate, capped; semi-Lagrangian: dt_mult times the
    Courant estimate, capped (it is not Courant-limited)."""
    est = float(adv.estimate_dt())
    return min(args.dt_mult * est, args.dt_max)       # implicit SUPG tolerates large steps too


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
        P = set_rheology(s, mesh, self.T.sym[0], self.zone.sym[0])
        if P is not None:
            P.director = sympy.Matrix([[float(self.n_hat[0]), float(self.n_hat[1])]])
        unit_r = mesh.CoordinateSystem.unit_e_0
        s.bodyforce = args.ra * self.T.sym[0] * unit_r
        s.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
        s.add_rotated_freeslip_bc(0, "Lower", normal=unit_r)
        if args.arm == "split":
            s.add_fault_bc(0, boundary="Trench")
        s.petsc_use_pressure_nullspace = True
        s.tolerance = args.tol
        set_custom_fmg(s, tail, field_id=0, fac_zone=self.band if args.fac else None)
        self.stokes = s
        self.t_build = time.perf_counter() - t0
        log(f"  add-on {tag}: {mesh.dm.getHeightStratum(0)[1]} cells, band {int(self.band.sum())}, "
            f"bridge {int(self.bridge.sum())}, trench at {np.degrees(theta_t):.3f} deg, built in {self.t_build:.1f} s")

    def solve(self):
        self.T.array[:, 0, 0] = np.asarray(uw.function.global_evaluate(
            T_b.sym[0], clamp_radii(np.asarray(self.T.coords)))).ravel()
        n_nan = int(comm.allreduce(int(np.isnan(self.T.array).sum()), op=MPI.SUM))
        tmin, tmax = gmin(np.nanmin(self.T.array) if self.T.array.shape[0] else 1.0), gmax(np.nanmax(self.T.array) if self.T.array.shape[0] else 0.0)
        log(f"  T on the add-on: nan {n_nan}, range {tmin:.3f} {tmax:.3f}")
        t0 = time.perf_counter()
        self.stokes.solve(picard=args.picard if args.yield0 > 0 else 0)
        info = self.stokes._rotated_freeslip_info
        self.t_solve = time.perf_counter() - t0
        if os.environ.get("ZG95_INSPECT"):
            try:
                ctx = (getattr(self.stokes, "_rotated_linear_cache", None) or {})["ctx"]
                velpc = ctx["ksp"].getPC().getFieldSplitSubKSP()[0].getPC()
                for l in range(velpc.getMGLevels()):
                    sm = velpc.getMGSmoother(l)
                    A = sm.getOperators()[0]
                    d = A.getDiagonal().array
                    dmin = gmin(np.abs(d).min() if d.size else np.inf)
                    dmax = gmax(np.abs(d).max() if d.size else 0.0)
                    b, x = A.createVecs(); b.setRandom(); x.set(0.0)
                    sm.solve(b, x)
                    log(f"    level {l}: n {A.getSize()[0]} diag min {dmin:.3e} max {dmax:.3e} "
                        f"smoother |x| {x.norm():.3e} reason {sm.getConvergedReason()}")
            except Exception as e:
                log("    inspection failed:", e)
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
    set_rheology(s, base, T_b.sym[0])
    s.bodyforce = args.ra * T_b.sym[0] * unit_r_b
    s.add_rotated_freeslip_bc(0, "Upper", normal=unit_r_b)
    s.add_rotated_freeslip_bc(0, "Lower", normal=unit_r_b)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = args.tol
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


_vb_coords = np.asarray(V_b.coords)
_vb_r = np.hypot(_vb_coords[:, 0], _vb_coords[:, 1])
_vb_surf = _vb_r > RO - 1e-6
_tb_coords = np.asarray(T_b.coords)
_tb_depth = RO - np.hypot(_tb_coords[:, 0], _tb_coords[:, 1])


def yielded_fraction(solver, Tsym, coords, depth):
    """Fraction of nodes (all, and in the lid: depth < 0.15) where the effective
    viscosity is below 0.9 of the temperature-dependent one, i.e. the plastic
    branch is active. Collective; NaN when the model has no yield stress."""
    if args.yield0 <= 0 or solver is None:
        return float("nan"), float("nan")
    try:
        cm = solver.constitutive_model
        ratio = np.asarray(uw.function.evaluate(cm._plastic_eff_viscosity.sym / eta_of(Tsym),
                                                clamp_radii(coords))).ravel()
    except Exception as e:
        log("  yielded fraction unavailable:", str(e)[:120])
        return float("nan"), float("nan")
    yielded = ratio < 0.9
    lid = depth < 0.15
    try:
        if coords.shape[0] == _tb_coords.shape[0] and np.allclose(coords[:8], _tb_coords[:8]):
            # the base: keep the indicator at the P1 nodes for the checkpoint
            from scipy.spatial import cKDTree
            _d, k = cKDTree(coords).query(np.asarray(Y_b.coords))
            Y_b.array[:, 0, 0] = ratio[k]
    except Exception:
        pass
    n_all = comm.allreduce(int(yielded.size), op=MPI.SUM)
    n_lid = comm.allreduce(int(lid.sum()), op=MPI.SUM)
    y_all = comm.allreduce(int(yielded.sum()), op=MPI.SUM)
    y_lid = comm.allreduce(int((yielded & lid).sum()), op=MPI.SUM)
    return y_all / max(n_all, 1), y_lid / max(n_lid, 1)


def flow_diagnostics():
    """vrms over the base nodes, the surface rms, their ratio (mobility), and the
    lid depth: where the horizontally averaged T first reaches 0.8 of its mid-depth
    value. Node averages, not integrals; collective."""
    v2 = np.einsum("ij,ij->i", V_b.array[:, 0, :], V_b.array[:, 0, :]) if V_b.array.shape[0] else np.zeros(0)
    n_all = comm.allreduce(int(v2.size), op=MPI.SUM)
    n_s = comm.allreduce(int(_vb_surf.sum()), op=MPI.SUM)
    vrms = np.sqrt(comm.allreduce(float(v2.sum()), op=MPI.SUM) / max(n_all, 1))
    vsurf = np.sqrt(comm.allreduce(float(v2[_vb_surf].sum()), op=MPI.SUM) / max(n_s, 1))
    nb = 45
    edges = np.linspace(0.0, RO - RI, nb + 1)
    Tv = T_b.array[:, 0, 0] if T_b.array.shape[0] else np.zeros(0)
    idx = np.clip(np.digitize(_tb_depth, edges) - 1, 0, nb - 1)
    sums = np.bincount(idx, weights=Tv, minlength=nb)
    cnts = np.bincount(idx, minlength=nb).astype(float)
    sums = comm.allreduce(sums, op=MPI.SUM)
    cnts = comm.allreduce(cnts, op=MPI.SUM)
    prof = sums / np.maximum(cnts, 1)
    mid = prof[nb // 2]
    hit = np.flatnonzero(prof >= 0.8 * mid)
    lid = float(edges[hit[0]] + 0.5 * (edges[1] - edges[0])) if hit.size else float("nan")
    return {"vrms": float(vrms), "vsurf_rms": float(vsurf),
            "mobility": float(vsurf / max(vrms, 1e-30)), "lid_depth": lid,
            "T_mid": float(mid)}, prof


def checkpoint(step, t):
    base.write_timestep("zg95", meshUpdates=False, meshVars=[T_b, V_b, Y_b], index=step, outputPath=args.out)
    if rank0:
        json.dump({"step": step, "t": t, "theta_t": float(globals().get("theta_t", args.theta0))},
                  open(os.path.join(args.out, f"state_{step:04d}.json"), "w"))
        json.dump(records, open(os.path.join(args.out, "records.json"), "w"), indent=1)


if args.spinup > 0 and args.restart < 0:
    s_b, v_b = base_stokes("spin")
    for k in range(args.spinup):
        t0 = time.perf_counter()
        s_b.solve(picard=args.picard if args.yield0 > 0 else 0)
        info = s_b._rotated_freeslip_info
        V_b.array[...] = v_b.array
        dt = next_dt()
        adv.solve(timestep=dt)
        t += dt
        vmax = gmax(np.linalg.norm(V_b.array[:, 0, :], axis=1).max() if V_b.array.shape[0] else 0.0)
        diag, prof = flow_diagnostics()
        y_all, y_lid = yielded_fraction(s_b, T_b.sym[0], _tb_coords, _tb_depth)
        rec = {"step": step, "phase": "spinup", "t": t, "dt": dt, "vel_its": info.get("vel_its_last"),
               "newton": info.get("nonlinear_iterations"), "vmax": vmax,
               "t_step": round(time.perf_counter() - t0, 1), "yield_frac": y_all, "yield_frac_lid": y_lid,
               **diag}
        if rank0 and step % args.fig_every == 0:
            np.save(os.path.join(args.out, f"Tprofile_{step:04d}.npy"), prof)
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
    dt = next_dt()
    adv.solve(timestep=dt)
    t += dt
    if args.migrate:
        theta_t += v_trench * dt / RO
    vmax = gmax(np.linalg.norm(V_b.array[:, 0, :], axis=1).max() if V_b.array.shape[0] else 0.0)
    diag, prof = flow_diagnostics()
    rec.update(diag)
    xa = np.asarray(addon.T.coords)
    y_all, y_lid = yielded_fraction(addon.stokes, addon.T.sym[0], xa, RO - np.hypot(xa[:, 0], xa[:, 1]))
    rec.update({"newton": addon.stokes._rotated_freeslip_info.get("nonlinear_iterations"),
                "yield_frac": y_all, "yield_frac_lid": y_lid})
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
