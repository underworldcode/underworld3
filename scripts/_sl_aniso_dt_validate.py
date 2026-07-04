"""Validate direction-aware CFL by running convection trajectories
on uniform R=1.0 and adapted R=3.0 starting from the step-125
state. Compare Nu / vrms to the isotropic-CFL baseline.

If adv-diff breaks (NaN T, T outside [0,1] envelope, vrms blowup,
Nu divergence vs baseline), stop and report the step.
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import sympy

import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells


p = argparse.ArgumentParser()
p.add_argument('--R', type=float, default=1.0,
               help='Adapt resolution_ratio (1.0 = uniform)')
p.add_argument('--n-steps', type=int, default=100)
p.add_argument('--cfl', type=float, default=0.5)
p.add_argument('--log-every', type=int, default=1)
p.add_argument('--src-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'uniform_res16_Ra1e7_dEta1e4'))
p.add_argument('--src-stem', type=str,
               default='sl_uniform_res16_Ra1e7_dEta1e4_step00125')
p.add_argument('--mode', type=str, default='aniso',
               choices=['aniso', 'iso'],
               help='aniso = direction-aware CFL (new); '
                    'iso = original inradius-based CFL')
p.add_argument('--out-tag', type=str, default=None)
p.add_argument('--snapshot-every', type=int, default=20)
p.add_argument('--resume', action='store_true',
               help='Resume from the latest snapshot in OUT_DIR '
                    '(and the history.npz partial). Skip the '
                    'restart-from-step125 setup.')
args = p.parse_args()

OUT_TAG = args.out_tag or f"R{args.R}_{args.mode}"
OUT_DIR = os.path.expanduser(
    f'~/+Simulations/StagnantLid/aniso_dt_validate/{OUT_TAG}')
os.makedirs(OUT_DIR, exist_ok=True)

Ra = 1.0e7
theta_FK = float(np.log(1.0e4))


# Resume: find the latest snapshot under OUT_DIR
def _find_latest_snapshot():
    import glob, re
    fs = glob.glob(os.path.join(
        OUT_DIR, "step*.mesh.00000.h5"))
    if not fs:
        return None
    idxs = []
    for f in fs:
        m = re.search(r"step(\d+)\.mesh\.00000\.h5$",
                      os.path.basename(f))
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        return None
    s_max = max(idxs)
    return s_max, f"step{s_max:04d}"


resume_info = _find_latest_snapshot() if args.resume else None
if resume_info is not None:
    resume_step, resume_label = resume_info
    print(f"[{OUT_TAG}] resuming from {resume_label} "
          f"(step {resume_step})")
    mesh = uw.discretisation.Mesh(
        os.path.join(OUT_DIR, f"{resume_label}.mesh.00000.h5"))
else:
    # Load step-125 from the uniform run
    resume_step = 0
    resume_label = None
    mesh = uw.discretisation.Mesh(
        os.path.join(args.src_dir,
                     f"{args.src_stem}.mesh.00000.h5"))

T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")
V = uw.discretisation.MeshVariable(
    "V_v2p1", mesh, vtype=uw.VarType.VECTOR,
    degree=2, continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    "P_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=1, continuous=True, varsymbol="p")

if resume_label is not None:
    # Resume from previous snapshot in OUT_DIR
    T.read_timestep(resume_label, "T_v2p1", 0, outputPath=OUT_DIR)
    V.read_timestep(resume_label, "V_v2p1", 0, outputPath=OUT_DIR)
    # P file may or may not exist depending on the snapshot's
    # vintage; fall back to zero if absent.
    try:
        P.read_timestep(resume_label, "P_v2p1", 0,
                        outputPath=OUT_DIR)
    except Exception:
        print("  (no P in snapshot; zeroing)")
        P.data[...] = 0.0
    print(f"loaded resume snapshot: T=[{T.data.min():.3f},"
          f"{T.data.max():.3f}], |v|max="
          f"{float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}")
else:
    T.read_timestep(args.src_stem, "T_v2p1", 0,
                    outputPath=args.src_dir)
    V.read_timestep(args.src_stem, "V_v2p1", 0,
                    outputPath=args.src_dir)
    P.read_timestep(args.src_stem, "P_v2p1", 0,
                    outputPath=args.src_dir)
    print(f"loaded step-125 uniform state: T=[{T.data.min():.3f},"
          f"{T.data.max():.3f}], |v|max="
          f"{float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}")

# Adapt if R > 1 — only when starting fresh (not on resume,
# the resume snapshot is already on the adapted mesh)
if args.R > 1.0 and resume_label is None:
    rho = uw.meshing.metric_density_from_gradient(
        mesh, T, amp=8.0, lo_percentile=50.0,
        hi_percentile=97.0, name=f"validate_R{args.R}")
    old_X = np.asarray(mesh.X.coords).copy()
    print(f"adapting to R={args.R}...")
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(resolution_ratio=args.R,
                           relax=0.2, n_outer=12))
    # FE-remap T to adapted mesh; explicit-zero V,P per the
    # post-adapt rule (NOT remap; NOT Lagrangian-carry).
    new_Tx = np.asarray(T.coords).copy()
    old_T = np.asarray(T.data).copy()
    new_X = np.asarray(mesh.X.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T
    rT = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = rT
    V.data[...] = 0.0
    P.data[...] = 0.0
else:
    print("uniform mesh (R=1.0) — no adapt")

# Build Stokes + AdvDiff (matches stagnant_lid_uniform.py)
X = mesh.CoordinateSystem.X
r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
unit_r = mesh.CoordinateSystem.unit_e_0

stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = (
    sympy.exp(theta_FK * (1 - T.sym[0])))
stokes.tolerance = 1.0e-5
stokes.penalty = 0.0
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
KFS = 1.0e6
fs = (KFS * V.sym.dot(unit_r) * unit_r)
stokes.add_natural_bc(fs, mesh.boundaries.Upper.name)
T_cond = sympy.log(r_sym) / sympy.log(0.5)
stokes.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r

adv = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=T, V_fn=V.sym, verbose=False,
    theta=1.0, monotone_mode='clamp')
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.tolerance = 1.0e-4
adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)


# Direction-aware CFL helper
tris = _tri_cells(mesh.dm)
if args.R > 1.0:
    # Need to redo tris after adapt (DOF indices preserved but
    # coords moved — tris are indices into mesh.X.coords, which
    # is correct)
    pass


def estimate_dt_direction_aware():
    coords = np.asarray(mesh.X.coords)
    cur_tris = _tri_cells(mesh.dm)
    centroids = coords[cur_tris].mean(axis=1)
    v_per_cell = np.asarray(uw.function.evaluate(
        V.sym, centroids)).reshape(centroids.shape[0], 2)
    vmag = np.linalg.norm(v_per_cell, axis=1)
    vhat = np.where(vmag[:, None] > 0,
                    v_per_cell / np.maximum(vmag[:, None], 1e-30),
                    0.0)
    Vverts = coords[cur_tris]
    D = Vverts - centroids[:, None, :]
    s = np.einsum('cvd,cd->cv', D, vhat)
    h_eff = s.max(axis=1) - s.min(axis=1)
    # Active cells only (avoid div-by-zero)
    active = vmag > vmag.max() * 1e-6
    if not active.any():
        return 1.0e-3
    dts = (h_eff[active] / vmag[active])
    return float(args.cfl * dts.min())


# Nu / vrms shells
Q_COND = 2.0 * np.pi / np.log(1.0 / 0.5)

# Create the Nu projection MeshVariable + solver ONCE outside
# the loop. Creating a new MV + Projection every step (as the
# original sl_uniform/sl_warm_puzzle scripts did) leaks ~50MB
# of PETSc SNES/KSP/PC hierarchy per call — 100 steps = 5 GB,
# and the page-fault / cache-miss costs start dominating around
# step 70 (per the slowdown we saw in the first long run).
_qf = uw.discretisation.MeshVariable(
    "qr_flux", mesh, vtype=uw.VarType.SCALAR,
    degree=2, continuous=True)
_qproj = uw.systems.Projection(mesh, _qf)
_qproj.smoothing = 0.0
_er = mesh.CoordinateSystem.unit_e_0
_qsym = ((V.sym[0] * _er[0] + V.sym[1] * _er[1]) * T.sym[0]
         - (T.sym[0].diff(X[0]) * _er[0]
            + T.sym[0].diff(X[1]) * _er[1]))
_qproj.uw_function = _qsym
_TH_EVAL = np.linspace(0, 2 * np.pi, 720, endpoint=False)
_R_EVAL = 0.75
_PTS_EVAL = np.column_stack(
    [_R_EVAL * np.cos(_TH_EVAL), _R_EVAL * np.sin(_TH_EVAL)])


def _shell_flow():
    _qproj.solve()
    q = np.asarray(uw.function.evaluate(
        _qf.sym[0], _PTS_EVAL)).reshape(-1)
    return float(q.mean() * _R_EVAL * 2.0 * np.pi)


# Cold solve to refresh V on the (possibly adapted) mesh.
# On resume we already have a valid V (from snapshot) — use it
# as warm guess; otherwise (cold start) and post-adapt, zero
# V,P and solve cold.
print(f"[{OUT_TAG}] initial Stokes solve ...", flush=True)
t0 = time.time()
_use_zero = (args.R > 1.0) and (resume_label is None)
stokes.solve(zero_init_guess=_use_zero)
print(f"  init done {time.time()-t0:.1f}s "
      f"|v|max={float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}",
      flush=True)


def snapshot(step):
    label = "init" if step == 0 else f"step{step:04d}"
    mesh.write_timestep(
        filename=label, index=0, outputPath=OUT_DIR,
        meshVars=[T, V, P], meshUpdates=True, create_xdmf=True)


# Initialise step counter / sim time / history from previous
# run if resuming, else start fresh.
hist = []
t_sim = 0.0
if resume_label is not None:
    # Restore history from disk
    hpath = os.path.join(OUT_DIR, "history.npz")
    if os.path.exists(hpath):
        z = np.load(hpath)
        for i in range(len(z['step'])):
            if int(z['step'][i]) > resume_step:
                continue   # drop later steps (defensive)
            hist.append((int(z['step'][i]),
                         float(z['t'][i]),
                         float(z['dt'][i]),
                         float(z['wall'][i]),
                         float(z['vrms'][i]),
                         float(z['Nu'][i]),
                         float(z['Tmin'][i]),
                         float(z['Tmax'][i])))
        if hist:
            t_sim = hist[-1][1]
            print(f"  resumed history: {len(hist)} entries, "
                  f"t={t_sim:.5f}")
else:
    snapshot(0)
START_STEP = resume_step + 1 if resume_label else 1
END_STEP = (resume_step if resume_label else 0) + args.n_steps + 1
print(f"[{OUT_TAG}] mode={args.mode} cfl={args.cfl} "
      f"running steps {START_STEP}..{END_STEP - 1} "
      f"(snapshot every {args.snapshot_every})")
print(f"{'step':>5} {'t':>9} {'dt':>10} {'wall':>7} "
      f"{'vrms':>10} {'Nu':>8} {'T[min,max]':>22} {'status'}")

for s in range(START_STEP, END_STEP):
    t_step_0 = time.time()
    if args.mode == 'aniso':
        dt = estimate_dt_direction_aware()
    else:
        dt = adv.estimate_dt()

    try:
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
    except Exception as e:
        print(f"  EXCEPTION at step {s}: {e}", flush=True)
        break
    t_sim += dt
    wall = time.time() - t_step_0

    T_arr = T.data[:, 0]
    if np.isnan(T_arr).any() or np.isinf(T_arr).any():
        print(f"  step {s}: NaN/Inf in T — ABORT", flush=True)
        break
    Tmin, Tmax = float(T_arr.min()), float(T_arr.max())
    if Tmax > 1.1 or Tmin < -0.1:
        print(f"  step {s}: T overshoot [{Tmin:+.4f},{Tmax:+.4f}]"
              f" — ABORT", flush=True)
        break

    v_sq = np.asarray(uw.function.evaluate(
        V.sym.dot(V.sym), mesh.X.coords))
    vrms = float(np.sqrt(np.mean(v_sq)))
    Nu = _shell_flow() / Q_COND

    hist.append((s, t_sim, dt, wall, vrms, Nu, Tmin, Tmax))
    # Atomic incremental history dump every step (never lose
    # progress to a crash; the live plot can read it any time).
    _h = np.asarray(hist)
    np.savez(os.path.join(OUT_DIR, "history.npz"),
             step=_h[:, 0], t=_h[:, 1], dt=_h[:, 2],
             wall=_h[:, 3], vrms=_h[:, 4], Nu=_h[:, 5],
             Tmin=_h[:, 6], Tmax=_h[:, 7])
    if s % args.snapshot_every == 0:
        snapshot(s)
    if s % args.log_every == 0 or s == 1:
        print(f"{s:>5d} {t_sim:>9.5f} {dt:>10.3e} "
              f"{wall:>6.2f}s {vrms:>10.3e} {Nu:>+8.3f} "
              f"[{Tmin:+.3f},{Tmax:+.3f}]  ok", flush=True)

    # Sanity: blowup detection
    if abs(vrms) > 1.0e6 or abs(Nu) > 50:
        print(f"  step {s}: blowup vrms={vrms:.2e} Nu={Nu:.2f}"
              f" — ABORT", flush=True)
        break


# Final snapshot (always)
if hist:
    snapshot(int(hist[-1][0]))
print(f"saved {OUT_DIR}/history.npz "
      f"({len(hist)} steps completed)")
