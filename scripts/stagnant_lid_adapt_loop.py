"""Stagnant-lid convection with PERIODIC adaptation in the time
loop. Adapts every K steps using the named strategy
(``--strategy med`` by default); the mover's
``skip_threshold`` is active so adapts get skipped when the
mesh is already aligned with the current metric.

Loop pattern per step:
  1. estimate dt
  2. if (step % adapt_every == 0): build metric, call mover
     with skip_threshold; FE-remap T, zero V,P if mesh moved
  3. solve advdiff
  4. solve Stokes (warm if no adapt this step; cold if adapt)
  5. log Nu, vrms, T extents
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import sympy
import underworld3 as uw


p = argparse.ArgumentParser()
p.add_argument('--src-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'uniform_res16_Ra1e7_dEta1e4'))
p.add_argument('--src-stem', type=str,
               default='sl_uniform_res16_Ra1e7_dEta1e4_step00125')
p.add_argument('--strategy', type=str, default='med',
               choices=list(uw.meshing.ADAPT_STRATEGIES.keys()))
p.add_argument('--adapt-every', type=int, default=5)
p.add_argument('--n-steps', type=int, default=100)
p.add_argument('--log-every', type=int, default=2)
p.add_argument('--snapshot-every', type=int, default=20)
p.add_argument('--out-tag', type=str, default=None)
p.add_argument('--resume', action='store_true')
p.add_argument('--grad-smooth-h0', type=float, default=0.0,
               help='gradient_smoothing_length expressed as a '
                    'multiple of mean h0 (background cell size). '
                    '0 = no smoothing; 2.0 = L = 2·h0 (the '
                    "production gradient-side de-noising).")
p.add_argument('--refinement', type=float, default=0.0,
               help='If > 0, use uw.meshing.follow_metric() with '
                    'this refinement value instead of the legacy '
                    'strategy-based path. coarsening="auto" '
                    '(= refinement^(1/d)) and metric='
                    '"front-following" are used. 0 = use the '
                    'legacy --strategy path.')
p.add_argument('--max-t', type=float, default=0.0,
               help='If > 0, stop the loop as soon as t_sim '
                    'reaches this value (in addition to the '
                    '--n-steps cap).')
p.add_argument('--from-perturbation', action='store_true',
               help='Start from the near-conductive initial '
                    'state (T_cond + small mode-5 perturbation, '
                    'V=P=0) instead of loading from --src-stem. '
                    'Builds a fresh Annulus(0.5, 1.0, '
                    'cellSize=1/16, qdegree=3) to match the '
                    'uniform-res16 setup.')
p.add_argument('--skip-threshold', type=float, default=-1.0,
               help='Override the adapt skip threshold. -1 (the '
                    'default) means use the strategy default '
                    '(typically 0.9). Set to a very high value '
                    '(e.g. 99) to never skip — adapt every '
                    '--adapt-every steps. 0 means always skip.')
p.add_argument('--dt-mult', type=float, default=1.0,
               help='Multiplier on estimate_dt (which returns '
                    'the single-cell crossing time, CFL=1). SLCN '
                    'is unconditionally stable, so multipliers '
                    '> 1 (e.g. 3-5) give larger physical-time '
                    'steps at modest accuracy cost. 1.0 is the '
                    'historic default.')
p.add_argument('--Ra', type=float, default=1.0e7,
               help='Rayleigh number (default 1e7).')
p.add_argument('--delta-eta', type=float, default=1.0e4,
               help='Frank-Kamenetskii viscosity contrast '
                    'eta(cold)/eta(hot). Default 1e4 (stiff '
                    'stagnant lid). 100 = much softer lid, more '
                    'dynamic flow.')
p.add_argument('--pert-mode', type=int, default=5,
               help='Azimuthal wavenumber of the initial T '
                    'perturbation. Mode 5 gives the classic '
                    'five-cell symmetric pattern; mode 1 breaks '
                    'symmetry, drives drifting / time-varying '
                    'convection.')
p.add_argument('--pert-amplitude', type=float, default=0.01,
               help='Amplitude of the initial T perturbation '
                    '(relative to T_cond ~ 1).')
args = p.parse_args()


tag = args.out_tag or f"adapt_loop_{args.strategy}_every{args.adapt_every}"
OUT_DIR = os.path.expanduser(
    f'~/+Simulations/StagnantLid/{tag}')
os.makedirs(OUT_DIR, exist_ok=True)

Ra = float(args.Ra)
theta_FK = float(np.log(float(args.delta_eta)))
STRAT = uw.meshing.ADAPT_STRATEGIES[args.strategy]
print(f"=== adaptive convection: strategy={args.strategy} "
      f"({STRAT['description']}) ===")
print(f"  every {args.adapt_every} steps, "
      f"skip_threshold={STRAT['skip_threshold']}, "
      f"R={STRAT['resolution_ratio']}")
print(f"  out: {OUT_DIR}")


# --- resume / fresh-start logic ---
def _latest_snapshot():
    import glob, re
    fs = glob.glob(os.path.join(OUT_DIR, "step*.mesh.00000.h5"))
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


resume_info = _latest_snapshot() if args.resume else None
if resume_info is not None:
    resume_step, resume_label = resume_info
    print(f"  resuming from {resume_label}")
    mesh = uw.discretisation.Mesh(
        os.path.join(OUT_DIR, f"{resume_label}.mesh.00000.h5"))
elif args.from_perturbation:
    resume_step = 0
    resume_label = None
    # Fresh Annulus matching the uniform-res16 setup.
    mesh = uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5,
        cellSize=1.0/16, qdegree=3)
else:
    resume_step = 0
    resume_label = None
    mesh = uw.discretisation.Mesh(
        os.path.join(args.src_dir,
                     f"{args.src_stem}.mesh.00000.h5"))

T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
V = uw.discretisation.MeshVariable(
    "V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2,
    continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    "P_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True, varsymbol="p")

if resume_label:
    T.read_timestep(resume_label, "T_v2p1", 0, outputPath=OUT_DIR)
    V.read_timestep(resume_label, "V_v2p1", 0, outputPath=OUT_DIR)
    try:
        P.read_timestep(resume_label, "P_v2p1", 0,
                        outputPath=OUT_DIR)
    except Exception:
        P.data[...] = 0.0
elif args.from_perturbation:
    # T_cond + amp · sin(m·θ) · sin(π(r-r_i)/(r_o-r_i))
    r_inner, r_o = 0.5, 1.0
    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0]**2 + X[1]**2)
    th_sym = sympy.atan2(X[1], X[0])
    T_cond = sympy.log(r_sym/r_o) / sympy.log(r_inner/r_o)
    init_T = (float(args.pert_amplitude)
              * sympy.sin(float(args.pert_mode) * th_sym)
              * sympy.sin(np.pi * (r_sym - r_inner)
                          / (r_o - r_inner))
              + T_cond)
    T.data[...] = np.asarray(uw.function.evaluate(
        init_T, T.coords)).reshape(-1, 1)
    V.data[...] = 0.0
    P.data[...] = 0.0
else:
    T.read_timestep(args.src_stem, "T_v2p1", 0,
                    outputPath=args.src_dir)
    V.read_timestep(args.src_stem, "V_v2p1", 0,
                    outputPath=args.src_dir)
    P.read_timestep(args.src_stem, "P_v2p1", 0,
                    outputPath=args.src_dir)
print(f"  loaded T=[{T.data.min():.3f},{T.data.max():.3f}]  "
      f"|v|max={float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}")


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
T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
stokes.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r

adv = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=T, V_fn=V.sym, verbose=False,
    theta=1.0, monotone_mode='clamp')
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.tolerance = 1.0e-4
adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)


# --- Nu evaluators ---
# Two variants:
#   _nu_surface()  surface heat flux on the cold (Upper) boundary
#                  via uw.maths.BdIntegral — Nu = 1 at conduction.
#                  The canonical thermal-convection Nusselt number.
#   _nu_midshell() advective+conductive radial flux integrated on a
#                  circle at r = R_EVAL via Projection + point sample.
#                  Cheaper, but susceptible to projection noise.
#
# Q_COND is the analytic ΔT=1 conductive surface flux on the annulus:
#   Q_COND = 2π / ln(R_o/R_i),  so Nu_conduction = 1.
Q_COND = 2.0 * np.pi / np.log(1.0 / 0.5)
_X = mesh.CoordinateSystem.X
_n = mesh.Gamma_N
# Outward conductive flux on the outer (Upper) cold boundary:
#   q_n = -∇T · n̂      (n̂ outward, T decreasing outward ⇒ q_n > 0)
_qn_outer = -(T.sym[0].diff(_X[0]) * _n[0]
              + T.sym[0].diff(_X[1]) * _n[1])
_bd_qn_upper = uw.maths.BdIntegral(
    mesh=mesh, fn=_qn_outer,
    boundary=mesh.boundaries.Upper.name)


def _nu_surface():
    """Surface Nusselt number via BdIntegral on the cold boundary."""
    return float(_bd_qn_upper.evaluate()) / Q_COND


# Legacy mid-shell variant, kept for cross-checking
_qf = uw.discretisation.MeshVariable(
    "qr_flux", mesh, vtype=uw.VarType.SCALAR,
    degree=2, continuous=True)
_qproj = uw.systems.Projection(mesh, _qf)
_qproj.smoothing = 0.0
_er = mesh.CoordinateSystem.unit_e_0
_qproj.uw_function = (
    (V.sym[0] * _er[0] + V.sym[1] * _er[1]) * T.sym[0]
    - (T.sym[0].diff(X[0]) * _er[0]
       + T.sym[0].diff(X[1]) * _er[1]))
_TH_EVAL = np.linspace(0, 2 * np.pi, 720, endpoint=False)
_R_EVAL = 0.75
_PTS_EVAL = np.column_stack([_R_EVAL * np.cos(_TH_EVAL),
                              _R_EVAL * np.sin(_TH_EVAL)])


def _nu_midshell():
    _qproj.solve()
    q = np.asarray(uw.function.evaluate(
        _qf.sym[0], _PTS_EVAL)).reshape(-1)
    return float(q.mean() * _R_EVAL * 2.0 * np.pi) / Q_COND


# Default Nu reported in the history is now the surface variant.
_nu = _nu_surface


def snapshot(step):
    label = "init" if step == 0 else f"step{step:04d}"
    mesh.write_timestep(filename=label, index=0,
                        outputPath=OUT_DIR,
                        meshVars=[T, V, P], meshUpdates=True,
                        create_xdmf=True)


def _adapt_step():
    """Build metric + invoke mover with skip_threshold; FE-remap
    T (V,P zeroed) if the mover actually moved nodes.
    Returns (moved, misalignment) tuple — misalignment is the
    current-mesh alignment score against the target metric BEFORE
    the adapt fires."""
    old_X = np.asarray(mesh.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    h0 = float(mesh._radii.mean())
    grad_L = (args.grad_smooth_h0 * h0
              if args.grad_smooth_h0 > 0 else None)
    # Resolve the effective skip threshold for THIS adapt
    if args.skip_threshold >= 0:
        sk = (None if args.skip_threshold > 10.0
              else args.skip_threshold)
    else:
        sk = STRAT["skip_threshold"]
    # Diagnostic: measure misalignment BEFORE adapting so we can
    # log it whether or not the adapt fires.
    coar_val = float(args.refinement) ** 0.5 if args.refinement > 0 else 1.0
    R = max(float(args.refinement), coar_val) if args.refinement > 0 else 1.0
    if args.refinement > 0:
        rho_diag = uw.meshing.metric_density_from_gradient(
            mesh, T, refinement=float(args.refinement),
            coarsening="auto", metric_choice="front-following",
            gradient_smoothing_length=grad_L, name="diag")
    else:
        rho_diag = uw.meshing.metric_density_from_gradient(
            mesh, T, strategy=args.strategy, name="diag",
            gradient_smoothing_length=grad_L)
    mm = uw.meshing.mesh_metric_mismatch(
        mesh, rho_diag, resolution_ratio=R)
    misalign = float(mm["misalignment"])
    print(f"  mismatch before adapt: misalignment={misalign:.3f} "
          f"(skip threshold {sk})", flush=True)
    if args.refinement > 0:
        moved = uw.meshing.follow_metric(
            mesh, T,
            refinement=args.refinement,
            coarsening="auto",
            metric="front-following",
            skip_threshold=sk,
            gradient_smoothing_length=grad_L,
            verbose=True,
        )
        new_X = np.asarray(mesh.X.coords).copy()
        if not moved:
            return False, misalign
    else:
        rho = uw.meshing.metric_density_from_gradient(
            mesh, T, strategy=args.strategy, name="loop",
            gradient_smoothing_length=grad_L)
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="anisotropic",
            strategy=args.strategy,
            method_kwargs=dict(relax=0.2, n_outer=12),
            verbose=True)
        new_X = np.asarray(mesh.X.coords).copy()
        if np.allclose(new_X, old_X):
            return False, misalign
    # FE-remap T; explicitly zero V,P post-adapt
    new_Tx = np.asarray(T.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T
    # global_evaluate: new (adapted) DOF coords may land in another rank's
    # subdomain, so the remap must resolve off-rank points (local evaluate
    # leaves stale T at the partition seams -> growing artefacts in parallel).
    rT = np.asarray(uw.function.global_evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = rT
    V.data[...] = 0.0
    P.data[...] = 0.0
    return True, misalign


# Initial Stokes solve
print("  initial Stokes solve...", flush=True)
t0 = time.time()
stokes.solve(zero_init_guess=False)
print(f"  init done {time.time()-t0:.1f}s "
      f"|v|max={float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}",
      flush=True)


hist = []
t_sim = 0.0
if resume_label:
    hpath = os.path.join(OUT_DIR, "history.npz")
    if os.path.exists(hpath):
        z = np.load(hpath)
        for i in range(len(z['step'])):
            if int(z['step'][i]) > resume_step:
                continue
            _mis = (float(z['misalignment'][i])
                    if 'misalignment' in z.files else float('nan'))
            hist.append((int(z['step'][i]),
                         float(z['t'][i]),
                         float(z['dt'][i]),
                         float(z['wall'][i]),
                         float(z['vrms'][i]),
                         float(z['Nu'][i]),
                         float(z['Tmin'][i]),
                         float(z['Tmax'][i]),
                         int(z['adapted'][i]),
                         _mis))
        if hist:
            t_sim = hist[-1][1]
            print(f"  resumed history: {len(hist)} entries, "
                  f"t={t_sim:.5f}")
else:
    snapshot(0)

START_STEP = resume_step + 1 if resume_label else 1
END_STEP = (resume_step if resume_label else 0) + args.n_steps + 1

print(f"  running steps {START_STEP}..{END_STEP - 1} "
      f"(snapshot every {args.snapshot_every}, "
      f"log every {args.log_every})")
print(f"{'step':>5} {'t':>9} {'dt':>10} {'wall':>7} "
      f"{'vrms':>10} {'Nu':>8} {'T[min,max]':>22} {'adapt'}")

n_adapt_skipped = 0
n_adapt_done = 0
for s in range(START_STEP, END_STEP):
    t_step_0 = time.time()
    did_adapt = False
    misalign = float('nan')
    if args.strategy != "off" and (s % args.adapt_every == 0):
        did_adapt, misalign = _adapt_step()
        if did_adapt:
            n_adapt_done += 1
        else:
            n_adapt_skipped += 1
    # Stokes BEFORE AdvDiff. Otherwise the AdvDiff step right
    # after an adapt uses V=0 (cold restart inside _adapt_step),
    # which causes a one-step pure-diffusion smearing of T at
    # the BL and a visible Nu dip (the artifact at t≈0.011 in
    # the previous run). With Stokes first, V is freshly
    # computed from the just-remapped T before AdvDiff uses it,
    # and the SLCN trace-back history stays consistent.
    try:
        stokes.solve(zero_init_guess=did_adapt)
        dt = adv.estimate_dt(direction_aware=True) * float(args.dt_mult)
        adv.solve(timestep=dt, zero_init_guess=False)
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
    Nu_val = _nu()

    hist.append((s, t_sim, dt, wall, vrms, Nu_val,
                 Tmin, Tmax, int(did_adapt), misalign))
    _h = np.asarray(hist)
    np.savez(os.path.join(OUT_DIR, "history.npz"),
             step=_h[:, 0], t=_h[:, 1], dt=_h[:, 2],
             wall=_h[:, 3], vrms=_h[:, 4], Nu=_h[:, 5],
             Tmin=_h[:, 6], Tmax=_h[:, 7], adapted=_h[:, 8],
             misalignment=_h[:, 9])
    if s % args.snapshot_every == 0:
        snapshot(s)
    if s % args.log_every == 0:
        print(f"{s:>5d} {t_sim:>9.5f} {dt:>10.3e} "
              f"{wall:>6.2f}s {vrms:>10.3e} {Nu_val:>+8.3f} "
              f"[{Tmin:+.3f},{Tmax:+.3f}]  "
              f"{'ADAPT' if did_adapt else ''}",
              flush=True)
    if args.max_t > 0 and t_sim >= args.max_t:
        print(f"  reached max_t={args.max_t} at step {s} "
              f"(t_sim={t_sim:.5f}) — STOPPING", flush=True)
        # Final snapshot for the movie
        if s % args.snapshot_every != 0:
            snapshot(s)
        break

print(f"=== done; adapts done={n_adapt_done}, "
      f"skipped={n_adapt_skipped} ===", flush=True)
if hist:
    snapshot(int(hist[-1][0]))
