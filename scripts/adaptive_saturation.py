"""One model of the 3-way saturation experiment (launch 3 in
parallel: --model ref24 | u16 | a16).

Ra=1e5 annulus convection (no-slip inner / free-slip outer — the
trusted benchmark BC, no Stokes nullspace). Run PAST the
exponential overshoot into the settled-Nu regime (this is an
exponentially-growing instability — the meaningful comparison is
the saturated state, not the perturbation-sensitive exponential
phase). Stop on a Nu-settle detector or a hard step/time cap.

  ref24 : uniform res-24 (the reference)
  u16   : uniform res-16
  a16   : res-16 + anisotropic adaptation every ADAPT_EVERY steps
          with the validated LOCAL-FE interp remap (topology-
          preserving + fixed domain ⇒ old P3 field evaluated at
          new nodes via uw.function.evaluate; max fidelity, no
          kd-tree).

Checkpointed regularly: UW write_timestep h5 (meshUpdates=True so
each checkpoint is self-contained for the pyvista plotter) + an
atomic per-step history npz. Plot with adaptive_saturation_plot.py
any time (reads partial progress).
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True,
               choices=["ref24", "u16", "a16", "a16p", "a16s",
                        "a16x"])
p.add_argument("--Ra", type=float, default=1.0e5)
p.add_argument("--adapt-every", type=int, default=5)
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--t-end", type=float, default=0.6)
p.add_argument("--ckpt-every", type=int, default=20)
p.add_argument("--resume", action="store_true",
               help="warm-start from this model's last checkpoint "
                    "+ history and run --max-steps MORE steps "
                    "(settle detector disabled — run the full "
                    "extension; for pushing vrms to steady state)")
args = p.parse_args()

RES = 24 if args.model == "ref24" else 16
ADAPT = args.model in ("a16", "a16p", "a16s", "a16x")
PRISTINE = args.model in ("a16p", "a16s", "a16x")  # from X0

# Per-model metric strength. a16p = the conservative validated
# defaults (was tuned vs the now-removed cumulative over-
# compression). a16s = "more aggressive gradient following": the
# documented clean-but-strong Pareto corner (cap 2→4 needs a
# gentler relax + more n_outer per the validation arc; amp 8→16).
# Pristine re-mesh keeps each event a single uniform→graded map,
# so the static single-adaptation Pareto applies (no compounding).
_MP = {
    "a16":  dict(amp=8.0,  aniso_cap=2.0, relax=0.2,  n_outer=8),
    "a16p": dict(amp=8.0,  aniso_cap=2.0, relax=0.2,  n_outer=8),
    "a16s": dict(amp=16.0, aniso_cap=4.0, relax=0.05, n_outer=25),
    # a16x = "slightly more aggressive bunching": amp 16→24 (the
    # density-bunching intensity). aniso_cap kept at 4 — it is the
    # binding stability lever (≥6 folds); the eigen-clamp caps
    # worst-case compression regardless of amp, and pristine
    # re-mesh prevents compounding, so the higher amp just
    # saturates the clamp over a broader band (stronger pull, same
    # quality floor).
    "a16x": dict(amp=24.0, aniso_cap=4.0, relax=0.05, n_outer=25),
}
MP = _MP.get(args.model, _MP["a16p"])
r_inner, r_o = 0.5, 1.0
DIR = "/tmp/metric_mesh/sat"
os.makedirs(DIR, exist_ok=True)
TAG = args.model
HIST = f"{DIR}/sat_{TAG}_hist.npz"


def build():
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=1.0 / RES, qdegree=3)
    r, th = mesh.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(
        "V", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    P = uw.discretisation.MeshVariable(
        "P", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    T = uw.discretisation.MeshVariable(
        "T", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v,
                               pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    unit_r = mesh.CoordinateSystem.unit_e_0
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_natural_bc(1.0e6 * v.sym.dot(unit_r) * unit_r,
                          mesh.boundaries.Upper.name)
    T_cond = (r_o - r) / (r_o - r_inner)
    stokes.bodyforce = args.Ra * (T.sym[0] - T_cond) * unit_r
    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=v.sym, verbose=False,
        theta=0.5, monotone_mode="clamp")
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.tolerance = 1.0e-4
    adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
    adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)
    init_t = (0.01 * sympy.sin(5.0 * th)
              * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
              + (r_o - r) / (r_o - r_inner))
    T.data[...] = np.asarray(uw.function.evaluate(
        init_t, T.coords)).reshape(-1, 1)
    return mesh, v, P, T, stokes, adv


# Analytic steady-conduction heat flux through the OUTER boundary
# of the annulus. The true conductive solution of ∇²T=0 with
# T(R_i)=1, T(R_o)=0 is LOGARITHMIC (NOT the linear slab profile
# the buoyancy reference uses): T_cond = ln(r/R_o)/ln(R_i/R_o) ⇒
# Q_cond = -∮_outer ∇T_cond·n dS = 2π / ln(R_o/R_i).
_Q_COND = 2.0 * np.pi / np.log(r_o / r_inner)


_NU_CACHE = {}
_R_MID = 0.5 * (r_inner + r_o)


def nusselt(mesh, T, v=None, h=None):
    r"""Nu = total radial heat flow through an INTERIOR shell /
    conductive flow. Total flux q_r = v_r·T - ∂T/∂r is projected
    to a nodal field (FE-consistent) and integrated on the
    mid-shell r=(R_i+R_o)/2 — robust: at steady state the flow is
    shell-independent (conservation) and the interior shell is
    immune to thermal-BL resolution (unlike the boundary ∂T/∂r).
    Validated (scripts/_nu_proper.py): analytic conduction ⇒
    Nu=1.0000 on every shell; settled checkpoints agree with the
    boundary method to ~1-2% and across shells. Q_cond = total
    annular log-conduction flow 2π/ln(R_o/R_i) (Nu→1 at
    conduction; verified). Cached per mesh (called every step).
    v required for the advective term; omit ⇒ diffusive only."""
    key = id(mesh)
    cache = _NU_CACHE.get(key)
    if cache is None:
        qf = uw.discretisation.MeshVariable(
            f"nu_qr_{key:x}", mesh, vtype=uw.VarType.SCALAR,
            degree=2, continuous=True)
        proj = uw.systems.Projection(mesh, qf)
        proj.smoothing = 0.0
        X = mesh.CoordinateSystem.X
        er = mesh.CoordinateSystem.unit_e_0
        gradT_r = (T.sym[0].diff(X[0]) * er[0]
                   + T.sym[0].diff(X[1]) * er[1])
        vr = ((v.sym[0] * er[0] + v.sym[1] * er[1])
              if v is not None else sympy.Integer(0))
        proj.uw_function = vr * T.sym[0] - gradT_r
        _NU_CACHE[key] = (qf, proj)
    else:
        qf, proj = cache
    proj.solve()
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    pts = np.column_stack([_R_MID * np.cos(th),
                           _R_MID * np.sin(th)])
    q = np.asarray(uw.function.evaluate(
        qf.sym[0], pts)).reshape(-1)
    return float(q.mean() * _R_MID * 2.0 * np.pi) / _Q_COND


def vrms(mesh, v):
    a = np.asarray(uw.function.evaluate(
        v.sym.dot(v.sym), mesh.X.coords))
    return float(np.sqrt(np.mean(a)))


def adapt_local_fe_interp(mesh, T, stokes):
    """Anisotropic adapt + local-FE remap (validated): old P3
    field evaluated at the new node positions; layout-invariant ⇒
    trivial restore, no migration; fixed domain ⇒ in-domain."""
    rho = metric_density_from_gradient(mesh, T, amp=MP["amp"],
                                       name="sat")
    old_X = np.asarray(mesh.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(aniso_cap=MP["aniso_cap"],
                           relax=MP["relax"],
                           n_outer=MP["n_outer"]))
    new_X = np.asarray(mesh.X.coords).copy()
    new_Tx = np.asarray(T.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T
    vals = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = vals
    # NO stokes here — the loop's single stokes.solve (now placed
    # AFTER adaptation) recomputes v on the adapted mesh; the old
    # in-adapt re-solve was redundant (v is stale-by-construction
    # in the segregated scheme; nothing reads it before that solve).


def adapt_pristine(mesh, T, stokes, X0, X0_Tx):
    """Re-adapt from the ORIGINAL mesh points each event (not the
    already-snuggled mesh) — the across-events analogue of making
    the metric Lagrangian-once within a call. Each event maps
    pristine X0 → graded(current T), so compression does NOT
    compound (minA/meanA bounded to single-adaptation quality
    instead of collapsing to ~0.07).

    Field handling stays the validated local-FE remap. Sequence:
    (1) put the *physical* T (currently on the previous graded
    mesh) onto pristine X0 by an FE evaluate at the pristine T-DOF
    coords; (2) build the metric from that pristine-mesh T;
    (3) mover baseline = pristine X0 ⇒ fresh non-compounding graded
    map; (4) FE-remap T onto the new graded mesh; (5) refresh v.
    """
    X_prev = np.asarray(mesh.X.coords).copy()
    T_prev = np.asarray(T.data).copy()
    # (1) physical T (mesh@X_prev, T_prev) → pristine X0 T-DOFs
    vals0 = np.asarray(uw.function.evaluate(
        T.sym[0], X0_Tx)).reshape(-1)
    mesh._deform_mesh(X0)
    T.data[:, 0] = vals0
    # (2) metric from the physical T now on the pristine mesh
    rho = metric_density_from_gradient(mesh, T, amp=MP["amp"],
                                       name="sat")
    # (3) mover baseline is pristine X0 (fresh, non-compounding)
    X0c = np.asarray(mesh.X.coords).copy()
    T0 = np.asarray(T.data).copy()
    smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(aniso_cap=MP["aniso_cap"],
                           relax=MP["relax"],
                           n_outer=MP["n_outer"]))
    new_X = np.asarray(mesh.X.coords).copy()
    new_Tx = np.asarray(T.coords).copy()
    # (4) FE-remap the pristine-mesh T onto the new graded mesh
    mesh._deform_mesh(X0c)
    T.data[...] = T0
    valsN = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = valsN
    # (5) NO stokes here — see adapt_local_fe_interp note; the
    # loop's single post-adaptation stokes.solve does it.


def save_hist(hist):
    a = np.array(hist, dtype=float)
    tmp = HIST + ".tmp.npz"
    np.savez(tmp, step=a[:, 0], t=a[:, 1], dt=a[:, 2],
             Nu=a[:, 3], vrms=a[:, 4])
    os.replace(tmp, HIST)


def settled(nu_hist):
    """Past the overshoot and flattened: Nu has exceeded the
    nonlinear threshold and the trailing window is flat."""
    if len(nu_hist) < 60:
        return False
    w = np.array(nu_hist[-40:])
    if np.max(np.abs(np.array(nu_hist))) < 3.0:
        return False                      # still pre-overshoot
    return (w.max() - w.min()) < 0.06 * abs(w.mean())


mesh, v, P, T, stokes, adv = build()
h = 1.0 / RES
# pristine reference captured once (mesh + T-DOF coords undeformed)
X0 = np.asarray(mesh.X.coords).copy()
X0_Tx = np.asarray(T.coords).copy()
if args.resume:
    import glob
    import re
    _fs = glob.glob(f"{DIR}/sat_{TAG}.mesh.T.*.h5")
    _idx = max(int(re.search(r"\.mesh\.T\.(\d+)\.h5$", f).group(1))
               for f in _fs)
    T.read_timestep(f"sat_{TAG}", "T", _idx, outputPath=DIR)
    v.read_timestep(f"sat_{TAG}", "V", _idx, outputPath=DIR)
    _z = np.load(HIST)
    hist = [[int(_z["step"][i]), float(_z["t"][i]),
             float(_z["dt"][i]), float(_z["Nu"][i]),
             float(_z["vrms"][i])] for i in range(len(_z["step"]))]
    STEP0 = int(_z["step"][-1])
    t_sim = float(_z["t"][-1])
    stokes.solve(zero_init_guess=False)   # sync v with loaded T
    print(f"=== sat {TAG} RESUME from step {STEP0} t={t_sim:.4f} "
          f"(+{args.max_steps} more steps, settle OFF — push "
          f"vrms to steady state) ===", flush=True)
else:
    stokes.solve(zero_init_guess=True)
    t_sim = 0.0
    hist = []
    STEP0 = 0
    print(f"=== sat {TAG} (res-{RES}, adapt={ADAPT}) "
          f"Ra={args.Ra:.0e} max_steps={args.max_steps} ===",
          flush=True)
for s in range(args.max_steps):
    STEP = STEP0 + s + 1
    dt = adv.estimate_dt()
    adv.solve(timestep=dt, zero_init_guess=False)
    # Loop-reorder fix: adapt BETWEEN adv.solve and the single
    # stokes.solve. The remesh+remap happens on the just-advected
    # T; the one stokes.solve below then recomputes v on the
    # adapted mesh (no redundant old-mesh solve, no in-adapt
    # re-solve). Physically equivalent end-of-step state, ~2 fewer
    # Stokes solves per adaptation step.
    if ADAPT and STEP % args.adapt_every == 0:
        if PRISTINE:
            adapt_pristine(mesh, T, stokes, X0, X0_Tx)
        else:
            adapt_local_fe_interp(mesh, T, stokes)
    stokes.solve(zero_init_guess=False)
    t_sim += dt
    Nu = nusselt(mesh, T, v)
    vr = vrms(mesh, v)
    hist.append([STEP, t_sim, dt, Nu, vr])
    if STEP % args.ckpt_every == 0 or (not args.resume and s == 0):
        save_hist(hist)
        mesh.write_timestep(f"sat_{TAG}", STEP, outputPath=DIR,
                            meshVars=[T, v], meshUpdates=True,
                            create_xdmf=False)
        tt = T.data[:, 0]
        print(f"  [{TAG}] step {STEP:3d} t={t_sim:.4f} "
              f"dt={dt:.2e} Nu={Nu:+.3f} vrms={vr:.3e} "
              f"T=[{tt.min():+.2f},{tt.max():+.2f}]", flush=True)
    if t_sim >= args.t_end:
        print(f"  [{TAG}] reached t_end={args.t_end}", flush=True)
        break
    if (not args.resume) and settled([hh[3] for hh in hist]):
        print(f"  [{TAG}] Nu settled at step {STEP} "
              f"t={t_sim:.4f} Nu≈{Nu:.2f}", flush=True)
        break
save_hist(hist)
mesh.write_timestep(f"sat_{TAG}", len(hist), outputPath=DIR,
                    meshVars=[T, v], meshUpdates=True,
                    create_xdmf=False)
print(f"  [{TAG}] DONE {len(hist)} steps, t={t_sim:.4f}, "
      f"Nu={hist[-1][3]:+.3f}, final ckpt idx={len(hist)}",
      flush=True)
