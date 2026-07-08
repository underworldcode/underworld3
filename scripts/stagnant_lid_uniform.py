"""Frank-Kamenetskii stagnant-lid convection, uniform mesh.

Stage-1 probe for the parallel-safe + scalable error-estimator
arc (B in the catalogue follow-ups). Purpose: characterise the
realised T- and η-gradient fields on a baseline uniform mesh,
so we can choose mover bunching strategy + GAMG-stress test
parameters for the adaptive run that follows.

Viscosity law (FK):
    η(T) = exp(θ · (1 - T)),  θ = ln(Δη)
With Δη = 1e4 ⇒ θ ≈ 9.21:
    T=0 (cold, outer) → η = Δη = 1e4     (the lid)
    T=1 (hot,  inner) → η = 1             (active layer reference)
This normalisation pins the active-layer viscosity at the
standard value of unity, so velocities and the pressure-velocity
coupling have a sensible scale (effective Ra ≈ Ra_input in the
active layer, with stiffness contrast Δη on top).

Annulus geometry, T_inner=1 / T_outer=0, free-slip outer +
no-slip inner (the existing trusted BC pair).  Ra=1e6 default.

Output: ~/+Simulations/StagnantLid/<tag>/ (XDMF + UW h5; viewable
in Finder / ParaView).  History saved as atomic .npz alongside
the checkpoints.
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import numpy as np
import sympy

import underworld3 as uw


# ---------------------- CLI -------------------------------------

p = argparse.ArgumentParser()
p.add_argument('--res', type=int, default=32,
               help='cellSize = 1/res')
p.add_argument('--n-steps', type=int, default=1500)
p.add_argument('--Ra', type=float, default=1.0e6)
p.add_argument('--delta-eta', type=float, default=1.0e4,
               help='η contrast: FK θ = ln(Δη)')
p.add_argument('--diffusion-theta', type=float, default=1.0,
               help='AdvDiff timestepping θ; 1.0 = BE (most '
                    'stable at high Ra, recommended for high-Ra)')
p.add_argument('--monotone-mode', type=str, default='clamp')
p.add_argument('--t-degree', type=int, default=3)
p.add_argument('--v-degree', type=int, default=2)
p.add_argument('--p-degree', type=int, default=1)
p.add_argument('--stokes-tol', type=float, default=1.0e-5)
p.add_argument('--stokes-snes-opt', type=str, default='direct',
               choices=['default', 'direct', 'gamg-noagr',
                        'gamg-noagrsor', 'gamg-full'],
               help='Stokes solver preset. Catalogue: at Δη=1e4 '
                    'default GAMG aggregation fails (DIVERGED_LINE_'
                    'SEARCH on every step); "direct" (MUMPS LU) is '
                    'the gold-standard probe solver. GAMG variants '
                    'available for the parallel-scaling sweep.')
p.add_argument('--capture-every', type=int, default=50)
p.add_argument('--log-every', type=int, default=10)
p.add_argument('--tag', type=str, default=None,
               help='run tag (defaults to res/Ra/dEta string)')
p.add_argument('--outdir', type=str,
               default=os.path.expanduser('~/+Simulations/StagnantLid'))
args = p.parse_args()


tag = (args.tag if args.tag else
       f"uniform_res{args.res}_Ra{args.Ra:.0e}_dEta{args.delta_eta:.0e}"
       .replace('+0', '').replace('-0', '-'))
run_dir = os.path.join(args.outdir, tag)
os.makedirs(run_dir, exist_ok=True)

theta_FK = float(np.log(args.delta_eta))


# ---------------------- mesh + fields ---------------------------

r_inner, r_o = 0.5, 1.0
cellsize = 1.0 / args.res
qdeg = max(3, args.v_degree + 1)

mesh = uw.meshing.Annulus(
    radiusOuter=r_o, radiusInner=r_inner,
    cellSize=cellsize, qdegree=qdeg,
)
r, th = mesh.CoordinateSystem.R
unit_r = mesh.CoordinateSystem.unit_e_0

pair_tag = f"v{args.v_degree}p{args.p_degree}"
v = uw.discretisation.MeshVariable(
    f"V_{pair_tag}", mesh, vtype=uw.VarType.VECTOR,
    degree=args.v_degree, continuous=True,
    varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    f"P_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
    degree=args.p_degree, continuous=True, varsymbol="p")
T = uw.discretisation.MeshVariable(
    f"T_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
    degree=args.t_degree, continuous=True, varsymbol="T")


# ---------------------- Stokes with FK viscosity ----------------

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Direct symbolic FK law — the c-tensor build path no longer
# calls simplify() (per docs/historical-notes), so the JIT
# compiler differentiates this correctly for Newton Jacobians.
stokes.constitutive_model.Parameters.shear_viscosity_0 = (
    sympy.exp(theta_FK * (1 - T.sym[0])))
stokes.tolerance = args.stokes_tol
stokes.penalty = 0.0

# Stokes solver preset (catalogue: solver-strategies-catalogue.md)
_SNES_OPT = {
    'default':      {},                                  # newtonls+bt+GAMG
    'direct':       {'ksp_type': 'preonly',
                     'pc_type': 'lu',
                     'pc_factor_mat_solver_type': 'mumps',
                     'mat_mumps_icntl_24': 1},
    'gamg-noagr':   {'pc_gamg_aggressive_coarsening': 0},
    'gamg-noagrsor':{'pc_gamg_aggressive_coarsening': 0,
                     'mg_levels_ksp_type': 'richardson',
                     'mg_levels_pc_type': 'sor',
                     'mg_levels_ksp_max_it': 2},
    'gamg-full':    {'pc_gamg_agg_nsmooths': 1,
                     'pc_gamg_threshold': 0.02,
                     'pc_gamg_threshold_scale': 0.5,
                     'pc_gamg_aggressive_coarsening': 0,
                     'mg_levels_ksp_type': 'richardson',
                     'mg_levels_pc_type': 'sor',
                     'mg_levels_ksp_max_it': 2},
}
for _k, _vopt in _SNES_OPT[args.stokes_snes_opt].items():
    stokes.petsc_options[_k] = _vopt

# No-slip inner, free-slip outer (trusted BC pair, no Stokes
# nullspace). Lid forms at the outer (cold) boundary.
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
KFS = 1.0e6
fs_term = (KFS * v.sym.dot(unit_r) * unit_r)
stokes.add_natural_bc(fs_term, mesh.boundaries.Upper.name)

# Buoyancy: Ra · (T - T_cond) · r̂ (T_cond = the logarithmic
# annular conduction reference so a still-conductive state is
# force-balanced)
T_cond = sympy.log(r / r_o) / sympy.log(r_inner / r_o)
stokes.bodyforce = args.Ra * (T.sym[0] - T_cond) * unit_r


# ---------------------- AdvDiff ---------------------------------

adv = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=T, V_fn=v.sym, verbose=False,
    theta=args.diffusion_theta,
    monotone_mode=(None if args.monotone_mode in ('None', 'none')
                   else args.monotone_mode),
)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.tolerance = 1.0e-4
adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)


# ---------------------- IC --------------------------------------

# Logarithmic conduction + mode-5 perturbation
init_T = (
    0.01 * sympy.sin(5.0 * th)
    * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
    + T_cond
)
T.data[...] = np.asarray(uw.function.evaluate(
    init_T, T.coords)).reshape(-1, 1)


# ---------------------- diagnostics -----------------------------

Q_COND = 2.0 * np.pi / np.log(r_o / r_inner)  # total cond flow


def _shell_flow(T_field, V_field, r_eval, n_th=720):
    """Total radial heat flow through circle r=r_eval:
        ∮ (v_r T - ∂T/∂r) r dθ
    Projects the symbolic flux to a nodal field so the integrand
    is FE-consistent (not raw ∂T)."""
    X = mesh.CoordinateSystem.X
    er = mesh.CoordinateSystem.unit_e_0
    grad_T_r = (T_field.sym[0].diff(X[0]) * er[0]
                + T_field.sym[0].diff(X[1]) * er[1])
    vr = (V_field.sym[0] * er[0] + V_field.sym[1] * er[1])
    qsym = vr * T_field.sym[0] - grad_T_r
    qf = uw.discretisation.MeshVariable(
        f"_qr_{int(time.time() * 1e6) % 10 ** 8}", mesh,
        vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    proj = uw.systems.Projection(mesh, qf)
    proj.uw_function = qsym
    proj.smoothing = 0.0
    proj.solve()
    th_eval = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    pts = np.column_stack(
        [r_eval * np.cos(th_eval), r_eval * np.sin(th_eval)])
    q = np.asarray(uw.function.evaluate(qf.sym[0], pts)).reshape(-1)
    return float(q.mean() * r_eval * 2.0 * np.pi)


def _diagnostics():
    """vrms, mid-shell Nu, T extents, realised η extents."""
    # vrms over the mesh interior
    v_sq = np.asarray(uw.function.evaluate(
        v.sym.dot(v.sym), mesh.X.coords))
    vrms = float(np.sqrt(np.mean(v_sq)))
    # mid-shell Nu (avoids near-BL stencil under-resolution)
    Nu = _shell_flow(T, v, 0.5 * (r_inner + r_o)) / Q_COND
    T_arr = T.data[:, 0]
    eta_arr = np.exp(theta_FK * (1 - T_arr))
    return {
        'vrms': vrms, 'Nu': Nu,
        'Tmin': float(T_arr.min()), 'Tmax': float(T_arr.max()),
        'eta_min': float(eta_arr.min()),
        'eta_max': float(eta_arr.max()),
    }


def _capture(step, t_sim):
    label = f"step{step:05d}" if step > 0 else 'init'
    mesh.write_timestep(
        filename=f"sl_{tag}_{label}", index=0, outputPath=run_dir,
        meshVars=[T, v, P], meshUpdates=True,
        create_xdmf=True)


# ---------------------- main loop -------------------------------

print(f"=== stagnant lid uniform-mesh probe ===", flush=True)
print(f"  tag           : {tag}", flush=True)
print(f"  out           : {run_dir}", flush=True)
print(f"  Ra            : {args.Ra:.2e}", flush=True)
print(f"  Δη (target)   : {args.delta_eta:.2e} "
      f"⇒ FK θ = {theta_FK:.4f}", flush=True)
print(f"  res / cellsz  : {args.res} / {cellsize:.4f}", flush=True)
print(f"  T/V/P degree  : {args.t_degree}/{args.v_degree}/"
      f"{args.p_degree}", flush=True)
print(f"  AdvDiff θ     : {args.diffusion_theta} "
      f"({'BE' if args.diffusion_theta == 1.0 else 'CN' if args.diffusion_theta == 0.5 else 'mixed'})",
      flush=True)
print(f"  monotone mode : {args.monotone_mode}", flush=True)
print(f"  Stokes preset : {args.stokes_snes_opt}", flush=True)

t0_wall = time.time()
stokes.solve(zero_init_guess=True)
print(f"  initial Stokes solve done "
      f"({time.time() - t0_wall:.1f}s wall)", flush=True)

t_sim = 0.0
history = {
    'step': [], 't_sim': [], 'dt': [], 'wall': [],
    'vrms': [], 'Nu': [], 'Tmin': [], 'Tmax': [],
    'eta_min': [], 'eta_max': [],
}


def _save_history():
    np.savez(os.path.join(run_dir, f"sl_{tag}_history.npz"),
             **{k: np.asarray(v) for k, v in history.items()})


# Initial snapshot
_capture(0, t_sim)
d0 = _diagnostics()
print(f"  IC: vrms={d0['vrms']:.3e}  Nu={d0['Nu']:+.3f}  "
      f"T=[{d0['Tmin']:+.3f},{d0['Tmax']:+.3f}]  "
      f"η=[{d0['eta_min']:.2e},{d0['eta_max']:.2e}]",
      flush=True)

for s in range(1, args.n_steps + 1):
    t_step_0 = time.time()
    dt = adv.estimate_dt()
    adv.solve(timestep=dt, zero_init_guess=False)
    stokes.solve(zero_init_guess=False)
    t_sim += dt
    wall = time.time() - t_step_0

    if s % args.log_every == 0 or s == 1:
        d = _diagnostics()
        history['step'].append(s)
        history['t_sim'].append(t_sim)
        history['dt'].append(dt)
        history['wall'].append(wall)
        for k in ('vrms', 'Nu', 'Tmin', 'Tmax',
                  'eta_min', 'eta_max'):
            history[k].append(d[k])
        _save_history()
        print(f"  step {s:5d}  t={t_sim:7.4f}  Δt={dt:.2e}  "
              f"wall={wall:5.2f}s  vrms={d['vrms']:.2e}  "
              f"Nu={d['Nu']:+5.2f}  T=[{d['Tmin']:+.3f},"
              f"{d['Tmax']:+.3f}]  η=[{d['eta_min']:.1e},"
              f"{d['eta_max']:.1e}]",
              flush=True)

    if args.capture_every > 0 and s % args.capture_every == 0:
        _capture(s, t_sim)

print(f"=== done ({time.time() - t0_wall:.1f}s wall, "
      f"sim t={t_sim:.4f}) ===", flush=True)
_capture(args.n_steps, t_sim)
_save_history()
