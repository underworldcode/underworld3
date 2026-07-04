"""Sweep Stokes-solver presets on the FK stagnant-lid problem,
holding T fixed at the step-125 settled state. Two meshes:
  (a) uniform res-16
  (b) adapted res-16 (equidist R=1.5 |∇T| metric)
For each preset (default, gamg-n1/thr/noagr/sor/full/noagrsor,
direct) report:
  - cold solve from zero IC: SNES reason, its, wall, ‖F0‖→‖F‖
  - warm solve from the stored V,P: same metrics
Designed to populate the catalogue's "test on a harder PDE"
follow-up with concrete data.
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import numpy as np
import sympy

import underworld3 as uw


PRESETS = {
    'default':       {},
    'gamg-n1':       {'pc_gamg_agg_nsmooths': 1},
    'gamg-thr':      {'pc_gamg_threshold': 0.02,
                      'pc_gamg_threshold_scale': 0.5},
    'gamg-noagr':    {'pc_gamg_aggressive_coarsening': 0},
    'gamg-sor':      {'mg_levels_ksp_type': 'richardson',
                      'mg_levels_pc_type': 'sor',
                      'mg_levels_ksp_max_it': 2},
    'gamg-noagrsor': {'pc_gamg_aggressive_coarsening': 0,
                      'mg_levels_ksp_type': 'richardson',
                      'mg_levels_pc_type': 'sor',
                      'mg_levels_ksp_max_it': 2},
    'gamg-full':     {'pc_gamg_agg_nsmooths': 1,
                      'pc_gamg_threshold': 0.02,
                      'pc_gamg_threshold_scale': 0.5,
                      'pc_gamg_aggressive_coarsening': 0,
                      'mg_levels_ksp_type': 'richardson',
                      'mg_levels_pc_type': 'sor',
                      'mg_levels_ksp_max_it': 2},
    'direct':        {'ksp_type': 'preonly',
                      'pc_type': 'lu',
                      'pc_factor_mat_solver_type': 'mumps',
                      'mat_mumps_icntl_24': 1},
}


p = argparse.ArgumentParser()
p.add_argument('--src-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'uniform_res16_Ra1e7_dEta1e4'),
               help='snapshot directory (uniform or adapted)')
p.add_argument('--src-stem', type=str, default=None,
               help='file stem (auto-detected if omitted)')
p.add_argument('--src-step', type=int, default=125)
p.add_argument('--Ra', type=float, default=1.0e7)
p.add_argument('--delta-eta', type=float, default=1.0e4)
p.add_argument('--stokes-tol', type=float, default=1.0e-5)
p.add_argument('--presets', type=str,
               default=','.join(PRESETS.keys()))
p.add_argument('--tag', type=str, default=None,
               help='label for this sweep (defaults to dir name)')
p.add_argument('--snes-atol-auto', action='store_true',
               help='Set snes_atol = rtol · ‖F(x=0)‖ on each '
                    'fresh solver (catalogue design-note fix). '
                    'When enabled, FNORM_ABS path becomes live '
                    'and warm guesses below the problem scale '
                    'converge in 0 Newton iterations.')
args = p.parse_args()

theta_FK = float(np.log(args.delta_eta))


# ---------------------- locate snapshot -------------------------

src_dir = args.src_dir
if args.src_stem is None:
    # heuristics: adapted/ → "adapted"; uniform → sl_<tag>_step<N>
    base = os.path.basename(src_dir.rstrip('/'))
    cand_adapted = os.path.join(src_dir, "adapted.mesh.00000.h5")
    cand_uniform = os.path.join(
        src_dir, f"sl_{base}_step{args.src_step:05d}.mesh.00000.h5")
    if os.path.exists(cand_adapted):
        stem = "adapted"
    elif os.path.exists(cand_uniform):
        stem = f"sl_{base}_step{args.src_step:05d}"
    else:
        sys.exit(f"can't auto-detect stem in {src_dir}")
else:
    stem = args.src_stem
print(f"loading {stem} from {src_dir}")

sweep_tag = args.tag or os.path.basename(src_dir.rstrip('/'))


# ---------------------- build problem ---------------------------

mesh = uw.discretisation.Mesh(
    os.path.join(src_dir, f"{stem}.mesh.00000.h5"))
X = mesh.CoordinateSystem.X
r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
unit_r = mesh.CoordinateSystem.unit_e_0

T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
V = uw.discretisation.MeshVariable(
    "V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2,
    continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    "P_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True, varsymbol="p")

T.read_timestep(stem, "T_v2p1", 0, outputPath=src_dir)
V.read_timestep(stem, "V_v2p1", 0, outputPath=src_dir)
P.read_timestep(stem, "P_v2p1", 0, outputPath=src_dir)
V_warm = np.asarray(V.data).copy()
P_warm = np.asarray(P.data).copy()
print(f"  loaded T=[{T.data.min():.3f},{T.data.max():.3f}], "
      f"|v|max(warm)={float(np.sqrt(V.data[:,0]**2 + V.data[:,1]**2).max()):.2e}")


def build_stokes(preset_options, snes_atol=None):
    """Build a fresh Stokes solver with the given PETSc options.
    A fresh solver per preset isolates option leakage."""
    s = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta_FK * (1 - T.sym[0])))
    s.tolerance = args.stokes_tol
    s.penalty = 0.0
    # no-slip inner + free-slip outer (the trusted pair)
    s.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    KFS = 1.0e6
    fs_term = (KFS * V.sym.dot(unit_r) * unit_r)
    s.add_natural_bc(fs_term, mesh.boundaries.Upper.name)
    # T-fixed buoyancy (the canonical Boussinesq form on a log
    # conductive reference; T is not advecting in this sweep)
    T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
    s.bodyforce = args.Ra * (T.sym[0] - T_cond) * unit_r
    for k, vopt in preset_options.items():
        s.petsc_options[k] = vopt
    if snes_atol is not None:
        s.petsc_options["snes_atol"] = float(snes_atol)
    return s


# Compute the problem-scale residual ‖F(x=0)‖ once (independent of
# preset; depends only on RHS and the operator structure).
F0 = None
atol = None
if args.snes_atol_auto:
    print("computing ‖F(x=0)‖ for snes_atol...", flush=True)
    V.data[...] = 0.0
    P.data[...] = 0.0
    _probe = build_stokes(PRESETS['default'])
    try:
        # First solve creates the SNES object (snes is None until
        # then). Then enable convergence-history capture and
        # re-solve to record ‖F(x=0)‖.
        _probe.solve(zero_init_guess=True)
        _probe.snes.setConvergenceHistory(reset=True)
        V.data[...] = 0.0
        P.data[...] = 0.0
        _probe.solve(zero_init_guess=True)
        _rh, _ = _probe.snes.getConvergenceHistory()
        F0 = float(_rh[0]) if _rh is not None and len(_rh) else None
    except Exception as e:
        print(f"  ‖F0‖ probe failed: {e}", flush=True)
        F0 = None
    if F0 is not None and F0 > 0.0:
        atol = float(args.stokes_tol) * F0
        print(f"  ‖F(x=0)‖ = {F0:.4e}  ⇒  "
              f"snes_atol = {atol:.4e}  (rtol={args.stokes_tol:.0e})",
              flush=True)
    else:
        print("  WARN: ‖F0‖ unavailable — running without atol",
              flush=True)


def diag_F0(stokes):
    """Compute ‖F(x=0)‖ — the problem-scale residual.  Useful as
    a 'success target' yardstick."""
    try:
        # No standard API: snes residual function is private.
        # Skip if unavailable.
        return None
    except Exception:
        return None


def run_one(preset_name, mode):
    """Run a single Stokes solve, recording outcome metrics.

    mode='cold' → zero IC; 'warm' → V,P from snapshot.
    """
    # Restore the appropriate IC
    if mode == 'cold':
        V.data[...] = 0.0
        P.data[...] = 0.0
        zero_init = True
    else:
        V.data[...] = V_warm
        P.data[...] = P_warm
        zero_init = False

    s = build_stokes(PRESETS[preset_name], snes_atol=atol)
    t0 = time.time()
    try:
        s.solve(zero_init_guess=zero_init)
        ok_call = True
        err = None
    except Exception as e:
        ok_call = False
        err = repr(e)
    wall = time.time() - t0

    # SNES outcome
    try:
        sn = s.snes
        reason = int(sn.getConvergedReason())
        its = int(sn.getIterationNumber())
        try:
            ksp = sn.getKSP()
            ksp_its = int(ksp.getIterationNumber())
        except Exception:
            ksp_its = None
    except Exception:
        reason, its, ksp_its = None, None, None

    vmax = float(np.sqrt(V.data[:, 0] ** 2
                         + V.data[:, 1] ** 2).max())
    return dict(preset=preset_name, mode=mode,
                ok_call=ok_call, err=err, wall=wall,
                snes_reason=reason, snes_its=its,
                ksp_its=ksp_its, vmax=vmax)


# ---------------------- sweep ----------------------------------

results = []
presets = args.presets.split(',')
print(f"\n[{sweep_tag}]  sweeping {len(presets)} presets × "
      f"{{cold, warm}}\n")
hdr = (f"{'preset':>14} {'mode':>5} {'reason':>6} {'its':>4} "
       f"{'ksp':>4} {'wall':>7} {'|v|max':>10}  {'ok'}")
print(hdr)
print("-" * len(hdr))

for ps in presets:
    if ps not in PRESETS:
        print(f"  unknown preset: {ps}, skip")
        continue
    for mode in ('cold', 'warm'):
        r = run_one(ps, mode)
        results.append(r)
        rstr = (f"{r['preset']:>14} {r['mode']:>5} "
                f"{str(r['snes_reason']):>6} "
                f"{str(r['snes_its']):>4} "
                f"{str(r['ksp_its']):>4} "
                f"{r['wall']:>6.2f}s {r['vmax']:>10.2e}  "
                f"{'ok' if r['ok_call'] else 'EXC'}")
        if r['err']:
            rstr += f"  {r['err'][:80]}"
        print(rstr, flush=True)


# ---------------------- save -----------------------------------

out_npz = os.path.join(src_dir, f"solver_sweep_{sweep_tag}.npz")
np.savez(out_npz,
         preset=np.asarray([r['preset'] for r in results]),
         mode=np.asarray([r['mode'] for r in results]),
         snes_reason=np.asarray([r['snes_reason'] for r in results]),
         snes_its=np.asarray([r['snes_its'] for r in results]),
         ksp_its=np.asarray([r['ksp_its'] for r in results]),
         wall=np.asarray([r['wall'] for r in results]),
         vmax=np.asarray([r['vmax'] for r in results]),
         ok_call=np.asarray([r['ok_call'] for r in results]))
print(f"\nsaved {out_npz}")
