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
                        "a16x", "a16y", "a16z", "a16c",
                        "a16c2", "a16c15", "a16e", "a16ed",
                        "a16r15", "a16r15p", "a16r15e",
                        "a16c2v", "a16r15r", "a16r15v",
                        "a16r15b","a16r15l2","a16r15tr","a16r15ko","a16r15a","a16r15d","a16r15_6","u16_6",
                        "a16r15-n1","a16r15-thr","a16r15-noagr","a16r15-sor","a16r15-full","a16r15-noagrsor",
                        "a16r15-n1-c","a16r15-thr-c","a16r15-noagr-c",
                        "a16r15-sor-c","a16r15-full-c","a16r15-noagrsor-c",
                        "a16r15-base-redo"])
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
p.add_argument("--resume-from", type=int, default=0,
               help="restart from THIS checkpoint index instead "
                    "of the max (0=max; e.g. 50 to test "
                    "path-dependence from a clean pre-failure "
                    "state)")
p.add_argument("--src-tag", type=str, default="",
               help="read the resume checkpoint+hist from this "
                    "tag, but write outputs under --model's tag "
                    "(so a probe can restart off another run "
                    "without clobbering its files)")
p.add_argument("--snes-debug", action="store_true",
               help="after each adv/stokes solve, query the SNES "
                    "converged reason + iteration count and tag "
                    "WHICH solver diverged; enable snes_monitor "
                    "for the residual history of the failure")
p.add_argument("--stokes-cold-recover", type=int, default=0,
               help="if the warm-started Stokes solve DIVERGES, "
                    "re-solve on the SAME mesh from a COLD start "
                    "(zero_init_guess=True) up to N times instead "
                    "of propagating the corrupted V/P to the next "
                    "step. Breaks the same-mesh failure burst.")
p.add_argument("--no-vp-remap", action="store_true",
               help="DISABLE the V,P remap across the mesh move "
                    "(default: remap V,P onto the new node "
                    "positions exactly like T — the correct/"
                    "complete behaviour; only the old buggy "
                    "T-only path leaves V,P scrambled). Use to "
                    "A/B the remap's effect on Stokes warm-fails.")
p.add_argument("--stokes-snes-opt",
               choices=["default","basic","l2","tr","ksponly","direct",
                        "gamg-n1","gamg-thr","gamg-noagr","gamg-sor","gamg-full","gamg-noagrsor",
                        "gamg-n1-corr","gamg-thr-corr","gamg-noagr-corr",
                        "gamg-sor-corr","gamg-full-corr","gamg-noagrsor-corr"],
               default="default",
               help="override the Stokes SNES: basic/l2 "
                    "line-search, tr (trust region), or ksponly "
                    "(treat the linear Stokes as linear). Warm-"
                    "start robustness investigation.")
p.add_argument("--stokes-snes-atol-auto", action="store_true",
               help="capture the COLD first-solve initial residual "
                    "‖F(x0=0)‖ and set a FIXED snes_atol = "
                    "snes_rtol·‖F0‖ — the guess-INDEPENDENT "
                    "problem-scale convergence criterion UW3's "
                    "tolerance setter omits. Confirmation of the "
                    "snes_atol root-cause fix.")
args = p.parse_args()

RES = 24 if args.model == "ref24" else 16
ADAPT = args.model in ("a16", "a16p", "a16s", "a16x", "a16y",
                       "a16z", "a16c", "a16c2", "a16c15", "a16e",
                       "a16ed", "a16r15", "a16r15p", "a16r15e",
                       "a16c2v", "a16r15r", "a16r15v",
                       "a16r15b","a16r15l2","a16r15tr","a16r15ko","a16r15a","a16r15d","a16r15_6","a16r15-n1",
                       "a16r15-thr","a16r15-noagr","a16r15-sor","a16r15-full","a16r15-noagrsor",
                       "a16r15-n1-c","a16r15-thr-c","a16r15-noagr-c",
                       "a16r15-sor-c","a16r15-full-c","a16r15-noagrsor-c",
                       "a16r15-base-redo")
PRISTINE = args.model in ("a16p", "a16s", "a16x", "a16y", "a16z",
                          "a16c", "a16c2", "a16c15", "a16e",
                          "a16ed", "a16r15", "a16r15p", "a16r15e",
                          "a16c2v", "a16r15r", "a16r15v",
                       "a16r15b","a16r15l2","a16r15tr","a16r15ko","a16r15a","a16r15d","a16r15_6","a16r15-n1",
                       "a16r15-thr","a16r15-noagr","a16r15-sor","a16r15-full","a16r15-noagrsor",
                       "a16r15-n1-c","a16r15-thr-c","a16r15-noagr-c",
                       "a16r15-sor-c","a16r15-full-c","a16r15-noagrsor-c",
                       "a16r15-base-redo")

# Per-model metric strength. a16p = the conservative validated
# defaults (was tuned vs the now-removed cumulative over-
# compression). a16s = "more aggressive gradient following": the
# documented clean-but-strong Pareto corner (cap 2→4 needs a
# gentler relax + more n_outer per the validation arc; amp 8→16).
# Pristine re-mesh keeps each event a single uniform→graded map,
# so the static single-adaptation Pareto applies (no compounding).
# `beta` (mover anisotropy strength) and `aniso_cap` (eigen-clamp)
# are the EFFECTIVE knobs; `amp` cancels for the anisotropic mover
# (no-op, see scripts/_amp_check.py). Existing entries keep
# beta=200 (the mover default they actually ran with).
_MP = {
    "a16":  dict(amp=8.0,  aniso_cap=2.0, relax=0.2,  n_outer=8,
                 beta=200.0),
    "a16p": dict(amp=8.0,  aniso_cap=2.0, relax=0.2,  n_outer=8,
                 beta=200.0),
    "a16s": dict(amp=16.0, aniso_cap=4.0, relax=0.05, n_outer=25,
                 beta=200.0),
    # a16x: amp 16→24. WARNING — for the anisotropic mover `amp`
    # is a NO-OP: M is built from |∇ρ|/max|∇ρ|, and with
    # ρ=1+amp·t both scale with amp ⇒ it cancels exactly (verified
    # to machine ε, scripts/_amp_check.py). a16x ≡ a16s in metric;
    # kept only as a control / extra movie. To actually intensify
    # bunching change `aniso_cap` (sharper peak, ≥6 folds) and/or
    # `beta` in the mover (wider band); the percentile window in
    # metric_density_from_gradient reshapes *where* it refines.
    # (`amp` IS effective for the isotropic spring/MA methods.)
    "a16x": dict(amp=24.0, aniso_cap=4.0, relax=0.05, n_outer=25,
                 beta=200.0),
    # a16y = GENUINELY more aggressive: the real levers — sharper
    # peak (aniso_cap 4→5, the binding lever; ≥6 folds) + wider
    # refined band (beta 200→300), with extra damping (relax
    # 0.05→0.04, n_outer 25→30) so cap=5 stays non-folding under
    # pristine re-mesh. NOT amp (no-op for the mover).
    "a16y": dict(amp=16.0, aniso_cap=5.0, relax=0.04, n_outer=30,
                 beta=300.0),
    # a16z = BUDGET-CONCENTRATION via the percentile window (the
    # genuine "more bunching" lever — amp/cap/β are flat). Same
    # *stable* metric strength as a16s (cap=4/β=200/relax=0.05/
    # n_outer=25) so any change is cleanly the percentile, not
    # cap=5. lo_pct 50→85, hi_pct 97→99: only the steepest ~15% of
    # |∇T| qualifies; the fixed node budget concentrates on the
    # sharpest fronts (BL flanks / weak plumes go coarse).
    # Numerics stress-test (sharper metric in a thin band ⇒ bigger
    # per-event displacement ⇒ fold risk) — watched.
    "a16z": dict(amp=16.0, aniso_cap=4.0, relax=0.05, n_outer=25,
                 beta=200.0, lo_pct=85.0, hi_pct=99.0),
    # a16c = the STRUCTURAL FIX: a16s metric strength + percentile
    # (cap=4 / β=200 / relax=0.05 / n_outer=25 / pct 50-97) but
    # coarsen_cap=4 turns the refine-only metric (M ⪰ base·I — flat
    # zones pinned at h0, single steepest feature starves the rest)
    # into its true anisotropic-EQUIDISTRIBUTION form: low-ρ nodes
    # de-resolve to h0·√4 = 2·h0 and release budget to ALL fronts
    # (BL *and* plumes). Clean A/B vs a16s (identical params,
    # coarsen_cap=1). Per-node max anisotropy now ≈cap·cc=16 —
    # the no-fold check (valid=True) is the point of this run.
    "a16c": dict(amp=16.0, aniso_cap=4.0, coarsen_cap=4.0,
                 relax=0.05, n_outer=25, beta=200.0),
    # a16c cc=4 over-coarsens: de-resolution mechanism proven
    # (p95/p05~5.4) but minA/meanA crashed 0.27→0.04-0.14 (slivers,
    # irregular mesh). Back coarsen_cap DOWN to find the quality
    # knee — flats to h0·√cc (cc=2 → 1.41·h0, cc=1.5 → 1.22·h0)
    # instead of cc=4's 2·h0. aniso_cap stays 4; per-node anisotropy
    # = cap·cc (8 / 6 here, vs a16c's 16). Same a16s base otherwise.
    "a16c2":  dict(amp=16.0, aniso_cap=4.0, coarsen_cap=2.0,
                   relax=0.05, n_outer=25, beta=200.0),
    # a16c2v = re-verify the "robust" cc=2 (regime 2) config on the
    # CURRENT build. The cited 0-DIVERGED result predates the
    # regime-1/2/3 restructuring + snes_max_it=1; this confirms the
    # regime-2 path is still bit-faithful & solver-robust now.
    # Identical params to a16c2; separate tag preserves provenance.
    "a16c2v": dict(amp=16.0, aniso_cap=4.0, coarsen_cap=2.0,
                   relax=0.05, n_outer=25, beta=200.0),
    # a16r15r = R=1.5 equidist (cleanest static mesh ⇒ V/P-guess
    # is the only variable) for the cold-start RECOVERY test:
    # run with --stokes-cold-recover N to see if breaking the
    # same-mesh failure-propagation burst makes equidist robust.
    "a16r15r": dict(amp=16.0, resolution_ratio=1.5,
                    relax=0.05, n_outer=25, beta=200.0),
    # a16r15v = R=1.5 equidist, V,P-remap ON (correct baseline),
    # NO cold-recover. Isolates the V,P-remap effect: warm STOKES
    # DIVERGED vs a16r15's recorded 30 (no remap). Expect the
    # adapt-step (every 5th) bad-guess→KSP-stall fails to vanish;
    # any residue = the genuinely-hard non-adapt transient.
    "a16r15v": dict(amp=16.0, resolution_ratio=1.5,
                    relax=0.05, n_outer=25, beta=200.0),
    # a16r15a = the snes_atol root-cause CONFIRMATION: R=1.5
    # equidist, warm start, V,P-remap ON, default newtonls+bt,
    # NO cold-recover, but --stokes-snes-atol-auto (fixed
    # problem-scale snes_atol). Predict: warm STOKES DIVERGED → 0
    # (vs a16r15v's 24) because convergence uses the guess-
    # independent absolute path.
    # GAMG-anisotropy probe variants: clones of a16r15 (R=1.5).
    # Each pairs with --stokes-snes-opt gamg-<...> on resume
    # from a16r15 ckpt 50 (the reproducible failure case).
    "a16r15-n1":   dict(amp=16.0, resolution_ratio=1.5,
                       relax=0.05, n_outer=25, beta=200.0),
    "a16r15-thr":  dict(amp=16.0, resolution_ratio=1.5,
                       relax=0.05, n_outer=25, beta=200.0),
    "a16r15-noagr":dict(amp=16.0, resolution_ratio=1.5,
                       relax=0.05, n_outer=25, beta=200.0),
    "a16r15-sor":  dict(amp=16.0, resolution_ratio=1.5,
                       relax=0.05, n_outer=25, beta=200.0),
    "a16r15-full": dict(amp=16.0, resolution_ratio=1.5,
                       relax=0.05, n_outer=25, beta=200.0),
    "a16r15-noagrsor": dict(amp=16.0, resolution_ratio=1.5,
                       relax=0.05, n_outer=25, beta=200.0),
    # CORRECT-SCOPE GAMG variants (paired with -corr presets;
    # _SNES_OPT correctly nests at fieldsplit_velocity_pc_gamg_*).
    "a16r15-n1-c":   dict(amp=16.0, resolution_ratio=1.5,
                          relax=0.05, n_outer=25, beta=200.0),
    "a16r15-thr-c":  dict(amp=16.0, resolution_ratio=1.5,
                          relax=0.05, n_outer=25, beta=200.0),
    "a16r15-noagr-c":dict(amp=16.0, resolution_ratio=1.5,
                          relax=0.05, n_outer=25, beta=200.0),
    "a16r15-sor-c":  dict(amp=16.0, resolution_ratio=1.5,
                          relax=0.05, n_outer=25, beta=200.0),
    "a16r15-full-c": dict(amp=16.0, resolution_ratio=1.5,
                          relax=0.05, n_outer=25, beta=200.0),
    "a16r15-noagrsor-c": dict(amp=16.0, resolution_ratio=1.5,
                              relax=0.05, n_outer=25, beta=200.0),
    # Fresh baseline (default SNES opt) for comparison.
    "a16r15-base-redo": dict(amp=16.0, resolution_ratio=1.5,
                              relax=0.05, n_outer=25, beta=200.0),
    "a16r15_6": dict(amp=16.0, resolution_ratio=1.5,
                     relax=0.05, n_outer=25, beta=200.0),
    "a16r15a": dict(amp=16.0, resolution_ratio=1.5,
                    relax=0.05, n_outer=25, beta=200.0),
    # a16r15d = the EXACT-inner-solve probe: R=1.5 equidist, warm,
    # V,P-remap ON, default newtonls+bt, NO cold-recover, NO
    # atol-auto, --stokes-snes-opt direct (MUMPS LU). Tests if an
    # exact Newton step kills the transient line-search failures.
    # vs a16r15v baseline 24.
    "a16r15d": dict(amp=16.0, resolution_ratio=1.5,
                    relax=0.05, n_outer=25, beta=200.0),
    # Warm-start SNES investigation arms — all R=1.5 equidist,
    # V,P-remap ON, no cold-recover; only --stokes-snes-opt
    # differs (separate tags = separate output files). Baseline =
    # a16r15v's 24 warm STOKES DIVERGED (default newtonls+bt).
    "a16r15b":  dict(amp=16.0, resolution_ratio=1.5,
                     relax=0.05, n_outer=25, beta=200.0),
    "a16r15l2": dict(amp=16.0, resolution_ratio=1.5,
                     relax=0.05, n_outer=25, beta=200.0),
    "a16r15tr": dict(amp=16.0, resolution_ratio=1.5,
                     relax=0.05, n_outer=25, beta=200.0),
    "a16r15ko": dict(amp=16.0, resolution_ratio=1.5,
                     relax=0.05, n_outer=25, beta=200.0),
    "a16c15": dict(amp=16.0, aniso_cap=4.0, coarsen_cap=1.5,
                   relax=0.05, n_outer=25, beta=200.0),
    # a16e = the FINAL single-knob equidistribution API:
    # resolution_ratio=2 only (no coarsen_cap/aniso_cap → regime 1,
    # s=base·ρ/G, complementary coarsening automatic by budget
    # conservation). Same a16s base (relax/n_outer/beta). Expected
    # ≈ a16c2 quality/physics (R=2 is the validated operating
    # point) but with the split parameter-free. Settled re-validation.
    "a16e":   dict(amp=16.0, resolution_ratio=2.0,
                   relax=0.05, n_outer=25, beta=200.0),
    # a16ed = a16e + EMA temporal damping of the equidist
    # normaliser G (geom_mean_smoothing=0.25, ln-space EMA across
    # adaptation events). Should kill the startup-transient
    # over-reaction + the steady-state grading-contrast pulse while
    # leaving settled quality/Nu = a16e. Clean A/B vs the saved
    # (undamped, geom_mean_smoothing=1.0) a16e checkpoints.
    "a16ed":  dict(amp=16.0, resolution_ratio=2.0,
                   geom_mean_smoothing=0.25,
                   relax=0.05, n_outer=25, beta=200.0),
    # a16r15 = the CORRECTED production config. Frozen-snapshot
    # sweep (scripts/_snap_sweep.py, step-20 overshoot field)
    # showed R is the binding cell-quality lever, not bump-keying
    # or EMA: R=2 sits on the poor-cell knee (Stokes-stressing
    # clusters), R=1.5 is clean (beats cc2's quality tail at the
    # worst instant, BIG&THIN=0) — with the DEFAULT construction
    # (iso-keyed bump, EMA off). So this is the minimal clean
    # single-knob config: resolution_ratio=1.5 only. Confirmation
    # target: zero DIVERGED (= a16c2), cc2-class quality through
    # the full lifecycle, settled Nu≈3.85 (≈ a16c2/a16e).
    "a16r15": dict(amp=16.0, resolution_ratio=1.5,
                   relax=0.05, n_outer=25, beta=200.0),
    # a16r15p = a16r15 config, separate output tag — for the
    # restart/instrumentation probe (reads a16r15 ckpt via
    # --src-tag, writes its own files, no clobber).
    "a16r15p": dict(amp=16.0, resolution_ratio=1.5,
                    relax=0.05, n_outer=25, beta=200.0),
    # a16r15e = task 34: the only untested combination — clean
    # clamp (R=1.5, no static poor cells per the snapshot) PLUS
    # EMA temporal damping of the equidist normaliser
    # (geom_mean_smoothing=0.25). The restart probe proved the
    # Stokes -6 failures are reproducible & dynamics-coupled (not
    # path/static), so the per-event normaliser jump during the
    # violent undershoot is the suspect — exactly what the EMA
    # smooths. Target: ZERO STOKES DIVERGED (= cc2 bar). a16ed was
    # R=2 (confounded by worse static quality); this is the clean
    # test. Run with --snes-debug to tag any residual failures.
    "a16r15e": dict(amp=16.0, resolution_ratio=1.5,
                    geom_mean_smoothing=0.25,
                    relax=0.05, n_outer=25, beta=200.0),
}
MP = _MP.get(args.model, _MP["a16p"])
r_inner, r_o = 0.5, 1.0
DIR = "/tmp/metric_mesh/sat"
os.makedirs(DIR, exist_ok=True)
TAG = args.model                       # outputs written here
SRC = args.src_tag or TAG              # resume state read here
HIST = f"{DIR}/sat_{TAG}_hist.npz"
SRC_HIST = f"{DIR}/sat_{SRC}_hist.npz"


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


def adapt_local_fe_interp(mesh, T, v, P, stokes):
    """Anisotropic adapt + local-FE remap (validated): old P3
    field evaluated at the new node positions; layout-invariant ⇒
    trivial restore, no migration; fixed domain ⇒ in-domain.
    V,P are remapped the same way as T (correct/complete) unless
    --no-vp-remap (the old T-only path that scrambled V,P)."""
    rho = metric_density_from_gradient(
        mesh, T, amp=MP["amp"], name="sat",
        lo_percentile=MP.get("lo_pct", 50.0),
        hi_percentile=MP.get("hi_pct", 97.0))
    old_X = np.asarray(mesh.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    old_v = np.asarray(v.data).copy()
    old_P = np.asarray(P.data).copy()
    smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(aniso_cap=MP.get("aniso_cap", 2.0),
                           coarsen_cap=MP.get("coarsen_cap", 1.0),
                           resolution_ratio=MP.get(
                               "resolution_ratio", 1.0),
                           geom_mean_smoothing=MP.get(
                               "geom_mean_smoothing", 1.0),
                           relax=MP["relax"],
                           n_outer=MP["n_outer"],
                           beta=MP["beta"]))
    new_X = np.asarray(mesh.X.coords).copy()
    new_Tx = np.asarray(T.coords).copy()
    new_Vx = np.asarray(v.coords).copy()
    new_Px = np.asarray(P.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T
    vals = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    if not args.no_vp_remap:
        # old_X geometry with old_v/old_P restored = the true
        # previous (v,P) field; sample it at the NEW dof positions
        # so the next Stokes warm-start is the previous solution
        # at the new nodes — not stale values on moved nodes.
        v.data[...] = old_v
        P.data[...] = old_P
        valsV = np.asarray(uw.function.evaluate(v.sym, new_Vx))
        valsP = np.asarray(uw.function.evaluate(
            P.sym[0], new_Px)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = vals
    if not args.no_vp_remap:
        v.data[...] = valsV.reshape(v.data.shape)
        P.data[:, 0] = valsP
    # NO stokes here — the loop's single stokes.solve (now placed
    # AFTER adaptation) recomputes v on the adapted mesh; the old
    # in-adapt re-solve was redundant (v is stale-by-construction
    # in the segregated scheme; nothing reads it before that solve).


def adapt_pristine(mesh, T, v, P, stokes, X0, X0_Tx):
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
    # v,P true previous solution lives in the X_prev geometry
    # (it was solved there); captured for the post-move remap.
    v_prev = np.asarray(v.data).copy()
    P_prev = np.asarray(P.data).copy()
    # (1) physical T (mesh@X_prev, T_prev) → pristine X0 T-DOFs
    vals0 = np.asarray(uw.function.evaluate(
        T.sym[0], X0_Tx)).reshape(-1)
    mesh._deform_mesh(X0)
    T.data[:, 0] = vals0
    # (2) metric from the physical T now on the pristine mesh
    rho = metric_density_from_gradient(
        mesh, T, amp=MP["amp"], name="sat",
        lo_percentile=MP.get("lo_pct", 50.0),
        hi_percentile=MP.get("hi_pct", 97.0))
    # (3) mover baseline is pristine X0 (fresh, non-compounding)
    X0c = np.asarray(mesh.X.coords).copy()
    T0 = np.asarray(T.data).copy()
    smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(aniso_cap=MP.get("aniso_cap", 2.0),
                           coarsen_cap=MP.get("coarsen_cap", 1.0),
                           resolution_ratio=MP.get(
                               "resolution_ratio", 1.0),
                           geom_mean_smoothing=MP.get(
                               "geom_mean_smoothing", 1.0),
                           relax=MP["relax"],
                           n_outer=MP["n_outer"],
                           beta=MP["beta"]))
    new_X = np.asarray(mesh.X.coords).copy()
    new_Tx = np.asarray(T.coords).copy()
    new_Vx = np.asarray(v.coords).copy()
    new_Px = np.asarray(P.coords).copy()
    # (4) FE-remap the pristine-mesh T onto the new graded mesh
    mesh._deform_mesh(X0c)
    T.data[...] = T0
    valsN = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = valsN
    # (4b) FE-remap V,P (the correct/complete behaviour): the true
    # previous solution lives in the X_prev geometry — restore it
    # there, sample at the NEW V/P dof positions, write onto the
    # graded mesh. Without this the next Stokes warm-start is the
    # old nodal values on moved nodes (a garbage guess → KSP stall
    # → SNES DIVERGED_LINE_SEARCH). --no-vp-remap keeps the old
    # T-only path for the controlled A/B.
    if not args.no_vp_remap:
        mesh._deform_mesh(X_prev)
        v.data[...] = v_prev
        P.data[...] = P_prev
        valsV = np.asarray(uw.function.evaluate(v.sym, new_Vx))
        valsP = np.asarray(uw.function.evaluate(
            P.sym[0], new_Px)).reshape(-1)
        mesh._deform_mesh(new_X)
        v.data[...] = valsV.reshape(v.data.shape)
        P.data[:, 0] = valsP
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

# Warm-start SNES investigation: override the Stokes nonlinear
# solver. This Stokes is effectively LINEAR (constant viscosity,
# T-fixed buoyancy) yet the default newtonls+bt line-search fails
# warm-started from the *true previous solution* on the equidist
# mesh through the violent transient (a16r15v: 24 fails, all
# non-adapt steps) while converging cold — a line-search
# misconfiguration signature, not a guess problem.
_SNES_OPT = {
    "default": {},                                  # newtonls+bt
    "basic":   {"snes_linesearch_type": "basic"},   # full step
    "l2":      {"snes_linesearch_type": "l2"},
    "tr":      {"snes_type": "newtontr"},            # trust region
    "ksponly": {"snes_type": "ksponly"},             # it's linear
    # EXACT inner Newton solve: MUMPS LU on the (small, res-16 2D)
    # Stokes Jacobian, replacing the iterative fieldsplit inner
    # solve. Tests whether the transient DIVERGED_LINE_SEARCH is
    # the bt line search rejecting an INEXACT Newton step — an
    # exact step should be accepted at λ=1. Keeps default
    # newtonls+bt (we test the *step*, not the line search).
    "direct":  {"ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1},            # null-pivot detect
    # Anisotropy-tuned GAMG (source-verified PETSc options).
    # AMG aggregation degrades on stretched/anisotropic cells
    # — these knobs counteract that without abandoning GAMG.
    "gamg-n1":    {"pc_gamg_agg_nsmooths": 1},  # 2→1 (PETSc default)
    "gamg-thr":   {"pc_gamg_threshold": 0.02,
                   "pc_gamg_threshold_scale": 0.5},
    "gamg-noagr": {"pc_gamg_aggressive_coarsening": 0},
    "gamg-sor":   {"mg_levels_ksp_type": "richardson",
                   "mg_levels_pc_type": "sor",
                   "mg_levels_ksp_max_it": 2},
    "gamg-full":  {"pc_gamg_agg_nsmooths": 1,
                   "pc_gamg_threshold": 0.02,
                   "pc_gamg_threshold_scale": 0.5,
                   "pc_gamg_aggressive_coarsening": 0,
                   "pc_gamg_mis_k_minimum_degree_ordering": True,
                   "mg_levels_ksp_type": "richardson",
                   "mg_levels_pc_type": "sor",
                   "mg_levels_ksp_max_it": 2},
    # Discriminator: aggregation-fix (noagr) PLUS smoother
    # swap (sor). On the 40-step probe both alone gave 0;
    # combined ought to also = 0. Use Krylov-iter / wall-time
    # as the finer discriminator if both produce 0 fails.
    "gamg-noagrsor": {"pc_gamg_aggressive_coarsening": 0,
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_pc_type": "sor",
                       "mg_levels_ksp_max_it": 2},
    # CORRECT-SCOPE variants: UW3 Stokes nests GAMG inside the
    # velocity Schur sub-block at prefix
    # `fieldsplit_velocity_pc_gamg_*`. The original (above)
    # entries set the options at the global (Solver_NN_) scope,
    # which is silently ignored by the velocity subsolver —
    # verified bit-identical KSP residuals to default (see
    # scripts/_sl_preset_verify.py). These -corr variants use
    # the correct prefix and DO change the KSP convergence
    # path. Run alongside the original (wrong-scope) entries
    # to A/B them.
    "gamg-n1-corr": {
        "fieldsplit_velocity_pc_gamg_agg_nsmooths": 1},
    "gamg-thr-corr": {
        "fieldsplit_velocity_pc_gamg_threshold": 0.02,
        "fieldsplit_velocity_pc_gamg_threshold_scale": 0.5},
    "gamg-noagr-corr": {
        "fieldsplit_velocity_pc_gamg_aggressive_coarsening": 0},
    "gamg-sor-corr": {
        "fieldsplit_velocity_mg_levels_ksp_type": "richardson",
        "fieldsplit_velocity_mg_levels_pc_type": "sor",
        "fieldsplit_velocity_mg_levels_ksp_max_it": 2},
    "gamg-full-corr": {
        "fieldsplit_velocity_pc_gamg_agg_nsmooths": 1,
        "fieldsplit_velocity_pc_gamg_threshold": 0.02,
        "fieldsplit_velocity_pc_gamg_threshold_scale": 0.5,
        "fieldsplit_velocity_pc_gamg_aggressive_coarsening": 0,
        "fieldsplit_velocity_pc_gamg_mis_k_minimum_degree_ordering": True,
        "fieldsplit_velocity_mg_levels_ksp_type": "richardson",
        "fieldsplit_velocity_mg_levels_pc_type": "sor",
        "fieldsplit_velocity_mg_levels_ksp_max_it": 2},
    "gamg-noagrsor-corr": {
        "fieldsplit_velocity_pc_gamg_aggressive_coarsening": 0,
        "fieldsplit_velocity_mg_levels_ksp_type": "richardson",
        "fieldsplit_velocity_mg_levels_pc_type": "sor",
        "fieldsplit_velocity_mg_levels_ksp_max_it": 2},
}
for _k, _vopt in _SNES_OPT.get(args.stokes_snes_opt, {}).items():
    stokes.petsc_options[_k] = _vopt
if args.stokes_snes_opt != "default":
    print(f"[stokes-snes-opt={args.stokes_snes_opt}] "
          f"{_SNES_OPT[args.stokes_snes_opt]}", flush=True)

# pristine reference captured once (mesh + T-DOF coords undeformed)
X0 = np.asarray(mesh.X.coords).copy()
X0_Tx = np.asarray(T.coords).copy()

# NB: do NOT enable the global PETSc `snes_converged_reason`
# viewer for debugging — it leaks into the mover's ksponly linear
# sub-solves (created later inside smooth_mesh_interior) and floods
# the log with misleading "DIVERGED_MAX_IT iterations 0" lines (a
# linear solve has no Newton iterations; the SNES wrapper
# mis-labels it). The programmatic _snes_chk below tags WHICH UW
# physics solver diverged + reason + iter-count with zero leakage.


def _snes_chk(solver, name, step):
    """Post-solve SNES convergence tag — reason<0 ⇒ diverged
    (DIVERGED_LINE_SEARCH = -6). Identifies which physics solve
    failed (the pyx retry message is solver-anonymous)."""
    if not args.snes_debug:
        return None
    try:
        sn = solver.snes
        reason = int(sn.getConvergedReason())
        its = int(sn.getIterationNumber())
        if reason < 0:
            print(f"  !! [step {step}] {name} SNES DIVERGED "
                  f"reason={reason} its={its}", flush=True)
        return reason
    except Exception as e:
        print(f"  !! [step {step}] {name} snes-chk error: {e}",
              flush=True)
        return None


if args.resume:
    import glob
    import re
    _fs = glob.glob(f"{DIR}/sat_{SRC}.mesh.T.*.h5")
    _all = sorted(int(re.search(r"\.mesh\.T\.(\d+)\.h5$",
                  f).group(1)) for f in _fs)
    _idx = (args.resume_from if args.resume_from > 0
            else max(_all))
    T.read_timestep(f"sat_{SRC}", "T", _idx, outputPath=DIR)
    v.read_timestep(f"sat_{SRC}", "V", _idx, outputPath=DIR)
    _z = np.load(SRC_HIST)
    hist = [[int(_z["step"][i]), float(_z["t"][i]),
             float(_z["dt"][i]), float(_z["Nu"][i]),
             float(_z["vrms"][i])] for i in range(len(_z["step"]))
            if int(_z["step"][i]) <= _idx]
    STEP0 = _idx
    t_sim = (hist[-1][1] if hist else 0.0)
    stokes.solve(zero_init_guess=False)   # sync v with loaded T
    print(f"=== sat {TAG} RESUME from {SRC} ckpt {STEP0} "
          f"t={t_sim:.4f} (+{args.max_steps} steps, "
          f"snes_debug={args.snes_debug}) ===", flush=True)
else:
    stokes.solve(zero_init_guess=True)        # solve 1: creates SNES
    if args.stokes_snes_atol_auto:
        # First-pass scale: the SNES object exists only after a
        # solve, so enable history NOW, then re-solve cold (same T,
        # x0=0) to record ‖F(x0=0)‖ ≈ the problem/RHS residual
        # scale. Set a FIXED snes_atol = snes_rtol·‖F0‖ so SNES
        # converges on the guess-INDEPENDENT absolute path
        # (SNESConvergedDefault snesut.c:752, evaluated even at
        # it==0 since forceiteration is off) — the criterion UW3's
        # tolerance setter never sets (snes_atol left at PETSc
        # ~1e-50). Same accuracy a working cold solve achieves,
        # just measured against the problem, not the guess.
        _F0 = 0.0
        try:
            stokes.snes.setConvergenceHistory(reset=True)
            stokes.solve(zero_init_guess=True)   # solve 2: records
            _rh, _ = stokes.snes.getConvergenceHistory()
            _F0 = float(_rh[0]) if _rh is not None and len(_rh) \
                else 0.0
        except Exception as _e:
            print(f"[snes-atol-auto] history read failed: {_e!r}",
                  flush=True)
        _rtol = 1.0e-5                       # build(): stokes.tolerance
        if _F0 > 0.0:
            _atol = _rtol * _F0
            stokes.petsc_options["snes_atol"] = _atol
            print(f"[snes-atol-auto] cold ‖F0‖={_F0:.4e} ⇒ "
                  f"snes_atol={_atol:.4e} (rtol={_rtol:.0e}); "
                  f"absolute, guess-independent", flush=True)
        else:
            print("[snes-atol-auto] WARN: ‖F0‖ unavailable; "
                  "atol unchanged (run is NOT a valid "
                  "confirmation)", flush=True)
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
    _snes_chk(adv, "ADV ", STEP)
    # Loop-reorder fix: adapt BETWEEN adv.solve and the single
    # stokes.solve. The remesh+remap happens on the just-advected
    # T; the one stokes.solve below then recomputes v on the
    # adapted mesh (no redundant old-mesh solve, no in-adapt
    # re-solve). Physically equivalent end-of-step state, ~2 fewer
    # Stokes solves per adaptation step.
    if ADAPT and STEP % args.adapt_every == 0:
        if PRISTINE:
            adapt_pristine(mesh, T, v, P, stokes, X0, X0_Tx)
        else:
            adapt_local_fe_interp(mesh, T, v, P, stokes)
    stokes.solve(zero_init_guess=False)
    _r = _snes_chk(stokes, "STOKES", STEP)
    # Same-mesh recovery: a diverged warm solve leaves V/P
    # corrupted ("solution vector may not have been updated");
    # propagating it to the next step makes the failure
    # self-sustaining (the same-mesh burst). Re-solve COLD on the
    # SAME mesh+T (Stokes is linear here ⇒ a converged cold solve
    # is the correct v,P) before moving on.
    if args.stokes_cold_recover and _r is not None and _r < 0:
        for _att in range(1, args.stokes_cold_recover + 1):
            print(f"  ~~ [step {STEP}] STOKES cold-recover "
                  f"attempt {_att} (zero_init_guess=True)",
                  flush=True)
            stokes.solve(zero_init_guess=True)
            _r = _snes_chk(stokes, f"STOKES.r{_att}", STEP)
            if _r is None or _r >= 0:
                break
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
