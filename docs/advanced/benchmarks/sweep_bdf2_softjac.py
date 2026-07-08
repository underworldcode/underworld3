"""Sweep — BDF-2 inexact-Newton line-search behaviour with softmin Jacobian.

Background: with Min residual + softmin Jacobian (δ=0.1), BDF-2 sees
195/413 line-search rejections and gives a more accurate answer than
pure Min/Min. Question: can we keep the accuracy and lose the noise?

Axes:
  A. Softness δ ∈ {0.05, 0.1, 0.2, 0.5}  (Jacobian-only smoothing)
  B. snes_linesearch_type ∈ {bt, cp, basic}  (basic = no LS, accept Newton)
  C. snes_atol ∈ {None, 1e-5}                (early termination if residual tiny)

We DON'T sweep the full cross product — too expensive. Run two
families in series, each with the matched axis fixed at the baseline
(δ=0.1, bt, atol=None):
  Family A: vary δ
  Family B: vary linesearch
  Family C: try atol=1e-5 with the baseline

Results land in output/benchmarks/sweep_bdf2_softjac/ — one .npz per
variant — and a summary table is printed at the end.
"""

import os
import time
import numpy as np
import sympy

from _bench_helpers import (
    DEFAULT_PARAMS, build_stokes, probe_centre,
    vep_square_wave, error_metrics,
)


V0 = 0.5
TAU_Y = 0.5
HALF_PERIOD = 2.0
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * HALF_PERIOD

DT_PLATEAU = 0.10
DT_FINE = 0.01
WINDOW = 0.1 * HALF_PERIOD

OUT_DIR = "../../../output/benchmarks/sweep_bdf2_softjac"


def schedule_dt(t_cur):
    flip_times = [HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2 - 1)]
    for f in flip_times:
        if abs(t_cur - f) <= WINDOW:
            return DT_FINE
    return DT_PLATEAU


def _capture_softmin_F1(stokes, softness):
    cm = stokes.constitutive_model
    saved_mode, saved_softness = cm._yield_mode, cm._yield_softness
    try:
        cm._yield_mode = "softmin"
        cm._yield_softness = softness
        soft_stress = cm.flux
        F1_softmin = soft_stress + stokes.penalty * stokes.div_u * sympy.eye(stokes.mesh.dim)
    finally:
        cm._yield_mode, cm._yield_softness = saved_mode, saved_softness
    return F1_softmin


def run_variant(label, softness, linesearch="bt", atol=None):
    """Run one BDF-2 variant. Returns dict with metrics + arrays."""
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = 2
    mesh, stokes, V_top, params = build_stokes(
        f"sweep_{label}", params, yield_stress=TAU_Y, yield_mode="min",
    )

    F1_jac = _capture_softmin_F1(stokes, softness)
    stokes.set_jacobian_F1_source(F1_jac)
    stokes.petsc_options["snes_linesearch_type"] = linesearch
    if atol is not None:
        stokes.petsc_options["snes_atol"] = atol

    times, dts, sigmas, gammas, reasons, iters = [], [], [], [], [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = schedule_dt(t_cur)
        flip_next = next((HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2)
                          if HALF_PERIOD * (k + 1) > t_cur + 1e-9), T_END)
        dt = min(dt, flip_next - t_cur, T_END - t_cur)
        t_end_step = t_cur + dt
        n_half = int(t_end_step / HALF_PERIOD - 1e-9)
        sign = 1.0 if n_half % 2 == 0 else -1.0
        v_now = sign * V0
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=0)
        sigmas.append(probe_centre(stokes))
        t_cur = t_end_step
        times.append(t_cur); dts.append(dt); gammas.append(2.0 * v_now / params["H"])
        reasons.append(int(stokes.snes.getConvergedReason()))
        iters.append(int(stokes.snes.getIterationNumber()))
    wall = time.time() - t0

    times = np.array(times); sigmas = np.array(sigmas)
    reasons = np.array(reasons); iters = np.array(iters)
    gamma_dot_0 = 2.0 * V0 / params["H"]
    sigma_ana = vep_square_wave(times, params["eta"], params["mu"],
                                gamma_dot_0, TAU_Y, HALF_PERIOD)
    err = error_metrics(sigmas, sigma_ana)

    out = dict(
        label=label, softness=softness, linesearch=linesearch, atol=atol,
        wall=wall, peak=float(np.abs(sigmas).max()),
        diverged=int((reasons < 0).sum()), mean_its=float(iters.mean()),
        max_err=float(err["max_abs"]), rms=float(err["rms"]),
        times=times, sigmas=sigmas, sigma_ana=sigma_ana,
        reasons=reasons, iters=iters,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"{label}.npz"), **{
        "times": times, "sigmas": sigmas, "sigma_ana": sigma_ana,
        "reasons": reasons, "iters": iters,
        "softness": softness, "linesearch": linesearch,
        "atol": (atol if atol is not None else -1.0),
        "wall": wall,
    })
    print(f"  [{label:<28}] δ={softness:<5} ls={linesearch:<5} atol={atol!r:<6} "
          f"wall={wall:6.1f}s  div={out['diverged']:3d}  its={out['mean_its']:5.2f}  "
          f"peak={out['peak']:.4f}  max|err|={out['max_err']:.3e}  rms={out['rms']:.3e}",
          flush=True)
    return out


def main():
    print("\n=== sweep_bdf2_softjac (Min residual, softmin Jacobian) ===\n", flush=True)
    print("Family A: vary softness δ (bt linesearch, no atol)", flush=True)
    family_A = [run_variant(f"deltaA_{d}", softness=d) for d in (0.05, 0.10, 0.20, 0.50)]

    print("\nFamily B: vary linesearch (δ=0.10, no atol)", flush=True)
    family_B = []
    for ls in ("cp", "basic", "l2"):
        family_B.append(run_variant(f"lsB_{ls}", softness=0.10, linesearch=ls))

    print("\nFamily C: snes_atol=1e-5 with baseline (δ=0.10, bt)", flush=True)
    family_C = [run_variant("atolC_1e-5", softness=0.10, linesearch="bt", atol=1e-5)]

    all_runs = family_A + family_B + family_C
    print("\n\n=== summary ===", flush=True)
    print(f"{'label':<28} {'δ':>5} {'ls':>6} {'atol':>8} {'wall':>7} {'div':>4} {'its':>5} {'peak|σ|':>7} {'max|err|':>10} {'rms':>10}",
          flush=True)
    for r in all_runs:
        atol_str = f"{r['atol']:.0e}" if r['atol'] is not None else "None"
        print(f"{r['label']:<28} {r['softness']:>5} {r['linesearch']:>6} {atol_str:>8} "
              f"{r['wall']:>7.1f} {r['diverged']:>4d} {r['mean_its']:>5.2f} "
              f"{r['peak']:>7.4f} {r['max_err']:>10.3e} {r['rms']:>10.3e}",
              flush=True)


if __name__ == "__main__":
    main()
