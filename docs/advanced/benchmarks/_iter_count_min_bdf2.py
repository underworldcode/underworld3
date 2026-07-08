"""Quick rerun: pure Min/Min BDF-2 with iter counts captured.

Confirms (or refutes) the hypothesis that the cleaner SNES record of
the Min/Min BDF-2 run is masking very few actual Newton iterations,
which would explain its larger answer error vs the softJac variants.
"""

import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, build_stokes, probe_centre,
    vep_square_wave, error_metrics, OUTPUT_DIR,
)


V0 = 0.5
TAU_Y = 0.5
HALF_PERIOD = 2.0
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * HALF_PERIOD
DT_PLATEAU = 0.10
DT_FINE = 0.01
WINDOW = 0.1 * HALF_PERIOD


def schedule_dt(t_cur):
    flip_times = [HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2 - 1)]
    for f in flip_times:
        if abs(t_cur - f) <= WINDOW:
            return DT_FINE
    return DT_PLATEAU


def main():
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = 2
    mesh, stokes, V_top, params = build_stokes(
        "iter_min_o2", params, yield_stress=TAU_Y, yield_mode="min",
    )
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
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
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

    from collections import Counter
    print(f"\nMin/Min BDF-2 (var-dt VEP square): wall={wall:.1f}s  steps={len(times)}",
          flush=True)
    print(f"  reasons:    {dict(sorted(Counter(reasons.tolist()).items()))}", flush=True)
    print(f"  iter dist:  {dict(sorted(Counter(iters.tolist()).items()))}", flush=True)
    print(f"  iter mean:  {iters.mean():.2f}  median: {np.median(iters):.0f}  "
          f"max: {int(iters.max())}", flush=True)
    print(f"  fraction with iters==0: {(iters == 0).sum()}/{len(iters)} = {(iters==0).mean():.1%}",
          flush=True)
    print(f"  fraction with iters==1: {(iters == 1).sum()}/{len(iters)} = {(iters==1).mean():.1%}",
          flush=True)
    print(f"  peak|σ|={float(np.abs(sigmas).max()):.4f}  "
          f"max|err|={err['max_abs']:.3e}  rms={err['rms']:.3e}", flush=True)

    import os
    np.savez(os.path.join(OUTPUT_DIR, "iter_count_min_bdf2.npz"),
             times=times, dts=dts, sigmas=sigmas, sigma_ana=sigma_ana,
             reasons=reasons, iters=iters, wall=wall)


if __name__ == "__main__":
    main()
