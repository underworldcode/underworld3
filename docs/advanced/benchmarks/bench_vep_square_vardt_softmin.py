"""Variable-dt VEP square-wave benchmark — *softmin* yield mode.

Counterpart to bench_vep_square_vardt (Min) and
bench_vep_square_vardt_smooth (smooth blend).  Same dt schedule and
forcing; the only change is ``yield_mode = "softmin"`` with the
default softness δ = 0.1.

Softmin replaces ``Min(η_ve, η_pl)`` with the smooth approximation

    η_eff = η_ve / g(f),    g(f) = 1 + (f − 1 + √((f−1)² + δ²))/2 − offset

which converges to true Min as δ → 0.  Default δ = 0.1 keeps continuous
derivatives at the yield kink while staying close to Min in magnitude.
"""

import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, build_stokes, probe_centre,
    vep_square_wave, save_run, error_metrics,
)


V0 = 0.5
TAU_Y = 0.5
HALF_PERIOD = 2.0
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * HALF_PERIOD

DT_PLATEAU = 0.10
DT_FINE_RATIO = 0.10
DT_FINE = DT_PLATEAU * DT_FINE_RATIO
WINDOW = 0.1 * HALF_PERIOD

LABEL = "vep_square_vardt_softmin"


def schedule_dt(t_cur):
    flip_times = [HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2 - 1)]
    for f in flip_times:
        if abs(t_cur - f) <= WINDOW:
            return DT_FINE
    return DT_PLATEAU


def _run_one(bdf_order):
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = bdf_order
    mesh, stokes, V_top, params = build_stokes(
        f"{LABEL}_o{bdf_order}", params,
        yield_stress=TAU_Y, yield_mode="softmin",
    )
    times, dts, sigmas, gammas, reasons = [], [], [], [], []
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
    return (np.array(times), np.array(dts), np.array(sigmas),
            np.array(gammas), np.array(reasons), time.time() - t0, params)


def main():
    times1, dts1, sig1, gam1, rea1, wall1, params = _run_one(1)
    times2, dts2, sig2, gam2, rea2, wall2, _      = _run_one(2)
    assert np.allclose(times1, times2)
    gamma_dot_0 = 2.0 * V0 / params["H"]
    sigma_ana_min = vep_square_wave(times1, params["eta"], params["mu"],
                                    gamma_dot_0, TAU_Y, HALF_PERIOD)
    err1 = error_metrics(sig1, sigma_ana_min)
    err2 = error_metrics(sig2, sigma_ana_min)
    peak1 = float(np.abs(sig1).max()); peak2 = float(np.abs(sig2).max())
    print(f"[{LABEL}]  steps={len(times1)}  τ_y={TAU_Y}  η·γ̇₀={params['eta']*gamma_dot_0:.4f}")
    print(f"  schedule: plateau dt={DT_PLATEAU}, fine dt={DT_FINE} (×{DT_FINE_RATIO}), window=±{WINDOW}")
    print(f"  yield_mode = softmin (default δ = 0.1)")
    print(f"  BDF-1 wall={wall1:.1f}s peak|σ|={peak1:.4f}  ({100*peak1/TAU_Y:.1f}% of τ_y)  max|err vs Min-clip|={err1['max_abs']:.4e}  rms={err1['rms']:.4e}")
    print(f"  BDF-2 wall={wall2:.1f}s peak|σ|={peak2:.4f}  ({100*peak2/TAU_Y:.1f}% of τ_y)  max|err vs Min-clip|={err2['max_abs']:.4e}  rms={err2['rms']:.4e}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, half_period=HALF_PERIOD, n_periods=N_PERIODS,
            tau_y=TAU_Y, gamma_dot_0=gamma_dot_0, t_end=T_END,
            dt_plateau=DT_PLATEAU, dt_fine=DT_FINE, dt_fine_ratio=DT_FINE_RATIO,
            window=WINDOW, yield_mode="softmin", yield_softness=0.1,
            peak_bdf1=peak1, peak_bdf2=peak2,
            err_max_bdf1=err1["max_abs"], err_rms_bdf1=err1["rms"],
            err_max_bdf2=err2["max_abs"], err_rms_bdf2=err2["rms"],
            wall_bdf1=wall1, wall_bdf2=wall2,
        ),
        times=times1, dts=dts1, gamma_dot=gam1, sigma_ana=sigma_ana_min,
        sigma_bdf1=sig1, sigma_bdf2=sig2,
        snes_reasons_bdf1=rea1, snes_reasons_bdf2=rea2,
    )


if __name__ == "__main__":
    main()
