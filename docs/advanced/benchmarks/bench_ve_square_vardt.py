"""Variable-dt VE square-wave benchmark.

Same physics as :mod:`bench_ve_square` but with a non-uniform timestep
schedule: dt is reduced by a factor of 10 in a small window around each
BC flip and held at the larger value on plateaux.  This tests the
projection-snapshot machinery on the exact path that previously
exhibited the implicit-projection drift (see
``tests/test_1052_VEP_stability_regression.py::test_vep_yield_lock_variable_dt``)
and confirms the same robustness on the pure-VE side.

Schedule (with ``T_{1/2} = 2 t_r`` and a window of ``±0.1 T_{1/2}`` around
each flip):
    plateau dt = ``DT_PLATEAU``
    flip-window dt = ``DT_PLATEAU / 10``

Run::

    pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_square_vardt.py
"""

import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, build_stokes, probe_centre,
    maxwell_square_wave, save_run, error_metrics,
)


V0 = 0.5
HALF_PERIOD = 2.0
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * HALF_PERIOD

DT_PLATEAU = 0.10           # plateau dt (same as bench_ve_square)
DT_FINE_RATIO = 0.10        # flip-window dt is 0.10 × plateau
DT_FINE = DT_PLATEAU * DT_FINE_RATIO
WINDOW = 0.1 * HALF_PERIOD  # ±0.20 t_r around each flip

LABEL = "ve_square_vardt"


def schedule_dt(t_cur):
    """Fine dt within ±WINDOW of any flip; plateau dt elsewhere."""
    flip_times = [HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2 - 1)]
    for f in flip_times:
        if abs(t_cur - f) <= WINDOW:
            return DT_FINE
    return DT_PLATEAU


def _run_one(bdf_order):
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = bdf_order
    mesh, stokes, V_top, params = build_stokes(f"{LABEL}_o{bdf_order}", params)

    times, dts, sigmas, gammas, reasons = [], [], [], [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = schedule_dt(t_cur)
        # Don't step past a flip boundary or past T_END
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
    sigma_ana = maxwell_square_wave(times1, params["eta"], params["mu"],
                                    gamma_dot_0, HALF_PERIOD)
    err1 = error_metrics(sig1, sigma_ana)
    err2 = error_metrics(sig2, sigma_ana)
    print(f"[{LABEL}]  steps={len(times1)}  σ_ss=η·γ̇₀={params['eta']*gamma_dot_0:.4f}")
    print(f"  schedule: plateau dt={DT_PLATEAU}, fine dt={DT_FINE} (×{DT_FINE_RATIO}), window=±{WINDOW}")
    print(f"  BDF-1 wall={wall1:.1f}s  max|err|={err1['max_abs']:.4e}  rms={err1['rms']:.4e}")
    print(f"  BDF-2 wall={wall2:.1f}s  max|err|={err2['max_abs']:.4e}  rms={err2['rms']:.4e}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, half_period=HALF_PERIOD, n_periods=N_PERIODS,
            gamma_dot_0=gamma_dot_0, t_end=T_END,
            dt_plateau=DT_PLATEAU, dt_fine=DT_FINE, dt_fine_ratio=DT_FINE_RATIO,
            window=WINDOW,
            err_max_bdf1=err1["max_abs"], err_rms_bdf1=err1["rms"],
            err_max_bdf2=err2["max_abs"], err_rms_bdf2=err2["rms"],
            wall_bdf1=wall1, wall_bdf2=wall2,
        ),
        times=times1, dts=dts1, gamma_dot=gam1, sigma_ana=sigma_ana,
        sigma_bdf1=sig1, sigma_bdf2=sig2,
        snes_reasons_bdf1=rea1, snes_reasons_bdf2=rea2,
    )


if __name__ == "__main__":
    main()
