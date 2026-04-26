"""Benchmark: Maxwell viscoelastic shear under square-wave forcing.

Drives the shear box with a square-wave :math:`V_{top}(t)` (sign flips
every ``half_period``) and compares the centre-point shear stress
against the closed-form piecewise-exponential solution.

Closed form
-----------
Within the n-th half-period (sign :math:`s_n = (-1)^n`):

.. math::
    \\sigma(t) = s_n \\sigma_{\\mathrm{ss}}
    + (\\sigma_{0,n} - s_n\\sigma_{\\mathrm{ss}})\\, e^{-(t-t_n)/t_r}

with :math:`\\sigma_{\\mathrm{ss}} = \\eta\\dot\\gamma_0` and
:math:`\\sigma_{0,n}` the stress at the start of half-period n.

Run
---
``pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_square.py``

Output: ``output/benchmarks/ve_square.npz``.
"""

import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, t_relax, build_stokes, probe_centre,
    maxwell_square_wave, save_run, error_metrics,
)


V0 = 0.5
HALF_PERIOD = 2.0    # in units of t_r
N_PERIODS = 4        # → t_end = 4 · 2 · t_r = 8 t_r
DT = 0.10            # 20 steps per half-period
T_END = N_PERIODS * 2.0 * HALF_PERIOD

LABEL = "ve_square"


def _run_one(bdf_order):
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = bdf_order
    mesh, stokes, V_top, params = build_stokes(f"{LABEL}_o{bdf_order}", params)

    times, dts, sigmas, gammas, reasons = [], [], [], [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        n_half = int((t_cur + 0.5 * dt) / HALF_PERIOD)
        sign = 1.0 if n_half % 2 == 0 else -1.0
        v_now = sign * V0
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        sigmas.append(probe_centre(stokes))
        t_cur += dt
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
    print(f"  BDF-1 wall={wall1:.1f}s  max|err|={err1['max_abs']:.4e}  rms={err1['rms']:.4e}")
    print(f"  BDF-2 wall={wall2:.1f}s  max|err|={err2['max_abs']:.4e}  rms={err2['rms']:.4e}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, half_period=HALF_PERIOD, n_periods=N_PERIODS,
            gamma_dot_0=gamma_dot_0, t_end=T_END, dt_nominal=DT,
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
