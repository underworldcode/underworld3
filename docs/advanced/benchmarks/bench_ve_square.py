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


def main():
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = 2
    mesh, stokes, V_top, params = build_stokes(LABEL, params)
    gamma_dot_0 = 2.0 * V0 / params["H"]

    times, dts, sigmas, gammas, reasons, signs = [], [], [], [], [], []
    t_cur = 0.0
    t_wall0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        # Sign at midpoint of step
        t_mid = t_cur + 0.5 * dt
        n_half = int(t_mid / HALF_PERIOD)
        sign = 1.0 if n_half % 2 == 0 else -1.0
        v_now = sign * V0
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        s = probe_centre(stokes)
        t_cur += dt
        times.append(t_cur); dts.append(dt); sigmas.append(s)
        gammas.append(2.0 * v_now / params["H"]); signs.append(sign)
        reasons.append(int(stokes.snes.getConvergedReason()))
    t_wall = time.time() - t_wall0
    times = np.array(times); dts = np.array(dts); sigmas = np.array(sigmas)
    gammas = np.array(gammas); reasons = np.array(reasons); signs = np.array(signs)

    sigma_ana = maxwell_square_wave(times, params["eta"], params["mu"],
                                    gamma_dot_0, HALF_PERIOD)
    err = error_metrics(sigmas, sigma_ana)
    print(f"[{LABEL}]  steps={len(times)}  wall={t_wall:.1f}s")
    print(f"  half_period={HALF_PERIOD} t_r,  σ_ss = η·γ̇₀ = {params['eta']*gamma_dot_0:.4f}")
    print(f"  max|err|={err['max_abs']:.4e}  rms={err['rms']:.4e}  rel={err['rel_max']:.4f}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, half_period=HALF_PERIOD, n_periods=N_PERIODS,
            gamma_dot_0=gamma_dot_0, t_end=T_END, dt_nominal=DT,
            err_max=err["max_abs"], err_rms=err["rms"], wall_time=t_wall,
        ),
        times=times, dts=dts, sigma=sigmas, sigma_ana=sigma_ana,
        gamma_dot=gammas, snes_reasons=reasons,
    )


if __name__ == "__main__":
    main()
