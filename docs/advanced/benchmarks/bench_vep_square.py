"""Benchmark: visco-elastic-plastic shear under square-wave forcing.

Same drive as :mod:`bench_ve_square` but with Min-mode plasticity
(yield stress :math:`\\tau_y < \\eta\\dot\\gamma_0`).  The closed-form
solution is the *clipped* version of the VE square-wave: within each
half-period the stress evolves under Maxwell exponentially toward
:math:`\\pm\\eta\\dot\\gamma_0`, but is held at :math:`\\pm\\tau_y` while the
material is yielding.  When the BC reverses, the next half-period
starts from the (clipped) value :math:`\\pm\\tau_y`.

Closed form
-----------
.. math::
    \\sigma(t) = \\mathrm{clip}\\bigl(s_n\\sigma_{\\mathrm{ss}}
    + (\\sigma_{0,n} - s_n\\sigma_{\\mathrm{ss}})\\, e^{-(t-t_n)/t_r},\\,
    -\\tau_y, +\\tau_y\\bigr)

with :math:`\\sigma_{0,n} = \\mathrm{clip}(\\sigma(t_n), \\pm\\tau_y)` —
i.e.\\ each new half-period starts from the clipped value at the
previous boundary.

Run
---
``pixi run -e amr-dev python docs/advanced/benchmarks/bench_vep_square.py``

Output: ``output/benchmarks/vep_square.npz``.
"""

import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, t_relax, build_stokes, probe_centre,
    vep_square_wave, save_run, error_metrics,
)


V0 = 0.5
TAU_Y = 0.5         # < η·γ̇₀ = 1, so material yields
HALF_PERIOD = 2.0
N_PERIODS = 4
DT = 0.10
T_END = N_PERIODS * 2.0 * HALF_PERIOD

LABEL = "vep_square"


def main():
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = 2
    mesh, stokes, V_top, params = build_stokes(
        LABEL, params, yield_stress=TAU_Y, yield_mode="min",
    )
    gamma_dot_0 = 2.0 * V0 / params["H"]

    times, dts, sigmas, gammas, reasons = [], [], [], [], []
    t_cur = 0.0
    t_wall0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
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
        gammas.append(2.0 * v_now / params["H"])
        reasons.append(int(stokes.snes.getConvergedReason()))
    t_wall = time.time() - t_wall0
    times = np.array(times); dts = np.array(dts); sigmas = np.array(sigmas)
    gammas = np.array(gammas); reasons = np.array(reasons)

    sigma_ana = vep_square_wave(times, params["eta"], params["mu"],
                                gamma_dot_0, TAU_Y, HALF_PERIOD)
    err = error_metrics(sigmas, sigma_ana)
    n_overshoot = int((np.abs(sigmas) > 1.001 * TAU_Y).sum())
    print(f"[{LABEL}]  steps={len(times)}  wall={t_wall:.1f}s")
    print(f"  τ_y = {TAU_Y},  η·γ̇₀ = {params['eta']*gamma_dot_0:.4f}")
    print(f"  peak|σ|={float(np.abs(sigmas).max()):.4f}")
    print(f"  steps with |σ| > 1.001·τ_y: {n_overshoot}")
    print(f"  max|err|={err['max_abs']:.4e}  rms={err['rms']:.4e}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, half_period=HALF_PERIOD, n_periods=N_PERIODS,
            tau_y=TAU_Y, gamma_dot_0=gamma_dot_0, t_end=T_END, dt_nominal=DT,
            peak_abs_sigma=float(np.abs(sigmas).max()),
            n_overshoot=n_overshoot,
            err_max=err["max_abs"], err_rms=err["rms"], wall_time=t_wall,
        ),
        times=times, dts=dts, sigma=sigmas, sigma_ana=sigma_ana,
        gamma_dot=gammas, snes_reasons=reasons,
    )


if __name__ == "__main__":
    main()
