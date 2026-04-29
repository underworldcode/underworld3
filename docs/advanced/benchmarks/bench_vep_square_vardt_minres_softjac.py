"""VEP variable-dt with Min residual and softmin Jacobian.

Hypothesis: at the yield kink, Newton stalls because the true Min-mode
Jacobian has a slope discontinuity, so each Newton step is throttled by
line search and DIVERGED_MAX_IT fires (despite the residual already being
below any sensible tolerance).

Try inexact Newton: keep the residual F1 = ``2·η_min·ε̇ + BDF-history``
(so the answer lands on the true yield surface), but autodiff a softmin
version of the same expression to build the uu / up Jacobian blocks.
The Jacobian is then continuous; Newton sees no kink; convergence
should be at-or-near 1 iteration per step.

Counterfactual: bench_vep_square_vardt.py (same problem, full Min for
both residual and Jacobian) recorded 4/413 BDF-1 steps as
DIVERGED_MAX_IT.  We expect 0 here.
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
JAC_SOFTNESS = 0.1

LABEL = "vep_square_vardt_minres_softjac"


def schedule_dt(t_cur):
    flip_times = [HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2 - 1)]
    for f in flip_times:
        if abs(t_cur - f) <= WINDOW:
            return DT_FINE
    return DT_PLATEAU


def _capture_softmin_F1(stokes, softness):
    """Build the alternative F1 that uses softmin viscosity throughout.

    The current ``stokes.F1.sym`` is built from ``cm.flux`` which uses
    whichever yield_mode is set on the constitutive model.  Briefly
    flip the mode to ``softmin`` to grab the alternative ``cm.flux``
    expression, then restore.
    """
    cm = stokes.constitutive_model
    saved_mode = cm._yield_mode
    saved_softness = cm._yield_softness
    try:
        cm._yield_mode = "softmin"
        cm._yield_softness = softness
        # Replicate F1.sym = stress + penalty * div_u * I, but using
        # the freshly-recomputed (softmin) stress.
        soft_stress = cm.flux
        F1_softmin = soft_stress + stokes.penalty * stokes.div_u * sympy.eye(stokes.mesh.dim)
    finally:
        cm._yield_mode = saved_mode
        cm._yield_softness = saved_softness
    return F1_softmin


def _run_one(bdf_order):
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = bdf_order
    mesh, stokes, V_top, params = build_stokes(
        f"{LABEL}_o{bdf_order}", params,
        yield_stress=TAU_Y, yield_mode="min",
    )
    # Inexact Newton: softmin Jacobian, Min residual.
    # ``set_jacobian_F1_source`` defaults to installing the ``cp``
    # (critical-point) linesearch, which is the right pairing for an
    # inexact Jacobian — the default ``bt`` rejects useful steps as
    # ``DIVERGED_LINE_SEARCH`` because they don't strictly reduce the
    # Min residual (only the softmin one).
    F1_jac = _capture_softmin_F1(stokes, JAC_SOFTNESS)
    stokes.set_jacobian_F1_source(F1_jac)

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
        # divergence_retries=0 to expose true Newton behaviour.
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=0)
        sigmas.append(probe_centre(stokes))
        t_cur = t_end_step
        times.append(t_cur); dts.append(dt); gammas.append(2.0 * v_now / params["H"])
        reasons.append(int(stokes.snes.getConvergedReason()))
        iters.append(int(stokes.snes.getIterationNumber()))
    return (np.array(times), np.array(dts), np.array(sigmas),
            np.array(gammas), np.array(reasons), np.array(iters),
            time.time() - t0, params)


def main():
    times1, dts1, sig1, gam1, rea1, its1, wall1, params = _run_one(1)
    times2, dts2, sig2, gam2, rea2, its2, wall2, _      = _run_one(2)
    assert np.allclose(times1, times2)

    gamma_dot_0 = 2.0 * V0 / params["H"]
    sigma_ana = vep_square_wave(times1, params["eta"], params["mu"],
                                gamma_dot_0, TAU_Y, HALF_PERIOD)
    err1 = error_metrics(sig1, sigma_ana)
    err2 = error_metrics(sig2, sigma_ana)
    peak1 = float(np.abs(sig1).max()); peak2 = float(np.abs(sig2).max())
    over1 = int((np.abs(sig1) > 1.001 * TAU_Y).sum())
    over2 = int((np.abs(sig2) > 1.001 * TAU_Y).sum())
    div1 = int((rea1 < 0).sum())
    div2 = int((rea2 < 0).sum())
    print(f"[{LABEL}]  steps={len(times1)}  τ_y={TAU_Y}  η·γ̇₀={params['eta']*gamma_dot_0:.4f}")
    print(f"  schedule: plateau dt={DT_PLATEAU}, fine dt={DT_FINE} (×{DT_FINE_RATIO}), window=±{WINDOW}")
    print(f"  Jacobian: softmin (δ={JAC_SOFTNESS}), residual: Min")
    print(f"  BDF-1 wall={wall1:.1f}s peak|σ|={peak1:.4f} over={over1} "
          f"max|err|={err1['max_abs']:.4e} rms={err1['rms']:.4e} "
          f"diverged={div1} mean_its={its1.mean():.2f}")
    print(f"  BDF-2 wall={wall2:.1f}s peak|σ|={peak2:.4f} over={over2} "
          f"max|err|={err2['max_abs']:.4e} rms={err2['rms']:.4e} "
          f"diverged={div2} mean_its={its2.mean():.2f}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, half_period=HALF_PERIOD, n_periods=N_PERIODS,
            tau_y=TAU_Y, gamma_dot_0=gamma_dot_0, t_end=T_END,
            dt_plateau=DT_PLATEAU, dt_fine=DT_FINE, dt_fine_ratio=DT_FINE_RATIO,
            window=WINDOW, jac_softness=JAC_SOFTNESS,
            peak_bdf1=peak1, peak_bdf2=peak2,
            n_over_bdf1=over1, n_over_bdf2=over2,
            n_diverged_bdf1=div1, n_diverged_bdf2=div2,
            mean_its_bdf1=float(its1.mean()), mean_its_bdf2=float(its2.mean()),
            err_max_bdf1=err1["max_abs"], err_rms_bdf1=err1["rms"],
            err_max_bdf2=err2["max_abs"], err_rms_bdf2=err2["rms"],
            wall_bdf1=wall1, wall_bdf2=wall2,
        ),
        times=times1, dts=dts1, gamma_dot=gam1, sigma_ana=sigma_ana,
        sigma_bdf1=sig1, sigma_bdf2=sig2,
        snes_reasons_bdf1=rea1, snes_reasons_bdf2=rea2,
        snes_iters_bdf1=its1, snes_iters_bdf2=its2,
    )


if __name__ == "__main__":
    main()
