"""Convergence sweep for the three VE/VEP benchmarks.

For each case (harmonic, square, VEP square) and each BDF order
(1, 2), runs the simulation at a range of timestep sizes and records
max-absolute and RMS error vs the closed-form solution.  Writes
``output/benchmarks/convergence_<case>.npz`` containing the full
sweep so the convergence figure can be regenerated without re-running.

Run
---
``pixi run -e amr-dev python docs/advanced/benchmarks/bench_convergence.py``

The full sweep is ~24 runs and takes a few minutes.
"""

import os
import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, t_relax, build_stokes, probe_centre,
    maxwell_oscillatory, maxwell_square_wave, vep_square_wave,
    save_run, error_metrics, OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Per-case runners.  Each takes (dt, bdf_order, **overrides) and returns
# (times, sigmas, sigma_ana, params).
# ---------------------------------------------------------------------------

def run_ve_harmonic(dt, bdf_order, V0=0.5, omega=np.pi/2.0, n_periods=4):
    """Endpoint V_top sampling — see bench_ve_harmonic.py for the rationale.

    Midpoint sampling is 1st-order accurate to the value BDF-2 wants
    at the step endpoint and would limit BDF-2 to slope-1 convergence.
    """
    label = f"ve_h_dt{dt:.4f}_o{bdf_order}"
    params = dict(DEFAULT_PARAMS); params["bdf_order"] = bdf_order
    _, stokes, V_top, params = build_stokes(label, params)
    gd0 = 2.0 * V0 / params["H"]
    t_end = n_periods * 2.0 * np.pi / omega + 0.5

    times, sigmas = [], []
    t_cur = 0.0
    while t_cur < t_end - 1e-9:
        ds = min(dt, t_end - t_cur)
        t_end_step = t_cur + ds
        v_now = V0 * float(np.sin(omega * t_end_step))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = ds
        stokes.solve(zero_init_guess=False, timestep=ds, divergence_retries=2)
        sigmas.append(probe_centre(stokes))
        t_cur = t_end_step
        times.append(t_cur)
    times = np.array(times); sigmas = np.array(sigmas)
    sigma_ana = maxwell_oscillatory(times, params["eta"], params["mu"], gd0, omega)
    return times, sigmas, sigma_ana, params


def run_ve_square(dt, bdf_order, V0=0.5, half_period=2.0, n_periods=4):
    label = f"ve_s_dt{dt:.4f}_o{bdf_order}"
    params = dict(DEFAULT_PARAMS); params["bdf_order"] = bdf_order
    _, stokes, V_top, params = build_stokes(label, params)
    gd0 = 2.0 * V0 / params["H"]
    t_end = n_periods * 2.0 * half_period

    times, sigmas = [], []
    t_cur = 0.0
    while t_cur < t_end - 1e-9:
        ds = min(dt, t_end - t_cur)
        n_half = int((t_cur + 0.5 * ds) / half_period)
        sign = 1.0 if n_half % 2 == 0 else -1.0
        V_top.sym = sympy.Float(sign * V0)
        stokes.constitutive_model.Parameters.dt_elastic = ds
        stokes.solve(zero_init_guess=False, timestep=ds, divergence_retries=2)
        sigmas.append(probe_centre(stokes))
        t_cur += ds
        times.append(t_cur)
    times = np.array(times); sigmas = np.array(sigmas)
    sigma_ana = maxwell_square_wave(times, params["eta"], params["mu"], gd0, half_period)
    return times, sigmas, sigma_ana, params


def run_vep_square(dt, bdf_order, V0=0.5, tau_y=0.5, half_period=2.0, n_periods=4):
    label = f"vep_s_dt{dt:.4f}_o{bdf_order}"
    params = dict(DEFAULT_PARAMS); params["bdf_order"] = bdf_order
    _, stokes, V_top, params = build_stokes(
        label, params, yield_stress=tau_y, yield_mode="min",
    )
    gd0 = 2.0 * V0 / params["H"]
    t_end = n_periods * 2.0 * half_period

    times, sigmas = [], []
    t_cur = 0.0
    while t_cur < t_end - 1e-9:
        ds = min(dt, t_end - t_cur)
        n_half = int((t_cur + 0.5 * ds) / half_period)
        sign = 1.0 if n_half % 2 == 0 else -1.0
        V_top.sym = sympy.Float(sign * V0)
        stokes.constitutive_model.Parameters.dt_elastic = ds
        stokes.solve(zero_init_guess=False, timestep=ds, divergence_retries=2)
        sigmas.append(probe_centre(stokes))
        t_cur += ds
        times.append(t_cur)
    times = np.array(times); sigmas = np.array(sigmas)
    sigma_ana = vep_square_wave(times, params["eta"], params["mu"],
                                gd0, tau_y, half_period)
    return times, sigmas, sigma_ana, params


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def sweep(case_name, runner, dts, orders, **runner_kwargs):
    """Run a sweep over (dt, order); return arrays + metrics dict.

    Also stores per-run traces so that re-plotting at any (order, dt)
    combination doesn't require re-running.  Trace arrays are stored
    as ``trace_t_o<order>_dt<dt:.4f>``, etc.
    """
    results = []
    extra_arrays = {}
    for order in orders:
        for dt in dts:
            t0 = time.time()
            times, sigmas, sigma_ana, params = runner(dt, order, **runner_kwargs)
            err = error_metrics(sigmas, sigma_ana)
            wall = time.time() - t0
            print(f"  [{case_name}]  order={order}  dt={dt:.4f}  "
                  f"steps={len(times)}  wall={wall:.1f}s  "
                  f"max|err|={err['max_abs']:.4e}  rms={err['rms']:.4e}",
                  flush=True)
            results.append(dict(
                order=order, dt=dt, n_steps=len(times),
                max_abs=err["max_abs"], rms=err["rms"], wall=wall,
            ))
            # Store traces for replotting — keyed by (order, dt)
            tag = f"o{order}_dt{dt:.4f}"
            extra_arrays[f"trace_t_{tag}"] = times
            extra_arrays[f"trace_sigma_{tag}"] = sigmas
            extra_arrays[f"trace_ana_{tag}"] = sigma_ana
    return dict(
        order=np.array([r["order"] for r in results]),
        dt=np.array([r["dt"] for r in results]),
        n_steps=np.array([r["n_steps"] for r in results]),
        max_abs=np.array([r["max_abs"] for r in results]),
        rms=np.array([r["rms"] for r in results]),
        wall=np.array([r["wall"] for r in results]),
        **extra_arrays,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reasonable dt range for each case.
    DTS_HARMONIC = [0.40, 0.20, 0.10, 0.05, 0.025]   # 5 values × 2 orders
    DTS_SQUARE   = [0.40, 0.20, 0.10, 0.05]          # 4 values; 0.025 not needed
    DTS_VEP      = [0.40, 0.20, 0.10, 0.05]          # same as VE square
    ORDERS = [1, 2]

    print("=== Convergence: VE harmonic (sin forcing) ===")
    res = sweep("ve_h", run_ve_harmonic, DTS_HARMONIC, ORDERS)
    save_run("convergence_ve_harmonic", params=DEFAULT_PARAMS,
             params_extra=dict(orders=list(ORDERS), dts=list(DTS_HARMONIC)),
             **res)

    print("\n=== Convergence: VE square wave ===")
    res = sweep("ve_s", run_ve_square, DTS_SQUARE, ORDERS)
    save_run("convergence_ve_square", params=DEFAULT_PARAMS,
             params_extra=dict(orders=list(ORDERS), dts=list(DTS_SQUARE)),
             **res)

    print("\n=== Convergence: VEP square wave (Min mode) ===")
    res = sweep("vep_s", run_vep_square, DTS_VEP, ORDERS)
    save_run("convergence_vep_square", params=DEFAULT_PARAMS,
             params_extra=dict(orders=list(ORDERS), dts=list(DTS_VEP)),
             **res)


if __name__ == "__main__":
    main()
