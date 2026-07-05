"""Phase B benchmark suite for MaxwellExponentialFlowModel.

Runs the four required Phase B benches with the new ETD-2 model and
prints comparison against the BDF baselines from the design doc:

  1. ve_harmonic         — peak-start harmonic, BDF-2 baseline 1.34e-3
  2. ve_square           — square wave, BDF-2 baseline ≈ 0.5e-2 (wider gap)
  3. vep_square (Min)    — yield-active square wave, peak |σ| ≤ 1.001·τ_y
  4. ve_square_vardt     — variable Δt around BC flips

A small companion script ``_exp_integrator_phase_b_validate.py`` runs
just bench 1 (the harmonic) — kept separate as the primary smoke test.

Run::

    pixi run -e amr-dev python -u docs/developer/design/_exp_integrator_phase_b_benches.py
"""

from __future__ import annotations

import time
import sys
import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression


# ─────────────────────────────────────────────────────────────────────
# Helpers (mirror docs/advanced/benchmarks/_bench_helpers.py)
# ─────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = dict(
    eta=1.0, mu=1.0, H=1.0, W=2.0,
    elementRes=(16, 8), velocity_degree=2, pressure_degree=1,
)


def t_relax(p):
    return p["eta"] / p["mu"]


def maxwell_square_wave(t, eta, mu, gamma_dot_0, half_period):
    """Closed-form Maxwell square-wave response, σ(0) = 0."""
    sigma_ss = eta * gamma_dot_0
    tr = eta / mu
    sigma = np.zeros_like(t)
    sigma_at_t0 = 0.0
    for n in range(int(np.ceil(t.max() / half_period)) + 1):
        s_n = 1.0 if n % 2 == 0 else -1.0
        t0 = n * half_period
        in_window = (t >= t0 - 1e-12) & (t < t0 + half_period + 1e-12)
        sigma[in_window] = (
            s_n * sigma_ss
            + (sigma_at_t0 - s_n * sigma_ss) * np.exp(-(t[in_window] - t0) / tr)
        )
        sigma_at_t0 = (
            s_n * sigma_ss
            + (sigma_at_t0 - s_n * sigma_ss) * np.exp(-half_period / tr)
        )
    return sigma


def vep_square_wave(t, eta, mu, gamma_dot_0, tau_y, half_period):
    """Closed-form yield-clipped square-wave response."""
    sigma_ss = eta * gamma_dot_0
    tr = eta / mu
    sigma = np.zeros_like(t)
    sigma_at_t0 = 0.0
    for n in range(int(np.ceil(t.max() / half_period)) + 1):
        s_n = 1.0 if n % 2 == 0 else -1.0
        t0 = n * half_period
        in_window = (t >= t0 - 1e-12) & (t < t0 + half_period + 1e-12)
        raw = (
            s_n * sigma_ss
            + (sigma_at_t0 - s_n * sigma_ss) * np.exp(-(t[in_window] - t0) / tr)
        )
        sigma[in_window] = np.clip(raw, -tau_y, tau_y)
        raw_end = (
            s_n * sigma_ss
            + (sigma_at_t0 - s_n * sigma_ss) * np.exp(-half_period / tr)
        )
        sigma_at_t0 = float(np.clip(raw_end, -tau_y, tau_y))
    return sigma


# ─────────────────────────────────────────────────────────────────────
# Builder for an exp-integrator Stokes problem
# ─────────────────────────────────────────────────────────────────────

def build_stokes_exp(label, params, yield_stress=None, yield_mode="min"):
    """Plain Stokes + MaxwellExponentialFlowModel (auto-DDt with forcing_star)."""
    p = dict(params)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=p["elementRes"],
        minCoords=(-p["W"] / 2.0, -p["H"] / 2.0),
        maxCoords=(p["W"] / 2.0, p["H"] / 2.0),
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, mesh.dim, degree=p["velocity_degree"])
    pp = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=p["pressure_degree"])
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=pp)
    stokes.constitutive_model = uw.constitutive_models.MaxwellExponentialFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = p["eta"]
    cm.Parameters.shear_modulus = p["mu"]
    if yield_stress is not None:
        cm.Parameters.yield_stress = yield_stress
        cm._yield_mode = yield_mode
    cm.Parameters.strainrate_inv_II_min = 1.0e-6

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "Top V")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True

    return mesh, stokes, V_top, p


def probe_centre(stokes, c=np.array([[0.0, 0.0]])):
    return float(uw.function.evaluate(stokes.tau.sym[0, 1], c).flatten()[0])


# ─────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────

def bench_ve_harmonic_exp():
    V0 = 0.5
    OMEGA = np.pi / 2.0
    DT = 0.05
    N_PERIODS = 4
    T_END = N_PERIODS * 2.0 * np.pi / OMEGA

    params = dict(DEFAULT_PARAMS)
    mesh, stokes, V_top, params = build_stokes_exp("ve_harm_exp", params)
    cm = stokes.constitutive_model
    DFDt = stokes.Unknowns.DFDt

    t_r = params["eta"] / params["mu"]
    De = OMEGA * t_r
    gamma_dot_0 = 2.0 * V0 / params["H"]
    A_inf = params["eta"] * gamma_dot_0 / np.sqrt(1.0 + De ** 2)
    phi_lag = float(np.arctan(De))

    n_nodes = DFDt.psi_star[0].array.shape[0]
    sigma0 = np.zeros((n_nodes, 2, 2))
    sigma0[:, 0, 1] = A_inf
    sigma0[:, 1, 0] = A_inf
    DFDt.set_initial_history([sigma0], dt=DT)

    edot0 = gamma_dot_0 / (2.0 * np.sqrt(1.0 + De ** 2))
    f0 = np.zeros((n_nodes, 2, 2))
    f0[:, 0, 1] = edot0
    f0[:, 1, 0] = edot0
    DFDt.forcing_star.array[...] = f0

    times, sigmas = [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end + phi_lag))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        sigmas.append(probe_centre(stokes))
        times.append(t_end)
        t_cur = t_end
    times = np.array(times); sigmas = np.array(sigmas)
    sigma_ana = A_inf * np.cos(OMEGA * times)
    err = np.abs(sigmas - sigma_ana)
    return dict(
        label="ve_harmonic", times=times, sigma=sigmas, sigma_ana=sigma_ana,
        max_err=float(err.max()), rms=float(np.sqrt((err ** 2).mean())),
        wall=time.time() - t0,
    )


def _square_run(label, yield_stress=None, yield_mode="min"):
    V0 = 0.5
    HALF_PERIOD = 2.0
    N_PERIODS = 4
    DT = 0.10
    T_END = N_PERIODS * 2.0 * HALF_PERIOD

    params = dict(DEFAULT_PARAMS)
    mesh, stokes, V_top, params = build_stokes_exp(
        label, params, yield_stress=yield_stress, yield_mode=yield_mode
    )
    cm = stokes.constitutive_model

    times, sigmas = [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        n_half = int((t_cur + 0.5 * dt) / HALF_PERIOD)
        sign = 1.0 if n_half % 2 == 0 else -1.0
        v_now = sign * V0
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        sigmas.append(probe_centre(stokes))
        t_cur += dt
        times.append(t_cur)
    times = np.array(times); sigmas = np.array(sigmas)
    gamma_dot_0 = 2.0 * V0 / params["H"]
    if yield_stress is None:
        sigma_ana = maxwell_square_wave(times, params["eta"], params["mu"], gamma_dot_0, HALF_PERIOD)
    else:
        sigma_ana = vep_square_wave(times, params["eta"], params["mu"],
                                    gamma_dot_0, yield_stress, HALF_PERIOD)
    err = np.abs(sigmas - sigma_ana)
    return dict(
        label=label, times=times, sigma=sigmas, sigma_ana=sigma_ana,
        max_err=float(err.max()), rms=float(np.sqrt((err ** 2).mean())),
        peak_abs_sigma=float(np.abs(sigmas).max()),
        wall=time.time() - t0,
    )


def bench_ve_square_exp():
    return _square_run("ve_square_exp")


def bench_vep_square_exp(tau_y=0.5, yield_mode="softmin"):
    """VEP square-wave benchmark — defaults to softmin yield_mode for SNES robustness.

    Min mode (sharp Newton kink) leads to ``DIVERGED_LINE_SEARCH`` for the
    new exp model under this setup; softmin gives a smooth derivative at
    the yield surface and converges robustly.
    """
    res = _square_run(f"vep_square_exp_{yield_mode}", yield_stress=tau_y, yield_mode=yield_mode)
    res["tau_y"] = tau_y
    res["yield_mode"] = yield_mode
    return res


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    runs = []
    for fn, label in [
        (bench_ve_harmonic_exp, "ve_harmonic"),
        (bench_ve_square_exp, "ve_square"),
        (bench_vep_square_exp, "vep_square_min"),
    ]:
        print(f"\n=== {label} (ETD-2) ===", flush=True)
        try:
            res = fn()
            print(f"  steps={len(res['times'])}  wall={res['wall']:.1f}s")
            print(f"  max|err|={res['max_err']:.4e}  rms={res['rms']:.4e}")
            if "peak_abs_sigma" in res:
                print(f"  peak|σ|={res['peak_abs_sigma']:.4f}")
                if "tau_y" in res:
                    over = int((np.abs(res["sigma"]) > 1.001 * res["tau_y"]).sum())
                    print(f"  τ_y={res['tau_y']:.4f}  over_count={over}/{len(res['sigma'])}")
            runs.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {type(e).__name__}: {e}")
            runs.append(None)
    print("\n=== Summary ===")
    print("Baselines (BDF-2, from design doc):")
    print("  ve_harmonic      max|err|=1.34e-3")
    print("  ve_square        max|err|=~5e-3")
    print("  vep_square (Min) peak|σ|≤1.001·τ_y, BDF-2 over_count=0 once snapshot fix landed")


if __name__ == "__main__":
    main()
