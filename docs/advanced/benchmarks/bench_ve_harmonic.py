"""Benchmark: Maxwell viscoelastic shear under sinusoidal forcing.

Drives the shear box with :math:`V_{top}(t) = V_0 \\sin(\\omega t)` and
compares the centre-point shear stress against the closed-form
solution.  Records amplitude, phase shift, and error norms.

Closed form
-----------
For Maxwell with constant :math:`\\eta, \\mu` driven by
:math:`\\dot\\gamma(t) = \\dot\\gamma_0 \\sin(\\omega t)`,

.. math::
    \\sigma(t) = \\frac{\\eta\\dot\\gamma_0}{1 + \\mathrm{De}^2}
    \\bigl[\\sin(\\omega t) - \\mathrm{De}\\cos(\\omega t)
    + \\mathrm{De}\\,e^{-t/t_r}\\bigr]

with :math:`\\mathrm{De} = \\omega t_r` (Deborah number).  Steady amplitude
:math:`A_{\\infty} = \\eta\\dot\\gamma_0 / \\sqrt{1+\\mathrm{De}^2}`, phase
lag :math:`\\varphi = \\arctan(\\mathrm{De})`.

Run
---
``pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_harmonic.py``

Output: ``output/benchmarks/ve_harmonic.npz`` containing the simulation
trace, the analytical reference at the same time points, and parameter
metadata.  See ``plot_benchmarks.py`` for plotting from the npz.
"""

import time
import numpy as np
import sympy
from _bench_helpers import (
    DEFAULT_PARAMS, t_relax, build_stokes, probe_centre,
    maxwell_oscillatory, save_run, error_metrics, fit_amp_phase,
)


# Run-specific parameters
V0 = 0.5                     # → γ̇₀ = 2·V0/H = 1.0 in the symmetric strain rate
OMEGA = np.pi / 2.0          # period 4·t_r → De = π/2 ≈ 1.57
DT = 0.05                    # ~80 steps per period; resolves the harmonic
N_PERIODS = 4                # 4 full periods (no warmup needed — see below)
T_END = N_PERIODS * 2.0 * np.pi / OMEGA   # 4 periods exactly

LABEL = "ve_harmonic"

# Initial condition design: start at a point in the steady-state cycle
# where σ̇ = 0, so σ(0) is consistent with the analytical and there is
# *no* startup transient.  Choose BC such that σ_ss(t) = A_∞·cos(ωt),
# i.e. peak at t=0.  Working backwards through the Maxwell phase
# response (lag φ = arctan(De)), this requires
#   V_top(t) = V_0 · cos(ωt + φ)
# so that ε̇_xy(t) = (V_0/H)·cos(ωt + φ) and the steady-state response
# σ_ss(t) = A_∞·cos(ωt + φ - φ) = A_∞·cos(ωt).
#
# The initial condition σ(0) = A_∞ matches the steady-state at t=0
# exactly, leaving no homogeneous (decaying) component — so the entire
# recorded trace is on the steady cycle.


def _run_one(bdf_order):
    """Run the simulation at one BDF order with peak-start initial condition.

    See module docstring above for why σ(0) = A_∞ paired with the cosine
    forcing eliminates the startup transient.  V_top is sampled at the
    *endpoint* of each step (BDF expects the value at the new time).
    """
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = bdf_order
    mesh, stokes, V_top, params = build_stokes(f"{LABEL}_o{bdf_order}", params)

    t_r = params["eta"] / params["mu"]
    De = OMEGA * t_r
    gamma_dot_0 = 2.0 * V0 / params["H"]
    A_inf = params["eta"] * gamma_dot_0 / np.sqrt(1.0 + De**2)
    phi = float(np.arctan(De))

    # Plant the steady-state cycle as the initial condition.  σ_ss(t) =
    # A_∞·cos(ωt) is the analytical solution under our cos forcing —
    # zero homogeneous component, σ̇(0) = 0.  History slot k is the
    # value at t = -k·Δt, which by cosine evenness is A_∞·cos(k·ω·Δt).
    #
    # Using the *exact* per-slot value (not just A_∞ for all k) is what
    # actually buys the benefit: the alternative drops O(Δt²) error into
    # ψ*[1], which then contaminates BDF-2's truncation right from
    # step 1 — exactly the kind of phase error we are trying to avoid.
    #
    # Also bypass the DDt's effective_order ramp: with all history slots
    # already populated, the very first solve can use full BDF order
    # rather than starting at BDF-1 and ramping up.
    ddt = stokes.DFDt
    ddt._history_initialised = True
    for k in range(ddt.order):
        val_k = A_inf * float(np.cos(OMEGA * k * DT))
        ddt.psi_star[k].array[:, 0, 1] = val_k
        ddt.psi_star[k].array[:, 1, 0] = val_k
    # Tell the DDt that bdf_order full history slots are already
    # populated, so effective_order = bdf_order from step 1.  Otherwise
    # _n_solves_completed = 0 forces effective_order = 1 on the first
    # solve, which would re-introduce BDF-1 startup error.
    ddt._n_solves_completed = ddt.order
    if bdf_order >= 2:
        ddt._dt_history = [DT] * ddt.order

    times, dts, sigmas, gammas, reasons = [], [], [], [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        # BC: V_top(t) = V_0·cos(ωt + φ) so σ_ss(t) = A_∞·cos(ωt).
        v_now = V0 * float(np.cos(OMEGA * t_end_step + phi))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        s = probe_centre(stokes)
        t_cur = t_end_step
        times.append(t_cur); dts.append(dt); sigmas.append(s)
        gammas.append(2.0 * v_now / params["H"])
        reasons.append(int(stokes.snes.getConvergedReason()))
    return (np.array(times), np.array(dts), np.array(sigmas),
            np.array(gammas), np.array(reasons), time.time() - t0, params)


def main():
    times1, dts1, sig1, gam1, rea1, wall1, params = _run_one(1)
    times2, dts2, sig2, gam2, rea2, wall2, _      = _run_one(2)
    assert np.allclose(times1, times2)

    t_r = t_relax(params)
    De = OMEGA * t_r
    gamma_dot_0 = 2.0 * V0 / params["H"]
    A_inf = params["eta"] * gamma_dot_0 / np.sqrt(1.0 + De**2)
    # Peak-start initial condition + cos(ωt + φ) forcing → no transient,
    # so the analytical is the steady-state cycle σ(t) = A_∞·cos(ωt).
    sigma_ana = A_inf * np.cos(OMEGA * times1)

    err1 = error_metrics(sig1, sigma_ana)
    err2 = error_metrics(sig2, sigma_ana)
    A1, phi1 = fit_amp_phase(times1, sig1, OMEGA)
    A2, phi2 = fit_amp_phase(times2, sig2, OMEGA)
    A_ana = params["eta"] * gamma_dot_0 / np.sqrt(1.0 + De**2)
    phi_ana = float(np.arctan(De))

    print(f"[{LABEL}]  steps={len(times1)}  De=ω·t_r={De:.4f}")
    print(f"  BDF-1 wall={wall1:.1f}s  max|err|={err1['max_abs']:.4e}  rms={err1['rms']:.4e}")
    print(f"        amp sim={A1:.4f} ana={A_ana:.4f}   phi sim={phi1:.4f} ana={phi_ana:.4f}")
    print(f"  BDF-2 wall={wall2:.1f}s  max|err|={err2['max_abs']:.4e}  rms={err2['rms']:.4e}")
    print(f"        amp sim={A2:.4f} ana={A_ana:.4f}   phi sim={phi2:.4f} ana={phi_ana:.4f}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, omega=OMEGA, gamma_dot_0=gamma_dot_0, De=De,
            t_end=T_END, dt_nominal=DT,
            A_bdf1=A1, A_bdf2=A2, A_ana=A_ana,
            phi_bdf1=phi1, phi_bdf2=phi2, phi_ana=phi_ana,
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
