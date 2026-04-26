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
N_PERIODS = 4                # 4 full periods after transient
T_END = N_PERIODS * 2.0 * np.pi / OMEGA + 0.5  # extra to capture tail

LABEL = "ve_harmonic"


def main():
    params = dict(DEFAULT_PARAMS)
    params["bdf_order"] = 2
    mesh, stokes, V_top, params = build_stokes(LABEL, params)

    t_r = t_relax(params)
    De = OMEGA * t_r
    gamma_dot_0 = 2.0 * V0 / params["H"]

    # Pre-allocate
    times, dts, sigmas, gammas, reasons = [], [], [], [], []
    t_cur = 0.0
    t_wall0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        # Set V_top at the midpoint of the step (centred-difference style)
        t_mid = t_cur + 0.5 * dt
        v_now = V0 * float(np.sin(OMEGA * t_mid))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        s = probe_centre(stokes)
        t_cur += dt
        times.append(t_cur)
        dts.append(dt)
        sigmas.append(s)
        gammas.append(2.0 * v_now / params["H"])
        reasons.append(int(stokes.snes.getConvergedReason()))
    t_wall = time.time() - t_wall0
    times = np.array(times)
    dts = np.array(dts)
    sigmas = np.array(sigmas)
    gammas = np.array(gammas)
    reasons = np.array(reasons)

    # Analytical reference
    sigma_ana = maxwell_oscillatory(times, params["eta"], params["mu"], gamma_dot_0, OMEGA)

    # Diagnostics
    err = error_metrics(sigmas, sigma_ana)
    A_sim, phi_sim = fit_amp_phase(times, sigmas, OMEGA)
    A_ana = params["eta"] * gamma_dot_0 / np.sqrt(1.0 + De**2)
    phi_ana = float(np.arctan(De))

    print(f"[{LABEL}]  steps={len(times)}  wall={t_wall:.1f}s")
    print(f"  De = ω·t_r = {De:.4f}")
    print(f"  Amplitude:  sim={A_sim:.4f}  ana={A_ana:.4f}  Δ={A_sim-A_ana:+.4f}")
    print(f"  Phase lag:  sim={phi_sim:.4f}  ana={phi_ana:.4f}  Δ={phi_sim-phi_ana:+.4f} rad")
    print(f"  max|err|={err['max_abs']:.4e}  rms={err['rms']:.4e}  rel={err['rel_max']:.4f}")

    save_run(
        LABEL,
        params=params,
        params_extra=dict(
            V0=V0, omega=OMEGA, gamma_dot_0=gamma_dot_0, De=De,
            t_end=T_END, dt_nominal=DT,
            A_sim=A_sim, A_ana=A_ana, phi_sim=phi_sim, phi_ana=phi_ana,
            err_max=err["max_abs"], err_rms=err["rms"], wall_time=t_wall,
        ),
        times=times, dts=dts, sigma=sigmas, sigma_ana=sigma_ana,
        gamma_dot=gammas, snes_reasons=reasons,
    )


if __name__ == "__main__":
    main()
