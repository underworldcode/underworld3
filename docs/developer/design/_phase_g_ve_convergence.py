"""Phase G v5b VE-only convergence: BDF-1 / ETD-1 / ETD-2 (with blends).

Tests temporal accuracy of the v5b ViscoPlasticExplicitElastic class
on the analytical Maxwell oscillatory shear problem. No yield active —
pure VE — so the only source of error is the time integrator.

Setup (matches bench_ve_harmonic):
  Maxwell shear box, η=μ=1 → t_r=1.
  γ̇(t) = γ̇₀·sin(ωt), γ̇₀ = 1.0, ω = π/2 (De = π/2 ≈ 1.57).
  σ(0) = 0 (zero-stress start, no phase-lag artefact).

Analytical:
  σ(t) = (η·γ̇₀/(1+De²)) · [sin(ωt) − De·cos(ωt) + De·exp(−t/τ)]

For each integrator config, sweep Δt and compute RMS error vs analytical.
A first-order method shows error ∝ Δt; a second-order method shows
error ∝ Δt². With v5b operator splitting + softmin yield (off here),
the integrator is the only thing changing.

Output: ``output/phase_g_ve_convergence.npz`` containing per-config
arrays of (Δt, error). Plotted by `_plot_phase_g_ve_convergence.py`.
"""

import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression


# ---------- common harness parameters (match bench_ve_harmonic) ----------
ETA, MU = 1.0, 1.0
T_RELAX = ETA / MU
GAMMA_DOT_0 = 1.0      # → V_top = 0.5·γ̇₀·H = 0.5
OMEGA = np.pi / 2.0    # De = ω·t_r = π/2
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * np.pi / OMEGA
H = 1.0; W = 2.0
RES = (16, 8)

OUT_DIR = "output"


def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    """σ(t) for σ(0)=0, γ̇(t)=γ̇₀·sin(ωt)."""
    t_r = eta / mu
    De = omega * t_r
    pre = eta * gamma_dot_0 / (1.0 + De**2)
    return pre * (np.sin(omega * t) - De * np.cos(omega * t)
                  + De * np.exp(-t / t_r))


def run_one(label, integrator, order, dt, etd_blend=1.0, bdf_blend=1.0):
    """Run a single Δt with the given integrator/order/blend on the
    Maxwell oscillatory test.  Returns (t_array, sigma_sim, sigma_ana, n_steps)."""

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=RES,
        minCoords=(-W/2, -H/2),
        maxCoords=(W/2, H/2),
    )
    u = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, mesh.dim, degree=2, vtype=VarType.VECTOR,
    )
    p = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    cm = uw.constitutive_models.ViscoPlasticExplicitElastic(
        stokes.Unknowns, integrator=integrator, order=order,
    )
    stokes.constitutive_model = cm
    cm.bdf_blend = bdf_blend
    cm.etd_blend = etd_blend
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = sympy.oo  # PURE VE — no yield
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6

    stokes.saddle_preconditioner = 1.0 / cm.K
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([-V_top, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])

    centre = np.array([[0.0, 0.0]])
    DFDt = stokes.Unknowns.DFDt

    t_arr = []; sigma_sim = []; sigma_ana = []
    t_cur = 0.0
    n_steps = int(np.round(T_END / dt))
    actual_dt = T_END / n_steps  # exact Δt that lands at T_END

    for step in range(1, n_steps + 1):
        t_end_step = step * actual_dt
        # γ̇(t)·H/2 = V_top  → V_top(t) = 0.5·γ̇₀·sin(ωt) at the top
        v_now = 0.5 * GAMMA_DOT_0 * float(np.sin(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = actual_dt

        try:
            stokes.solve(zero_init_guess=False, timestep=actual_dt)
        except Exception as exc:
            print(f"  {label} dt={dt:.4g}: solve raised at step {step} — {exc}",
                  flush=True)
            break

        # Probe σ_xy at centre
        sigma_xy = float(uw.function.evaluate(
            stokes.tau.sym[0, 1], centre,
        ).flatten()[0])
        t_arr.append(t_end_step)
        sigma_sim.append(sigma_xy)
        sigma_ana.append(
            maxwell_oscillatory(t_end_step, ETA, MU, GAMMA_DOT_0, OMEGA)
        )
        t_cur = t_end_step

    return (np.array(t_arr), np.array(sigma_sim), np.array(sigma_ana), n_steps)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Δt sweep — sample 4 decades of resolution
    dt_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    # Configs to test
    configs = [
        ("bdf1",       "bdf", 1, 1.0, 1.0),
        ("etd1",       "etd", 1, 1.0, 1.0),
        ("etd2",       "etd", 2, 1.0, 1.0),    # full ETD-2
        ("etd2_b75",   "etd", 2, 0.75, 1.0),   # etd_blend=0.75
        ("etd2_b50",   "etd", 2, 0.50, 1.0),   # etd_blend=0.50
        ("etd2_b25",   "etd", 2, 0.25, 1.0),   # etd_blend=0.25
    ]

    results = {label: {"dt": [], "rms_err": [], "max_err": [], "n_steps": []}
               for label, *_ in configs}

    for label, integrator, order, etd_blend, bdf_blend in configs:
        print(f"\n=== {label} (integrator={integrator}, order={order}, "
              f"etd_blend={etd_blend}, bdf_blend={bdf_blend}) ===", flush=True)
        for dt in dt_values:
            t_run0 = time.time()
            t_arr, sim, ana, n_steps = run_one(
                f"{label}_dt{dt:.4g}".replace(".", "p"),
                integrator, order, dt,
                etd_blend=etd_blend, bdf_blend=bdf_blend,
            )
            elapsed = time.time() - t_run0
            if len(t_arr) < n_steps:
                print(f"  Δt={dt:.4g}: blew up at step {len(t_arr)}/{n_steps}",
                      flush=True)
                # Don't store — convergence point is invalid
                continue
            err = sim - ana
            rms = float(np.sqrt(np.mean(err**2)))
            mx = float(np.max(np.abs(err)))
            results[label]["dt"].append(dt)
            results[label]["rms_err"].append(rms)
            results[label]["max_err"].append(mx)
            results[label]["n_steps"].append(n_steps)
            print(f"  Δt={dt:.4g}  steps={n_steps}  RMS={rms:.4e}  "
                  f"max={mx:.4e}  ({elapsed:.1f}s)", flush=True)

    # Save aggregated results
    flat = {}
    for label, data in results.items():
        for k, v in data.items():
            flat[f"{label}_{k}"] = np.array(v)
    np.savez(os.path.join(OUT_DIR, "phase_g_ve_convergence.npz"), **flat)
    print(f"\n  saved → {os.path.join(OUT_DIR, 'phase_g_ve_convergence.npz')}",
          flush=True)


if __name__ == "__main__":
    main()
