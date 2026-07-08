"""TI-VEP harmonic benchmark — variant with σ=0 initial condition.

Sanity check on the previous run (peak-start IC) which produced
catastrophic BDF-2 blow-up.  Hypothesis: planting σ_xy = A_∞
uniformly puts the fault yield zone at 3-4× its yield stress at
t=0, and BDF-2's inconsistent ψ*₀/ψ*₁ history then drives an
unstable plastic correction that grows.  σ=0 IC avoids that.

Driving uses a cos forcing that *does* start at peak (V_top = V0
at t=0), so we expect a transient before settling on the steady
cycle — but the solver should remain stable throughout.

Same suite (3 angles × 2 τ_y × 2 BDF orders).
"""

import os
import time
import numpy as np
import sympy

from _bench_helpers import OUTPUT_DIR
from bench_ti_vep_harmonic import (
    V0, OMEGA, DT, T_END, ETA_0, ETA_1, MU,
    FAULT_LENGTH, FAULT_WIDTH, RES,
    ANGLES_DEG, TAU_Y_LIST, BDF_ORDERS,
    build_ti_stokes, probe_stress,
)


def _run_one(theta_deg, tau_y, bdf_order, label):
    mesh, stokes, V_top, n_vec = build_ti_stokes(
        label, theta_deg, tau_y, bdf_order,
    )
    # σ=0 IC — let DDt initialise history from current value (which is 0)
    # on the first solve. No set_initial_history call.

    t_r = ETA_1 / MU
    De = OMEGA * t_r
    # BCs: Top moves, Bottom fixed → γ̇_0 = V0/H (not 2·V0/H).
    gamma_dot_0 = V0 / 1.0
    A_inf = ETA_1 * gamma_dot_0 / np.sqrt(1.0 + De**2)
    phi = float(np.arctan(De))

    times, sxy_h, tres_h, reasons, iters = [], [], [], [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step + phi))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        sxy, tres = probe_stress(stokes, n_vec)
        t_cur = t_end_step
        times.append(t_cur); sxy_h.append(sxy); tres_h.append(tres)
        reasons.append(int(stokes.snes.getConvergedReason()))
        iters.append(int(stokes.snes.getIterationNumber()))
    wall = time.time() - t0

    times = np.array(times); sxy_h = np.array(sxy_h); tres_h = np.array(tres_h)
    reasons = np.array(reasons); iters = np.array(iters)
    sigma_ve = A_inf * np.cos(OMEGA * times)
    return dict(
        times=times, sigma_xy=sxy_h, tau_resolved=tres_h,
        sigma_ve=sigma_ve, reasons=reasons, iters=iters,
        wall=wall, A_inf=A_inf, phi=phi, De=De, gamma_dot_0=gamma_dot_0,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    for theta_deg in ANGLES_DEG:
        for tau_y in TAU_Y_LIST:
            results = {}
            for bdf in BDF_ORDERS:
                lbl = f"tivep_zIC_o{bdf}_th{theta_deg:+.0f}_ty{tau_y:.2f}".replace(
                    ".", "p"
                )
                print(f"\n--- {lbl}: θ={theta_deg}°, τ_y={tau_y}, BDF-{bdf} (σ=0 IC) ---",
                      flush=True)
                results[bdf] = _run_one(theta_deg, tau_y, bdf, lbl)
                r = results[bdf]
                ndiv = int((r["reasons"] < 0).sum())
                print(f"    wall={r['wall']:.1f}s  steps={len(r['times'])}  "
                      f"diverged={ndiv}  mean_its={float(r['iters'].mean()):.2f}  "
                      f"peak|τ_resolved|={float(np.abs(r['tau_resolved']).max()):.4f}  "
                      f"peak|σ_xy|={float(np.abs(r['sigma_xy']).max()):.4f}",
                      flush=True)
                summary.append(dict(
                    label=lbl, theta=theta_deg, tau_y=tau_y, bdf=bdf,
                    wall=r["wall"], diverged=ndiv,
                    mean_its=float(r["iters"].mean()),
                    peak_resolved=float(np.abs(r["tau_resolved"]).max()),
                    peak_sxy=float(np.abs(r["sigma_xy"]).max()),
                ))

            tag = f"ti_vep_harmonic_zIC_th{theta_deg:+.0f}_ty{tau_y:.2f}".replace(
                ".", "p"
            )
            np.savez(
                os.path.join(OUTPUT_DIR, f"{tag}.npz"),
                theta_deg=theta_deg, tau_y=tau_y,
                times=results[1]["times"],
                sigma_xy_bdf1=results[1]["sigma_xy"],
                sigma_xy_bdf2=results[2]["sigma_xy"],
                tau_resolved_bdf1=results[1]["tau_resolved"],
                tau_resolved_bdf2=results[2]["tau_resolved"],
                sigma_ve=results[1]["sigma_ve"],
                reasons_bdf1=results[1]["reasons"],
                reasons_bdf2=results[2]["reasons"],
                iters_bdf1=results[1]["iters"],
                iters_bdf2=results[2]["iters"],
                A_inf=results[1]["A_inf"], De=results[1]["De"],
                gamma_dot_0=results[1]["gamma_dot_0"],
                wall_bdf1=results[1]["wall"], wall_bdf2=results[2]["wall"],
                V0=V0, OMEGA=OMEGA, DT=DT, T_END=T_END,
                ETA_0=ETA_0, ETA_1=ETA_1, MU=MU,
                FAULT_WIDTH=FAULT_WIDTH, FAULT_LENGTH=FAULT_LENGTH, RES=RES,
            )
            print(f"  saved → {tag}.npz", flush=True)

    print("\n=== summary (σ=0 IC) ===", flush=True)
    print(f"{'label':<40} {'θ°':>4} {'τ_y':>5} {'BDF':>4} {'wall':>6} "
          f"{'div':>4} {'its':>5} {'peak|τ_res|':>11} {'peak|σ_xy|':>11}",
          flush=True)
    for s in summary:
        print(f"{s['label']:<40} {s['theta']:>4.0f} {s['tau_y']:>5.2f} "
              f"{s['bdf']:>4d} {s['wall']:>6.1f} {s['diverged']:>4d} "
              f"{s['mean_its']:>5.2f} {s['peak_resolved']:>11.4f} "
              f"{s['peak_sxy']:>11.4f}", flush=True)


if __name__ == "__main__":
    main()
