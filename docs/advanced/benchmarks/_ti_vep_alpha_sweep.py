"""Find the bdf_blend α threshold for TI-VEP + spatial τ_y stability.

Known so far at T=16, harmonic forcing, BDF-2:
  α = 1.0  →  peak|σ_xy| ≈ 7-10           (blows up modestly)
  α = 0.5  →  peak|σ_xy| ≈ 7-30000         (blow-up reduced but still)
  α = 0.0  →  peak|σ_xy| ≈ 0.30 (BDF-1)    (stable)

What's the smallest α that still blows up?  Sweep at θ=15° (the worst
case in earlier tests) and at θ=0° (where 1D-y blow-up is also seen).
Use the same setup as bench_ti_vep_harmonic_zeroIC at τ_y=0.30.
"""

import time
import numpy as np
import sympy
from bench_ti_vep_harmonic import build_ti_stokes, probe_stress, V0, OMEGA, DT


T_END = 16.0
TAU_Y = 0.30


def run(theta_deg, alpha, label):
    stokes, V_top, n_vec = (None, None, None)
    mesh, stokes, V_top, n_vec = build_ti_stokes(label, theta_deg, TAU_Y, bdf_order=2)
    stokes.constitutive_model._bdf_blend = alpha

    phi = float(np.arctan(OMEGA))
    n_steps = int(T_END / DT)
    sxy = []; tres = []
    div = 0; iters_total = 0
    t0 = time.time()
    for step in range(n_steps):
        t = (step + 1) * DT
        v_now = V0 * float(np.cos(OMEGA * t + phi))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = DT
        stokes.solve(zero_init_guess=False, timestep=DT, divergence_retries=2)
        if stokes.snes.getConvergedReason() < 0:
            div += 1
        iters_total += stokes.snes.getIterationNumber()
        sxy_v, tres_v = probe_stress(stokes, n_vec)
        sxy.append(sxy_v); tres.append(tres_v)
    wall = time.time() - t0
    sxy = np.array(sxy); tres = np.array(tres)
    return dict(label=label, alpha=alpha, theta=theta_deg, wall=wall,
                peak_sxy=float(np.abs(sxy).max()),
                peak_tres=float(np.abs(tres).max()),
                div=div, mean_its=iters_total / max(1, len(sxy)))


def main():
    cases = []
    # θ=0° — easier; α≥0.5 already blows up modestly
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        cases.append((0.0, alpha))
    # θ=15° — harder; α≥0.5 still blows up massively
    for alpha in (0.0, 0.10, 0.25, 0.50):
        cases.append((15.0, alpha))

    print(f"\n{'label':<22} {'θ°':>4} {'α':>5} {'wall':>6} {'div':>4} {'its':>5} "
          f"{'peak|τ_res|':>11} {'peak|σ_xy|':>12}", flush=True)
    for theta, alpha in cases:
        label = f"th{theta:+.0f}_a{alpha:.2f}".replace(".", "p")
        print(f"--- running {label} ---", flush=True)
        r = run(theta, alpha, label)
        print(f"{r['label']:<22} {r['theta']:>4.0f} {r['alpha']:>5.2f} "
              f"{r['wall']:>6.1f} {r['div']:>4d} {r['mean_its']:>5.2f} "
              f"{r['peak_tres']:>11.4e} {r['peak_sxy']:>12.4e}", flush=True)


if __name__ == "__main__":
    main()
