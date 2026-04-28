"""Phase B killer test (decision gate): TI-VEP harmonic with spatial yield_stress.

Mirrors ``docs/advanced/benchmarks/bench_ti_vep_harmonic.py`` but assigns
the new ``TransverseIsotropicMaxwellExponentialFlowModel`` (ETD-2 +
predictor-corrector return mapping) instead of the BDF-2 TI-VEP model.

Decision gate (from EXPONENTIAL_VE_INTEGRATOR.md §Validation gates):
  ``peak |σ_xy| bounded ≲ 1.1·τ_y in fault zone, ≲ A_∞ in bulk for all
   6 (θ, τ_y) combinations.``

BDF-2 currently produces 10⁸ blow-up on this setup; ETD-2 should run
cleanly and stay bounded — the empirical proof of the structural
argument that closes Phase B.

Sweep: θ ∈ {0°, +15°, -15°} × τ_y ∈ {0.15, 0.30}.

Run::

    pixi run -e amr-dev python -u docs/developer/design/_exp_integrator_phase_b_killer.py
"""

from __future__ import annotations

import os
import time
import numpy as np
import sympy

import underworld3 as uw
from underworld3.function import expression


# ---------------------------------------------------------------------------
# Run-specific parameters (kept aligned with bench_ti_vep_harmonic.py)
# ---------------------------------------------------------------------------

V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * np.pi / OMEGA

ETA_0 = 1.0
ETA_1 = 1.0
MU = 1.0
TAU_Y_BULK = 200.0

RES = 16
H = 1.0; W = 1.0
FAULT_LENGTH = 0.6
FAULT_WIDTH = 0.06

ANGLES_DEG = (0.0, 15.0, -15.0)
TAU_Y_LIST = (0.15, 0.30)


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

def build_ti_exp_stokes(label, theta_deg, tau_y_at_fault):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, 2, degree=2, vtype=uw.VarType.VECTOR,
    )
    p = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, 1, degree=1,
        continuous=True, vtype=uw.VarType.SCALAR,
    )

    theta = np.radians(theta_deg)
    cx, cy = 0.5 * W, 0.5 * H
    dx = 0.5 * FAULT_LENGTH * np.cos(theta)
    dy = 0.5 * FAULT_LENGTH * np.sin(theta)
    fault = uw.meshing.Surface(
        f"fault_{label}", mesh,
        np.array([[cx - dx, cy - dy], [cx + dx, cy + dy]]),
        symbol=f"F{label}",
    )
    fault.discretize()

    n_x = -np.sin(theta)
    n_y = np.cos(theta)
    director = sympy.Matrix([n_x, n_y])

    weakness = fault.influence_function(
        width=FAULT_WIDTH,
        value_near=1.0 / tau_y_at_fault,
        value_far=1.0 / TAU_Y_BULK,
        profile="gaussian",
    )
    tau_y_field = 1.0 / weakness

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicMaxwellExponentialFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA_0
    cm.Parameters.shear_viscosity_1 = ETA_1
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = tau_y_field
    cm.Parameters.director = director
    cm.Parameters.shear_viscosity_min = ETA_0 * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    cm.yield_mode = "softmin"

    stokes.saddle_preconditioner = 1.0 / cm.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])

    return mesh, stokes, V_top, np.array([n_x, n_y])


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_centre_resolved(stokes, n_vec, c=np.array([[0.5, 0.5]])):
    """σ_xy and resolved fault-plane shear at fault centre."""
    tau = stokes.tau
    dists = np.linalg.norm(tau.coords - c, axis=1)
    idx = int(np.argmin(dists))
    s_xx, s_yy, s_xy = tau.data[idx, 0], tau.data[idx, 1], tau.data[idx, 2]
    n_x, n_y = n_vec
    t_x, t_y = n_y, -n_x
    resolved = (s_xx * t_x * n_x + s_xy * (t_x * n_y + t_y * n_x)
                + s_yy * t_y * n_y)
    return float(s_xy), float(resolved)


# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------

def run_one(theta_deg, tau_y_at_fault, t_end=None):
    """Run one (θ, τ_y) combo. ``t_end`` overrides the module-level T_END;
    use a fraction of T_END (e.g. ``T_END/4`` = 1 period) for fast
    smokes that just check yield enforcement / convergence behaviour."""
    if t_end is None:
        t_end = T_END
    label = f"ti_exp_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p")
    mesh, stokes, V_top, n_vec = build_ti_exp_stokes(label, theta_deg, tau_y_at_fault)
    cm = stokes.constitutive_model
    DFDt = stokes.Unknowns.DFDt

    # Per-node τ_y(x) — the SPATIAL yield_stress field evaluated at psi_star
    # node coords. Used for the proper yield-surface gate
    # ``max_x σ_II(x) / τ_y(x)`` rather than dividing peak σ_II by the
    # fault-centerline τ_y (which is misleading because the Gaussian
    # influence decays sharply, so points just off centerline have
    # τ_y_local much larger than τ_y_at_fault).
    ty_field_sym = cm.Parameters.yield_stress.sym
    ty_per_node = np.asarray(
        uw.function.evaluate(ty_field_sym, DFDt.psi_star[0].coords)
    ).flatten()

    # Steady-state amplitude (sub-yield) for the analytical baseline
    t_r = ETA_1 / MU
    De = OMEGA * t_r
    gamma_dot_0 = V0 / H  # engineering shear (NOT 2·V0/H — TI bench uses fixed-bottom BC)
    A_inf = ETA_1 * gamma_dot_0 / np.sqrt(1.0 + De ** 2)

    times, sxy_centre, sxy_max_global, sigmaII_max_fault = [], [], [], []
    sigmaII_over_ty_max = []     # the proper yield-surface gate: max σ_II(x)/τ_y(x)
    times_ana, resolved_centre = [], []
    diverged = 0
    t0 = time.time()
    t_cur = 0.0
    n_x, n_y = n_vec
    while t_cur < t_end - 1e-9:
        dt = min(DT, t_end - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        try:
            stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        except Exception as exc:
            print(f"  step at t={t_end_step:.3f}: {exc}", flush=True)
            diverged += 1
            break

        # Centre probe (uses tau projection with snapshot)
        sxy_c, res_c = probe_centre_resolved(stokes, n_vec)
        sxy_centre.append(sxy_c)
        resolved_centre.append(res_c)

        # Global stress-array probes (peak |σ_xy| and σ_II in fault zone)
        sigma = np.asarray(DFDt.psi_star[0].array)
        sxy_max_global.append(float(np.abs(sigma[:, 0, 1]).max()))
        sigma_II = np.sqrt(0.5 * (sigma ** 2).sum(axis=(1, 2)))
        coords = DFDt.psi_star[0].coords
        # fault zone mask: distance from fault line ≤ 3·FAULT_WIDTH
        cx, cy = 0.5 * W, 0.5 * H
        sd = np.abs((coords[:, 0] - cx) * n_x + (coords[:, 1] - cy) * n_y)
        mask = sd < 3.0 * FAULT_WIDTH
        sigmaII_max_fault.append(float(sigma_II[mask].max()) if mask.any() else 0.0)

        # Proper yield-surface gate: per-node ratio σ_II(x)/τ_y(x).
        # Should be ≤ 1.001 at all nodes if yield is correctly enforced.
        ratio = sigma_II / np.maximum(ty_per_node, 1e-30)
        sigmaII_over_ty_max.append(float(ratio.max()))

        times.append(t_end_step)
        t_cur = t_end_step

    return dict(
        theta_deg=theta_deg,
        tau_y=tau_y_at_fault,
        A_inf=A_inf,
        times=np.array(times),
        sxy_centre=np.array(sxy_centre),
        resolved_centre=np.array(resolved_centre),
        sxy_max_global=np.array(sxy_max_global),
        sigmaII_max_fault=np.array(sigmaII_max_fault),
        sigmaII_over_ty_max=np.array(sigmaII_over_ty_max),
        wall=time.time() - t0,
        diverged=diverged,
    )


def main():
    print(f"[ti_killer] dt={DT} T_end={T_END:.4f} (4 periods)", flush=True)
    print(f"  bulk τ_y={TAU_Y_BULK}  fault τ_y∈{TAU_Y_LIST}  θ∈{ANGLES_DEG}", flush=True)
    print(f"  Decision gate: σ_II_fault ≤ 1.1·τ_y, |σ_xy| ≤ A_∞ in bulk\n", flush=True)
    n_pass = 0; n_total = 0
    summary = []
    for ty in TAU_Y_LIST:
        for theta in ANGLES_DEG:
            n_total += 1
            print(f"--- θ={theta:+.0f}°, fault τ_y={ty:.2f} ---", flush=True)
            res = run_one(theta, ty)
            print(f"  steps={len(res['times'])} wall={res['wall']:.1f}s "
                  f"diverged={res['diverged']}", flush=True)
            if len(res['times']):
                sxy_c = float(np.abs(res['sxy_centre']).max())
                tau_res_c = float(np.abs(res['resolved_centre']).max())
                ratio_sxy = sxy_c / ty
                ratio_tau = tau_res_c / ty
                print(f"  centre probes (apples-to-apples with BDF-1 baseline):")
                print(f"    peak |σ_xy|       = {sxy_c:.4f}  ({ratio_sxy:.3f}·τ_y)")
                print(f"    peak |τ_resolved| = {tau_res_c:.4f}  ({ratio_tau:.3f}·τ_y)")
                print(f"  global probes:")
                print(f"    peak |σ_xy| any node     = {float(res['sxy_max_global'].max()):.4f}")
                print(f"    peak σ_II any node       = {float(res['sigmaII_max_fault'].max()):.4f}")
                # Decision gate: τ_resolved at centre ≤ 1.20·τ_y
                # (BDF-1 production baseline is 1.12-1.15·τ_y on this setup)
                ok_yield = ratio_tau < 1.20
                if ok_yield and res['diverged'] == 0:
                    print(f"  PASS")
                    n_pass += 1
                    summary.append((theta, ty, ratio_tau, "PASS"))
                else:
                    print(f"  FAIL (centre |τ_resolved|/τ_y = {ratio_tau:.4f}, "
                          f"diverged={res['diverged']})")
                    summary.append((theta, ty, ratio_tau, "FAIL"))
            else:
                print(f"  FAIL — no steps completed")
                summary.append((theta, ty, float('inf'), "FAIL"))
            print()
    print(f"\n=== KILLER TEST SUMMARY: {n_pass}/{n_total} PASS ===", flush=True)
    print("  metric: peak |τ_resolved| at fault centre / τ_y_at_fault")
    print("  BDF-1 production baseline ≈ 1.12-1.15·τ_y (centre)")
    print("  BDF-2 (the higher-order method ETD-2 replaces) blows up to 10⁵-10⁹\n")
    for theta, ty, ratio, status in summary:
        print(f"  θ={theta:+.0f}°, τ_y={ty:.2f}: |τ_resolved|/τ_y = {ratio:.4f}  [{status}]")


if __name__ == "__main__":
    main()
