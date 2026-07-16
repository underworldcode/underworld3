"""Phase B intermediate test: bench_ti_vep_harmonic geometry with the ISOTROPIC
MaxwellExponentialFlowModel + spatial yield_stress field.

This is NOT the killer test as designed (which uses TransverseIsotropic
rank-4 tensor) — it's a structural sanity check before investing in the
TI extension. Goal: confirm that the predictor-corrector return mapping
on the spatial yield_stress field stays bounded, i.e. σ_II ≤ 1.001·τ_y
everywhere. If yes: the exp framework's structural argument extends to
spatial yield, and TI extension is tensor-bookkeeping. If no: the spatial
yield handling itself needs more work.

Note: this uses the ``zIC`` (zero IC) variant of the bench, since the
peak-start TI IC requires resolving stress onto the fault tangent —
which is TI-specific. Zero IC + smooth ramp-up is a cleaner test.

Run::

    pixi run -e amr-dev python -u docs/developer/design/experiments/exp-integrator/_exp_integrator_phase_b_ti_iso.py
"""

import time
import numpy as np
import sympy

import underworld3 as uw
from underworld3.function import expression


# ---------------------------------------------------------------------------
# Parameters (kept aligned with bench_ti_vep_harmonic.py)
# ---------------------------------------------------------------------------

V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * np.pi / OMEGA

ETA = 1.0          # use single isotropic viscosity (η₀ in TI nomenclature)
MU = 1.0
TAU_Y_BULK = 200.0

RES = 16
H = 1.0; W = 1.0
FAULT_LENGTH = 0.6
FAULT_WIDTH = 0.06

ANGLES_DEG = (0.0,)        # start with 0° to keep the iso comparison simple
TAU_Y_LIST = (0.30, 0.15)


def build_iso_exp_stokes(label, theta_deg, tau_y_at_fault):
    """Plain Stokes + MaxwellExponentialFlowModel + spatial yield_stress field."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1, continuous=True)

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

    weakness = fault.influence_function(
        width=FAULT_WIDTH,
        value_near=1.0 / tau_y_at_fault,
        value_far=1.0 / TAU_Y_BULK,
        profile="gaussian",
    )
    tau_y_field = 1.0 / weakness

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.MaxwellExponentialFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = tau_y_field
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    cm._yield_mode = "softmin"

    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "Top V")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])

    return mesh, stokes, V_top, tau_y_field


def run_iso_zIC(theta_deg, tau_y_at_fault):
    label = f"ti_iso_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p")
    mesh, stokes, V_top, ty_field = build_iso_exp_stokes(label, theta_deg, tau_y_at_fault)
    cm = stokes.constitutive_model
    DFDt = stokes.Unknowns.DFDt

    times, peak_sxy_global, peak_sigmaII_fault = [], [], []
    n_diverged = 0
    t0 = time.time()
    t_cur = 0.0
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end = t_cur + dt
        # Forcing: V_top(t) = V0·cos(ωt + φ_lag) — same as bench
        v_now = V0 * float(np.cos(OMEGA * t_end))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        try:
            stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        except Exception as exc:
            print(f"  step at t={t_end:.3f}: {exc}", flush=True)
            n_diverged += 1
            break
        # Probe
        sigma = np.asarray(DFDt.psi_star[0].array)
        sxy_global = float(np.abs(sigma[:, 0, 1]).max())
        sigma_II = np.sqrt(0.5 * (sigma ** 2).sum(axis=(1, 2)))
        coords = DFDt.psi_star[0].coords
        # fault-zone mask (within ~3·FAULT_WIDTH of the centerline)
        dist = np.abs(coords[:, 1] - 0.5 * H)  # 2D, theta=0 simplification
        mask = dist < 3.0 * FAULT_WIDTH
        sigmaII_fault = float(sigma_II[mask].max()) if mask.any() else 0.0
        peak_sxy_global.append(sxy_global)
        peak_sigmaII_fault.append(sigmaII_fault)
        times.append(t_end)
        t_cur = t_end
    return dict(
        times=np.array(times),
        peak_sxy_global=np.array(peak_sxy_global),
        peak_sigmaII_fault=np.array(peak_sigmaII_fault),
        wall=time.time() - t0,
        n_diverged=n_diverged,
        tau_y=tau_y_at_fault,
    )


def main():
    print(f"[ti_iso_zIC] dt={DT} T_end={T_END:.4f} (4 periods)", flush=True)
    print(f"  bulk τ_y={TAU_Y_BULK}, fault τ_y values: {TAU_Y_LIST}\n", flush=True)
    for ty in TAU_Y_LIST:
        for theta in ANGLES_DEG:
            print(f"--- θ={theta:+.0f}°, fault τ_y={ty:.2f} ---", flush=True)
            res = run_iso_zIC(theta, ty)
            print(f"  steps={len(res['times'])} wall={res['wall']:.1f}s "
                  f"diverged={res['n_diverged']}", flush=True)
            sxy = res['peak_sxy_global']
            sii_fault = res['peak_sigmaII_fault']
            if len(sxy):
                print(f"  peak|σ_xy| (global): {sxy.max():.4f}", flush=True)
                print(f"  peak σ_II (fault):   {sii_fault.max():.4f}", flush=True)
                print(f"  ratio σ_II_fault/τ_y: {sii_fault.max()/ty:.3f}", flush=True)
                if sii_fault.max() < 1.1 * ty:
                    print(f"  PASS (σ_II_fault ≤ 1.1·τ_y)", flush=True)
                else:
                    print(f"  FAIL (σ_II_fault = {sii_fault.max():.4f} > 1.1·τ_y = {1.1*ty:.4f})", flush=True)
            print()


if __name__ == "__main__":
    main()
