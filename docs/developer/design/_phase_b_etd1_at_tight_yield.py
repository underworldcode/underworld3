"""ETD-1 (first-order) trajectory at τ_y=0.05, θ=+15°.

Hypothesis (after the lesson from BDF-2/ETD-2 lumped/split/hybrid):
all higher-order time integrators show some flavour of growing
instability on this tight-yield TI fault problem; only first-order
BDF stays stable. The cause is L-stability / numerical dissipation,
not algorithm specifics. ETD-1 is the first-order ETD analogue:

    σ^{n+1} = α·σ^n + 2η(1-α)·ε̇^{n+1},   α = exp(-Δt/τ)

Single step, no φ, no ε̇* history. Implemented as ETD-2 with
φ = α (set in _update_history_coefficients), which zeroes the
(φ-α)·ε̇* term and turns (1-φ)·ε̇ into (1-α)·ε̇.

Per-step logging every 5 steps + runaway guard, per the project
memory on per-step diagnostics.

Run::

    pixi run -e amr-dev python -u docs/developer/design/_phase_b_etd1_at_tight_yield.py
"""

import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression


V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
H = 1.0; W = 1.0
FAULT_LENGTH = 0.6
FAULT_WIDTH = 0.06
ETA_0 = 1.0; ETA_1 = 1.0; MU = 1.0
TAU_Y_BULK = 200.0
RES = 32

OUT_DIR = "output"


def run_etd1(theta_deg, tau_y_at_fault, n_periods=1.5):
    label = f"etd1_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p")

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    u = uw.discretisation.MeshVariable(f"U_{label}", mesh, 2, degree=2,
                                        vtype=VarType.VECTOR)
    p_sol = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1,
                                            continuous=True, vtype=VarType.SCALAR)

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

    n_x = -np.sin(theta); n_y = np.cos(theta)
    director = sympy.Matrix([n_x, n_y])
    weakness = fault.influence_function(
        width=FAULT_WIDTH,
        value_near=1.0 / tau_y_at_fault, value_far=1.0 / TAU_Y_BULK,
        profile="gaussian",
    )
    tau_y_field = 1.0 / weakness

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p_sol)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, integrator="etd1",
    )
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

    DFDt = stokes.Unknowns.DFDt

    T_END = n_periods * 2.0 * np.pi / OMEGA
    iters = []; reasons = []
    sigma_II_max_per_step = []
    u_y_max_per_step = []
    sigma_xy_centre = []
    sigma_par_centre = []
    centre = np.array([[cx, cy]])
    n_x_val = -float(np.sin(theta))
    n_y_val = float(np.cos(theta))

    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        try:
            stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        except Exception as exc:
            print(f"  step at t={t_end_step:.3f}: solve raised — {exc}", flush=True)
            iters.append(-1)
            reasons.append(-99)
            break

        iters.append(int(stokes.snes.getIterationNumber()))
        reasons.append(int(stokes.snes.getConvergedReason()))

        sigma_arr = np.asarray(DFDt.psi_star[0].array)
        sigma_II = np.sqrt(0.5 * (sigma_arr ** 2).sum(axis=(1, 2)))
        sigma_II_max_per_step.append(float(sigma_II.max()))
        u_arr = np.asarray(u.array).reshape(-1, 2)
        u_y_max_per_step.append(float(np.abs(u_arr[:, 1]).max()))
        sxy_centre = float(uw.function.evaluate(stokes.tau.sym[0, 1], centre).flatten()[0])
        sigma_xy_centre.append(sxy_centre)
        sxx_c = float(uw.function.evaluate(stokes.tau.sym[0, 0], centre).flatten()[0])
        syy_c = float(uw.function.evaluate(stokes.tau.sym[1, 1], centre).flatten()[0])
        T_x = sxx_c * n_x_val + sxy_centre * n_y_val
        T_y = sxy_centre * n_x_val + syy_c * n_y_val
        sig_nn = T_x * n_x_val + T_y * n_y_val
        sig_par = float(np.sqrt(max(T_x ** 2 + T_y ** 2 - sig_nn ** 2, 0.0)))
        sigma_par_centre.append(sig_par)

        # Per-step logging
        step_idx = len(iters)
        if step_idx <= 5 or step_idx % 5 == 0:
            print(
                f"  step {step_idx:3d}/120  t={t_end_step:5.3f}  "
                f"V={v_now:+.3f}  iters={iters[-1]:2d}  "
                f"|σ|_II={sigma_II_max_per_step[-1]:.3e}  "
                f"|u_y|={u_y_max_per_step[-1]:.3e}  "
                f"|σ_∥|={sig_par:.3e}",
                flush=True,
            )

        # Runaway guard
        if sigma_II_max_per_step[-1] > 100.0 or u_y_max_per_step[-1] > 10.0:
            print(
                f"  *** runaway detected at step {step_idx}: "
                f"|σ|_II={sigma_II_max_per_step[-1]:.3e}, "
                f"|u_y|={u_y_max_per_step[-1]:.3e} — breaking ***",
                flush=True,
            )
            break

        t_cur = t_end_step

    iters_arr = np.array(iters)
    reasons_arr = np.array(reasons)

    print(
        f"  ran {len(iters)} steps in {time.time()-t0:.1f}s; "
        f"ETD-1, τ_y_fault={tau_y_at_fault}",
        flush=True,
    )
    if iters_arr.size > 0 and (iters_arr >= 0).any():
        print(
            f"  SNES iters per step (etd1): mean={iters_arr[iters_arr>=0].mean():.1f} "
            f"median={int(np.median(iters_arr[iters_arr>=0]))} "
            f"max={iters_arr[iters_arr>=0].max()} "
            f"diverged={int((reasons_arr<0).sum())}/{len(reasons_arr)}",
            flush=True,
        )
    if sigma_II_max_per_step:
        print(
            f"  max |σ|_II per step: end={sigma_II_max_per_step[-1]:.4f}  "
            f"global max={max(sigma_II_max_per_step):.4f}",
            flush=True,
        )
        print(
            f"  max |u_y| per step:  end={u_y_max_per_step[-1]:.4f}  "
            f"global max={max(u_y_max_per_step):.4f}",
            flush=True,
        )
        print(
            f"  centre |σ_∥| (resolved): "
            f"end={sigma_par_centre[-1]:.4f}  "
            f"peak={max(sigma_par_centre):.4f}  "
            f"({max(sigma_par_centre)/tau_y_at_fault:.2f}·τ_y_fault)",
            flush=True,
        )

    out_npz = os.path.join(
        OUT_DIR,
        f"phase_b_etd1_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p") + ".npz",
    )
    np.savez(
        out_npz,
        iters=iters_arr,
        reasons=reasons_arr,
        sigma_II_max_per_step=np.asarray(sigma_II_max_per_step),
        u_y_max_per_step=np.asarray(u_y_max_per_step),
        sigma_xy_centre=np.asarray(sigma_xy_centre),
        sigma_par_centre=np.asarray(sigma_par_centre),
        theta_deg=np.array(theta_deg),
        tau_y_at_fault=np.array(tau_y_at_fault),
        T_END=np.array(T_END),
        n_steps=np.array(len(iters)),
        wall_seconds=np.array(time.time() - t0),
    )
    print(f"  saved → {out_npz}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = os.path.join(OUT_DIR, "phase_b_etd1_th+15_ty0p05.npz")
    if os.path.exists(cache):
        print(f"=== ETD-1 cache hit: {cache} — skipping run ===", flush=True)
        return
    print("=== ETD-1: θ=+15°, τ_y=0.05 ===", flush=True)
    run_etd1(15.0, 0.05, n_periods=1.5)


if __name__ == "__main__":
    main()
