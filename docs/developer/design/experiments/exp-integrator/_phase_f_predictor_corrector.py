"""Phase F: ETD-2 VE predictor + J2 radial-return corrector (isotropic).

Implements the predictor-corrector architecture from the web advice
(``vep_stress_update_full_latex.md`` §15 Stage 4) on the isotropic VEP
case. The TI extension is the end goal but isotropic is a cleaner first
test of the architecture.

Per-step structure:
  1. Stokes solve with constitutive_model = isotropic Maxwell ETD-2
     (yield_stress = ∞, so no in-residual yield clipping).
  2. Read psi_star (= unclipped VE trial stress).
  3. J2 radial return at each quadrature node: if |σ|_eq > σ_y, scale
     σ ← (σ_y/|σ|_eq)·σ. Overwrite psi_star.
  4. The corrected psi_star becomes σⁿ for the next timestep's
     history term (α·σⁿ in the ETD-2 update).

This is "predictor-corrector without outer Picard" — single-shot
correction per timestep. If stability fails, add an outer Picard
loop with stress damping (advice §10, ω_τ ≈ 0.5).

Comparison: same harmonic shear box geometry as the existing killer
test but isotropic (no fault, no director, uniform σ_y). Compare
trajectory against BDF-1 and ETD-1 baselines run in this script
in the same setup.

Per-step diagnostics every 5 steps; runaway guard.

Run::

    pixi run -e amr-dev python -u docs/developer/design/experiments/exp-integrator/_phase_f_predictor_corrector.py
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
ETA = 1.0; MU = 1.0
# Spatially-varying σ_y(x): small in the fault influence zone, large in
# the bulk. Isotropic von Mises elsewhere — no director, no rank-4
# projector, just the same fault influence-function used to localise
# yielding. Constant σ_y everywhere would fail everywhere (the correct
# solution); the localised weak zone gives partial yielding.
TAU_Y_FAULT = 0.05      # yield stress in the weak zone
TAU_Y_BULK  = 200.0     # effectively no yield outside
THETA_DEG = 15.0        # weak-zone tilt (no director use; just the geometry)
RES = 32

OUT_DIR = "output"


def _build_isotropic_stokes(label, integrator, order, yield_stress_value):
    """Common Stokes + isotropic VEP setup with spatial yield-stress
    field via the fault influence function (isotropic von Mises;
    director NOT used).

    yield_stress_value: pass the spatial sympy expression to apply the
    localised yield zone, or sympy.oo to disable yielding entirely
    (predictor-corrector path applies J2 return mapping externally).
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    # Build the localised weak zone using the fault influence-function
    # (same geometry as the killer test). σ_y(x) small near the layer
    # axis, large in the bulk. We're NOT using the director — this is
    # just a spatial yield-stress field that happens to be drawn from
    # the fault helper.
    cx, cy = 0.5 * W, 0.5 * H
    theta = np.radians(THETA_DEG)
    dx_layer = 0.5 * FAULT_LENGTH * np.cos(theta)
    dy_layer = 0.5 * FAULT_LENGTH * np.sin(theta)
    fault = uw.meshing.Surface(
        f"layer_{label}", mesh,
        np.array([[cx - dx_layer, cy - dy_layer],
                  [cx + dx_layer, cy + dy_layer]]),
        symbol=f"L{label}",
    )
    fault.discretize()
    weakness = fault.influence_function(
        width=FAULT_WIDTH,
        value_near=1.0 / TAU_Y_FAULT,
        value_far=1.0 / TAU_Y_BULK,
        profile="gaussian",
    )
    tau_y_field = 1.0 / weakness

    u = uw.discretisation.MeshVariable(f"U_{label}", mesh, 2, degree=2,
                                        vtype=VarType.VECTOR)
    p_sol = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1,
                                            continuous=True, vtype=VarType.SCALAR)

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p_sol)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, integrator=integrator, order=order,
    )
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    # Use the spatial yield-stress field by default; sympy.oo overrides
    # for the no-yield-in-residual predictor-corrector path.
    cm.Parameters.yield_stress = (
        yield_stress_value if yield_stress_value is not None else tau_y_field
    )
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
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

    return dict(mesh=mesh, stokes=stokes, u=u, p=p_sol, V_top=V_top, cm=cm)


def _j2_radial_return(sigma_arr, sigma_y):
    """Apply J2 radial return at each quadrature node.

    sigma_arr: shape (n_nodes, dim, dim) symmetric tensor field.
    Returns the corrected array (in-place not allowed — return new copy
    so the caller can decide whether to overwrite).
    """
    # Equivalent stress: σ_eq = sqrt(3/2 · σ:σ) for deviatoric σ.
    sig_dot_sig = (sigma_arr * sigma_arr).sum(axis=(1, 2))  # σ_ij σ_ij
    sigma_eq = np.sqrt(1.5 * sig_dot_sig)
    # Avoid divide-by-zero
    safe_eq = np.where(sigma_eq > 1.0e-12, sigma_eq, 1.0)
    scale = np.where(sigma_eq > sigma_y, sigma_y / safe_eq, 1.0)
    return sigma_arr * scale[:, None, None]


def run_case(label, integrator, order, n_periods=1.5,
             apply_radial_return=False, in_residual_yield=True,
             max_picard_iters=1, picard_tol=1.0e-3, omega_tau=0.5):
    """Generic runner.

    integrator/order: passed to constitutive model.
    apply_radial_return: if True, apply J2 radial return post-solve.
    in_residual_yield: if False, set yield_stress=∞ in the model so it
                       doesn't clip in-residual; only post-solve return
                       maps. If True, use the spatial yield_stress
                       field (small near layer, large in bulk).
    max_picard_iters: outer Picard within a timestep. With value 1 the
                       Stokes solve runs once (no equilibration after
                       correction). With value > 1 we save σ_n at step
                       start, run Stokes, apply correction, damp σ, and
                       re-solve with the corrected psi_star until σ
                       converges.
    omega_tau: stress damping coefficient inside the Picard loop
               (advice §10; ω_τ ~ 0.5 is the standard value).
    """
    # None → use the spatial weak-zone field built inside _build_..; oo
    # → disable in-residual yielding (predictor-corrector path).
    yield_stress_value = None if in_residual_yield else sympy.oo
    obj = _build_isotropic_stokes(label, integrator, order, yield_stress_value)
    mesh = obj["mesh"]; stokes = obj["stokes"]
    u = obj["u"]; V_top = obj["V_top"]; cm = obj["cm"]
    DFDt = stokes.Unknowns.DFDt

    # Per-step trace file — updated each step (flush every line). Lets a
    # killed run still leave usable data, and lets the plot script parse
    # it without rerunning. Pattern from feedback_per_step_logging.md.
    trace_path = os.path.join(
        os.path.dirname(__file__), f"_phase_f_{label}.trace.txt"
    )
    trace_fh = open(trace_path, "w")
    trace_fh.write(
        f"# Phase F predictor-corrector trace: {label}\n"
        f"# integrator={integrator!r} order={order} "
        f"apply_radial_return={apply_radial_return} "
        f"in_residual_yield={in_residual_yield}\n"
        f"# columns: step, t, V_top, snes_iters_total, picard_iters, "
        f"sigma_eq_max, sigma_eq_max_after_correction, u_y_max, yielded_fraction\n"
    )
    trace_fh.flush()

    # Evaluate spatial σ_y(x) at psi_star coords ONCE — the yield stress
    # field doesn't change in time (just space). Used by the radial
    # return corrector AND by the yielded-fraction diagnostic.
    sigma_coords = DFDt.psi_star[0].coords
    cx, cy = 0.5 * W, 0.5 * H
    theta = np.radians(THETA_DEG)
    n_x_l = -np.sin(theta); n_y_l = np.cos(theta)
    # Re-derive the same Gaussian as in _build (signed-distance to the
    # layer axis). Avoids needing to evaluate the cm's yield_stress
    # symbolically for every node — which can be expensive.
    sd = np.abs((sigma_coords[:, 0] - cx) * n_x_l
                + (sigma_coords[:, 1] - cy) * n_y_l)
    half_length = 0.5 * FAULT_LENGTH
    along = (sigma_coords[:, 0] - cx) * n_y_l - (sigma_coords[:, 1] - cy) * n_x_l
    in_extent = np.abs(along) <= half_length
    # The fault influence_function uses a Gaussian normal to the layer
    # axis, restricted to the layer extent. value_near at sd=0 (and
    # within extent), value_far at large sd (or outside extent).
    weakness_arr = np.where(
        in_extent,
        (1.0 / TAU_Y_FAULT) * np.exp(-(sd / FAULT_WIDTH) ** 2)
        + (1.0 / TAU_Y_BULK)
        * (1.0 - np.exp(-(sd / FAULT_WIDTH) ** 2)),
        1.0 / TAU_Y_BULK * np.ones_like(sd),
    )
    sigma_y_at_nodes = 1.0 / weakness_arr

    T_END = n_periods * 2.0 * np.pi / OMEGA
    iters = []; reasons = []
    sigma_eq_max_per_step = []
    sigma_eq_centre_per_step = []
    u_y_max_per_step = []
    yielded_fraction_per_step = []
    centre = np.array([[0.5 * W, 0.5 * H]])

    t_cur = 0.0
    t0 = time.time()
    picard_iters_per_step = []  # diagnostic: how many Picard iters used
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt

        # Save σ_n at start of step — needed if we Picard-iterate within
        # the step (psi_star has to read σ_n for E_eff history each
        # solve, but the post-solve correction overwrites it).
        sigma_n = np.asarray(DFDt.psi_star[0].array).copy()
        sigma_iter = sigma_n.copy()  # current best estimate of σ_{n+1}

        snes_iters_total = 0
        snes_reason_last = 0
        picard_k_used = max_picard_iters
        for picard_k in range(max_picard_iters):
            # Restore start-of-step state so model sees σ_n as history
            DFDt.psi_star[0].array[...] = sigma_n
            try:
                stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
            except Exception as exc:
                print(f"  step at t={t_end_step:.3f} picard={picard_k}: "
                      f"solve raised — {exc}", flush=True)
                snes_iters_total = -1
                snes_reason_last = -99
                picard_k_used = picard_k
                break
            snes_iters_total += int(stokes.snes.getIterationNumber())
            snes_reason_last = int(stokes.snes.getConvergedReason())

            sigma_trial = np.asarray(DFDt.psi_star[0].array).copy()
            if apply_radial_return:
                sigma_corrected = _j2_radial_return(sigma_trial, sigma_y_at_nodes)
            else:
                sigma_corrected = sigma_trial

            if max_picard_iters == 1:
                # Single-shot mode: just accept the correction
                sigma_iter = sigma_corrected
                picard_k_used = 1
                break

            # Damp the σ update for outer-Picard convergence (advice §10)
            sigma_new = (1.0 - omega_tau) * sigma_iter + omega_tau * sigma_corrected
            # Convergence check
            denom = max(np.linalg.norm(sigma_new), 1e-12)
            diff = np.linalg.norm(sigma_new - sigma_iter) / denom
            sigma_iter = sigma_new
            if diff < picard_tol:
                picard_k_used = picard_k + 1
                break

        # Final accepted state
        DFDt.psi_star[0].array[...] = sigma_iter
        picard_iters_per_step.append(picard_k_used)

        if snes_iters_total < 0:
            iters.append(-1); reasons.append(snes_reason_last)
            break
        iters.append(snes_iters_total)
        reasons.append(snes_reason_last)

        # Diagnostics on the FINAL accepted σ
        sig_dot_sig = (sigma_iter * sigma_iter).sum(axis=(1, 2))
        sigma_eq = np.sqrt(1.5 * sig_dot_sig)
        sigma_eq_max_per_step.append(float(sigma_eq.max()))
        n_yielded = int((sigma_eq > sigma_y_at_nodes * 0.99).sum())
        yielded_fraction_per_step.append(n_yielded / sigma_eq.size)
        # σ_eq AFTER correction (same as final state since sigma_iter is corrected)
        sigma_eq_centre_per_step.append(float(sigma_eq.max()))

        u_arr = np.asarray(u.array).reshape(-1, 2)
        u_y_max_per_step.append(float(np.abs(u_arr[:, 1]).max()))

        step_idx = len(iters)
        # Persistent per-step trace — written EVERY step, flushed
        trace_fh.write(
            f"{step_idx:4d} {t_end_step:7.4f} {v_now:+.4f} "
            f"{iters[-1]:3d} {picard_k_used:2d} "
            f"{sigma_eq_max_per_step[-1]:.6e} "
            f"{sigma_eq_centre_per_step[-1]:.6e} "
            f"{u_y_max_per_step[-1]:.6e} "
            f"{yielded_fraction_per_step[-1]:.6f}\n"
        )
        trace_fh.flush()
        if step_idx <= 5 or step_idx % 5 == 0:
            picard_str = f" pic={picard_k_used:d}" if max_picard_iters > 1 else ""
            print(
                f"  step {step_idx:3d}/120  t={t_end_step:5.3f}  "
                f"V={v_now:+.3f}  iters={iters[-1]:2d}{picard_str}  "
                f"|σ|_eq_max={sigma_eq_max_per_step[-1]:.3e}  "
                f"|u_y|={u_y_max_per_step[-1]:.3e}  "
                f"yielded={yielded_fraction_per_step[-1]:.2%}",
                flush=True,
            )

        if sigma_eq_max_per_step[-1] > 100.0 or u_y_max_per_step[-1] > 10.0:
            print(f"  *** runaway at step {step_idx} — breaking ***", flush=True)
            break

        t_cur = t_end_step

    iters_arr = np.array(iters)
    reasons_arr = np.array(reasons)
    print(
        f"  ran {len(iters)} steps in {time.time()-t0:.1f}s; "
        f"{label} (integrator={integrator}, order={order}, "
        f"radial_return={apply_radial_return}, in_residual_yield={in_residual_yield})",
        flush=True,
    )
    if iters_arr.size > 0 and (iters_arr >= 0).any():
        print(
            f"  SNES iters mean={iters_arr[iters_arr>=0].mean():.1f} "
            f"max={iters_arr[iters_arr>=0].max()} "
            f"diverged={int((reasons_arr<0).sum())}/{len(reasons_arr)}",
            flush=True,
        )
    if sigma_eq_max_per_step:
        print(
            f"  σ_eq_max: end={sigma_eq_max_per_step[-1]:.4f}  "
            f"global max={max(sigma_eq_max_per_step):.4f}  "
            f"({max(sigma_eq_max_per_step)/TAU_Y_FAULT:.2f}·τ_y_fault)",
            flush=True,
        )
        print(
            f"  |u_y|_max: end={u_y_max_per_step[-1]:.4f}  "
            f"global max={max(u_y_max_per_step):.4f}",
            flush=True,
        )
        print(
            f"  yielded fraction: end={yielded_fraction_per_step[-1]:.2%}  "
            f"max={max(yielded_fraction_per_step):.2%}",
            flush=True,
        )

    out_npz = os.path.join(OUT_DIR, f"phase_f_{label}.npz")
    np.savez(
        out_npz,
        iters=iters_arr, reasons=reasons_arr,
        sigma_eq_max_per_step=np.asarray(sigma_eq_max_per_step),
        sigma_eq_centre_per_step=np.asarray(sigma_eq_centre_per_step),
        u_y_max_per_step=np.asarray(u_y_max_per_step),
        yielded_fraction_per_step=np.asarray(yielded_fraction_per_step),
        T_END=np.array(T_END),
        n_steps=np.array(len(iters)),
        wall_seconds=np.array(time.time() - t0),
    )
    print(f"  saved → {out_npz}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Each tuple: (label, integrator, order, apply_pc, max_picard_iters)
    # apply_pc=True → yield_stress=∞ in model, radial-return correction
    # max_picard_iters=1 → single shot; >1 → outer Picard equilibrate
    cases = [
        # Baseline: BDF-1 yield-in-residual via softmin (works)
        ("bdf1_iso",           "bdf", 1, False, 1),
        # NOTE: ETD-1 / ETD-2 with yield_stress=spatial-field +
        # yield_mode=softmin in the parent ViscoElasticPlasticFlowModel
        # currently produces SNES line-search divergence (separate bug,
        # filed for follow-up). Skipping those baselines here — the
        # predictor-corrector path below sets yield_stress=∞ in the
        # model so it never enters that broken in-residual code path.
        # Predictor-corrector: single shot
        ("etd2_pc1",           "etd", 2, True,  1),
        ("etd1_pc1",           "etd", 1, True,  1),
        # Predictor-corrector with outer Picard equilibration (ω_τ=0.5)
        ("etd2_pc_picard",     "etd", 2, True,  6),
        ("etd1_pc_picard",     "etd", 1, True,  6),
    ]
    for label, integrator, order, apply_pc, max_picard in cases:
        cache = os.path.join(OUT_DIR, f"phase_f_{label}.npz")
        if os.path.exists(cache):
            print(f"\n=== {label}: cache hit, skipping ===", flush=True)
            continue
        if apply_pc:
            mode = ("predictor-corrector single-shot"
                    if max_picard == 1
                    else f"predictor-corrector + outer Picard ({max_picard} iters)")
            print(
                f"\n=== {label}: integrator={integrator!r}, order={order}, "
                f"{mode} ===",
                flush=True,
            )
            run_case(label, integrator, order, apply_radial_return=True,
                     in_residual_yield=False, max_picard_iters=max_picard)
        else:
            print(
                f"\n=== {label}: integrator={integrator!r}, order={order}, "
                f"yield-in-residual (softmin in model) ===",
                flush=True,
            )
            run_case(label, integrator, order, apply_radial_return=False,
                     in_residual_yield=True, max_picard_iters=1)


if __name__ == "__main__":
    main()
