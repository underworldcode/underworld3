"""Variable-dt VEP via scheduled BDF-1 restart at the dt-change boundary.

Hypothesis: a single BDF-1 step taken at the new dt creates uniform-spaced
psi_star history for subsequent BDF-2 steps:

    halving (dt -> dt/2):  psi_star = [sigma(t_n + dt/2), sigma(t_n)]  -> -dt/2 spacing
    doubling (dt -> 2*dt): psi_star = [sigma(t_n + 2dt), sigma(t_n)]   -> -2*dt spacing

Both yield uniform spacing for the next BDF-2 step.

This experiment uses an oracle-driven dt schedule (knowledge of when BC
flips happen) to halve dt around each transition and double back on the
plateau. We run four configurations on each of pure-VE and VEP-min:

  1. fixed dt = dt_max                    -- coarse reference
  2. fixed dt = dt_min                    -- fine reference
  3. scheduled halve/double, ONE-SIDED    -- old (pre-fix) behaviour
  4. scheduled halve/double, SYMMETRIC    -- with the BDF-1 restart fix

Configurations 3 and 4 differ only in `ViscoElasticPlasticFlowModel.
_update_bdf_coefficients`: with the fix, the threshold is symmetric so
any |ratio - 1| > log2(threshold) triggers BDF-1 restart, whereas the
old code only triggered on ratio > threshold (i.e. doubling but not
halving).

To run config 3 we monkey-patch the method back to its one-sided form
on the active model instance.

Outputs: per-config npz under output/scheduled_dt/, plus a 2-panel
comparison figure docs/advanced/figures/scheduled_dt_handoff.png.
"""

import os
import time
import types
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression
from underworld3.systems.ddt import _bdf_coefficients


# ---------------------------------------------------------------- analytical

def step_square_wave_stress(t, eta, mu, gamma_dot_0, half_period):
    """Closed-form Maxwell stress under step-change V_top: piecewise
    relaxation toward eta * gamma_dot_0 with sign flips every half_period."""
    t_r = eta / mu
    sigma_ss = eta * gamma_dot_0
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        n = int(ti / half_period)
        t_local = ti - n * half_period
        sigma_start = 0.0
        for j in range(n):
            sign = 1.0 if j % 2 == 0 else -1.0
            sigma_target = sign * sigma_ss
            sigma_start = sigma_target + (sigma_start - sigma_target) * np.exp(-half_period / t_r)
        sign = 1.0 if n % 2 == 0 else -1.0
        sigma_target = sign * sigma_ss
        out[i] = sigma_target + (sigma_start - sigma_target) * np.exp(-t_local / t_r)
    return out


# --------------------------- experimental BDF-1 restart on every dt-change

def _update_bdf_coefficients_symmetric_restart(self):
    """Experimental: trigger BDF-1 fallback on dt change in either direction
    (i.e. ratio >= threshold OR ratio <= 1/threshold). The hypothesis is that
    a single BDF-1 step at the new dt creates uniform-spaced psi_star history
    for subsequent BDF-2 steps, rescuing variable-dt accuracy.
    """
    order = self.effective_order
    if self.Unknowns is not None and self.Unknowns.DFDt is not None:
        dt_current = self.Parameters.dt_elastic
        if hasattr(dt_current, 'sym'):
            dt_current = dt_current.sym
        dt_history = self.Unknowns.DFDt._dt_history
        if order >= 2 and len(dt_history) > 0 and dt_history[0] is not None:
            try:
                ratio = float(dt_current) / float(dt_history[0])
                threshold = self._max_dt_ratio_for_higher_order
                if ratio >= threshold or ratio <= 1.0 / threshold:
                    order = 1
            except (TypeError, ZeroDivisionError):
                pass
        coeffs = _bdf_coefficients(order, dt_current, dt_history)
        alpha = self.bdf_blend
        if 0 < alpha < 1 and order >= 2:
            coeffs_o1 = _bdf_coefficients(1, dt_current, dt_history)
            while len(coeffs_o1) < len(coeffs):
                coeffs_o1.append(sympy.Integer(0))
            coeffs = [
                (1 - alpha) * c1 + alpha * ck
                for c1, ck in zip(coeffs_o1, coeffs)
            ]
    else:
        coeffs = _bdf_coefficients(order, None, [])
    while len(coeffs) < 4:
        coeffs.append(sympy.Integer(0))
    self._bdf_c0.sym = coeffs[0]
    self._bdf_c1.sym = coeffs[1]
    self._bdf_c2.sym = coeffs[2]
    self._bdf_c3.sym = coeffs[3]


# ---------------------------------------------------------------- dt schedule

def schedule_dt(t_cur, dt_max, dt_min, flip_times, window):
    """Halve dt within `window` of any scheduled BC flip; otherwise dt_max."""
    for f in flip_times:
        if abs(t_cur - f) <= window or (0 <= f - t_cur <= window):
            return dt_min
    return dt_max


# ---------------------------------------------------------------- single run

def run(label, schedule_kind, fix_kind, yield_mode, save_path):
    """schedule_kind: 'fixed_max', 'fixed_min', 'scheduled'
       fix_kind: 'current'    -- main code unchanged (strict ``ratio > threshold``)
                 'experiment' -- monkey-patched symmetric BDF-1 restart on every
                                 dt change (ratio >= threshold OR ratio <= 1/threshold)
       yield_mode: 'VE' (no yielding) or 'min' (sharp Min yielding)"""
    ETA, MU, H, W = 1.0, 1.0, 1.0, 2.0
    V0 = 0.5
    tau_y_val = 0.5 if yield_mode == "min" else 1.0e6  # effectively infinite
    t_r = ETA / MU
    half_period = 2.0 * t_r
    t_end = 4 * half_period  # 8 t_r
    dt_max = 0.20 * t_r
    dt_min = 0.10 * t_r
    window = 0.40 * t_r  # +/- 2 small steps around each flip

    flip_times = [k * half_period for k in range(1, int(t_end / half_period))]

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W/2, -H/2), maxCoords=(W/2, H/2),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=2)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.yield_stress = tau_y_val
    stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6
    stokes.constitutive_model._yield_mode = "min"

    # Apply experimental symmetric BDF-1 restart on this instance only
    if fix_kind == "experiment":
        stokes.constitutive_model._update_bdf_coefficients = types.MethodType(
            _update_bdf_coefficients_symmetric_restart, stokes.constitutive_model
        )

    V_top = expression(R"V_{top}", sympy.Float(V0), "Top V")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True

    centre = np.array([[0.0, 0.0]])
    times, taus, dts, reasons, orders = [], [], [], [], []
    t_cur = 0.0
    step = 0
    t0 = time.time()
    while t_cur < t_end - 1e-9 and step < 3000:
        if schedule_kind == "fixed_max":
            dt = dt_max
        elif schedule_kind == "fixed_min":
            dt = dt_min
        else:  # scheduled
            dt = schedule_dt(t_cur, dt_max, dt_min, flip_times, window)
        # Don't step past t_end
        dt = min(dt, t_end - t_cur)

        n_half = int((t_cur + 0.5 * dt) / half_period)
        V_sign = 1.0 if (n_half % 2 == 0) else -1.0
        V_top.sym = sympy.Float(V_sign * V0)
        stokes.constitutive_model.Parameters.dt_elastic = dt

        # Record what BDF order will be used for this step (post-update_bdf)
        # (we re-derive it from the same logic here for diagnostics)
        dt_history = stokes.DFDt._dt_history
        used_order = stokes.constitutive_model.effective_order
        threshold = stokes.constitutive_model._max_dt_ratio_for_higher_order
        if used_order >= 2 and len(dt_history) > 0 and dt_history[0] is not None:
            try:
                ratio = float(dt) / float(dt_history[0])
                if fix_kind == "experiment":
                    if ratio >= threshold or ratio <= 1.0 / threshold:
                        used_order = 1
                else:  # current code: strict > threshold
                    if ratio > threshold:
                        used_order = 1
            except (TypeError, ZeroDivisionError):
                pass

        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=1)
        s = float(uw.function.evaluate(stokes.tau.sym[0, 1], centre).flatten()[0])
        reason = int(stokes.snes.getConvergedReason())

        t_cur += dt
        step += 1
        times.append(t_cur)
        taus.append(s)
        dts.append(dt)
        reasons.append(reason)
        orders.append(used_order)

    t_wall = time.time() - t0
    times = np.array(times)
    taus = np.array(taus)
    dts = np.array(dts)
    reasons = np.array(reasons)
    orders = np.array(orders)
    peak = float(np.abs(taus).max())
    over = int((np.abs(taus) > 1.05 * (tau_y_val if yield_mode == "min" else 1e9)).sum())
    n_o1 = int((orders == 1).sum())
    n_o2 = int((orders >= 2).sum())

    if yield_mode == "VE":
        sigma_ana = step_square_wave_stress(times, ETA, MU, 2.0 * V0 / H, half_period)
        max_err = float(np.max(np.abs(taus - sigma_ana)))
        rms_err = float(np.sqrt(np.mean((taus - sigma_ana) ** 2)))
        msg_extra = f"max|err|={max_err:.4f}  rms|err|={rms_err:.4f}"
    else:
        max_err = rms_err = float("nan")
        msg_extra = f"peak|sigma|={peak:.4f}  overshoots>1.05*tau_y: {over}"

    print(f"  [{label:30s}] {len(taus):3d}st  wall={t_wall:5.1f}s  "
          f"BDF1={n_o1:3d} BDF2={n_o2:3d}  "
          f"dt in [{dts.min():.3f},{dts.max():.3f}]  {msg_extra}",
          flush=True)

    np.savez(save_path,
             times=times, stress=taus, dts=dts, reasons=reasons, orders=orders,
             peak=peak, overshoots=over, max_err=max_err, rms_err=rms_err,
             label=label, schedule_kind=schedule_kind, fix_kind=fix_kind,
             yield_mode=yield_mode)
    del stokes, mesh


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    out_dir = "output/scheduled_dt"
    os.makedirs(out_dir, exist_ok=True)

    print("=== Pure VE (yield_stress -> infinity) ===", flush=True)
    run("VE  fixed dt_max",        "fixed_max",  "current",    "VE",
        f"{out_dir}/ve_fixed_max.npz")
    run("VE  fixed dt_min",        "fixed_min",  "current",    "VE",
        f"{out_dir}/ve_fixed_min.npz")
    run("VE  scheduled, current",  "scheduled",  "current",    "VE",
        f"{out_dir}/ve_sched_current.npz")
    run("VE  scheduled, BDF1-restart", "scheduled", "experiment", "VE",
        f"{out_dir}/ve_sched_experiment.npz")

    print("=== VEP min (tau_y = 0.5) ===", flush=True)
    run("VEP fixed dt_max",        "fixed_max",  "current",    "min",
        f"{out_dir}/vep_fixed_max.npz")
    run("VEP fixed dt_min",        "fixed_min",  "current",    "min",
        f"{out_dir}/vep_fixed_min.npz")
    run("VEP scheduled, current",  "scheduled",  "current",    "min",
        f"{out_dir}/vep_sched_current.npz")
    run("VEP scheduled, BDF1-restart", "scheduled", "experiment", "min",
        f"{out_dir}/vep_sched_experiment.npz")
