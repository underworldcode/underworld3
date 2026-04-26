"""Reproduce the Min-BDF-1 NaN-on-plateau divergence with SNES monitoring."""
import numpy as np
import sympy
from _bench_helpers import DEFAULT_PARAMS, build_stokes, probe_centre
V0 = 0.5
TAU_Y = 0.5
HALF_PERIOD = 2.0
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * HALF_PERIOD
DT_PLATEAU = 0.10
DT_FINE = 0.01
WINDOW = 0.2

def schedule_dt(t_cur):
    flip_times = [HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2 - 1)]
    for f in flip_times:
        if abs(t_cur - f) <= WINDOW:
            return DT_FINE
    return DT_PLATEAU

params = dict(DEFAULT_PARAMS)
params["bdf_order"] = 1
mesh, stokes, V_top, params = build_stokes(
    "minfail_o1", params, yield_stress=TAU_Y, yield_mode="min",
)

# enable SNES monitor — prints |F| at every SNES iteration
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_converged_reason"] = None

t_cur = 0.0
step_idx = 0
target_steps = {348, 404, 405, 406}
# also peek at adjacent steps for context
context_steps = target_steps | {347, 403}

while t_cur < T_END - 1e-9:
    dt = schedule_dt(t_cur)
    flip_next = next((HALF_PERIOD * (k + 1) for k in range(N_PERIODS * 2)
                      if HALF_PERIOD * (k + 1) > t_cur + 1e-9), T_END)
    dt = min(dt, flip_next - t_cur, T_END - t_cur)
    t_end_step = t_cur + dt
    n_half = int(t_end_step / HALF_PERIOD - 1e-9)
    sign = 1.0 if n_half % 2 == 0 else -1.0
    v_now = sign * V0
    V_top.sym = sympy.Float(v_now)
    stokes.constitutive_model.Parameters.dt_elastic = dt

    # Only enable verbose monitoring at the steps we care about
    verbose = (step_idx in context_steps)
    if verbose:
        print(f"\n===== step {step_idx} t={t_end_step:.3f} dt={dt:.4f} sign={sign:+.0f} =====", flush=True)

    if not verbose:
        # silence the monitors temporarily
        stokes.petsc_options.delValue("snes_monitor")
        stokes.petsc_options.delValue("snes_converged_reason")
    else:
        stokes.petsc_options["snes_monitor"] = None
        stokes.petsc_options["snes_converged_reason"] = None

    stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)

    if verbose:
        sigma = probe_centre(stokes)
        reason = int(stokes.snes.getConvergedReason())
        its = int(stokes.snes.getIterationNumber())
        print(f"  → sigma_xy={sigma:.6f}  SNES reason={reason}  iters={its}", flush=True)
    t_cur = t_end_step
    step_idx += 1
    if step_idx > max(target_steps) + 2:
        break

print("\n--- done ---", flush=True)
