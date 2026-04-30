"""Phase G v2: two-Stokes operator-split VEP — α-decoupled formulation.

User's clarifications (2026-04-29):

* Stage 2 is *the* point of truth: a single nonlinear Stokes solve
  whose converged (σ, ε̇) become the recorded history. The yield
  enforcement happens INSIDE the SNES Newton via a yield-aware effective
  viscosity (no external Picard, no numpy J2 radial return).
* The Maxwell relaxation coefficient α must NOT be coupled to the
  within-step Newton-iterating yield-aware viscosity. Instead, α is
  frozen at the start of each timestep from a persistent meshvar
  ``η_lagged`` carrying the prior step's converged η_active.
* The VE predictor (Stage 1) ALSO uses ``η_lagged`` — without it,
  previously-failed zones build up phantom η_VE·ε̇·Δt elastic stress
  that Stage 2 has to do equal-and-opposite work to clip. Lagging η
  forward avoids that cancellation drift.

Architecture:

  Persistent across timesteps:
      η_lagged(x)        — yield-aware viscosity from prior step
                          (init: η_eff_pure_VE on step 0)
      σ_history(x)       — stored stress history (rank-2 sym tensor)

  At start of each timestep:
      η_eff_lag(x) = η_lagged · μΔt / (η_lagged + μΔt)
      α_lagged(x)  = η_lagged / (η_lagged + μΔt)
      Both frozen for this step.

  Stage 1 (linear VE warm-start):
      Stokes with η = η_eff_lag(x), bodyforce = −∇·(α_lagged · σ_history)
      Linear (η_eff_lag is frozen). Output: v_VE warm start.

  Stage 2 (nonlinear VEP, point of truth):
      Stokes with η = softmin(η_eff_lag(x), σ_y/(2|ε̇|)),
      bodyforce = −∇·(α_lagged · σ_history)  (same).
      SNES Newton iterates the yield-aware viscosity to convergence.
      Output: v_pl, p_pl.

  End-of-step:
      σ_new   = α_lagged·σ_history + 2·η_active(converged)·ε̇(v_pl)
      σ_history    ← σ_new
      η_lagged     ← η_active(converged)

Step 0 has σ_history = 0 → relaxation body force is zero; the solve
reduces to a pure viscoplastic Stokes (yield-aware viscous, no elastic
history). User-approved as the natural bootstrap.

Test harness: isotropic VEP shear box, RES=32, localised weak zone
(τ_y_fault=0.05, τ_y_bulk=200, θ=15°), 1.5 forcing periods.

Per-step trace files (flushed); runaway guard.
"""

import math
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
TAU_Y_FAULT = 0.05
TAU_Y_BULK  = 200.0
THETA_DEG = 15.0
RES = 32

OUT_DIR = "output"

# Pure-VE BDF-1 effective viscosity (used for η_lagged init)
ETA_EFF_PURE_VE = ETA * MU * DT / (ETA + MU * DT)


def _build_setup(label):
    """Mesh + fault + persistent-state meshvars."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )

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
    sigma_y_sym = 1.0 / weakness  # spatial yield stress

    # Velocity / pressure for both stages (separate fields)
    u_VE = uw.discretisation.MeshVariable(f"U_VE_{label}", mesh, 2, degree=2,
                                           vtype=VarType.VECTOR)
    p_VE = uw.discretisation.MeshVariable(f"P_VE_{label}", mesh, 1, degree=1,
                                           continuous=True, vtype=VarType.SCALAR)
    u_pl = uw.discretisation.MeshVariable(f"U_pl_{label}", mesh, 2, degree=2,
                                           vtype=VarType.VECTOR)
    p_pl = uw.discretisation.MeshVariable(f"P_pl_{label}", mesh, 1, degree=1,
                                           continuous=True, vtype=VarType.SCALAR)

    # Persistent state. η_lag stores the EFFECTIVE (yield-aware) BDF-1
    # viscosity from the prior step's converged Stage 2 — i.e., the
    # quantity that multiplies 2·ε̇ in σⁿ⁺¹ = α·σⁿ + 2·η_eff·ε̇. No
    # further BDF reduction is applied to it. α_lag is derived as
    # α = η_lag / (μΔt), which gives α = η_VE/(η_VE+μΔt) in the elastic
    # limit (η_lag = η_eff_pure) and α → 0 in the deep-yield limit
    # (η_lag → 0).
    eta_lag = uw.discretisation.MeshVariable(
        f"eta_lag_{label}", mesh, 1, degree=2,
        continuous=True, vtype=VarType.SCALAR,
    )
    alpha_lag = uw.discretisation.MeshVariable(
        f"alpha_lag_{label}", mesh, 1, degree=2,
        continuous=True, vtype=VarType.SCALAR,
    )
    sigma_hist = uw.discretisation.MeshVariable(
        f"sigma_hist_{label}", mesh, (2, 2), degree=2,
        vtype=VarType.SYM_TENSOR,
    )

    # Initial state: η_lag = η_eff_pure_VE everywhere (full elastic),
    # α_lag = η_eff_pure_VE / μΔt (pure-VE α), σ_hist = 0.
    eta_lag.array[...] = ETA_EFF_PURE_VE
    alpha_lag.array[...] = ETA_EFF_PURE_VE / (MU * DT)
    sigma_hist.array[...] = 0.0

    return dict(
        mesh=mesh, fault=fault, sigma_y_sym=sigma_y_sym,
        u_VE=u_VE, p_VE=p_VE, u_pl=u_pl, p_pl=p_pl,
        eta_lag=eta_lag, alpha_lag=alpha_lag, sigma_hist=sigma_hist,
    )


def _div_rank2_sym(tensor_sym, mesh):
    """Divergence of a symbolic rank-2 tensor → vector."""
    cs_x = mesh.CoordinateSystem.N[0]
    cs_y = mesh.CoordinateSystem.N[1]
    return sympy.Matrix([
        [tensor_sym[0, 0].diff(cs_x) + tensor_sym[0, 1].diff(cs_y)],
        [tensor_sym[1, 0].diff(cs_x) + tensor_sym[1, 1].diff(cs_y)],
    ])


def _build_stage1(setup, label):
    """Stage 1: linear VE warm-start with η = η_lag(x).

    Body force = −∇·(α_lag · σ_history) (the relaxation history
    acting as a known stress source). Pure linear viscous solve since
    η_lag and α_lag are frozen at the start of the step (carrying the
    prior step's converged yield-aware effective viscosity).
    """
    mesh = setup["mesh"]
    u_VE = setup["u_VE"]; p_VE = setup["p_VE"]
    eta_lag = setup["eta_lag"]
    alpha_lag = setup["alpha_lag"]
    sigma_hist = setup["sigma_hist"]

    stokes = uw.systems.Stokes(mesh, velocityField=u_VE, pressureField=p_VE)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    # Scalar meshvars carry their .sym as Matrix(1,1); unwrap to a plain
    # scalar so it composes cleanly with rank-2 tensors and other scalars.
    eta_lag_scalar = eta_lag.sym[0, 0]
    alpha_scalar = alpha_lag.sym[0, 0]

    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_lag_scalar

    stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,VE,{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")

    relax_history = alpha_scalar * sigma_hist.sym
    stokes.bodyforce = -_div_rank2_sym(relax_history, mesh)

    return stokes, V_top


def _build_stage2(setup, label):
    """Stage 2: nonlinear yield-aware Stokes (the point of truth).

    Active viscosity: softmin(η_lag(x), σ_y/(2|ε̇|)) — yield-aware,
    SNES Newton-iterated. Body force matches Stage 1 (same α·σ_hist
    relaxation). The frozen α_lag keeps the elastic relaxation rate
    decoupled from the within-step yield-aware iterates.
    """
    mesh = setup["mesh"]
    u_pl = setup["u_pl"]; p_pl = setup["p_pl"]
    eta_lag = setup["eta_lag"]
    alpha_lag = setup["alpha_lag"]
    sigma_hist = setup["sigma_hist"]
    sigma_y_sym = setup["sigma_y_sym"]

    stokes = uw.systems.Stokes(mesh, velocityField=u_pl, pressureField=p_pl)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)

    # Unwrap scalar meshvars from Matrix(1,1) to plain scalars
    eta_lag_scalar = eta_lag.sym[0, 0]
    alpha_scalar = alpha_lag.sym[0, 0]

    # Yield-aware viscosity: softmin(η_lag(x), σ_y/(2|ε̇|))
    E = stokes.Unknowns.E
    edot_inv_II = sympy.sqrt((E**2).trace() / 2 + sympy.Float(1.0e-12))
    eta_yield = sigma_y_sym / (2 * edot_inv_II)

    delta = 0.1
    f = eta_lag_scalar / eta_yield
    offset = (-1 + math.sqrt(1 + delta**2)) / 2
    g = 1 + (f - 1 + sympy.sqrt((f - 1)**2 + delta**2)) / 2 - offset
    eta_active_raw = eta_lag_scalar / g

    # Smooth lower floor: η_active ← (η_active_raw + √(η_raw² + 4·η_min²))/2
    # behaves as max(η_raw, η_min) but C¹-smooth, so SNES Newton has a
    # well-conditioned Jacobian everywhere (no kinks at the floor).
    # 1e-2 is a generous floor — η_active is just an estimate driving
    # α and the linear viscosity in next step's Stage 1.
    eta_min = sympy.Float(1.0e-2)
    eta_active = (eta_active_raw + sympy.sqrt(
        eta_active_raw**2 + 4 * eta_min**2
    )) / 2

    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_active

    stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,pl,{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")

    relax_history = alpha_scalar * sigma_hist.sym
    stokes.bodyforce = -_div_rank2_sym(relax_history, mesh)

    return stokes, V_top, eta_active


def run_two_stokes_v2(label, n_periods=1.5, use_stage1_warmstart=True):
    setup = _build_setup(label)
    mesh = setup["mesh"]
    eta_lag = setup["eta_lag"]
    alpha_lag = setup["alpha_lag"]
    sigma_hist = setup["sigma_hist"]

    stokes_VE, V_top_VE = _build_stage1(setup, label)
    stokes_pl, V_top_pl, eta_active_sym = _build_stage2(setup, label)

    u_VE = setup["u_VE"]; u_pl = setup["u_pl"]

    # Pre-compute σ_y at psi_star coords for diagnostic comparisons
    cx, cy = 0.5 * W, 0.5 * H
    theta = np.radians(THETA_DEG)
    n_x_l = -np.sin(theta); n_y_l = np.cos(theta)
    sigma_coords = sigma_hist.coords
    sd = np.abs((sigma_coords[:, 0] - cx) * n_x_l
                + (sigma_coords[:, 1] - cy) * n_y_l)
    half_length = 0.5 * FAULT_LENGTH
    along = (sigma_coords[:, 0] - cx) * n_y_l - (sigma_coords[:, 1] - cy) * n_x_l
    in_extent = np.abs(along) <= half_length
    weakness_arr = np.where(
        in_extent,
        (1.0 / TAU_Y_FAULT) * np.exp(-(sd / FAULT_WIDTH) ** 2)
        + (1.0 / TAU_Y_BULK) * (1.0 - np.exp(-(sd / FAULT_WIDTH) ** 2)),
        1.0 / TAU_Y_BULK * np.ones_like(sd),
    )
    sigma_y_at_nodes = 1.0 / weakness_arr

    # Coords for projecting η_active (scalar) — same nodes as eta_lag
    eta_coords = eta_lag.coords

    trace_path = os.path.join(
        os.path.dirname(__file__), f"_phase_g_{label}.trace.txt"
    )
    trace_fh = open(trace_path, "w")
    trace_fh.write(
        f"# Phase G v2 α-decoupled two-Stokes: {label}\n"
        f"# η_eff_pure_VE={ETA_EFF_PURE_VE:.6e}\n"
        f"# columns: step, t, V_top, snes_VE, snes_pl, "
        f"sigma_eq_max_VE, sigma_eq_max_total, "
        f"u_VE_y_max, u_pl_y_max, eta_lag_min, eta_lag_max, "
        f"yielded_fraction\n"
    )
    trace_fh.flush()

    T_END = n_periods * 2.0 * np.pi / OMEGA

    iters_VE = []; iters_pl = []
    sigma_eq_VE_per_step = []
    sigma_eq_total_per_step = []
    u_VE_y_max_per_step = []
    u_pl_y_max_per_step = []
    eta_lag_min_per_step = []
    eta_lag_max_per_step = []
    yielded_fraction_per_step = []

    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top_VE.sym = sympy.Float(v_now)
        V_top_pl.sym = sympy.Float(v_now)

        # ----- start of step: freeze α from η_lag -----
        # η_lag is already the EFFECTIVE (yield-aware) BDF-1 viscosity
        # from the prior step's converged Stage 2. We only need to
        # derive α_lag = η_lag/(μΔt) here. No further BDF reduction.
        eta_lag_arr = np.asarray(eta_lag.array).copy()
        alpha_arr = eta_lag_arr / (MU * dt)
        alpha_lag.array[...] = alpha_arr
        eta_lag_min_per_step.append(float(eta_lag_arr.min()))
        eta_lag_max_per_step.append(float(eta_lag_arr.max()))

        # ----- Stage 1: linear VE warm-start -----
        snes_VE_iters = 0
        if use_stage1_warmstart:
            try:
                stokes_VE.solve(zero_init_guess=False)
            except Exception as exc:
                print(f"  step t={t_end_step:.3f}: VE solve raised — {exc}",
                      flush=True)
                break
            snes_VE_iters = int(stokes_VE.snes.getIterationNumber())
            # Copy v_VE → u_pl as initial guess
            u_pl.array[...] = np.asarray(u_VE.array)

        u_VE_arr = np.asarray(u_VE.array).reshape(-1, 2)
        u_VE_y_max_per_step.append(float(np.abs(u_VE_arr[:, 1]).max()))

        # ----- Stage 2: nonlinear yield-aware Stokes (point of truth) -----
        try:
            stokes_pl.solve(zero_init_guess=False)
        except Exception as exc:
            print(f"  step t={t_end_step:.3f}: pl solve raised — {exc}",
                  flush=True)
            break
        snes_pl_iters = int(stokes_pl.snes.getIterationNumber())

        u_pl_arr = np.asarray(u_pl.array).reshape(-1, 2)
        u_pl_y_max_per_step.append(float(np.abs(u_pl_arr[:, 1]).max()))

        # ----- Compute σ_total = α_lag·σ_old + 2·η_active(ε̇)·ε̇ at sigma_coords -----
        # (sigma_hist will hold this for next step)
        E_pl_sym = stokes_pl.Unknowns.E
        edot_xx = np.asarray(uw.function.evaluate(E_pl_sym[0, 0], sigma_coords)).flatten()
        edot_xy = np.asarray(uw.function.evaluate(E_pl_sym[0, 1], sigma_coords)).flatten()
        edot_yy = np.asarray(uw.function.evaluate(E_pl_sym[1, 1], sigma_coords)).flatten()
        eta_active_at_sigma = np.asarray(
            uw.function.evaluate(eta_active_sym, sigma_coords)
        ).flatten()
        alpha_lag_at_sigma = np.asarray(
            uw.function.evaluate(alpha_lag.sym, sigma_coords)
        ).flatten()
        sigma_old = np.asarray(sigma_hist.array).copy()
        sigma_total = np.zeros_like(sigma_old)
        sigma_total[:, 0, 0] = (alpha_lag_at_sigma * sigma_old[:, 0, 0]
                                + 2 * eta_active_at_sigma * edot_xx)
        sigma_total[:, 0, 1] = (alpha_lag_at_sigma * sigma_old[:, 0, 1]
                                + 2 * eta_active_at_sigma * edot_xy)
        sigma_total[:, 1, 0] = sigma_total[:, 0, 1]
        sigma_total[:, 1, 1] = (alpha_lag_at_sigma * sigma_old[:, 1, 1]
                                + 2 * eta_active_at_sigma * edot_yy)
        sigma_eq_total = np.sqrt(1.5 * (sigma_total * sigma_total).sum(axis=(1, 2)))
        sigma_eq_total_per_step.append(float(sigma_eq_total.max()))

        # σ_VE diagnostic: stress as if no yield (using η_lag instead of η_active)
        eta_eff_at_sigma = np.asarray(
            uw.function.evaluate(eta_lag.sym, sigma_coords)
        ).flatten()
        # Use Stage 1 strain rate for σ_VE — what stage-1 v_VE produces
        E_VE_sym = stokes_VE.Unknowns.E
        edot_VE_xx = np.asarray(uw.function.evaluate(E_VE_sym[0, 0], sigma_coords)).flatten()
        edot_VE_xy = np.asarray(uw.function.evaluate(E_VE_sym[0, 1], sigma_coords)).flatten()
        edot_VE_yy = np.asarray(uw.function.evaluate(E_VE_sym[1, 1], sigma_coords)).flatten()
        sigma_VE = np.zeros_like(sigma_old)
        sigma_VE[:, 0, 0] = (alpha_lag_at_sigma * sigma_old[:, 0, 0]
                              + 2 * eta_eff_at_sigma * edot_VE_xx)
        sigma_VE[:, 0, 1] = (alpha_lag_at_sigma * sigma_old[:, 0, 1]
                              + 2 * eta_eff_at_sigma * edot_VE_xy)
        sigma_VE[:, 1, 0] = sigma_VE[:, 0, 1]
        sigma_VE[:, 1, 1] = (alpha_lag_at_sigma * sigma_old[:, 1, 1]
                              + 2 * eta_eff_at_sigma * edot_VE_yy)
        sigma_eq_VE = np.sqrt(1.5 * (sigma_VE * sigma_VE).sum(axis=(1, 2)))
        sigma_eq_VE_per_step.append(float(sigma_eq_VE.max()))

        # Yielded fraction (post-converged σ vs σ_y locally)
        n_yielded = int((sigma_eq_total > sigma_y_at_nodes * 0.99).sum())
        yielded_fraction_per_step.append(n_yielded / sigma_eq_total.size)

        # ----- Update persistent state for next step -----
        sigma_hist.array[...] = sigma_total

        # η_lag is updated by projecting η_active at eta_coords. This
        # is just a viscosity ESTIMATE used to set α and the linear-VE
        # viscosity in the NEXT step's Stage 1 — it does not appear in
        # any conservation law, so we can be liberal with regularization.
        # Floor 1e-4 prevents boundary-corner numerical artifacts from
        # collapsing η to ~1e-19 (which would give α=0 there and let
        # Stage 1 produce unbounded σ_VE). Cap at η_eff_pure_VE since
        # η_active ≤ η_eff_pure_VE algebraically; projection overshoots
        # above this are non-physical.
        eta_active_at_eta = np.asarray(
            uw.function.evaluate(eta_active_sym, eta_coords)
        ).flatten()
        eta_active_at_eta = np.clip(eta_active_at_eta,
                                     1.0e-2, ETA_EFF_PURE_VE)
        eta_lag.array[...] = eta_active_at_eta.reshape(-1, 1, 1)

        iters_VE.append(snes_VE_iters); iters_pl.append(snes_pl_iters)

        step_idx = len(iters_pl)
        trace_fh.write(
            f"{step_idx:4d} {t_end_step:7.4f} {v_now:+.4f} "
            f"{snes_VE_iters:3d} {snes_pl_iters:3d} "
            f"{sigma_eq_VE_per_step[-1]:.6e} "
            f"{sigma_eq_total_per_step[-1]:.6e} "
            f"{u_VE_y_max_per_step[-1]:.6e} "
            f"{u_pl_y_max_per_step[-1]:.6e} "
            f"{eta_lag_min_per_step[-1]:.6e} "
            f"{eta_lag_max_per_step[-1]:.6e} "
            f"{yielded_fraction_per_step[-1]:.6f}\n"
        )
        trace_fh.flush()
        if step_idx <= 5 or step_idx % 5 == 0:
            print(
                f"  step {step_idx:3d}/120  t={t_end_step:5.3f}  "
                f"V={v_now:+.3f}  VE={snes_VE_iters} pl={snes_pl_iters}  "
                f"|σ_VE|={sigma_eq_VE_per_step[-1]:.3e}  "
                f"|σ_tot|={sigma_eq_total_per_step[-1]:.3e}  "
                f"|u_pl_y|={u_pl_y_max_per_step[-1]:.3e}  "
                f"η_lag=[{eta_lag_min_per_step[-1]:.2e},{eta_lag_max_per_step[-1]:.2e}]  "
                f"yielded={yielded_fraction_per_step[-1]:.2%}",
                flush=True,
            )

        if (sigma_eq_total_per_step[-1] > 100.0
                or u_pl_y_max_per_step[-1] > 10.0):
            print(f"  *** runaway at step {step_idx} — breaking ***", flush=True)
            break

        t_cur = t_end_step

    trace_fh.close()

    iters_VE_arr = np.array(iters_VE); iters_pl_arr = np.array(iters_pl)
    print(
        f"  ran {len(iters_pl)} steps in {time.time()-t0:.1f}s; "
        f"two-Stokes α-decoupled (η_eff_pure_VE={ETA_EFF_PURE_VE:.3e})",
        flush=True,
    )
    if len(iters_pl) > 0:
        print(
            f"  SNES iters: VE mean={iters_VE_arr.mean():.1f} max={iters_VE_arr.max()} "
            f"| pl mean={iters_pl_arr.mean():.1f} max={iters_pl_arr.max()}",
            flush=True,
        )
        print(
            f"  σ_eq_total max: {max(sigma_eq_total_per_step):.4f}  "
            f"σ_eq_VE max: {max(sigma_eq_VE_per_step):.4f}",
            flush=True,
        )
        print(
            f"  |u_VE_y|_max: {max(u_VE_y_max_per_step):.4f}  "
            f"|u_pl_y|_max: {max(u_pl_y_max_per_step):.4f}",
            flush=True,
        )
        print(
            f"  yielded fraction: max={max(yielded_fraction_per_step):.2%}  "
            f"η_lag final range: [{eta_lag_min_per_step[-1]:.2e}, "
            f"{eta_lag_max_per_step[-1]:.2e}]",
            flush=True,
        )

    out_npz = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
    np.savez(
        out_npz,
        iters_VE=iters_VE_arr, iters_pl=iters_pl_arr,
        sigma_eq_VE_per_step=np.asarray(sigma_eq_VE_per_step),
        sigma_eq_total_per_step=np.asarray(sigma_eq_total_per_step),
        u_VE_y_max_per_step=np.asarray(u_VE_y_max_per_step),
        u_pl_y_max_per_step=np.asarray(u_pl_y_max_per_step),
        eta_lag_min_per_step=np.asarray(eta_lag_min_per_step),
        eta_lag_max_per_step=np.asarray(eta_lag_max_per_step),
        yielded_fraction_per_step=np.asarray(yielded_fraction_per_step),
        eta_eff_pure_VE=np.array(ETA_EFF_PURE_VE),
        T_END=np.array(T_END),
        n_steps=np.array(len(iters_pl)),
        wall_seconds=np.array(time.time() - t0),
    )
    print(f"  saved → {out_npz}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = [
        ("v2_no_warmstart", False),    # Stage 2 only — point-of-truth test
        ("v2_warmstart", True),         # also use Stage 1 as warm start
    ]
    for label, use_warm in cases:
        cache = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
        if os.path.exists(cache):
            print(f"\n=== {label}: cache hit, skipping ===", flush=True)
            continue
        print(
            f"\n=== Phase G v2 α-decoupled "
            f"(stage-1 warm-start={use_warm}) ===",
            flush=True,
        )
        run_two_stokes_v2(label, n_periods=1.5,
                          use_stage1_warmstart=use_warm)


if __name__ == "__main__":
    main()
