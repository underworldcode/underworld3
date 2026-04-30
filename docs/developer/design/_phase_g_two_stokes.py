"""Phase G v3: minimal η-lag modification + predictor-warmstart variants.

The custom α-decoupled two-Stokes architecture in v2 (archived as
_phase_g_v2_archive.py) had boundary-corner artifacts plummeting η_lag
to ~1e-19, then triggering huge ∇·(α·σ_hist) body forces and runaway.
Too many moving parts to debug cleanly.

Three variants compared on the same isotropic VEP harness:

  1. ``baseline``: standard BDF-1 VEP with constant η_VE — reference.
  2. ``lag``:      single Stokes per step, ``shear_viscosity_0 = η_lag(x)``
                   updated from yield-aware viscosity each step
                   (η_lag floor 0.1 to keep ≤10× viscosity contrast).
  3. ``predictor``: TWO Stokes solves per step:
                    Stage 1 = pure VE (yield_stress=∞, η=ETA constant)
                    Stage 2 = standard BDF-1 VEP softmin, warm-started
                              from v_VE.
                    Stage 2 is the "well-formed" yield solve; σ_VE
                    from Stage 1 enters only as a warm start (no body
                    force, avoiding the discontinuity pathology that
                    broke the original two-Stokes formulation).

All three use the SAME existing UW3 BDF-1 VEP solver (no new
constitutive code) — only the wrapper logic differs.

Test harness: isotropic VEP shear box, RES=32, localised weak zone
(τ_y_fault=0.05, τ_y_bulk=200, θ=15°), 1.5 forcing periods.
Per-step trace files; runaway guard.
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
TAU_Y_FAULT = 0.05
TAU_Y_BULK  = 200.0
THETA_DEG = 15.0
RES = 32

ETA_LAG_FLOOR = 1.0e-1     # floor at 10× below η_VE — keeps solver-tractable contrast
ETA_LAG_CEIL  = ETA         # raw elastic upper bound

OUT_DIR = "output"


def _build_setup(label):
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
    sigma_y_sym = 1.0 / weakness

    u = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, 2, degree=2, vtype=VarType.VECTOR,
    )
    p = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )
    # Separate fields for the predictor (Stage 1) variant — needed only
    # so Stage 1's purely-viscous solve doesn't perturb Stage 2's velocity
    # warm-start (which is the previous-step converged velocity in u).
    u_VE = uw.discretisation.MeshVariable(
        f"U_VE_{label}", mesh, 2, degree=2, vtype=VarType.VECTOR,
    )
    p_VE = uw.discretisation.MeshVariable(
        f"P_VE_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )
    eta_lag = uw.discretisation.MeshVariable(
        f"eta_lag_{label}", mesh, 1, degree=2,
        continuous=True, vtype=VarType.SCALAR,
    )
    eta_lag.array[...] = ETA  # init: raw elastic everywhere

    # σ_hist (rank-2 sym tensor) — externally-managed stress history for
    # the v4 (explicit-elastic) variant only. Stores σⁿ from the prior
    # step's converged result. Initialised to zero on step 0.
    # degree=1 to match psi_star's layout (the DDt allocates psi_star
    # at degree=1 for SemiLagrangian stress history).
    sigma_hist = uw.discretisation.MeshVariable(
        f"sigma_hist_{label}", mesh, (2, 2), degree=1,
        vtype=VarType.SYM_TENSOR,
    )
    sigma_hist.array[...] = 0.0

    return dict(
        mesh=mesh, fault=fault, sigma_y_sym=sigma_y_sym,
        u=u, p=p, u_VE=u_VE, p_VE=p_VE,
        eta_lag=eta_lag, sigma_hist=sigma_hist,
    )


def _div_rank2_sym(tensor_sym, mesh):
    """Divergence of a symbolic rank-2 tensor → vector (2D)."""
    cs_x = mesh.CoordinateSystem.N[0]
    cs_y = mesh.CoordinateSystem.N[1]
    return sympy.Matrix([
        [tensor_sym[0, 0].diff(cs_x) + tensor_sym[0, 1].diff(cs_y)],
        [tensor_sym[1, 0].diff(cs_x) + tensor_sym[1, 1].diff(cs_y)],
    ])


def _build_stokes(setup, label, use_lag, use_predictor):
    """Stage 2 (the main yield-aware VEP solver).

    For all three variants this is the standard BDF-1 VEP solver:
      - baseline:  shear_viscosity_0 = ETA constant
      - lag:       shear_viscosity_0 = η_lag(x) updated from prior step's |ε̇|
      - predictor: shear_viscosity_0 = η_lag(x) updated from current
                   step's Stage-1 |ε̇| (set by the runner before each solve)

    The yield_stress softmin handles within-SNES yield iteration in all
    cases. Only the source of η_lag differs.
    """
    mesh = setup["mesh"]
    u = setup["u"]; p = setup["p"]
    eta_lag = setup["eta_lag"]
    sigma_y_sym = setup["sigma_y_sym"]

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, integrator='bdf', order=1,
    )
    cm = stokes.constitutive_model

    if use_lag or use_predictor:
        cm.Parameters.shear_viscosity_0 = eta_lag.sym[0, 0]
    else:
        cm.Parameters.shear_viscosity_0 = ETA

    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = sigma_y_sym
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6

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

    return stokes, V_top, cm


def _build_stokes_v4(setup, label, use_yield_lagged_alpha):
    """v4 Stage 2: explicit elastic term + visco-plastic solver.

    Trick: use the standard BDF-1 VEP model (μ=MU finite, yield-aware
    softmin in residual), but force psi_star=0 at the start of each
    step. With psi_star=0, the model's internal history term
    α·psi_star contributes 0, so its residual becomes:
        σ_residual = 2·viscosity_yield_aware·ε̇
    i.e., a pure visco-plastic stress. We then ADD the elastic
    relaxation as an external body force:
        bodyforce = −∇·(α · σ_hist_prior)
        α = η_VE/(η_VE+μΔt)                  (use_yield_lagged_alpha=False)
          = η_lag(x)/(η_lag(x)+μΔt)          (use_yield_lagged_alpha=True)

    Effective full stress at convergence:
        σⁿ⁺¹ = α·σ_hist + 2·viscosity_yield_aware·ε̇(v)

    The α is precomputed at start of step and frozen — never iterated
    inside SNES. The within-SNES nonlinearity is purely the yield-aware
    viscosity on |ε̇|. This is the "explicit elastic, VP solver"
    architecture.

    No shear_modulus=∞ shenanigans (which crashed with FNORM_NAN);
    instead we zero psi_star each step in the runner before solve.
    """
    mesh = setup["mesh"]
    u = setup["u"]; p = setup["p"]
    sigma_y_sym = setup["sigma_y_sym"]
    sigma_hist = setup["sigma_hist"]
    eta_lag = setup["eta_lag"]

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, integrator='bdf', order=1,
    )
    cm = stokes.constitutive_model
    # Standard BDF-1 VEP setup — same as baseline. Maxwell relaxation
    # IS present in the model, but we'll zero psi_star before each solve
    # so its history contribution vanishes.
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = sigma_y_sym
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6

    stokes.saddle_preconditioner = 1.0 / cm.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,v4_{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")

    # Explicit elastic body force = −∇·(α · σ_hist).
    # For const-α: α is a scalar.
    # For lagged-α: α(x) = η_lag(x)/(η_lag(x)+μΔt) — meshvar expression.
    if use_yield_lagged_alpha:
        eta_lag_scalar = eta_lag.sym[0, 0]
        alpha_sym = eta_lag_scalar / (eta_lag_scalar + MU * DT)
    else:
        alpha_sym = sympy.Float(ETA / (ETA + MU * DT))

    relax_history = alpha_sym * sigma_hist.sym
    stokes.bodyforce = -_div_rank2_sym(relax_history, mesh)

    return stokes, V_top, cm, alpha_sym


def _build_stokes_VE_predictor(setup, label):
    """Stage 1 (predictor variant only): pure viscous Stokes with η_VE
    constant. No yield, no elasticity (no DDt). Cheap linear solve whose
    velocity field gives a fresh estimate of |ε̇| at the current loading
    state, used to compute η_lag for Stage 2.
    """
    mesh = setup["mesh"]
    u_VE = setup["u_VE"]; p_VE = setup["p_VE"]

    stokes = uw.systems.Stokes(mesh, velocityField=u_VE, pressureField=p_VE)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,VE,{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])

    return stokes, V_top


def run_lag_case(label, n_periods=1.5, use_lag=False, use_predictor=False):
    setup = _build_setup(label)
    mesh = setup["mesh"]
    eta_lag = setup["eta_lag"]
    u = setup["u"]
    sigma_y_sym = setup["sigma_y_sym"]

    stokes, V_top, cm = _build_stokes(setup, label, use_lag, use_predictor)
    DFDt = stokes.Unknowns.DFDt

    # Stage 1 (predictor variant only): pure viscous, gives v_VE → |ε̇_VE|
    if use_predictor:
        stokes_VE, V_top_VE = _build_stokes_VE_predictor(setup, label)
    else:
        stokes_VE, V_top_VE = None, None

    # σ_y at psi_star coords (for diagnostics)
    cx, cy = 0.5 * W, 0.5 * H
    theta = np.radians(THETA_DEG)
    n_x_l = -np.sin(theta); n_y_l = np.cos(theta)
    sigma_coords = DFDt.psi_star[0].coords
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

    eta_coords = eta_lag.coords

    trace_path = os.path.join(
        os.path.dirname(__file__), f"_phase_g_{label}.trace.txt"
    )
    trace_fh = open(trace_path, "w")
    trace_fh.write(
        f"# Phase G v3 minimal lag (use_lag={use_lag}): {label}\n"
        f"# columns: step, t, V_top, snes, sigma_eq_max, "
        f"u_y_max, eta_lag_min, eta_lag_max, yielded_fraction\n"
    )
    trace_fh.flush()

    T_END = n_periods * 2.0 * np.pi / OMEGA

    iters = []
    sigma_eq_per_step = []
    u_y_max_per_step = []
    eta_lag_min_per_step = []
    eta_lag_max_per_step = []
    yielded_fraction_per_step = []

    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt

        # ----- Stage 1 (predictor variant only): pure viscous solve to
        # estimate |ε̇| at current loading. Use this to update η_lag for
        # Stage 2's effective viscosity.
        snes_VE_iters = 0
        if use_predictor:
            V_top_VE.sym = sympy.Float(v_now)
            try:
                stokes_VE.solve(zero_init_guess=False)
            except Exception as exc:
                print(f"  step t={t_end_step:.3f}: VE pre-solve raised — {exc}",
                      flush=True)
                break
            snes_VE_iters = int(stokes_VE.snes.getIterationNumber())
            # Update η_lag from |ε̇(v_VE)| at eta_coords
            E_VE_sym = stokes_VE.Unknowns.E
            edot_xx = np.asarray(uw.function.evaluate(E_VE_sym[0, 0], eta_coords)).flatten()
            edot_xy = np.asarray(uw.function.evaluate(E_VE_sym[0, 1], eta_coords)).flatten()
            edot_yy = np.asarray(uw.function.evaluate(E_VE_sym[1, 1], eta_coords)).flatten()
            edot_inv_II = np.sqrt(
                0.5 * (edot_xx**2 + 2 * edot_xy**2 + edot_yy**2)
                + 1.0e-12
            )
            sigma_y_at_eta = np.asarray(
                uw.function.evaluate(sigma_y_sym, eta_coords)
            ).flatten()
            eta_yield = sigma_y_at_eta / (2 * edot_inv_II)
            eta_active = np.minimum(ETA, eta_yield)
            eta_active = np.clip(eta_active, ETA_LAG_FLOOR, ETA_LAG_CEIL)
            eta_lag.array[...] = eta_active.reshape(-1, 1, 1)

        # Record start-of-step η_lag for diagnostics
        eta_lag_arr = np.asarray(eta_lag.array).copy()
        eta_lag_min_per_step.append(float(eta_lag_arr.min()))
        eta_lag_max_per_step.append(float(eta_lag_arr.max()))

        try:
            stokes.solve(zero_init_guess=False, timestep=dt)
        except Exception as exc:
            print(f"  step t={t_end_step:.3f}: solve raised — {exc}",
                  flush=True)
            break
        snes_iters = int(stokes.snes.getIterationNumber())

        # Read out the converged stress (DDt has already shifted ψ_star
        # in update_post_solve; psi_star[0] is now σⁿ⁺¹).
        sigma_arr = np.asarray(DFDt.psi_star[0].array).copy()
        sigma_eq = np.sqrt(1.5 * (sigma_arr * sigma_arr).sum(axis=(1, 2)))
        sigma_eq_per_step.append(float(sigma_eq.max()))
        n_yielded = int((sigma_eq > sigma_y_at_nodes * 0.99).sum())
        yielded_fraction_per_step.append(n_yielded / sigma_eq.size)
        u_arr = np.asarray(u.array).reshape(-1, 2)
        u_y_max_per_step.append(float(np.abs(u_arr[:, 1]).max()))

        # ----- Update η_lag from |ε̇| and σ_y at eta_coords -----
        # Compute the yield-aware *raw* viscosity directly:
        #     η_lag_new = min(ETA, σ_y / (2|ε̇|))
        # This is what we want to inject into shear_viscosity_0 next step
        # so the model's own BDF reduction gives the right ve_eff. We
        # bypass cm.viscosity (which is post-BDF-reduction and produces
        # double-reduction on round-trip; also slow to evaluate).
        if use_lag:
            E_sym = stokes.Unknowns.E
            edot_xx = np.asarray(
                uw.function.evaluate(E_sym[0, 0], eta_coords)
            ).flatten()
            edot_xy = np.asarray(
                uw.function.evaluate(E_sym[0, 1], eta_coords)
            ).flatten()
            edot_yy = np.asarray(
                uw.function.evaluate(E_sym[1, 1], eta_coords)
            ).flatten()
            edot_inv_II = np.sqrt(
                0.5 * (edot_xx**2 + 2 * edot_xy**2 + edot_yy**2)
                + 1.0e-12
            )
            sigma_y_at_eta = np.asarray(
                uw.function.evaluate(sigma_y_sym, eta_coords)
            ).flatten()
            eta_yield = sigma_y_at_eta / (2 * edot_inv_II)
            eta_active = np.minimum(ETA, eta_yield)
            eta_active = np.clip(eta_active, ETA_LAG_FLOOR, ETA_LAG_CEIL)
            eta_lag.array[...] = eta_active.reshape(-1, 1, 1)

        iters.append(snes_iters)
        step_idx = len(iters)
        trace_fh.write(
            f"{step_idx:4d} {t_end_step:7.4f} {v_now:+.4f} "
            f"{snes_iters:3d} "
            f"{sigma_eq_per_step[-1]:.6e} "
            f"{u_y_max_per_step[-1]:.6e} "
            f"{eta_lag_min_per_step[-1]:.6e} "
            f"{eta_lag_max_per_step[-1]:.6e} "
            f"{yielded_fraction_per_step[-1]:.6f}\n"
        )
        trace_fh.flush()
        if step_idx <= 5 or step_idx % 5 == 0:
            print(
                f"  step {step_idx:3d}/120  t={t_end_step:5.3f}  "
                f"V={v_now:+.3f}  snes={snes_iters}  "
                f"|σ|={sigma_eq_per_step[-1]:.3e}  "
                f"|u_y|={u_y_max_per_step[-1]:.3e}  "
                f"η_lag=[{eta_lag_min_per_step[-1]:.2e},"
                f"{eta_lag_max_per_step[-1]:.2e}]  "
                f"yielded={yielded_fraction_per_step[-1]:.2%}",
                flush=True,
            )

        if (sigma_eq_per_step[-1] > 100.0
                or u_y_max_per_step[-1] > 10.0):
            print(f"  *** runaway at step {step_idx} — breaking ***",
                  flush=True)
            break

        t_cur = t_end_step

    trace_fh.close()

    iters_arr = np.array(iters)
    print(
        f"  ran {len(iters)} steps in {time.time()-t0:.1f}s "
        f"(use_lag={use_lag})",
        flush=True,
    )
    if len(iters) > 0:
        print(
            f"  SNES iters: mean={iters_arr.mean():.1f} max={iters_arr.max()}",
            flush=True,
        )
        print(
            f"  σ_eq max: {max(sigma_eq_per_step):.4f}  "
            f"|u_y|_max: {max(u_y_max_per_step):.4f}  "
            f"yielded max: {max(yielded_fraction_per_step):.2%}",
            flush=True,
        )
        print(
            f"  η_lag final range: [{eta_lag_min_per_step[-1]:.2e}, "
            f"{eta_lag_max_per_step[-1]:.2e}]",
            flush=True,
        )

    out_npz = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
    np.savez(
        out_npz,
        iters=iters_arr,
        sigma_eq_per_step=np.asarray(sigma_eq_per_step),
        u_y_max_per_step=np.asarray(u_y_max_per_step),
        eta_lag_min_per_step=np.asarray(eta_lag_min_per_step),
        eta_lag_max_per_step=np.asarray(eta_lag_max_per_step),
        yielded_fraction_per_step=np.asarray(yielded_fraction_per_step),
        T_END=np.array(T_END),
        n_steps=np.array(len(iters)),
        wall_seconds=np.array(time.time() - t0),
    )
    print(f"  saved → {out_npz}", flush=True)


def run_v4_explicit_elastic(label, n_periods=1.5, use_yield_lagged_alpha=False):
    """v4: explicit elastic term as body force, visco-plastic solver in residual."""
    setup = _build_setup(label)
    mesh = setup["mesh"]
    sigma_hist = setup["sigma_hist"]
    eta_lag = setup["eta_lag"]
    u = setup["u"]
    sigma_y_sym = setup["sigma_y_sym"]

    eta_eff_const = ETA * MU * DT / (ETA + MU * DT)
    alpha_const_value = ETA / (ETA + MU * DT)

    stokes, V_top, cm, alpha_sym = _build_stokes_v4(
        setup, label, use_yield_lagged_alpha,
    )
    DFDt = stokes.Unknowns.DFDt

    # σ_y at sigma_coords (for diagnostics)
    sigma_coords = sigma_hist.coords
    cx, cy = 0.5 * W, 0.5 * H
    theta = np.radians(THETA_DEG)
    n_x_l = -np.sin(theta); n_y_l = np.cos(theta)
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

    eta_coords = eta_lag.coords

    trace_path = os.path.join(
        os.path.dirname(__file__), f"_phase_g_{label}.trace.txt"
    )
    trace_fh = open(trace_path, "w")
    trace_fh.write(
        f"# Phase G v4 explicit elastic + VP solver (use_yield_lagged_alpha="
        f"{use_yield_lagged_alpha}): {label}\n"
        f"# η_eff_const={eta_eff_const:.6e}  α_const={alpha_const_value:.6e}\n"
        f"# columns: step, t, V_top, snes, sigma_eq_max, "
        f"u_y_max, eta_lag_min, eta_lag_max, yielded_fraction\n"
    )
    trace_fh.flush()

    T_END = n_periods * 2.0 * np.pi / OMEGA
    iters = []
    sigma_eq_per_step = []
    u_y_max_per_step = []
    eta_lag_min_per_step = []
    eta_lag_max_per_step = []
    yielded_fraction_per_step = []

    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt  # not used (μ=∞) but kept consistent

        # Record start-of-step η_lag (only meaningful for yield-lagged α)
        eta_lag_arr = np.asarray(eta_lag.array).copy()
        eta_lag_min_per_step.append(float(eta_lag_arr.min()))
        eta_lag_max_per_step.append(float(eta_lag_arr.max()))

        # Snapshot σ_hist (prior step's σⁿ) for body force, then ZERO
        # the model's psi_star so its internal history term contributes
        # 0 — only the active part 2·viscosity·ε̇ comes out of the residual.
        sigma_old = np.asarray(sigma_hist.array).copy()
        DFDt.psi_star[0].array[...] = 0.0

        try:
            stokes.solve(zero_init_guess=False, timestep=dt)
        except Exception as exc:
            print(f"  step t={t_end_step:.3f}: solve raised — {exc}",
                  flush=True)
            break
        snes_iters = int(stokes.snes.getIterationNumber())

        # After solve, psi_star[0] holds the model's post-converged stress
        # which (with psi_star_prev=0) equals 2·viscosity·ε̇ — the active
        # visco-plastic stress, yield-clipped via softmin. Combine with
        # the explicit elastic part α·σ_hist to get σⁿ⁺¹.
        sigma_active = np.asarray(DFDt.psi_star[0].array).copy()

        if use_yield_lagged_alpha:
            alpha_at_sigma = np.asarray(
                uw.function.evaluate(alpha_sym, sigma_coords)
            ).flatten()
        else:
            alpha_at_sigma = np.full(sigma_active.shape[0], alpha_const_value)

        sigma_new = np.zeros_like(sigma_old)
        sigma_new[:, 0, 0] = alpha_at_sigma * sigma_old[:, 0, 0] + sigma_active[:, 0, 0]
        sigma_new[:, 0, 1] = alpha_at_sigma * sigma_old[:, 0, 1] + sigma_active[:, 0, 1]
        sigma_new[:, 1, 0] = sigma_new[:, 0, 1]
        sigma_new[:, 1, 1] = alpha_at_sigma * sigma_old[:, 1, 1] + sigma_active[:, 1, 1]

        sigma_eq = np.sqrt(1.5 * (sigma_new * sigma_new).sum(axis=(1, 2)))
        sigma_eq_per_step.append(float(sigma_eq.max()))
        n_yielded = int((sigma_eq > sigma_y_at_nodes * 0.99).sum())
        yielded_fraction_per_step.append(n_yielded / sigma_eq.size)
        u_arr = np.asarray(u.array).reshape(-1, 2)
        u_y_max_per_step.append(float(np.abs(u_arr[:, 1]).max()))

        # Store σⁿ⁺¹ for next step's body force
        sigma_hist.array[...] = sigma_new

        # If using yield-lagged α, also update η_lag from converged viscosity
        # for next step's α computation.
        if use_yield_lagged_alpha:
            visc_at_eta = np.asarray(
                uw.function.evaluate(cm.viscosity, eta_coords)
            ).flatten()
            visc_at_eta = np.clip(visc_at_eta, ETA_LAG_FLOOR, ETA_LAG_CEIL)
            eta_lag.array[...] = visc_at_eta.reshape(-1, 1, 1)

        iters.append(snes_iters)
        step_idx = len(iters)
        trace_fh.write(
            f"{step_idx:4d} {t_end_step:7.4f} {v_now:+.4f} "
            f"{snes_iters:3d} "
            f"{sigma_eq_per_step[-1]:.6e} "
            f"{u_y_max_per_step[-1]:.6e} "
            f"{eta_lag_min_per_step[-1]:.6e} "
            f"{eta_lag_max_per_step[-1]:.6e} "
            f"{yielded_fraction_per_step[-1]:.6f}\n"
        )
        trace_fh.flush()
        if step_idx <= 5 or step_idx % 5 == 0:
            print(
                f"  step {step_idx:3d}/120  t={t_end_step:5.3f}  "
                f"V={v_now:+.3f}  snes={snes_iters}  "
                f"|σ|={sigma_eq_per_step[-1]:.3e}  "
                f"|u_y|={u_y_max_per_step[-1]:.3e}  "
                f"yielded={yielded_fraction_per_step[-1]:.2%}",
                flush=True,
            )

        if (sigma_eq_per_step[-1] > 100.0
                or u_y_max_per_step[-1] > 10.0):
            print(f"  *** runaway at step {step_idx} — breaking ***",
                  flush=True)
            break

        t_cur = t_end_step

    trace_fh.close()

    iters_arr = np.array(iters)
    print(
        f"  ran {len(iters)} steps in {time.time()-t0:.1f}s "
        f"(use_yield_lagged_alpha={use_yield_lagged_alpha})",
        flush=True,
    )
    if len(iters) > 0:
        print(
            f"  SNES iters: mean={iters_arr.mean():.1f} max={iters_arr.max()}",
            flush=True,
        )
        print(
            f"  σ_eq max: {max(sigma_eq_per_step):.4f}  "
            f"|u_y|_max: {max(u_y_max_per_step):.4f}  "
            f"yielded max: {max(yielded_fraction_per_step):.2%}",
            flush=True,
        )

    out_npz = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
    np.savez(
        out_npz,
        iters=iters_arr,
        sigma_eq_per_step=np.asarray(sigma_eq_per_step),
        u_y_max_per_step=np.asarray(u_y_max_per_step),
        eta_lag_min_per_step=np.asarray(eta_lag_min_per_step),
        eta_lag_max_per_step=np.asarray(eta_lag_max_per_step),
        yielded_fraction_per_step=np.asarray(yielded_fraction_per_step),
        T_END=np.array(T_END),
        n_steps=np.array(len(iters)),
        wall_seconds=np.array(time.time() - t0),
    )
    print(f"  saved → {out_npz}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = [
        # (label, kind, use_lag, use_predictor, use_yield_lagged_alpha)
        ("v3_baseline_const_eta", "v3", False, False, None),  # baseline reference
        ("v4_const_alpha",        "v4", None,  None,  False), # α from η_VE constant
        ("v4_lagged_alpha",       "v4", None,  None,  True),  # α from prior η_lag
    ]
    for label, kind, use_lag, use_predictor, use_lagged_alpha in cases:
        cache = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
        if os.path.exists(cache):
            print(f"\n=== {label}: cache hit, skipping ===", flush=True)
            continue
        if kind == "v3":
            print(f"\n=== Phase G v3 baseline ({label}) ===", flush=True)
            run_lag_case(label, n_periods=1.5,
                         use_lag=use_lag, use_predictor=use_predictor)
        else:
            print(
                f"\n=== Phase G v4 explicit-elastic+VP ({label}, "
                f"use_yield_lagged_alpha={use_lagged_alpha}) ===",
                flush=True,
            )
            run_v4_explicit_elastic(label, n_periods=1.5,
                                     use_yield_lagged_alpha=use_lagged_alpha)


if __name__ == "__main__":
    main()
