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

ETA_LAG_FLOOR = 1.0e-3     # below physical yield range so floor isn't activated
ETA_LAG_CEIL  = ETA         # raw elastic upper bound

# Save spatial snapshots at these step indices (across the run, not just end)
# so we can compare fields at multiple loading-cycle phases.
SNAPSHOT_STEPS = (30, 60, 90, 120)

# Point-wise diagnostics: σ_eq at fault midpoint (centre of weak zone)
# and at a bulk reference point (well inside the elastic region).
DIAG_FAULT_POINT = np.array([[0.5 * W, 0.5 * H]])     # mesh centre
DIAG_BULK_POINT  = np.array([[0.25 * W, 0.75 * H]])   # NW quadrant (in bulk)

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
    # η_lag stores the EFFECTIVE (post-BDF-reduction) viscosity, so it
    # should be initialised to ve_eff_pure = η_VE·μΔt/(η_VE+μΔt), not raw η_VE.
    # Otherwise step 1 of the lagged-α variant uses α = η_lag/μΔt = 20 (huge)
    # which inherits a wrong-state ψ_star post-solve and accumulates over
    # subsequent steps. (Doesn't matter for the η_lag injection variant
    # which uses η_lag in shear_viscosity_0 and gets BDF-reduction applied
    # downstream — but harmless to be consistent.)
    eta_lag.array[...] = ETA * MU * DT / (ETA + MU * DT)

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


def _save_spatial_snapshot(label, step_idx, t, V_top_value, stokes, cm,
                           sigma_coords, sigma_arr):
    """Save spatial fields (velocity, ε̇, σ, viscosity) at a given step.

    Saved as ``output/phase_g_{label}_spatial_step{step_idx:03d}.npz``
    so multiple snapshots through the run are kept side-by-side.
    """
    u = stokes.Unknowns.u
    u_coords = np.asarray(u.coords).copy()
    u_array = np.asarray(u.array).reshape(-1, 2).copy()

    E_sym = stokes.Unknowns.E
    edot_xx = np.asarray(uw.function.evaluate(E_sym[0, 0], sigma_coords)).flatten()
    edot_xy = np.asarray(uw.function.evaluate(E_sym[0, 1], sigma_coords)).flatten()
    edot_yy = np.asarray(uw.function.evaluate(E_sym[1, 1], sigma_coords)).flatten()
    edot_inv_II = np.sqrt(0.5 * (edot_xx**2 + 2 * edot_xy**2 + edot_yy**2))

    viscosity = np.asarray(uw.function.evaluate(cm.viscosity, sigma_coords)).flatten()

    sigma_eq = np.sqrt(1.5 * (sigma_arr * sigma_arr).sum(axis=(1, 2)))

    out_path = os.path.join(
        OUT_DIR, f"phase_g_{label}_spatial_step{int(step_idx):03d}.npz"
    )
    np.savez(
        out_path,
        step_idx=np.array(int(step_idx)),
        t=np.array(t),
        V_top=np.array(V_top_value),
        u_coords=u_coords, u_array=u_array,
        sigma_coords=np.asarray(sigma_coords),
        sigma=sigma_arr,
        sigma_eq=sigma_eq,
        edot_xx=edot_xx, edot_xy=edot_xy, edot_yy=edot_yy,
        edot_inv_II=edot_inv_II,
        viscosity=viscosity,
    )
    print(f"  snapshot @ step {step_idx} → {out_path}", flush=True)


# ===========================================================================
# v5b: Custom constitutive law — operator-split yield-on-total VEP
# ===========================================================================
# As of 2026-05-01 this class lives in
# ``src/underworld3/constitutive_models.py`` as
# ``uw.constitutive_models.ViscoPlasticExplicitElastic``. The local copy
# below mirrors that production class so this runner remains self-contained
# (and runnable from worktrees that may not have the production class yet).
class ViscoPlasticExplicitElastic(
    uw.constitutive_models.ViscoElasticPlasticFlowModel
):
    """BDF-1 VEP with the elastic relaxation α decoupled from the
    within-SNES yield-aware viscosity.

    Standard BDF-1 VEP flux (from base class ``stress()``):
        flux = 2·viscosity·E_eff
             = 2·viscosity·E + (viscosity/(μΔt))·σ_star

    Notice the σ_star coefficient is the yield-aware ``viscosity``
    that SNES iterates — so the relaxation rate IS coupled to the
    plastic-iteration viscosity. That's what we're trying to undo.

    Custom flux here:
        flux = 2·viscosity·E + α_explicit·σ_star

    where:
      * ``viscosity`` is the standard yield-aware effective viscosity
        (softmin of η_eff and σ_y/(2|ε̇|)) — iterated within SNES, used
        only for the active strain-rate term;
      * ``α_explicit`` is a precomputed scalar or meshvar expression
        (set via ``set_alpha_explicit()``), frozen for the whole
        timestep — never iterated inside SNES.

    Two flavours of α_explicit are useful:
      - constant: ``α = η_VE/(η_VE + μΔt)`` (pure-VE relaxation,
        truly independent of yield);
      - yield-lagged: ``α(x) = η_lag(x)/(η_lag(x) + μΔt)`` from the
        prior step's converged effective viscosity (so failed zones
        still relax faster, but the spatial structure is locked in
        before the SNES iteration starts, not iterated within it).

    Notes:
      * The ``viscosity`` property still uses ``E_eff`` for the yield
        criterion (``vp_eff = σ_y/(2|E_eff|_II)``). That's the same
        coupling baseline has — kept here so the yield clip is
        comparable, only the σ_star *coefficient* differs.
      * Keeps psi_star management to the DDt; no manual zeroing.
    """

    def __init__(self, *args, alpha_explicit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha_explicit = alpha_explicit

    def set_alpha_explicit(self, alpha_sym):
        self._alpha_explicit = alpha_sym
        # Force re-setup so the new flux expression is picked up.
        self._is_setup = False
        self._solver_is_setup = False

    def stress(self):
        # Non-elastic / no DDt fallback — just the active term.
        if not self.is_elastic or self.Unknowns.DFDt is None:
            return 2 * self.viscosity * self.grad_u

        # σ_trial = 2 · η_VE_unclipped · E_eff
        #
        # The base class's `_unclipped_ve_viscosity` and `E_eff.sym`
        # together encode the high-order VE prediction:
        #   BDF-1: η_unclipped = ve_eff = η·μΔt/(η+μΔt),
        #          E_eff = E + σ_star/(2μΔt)
        #          → σ_trial = 2·ve_eff·E + (ve_eff/μΔt)·σ_star
        #                    = α_BDF·σ_star + 2·ve_eff·E
        #   BDF-2: η_unclipped = ve_eff(c_0=1.5),
        #          E_eff = E - c_1·σ_star/(2μΔt) - c_2·σ_2star/(2μΔt)
        #          (two-history-slot retention via c_1, c_2)
        #   ETD-1: η_unclipped = η_VE (raw),
        #          E_eff = (1-α)·E + α/(2η)·σ_star
        #          (φ=α makes the (φ-α)·ε̇_old term zero)
        #   ETD-2: η_unclipped = η_VE,
        #          E_eff = (1-φ)·E + α/(2η)·σ_star + (φ-α)·ε̇_old
        #          (with forcing_star)
        #
        # In all cases the σ_star (and ε̇_old) terms are frozen for the
        # timestep — only the bare E in E_eff iterates within SNES. So
        # the high-order time integrator is fully decoupled from the
        # nonlinear yield iteration.
        eta_unclipped = self._unclipped_ve_viscosity
        eta_unclipped_sym = (
            eta_unclipped.sym if hasattr(eta_unclipped, 'sym') else eta_unclipped
        )
        sigma_trial = 2 * eta_unclipped_sym * self.E_eff.sym

        # Apply yield correction on the TOTAL trial stress (softmin clip).
        # Same δ-smoothness as base-class viscosity property so the SNES
        # Jacobian is smooth at the yield boundary.
        yield_stress = self.Parameters.yield_stress
        if yield_stress.sym == sympy.oo:
            return sigma_trial

        delta = self._yield_softness  # 0.1 by default
        sigma_trial_inv_II = sympy.sqrt((sigma_trial**2).trace() / 2)
        f = sigma_trial_inv_II / yield_stress
        import math as _math
        offset = (-1 + _math.sqrt(1 + delta**2)) / 2
        g = 1 + (f - 1 + sympy.sqrt((f - 1)**2 + delta**2)) / 2 - offset
        # σⁿ⁺¹ = σ_trial · min(1, σ_y/|σ_trial|) ≈ σ_trial / g
        return sigma_trial / g


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

        # Spatial snapshot at instrumented checkpoint steps
        if step_idx in SNAPSHOT_STEPS:
            try:
                sigma_arr_now = np.asarray(DFDt.psi_star[0].array).copy()
                _save_spatial_snapshot(
                    label, step_idx, t_end_step, v_now,
                    stokes, cm, sigma_coords, sigma_arr_now,
                )
            except Exception as exc:
                print(f"  snapshot at step {step_idx} failed: {exc}",
                      flush=True)

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

        # Spatial snapshot at instrumented checkpoint steps
        if step_idx in SNAPSHOT_STEPS:
            try:
                sigma_arr_now = np.asarray(DFDt.psi_star[0].array).copy()
                _save_spatial_snapshot(
                    label, step_idx, t_end_step, v_now,
                    stokes, cm, sigma_coords, sigma_arr_now,
                )
            except Exception as exc:
                print(f"  snapshot at step {step_idx} failed: {exc}",
                      flush=True)

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


def run_v5_custom_constitutive(label, n_periods=1.5, use_yield_lagged_alpha=False,
                                 integrator='bdf', order=1,
                                 bdf_blend=1.0, etd_blend=1.0):
    """v5: custom constitutive law — operator-split VEP with yield-on-total.

    Uses ``ViscoPlasticExplicitElastic`` (defined above) which overrides
    the flux. The form depends on the time integrator:

      BDF-1: flux = clip(α·σ_old + 2·η_VE_eff·E, σ_y)
      ETD-1: same shape with α from exp(-Δt/τ)
      ETD-2: flux = clip(α·σ_old + 2·η_VE·(φ-α)·ε̇_old + 2·η_VE·(1-φ)·E, σ_y)

    Yield correction is a softmin clip on the TOTAL trial stress —
    matches the ground-truth radial-return behaviour. The σ_old (and
    ε̇_old for ETD-2) terms are explicit/frozen for the timestep, so
    the time-integration order is decoupled from the SNES iteration.
    """
    setup = _build_setup(label)
    mesh = setup["mesh"]
    eta_lag = setup["eta_lag"]
    u = setup["u"]
    sigma_y_sym = setup["sigma_y_sym"]

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=setup["p"])

    # α_explicit only used by the BDF path (ETD reads α/φ from DDt directly).
    if use_yield_lagged_alpha:
        eta_lag_scalar = eta_lag.sym[0, 0]
        alpha_explicit = eta_lag_scalar / (MU * DT)
    else:
        alpha_explicit = sympy.Float(ETA / (ETA + MU * DT))

    # Use the production class from constitutive_models.py — it has the
    # damping knobs (bdf_blend / etd_blend) the local class lacks.
    cm = uw.constitutive_models.ViscoPlasticExplicitElastic(
        stokes.Unknowns, integrator=integrator, order=order,
    )
    stokes.constitutive_model = cm
    cm.bdf_blend = bdf_blend
    cm.etd_blend = etd_blend
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

    DFDt = stokes.Unknowns.DFDt

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
        f"# Phase G v5 custom constitutive (use_yield_lagged_alpha="
        f"{use_yield_lagged_alpha}): {label}\n"
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

        # σⁿ⁺¹ now lives in psi_star[0] (DDt-projected from cm.flux).
        sigma_arr = np.asarray(DFDt.psi_star[0].array).copy()
        sigma_eq = np.sqrt(1.5 * (sigma_arr * sigma_arr).sum(axis=(1, 2)))
        sigma_eq_per_step.append(float(sigma_eq.max()))
        n_yielded = int((sigma_eq > sigma_y_at_nodes * 0.99).sum())
        yielded_fraction_per_step.append(n_yielded / sigma_eq.size)
        u_arr = np.asarray(u.array).reshape(-1, 2)
        u_y_max_per_step.append(float(np.abs(u_arr[:, 1]).max()))

        # If using yield-lagged α, refresh η_lag from converged viscosity
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

        # Spatial snapshot at instrumented checkpoint steps
        if step_idx in SNAPSHOT_STEPS:
            try:
                sigma_arr_now = np.asarray(DFDt.psi_star[0].array).copy()
                _save_spatial_snapshot(
                    label, step_idx, t_end_step, v_now,
                    stokes, cm, sigma_coords, sigma_arr_now,
                )
            except Exception as exc:
                print(f"  snapshot at step {step_idx} failed: {exc}",
                      flush=True)

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
        # (label, kind, use_lag, use_predictor, use_yield_lagged_alpha,
        #  integrator, order, bdf_blend, etd_blend)
        ("v3_baseline_const_eta", "v3", False, False, None,  None,  None, 1.0, 1.0),  # baseline
        ("v5b_bdf1",              "v5", None,  None,  False, "bdf", 1,    1.0, 1.0),
        ("v5b_bdf2",              "v5", None,  None,  False, "bdf", 2,    1.0, 1.0),
        ("v5b_bdf2_blend25",      "v5", None,  None,  False, "bdf", 2,    0.25, 1.0),  # damped BDF-2
        ("v5b_etd1",              "v5", None,  None,  False, "etd", 1,    1.0, 1.0),
        ("v5b_etd2",              "v5", None,  None,  False, "etd", 2,    1.0, 1.0),
        ("v5b_etd2_blend25",      "v5", None,  None,  False, "etd", 2,    1.0, 0.25),  # damped ETD-2
    ]
    for label, kind, use_lag, use_predictor, use_lagged_alpha, integrator, order, bdf_blend, etd_blend in cases:
        cache = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
        if os.path.exists(cache):
            print(f"\n=== {label}: cache hit, skipping ===", flush=True)
            continue
        if kind == "v3":
            print(f"\n=== Phase G v3 baseline ({label}) ===", flush=True)
            run_lag_case(label, n_periods=1.5,
                         use_lag=use_lag, use_predictor=use_predictor)
        elif kind == "v4":
            print(
                f"\n=== Phase G v4 explicit-elastic+VP ({label}, "
                f"use_yield_lagged_alpha={use_lagged_alpha}) ===",
                flush=True,
            )
            run_v4_explicit_elastic(label, n_periods=1.5,
                                     use_yield_lagged_alpha=use_lagged_alpha)
        elif kind == "v5":
            print(
                f"\n=== Phase G v5 custom constitutive ({label}, "
                f"integrator={integrator!r} order={order} "
                f"bdf_blend={bdf_blend} etd_blend={etd_blend}) ===",
                flush=True,
            )
            run_v5_custom_constitutive(
                label, n_periods=1.5,
                use_yield_lagged_alpha=use_lagged_alpha,
                integrator=integrator, order=order,
                bdf_blend=bdf_blend, etd_blend=etd_blend,
            )


if __name__ == "__main__":
    main()
