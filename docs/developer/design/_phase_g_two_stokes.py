"""Phase G: two-Stokes operator-split VEP — first prototype.

Architecture (from VEP_TWO_STOKES_OPERATOR_SPLIT.md and the user's framing):

  Stage 1  VE Stokes solve              →  v_VE, p_VE, σ_VE
  Stage 2  viscous Stokes solve          →  v_pl, p_pl
           - effective viscosity FROZEN at stage-1 value
           - σ_VE on RHS as ∇·σ_VE body force (elastic stresses fully explicit)
  J2 radial return on σ_total = σ_VE + 2η·ε̇(v_pl) → σ_admissible
  psi_star ← σ_admissible (next step's VE history)
  velocity ← v_pl (the plasticity-corrected velocity is the actual one)

Test harness identical to Phase F: isotropic VEP shear box, RES=32,
localised weak zone (fault.influence_function with τ_y_fault=0.05,
τ_y_bulk=200, θ=15°).

VE predictor: BDF-1 (this branch is off development — pre-ETD-1 work
in PR #161). The architecture is integrator-agnostic; once PR #161
merges we'll switch to ETD-1 (and try ETD-2 to see if the second
Stokes rescues higher-order under yielding).

Per-step trace files written each step (flushed); npz at end.
Runaway guard.
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

OUT_DIR = "output"


def _build_setup(label):
    """Build mesh, weak zone, and shared infrastructure used by both stages.

    Returns a dict with mesh, fault geometry, σ_y(x) field, and the
    two velocity / pressure meshvars (one pair per stage).
    """
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
    tau_y_field = 1.0 / weakness

    # Stage 1 (VE) variables
    u_VE = uw.discretisation.MeshVariable(f"U_VE_{label}", mesh, 2, degree=2,
                                           vtype=VarType.VECTOR)
    p_VE = uw.discretisation.MeshVariable(f"P_VE_{label}", mesh, 1, degree=1,
                                           continuous=True, vtype=VarType.SCALAR)

    # Stage 2 (plastic) variables — separate meshvars on the same mesh
    u_pl = uw.discretisation.MeshVariable(f"U_pl_{label}", mesh, 2, degree=2,
                                           vtype=VarType.VECTOR)
    p_pl = uw.discretisation.MeshVariable(f"P_pl_{label}", mesh, 1, degree=1,
                                           continuous=True, vtype=VarType.SCALAR)

    return dict(
        mesh=mesh, fault=fault,
        tau_y_field=tau_y_field,
        u_VE=u_VE, p_VE=p_VE,
        u_pl=u_pl, p_pl=p_pl,
    )


def _build_stage1_VE(setup, label, integrator="bdf", order=1):
    """Stage 1: VE Stokes solver (configurable integrator)."""
    mesh = setup["mesh"]
    u_VE = setup["u_VE"]; p_VE = setup["p_VE"]

    stokes_VE = uw.systems.Stokes(mesh, velocityField=u_VE, pressureField=p_VE)
    stokes_VE.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes_VE.Unknowns, integrator=integrator, order=order,
    )
    cm = stokes_VE.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = sympy.oo  # no in-residual yield
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6

    stokes_VE.saddle_preconditioner = 1.0 / cm.K
    stokes_VE.tolerance = 1.0e-4
    stokes_VE.petsc_options["ksp_type"] = "fgmres"
    stokes_VE.petsc_options["snes_force_iteration"] = True

    V_top_VE = expression(rf"V_{{top,VE,{label}}}", sympy.Float(0.0), "Top BC, stage 1")
    stokes_VE.add_essential_bc(sympy.Matrix([V_top_VE, 0.0]), "Top")
    stokes_VE.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes_VE.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes_VE.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes_VE.bodyforce = sympy.Matrix([0.0, 0.0])

    return stokes_VE, V_top_VE


def _build_stage2_plastic(setup, label, eta_eff_frozen, sigma_VE_sym):
    """Stage 2: viscous Stokes with frozen η and σ_VE on RHS.

    eta_eff_frozen: scalar value (the same effective viscosity that
                    stage 1 sees — for BDF-1 this is η·μΔt/(c0·η+μΔt))
    sigma_VE_sym:   sympy Matrix expression for σ_VE (read from
                    psi_star.sym after stage 1 solves). The body force
                    is −∇·σ_VE so that ∇·(σ_VE + 2η·ε̇(v_pl)) balances.
    """
    mesh = setup["mesh"]
    u_pl = setup["u_pl"]; p_pl = setup["p_pl"]

    stokes_pl = uw.systems.Stokes(mesh, velocityField=u_pl, pressureField=p_pl)
    stokes_pl.constitutive_model = uw.constitutive_models.ViscousFlowModel(
        stokes_pl.Unknowns,
    )
    cm = stokes_pl.constitutive_model
    cm.Parameters.shear_viscosity_0 = eta_eff_frozen

    stokes_pl.saddle_preconditioner = 1.0 / cm.K
    stokes_pl.tolerance = 1.0e-4
    stokes_pl.petsc_options["ksp_type"] = "fgmres"
    stokes_pl.petsc_options["snes_force_iteration"] = True

    V_top_pl = expression(rf"V_{{top,pl,{label}}}", sympy.Float(0.0), "Top BC, stage 2")
    stokes_pl.add_essential_bc(sympy.Matrix([V_top_pl, 0.0]), "Top")
    stokes_pl.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes_pl.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes_pl.add_essential_bc((sympy.oo, 0.0), "Right")

    # Body force = −∇·σ_VE so total stress (σ_VE + 2η·ε̇(v_pl)) is
    # divergence-free in absence of external body force.
    # σ_VE comes in via psi_star.sym (rank-2 sym tensor); divergence
    # is the symbolic gradient summed over j.
    coords = mesh.X.coords  # symbolic coordinate names
    # For a symmetric 2x2 σ: div(σ)_i = ∂σ_ix/∂x + ∂σ_iy/∂y
    # Use sympy diff over the BaseScalar coordinate atoms.
    cs_x = mesh.CoordinateSystem.N[0]
    cs_y = mesh.CoordinateSystem.N[1]
    div_sigma = sympy.Matrix([
        [sigma_VE_sym[0, 0].diff(cs_x) + sigma_VE_sym[0, 1].diff(cs_y)],
        [sigma_VE_sym[1, 0].diff(cs_x) + sigma_VE_sym[1, 1].diff(cs_y)],
    ])
    stokes_pl.bodyforce = -div_sigma

    return stokes_pl, V_top_pl


def _j2_radial_return(sigma_arr, sigma_y_arr):
    """J2 radial return at each quadrature node."""
    sig_dot_sig = (sigma_arr * sigma_arr).sum(axis=(1, 2))
    sigma_eq = np.sqrt(1.5 * sig_dot_sig)
    safe_eq = np.where(sigma_eq > 1.0e-12, sigma_eq, 1.0)
    scale = np.where(sigma_eq > sigma_y_arr, sigma_y_arr / safe_eq, 1.0)
    return sigma_arr * scale[:, None, None]


def _eta_eff_bdf1(eta_raw, mu_raw, dt):
    """BDF-1 VE-effective viscosity (c0=1)."""
    return eta_raw * mu_raw * dt / (eta_raw + mu_raw * dt)


def run_two_stokes(label, n_periods=1.5, integrator="bdf", order=1):
    setup = _build_setup(label)
    mesh = setup["mesh"]

    # Pre-compute σ_y at psi_star coords for the radial return
    cx, cy = 0.5 * W, 0.5 * H
    theta = np.radians(THETA_DEG)
    n_x_l = -np.sin(theta); n_y_l = np.cos(theta)

    # Stage 1 builds the VE Stokes — psi_star lives on this solver's DDt
    stokes_VE, V_top_VE = _build_stage1_VE(setup, label,
                                            integrator=integrator, order=order)
    DFDt = stokes_VE.Unknowns.DFDt

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

    # Stage 2 needs σ_VE.sym from psi_star (which is the stage-1 history).
    # σ_VE in psi_star post-stage-1 IS the rank-2 sym tensor we want.
    # Pass the .sym expression to the stage-2 builder.
    sigma_VE_sym = DFDt.psi_star[0].sym
    eta_eff_frozen = _eta_eff_bdf1(ETA, MU, DT)
    stokes_pl, V_top_pl = _build_stage2_plastic(setup, label, eta_eff_frozen, sigma_VE_sym)

    u_VE = setup["u_VE"]; u_pl = setup["u_pl"]

    # Trace file
    trace_path = os.path.join(
        os.path.dirname(__file__), f"_phase_g_{label}.trace.txt"
    )
    trace_fh = open(trace_path, "w")
    trace_fh.write(
        f"# Phase G two-Stokes operator-split: {label}\n"
        f"# η_eff_frozen={eta_eff_frozen:.6e}\n"
        f"# columns: step, t, V_top, snes_VE, snes_pl, "
        f"sigma_eq_max_VE, sigma_eq_max_total, sigma_eq_max_admissible, "
        f"u_VE_y_max, u_pl_y_max, yielded_fraction\n"
    )
    trace_fh.flush()

    T_END = n_periods * 2.0 * np.pi / OMEGA
    iters_VE = []; iters_pl = []
    sigma_eq_VE_per_step = []
    sigma_eq_total_per_step = []
    sigma_eq_admissible_per_step = []
    u_VE_y_max_per_step = []
    u_pl_y_max_per_step = []
    yielded_fraction_per_step = []

    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top_VE.sym = sympy.Float(v_now)
        V_top_pl.sym = sympy.Float(v_now)
        stokes_VE.constitutive_model.Parameters.dt_elastic = dt

        # Stage 1: VE solve
        try:
            stokes_VE.solve(zero_init_guess=False, timestep=dt)
        except Exception as exc:
            print(f"  step at t={t_end_step:.3f}: VE solve raised — {exc}", flush=True)
            break
        snes_VE_iters = int(stokes_VE.snes.getIterationNumber())

        sigma_VE = np.asarray(DFDt.psi_star[0].array).copy()
        sigma_eq_VE = np.sqrt(1.5 * (sigma_VE * sigma_VE).sum(axis=(1, 2)))
        sigma_eq_VE_per_step.append(float(sigma_eq_VE.max()))
        u_VE_arr = np.asarray(u_VE.array).reshape(-1, 2)
        u_VE_y_max_per_step.append(float(np.abs(u_VE_arr[:, 1]).max()))

        # NOTE on architecture (commit history):
        # 1. Single-shot with body force = -∇·σ_VE (uncorrected): works
        #    for BDF-1, blows up for ETD-* because σ_VE → σ_admissible
        #    via radial return creates a momentum imbalance.
        # 2. Single-shot with body force = -∇·σ_admissible (apply RR
        #    on σ_VE FIRST, then use as body-force): blows up for ALL
        #    three because σ_admissible has a sharp discontinuity at
        #    the yield boundary, ∇·σ_admissible is large+localised
        #    there, drives a big v_pl correction, σ_total overshoots,
        #    next clip is harder, runaway.
        # The architecture really needs INNER Picard within a timestep
        # to converge σ_admissible and v_pl simultaneously. This file
        # currently runs version 1 (uncorrected) — Picard pending.

        # Stage 2: viscous Stokes with σ_VE on RHS, frozen η
        try:
            stokes_pl.solve(zero_init_guess=False)
        except Exception as exc:
            print(f"  step at t={t_end_step:.3f}: pl solve raised — {exc}", flush=True)
            break
        snes_pl_iters = int(stokes_pl.snes.getIterationNumber())

        u_pl_arr = np.asarray(u_pl.array).reshape(-1, 2)
        u_pl_y_max_per_step.append(float(np.abs(u_pl_arr[:, 1]).max()))

        # Compute σ_total = σ_VE + 2η·ε̇(v_pl) at psi_star coords by
        # evaluating the symbolic strain rate of u_pl
        # NOTE: for simplicity, evaluate ε̇(v_pl) symbolically and add
        # 2·η_eff·ε̇ to σ_VE. Use the existing symbolic strain-rate of
        # the plastic-solver Unknowns.
        E_pl_sym = stokes_pl.Unknowns.E
        edot_xx = np.asarray(uw.function.evaluate(E_pl_sym[0, 0], sigma_coords)).flatten()
        edot_xy = np.asarray(uw.function.evaluate(E_pl_sym[0, 1], sigma_coords)).flatten()
        edot_yy = np.asarray(uw.function.evaluate(E_pl_sym[1, 1], sigma_coords)).flatten()
        # σ_total = σ_VE + 2η·ε̇(v_pl). Reverting to v1 architecture
        # (uncorrected body force) which works for BDF-1.
        sigma_total = sigma_VE.copy()
        sigma_total[:, 0, 0] += 2 * eta_eff_frozen * edot_xx
        sigma_total[:, 0, 1] += 2 * eta_eff_frozen * edot_xy
        sigma_total[:, 1, 0] += 2 * eta_eff_frozen * edot_xy
        sigma_total[:, 1, 1] += 2 * eta_eff_frozen * edot_yy
        sigma_eq_total = np.sqrt(1.5 * (sigma_total * sigma_total).sum(axis=(1, 2)))
        sigma_eq_total_per_step.append(float(sigma_eq_total.max()))

        # J2 radial return on σ_total
        sigma_admissible = _j2_radial_return(sigma_total, sigma_y_at_nodes)
        sigma_eq_admissible = np.sqrt(1.5 * (sigma_admissible * sigma_admissible).sum(axis=(1, 2)))
        sigma_eq_admissible_per_step.append(float(sigma_eq_admissible.max()))
        n_yielded = int((sigma_eq_total > sigma_y_at_nodes * 0.99).sum())
        yielded_fraction_per_step.append(n_yielded / sigma_eq_total.size)

        # Update psi_star with the admissible stress for next step's VE history
        DFDt.psi_star[0].array[...] = sigma_admissible

        iters_VE.append(snes_VE_iters); iters_pl.append(snes_pl_iters)

        step_idx = len(iters_VE)
        trace_fh.write(
            f"{step_idx:4d} {t_end_step:7.4f} {v_now:+.4f} "
            f"{snes_VE_iters:3d} {snes_pl_iters:3d} "
            f"{sigma_eq_VE_per_step[-1]:.6e} "
            f"{sigma_eq_total_per_step[-1]:.6e} "
            f"{sigma_eq_admissible_per_step[-1]:.6e} "
            f"{u_VE_y_max_per_step[-1]:.6e} "
            f"{u_pl_y_max_per_step[-1]:.6e} "
            f"{yielded_fraction_per_step[-1]:.6f}\n"
        )
        trace_fh.flush()
        if step_idx <= 5 or step_idx % 5 == 0:
            print(
                f"  step {step_idx:3d}/120  t={t_end_step:5.3f}  "
                f"V={v_now:+.3f}  VE={snes_VE_iters:2d} pl={snes_pl_iters:2d}  "
                f"|σ_VE|_eq={sigma_eq_VE_per_step[-1]:.3e}  "
                f"|σ_tot|_eq={sigma_eq_total_per_step[-1]:.3e}  "
                f"|σ_adm|_eq={sigma_eq_admissible_per_step[-1]:.3e}  "
                f"|u_pl_y|={u_pl_y_max_per_step[-1]:.3e}  "
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
        f"  ran {len(iters_VE)} steps in {time.time()-t0:.1f}s; "
        f"two-Stokes (BDF-1 VE, viscous plastic, η_frozen={eta_eff_frozen:.3e})",
        flush=True,
    )
    print(
        f"  SNES iters: VE mean={iters_VE_arr.mean():.1f} max={iters_VE_arr.max()} "
        f"| pl mean={iters_pl_arr.mean():.1f} max={iters_pl_arr.max()}",
        flush=True,
    )
    print(
        f"  σ_eq_total max: {max(sigma_eq_total_per_step):.4f}  "
        f"σ_eq_admissible max: {max(sigma_eq_admissible_per_step):.4f}",
        flush=True,
    )
    print(
        f"  |u_VE_y|_max: {max(u_VE_y_max_per_step):.4f}  "
        f"|u_pl_y|_max: {max(u_pl_y_max_per_step):.4f}",
        flush=True,
    )
    print(
        f"  yielded fraction: max={max(yielded_fraction_per_step):.2%}",
        flush=True,
    )

    out_npz = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
    np.savez(
        out_npz,
        iters_VE=iters_VE_arr, iters_pl=iters_pl_arr,
        sigma_eq_VE_per_step=np.asarray(sigma_eq_VE_per_step),
        sigma_eq_total_per_step=np.asarray(sigma_eq_total_per_step),
        sigma_eq_admissible_per_step=np.asarray(sigma_eq_admissible_per_step),
        u_VE_y_max_per_step=np.asarray(u_VE_y_max_per_step),
        u_pl_y_max_per_step=np.asarray(u_pl_y_max_per_step),
        yielded_fraction_per_step=np.asarray(yielded_fraction_per_step),
        eta_eff_frozen=np.array(eta_eff_frozen),
        T_END=np.array(T_END),
        n_steps=np.array(len(iters_VE)),
        wall_seconds=np.array(time.time() - t0),
    )
    print(f"  saved → {out_npz}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = [
        # (label, integrator, order)
        ("bdf1",  "bdf", 1),
        ("etd1",  "etd", 1),
        ("etd2",  "etd", 2),  # headline experiment
    ]
    for label, integrator, order in cases:
        cache = os.path.join(OUT_DIR, f"phase_g_{label}.npz")
        if os.path.exists(cache):
            print(f"\n=== {label}: cache hit, skipping ===", flush=True)
            continue
        print(
            f"\n=== Phase G two-Stokes ({integrator.upper()}-{order} VE "
            f"+ viscous plastic) ===",
            flush=True,
        )
        run_two_stokes(label, n_periods=1.5, integrator=integrator, order=order)


if __name__ == "__main__":
    main()
