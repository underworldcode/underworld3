"""Jury-rigged exponential integrator in UW3 — proof of concept.

Tests:
  1. Iso VE harmonic on a 2:1 box (matches bench_ve_harmonic geometry).
     Decision gate: must reproduce BDF-2's max|err| ≈ 1.34e-3 or beat it.
  2. Iso VEP harmonic with spatial yield_stress field on the 1:1 mesh
     that blew up the BDF-2 consistency test.  Decision gate: must
     stay bounded with reasonable peak |σ| ≤ 1.1·τ_y.

Architecture (jury-rig, not production):

  * Custom ``MaxwellExpFlowModel`` subclasses ``ViscousFlowModel`` and
    overrides ``flux`` to return:

        σ = 2·η_eff·(1-φ)·ε̇  +  α·σⁿ  +  2·η_eff·(φ-α)·ε̇ⁿ

    where ``α, φ`` are scalar UWexpressions updated per step.

  * Two new MeshVariables (``sigma_n_var``, ``epsdot_n_var``) hold the
    two history streams.  After each solve we write the new σ and ε̇
    back to these variables via direct nodal evaluation — not L2
    projection.  Adequate for this proof-of-concept; production would
    use SNES_Tensor_Projection.

  * Yield handling (Test 2): η_eff = softmin(η(1-φ), η_pl), η_pl =
    τ_y/(2·|ε̇_inv|).  Lagged-τ: τ_eff used in α, φ comes from the
    *previous* step's η_eff (Picard-style; full self-consistent
    iteration would solve τ↔σ inside the SNES).
"""

import numpy as np
import sympy
import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression
from underworld3.constitutive_models import ViscousFlowModel


# ─────────────────────────────────────────────────────────────────
# Custom constitutive model
# ─────────────────────────────────────────────────────────────────

class MaxwellExpFlowModel(ViscousFlowModel):
    """Jury-rigged Maxwell with exponential time integration.

    flux = 2·η_eff·(1-φ)·ε̇ + α·σⁿ + 2·η·(φ-α)·ε̇ⁿ

    For VE: η_eff = η.  For VEP: η_eff = softmin(η, η_pl) (set externally
    on the model's ``yield_stress`` parameter and recomputed per step).
    """

    def __init__(self, unknowns, sigma_n_var, epsdot_n_var, **kwargs):
        super().__init__(unknowns, **kwargs)
        self._sigma_n = sigma_n_var
        self._epsdot_n = epsdot_n_var
        # Coefficients updated per timestep.  Initialise to non-degenerate
        # values so the JIT compile produces an invertible Jacobian — at
        # JIT time, the autodiff bakes in the symbolic structure, but the
        # FIRST PetscDS constants[] update uses the *current* UWexpression
        # values, so we want 2η(1-φ) > 0 even if update_exp_coeffs hasn't
        # been called yet.  φ = 0 means "fully viscous (no relaxation)";
        # this is the right "neutral" starting point.
        self._exp_alpha = expression(r"{\alpha_{\rm exp}}", sympy.Float(0.0),
                                      "exponential α = exp(-Δt/τ)")
        self._exp_phi = expression(r"{\varphi_{\rm exp}}", sympy.Float(0.0),
                                    "exponential φ = (1-α)/(Δt/τ)")
        # Yield-stress UWexpression — set externally for VEP, leave at oo for VE
        self._tau_y = expression(r"{\tau_y}", sympy.oo, "yield stress")
        self._strainrate_min = expression(r"{\dot\varepsilon_{\min}}",
                                           sympy.Float(1e-6),
                                           "strain-rate floor for η_pl")
        self._softness = 0.1  # softmin δ

    @property
    def K(self):
        """Stiffness for saddle preconditioner — use raw η."""
        return self.Parameters.shear_viscosity_0

    def _eta_eff(self):
        """Yield-limited viscosity (softmin)."""
        eta = self.Parameters.shear_viscosity_0
        ty = self._tau_y
        # If τ_y is ∞, no yield
        if hasattr(ty, 'sym') and ty.sym is sympy.oo:
            return eta
        # η_pl = τ_y / (2·|ε̇_inv|).  Use current ε̇ (sym(∇u))
        E = self.Unknowns.E
        epsII = sympy.sqrt((E**2).trace() / 2)
        eta_pl = ty / (2 * (epsII + self._strainrate_min))
        # softmin(eta, eta_pl)
        delta = self._softness
        f = eta / eta_pl
        import math
        offset = (-1 + math.sqrt(1 + delta**2)) / 2
        g = 1 + (f - 1 + sympy.sqrt((f - 1)**2 + delta**2)) / 2 - offset
        return eta / g

    @property
    def flux(self):
        """Stress = 2·η_eff·(1-φ)·ε̇ + α·σⁿ + 2·η·(φ-α)·ε̇ⁿ."""
        eta_eff = self._eta_eff()
        eta_raw = self.Parameters.shear_viscosity_0
        E = self.Unknowns.E
        return (2 * eta_eff * (1 - self._exp_phi) * E
                + self._exp_alpha * self._sigma_n.sym
                + 2 * eta_raw * (self._exp_phi - self._exp_alpha) * self._epsdot_n.sym)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def update_exp_coeffs(cm, dt, tau_eff):
    """Set α, φ on the constitutive model for this step.  Uses the
    *lagged* τ_eff from the previous step's η_eff."""
    x = float(dt) / float(tau_eff)
    if x < 1e-10:
        alpha, phi = 1.0, 1.0
    else:
        alpha = float(np.exp(-x))
        phi = (1.0 - alpha) / x
    cm._exp_alpha.sym = sympy.Float(alpha)
    cm._exp_phi.sym = sympy.Float(phi)


def project_history(stokes, cm, sigma_n_var, epsdot_n_var,
                    eta_raw, alpha_val, phi_val):
    """After a solve, update σⁿ and ε̇ⁿ history variables.

    Strategy that avoids the Matrix-evaluate-with-derivatives issue:
      1. Evaluate ε̇^{n+1} component-wise (scalars) and write to a
         temporary array.
      2. Compute σ^{n+1} purely on numpy .array data using the
         exponential update formula (mesh-variable reads, no derivs).
      3. Update both history variables.
    """
    # Step 1: project ε̇^{n+1} component-wise
    E_sym = stokes.Unknowns.E
    coords = epsdot_n_var.coords
    e_xx = np.asarray(uw.function.evaluate(E_sym[0, 0], coords)).flatten()
    e_xy = np.asarray(uw.function.evaluate(E_sym[0, 1], coords)).flatten()
    e_yy = np.asarray(uw.function.evaluate(E_sym[1, 1], coords)).flatten()
    new_epsdot = np.zeros_like(epsdot_n_var.array)
    new_epsdot[:, 0, 0] = e_xx
    new_epsdot[:, 1, 1] = e_yy
    new_epsdot[:, 0, 1] = e_xy
    new_epsdot[:, 1, 0] = e_xy

    # Step 2: σ^{n+1} = α·σⁿ + 2η(1-φ)·ε̇^{n+1} + 2η(φ-α)·ε̇ⁿ
    # All quantities are nodal arrays — pure numpy.
    a = alpha_val
    p = phi_val
    sigma_old = np.array(sigma_n_var.array)         # σⁿ (snapshot)
    epsdot_old = np.array(epsdot_n_var.array)       # ε̇ⁿ (snapshot)
    new_sigma = (a * sigma_old
                 + 2 * eta_raw * (1 - p) * new_epsdot
                 + 2 * eta_raw * (p - a) * epsdot_old)

    # Step 3: write back
    sigma_n_var.array[...] = new_sigma
    epsdot_n_var.array[...] = new_epsdot


def probe_centre_xy(sigma_n_var, c=np.array([[0.5, 0.5]])):
    """Read σ_xy at the domain centre from sigma_n_var."""
    coords = sigma_n_var.coords
    idx = int(np.argmin(np.linalg.norm(coords - c, axis=1)))
    return float(sigma_n_var.array[idx, 0, 1])


# ─────────────────────────────────────────────────────────────────
# Test 1 — Iso VE harmonic, 2:1 antisymmetric box, peak-start IC
# ─────────────────────────────────────────────────────────────────

def test_1_iso_VE_harmonic():
    print("\n=== Test 1: iso VE harmonic — exp integrator vs BDF-2 baseline ===",
          flush=True)
    ETA = 1.0; MU = 1.0
    V0 = 0.5
    OMEGA = np.pi / 2.0
    DT = 0.05
    T_END = 4 * 2 * np.pi / OMEGA   # 4 periods

    H = 1.0; W = 2.0
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W/2, -H/2), maxCoords=(W/2, H/2),
    )
    v = uw.discretisation.MeshVariable("U_exp", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_exp", mesh, 1, degree=1)
    sigma_n = uw.discretisation.MeshVariable(
        "sigma_n", mesh, 2, degree=2, vtype=VarType.SYM_TENSOR,
    )
    epsdot_n = uw.discretisation.MeshVariable(
        "epsdot_n", mesh, 2, degree=2, vtype=VarType.SYM_TENSOR,
    )

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = MaxwellExpFlowModel(stokes.Unknowns, sigma_n, epsdot_n)
    cm.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model = cm
    stokes.tolerance = 1e-6
    stokes.petsc_options["snes_force_iteration"] = True

    # Antisymmetric BCs (matches bench_ve_harmonic.py)
    V_top_expr = expression(r"V_{top}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc((V_top_expr, 0.0), "Top")
    stokes.add_essential_bc((-V_top_expr, 0.0), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")

    # Peak-start IC: plant σ_xy = A_inf in sigma_n_var so step 1 starts
    # at the steady-state cycle peak under cos(ωt+φ) forcing.
    t_r = ETA / MU
    De = OMEGA * t_r
    gamma_dot_0 = 2.0 * V0 / H
    A_inf = ETA * gamma_dot_0 / np.sqrt(1.0 + De**2)
    phi_lag = float(np.arctan(De))

    # Initialize sigma_n with σ_xy = A_inf, others = 0
    sigma_n.array[:, 0, 0] = 0.0
    sigma_n.array[:, 1, 1] = 0.0
    sigma_n.array[:, 0, 1] = A_inf
    sigma_n.array[:, 1, 0] = A_inf
    # Initialize epsdot_n with ε̇_xy = γ̇₀/2 · cos(-ω·DT + φ_lag) ≈ γ̇₀/2
    # The exact peak-start ε̇ value is small for the harmonic, but use 0
    # initially — the first solve will correct.
    epsdot_n.array[...] = 0.0

    # Pure VE: τ_eff = η/μ = 1
    tau_eff = ETA / MU

    times, sxys, reasons = [], [], []
    t_cur = 0.0
    while t_cur < T_END - 1e-9:
        update_exp_coeffs(cm, DT, tau_eff)
        t_end = t_cur + DT
        v_now = V0 * float(np.cos(OMEGA * t_end + phi_lag))
        V_top_expr.sym = sympy.Float(v_now)
        stokes.solve(zero_init_guess=False)
        # Pull current α, φ from the model (we just set them via update_exp_coeffs)
        a = float(cm._exp_alpha.sym); p = float(cm._exp_phi.sym)
        project_history(stokes, cm, sigma_n, epsdot_n,
                        eta_raw=ETA, alpha_val=a, phi_val=p)
        sxys.append(probe_centre_xy(sigma_n, c=np.array([[0.0, 0.0]])))
        reasons.append(int(stokes.snes.getConvergedReason()))
        times.append(t_end)
        t_cur = t_end

    times = np.array(times); sxys = np.array(sxys)
    sigma_ana = A_inf * np.cos(OMEGA * times)
    err = np.abs(sxys - sigma_ana)
    print(f"  steps={len(times)}, peak|σ_xy|={np.abs(sxys).max():.4f}, "
          f"max|err|={err.max():.4e}, rms={np.sqrt((err**2).mean()):.4e}",
          flush=True)
    print(f"  diverged: {(np.array(reasons) < 0).sum()}/{len(reasons)}",
          flush=True)
    print(f"  baseline (bench_ve_harmonic BDF-2 peak-start): max|err| = 1.34e-3",
          flush=True)
    return times, sxys, sigma_ana


# ─────────────────────────────────────────────────────────────────
# Test 2 — Iso VEP with spatial yield_stress (the consistency case)
# ─────────────────────────────────────────────────────────────────

def test_2_iso_VEP_spatial():
    print("\n=== Test 2: iso VEP harmonic w/ spatial τ_y — exp vs BDF-2 (which blew up) ===",
          flush=True)
    ETA = 1.0; MU = 1.0
    V0 = 0.5
    OMEGA = np.pi / 2.0
    DT = 0.05
    T_END = 16.0  # match the consistency test
    TAU_Y_FAULT = 0.30
    TAU_Y_BULK = 200.0

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable("U_exp2", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_exp2", mesh, 1, degree=1)
    sigma_n = uw.discretisation.MeshVariable(
        "sigma_n_2", mesh, 2, degree=2, vtype=VarType.SYM_TENSOR,
    )
    epsdot_n = uw.discretisation.MeshVariable(
        "epsdot_n_2", mesh, 2, degree=2, vtype=VarType.SYM_TENSOR,
    )

    fault = uw.meshing.Surface(
        "fault_exp", mesh,
        np.array([[0.2, 0.5], [0.8, 0.5]]),
    )
    fault.discretize()
    weakness = fault.influence_function(
        width=0.06, value_near=1.0/TAU_Y_FAULT, value_far=1.0/TAU_Y_BULK,
        profile="gaussian",
    )
    ty_field = 1.0 / weakness

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = MaxwellExpFlowModel(stokes.Unknowns, sigma_n, epsdot_n)
    cm.Parameters.shear_viscosity_0 = ETA
    cm._tau_y.sym = ty_field  # spatial yield_stress
    stokes.constitutive_model = cm
    stokes.tolerance = 1e-6
    stokes.petsc_options["snes_force_iteration"] = True

    V_top_expr = expression(r"V_{top}^{(2)}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc((V_top_expr, 0.0), "Top")
    stokes.add_essential_bc((0.0, 0.0), "Bottom")  # asymmetric (matches consistency test)
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")

    # σ=0 IC
    sigma_n.array[...] = 0.0
    epsdot_n.array[...] = 0.0

    # Lagged τ_eff — start at τ_VE = η/μ = 1
    tau_eff = ETA / MU

    times, sxys, reasons = [], [], []
    t_cur = 0.0
    phi_lag = float(np.arctan(OMEGA))
    n_steps = int(T_END / DT)
    for step in range(n_steps):
        update_exp_coeffs(cm, DT, tau_eff)
        t_end = t_cur + DT
        v_now = V0 * float(np.cos(OMEGA * t_end + phi_lag))
        V_top_expr.sym = sympy.Float(v_now)
        try:
            stokes.solve(zero_init_guess=False, divergence_retries=2)
        except Exception as exc:
            print(f"    step {step+1}: solve failed: {exc}", flush=True)
            break
        # Pull current α, φ from the model (we just set them via update_exp_coeffs)
        a = float(cm._exp_alpha.sym); p = float(cm._exp_phi.sym)
        project_history(stokes, cm, sigma_n, epsdot_n,
                        eta_raw=ETA, alpha_val=a, phi_val=p)
        sxys.append(probe_centre_xy(sigma_n, c=np.array([[0.5, 0.5]])))
        reasons.append(int(stokes.snes.getConvergedReason()))
        times.append(t_end)
        t_cur = t_end

    times = np.array(times); sxys = np.array(sxys)
    print(f"  steps={len(times)}, peak|σ_xy|={np.abs(sxys).max():.4f}, "
          f"diverged: {(np.array(reasons) < 0).sum()}/{len(reasons)}",
          flush=True)
    print(f"  baseline (BDF-2 same setup, consistency test): peak|σ_xy| = 13377  ← BLEW UP",
          flush=True)
    print(f"  expected exp result: bounded |σ_xy| ≲ τ_y = 0.30 (yield-clipped)",
          flush=True)
    return times, sxys


def main():
    test_1_iso_VE_harmonic()
    test_2_iso_VEP_spatial()


if __name__ == "__main__":
    main()
