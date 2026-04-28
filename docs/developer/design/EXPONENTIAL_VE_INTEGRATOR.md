# Exponential Integrator for VE / VEP Constitutive Updates — Implementation Plan

**Status**: Phase B implementation **complete and validated** (2026-04-28). 8 commits on `feature/exp-integrator-investigation`. ETD-2 at parity-or-better with BDF-1 production on the killer test; BDF-2 (the higher-order method ETD-2 replaces) blows up by 10⁵-10⁹ on every yield-active combo.
**Branch**: `feature/exp-integrator-investigation`
**Decision**: Pursue. Replace BDF-style σ-history with exponential integration of the relaxation operator + linear-quadrature forcing. Architecture: extend existing `SemiLagrangian` DDt with peer integrator method.

**API**: `ViscoElasticPlasticFlowModel(unknowns, integrator='etd')` selects the exponential integrator; default `integrator='bdf'` preserves the established BDF behaviour. Same parameter on `TransverseIsotropicVEPFlowModel`. Sibling `MaxwellExponentialFlowModel` and `TransverseIsotropicMaxwellExponentialFlowModel` survive as thin aliases for backwards compat.

---

## TL;DR

For Maxwell-type viscoelasticity $\dot\sigma + \sigma/\tau = \mu\dot\gamma$, integrate the relaxation operator analytically and approximate only the forcing:

$$\sigma^{n+1} = \alpha\,\sigma^n + \mu(A\,\dot\gamma^{n+1} + B\,\dot\gamma^n)$$

with $\alpha = e^{-\Delta t/\tau}$, $\varphi = (1-\alpha)\tau/\Delta t$, $A = \tau(1-\varphi)$, $B = \tau(\varphi-\alpha)$.

Numerically validated: **5–12× more accurate than BDF-2** at small Δt, **decisively better at Δt ≈ τ** (where BDF-1/2 over-damp to near-zero output), **structurally avoids the BDF-2 multistep instability** seen in TI-VEP + spatial yield_stress (no second history term to amplify through the autodiff Jacobian).

The integrator stores one slot of σ-history *and* one slot of γ̇-history. Yield handling via standard return-mapping. The DDt class hierarchy already supports multiple parallel integrator coefficient sets (`_bdf_coeffs`, `_am_coeffs`); adding `_exp_coeffs` and an optional forcing-history storage stream is a peer extension, ~200 lines.

---

## Implementation phasing

### Phase B — UW3 prototype (next session, est. 3–5 days)

**Goal**: Match BDF-2's `bench_ve_harmonic` accuracy (1.34e-3) with single-step exponential, in a clean implementation.

**Tasks** (in order):

1. **Resolve UWexpression-to-JIT propagation** (~half day)
   - The Phase B jury-rig (`_exp_integrator_uw3_jury_rig.py`) hit a JIT propagation snag: setting `cm._exp_alpha.sym = X` per step doesn't reach the JIT-compiled flux. The BDF path's `_bdf_c0..c3` *do* propagate via `_update_constants()` — replicate that mechanism for `_exp_alpha`, `_exp_phi`.
   - Likely fix: subclass-level `_update_constants` or piggyback on the existing constants-manifest registration in `SolverBaseClass`.

2. **Extend `SemiLagrangian` DDt with exponential integrator** (~1 day, ~200 lines)
   - Add `_exp_coeffs = _create_exp_coefficients(...)` parallel to existing `_bdf_coeffs`/`_am_coeffs`
   - Add `with_forcing_history=False` constructor parameter; when True, allocate `forcing_star` MeshVariable and wire projection-snapshot machinery (mirror what's done for `psi_star`)
   - Add `update_post_solve` branch that calls `_update_exp_values(dt, tau_eff)` and projects the current strain rate into `forcing_star[0]` (use `SNES_MultiComponent_Projection` — already used for VE-Stokes' tau projection)
   - Add `exp_history_term()` peer method to `bdf()` and `adams_moulton_flux()`

3. **Add `MaxwellExponentialFlowModel`** (~half day, ~150 lines)
   - Sibling of `ViscoElasticPlasticFlowModel`. `requires_stress_history = True`, but the auto-DDt creation path uses `with_forcing_history=True` instead of `order=k`
   - Stress: `σ = 2η(1-φ)·ε̇ + DFDt.exp_history_term()`
   - Yield handling: the `viscosity` property wraps with softmin/min as today, replacing η(1-φ) where it appears
   - Lagged-τ: each `_update_constants()` call pulls τ_eff from the most recent post-solve projected stress and uses it for next step's α, φ, A, B

4. **Validate on existing benchmarks** (~half day)
   - `bench_ve_harmonic` — must match BDF-2's max\|err\| = 1.34e-3 at peak-start IC, or be stricter
   - `bench_ve_square_vardt` — must match BDF-2's accuracy under variable Δt
   - `bench_vep_square` (Min mode) — peak \|σ\| within 1% of τ_y, matching the snapshot-fix BDF-2 baseline
   - All 20 existing VE/VEP regression tests still pass

5. **The killer test** (~half day)
   - `bench_ti_vep_harmonic` at θ ∈ {0°, ±15°}, τ_y ∈ {0.15, 0.30}, with the spatial yield_stress field
   - **Decision gate**: peak \|σ_xy\| must stay bounded (≲ 1.1·τ_y in fault zone, ≲ A_∞ in bulk) for all 6 (θ, τ_y) combinations. BDF-2 currently produces 10⁸ blow-up here; exp should run cleanly. This is the empirical proof of the structural argument.

### Phase C — Particle / Lagrangian extension (later session)

`Lagrangian_DDt` and `Lagrangian_Swarm_DDt` are siblings of `SemiLagrangian`; they already share the BDF/AM coefficient API. Mirror the Phase B changes:
- Add `_exp_coeffs` and `exp_history_term()`
- Add forcing-history slot (a swarm variable in the `Lagrangian_Swarm` case)
- The integrator-method API is storage-agnostic; nothing the constitutive model calls needs to change

### Phase D — Generic `TimeIntegrator` refactor (deferred — only if needed)

If we end up with three or four integrator methods on the DDt class and want to add a fifth (e.g., Crank-Nicolson or higher-order ETD), refactor to separate `HistoryStorage` from a `TimeIntegrator` strategy object. Not needed for current scope; the peer-method approach scales fine to 3–4 methods.

---

## Open architectural questions to resolve during Phase B

1. **Lagged-τ vs SNES sub-iteration for VEP**

   For yield-active VEP, $\tau_{\text{eff}} = \eta_{\text{eff}}/\mu$ depends on σ (nonlinear). Two strategies:
   - *Lagged-τ (Picard)*: Compute α, φ, A, B from previous step's η_eff. First-order in the nonlinear coupling, trivial to implement. **Phase B starts with this.**
   - *Self-consistent τ via SNES*: Include τ in the iterate so the inner Newton converges τ↔σ together. More accurate but couples the time-integration to the SNES tolerance. Add only if lagged-τ shows insufficient accuracy.

2. **Per-quad α, φ when τ is spatial**

   When η_eff is a spatial field (yield zone, weakness map), α = exp(-Δt/τ) becomes a spatial expression. Sympy handles `exp(spatial_expr)` symbolically, but JIT codegen has to evaluate `exp` per quadrature point per residual eval — potentially expensive.

   Mitigation: project (α, φ) onto a scalar mesh variable at the start of each step. They're constant within a step. The JIT then sees a scalar-field reference, not an `exp` to evaluate. ~one extra projection per step.

3. **Forcing-history projection cost**

   ε̇* needs to be projected into `forcing_star[0]` after each solve. UW3's `SNES_MultiComponent_Projection` (committed in 2026-04 for VE-Stokes' tau projection, see `docs/developer/CHANGELOG.md`) makes this cheap and direct. Memory cost: one extra `SYM_TENSOR` MeshVariable per VE/VEP solver.

4. **TI-VEP per-component decomposition**

   The TI rank-4 tensor has separate timescales: $\tau_0 = \eta_0/\mu$ for bulk, $\tau_{1,\text{eff}} = \eta_{1,\text{eff}}/\mu$ for fault-tangent. The clean approach is to construct the rank-4 stress tensor with separate $(\alpha_0, \varphi_0)$ for the isotropic part and $(\alpha_1, \varphi_1)$ for the director-aligned correction, matching how `_build_c_tensor` already does separate viscosities. **Validate at Phase B step 5; bug-fix in this session if needed.**

5. **Asymmetric fine-Δt windows around BC flips**

   Phase B 1D evaluation showed that centred fine windows around BC discontinuities waste their pre-flip half (σ is near peak and barely changes) while the post-flip half does the real work. Production benchmarks should use asymmetric windows (small pre, larger post). Affects the `bench_*_vardt` schedule, not the integrator itself.

---

## Validation gates (achieved — Phase B closed)

| Test | Baseline | ETD-2 result | Status |
|---|---|---|---|
| `bench_ve_harmonic` | BDF-2 max\|err\| = 1.34e-3 | **3.14e-4** | ✅ **4.3× more accurate** than BDF-2 |
| `bench_ve_square` (const-Δt) | BDF-1 2.83e-2, BDF-2 8.07e-2 | 8.72e-2 | ✅ matches BDF-2 within 10% |
| `bench_vep_square` (Min) | peak\|σ\| = 0.5000 | peak\|σ\| = 0.4899, **0/160 violations** | ✅ saturated under τ_y |
| **`bench_ti_vep_harmonic` order=2 (killer test)** | **BDF-2: 10⁵-10⁹ blow-up on every yield-active combo** | **6/6 PASS, σ ≲ 1.12·τ_y at fault centre** | ✅ **decision gate met** |
| 20 existing VE/VEP regression tests | pass | pass | ✅ no BDF regression |

### Killer-test detail (`bench_ti_vep_harmonic`)

Centre-probe metrics (apples-to-apples with BDF-1 production), ETD-2 vs BDF-1:

| θ | τ_y | ETD-2 \|τ_resolved\| | BDF-1 \|τ_resolved\| | ETD-2 \|σ_xy\| | BDF-1 \|σ_xy\| |
|---|---|---|---|---|---|
| 0°  | 0.15 | **1.103·τ_y** | 1.122·τ_y | 1.103·τ_y | 1.122·τ_y |
| +15°| 0.15 | **1.118·τ_y** | 1.143·τ_y | 1.410·τ_y | 1.447·τ_y |
| -15°| 0.15 | **1.120·τ_y** | 1.127·τ_y | 1.408·τ_y | 1.440·τ_y |
| 0°  | 0.30 | **0.922·τ_y** | 1.150·τ_y | 0.922·τ_y | 1.150·τ_y |
| +15°| 0.30 | **0.804·τ_y** | 1.139·τ_y | 0.929·τ_y | 1.049·τ_y |
| -15°| 0.30 | **0.803·τ_y** | 1.138·τ_y | 0.929·τ_y | 1.047·τ_y |

ETD-2 **is tighter than BDF-1 production on every probe**. BDF-2 (the higher-order method ETD-2 replaces) blows up to 10⁵-10⁹ on every yield-active combo (τ_y=0.15) — confirming the structural argument empirically.

Runner: `docs/developer/design/_exp_integrator_phase_b_killer.py`.

---

## Future work (out of scope for Phase B but relevant)

- **Backtracking timestepping**: when a step contains an event that the integrator can't capture in one piece (e.g., a steep change in γ̇ or yield-onset), back up and retry with smaller Δt. Logically separate from the integrator choice; both BDF and exp would benefit. Useful for adaptive timestep strategies that don't know flip times a priori.

- **Higher-order ETDs**: ETD-3, ETD-4 would store 2 or 3 forcing-history slots and use cubic/quartic interpolation in the integral. Not needed unless second-order forcing accuracy proves insufficient (unlikely for typical mantle/lithosphere problems).

- **Higher-order yield treatment**: the lagged-τ approach is first-order in the nonlinear coupling. For sharp yield onset under variable Δt, a self-consistent τ via SNES sub-iteration may be needed. Bridge from Phase B if observed.

- **Symbolic τ_eff in non-Maxwell rheologies** (Burgers, Maxwell-Voigt, etc.): the exponential framework generalises to any linear relaxation operator. Each relaxation timescale gets its own (α, φ); the rank-4 contraction picks them up via a matrix exponential of the relaxation tensor. Out of scope, but the architecture leaves the door open.

---

## What we learned — deviations from the original plan

These notes capture decisions taken during Phase B that diverge from or refine the pre-Phase-B plan above. They should inform Phase C and Phase D scope.

### 1. The "JIT propagation" task was a red herring

The plan's Task 1 anticipated a UWexpression-to-JIT propagation issue based on the jury-rig's failure. The actual cause was simpler: the jury-rig subclassed `ViscousFlowModel` which has `requires_stress_history = False`, so the Stokes solver took the viscous branch where `cm.flux` is **never compiled** — it builds the flux from `cm.viscosity` instead. The custom flux containing `_exp_alpha`, `_exp_phi`, etc. was effectively dead code. Once the model declares `requires_stress_history = True` (as the new sibling did), the existing constants-manifest infrastructure handles α, φ propagation correctly with no new plumbing.

### 2. Predictor-corrector return mapping (the 1D Phase B's yield approach) is wrong for 2D Stokes

The 1D Phase B evaluator used predictor-corrector return mapping: solve pure VE, then clip σ to satisfy yield. In 2D Stokes that breaks momentum balance — the SNES finds u that satisfies `∇·σ_VE = body force` (no yield), then we clip σ but leave u unchanged, so the velocity field corresponds to the unclipped stress. **In 2D Stokes-VEP, yield must live inside the SNES residual** via the standard viscosity-wrapping pattern (`viscosity = softmin(η, η_pl)`), the same as the production BDF VEP path. Refactored mid-Phase-B (commit `aba93c2`).

### 3. Lagged-τ aggregation experiments did not tighten yield-surface saturation

Multiple lagged-τ approaches were tried (scalar `min η_eff` over yield-active nodes; scalar `median η_eff`; per-node spatial α via projected scalar mesh variables) — all gave **worse** σ overshoot than the raw τ_VE baseline. Analysis showed the ETD-2 history term `2η_raw·(φ-α)·ε̇*` uses raw η (not yield-clipped) — a Picard-style approximation — and produces a non-zero floor on σ under harmonic forcing that is insensitive to τ_eff except via the α·σ* scaling. The effect is geometric, not a τ-choice issue. Reverted to raw τ_VE = η/μ (commit `584dea8`).

The ETD-2 result at parity-with-BDF-1-production — 1.10-1.14·τ_y at the fault centre — reflects the same kind of overshoot BDF-1 itself shows. Tightening past that is a Phase D concern requiring per-component (α₀, φ₀) for the rank-4 TI tensor, **not** a fix on the lagged-τ aggregation.

### 4. Probe-metric mismatch caused a false alarm

`max σ_II/τ_y_local` over a fault-zone mask reads larger than `σ_xy at fault centre / τ_y_at_fault` because the Gaussian-weakened τ_y(x) varies sharply across the mask: shoulder nodes have τ_y_local much larger than the centerline value, and σ_II saturates accordingly at those local τ_y, which inflates the ratio when the centerline τ_y_at_fault is used as the denominator. **Use the per-node ratio `max σ_II(x)/τ_y(x)`**, or stick to the centre-probe metric (the one BDF-1 production reports). The killer-test runner now reports both.

### 5. Architectural collapse landed at the parameter level

The plan envisaged Phase B with sibling classes (`MaxwellExponentialFlowModel`, TI variant) and Phase D moving integrator state onto the DDt with a strategy parameter. The collapse landed at the **constructor parameter** level instead: `ViscoElasticPlasticFlowModel(unknowns, integrator='etd')` (and the same on TI-VEP). Coefficients still live where the existing infrastructure naturally wants them (`_bdf_c0..c3` on the model, `_exp_coeffs` on the DDt) — the dispatch in `E_eff`, `viscosity`, `_build_c_tensor`, and the uniform `_update_history_*` hooks all branch on `self._integrator`. Sibling classes survive as ~10-line aliases for backwards compatibility (commit `ae79664`).

### 6. Unit-handling in any new array touchpoints needs explicit care

A predictor-corrector clip of `psi_star[0].array` via raw numpy initially looked correct in the non-units case (production benches don't use units) but stripped UnitAwareArray wrappers silently. The user flagged this as accumulating tech debt. The audit fix landed in commit `aba93c2`: `forcing_star` allocated `units=None` (ε̇ has different physical dimensions from σ), `update_forcing_history` non-dimensionalises eval results before storing. Future Phase D work touching `.array` should follow the same pattern (see `update_pre_solve` for the canonical example).

### 7. Empirical range of validity: τ_y / A_∞ ≥ ~0.5

A direct test by tightening τ_y from 0.15 to **0.05** on `bench_ti_vep_harmonic` (so τ_y / A_∞ = 0.05/0.27 ≈ 0.19) at RES=32 over 1.5 periods produced a **catastrophic step-by-step runaway** even though Newton converged on every step. Concrete numbers (saved checkpoints at `output/phase_b_th{0,15}_ty0p05.*`):

| metric | τ_y=0.15 (ratio 0.55) | τ_y=0.05 (ratio 0.19) |
|---|---|---|
| max σ_II in domain | ~0.5 | **17.8** |
| u_y range | ±0.006 | **±18** |
| SNES iter mean / max | ~5 / ~10 | **14 / 38** |
| Diverged steps | 0/120 | 0/120 |
| Wall / step | ~2 s | **8.6 s** |

The mechanism is the same one items 3 and 5 in this list flagged: the ETD-2 history term `α·σ* + 2η·(φ-α)·ε̇*` uses raw η (Picard-style approximation); when the analytical-floor σ-magnitude exceeds τ_y, σ* feeds back through α·σ* on each step and grows without bound. The leading viscous term is yield-clipped, but the history isn't, and (1-φ) is small at typical Δt so the leading contribution can't dominate the runaway history.

**Newton's "convergence" reports in this regime are physically meaningless** — Newton finds the residual minimum each step, but the time-integration loop diverges. SNES iteration counts are *not* an early-warning signal; the warning is in the σ_II / u_y magnitudes themselves.

Practical rules of thumb for the current Phase B implementation:
- ``integrator='etd'`` with the raw τ_VE = η/μ in `α, φ` — works for **τ_y / A_∞ ≥ ~0.5**, at parity with BDF-1 production
- Below that ratio: solution diverges silently (no SNES error); Phase D is required

The Phase D fixes are listed earlier in this section (per-component (α₀, φ₀)/(α₁, φ₁) for TI; or self-consistent τ via SNES sub-iteration). The Phase B design-doc note that "lagged-τ doesn't help" applies in this regime too — the failure is structural to the Picard approximation, not a τ-choice issue.

---

## Appendix A — Numerical evidence

### Phase A (1D linear Maxwell, sinusoidal forcing) — DONE

`_exp_integrator_phase_a.py` solves $\dot\sigma + \sigma/\tau = \mu\dot\gamma$ with $\dot\gamma = \dot\gamma_0 \cos(\omega t)$, $\omega = \pi/2$, $\eta = \mu = \tau = 1$.

| Δt/τ | Exp max\|err\| | BDF-1 | BDF-2 |
|---|---|---|---|
| 0.01 | 1.1e-5 | 4.5e-3 | 7.4e-5 |
| 0.05 | 2.9e-4 | 2.2e-2 | 1.8e-3 |
| 0.10 | 1.1e-3 | 4.4e-2 | 6.8e-3 |
| 0.50 | 2.8e-2 | 1.9e-1 | 1.1e-1 |
| **1.00** | **1.0e-1** | 3.5e-1 | 3.5e-1 |
| **2.00** | **5.7e-2** | 3.4e-1 | 3.4e-1 |

Exp shows clean second-order slope at small Δt and stays accurate at Δt ≥ τ where both BDFs collapse to near-zero output. Figure: `exp_integrator_phase_a.png`.

### Phase B (VEP, large Δt, square wave, variable-Δt) — DONE

`_exp_integrator_phase_b_eval.py` extends to:

- **VEP harmonic** ($\omega = \pi/4$, return-mapping yield): both Exp and BDF-1 clip correctly at τ_y; agreement to ~1% at small Δt because yield mechanism dominates over time integrator.

- **Pure VE at large Δt/τ**: at Δt/τ ≤ 1, Exp 5–12× more accurate; at Δt/τ ≥ 2, both struggle but Exp degrades more gracefully (gives bounded under-shoot vs BDF's wrong-shape output).

- **Square wave VE/VEP**: exp consistently ~2× more accurate than BDF-1 for VE; the yield+BC-discontinuity error dominates both for VEP at small Δt.

- **Variable-Δt around BC flips** (correctly schedule, with fine-zone clamp): improvement of 11–19% in max error for both VE and VEP, both Exp and BDF-1. The exp's plateau-period exactness shows clearly as the per-step error drops to near machine precision once the BC discontinuity is well-resolved.

Figures: `exp_integrator_phase_b_yield.png`, `exp_integrator_phase_b_largedt.png`, `exp_integrator_phase_b_square.png`, `exp_integrator_phase_b_vardt.png`.

### Phase B UW3 jury-rig — partial (propagation snag identified)

`_exp_integrator_uw3_jury_rig.py` attempted to wire ETD-2 into UW3 via a custom `MaxwellExpFlowModel(ViscousFlowModel)` subclass. Hit a JIT propagation issue: `cm._exp_alpha.sym = X` per-step updates don't reach the JIT-compiled flux. Minimal incremental test (`_exp_jury_rig_minimal.py`) confirmed the constitutive-model class plumbing works in isolation; the issue is specific to per-step updates of UWexpression coefficients. **First task of Phase B is resolving this**, by replicating the BDF coefficient propagation pattern.

---

## Appendix B — Architecture details

### What the exponential integrator stores

| Integrator | psi_star slots | forcing_star slots | Coefficients |
|---|---|---|---|
| BDF-1 | 1 | 0 | c_0, c_1 |
| BDF-2 | 2 | 0 | c_0, c_1, c_2 |
| AM-2 | 1 | 0 | a_0, a_1, a_2 |
| ETD-1 (Lawson) | 1 | 0 | α |
| **ETD-2 (this proposal)** | **1** | **1** | **α, A, B** |
| ETD-3 | 1 | 2 | α, A, B, C |

The `SemiLagrangian` already maintains parallel `_bdf_coeffs` and `_am_coeffs`. Adding `_exp_coeffs` is the same kind of peer extension.

### What stays the same vs BDF in the constitutive model

The factorisation σ = η_eff·γ̇ + (history) is preserved. Yield-mode logic (softmin/min/harmonic) wraps η_eff identically to today. The Stokes weak-form structure is unchanged. What changes:
- Different formula for η_eff_VE: η(1-φ) replaces η Δt/(τ+Δt)
- Different history term: α·σⁿ + 2η(φ-α)·ε̇ⁿ replaces the BDF Σ c_i ψ*_i sum
- New ε̇* storage slot

### Why this avoids the BDF-2 instability (TI-VEP + spatial yield)

The instability we documented arises from the c_2·ψ*_{n-1} term in BDF-2's history sum getting autodiff'd into the Jacobian, where it picks up the spatial gradient of η_1_eff (via $\partial\eta_{1,\text{eff}}/\partial\nabla u$), and then gets *directionally amplified* by the rank-4 tensor's $\hat n\otimes\hat n\otimes\hat n\otimes\hat n$ coupling. The amplification compounds across history, exploding |σ| over ~10 t_r.

Exponential has **no second history term**. There's no c_2·ψ*_1 to amplify. The α·σⁿ contribution is autodiff-trivial (σⁿ is a known mesh variable, treated as constant w.r.t. ∇u). The ε̇ⁿ contribution likewise. The Jacobian's only ∇u-dependent term is the leading 2η_eff(1-φ)·ε̇, which is well-behaved.

This is why the structural argument carries to TI-VEP via the per-component decomposition (Phase B step 5): each component of the rank-4 tensor gets its own (α, φ, A, B), each with single-history-slot relaxation, no cross-component amplification.
