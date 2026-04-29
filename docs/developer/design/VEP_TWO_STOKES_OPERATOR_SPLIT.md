# VEP Two-Stokes Operator Split — Investigation Plan

> **Status**: planned, not implemented (2026-04-29). New investigation branching off the ETD integrator work (PR #161). Captures architectural context while it's fresh; first session of implementation will likely build the second-stage solver and run a comparison vs ETD-1 + in-residual yield (the production path that ships in PR #161).

## Motivation

The exponential-integrator investigation (Phase A–F, see `EXPONENTIAL_VE_INTEGRATOR.md`) eliminated several VE+plasticity strategies and converged on **ETD-1 + yield-in-residual softmin** as the production answer. That works because:

1. ETD-1 is L-stable (advice §13 in this doc → `EXPONENTIAL_VE_INTEGRATOR.md` lesson #13).
2. The in-residual softmin yield is a coupled Newton solve where σ and u find each other through the residual.

But the in-residual softmin has known imperfections:
* **Yield surface saturation** is approximate (softmin with finite δ allows σ to drift above τ_y, especially under variable Δt — see project memory `project_vep_variable_dt_yield_violation.md`).
* **Higher-order ETD-2** beats BDF-2 by 4× on smooth VE but blows up under in-residual yield in tight-yield regimes (lesson #11 — fundamental, not patchable). For *fully-VE problems* ETD-2 is shippable; for *VEP*, only ETD-1 is.
* **No clean predictor-corrector path**: the radial-return-after-VE-predictor pattern (Phase F) only worked with ETD-1, and adding it to ETD-2 didn't rescue ETD-2 (lesson: σ-damping in outer Picard isn't sufficient — need to update η_eff between iters too).

The web-advice doc (`/Users/lmoresi/Downloads/vep_stress_update_full_latex.md`) prescribes the architecture that *is* the standard robust pattern in production geodynamics codes:

```
Stokes solve with lagged viscosity → VE exponential predictor → plastic return mapping → damped outer Picard iteration
```

What we built in Phase F was the **single-Stokes** version of this: one Stokes solve per Picard iter, σ damped between iters, η_eff fixed at η_VE. That's not the full architecture. The advice (and the user's framing in the conversation) is a **two-Stokes** operator split:

> **Stage 1**: VE Stokes solve — find v, p, σ_VE assuming pure viscoelastic.
> **Stage 2**: plasticity Stokes solve where σ_VE is *fully explicit* (a known stress source) and viscosity plays the role of the plastic multiplier.

Adding the second momentum-balanced solve is what's missing. With it:
* **ETD-2 might become viable for VEP+yield** because the second Stokes equilibrates the velocity field with the corrected stress field via a proper momentum solve, not just σ damping.
* **Pointwise plastic correction is exact** for J2 (closed-form radial return). The second Stokes restores momentum balance globally given the corrected stress field.
* **Anisotropic case (TI-VEP)** becomes a clean extension: stage 1 uses TI VE; stage 2 uses J2 (or Drucker–Prager) on the resolved fault-shear with the same momentum-balanced equilibration.

## Architecture

### Stage 1 (VE predictor)

Standard Stokes with the existing `ViscoElasticPlasticFlowModel`:
* `integrator='etd', order=2` (or `order=1`)
* `yield_stress = ∞` (no in-residual yield)
* solves `∇·σ_VE - body_force - ∇p_VE = 0`

After solve: `psi_star.array` holds σ_VE (the VE trial stress, the predictor's output).

### Stage 2 (plasticity corrector — the new piece)

A separate Stokes-like solver with:

* **Constitutive**: pure viscous, `σ_pl = 2η_pl(x)·ε̇(v_pl)`. The `ViscousFlowModel` works for this; `Parameters.shear_viscosity_0` is set to a meshvar field.
* **Body force**: `−∇·σ_VE` so the total stress satisfies momentum: `∇·(σ_VE + 2η_pl·ε̇(v_pl)) = body_force_external`.
  * Computed symbolically from `psi_star.sym` — UW3 supports `mesh.vector.divergence(rank-2 sym)` or equivalent.
  * The bodyforce expression goes into `stokes_pl.bodyforce`.
* **Effective plastic viscosity** `η_pl(x)`:
  * **Interpretation (a) — coupled solve**: `η_pl` is determined implicitly so that `|σ_total|_eq ≤ σ_y` everywhere with equality where yielded. Solver iterates Newton on this; `η_pl` is a non-linear function of v_pl. This is the rigorous form, equivalent to a yield-aware viscosity in stage 2.
  * **Interpretation (b) — explicit**: compute `η_pl` from stage-1 data: `η_pl = σ_y/(2|γ̇_VE|)` where `|σ_VE|_eq > σ_y`, large value otherwise. Stage 2 is a *linear* viscous solve. Outer Picard iterates over (a) or (b) blend until consistent.

Phase F tested neither (a) nor (b) properly — it iterated σ via single-Stokes Picard with `η_eff = η_VE` fixed, which doesn't add momentum-balanced equilibration. The first session of this investigation should test (b) as the simpler scaffold; (a) requires either a yield-aware constitutive law for `ViscousFlowModel` or Newton-iterating the linear-viscous-with-spatial-η problem.

After Stage 2: corrected velocity `v` and stress `σ_total = σ_VE + 2η_pl·ε̇(v_pl)` (or `v_total = v_VE + v_pl` depending on framing).

### Outer iteration

Wrap the two stages in a Picard loop with η damping (advice §9, ω_η ≈ 0.3) and σ damping (§10, ω_τ ≈ 0.5):

```
for k = 1, 2, ...:
    Stage 1 with η_eff_k from previous iter        →  σ_VE^k
    Compute η_pl^k from σ_VE^k (interpretation b)   or
    Stage 2 (linear or nonlinear) with η_pl^k       →  v_pl^k, p_pl^k
    Damp:
        σ ← (1-ω_τ)·σ_old + ω_τ·σ_total^k
        η ← (1-ω_η)·η_old + ω_η·η_pl^k
    Convergence check: ||σ_k - σ_{k-1}|| / ||σ_k|| < tol
```

Within a single timestep. After convergence, `psi_star ← σ_total` (or σ_VE — design choice).

## Implementation challenges in UW3

1. **Two Stokes objects on the same mesh.** Each wants its own velocity/pressure meshvar. Currently the codebase has one Stokes per setup; we'd build a separate `Stokes(mesh, velocityField=v_pl, pressureField=p_pl)` and configure its constitutive model independently.

2. **Spatial η_pl as a meshvar in the constitutive law.** `ViscousFlowModel.Parameters.shear_viscosity_0 = eta_pl_field.sym` should work — model uses the symbolic expression. Need to verify the JIT path handles a meshvar reference correctly (it does for σ_y already).

3. **Body force = −∇·σ_VE**. `psi_star` is a SYM_TENSOR meshvar. Its symbolic divergence is `sympy.diff(psi_star.sym[i, j], coord_j)` summed over j. The bodyforce expression assembles to a vector. Cost: per-quadrature-point evaluation of three `sympy.diff` expressions on a tensor field — should be cheap.

4. **What stays in psi_star at end of step.** Three options:
   * `σ_total = σ_VE + 2η_pl·ε̇(v_pl)` — full corrected stress. Best for next-step ETD history term `α·σⁿ`.
   * `σ_VE` — the VE-only stress. Cleaner separation but loses plasticity history.
   * Apply final return-mapping cleanup on `σ_VE + 2η_pl·ε̇(v_pl)` to enforce yield exactly.

5. **Velocity composition.** Does `v_pl` *replace* v_VE or *add* to it? If stage 2 has body force `−∇·σ_VE`, the `v_pl` from stage 2 is the velocity that, combined with `σ_VE` as "baseline stress", balances momentum. So `v_pl` IS the corrected velocity field at end of step (not v_VE + v_pl). For Lagrangian advection, use `v_pl`.

6. **Boundary conditions.** Stage 2 inherits the same BCs as stage 1 (kinematic boundary conditions on v_pl). The VE stress σ_VE on the boundary is consistent with v_VE; stage 2 finds v_pl satisfying BCs and the plastic-balanced momentum.

## Validation plan

Reuse the Phase F harness (isotropic VEP, localised weak zone, harmonic loading). Compare:

* **BDF-1 yield-in-residual** — production reference (already validated).
* **ETD-1 + two-Stokes operator split (interpretation b)** — does it match BDF-1?
* **ETD-2 + two-Stokes operator split** — does the second momentum solve rescue ETD-2 from the drift seen in Phase F? *This is the headline test.*

If the answer is yes for ETD-2: we have a robust path to the higher-accuracy integrator for VEP+yield, and the path to TI-VEP fault mechanics is open.

If no: the residual drift mechanism is structural beyond two-Stokes equilibration, and ETD-1 + softmin yield-in-residual remains the right answer.

## Investigation outcome (2026-04-29)

The first-cut implementation in `_phase_g_two_stokes.py` is shippable as a study tool but the architecture as proposed **does not work** for sustained yielding. Across all four configurations:

| Case               | Stage-1 integrator | Inner Picard | Result                        |
| ------------------ | ------------------ | ------------ | ----------------------------- |
| `bdf1_pic1`        | BDF-1              | 1 (no Picard)| Runaway at step 39 (σ → 133)  |
| `bdf1_pic6`        | BDF-1              | 6, ω_τ=0.5   | Runaway at step 4 (σ → 144)   |
| `etd1_pic6`        | ETD-1              | 6, ω_τ=0.5   | Runaway at step 4 (σ → 145)   |
| `etd2_pic6`        | ETD-2              | 6, ω_τ=0.5   | Runaway at step 4 (σ → 128)   |

(Test setup: 1.5 periods of harmonic shear, V₀=0.5, ω=π/2, RES=32, fault layer τ_y=0.05 vs bulk τ_y=200, θ=15°.)

Even single-shot BDF-1 fails. The blow-up pattern is identical across all cases: σ_eq grows ~2.3× per step in the late phase of the second loading half-cycle (steps 35–39 for single-shot, steps 1–4 for inner Picard).

### Failure mechanism — feedback loop on the yield-boundary discontinuity

1. Stage 1 (yield_stress=∞) inherits `psi_star = σ_admissible` from the previous step. `σ_admissible` is **discontinuous** at the yield-zone boundary because the J2 radial return is a hard projection (σ → (σ_y/|σ|)·σ in yielded cells, identity elsewhere).
2. The VE update `σ_VE = α·σ_admissible + 2η_eff(1-α)·ε̇_VE` propagates this discontinuity into σ_VE.
3. Stage 2 body force `−∇·σ_VE` therefore has a near-delta-function source at the yield boundary. With η_frozen ≈ 0.048, the `v_pl` correction has gradients ~σ_jump / (η_frozen·h) ≈ thousands.
4. `σ_total = σ_VE + 2η·ε̇(v_pl)` overshoots σ_y far more than σ_VE alone did. Radial return clips harder. The clip-induced jump in `σ_admissible` is now larger.
5. Goto step 1. Each step amplifies the discontinuity-driven correction. After ~30 steps the bulk hasn't yielded but the fault-boundary discontinuity dominates the response.

### Why inner Picard makes it worse

With max_picard=6 and ω_τ=0.5, each iteration solves stage 2 against an increasingly clipped `σ_body`. Each stage-2 solve still sees the discontinuity at the yield boundary (clipped vs unclipped), but now with body force = −∇·(σ_body damped between σ_VE and σ_admissible). The damping does *not* smooth the jump — it just averages across iterates with the same jump location. v_pl stays large; `σ_total = σ_body + 2η·ε̇(v_pl)` keeps growing per inner iteration. Six iterations of growth is enough to blow up immediately.

### The damping bug fix (kept, no longer relevant on its own)

The Picard implementation initially wrote the damped `σ_body = (1-ω)·σ_VE + ω·σ_admissible` as the new `psi_star` at the end of the timestep. That was wrong: only `σ_admissible` is yield-admissible. The fix (write `σ_admissible`) restored single-shot byte-identical to 3af9751 (which also fails at step 39, identical numbers). The damping bug is now fixed but the underlying architecture remains broken.

### Architectural conclusion — what's actually needed

The two-Stokes operator split with `bodyforce = −∇·σ_VE` is provably wrong as posed:

* Either σ_VE is unclipped (the predictor) — then it overshoots σ_y arbitrarily far in yielded cells, and `−∇·σ_VE` is enormous;
* Or σ_VE is clipped (`σ_admissible`) — then it's discontinuous and `∇·σ_admissible` has a delta source at the yield boundary.

Both fail. The "correct" stress to put on the right-hand side is one that is *both* yield-admissible *and* smooth, which only happens if it's a regularised yield surface (softmin) — which is exactly the in-residual softmin that ETD-1 + yield-in-residual already does in PR #161.

Three candidate paths forward, in increasing order of departure from the current design:

1. **Replace `−∇·σ_admissible` body force with a smoothing**: project `σ_admissible` onto a continuous P1 mesh variable before differentiating. Smooths the jump but adds spatial diffusion of stress; not obviously consistent.
2. **Spatially-varying η_pl (interpretation b)**: stage 2 has no σ_VE body force at all. Instead, η_pl(x) = σ_y/(2|γ̇_VE|) where σ_VE > σ_y, large value otherwise. Stage 2 is a pure viscous Stokes with non-uniform viscosity. The yield is enforced through the constitutive law, not through a body force. This is much closer to the standard production geodynamics pattern. Untested in this branch.
3. **Augmented Lagrangian / Uzawa for the plastic constraint**: introduce a Lagrange multiplier for |σ| ≤ σ_y; iterate momentum balance and constraint together. This is the rigorous equivalent of (a) in §2 of this doc.

**Net assessment**: ETD-1 + yield-in-residual softmin (the PR #161 production answer) remains the best available path. The two-Stokes architecture as proposed in this document is rejected. Interpretation (b) — yield-aware viscosity in stage 2 with no σ_VE body force — is the natural next investigation if anyone wants to revisit; it may also be the architecture the web-advice doc actually intended.

## Code organisation

Suggested new files (on a branch off `development` after PR #161 merges):

```
docs/developer/design/_phase_g_two_stokes.py         # runner with stage 1 + stage 2 + outer Picard
docs/developer/design/_plot_phase_g.py               # comparison plot, includes Phase F traces
docs/developer/design/_phase_g_*.trace.txt           # per-step traces
docs/developer/design/VEP_TWO_STOKES_OPERATOR_SPLIT.md  # this document
```

No production-API changes expected unless the architecture proves itself for VEP+yield — in which case the second-stage solver might land as a new helper or a method on `ViscoElasticPlasticFlowModel`.

## Connecting back

The user's framing closing the ETD investigation: *"the radial return, correctly computed as a sequence of solves ... offers the potential for a very robust VEP solver"*. That's exactly what this branch tests. Reference points:

* `EXPONENTIAL_VE_INTEGRATOR.md` lesson #13 — first-order dissipation explanation
* Phase F results — what radial return alone (without two-Stokes) achieves and where it fails
* Web advice `/Users/lmoresi/Downloads/vep_stress_update_full_latex.md` §3, §11, §15 — the canonical predictor-corrector + outer Picard architecture

The two-Stokes investigation is the bridge between the ETD work and a production-quality VEP+yield solver.
