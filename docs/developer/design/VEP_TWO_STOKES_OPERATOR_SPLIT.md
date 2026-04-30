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

## Follow-up investigation v3/v4 (2026-04-30)

Building on the v1/v2 post-mortem above, two further architectures were tested as minimal modifications of the existing UW3 BDF-1 VEP solver. The user's framing for these was: keep "second-solve is the truth" reliability of standard BDF-1 VEP, but stabilise the relaxation by taking the elastic part out of the SNES iteration.

### v3 (η-lag injection)

Single-Stokes BDF-1 VEP with `shear_viscosity_0 = η_lag.sym[0,0]` instead of constant — a persistent lagged meshvar updated each step from `min(η_VE, σ_y/(2|ε̇|))` at converged state, floored at `η_VE/10` to keep viscosity contrast tractable.

| Variant | Source of η_lag for next step | Result |
| --- | --- | --- |
| baseline | n/a (constant η_VE) | 120 steps, σ_max 1.456, mean 13 SNES iters |
| `lag` | prior step's converged \|ε̇\| | tracks baseline σ closely; SNES line-search failures from step ~35 absorbed by `snes_force_iteration` |
| `predictor` | current step's Stage-1 (pure-VE) \|ε̇\| | abandoned mid-development, similar behaviour expected |

The η-lag injection didn't actually decouple α from SNES, because in the existing UW3 flux structure `flux = 2·viscosity·E_eff = 2·viscosity·E + (viscosity/(μΔt))·σ_star`, the σ_star coefficient `viscosity/(μΔt)` IS the within-SNES yield-aware viscosity divided by μΔt. Even with `shear_viscosity_0 = η_lag` (which makes the *unclipped* ve_eff spatial), the softmin yield-clip in the residual still enters that coefficient. So we were just adding spatial variation to the SNES residual without actually pulling α out of it — explaining why lag is *less* stable than baseline rather than more.

### v4 (explicit elastic + visco-plastic solver)

Genuine α decoupling via the trick of forcing `psi_star = 0` at the start of each step (so the model's internal history term contributes zero) plus an external body force `−∇·(α·σ_hist)` where σ_hist is independently managed and α is precomputed at start of step:

| Variant | α | Result |
| --- | --- | --- |
| `v4_const_alpha`  | constant `η_VE/(η_VE+μΔt) = 0.952`     | runaway at step 16, σ → 7+ |
| `v4_lagged_alpha` | spatial `η_lag(x)/(η_lag(x)+μΔt)`       | runaway at step 35, σ → 12 |

Lagged-α survived longer than const-α (matching the physics that yielded zones should relax faster), but neither completed the run. The failure mechanism is the same yield-boundary discontinuity that killed v1/v2: even with smooth softmin σ_hist, the spatial gradient of `α(x)·σ_hist(x)` is steep across the yield boundary, and `−∇·(α·σ_hist)` becomes a localised body-force spike that drives `v_pl` overshoot when σ becomes large mid-cycle. Pulling α out of the SNES iteration via body-force redirection introduces its own destabilising spatial structure.

### v3/v4 conclusion

Both follow-up architectures are **less stable than baseline** on this isotropic VEP harmonic test. The puzzle ("decoupling α should be more stable, why isn't it?"): pulling α out via body force trades smooth-in-residual coupling for sharp-at-yield-boundary external forcing. Baseline keeps α and σ_old multiplied by the *same* yield-aware viscosity inside the residual, which is what keeps the implicit body-force-equivalent term smooth as the yield boundary moves.

The cleaner remaining option for true α decoupling is a **custom constitutive class** that has the flux as `2·η_yield_aware·E + α_const·σ_star` directly — no body-force redirection, no `psi_star=0` trickery. That requires modifying `ViscoElasticPlasticFlowModel` itself rather than wrapping it externally; substantial UW3 dev work, deferred.

**Updated net assessment (after v3/v4)**: ETD-1 + yield-in-residual softmin (PR #161) remains the production answer. The body-force decoupling family of architectures (v1, v2, v4) is structurally rejected. The η-lag injection family (v3) tracks baseline but adds no observable benefit on this problem. A custom constitutive class for true α decoupling is the only untested path — see v5 below.

## v5 — Custom constitutive (the working architecture, 2026-04-30)

The user's framing closing the v3/v4 work: implement the last-chance variant via a new constitutive law.

### Architecture

`ViscoPlasticExplicitElastic` subclasses `ViscoElasticPlasticFlowModel` and overrides `stress()`:

```
flux = 2·viscosity·E + α_explicit·σ_star
       └ in SNES residual ┘   └ frozen at start of step ┘
```

vs the baseline UW3 BDF-1 VEP flux:

```
flux = 2·viscosity·E_eff = 2·viscosity·E + (viscosity/(μΔt))·σ_star
                                            └ iterated within SNES ┘
```

The σ_star coefficient is now `α_explicit` (a precomputed sympy expression — scalar or meshvar — not a property of the SNES iterate). The yield-aware `viscosity` still iterates within SNES, but only multiplies the active strain-rate term, not the history term. This is "second-solve is the truth, with the relaxation stabilised."

Two flavours of `α_explicit`:
- **const**: `α = η_VE/(η_VE + μΔt)` — uniform, true decoupling
- **lagged**: `α(x) = η_lag(x)/(η_lag(x) + μΔt)` — preserves the physics of "yielded zones relax faster" but freezes that spatial structure at start of step

### Result

| Variant | Steps | σ_max | Peak yielded | Mean SNES | Wall (s) |
| --- | --- | --- | --- | --- | --- |
| BDF-1 baseline           | 120 | 1.456 | 9.09% | **13.2** | 854 |
| v5 const-α               | 120 | 0.681 | 9.73% |  1.7  | 518 |
| v5 lagged-α              | 120 | 0.159 | 6.34% |  1.5  | 2240⁺ |

⁺ Wall time inflated by per-step `uw.function.evaluate(cm.viscosity, eta_coords)` projection to refresh η_lag — easy optimisation by computing η_lag from |ε̇| numpy-side instead.

**The architecture works.** Both variants run the full 1.5 forcing periods (120 steps) without a single SNES failure. Mean SNES iteration count drops from baseline's **13.2 to 1.5–1.7** — ~8× speed-up on the nonlinear solve, exactly the conditioning improvement predicted by pulling α out of the iteration.

### Trajectory differences from baseline

The three architectures give three different physics curves on this test:

* **Baseline**: σ_star coefficient is the iterated yield-aware viscosity. Failed zones have *both* small viscosity (yield clip) AND consequently small σ_star coefficient — i.e., yield indirectly accelerates relaxation through the same coupling that causes the SNES conditioning headache. Net effect: σ peaks at 1.46 in steady state.
* **v5 const-α**: α uniform 0.952 everywhere; failed zones don't relax faster than elastic zones. Stress retention is uniformly slow. σ peaks at 0.68 (cleaner periodic).
* **v5 lagged-α**: α(x) drops to ~0.1 in failed zones (matching baseline's qualitative behaviour) but spatially frozen for the step. σ peaks at 0.16 — smallest residual stress, cleanest limit cycle.

For physics comparison the lagged-α variant is closest to baseline's *intent* (failed zones relax faster) without the within-SNES coupling. The const-α variant is the cleanest "fully decoupled" reference.

### Architectural note: `frozen_flux` as a generalisation

User's suggestion (after seeing v5 work): make `frozen_flux` a property of the base `Constitutive_Model` — a tensor expression added verbatim to the residual flux, not iterated. The `α_explicit·σ_star` term in v5 is one specific use; other use cases include explicit elastic predictors, prescribed stress sources, and any scenario where part of the constitutive contribution is known up-front and shouldn't enter the nonlinear iteration. If v5 ships, `frozen_flux` is the natural API to land alongside it.

### Outcome

| Architecture | Result |
| --- | --- |
| v1, v2 (two-Stokes with body force = −∇·σ_VE)            | rejected — yield-boundary discontinuity feedback |
| v3 (η-lag injection into shear_viscosity_0)              | tracks baseline, no benefit (doesn't actually decouple α) |
| v4 (psi_star=0 + body force = −∇·(α·σ_hist))             | rejected — same boundary-discontinuity pathology in α-gradient |
| **v5 (custom constitutive: flux = 2·η·E + α·σ_star)**    | **works** — 8× SNES speed-up, stable through full run |

**Final net assessment**: v5 is the working architecture for "explicit elastic + visco-plastic solver." It produces a different physics trajectory than baseline (because α decoupling IS a physics change, not just a numerical reorganisation) but is dramatically better-conditioned. Whether v5 ships as a UW3 production option depends on validation against physics expectations on a wider range of problems; this branch establishes the architecture is viable and identifies `frozen_flux` as the natural generalisation pattern.

### v5 follow-up (2026-04-30 evening) — α-formula bug + structural lagged-α failure

After the initial v5 result, instrumented re-runs with multi-step spatial checkpoints (steps 30/60/90/120) and a lower η_lag floor (1e-3 instead of 1e-1) revealed two issues with the lagged-α variant:

1. **α-formula bug (resolved)**: The original v5_lagged-α used `α = η_lag/(η_lag + μΔt)`, treating η_lag as raw elastic viscosity. But η_lag stores `cm.viscosity` which is the *post-BDF-reduction* effective viscosity. The correct match to baseline's σ_star coefficient (`viscosity/(μΔt)`) is `α = η_lag/(μΔt)`. With the wrong formula, bulk α was 0.488 instead of 0.952 — half-retention explained the 13× σ-amplitude shortfall.

2. **Structural failure of lagged-α (unresolvable)**: With the corrected formula and proper η_lag floor (so η_lag genuinely varies spatially as cm.viscosity does), v5_lagged-α runs away at step 8. The runaway is the same yield-boundary discontinuity that killed v4: with α(x) varying sharply across the yield boundary, the `α(x)·σ_star` term in the flux acts as a sharp-gradient stress source that drives v_pl overshoot. Mathematically equivalent to v4's body force `−∇·(α·σ_hist)` (just by divergence theorem) — moving it inside the flux doesn't smooth the gradient. The first stable lagged-α run was a numerical accident: the high floor (0.1 > cm.viscosity_max ≈ 0.0476) clamped η_lag to a uniform 0.1 everywhere, making it effectively const-α at α=0.667. Once the floor allows real spatial variation, the architecture collapses.

**Why baseline tolerates spatial α variation but v5_lagged doesn't**: Baseline's `viscosity/(μΔt)` is *iterated within SNES*. As Newton refines the velocity, viscosity at every quad point updates to the current state — yield-zone viscosity, fault-tip stress concentration, etc. all reach self-consistency. v5_lagged freezes α(x) from the prior step, so within the current step the SNES sees a fixed-spatial-pattern coefficient that may not match the current state's natural pattern. The mismatch drives instability when the loading reverses or yields shift.

### Final v5 conclusions

| Architecture | Result | Physics |
| --- | --- | --- |
| v5 const-α (uniform α=0.952)         | works, 8× SNES win, 120 steps clean | **different** from baseline: no plastic acceleration of relaxation in failed zones |
| v5 lagged-α with high floor (~0.1)   | works, but η_lag floor pinned uniform → effectively const-α at smaller α | not genuinely lagged |
| v5 lagged-α with low floor (~1e-3)   | runaway at step 8 | structural — equivalent to v4 pathology |

The "true α decoupling matching baseline physics" remains structurally unattainable through frozen-α architectures. baseline's α-yield coupling **must** iterate inside SNES to remain stable when yield-zones move spatially. The v5 const-α architecture is a viable production option **if** you accept the physics change (no plastic acceleration of relaxation in yielded zones) — appropriate for problems where this is qualitatively acceptable, gives an 8× SNES speed-up.

The `frozen_flux` generalisation idea remains valid for *uniform* frozen contributions (predictors, prescribed sources, constant-α terms). Spatially-varying frozen contributions need to be applied with care because of the discontinuity-amplification mechanism.

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
