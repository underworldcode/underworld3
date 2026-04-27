# Exponential Integrator for VE / VEP Constitutive Updates

**Status**: Investigation / design (started 2026-04-27)
**Branch**: TBD (started on `bugfix/vep-investigation-fixes` for context, will move)
**Motivation**: planning entry under "Nice to Have" in `~/Library/CloudStorage/Box-Box/Planning/underworld.md`

## Problem statement

The current VE / VEP / TI-VEP constitutive update uses a BDF-k discretisation of the Maxwell ODE

$$\dot\sigma + \sigma/\tau = 2\mu\,\dot\varepsilon$$

with $\tau = \eta/\mu$ the relaxation time. BDF-2 has known issues we documented on `bugfix/vep-investigation-fixes`:

1. The $c_2 \psi^*_{n-1}$ term in the multistep formula gets amplified through the autodiff Jacobian whenever the constitutive viscosity has spatial gradients (TI tensor + spatial $\tau_y$). Result: |σ| → ∞ over ~10 t_r at BDF-2.
2. BDF-1 is robust but only first-order accurate. With our `bdf_blend = 0.10` workaround we get effectively BDF-1 anyway.
3. BDF-style derivative approximations are inherently sensitive when $\Delta t \gg \tau$ (excessive numerical diffusion) and when $\dot\varepsilon$ is noisy from the Stokes solve.

The Maxwell relaxation operator is **linear and analytically integrable**. Discretising the time derivative is unnecessary work — we can integrate exactly and only approximate the forcing.

## Mathematical formulation

### The operator-split form

Treat the Maxwell ODE as a linear inhomogeneous equation. The integrating factor is $e^{t/\tau}$, giving the closed-form solution over a step $[t^n, t^{n+1}]$:

$$\sigma^{n+1} = e^{-\Delta t/\tau}\,\sigma^n
            + 2\mu \int_{t^n}^{t^{n+1}} e^{-(t^{n+1}-s)/\tau}\,\dot\varepsilon(s)\,ds$$

Only $\dot\varepsilon(s)$ inside the integral needs to be approximated.

### Two-point linear quadrature (recommended)

If we approximate $\dot\varepsilon$ as a linear interpolant between $\dot\varepsilon^n$ and $\dot\varepsilon^{n+1}$ over the step, the integral has a closed form:

$$\sigma^{n+1} = \alpha\,\sigma^n + 2\mu\,\bigl(A\,\dot\varepsilon^{n+1} + B\,\dot\varepsilon^n\bigr)$$

with

$$\alpha = e^{-\Delta t/\tau},\quad
\varphi = \frac{1-\alpha}{\Delta t/\tau},\quad
A = \tau(1-\varphi),\quad
B = \tau(\varphi - \alpha).$$

Properties to verify before we trust this:

- **Limits**:
  - $\Delta t/\tau \to 0$: $\alpha \to 1$, $A, B \to \Delta t/2$ → trapezoidal-rule limit
  - $\Delta t/\tau \to \infty$: $\alpha \to 0$, $A \to \tau$, $B \to 0$ → viscous limit ($\sigma \to 2\eta\dot\varepsilon^{n+1}$)
- **Stability**: $|\alpha| = e^{-\Delta t/\tau} < 1$ unconditionally
- **Order**: second-order in $\Delta t$ for smooth $\dot\varepsilon$ (linear interpolant in the integral)
- **Storage**: one history slot ($\sigma^n$) and one ($\dot\varepsilon^n$). No multistep history needed.

### Comparison to BDF

| | BDF-1 | BDF-2 | Exp (this) |
|---|---|---|---|
| Order | 1 | 2 | 2 |
| Stability for $\Delta t \gg \tau$ | OK (over-damped) | OK at α=1 (constant coefficients only) | exact |
| History slots | 1 | 2 | 1 (σ) + 1 (ε̇) |
| Multistep autodiff issue with spatial yield | n/a | yes (this branch) | n/a |

## Implementation approach

### Phase A — 1D linear Maxwell validator (standalone) — DONE

Pure Python, no UW3. `_exp_integrator_phase_a.py`. Solves engineering Maxwell $\dot\sigma + \sigma/\tau = \mu\dot\gamma$ for prescribed $\dot\gamma(t)$ and compares exponential integrator to BDF-1, BDF-2, and the analytical phasor solution.

**Results — sinusoidal forcing** ($\dot\gamma = \dot\gamma_0 \cos(\omega t)$, $\omega = \pi/2$, $\eta = \mu = \tau = 1$, $T = 8\tau$):

| $\Delta t/\tau$ | **Exp max\|err\|** | BDF-1 max | BDF-2 max |
|---|---|---|---|
| 0.01 | **1.1e-5** | 4.5e-3 | 7.4e-5 |
| 0.05 | **2.9e-4** | 2.2e-2 | 1.8e-3 |
| 0.1  | **1.1e-3** | 4.4e-2 | 6.8e-3 |
| 0.25 | **7.1e-3** | 1.1e-1 | 3.7e-2 |
| 0.5  | **2.8e-2** | 1.9e-1 | 1.1e-1 |
| 1.0  | **1.0e-1** | 3.5e-1 | 3.5e-1 |
| 2.0  | **5.7e-2** | 3.4e-1 | 3.4e-1 |

Conclusions:

1. **Exponential is consistently 5–6× more accurate than BDF-2 at small $\Delta t$** and shows clean second-order convergence.
2. **At $\Delta t \geq \tau$, BDF-1 and BDF-2 collapse together to ~full-amplitude error (3.4e-1)** — they over-damp the response so heavily that simulated $\sigma$ stays near zero, regardless of multistep order. The trace plot makes this stark: at $\Delta t/\tau = 1$ both BDFs give a flat near-zero output while the exponential tracks the analytical sinusoid faithfully.
3. **The exponential integrator's error actually *decreases* from $\Delta t/\tau = 1$ to 2** because the linear-interpolant quadrature error is dominated by the trapezoidal-rule limit, which is well-behaved for slow forcing.

**Square-wave** results are less clean because of discontinuity-handling at sign flips. All three integrators have errors of similar order (~0.5–0.8 of full amplitude). Likely needs the same fine-dt-around-flips treatment we already use for `bench_*_vardt`. Not a blocker — this is a feature of *forcing approximation*, not the relaxation operator.

**Decision gate**: cleared. Proceed to Phase B.

Figure: `exp_integrator_phase_a.png` (convergence + trace).

### Phase B numerical evaluation — VEP and large-Δt regimes — DONE

Standalone validator extended (`_exp_integrator_phase_b_eval.py`) to test (i) yield-active VEP with return-mapping clip and (ii) the large-Δt/τ regime where BDF over-damps.

**VEP harmonic** ($\omega = \pi/4$, $\Delta t = 0.1\tau$, A_∞ = 0.79):

| τ_y | regime | Exp peak\|σ\| | BDF-1 peak\|σ\| | ratio |
|---|---|---|---|---|
| 0.10 | yielding | 0.100 | 0.100 | 1.000 |
| 0.20 | yielding | 0.200 | 0.199 | 1.003 |
| 0.30 | yielding | 0.299 | 0.299 | 1.002 |
| 0.50 | yielding | 0.499 | 0.494 | 1.010 |

When *both* integrators use the same return-mapping yield treatment, **they agree to ~1%** at small Δt with active yielding. The yield mechanism dominates over the time integrator. Implication: in the VEP regime where dt is small relative to τ_VE, switching to exponential doesn't give a *direct* accuracy win — the win is structural (no BDF-2 instability, single history, simpler autodiff path).

**Pure VE at large Δt/τ** (no yield, smooth forcing):

| Δt/τ | Exp max\|err\| | BDF-1 max\|err\| | Notes |
|---|---|---|---|
| 0.5 | 0.010 | 0.126 | Exp 12× better |
| 1.0 | 0.040 | 0.225 | Exp 6× better |
| 2.0 | 0.119 | 0.402 | BDF-1 wrong shape; Exp under-shoots peaks |
| 5.0 | 0.624 | 0.584 | Both bad — quadrature too coarse |

This is the regime that matters for **yield-active fault zones**: η_eff drops to η_pl, τ_eff = η_pl/μ shrinks, and dt/τ_eff can grow into the Δt/τ ≈ 1 regime even when dt/τ_VE in the bulk is small. Exponential continues to give physically meaningful answers; BDF-1 over-damps and BDF-2 hits the documented multistep instability.

Figures: `exp_integrator_phase_b_yield.png`, `exp_integrator_phase_b_largedt.png`.

---

## DDt generalisation — architectural sketch

The current `SemiLagrangian` DDt is BDF-centric: it holds a list `psi_star[0..order-1]` of past stress history slots, exposes a `bdf()` method that computes $\sum c_i \psi^*_i$ symbolically, and updates `_bdf_c0..c3` UWexpressions per timestep via `_update_bdf_coefficients`.

For the exponential integrator we need a **second history field** ($\dot\varepsilon^*$ — strain rate at the previous step) and a **different history-term computation**. Three implementation strategies, in increasing surgical scope:

### Strategy 1 — Subclass: `ExponentialMaxwell_DDt(SemiLagrangian)`

Inherit the snapshot/projection machinery; add ε̇* slot; provide `exp_history()` alongside `bdf()`.

```python
class ExponentialMaxwell_DDt(SemiLagrangian):
    def __init__(self, ..., order=1):  # exp is single-step (order=1 always)
        super().__init__(..., order=1)
        # Parallel slot for strain-rate history
        self._epsdot_star = MeshVariable(...)
        # Time-integration coefficients (UWexpressions, updated per step)
        self._exp_alpha = expression(r"\alpha", sympy.Float(1.0), ...)
        self._exp_phi   = expression(r"\varphi", sympy.Float(1.0), ...)
        self._exp_A     = expression(r"A", sympy.Float(0.0), ...)
        self._exp_B     = expression(r"B", sympy.Float(0.0), ...)

    def update_post_solve(self, dt, ...):
        super().update_post_solve(dt, ...)
        # Project current ε̇ into ε̇* slot (parallel to ψ* projection)
        self._project_epsdot_star()

    def exp_history(self, mu_dt):
        """Returns the symbolic history term: α σⁿ + μ B γ̇ⁿ."""
        return (self._exp_alpha * self.psi_star[0].sym
                + 2 * mu_dt / self._dt * self._exp_B * self._epsdot_star.sym)
```

The constitutive model selects which DDt class to use based on a `time_integrator` parameter. When set to `"exponential"`:
- The DDt is `ExponentialMaxwell_DDt`
- `stress()` builds $\sigma = 2\eta(1-\varphi)\,\dot\varepsilon + \mathrm{exp\_history}$
- `_update_time_coefficients()` computes $\alpha, \varphi, A, B$ from current dt and (lagged) τ

**Pros**: minimal disturbance to existing BDF code paths. Reuses snapshot fix.
**Cons**: introduces a parallel hierarchy; if both BDF and exp must coexist for transition, we have two diverging code paths in the constitutive model.

### Strategy 2 — Refactor `SemiLagrangian` into history-storage + integrator

Separate concerns:

- **`HistoryStorage`** — a list of mesh variables for past states, the projection-snapshot machinery, the SemiLagrangian advection. *No* opinion about how the history is consumed.
- **`TimeIntegrator`** — given a `HistoryStorage` and current state, produces the residual contribution. Subclasses: `BDF_TimeIntegrator(order)`, `ExponentialMaxwell_TimeIntegrator()`.

```python
class TimeIntegrator(ABC):
    @abstractmethod
    def history_term(self, current_state) -> sympy.Expr:
        """Symbolic history contribution to the residual."""
    @abstractmethod
    def update_coefficients(self, dt, **kwargs):
        """Per-step coefficient update."""
    @abstractmethod
    def required_history_slots(self) -> list[VarType]:
        """What state variables to track (stress, strain rate, ...)."""
```

The constitutive model holds a `time_integrator: TimeIntegrator` attribute and asks it for the history term. The DDt instance becomes a thin wrapper that just owns the storage.

**Pros**: clean separation; new integrators (RK, Crank-Nicolson, etc.) drop in without touching constitutive model.
**Cons**: substantial refactor of `ddt.py` and the constitutive-model assignment path. High up-front cost.

### Strategy 3 — Keep BDF DDt, add exp as a special configuration

Treat the exponential integrator as "BDF-1 with α∈(0,1) replaced by exp(-Δt/τ) and the leading viscosity replaced by η(1-φ)". The DDt machinery technically still stores one history slot and provides BDF-1 coefficients; we just plug in different expressions in the constitutive model's stress formula.

**Pros**: zero changes to DDt.
**Cons**: confuses the BDF intent; the ε̇* requirement still needs handling somewhere; coefficient computations in ``_update_bdf_coefficients`` would have to dual-purpose.

### Recommendation

**Start with Strategy 1** (subclass). Lowest cost-to-prototype, validates the formulation in production-like UW3 plumbing before committing to a refactor. If it works and we want to keep both BDF and exponential as first-class options long-term, **promote to Strategy 2** in a separate session — at that point we know which interfaces matter.

Strategy 3 is rejected: too clever, hard to debug.

### Phase B implementation work-plan (concrete)

1. Create `ExponentialMaxwell_DDt` in `systems/ddt.py`. Re-use `SemiLagrangian` projection-snapshot. Add `_epsdot_star` mesh variable + projection. Add `α, φ, A, B` UWexpressions and `_update_exp_coefficients(dt, tau)`.

2. Create `MaxwellExponentialFlowModel` in `constitutive_models.py`, sibling of `ViscoElasticPlasticFlowModel`. `requires_stress_history = True` (re-uses solver auto-DDt path) but with the exp DDt class. Stress:
   ```python
   def stress(self):
       phi, alpha = self._exp_phi, self._exp_alpha
       eta = self.Parameters.shear_viscosity_0
       sigma_h = self.Unknowns.DFDt.exp_history(self.Parameters.dt_elastic *
                                                 self.Parameters.shear_modulus)
       return 2 * eta * (1 - phi) * self.Unknowns.E + sigma_h
   ```

3. Wire VEP via lagged-τ: each step, after solve, compute new $\tau_{\text{eff}} = \eta_{\text{eff}}/\mu$ from the projected stress and update for next step's α, φ.

4. Add `bench_ve_harmonic_exp.py` — run the harmonic benchmark with the new model. Decision gate: must match BDF-2's max\|err\| of 1.34e-3 (or beat it).

5. If (4) passes, run `bench_ti_vep_harmonic_exp.py` — the headline test. Should sidestep the BDF-2 instability entirely.

### Open architectural questions (to resolve during Phase B implementation)

1. **Where does ε̇* live?** A mesh variable is the obvious choice (parallel to ψ*). But it's symmetric (sym-tensor) and needs a SemiLagrangian advection like ψ*. The projection cost roughly doubles vs current BDF.

2. **Per-quadrature-point vs scalar α, φ**? When η is uniform, α and φ are scalars and constants[] handles them cheaply. When η is spatial (yield zones, weakness fields), α, φ become spatial expressions — sympy handles `exp` of a spatial expression, but JIT codegen has to evaluate `exp` per quad point per solve. Need to measure overhead before committing. Worst case: project (α, φ) onto a scalar mesh variable per step, evaluated once.

3. **Yield kink in the exponential operator**? The "linear relaxation" model assumes constant τ over the step. When yielding kicks in mid-step, that's wrong. Lagged-τ is first-order in this nonlinearity; sub-iteration via SNES is exact. Phase C decides which.

4. **Snapshot machinery compatibility**? The existing snapshot fix (commits 8f2b0dd, 31abad1) is what makes BDF VEP variable-dt stable. The exponential integrator has different history structure but the snapshot mechanism (project actual stress → use that for next step's `psi_star[0]`) should apply identically. Verify on Phase B variable-dt test.

5. **Default for VEP/TI-VEP?** Even after exponential is wired in and validated, when should it become the *default*? Probably keep BDF as the default for backward-compat and select via `time_integrator="exponential"` until the exp path has soak-tested across the full benchmark suite.

### Phase B — UW3 prototype: pure VE

New constitutive model `MaxwellExponentialFlowModel` (sibling of `ViscousFlowModel`). Single history slot for σ, one for ε̇. Stress expression for the residual:

```python
sigma = alpha * sigma_star + 2*mu*A * E + 2*mu*B * E_star
```

Where `alpha`, `A`, `B` are scalar UWexpressions updated per timestep from current `dt` and `tau`. `E` is `Unknowns.E` (current strain rate), `E_star` is the previous step's strain rate.

Validate against `bench_ve_harmonic.py` — should match or beat BDF-2's 1.34e-3 max\|err\|.

**Decision gate**: if VE prototype passes the existing benchmark, proceed to Phase C.

### Phase C — VEP extension

For yield-active VEP, the relaxation time $\tau$ depends on stress (plastic viscosity grows when yielding). The "linear relaxation" framing breaks down.

Three options:

1. **Lagged-τ (Picard-style)**: use $\tau$ from the previous step in $\alpha, A, B$. First-order accurate in the nonlinear coupling but trivial to implement.
2. **Sub-iteration within step**: iterate $\tau \to \sigma \to \tau$ to convergence inside each step. Robust but expensive.
3. **Exponential of nonlinear operator**: keep the integrating-factor form but with $\tau$ as a function of the iterate. Gets clever and we'd need to think carefully about what's exact.

**Plan**: start with (1), validate on `bench_vep_square_vardt` against existing BDF-1 and BDF-2 traces. Move to (2) only if (1) is too inaccurate at yield onset.

### Phase D — TI-VEP extension

Two viscosities ($\eta_0$ bulk, $\eta_1$ fault-plane), two relaxation times ($\tau_0 = \eta_0/\mu$, $\tau_1 = \eta_1/\mu$). The exponential update needs to handle the rank-4 tensor structure.

Open question: does the exponential decompose component-wise along director vs perpendicular, or do we need a tensor exponential?

**Plan**: defer until B and C land. Validate on `bench_ti_vep_harmonic` with the spatial-yield-stress fault — this is the case where the BDF-2 path was unstable. Exponential should sidestep entirely.

## Open questions / risks

1. **The $\dot\varepsilon^n$ history term** — currently UW3's `DFDt` machinery stores $\sigma^*$ (stress history). Adding $\dot\varepsilon^*$ is structurally new. Could re-use the same projection-snapshot machinery used for $\sigma^*$ but applied to $\dot\varepsilon$. Worth thinking about whether this complicates SemiLagrangian advection or the projection-fix.

2. **Spatial $\tau$** — when $\eta$ varies spatially (yield zone, weakness fields), $\tau = \eta/\mu$ varies. The exponential coefficients $\alpha, A, B$ are then per-quadrature-point, not constants. Symbolically this is fine (sympy handles $\exp$ of a spatial expression) but JIT codegen has to evaluate $\exp$ per quadrature point per solve — potentially expensive if $\eta$ is complex.

3. **Yield kink** — at the yield surface, $\eta_{\text{eff}}$ has a jump (Min mode) or steep transition (softmin). The "linear relaxation" model is wrong precisely at that boundary. Need to think about what the right object is.

4. **BC discontinuities** — in our square-wave benchmarks, $\dot\varepsilon^n$ and $\dot\varepsilon^{n+1}$ can differ by $\dot\varepsilon_0$ (sign flip). The trapezoidal-style $A, B$ quadrature can lose accuracy at the discontinuity. Possibly need a "fine-dt around flips" adapter (we already have this for BDF in `bench_*_vardt`).

5. **Integration with `set_jacobian_F1_source`** — once exponential is the residual, the Jacobian autodiff is qualitatively different (no multistep $\psi^*_1$ term). Probably *easier*: no inexact-Newton tricks needed because there's no instability.

## Validation against existing benchmarks

Once Phase B–D are in:

| Benchmark | Current (BDF-2) | Exp target |
|---|---|---|
| `bench_ve_harmonic` | 1.34e-3 (peak-start) | match or beat |
| `bench_ve_square` | passes | passes |
| `bench_ve_square_vardt` | passes | passes |
| `bench_vep_square` | passes (Min) | passes (with lagged-τ) |
| `bench_vep_square_vardt` | passes | passes |
| `bench_ti_vep_harmonic` | **unstable at order=2** | **stable at full order** |

The TI-VEP fault row is the strongest motivation: BDF-2 currently can't run that problem at all without the `bdf_blend=0.10` damping (which throws away the order-2 benefit). Exponential should run it cleanly.

## Next concrete step

Implement and run the Phase A 1D Maxwell validator. Should take ~1 hour and definitively answer "is this formulation actually as good as the math suggests".
