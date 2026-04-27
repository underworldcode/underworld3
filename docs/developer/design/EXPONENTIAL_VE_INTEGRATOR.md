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
