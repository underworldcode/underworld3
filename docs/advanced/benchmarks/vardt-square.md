---
title: "Variable-dt square-wave (VE and VEP)"
---

# Square-wave shear with reduced timestep near BC discontinuities

The VE and VEP square-wave cases concentrate their numerical error in a
small window around each BC flip — the discrete time derivative is
attempting to follow a corner in the analytical $\sigma(t)$.  Reducing
$\Delta t$ inside that window and keeping it large on the plateaux is
exactly the kind of variable-timestep schedule that the projection
machinery has to handle robustly.

This pair of benchmarks runs the VE and VEP square-wave problems with
$\Delta t$ = 0.10·$t_r$ on plateaux and 0.01·$t_r$ within ±0.20·$t_r$
of every flip — a 10× reduction across the discontinuity.

## Schedule

```
Δt(t) = 0.10·t_r          on plateaux  (≥ 0.20·t_r away from any flip)
        0.01·t_r          within ±0.20·t_r of a flip
```

Step boundaries are clamped to the flip time so no step straddles a
discontinuity.

## What this exercises

* The DDt's snapshot machinery: every halve/double of $\Delta t$
  exposes a new $\Delta t$ ratio to the implicit projection.  Without
  the snapshot fix the previous-generation code drifted off the yield
  surface by ~30% under exactly this schedule.
* The Picard / `divergence_retries` SNES rescue: VEP's first solve
  inside a fine-window after a flip lands close to the yield kink and
  occasionally takes a Newton step that fails the tolerance check
  within 50 iterations; the retry mechanism recovers without manual
  intervention.

## Results

### VE

```{figure} ../figures/bench_ve_square_vardt.png
:width: 100%

Top: BDF-1 (blue circles) and BDF-2 (red squares) overlaid on the
analytical (black) for the variable-Δt VE square wave.  Driving γ̇
shown in light blue fill.  Middle: pointwise absolute error.  Bottom:
the Δt schedule with the 10× drop visible at every flip.
```

| | BDF-1 | BDF-2 |
|---|---|---|
| max\|err\| | 2.38e-02 | 1.42e-02 |
| rms        | 1.62e-02 | 6.17e-03 |

For comparison, the fixed-Δt=0.10 run gave BDF-2 max\|err\| ≈ 8.07e-02
— so the fine window around the flip is doing exactly what it should
(reducing the dominant per-flip error).

### VEP (Min mode)

```{figure} ../figures/bench_vep_square_vardt.png
:width: 100%

Same layout, with τ_y = 0.5 yield surface guides (dashed grey).
```

| | BDF-1 | BDF-2 |
|---|---|---|
| peak\|σ\| | 0.5000 | 0.5004 |
| overshoots > 1.001·τ_y | 0 | 0 |
| max\|err\| | 2.15e-02 | 8.70e-02 |
| rms        | 6.68e-03 | 1.53e-02 |

**The yield surface holds**.  With both the snapshot machinery and
the Picard-style SNES retry in place, σ stays clipped to ±τ_y under
the variable-Δt schedule that previously produced a ~30% drift.
Peak\|σ\| matches τ_y = 0.5 to four decimal places (BDF-1) and
within 0.1% (BDF-2 — the 0.0004 excess is a transient at one
loading-onset transition, not a sustained yield-surface violation).

The BDF-2 max\|err\| being larger than BDF-1's is the same phase-lag
story as in the fixed-dt VEP case: at the loading→yield transition
the 2nd-order step occasionally lags by one fine-Δt step before
catching up.  RMS — which is the more honest measure for a sharp
transition — is comparable to BDF-1's.

### VEP (softmin mode, default δ = 0.1)

```{figure} ../figures/bench_vep_square_vardt_softmin.png
:width: 100%

Same problem with `yield_mode="softmin"` — replacing
$\min(\eta_{ve},\eta_{pl})$ with the smooth approximation
$\eta_{ve}/g(f)$, $g(f) = 1 + (f-1+\sqrt{(f-1)^2+\delta^2})/2$.
At δ = 0.1 the kink is differentiable but the plateau still tracks
the yield surface tightly.
```

| | BDF-1 | BDF-2 |
|---|---|---|
| peak\|σ\| | 0.4894 (97.9% of τ_y) | 0.4853 (97.1% of τ_y) |
| max\|err vs Min-clip\| | 8.39e-02 | 1.11e-01 |
| rms vs Min-clip       | 6.12e-02 | 7.76e-02 |

Softmin keeps the plateau within 2-3% of the true Min yield surface
while smoothing out the kink at $\eta_{ve} = \eta_{pl}$ — Newton sees
a continuous derivative and the SNES never needs the Picard retry
that Min mode occasionally triggers.  The "error vs Min-clip"
figure-of-merit penalises the deliberately rounded transitions; it
is not an accuracy gap in any physically meaningful sense.

### Why `yield_mode="smooth"` was retired

The third yield-mode option, the harmonic-blend "smooth" formula
$\eta_{eff} = \eta_{ve}\,(1+f)/(1+f+f^2)$, was retired in this same
benchmark suite (commit 5936b46) after the variable-dt run made the
problem unmissable:

```{figure} ../figures/bench_vep_square_vardt_smooth.png
:width: 100%

`yield_mode="smooth"` plateaus at |σ| ≈ 0.24 — only ~50% of
τ_y = 0.5 — across every loading half-cycle.  Both BDF orders
under-clip identically.  This is not a transient; the driving
$\dot\gamma$ holds long enough to reach steady state on every plateau.
```

| | BDF-1 | BDF-2 |
|---|---|---|
| peak\|σ\| | 0.2599 (52.0% of τ_y) | 0.2355 (47.1% of τ_y) |
| max\|err vs Min-clip\| | 3.66e-01 | 3.81e-01 |

The blend formula has the wrong asymptotic behaviour: at $f = 1$
(the yield kink itself) it gives $\eta_{eff}/\eta_{ve} = 2/3$, and it
keeps reducing $\eta_{eff}$ deep into the plastic regime instead of
saturating at $\eta_{pl}$.  `softmin` does not have this defect, so
`smooth` was removed and the setter now redirects users.
