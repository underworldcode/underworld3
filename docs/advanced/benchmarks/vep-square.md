---
title: "VEP — square-wave shear (Min mode)"
---

# Visco-elastic-plastic shear under square-wave forcing

Add a yield surface to the square-wave VE benchmark and the closed-form
solution is just the *clipped* version of the VE square-wave: within each
half-period the stress evolves exponentially toward
$\pm\sigma_{\mathrm{ss}}$ but is held at $\pm\tau_y$ while the material
is yielding.

This benchmark verifies the implementation of Min-mode plasticity, the
yield-surface clip itself, and — under variable timestep — the
projection-snapshot machinery in `SemiLagrangian` DDt that prevents the
implicit-projection drift at the Min kink (see the regression test in
`tests/test_1052_VEP_stability_regression.py`).

## Governing equation

Maxwell evolution with a Min-mode yield surface:

$$
\dot\sigma + \frac{\sigma}{t_r} = \mu\,\dot\gamma(t),
\qquad
\eta_{\mathrm{eff}} = \min\bigl(\eta_{\mathrm{ve}},\,\eta_{\mathrm{pl}}\bigr),
\qquad
\eta_{\mathrm{pl}} = \frac{\tau_y}{2\,|\dot\varepsilon_{\mathrm{eff}}|}.
$$

Within each half-period the analytical solution is

$$
\sigma(t) = \mathrm{clip}\Bigl(
s_n\sigma_{\mathrm{ss}}
+ (\sigma_{0,n} - s_n\sigma_{\mathrm{ss}})\,e^{-(t-t_n)/t_r},
\;-\tau_y,\;+\tau_y
\Bigr).
$$

Because the yielded portion holds $\sigma = \pm\tau_y$ exactly, the
*clipped* value carries forward as the next half-period's initial
condition:

$$
\sigma_{0,n+1} = \mathrm{clip}\bigl(\sigma(t_n+T_{1/2}),
\;-\tau_y,\;+\tau_y\bigr).
$$

When $\eta\,\dot\gamma_0 > \tau_y$ (yielding occurs) the response
saturates at $\pm\tau_y$ during the second half of each half-period.

## Setup

| | |
|---|---|
| Mesh | `StructuredQuadBox` 16×8 over $\bigl(\pm 1,\pm 0.5\bigr)$ |
| Velocity field | $\mathbb{P}^2$ |
| Pressure field | $\mathbb{P}^1$ |
| Time integration | BDF-2, $\Delta t = 0.10\,t_r$ |
| Shear viscosity | $\eta = 1$ |
| Shear modulus | $\mu = 1$ |
| Yield stress | $\tau_y = 0.5$ (so $\eta\dot\gamma_0 / \tau_y = 2$) |
| Yield mode | `min` |
| Top velocity amplitude | $V_0 = 0.5$ → $\dot\gamma_0 = 1$ |
| Half-period | $T_{1/2} = 2\,t_r$ |
| Run length | 4 full periods |

## Run

```bash
pixi run -e amr-dev python docs/advanced/benchmarks/bench_vep_square.py
pixi run -e amr-dev python docs/advanced/benchmarks/plot_benchmarks.py
```

Logs to `output/benchmarks/vep_square.npz`.

## Results

```{figure} ../figures/bench_vep_square.png
:width: 100%

Top: BDF-1 (blue open circles) and BDF-2 (red filled squares)
overlaid on the analytical clipped solution (black), yield surface
guides $\pm\tau_y$ (dashed grey), and rescaled forcing (light blue
fill).  Middle: pointwise absolute error for both orders on a log
scale — note the dramatic drop to $\sim 10^{-6}$ during yielded
plateaux where simulation and analytical both sit at $\pm\tau_y$ to
machine precision.  Bottom: time-step.
```

Two things to read from the per-case plot:

1. **The yield surface holds for both orders**.  Peak $|\sigma|$
   matches $\tau_y = 0.5$ to four decimal places at the canonical dt
   for both BDF-1 and BDF-2; the count of overshoots
   ($|\sigma| > 1.001\,\tau_y$) is zero in both runs.  This is the
   regression that the
   [variable-dt yield-lock test](../../../tests/test_1052_VEP_stability_regression.py)
   protects against re-introduction.

2. **The error has structure**.  During yielded plateaux the
   simulation matches the analytical to machine precision (the
   $\sim 10^{-6}$ floor is the projection's L2 residual).  During the
   elastic loading/unloading transients the per-step truncation error
   peaks just after each BC flip and decays within the half-period.

```{figure} ../figures/bench_convergence.png
:width: 100%

Convergence sweep — right panel is the VEP case.  BDF-1 shows clean
slope 1.  BDF-2 follows the same trend with a constant ratio above
BDF-1, until the smallest $\Delta t$ where a transient overshoot
arrives one step late on the yield-onset transition; this shows up
in the max-norm but not the rms.  Peak $|\sigma|$ stays within 1.3 %
of $\tau_y$ at every $\Delta t$ tested.
```

The benchmark's strict accuracy requirement is the yield-surface peak,
not the transient error: any future change that produces $|\sigma| >
\tau_y$ on a fixed-dt yielded plateau by more than the yield-lock
test's tolerance fails the regression suite.
