---
title: "VE — square-wave shear"
---

# Maxwell viscoelastic shear under square-wave forcing

A Maxwell material driven by a square-wave shear rate also has a
closed-form solution: within each half-period the stress relaxes
exponentially toward the new steady-state value.  This benchmark
exercises the BDF-2 stress-history integrator at the BC discontinuities,
where the time derivative has jumps.

## Governing equation

Same Maxwell ODE as the harmonic case,

$$
\dot\sigma + \frac{\sigma}{t_r} = \mu\,\dot\gamma(t),
$$

but now $\dot\gamma(t) = s_n\,\dot\gamma_0$ where $s_n = (-1)^n$ is the
sign during half-period $n$ of length $T_{1/2}$.  Within half-period
$n$ (with $t_n = n\,T_{1/2}$ and initial value $\sigma_{0,n}$),

$$
\sigma(t) = s_n\sigma_{\mathrm{ss}}
+ \bigl(\sigma_{0,n} - s_n\sigma_{\mathrm{ss}}\bigr)\,
e^{-(t-t_n)/t_r},
\qquad
\sigma_{\mathrm{ss}} = \eta\,\dot\gamma_0,
$$

and the next half-period's initial value is the previous one's end
value:

$$
\sigma_{0,n+1} = s_n\sigma_{\mathrm{ss}}
+ \bigl(\sigma_{0,n} - s_n\sigma_{\mathrm{ss}}\bigr)\,
e^{-T_{1/2}/t_r}.
$$

After a few periods the response settles into a periodic envelope
between $\pm\sigma_{\mathrm{ss}}\tanh\bigl(T_{1/2}/(2 t_r)\bigr)$.

## Setup

| | |
|---|---|
| Mesh | `StructuredQuadBox` 16×8 over $\bigl(\pm 1,\pm 0.5\bigr)$ |
| Velocity field | $\mathbb{P}^2$ |
| Pressure field | $\mathbb{P}^1$ |
| Time integration | BDF-2, $\Delta t = 0.10\,t_r$ |
| Shear viscosity | $\eta = 1$ |
| Shear modulus | $\mu = 1$ |
| Top velocity amplitude | $V_0 = 0.5$ → $\dot\gamma_0 = 1$ |
| Half-period | $T_{1/2} = 2\,t_r$ |
| Run length | 4 full periods (= $8\,T_{1/2}$) |

## Run

```bash
pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_square.py
pixi run -e amr-dev python docs/advanced/benchmarks/plot_benchmarks.py
```

Logs to `output/benchmarks/ve_square.npz`.

## Results

```{figure} ../figures/bench_ve_square.png
:width: 100%

Top: simulated stress (red points) and analytical envelope (black) over
four periods of the square-wave forcing (light blue fill).  Middle:
pointwise absolute error on a log scale; the bumps coincide with the BC
flips at $t = 2, 4, 6,\ldots\,t_r$ where the analytical $\dot\sigma$ has
a jump.  Bottom: time-step (constant for this run).
```

The simulation tracks the analytical envelope with a max error of
$\sim 8\times 10^{-2}$, concentrated immediately after each BC flip.
The error decays exponentially within each half-period as the
discrete BDF-2 history catches up with the new ramp.  The decay rate
matches the Maxwell relaxation time $t_r$.

The asymptotic per-period envelope amplitude is
$\sigma_{\mathrm{ss}}\tanh(T_{1/2}/(2t_r)) = \tanh(1) \approx 0.762$,
which the simulation reaches within the first two periods.
