---
title: "VE — sinusoidal shear"
---

# Maxwell viscoelastic shear under sinusoidal forcing

A Maxwell material driven by a sinusoidal shear-rate has a closed-form
stress response.  This benchmark drives the simple-shear box with
$V_{\mathrm{top}}(t) = V_0 \sin(\omega t)$ and compares the centre-point
shear stress to the analytical solution.  The check covers the
amplitude attenuation and phase lag that the Deborah number predicts.

## Governing equation

Maxwell constitutive law in shear:

$$
\dot\sigma + \frac{\sigma}{t_r} = \mu\,\dot\gamma(t),
\qquad t_r = \frac{\eta}{\mu}.
$$

For $\dot\gamma(t) = \dot\gamma_0 \sin(\omega t)$ with $\sigma(0) = 0$,
the closed-form solution is

$$
\sigma(t) = \frac{\eta\,\dot\gamma_0}{1 + \mathrm{De}^2}
\left[\sin(\omega t) - \mathrm{De}\,\cos(\omega t)
+ \mathrm{De}\,e^{-t/t_r}\right]
$$

with $\mathrm{De} = \omega\,t_r$ the Deborah number.  After the
exponential transient (a few $t_r$) the stress oscillates as

$$
\sigma_\infty(t) = A_\infty \sin(\omega t - \varphi),
\qquad
A_\infty = \frac{\eta\,\dot\gamma_0}{\sqrt{1+\mathrm{De}^2}},
\qquad
\varphi = \arctan(\mathrm{De}).
$$

## Setup

| | |
|---|---|
| Mesh | `StructuredQuadBox` 16×8 over $\bigl(\pm 1,\pm 0.5\bigr)$ |
| Velocity field | $\mathbb{P}^2$ |
| Pressure field | $\mathbb{P}^1$ |
| Boundary conditions | top/bottom velocity = $\pm V_0\sin(\omega t)$, free at left/right |
| Time integration | BDF-2, $\Delta t = 0.05\,t_r$ |
| Shear viscosity | $\eta = 1$ |
| Shear modulus | $\mu = 1$ |
| Top velocity amplitude | $V_0 = 0.5$ → $\dot\gamma_0 = 1$ |
| Forcing frequency | $\omega = \pi/2$ → period $4\,t_r$, $\mathrm{De} = \pi/2 \approx 1.57$ |
| Run length | $4$ full periods |

The strain rate uses the symmetric tensor convention
$\dot\varepsilon_{xy} = (\partial_y u_x + \partial_x u_y)/2$, so
$\dot\gamma = 2 V_0 / H = 1$ for $V_0 = 0.5$, $H = 1$.

## Run

```bash
pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_harmonic.py
pixi run -e amr-dev python docs/advanced/benchmarks/plot_benchmarks.py
```

The simulation logs to `output/benchmarks/ve_harmonic.npz` (per-step
trace + analytical reference at the same time points).  Re-running the
plot script doesn't re-run the simulation.

## Results

```{figure} ../figures/bench_ve_harmonic.png
:width: 100%

Top: simulated stress (red points), closed-form solution (black), and
the rescaled square-wave forcing (light blue fill) for context.  Middle:
pointwise absolute error.  Bottom: time-step.  Inset shows the
fitted-vs-analytical amplitude and phase.
```

At $\mathrm{De} = \pi/2$ the analytical amplitude is
$A_\infty = 1/\sqrt{1+\pi^2/4} \approx 0.537$ and the phase lag is
$\varphi = \arctan(\pi/2) \approx 1.004$ rad.  The benchmark recovers
both: the fitted amplitude matches to within $10^{-3}$ and the phase
lag to within a few percent over the post-transient window.  The
pointwise error sits below $\sim 2\times 10^{-2}$ — the BDF-2 phase
error at $\Delta t / T = 0.0125$ explains essentially all of it.
