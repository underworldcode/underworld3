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
| Time integration | BDF-1 *and* BDF-2 at $\Delta t = 0.05\,t_r$, plus a sweep over $\Delta t \in \{0.025, 0.05, 0.10, 0.20, 0.40\}\,t_r$ |
| BC sampling | $V_{\mathrm{top}}$ evaluated at the *endpoint* of each step |
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

Top: BDF-1 (blue open circles) and BDF-2 (red filled squares)
overlaid on the closed-form solution (black) and the rescaled
sinusoidal forcing (light blue fill) for context.  Middle: pointwise
absolute error for both orders.  Bottom: time-step.  Inset compares
fitted vs analytical amplitude and phase lag.
```

At $\mathrm{De} = \pi/2$ the analytical amplitude is
$A_\infty = 1/\sqrt{1+\pi^2/4} \approx 0.537$ and the phase lag is
$\varphi = \arctan(\pi/2) \approx 1.004$ rad.  At $\Delta t =
0.05\,t_r$ BDF-2 recovers both the amplitude and the phase lag to
within $10^{-3}$ rad; BDF-1 is off by a few percent in the phase
(the residual O($\Delta t$) error of an implicit-Euler scheme).

```{figure} ../figures/bench_convergence.png
:width: 100%

Convergence sweep — left panel is the harmonic case.  BDF-1 sits on
slope 1 (dotted reference); BDF-2 (rms, lower red dotted line)
hits slope 2 (dashed reference) cleanly between $\Delta t = 0.4$ and
$0.1$ before levelling off at the fine end (where the BDF-2 startup
transient — first one or two steps that effectively run at BDF-1 —
becomes the dominant contribution).
```

The benchmark surfaces a subtle but important detail: $V_{\mathrm{top}}$
is sampled at the step *endpoint* (i.e.\\ the time BDF's implicit step
solves for), not at the midpoint.  Midpoint sampling is only
1st-order accurate to the endpoint value; using it would limit BDF-2
to slope-1 convergence even though the time integrator itself is
2nd-order.  Same nominal mesh, dt schedule, and tolerance — only the
BC sampling differs.
