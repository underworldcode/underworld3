---
title: "Viscoelastic Square-Wave Shear Benchmark"
---

# Maxwell Square-Wave Shear

This benchmark validates the viscoelastic Stokes solver with **variable timesteps**
against an analytical solution for a Maxwell material under square-wave shear forcing.

It tests both the variable-dt BDF-2 coefficients and the PetscDS constants mechanism
that routes these coefficients to the compiled pointwise functions at runtime.

## Problem Setup

Same geometry as the oscillatory shear benchmark: a box with height $H$ and width $2H$,
sheared by top/bottom boundary velocities. The shear rate is a truncated Fourier series
approximation of a square wave:

$$\dot\gamma(t) = \frac{4\dot\gamma_0}{\pi}
\sum_{k=1}^{N} \frac{\sin\bigl((2k-1)\omega t\bigr)}{2k-1}$$

The sharp transitions between positive and negative shear demand small timesteps
near the transition points, while the plateaux can use much larger steps. This
makes it a natural test for adaptive (variable) timestepping.

## Analytical Solution

Since the Maxwell equation is linear, the stress is the superposition of
single-frequency Maxwell solutions at each Fourier harmonic:

$$\sigma_{xy}(t) = \sum_{k=1}^{N} \sigma_k(t)$$

where each $\sigma_k$ is the oscillatory Maxwell solution with amplitude
$a_k = 4\dot\gamma_0 / (\pi(2k-1))$ and frequency $\omega_k = (2k-1)\omega$:

$$\sigma_k(t) = \frac{\eta\, a_k}{1 + \text{De}_k^2}
\left[\sin(\omega_k t) - \text{De}_k\cos(\omega_k t)
+ \text{De}_k\,e^{-t/t_r}\right]$$

with $\text{De}_k = \omega_k t_r$.

## Adaptive Timestep Strategy

The timestep varies between `dt_min` near transitions and `dt_max` on plateaux,
based on distance to the nearest square-wave transition point:

$$\Delta t = \Delta t_{\min} + (\Delta t_{\max} - \Delta t_{\min})\, f^2$$

where $f \in [0,1]$ measures the normalised distance from the nearest transition.

## Variable-dt BDF Coefficients

With uniform timesteps, BDF-2 uses constant coefficients $[3/2, -2, 1/2]$.
With variable timesteps (ratio $r = \Delta t_n / \Delta t_{n-1}$), the
coefficients become:

$$c_0 = \frac{1+2r}{1+r}, \quad c_1 = -(1+r), \quad c_2 = \frac{r^2}{1+r}$$

These coefficients are stored as UWexpressions and updated each step via
`_update_bdf_coefficients()`, flowing through PetscDS `constants[]` to the
compiled pointwise functions without JIT recompilation.

## Results (De=1.5, BDF-2, 10 harmonics)

| Run | Steps | L2 Error | Ratio |
|-----|-------|----------|-------|
| Adaptive dt | 295 | 9.55e-04 | 1.78x |
| Uniform dt | 629 | 5.35e-04 | 1.0x |

The adaptive run uses 53% fewer steps at only 1.78x the error.

## Running the Benchmark

```bash
pixi run -e default python docs/advanced/benchmarks/run_ve_square_wave.py
```

The script runs both adaptive and uniform timestep cases, prints convergence
data, and saves results to `.npz` files.
