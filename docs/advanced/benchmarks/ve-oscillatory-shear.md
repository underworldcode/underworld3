---
title: "Viscoelastic Oscillatory Shear Benchmark"
---

# Maxwell Oscillatory Shear

This benchmark validates the viscoelastic Stokes solver against the analytical
solution for a Maxwell material under oscillatory simple shear.

## Problem Setup

A box with height $H$ and width $2H$ is sheared by imposing time-dependent
velocities on the top and bottom boundaries:

$$v_x(y=\pm H/2, t) = \pm V_0 \sin(\omega t)$$

The left and right boundaries are free-slip (no vertical velocity). The shear
rate is $\dot\gamma(t) = \dot\gamma_0 \sin(\omega t)$ where $\dot\gamma_0 = 2V_0/H$.

## Analytical Solution

The Maxwell constitutive law gives:

$$\dot\sigma_{xy} + \frac{\sigma_{xy}}{t_r} = \mu \dot\gamma_0 \sin(\omega t)$$

where $t_r = \eta/\mu$ is the relaxation time. With $\sigma(0) = 0$, the full
solution (including the startup transient) is:

$$\sigma_{xy}(t) = \frac{\eta \dot\gamma_0}{1 + \text{De}^2}
\left[\sin(\omega t) - \text{De}\cos(\omega t) + \text{De}\,e^{-t/t_r}\right]$$

where $\text{De} = \omega t_r$ is the Deborah number.

**Steady-state properties** (after transient decays):

- Amplitude: $A = \eta \dot\gamma_0 / \sqrt{1 + \text{De}^2}$
- Phase lag: $\delta = \arctan(\text{De})$

At $\text{De} = 0$ (viscous limit): $A = \eta\dot\gamma_0$, $\delta = 0$.
At $\text{De} \to \infty$ (elastic limit): $A \to 0$, $\delta \to 90°$.

## Convergence with BDF Order

The VE solver uses BDF-$k$ time integration ($k = 1, 2, 3$). The convergence
study (constant shear, $\text{De} = 1$) shows:

| $\Delta t / t_r$ | BDF-1 error | BDF-2 error | BDF-3 error |
|-------------------|-------------|-------------|-------------|
| 0.200 | 3.0e-02 | 4.0e-03 | 6.4e-03 |
| 0.100 | 1.5e-02 | 9.3e-04 | 1.7e-03 |
| 0.050 | 7.8e-03 | 2.3e-04 | 4.3e-04 |
| 0.020 | 3.1e-03 | 3.7e-05 | — |

BDF-2 achieves second-order convergence (~4x error reduction per halving) and
is the recommended default. BDF-1 is first-order. BDF-3 converges at nearly
second order but with a larger error constant.

## Resolution Study (Oscillatory, De = 5)

At high Deborah number, the oscillation period is short relative to the
relaxation time, requiring fine time resolution. The plot below shows the
effect of timestep size at $\text{De} = 5$ ($\omega t_r = 5$, phase lag = 79°):

- **63 pts/period** ($\Delta t/t_r = 0.02$): both orders match analytical
- **31 pts/period** ($\Delta t/t_r = 0.04$): O1 shows slight amplitude reduction, O2 still accurate
- **16 pts/period** ($\Delta t/t_r = 0.08$): O1 amplitude visibly damped, O2 remains good

```{note}
The amplitude reduction at coarse timesteps is numerical dissipation from
the BDF-1 discrete transfer function, not a cumulative error. The discrete
steady-state amplitude is a fixed fraction of the analytical amplitude,
determined by $\omega \Delta t$.
```

## Running the Benchmarks

```bash
# Oscillatory validation (De=1.5, order 1 and 2)
python tests/plot_ve_oscillatory_validation.py

# Resolution study (De=5, three timestep sizes)
# Saves .npz data files for re-analysis
python tests/plot_ve_oscillatory_validation.py

# Replot from saved data (no re-running)
python tests/plot_ve_oscillatory_validation.py --replot
```

## Notes on `dt_elastic`

The parameter `dt_elastic` on the constitutive model is the elastic relaxation
timescale used in the BDF discretisation. It controls the effective viscosity
$\eta_{\text{eff}}$ and the stress history weighting:

- BDF-1: $\eta_{\text{eff}} = \eta\mu\Delta t_e / (\eta + \mu\Delta t_e)$
- BDF-2: $\eta_{\text{eff}} = 2\eta\mu\Delta t_e / (3\eta + 2\mu\Delta t_e)$
- BDF-3: $\eta_{\text{eff}} = 6\eta\mu\Delta t_e / (11\eta + 6\mu\Delta t_e)$

When `timestep` is passed to `VE_Stokes.solve()`, it controls the advection
step for semi-Lagrangian history transport. It does **not** overwrite
`dt_elastic` — these are independent parameters.

A running-average approach for accumulating history when $\Delta t \ll \Delta t_e$
was investigated but found to be extremely diffusive for semi-Lagrangian
transport and is not implemented. To prevent runaway or unstable behaviour
when timesteps become small (e.g. due to CFL constraints or failure events),
we advise limiting the minimum effective viscosity, in line with the physics
of the problem.
