---
title: "Semi-Lagrangian Time Integration (SLCN / SL-BDF2)"
---

# Semi-Lagrangian time integration: SLCN and SL-BDF2

`AdvDiffusionSLCN` (and the `SemiLagrangian` history operator behind it)
solves advection–diffusion

$$\frac{\partial u}{\partial t} + \mathbf{v}\cdot\nabla u
   - \nabla\cdot(\kappa\nabla u) = f$$

by tracing characteristics backward in time for the advective part and
treating diffusion implicitly. This page explains the **two independent
order knobs** in the scheme and how to pair them correctly — the common
pitfall is mixing them.

## The scheme has two time-integration choices

The discrete residual assembled by the solver is

$$\underbrace{\frac{c_0\,u^{n+1} + c_1\,u^{*} + c_2\,u^{**} + \cdots}{\Delta t}}_{\textbf{time derivative — BDF}}
  \;=\;
  \nabla\cdot\underbrace{\big[a_0\,\mathbf{F}^{n+1} + a_1\,\mathbf{F}^{*} + \cdots\big]}_{\textbf{flux — Adams–Moulton / }\theta}
  \;+\; f$$

where $u^{*}, u^{**}, \mathbf{F}^{*}$ are quantities sampled at the
**departure points** of the characteristics (one step, two steps, …
upstream).

These two sides are controlled separately:

| term | operator | knob | meaning |
|------|----------|------|---------|
| time derivative $\mathrm{D}u/\mathrm{D}t$ | `DuDt` | **BDF `order`** | backward-difference stencil along the characteristic |
| diffusive flux | `DFDt` | **`theta`** (Adams–Moulton) | time-centring of the flux |

`DuDt` and `DFDt` are both `SemiLagrangian` operators, reachable as
`adv_diff.DuDt` and `adv_diff.DFDt`.

:::{important}
**BDF** (backward differentiation) and **Adams–Moulton/θ** are *distinct*
linear-multistep families. A BDF stencil is one-sided implicit and expects
the flux evaluated **at $n+1$ only**; the trapezoidal (Crank–Nicolson) flux
is *centred* between $n$ and $n+1$ and pairs with a single-step difference.
They must be chosen as a **matched pair**.
:::

## The two standard schemes

### SLCN — Semi-Lagrangian Crank–Nicolson (default)

`order = 1`, `theta = 0.5`:

$$\frac{u^{n+1} - u^{*}}{\Delta t}
  = \tfrac12\,\nabla\cdot(\mathbf{F}^{n+1} + \mathbf{F}^{*}) + f .$$

This is the classic scheme of Spiegelman & Katz (2006). Although the
stencil and the departure point are formally first order, the
trapezoidal-along-the-trajectory structure recovers **second-order**
accuracy. It is A-stable but not L-stable, so it can ring on stiff,
under-resolved gradients (see `theta` below).

```python
adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym, order=1)
# theta defaults to 0.5 (Crank-Nicolson)
```

### SL-BDF2 — Semi-Lagrangian BDF2

`DuDt` BDF `order = 2`, flux **`theta = 1.0`**:

$$\frac{\tfrac32 u^{n+1} - 2u^{*} + \tfrac12 u^{**}}{\Delta t}
  = \nabla\cdot\mathbf{F}^{n+1} + f .$$

BDF2 uses **two** departure points and evaluates the flux **implicitly at
$n+1$** (hence `theta = 1.0`, not Crank–Nicolson). It is also second-order
accurate but, unlike CN, avoids the spurious resonance / sign-flip ringing
CN can show on stiff modes (Bonaventura et al., 2021).

Because the internally-built `DuDt` is fixed at BDF order 1, SL-BDF2 is
selected by supplying an explicit order-2 `DuDt` and setting the flux
`theta`:

```python
duDt = uw.systems.ddt.SemiLagrangian(
    mesh, T.sym, v.sym,
    vtype=uw.VarType.SCALAR, degree=T.degree,
    continuous=T.continuous, order=2,          # BDF2 stencil
)
adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym, DuDt=duDt, order=1)
adv.DFDt.theta = 1.0                           # flux implicit at n+1
```

:::{warning}
**Do not mix a BDF2 stencil with a Crank–Nicolson flux** (`order=2` +
`theta=0.5`). The left side is centred at $n+1$ while the right side is
centred at $n+\tfrac12$; the combination is *not* a consistent second-order
scheme. Use SLCN (`order=1`, `theta=0.5`) **or** SL-BDF2 (`order=2`,
`theta=1.0`), not a hybrid.
:::

## `theta` — Crank–Nicolson vs Backward Euler (flux)

For the order-1 flux integrator the Adams–Moulton coefficients are
$[\theta,\,1-\theta]$:

- `theta = 0.5` — **Crank–Nicolson** (trapezoidal): second-order, A-stable.
  The default. Can ring on stiff, under-resolved diffusion.
- `theta = 1.0` — **Backward Euler**: L-stable and monotone for diffusion,
  first-order on its own — and the correct flux centring for SL-BDF2.

`theta` is settable after construction: `adv_diff.DFDt.theta = 1.0`.

## Related options

- **`monotone_mode`** (`"clamp"` / `"pick"`) bounds the semi-Lagrangian
  trace-back to the local data range, curing FE-overshoot ("pepper")
  speckles in cells with sharp gradients. Forwarded to both `DuDt` and
  `DFDt`. See the `AdvDiffusionSLCN` docstring.
- **`old_frame_traceback`** samples the history on the *previous-step*
  mesh geometry rather than the new one — the cure for the semi-Lagrangian
  blow-up on a moving (free-surface or adapting) mesh. It is order-agnostic
  (works for SLCN and SL-BDF2 alike). See
  [`docs/developer/design/lagged-clone-sl-history.md`](../developer/design/lagged-clone-sl-history.md).

## References

- Spiegelman, M., & Katz, R. F. (2006). *A semi-Lagrangian Crank–Nicolson
  algorithm for the numerical solution of advection–diffusion problems.*
  Geochemistry, Geophysics, Geosystems, 7(4).
  <https://doi.org/10.1029/2005GC001073>
- Bonaventura, L., Calzola, E., Carlini, E., & Ferretti, R. (2021).
  *Second order fully semi-Lagrangian discretizations of
  advection–diffusion–reaction systems.* Journal of Scientific Computing,
  88, 23. <https://doi.org/10.1007/s10915-021-01518-8>
