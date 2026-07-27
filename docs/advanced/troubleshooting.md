---
title: "Troubleshooting"
---

# Troubleshooting

Notes on behaviour changes and common pitfalls that show up as "my old script
runs but gives different answers / different performance".

## Stokes penalty is now viscosity-scaled (June 2026)

The Stokes augmented-Lagrangian (grad-div) incompressibility penalty changed
from a bare constant to a **viscosity-scaled** term. The operator changed
from

$$
\sigma + \lambda \, (\nabla \cdot \mathbf{u}) \, I
$$

to

$$
\sigma + \lambda \, \mu \, (\nabla \cdot \mathbf{u}) \, I
$$

where $\mu$ is the local viscosity (`constitutive_model.K`) and $\lambda$ is
the value you set with `stokes.penalty`.

Why: with spatially variable viscosity, a constant penalty is enormous
relative to the stress in low-viscosity regions (over-stiffening them into
velocity locking) and negligible in high-viscosity regions. Scaling by the
local viscosity keeps the penalty proportional to the local stress scale
everywhere.

**What to change in older scripts:**

1. `stokes.penalty` is now a *dimensionless* number of order 1. Scripts that
   tuned a large constant against the viscosity magnitude (say `penalty=1e6`
   for a model with $\mu \sim 1$, or a value chosen to sit above the largest
   viscosity in a contrast model) should drop that tuning — use `1.0` (or a
   modest multiple) instead. Keeping the old large value multiplies it by
   the viscosity again and can lock the velocity field or destroy the
   conditioning of the velocity block.
2. A penalty is still **off by default** (`penalty = 0`) and usually
   unnecessary: the pressure Schur complement is already preconditioned with
   the local $1/\mu$ scaling.
3. The penalty term is part of the operator, so converged pressures include
   the grad-div contribution; diagnostics that compared pressure against a
   constant-penalty run will differ at the level of the divergence residual.

See the `Stokes.penalty` property docstring for the full description, and the
`CONSTRAINED_FREESLIP_MULTIPLIER` design note for the derivation.
