---
name: transport-schemes
description: Which transport scheme and which time history to use in Underworld3, and why — nodal, integration-point or grid histories; semi-Lagrangian, Eulerian SUPG or Lagrangian swarm transport; the Courant number to run at; what each choice does to a settled state and to a peak. The evidence is the tests and notes named beside each ruling.
families: [AdvDiffusion, NavierStokes, Eulerian, EulerianSUPG, Lagrangian, Lagrangian_Swarm,
  BackwardNodesSemiLagrangian, BackwardIntegrationPointsSemiLagrangian,
  ForwardIntegrationPointsSemiLagrangian, ForwardNodesSemiLagrangian]
kind: guide
status: draft, rulings to be confirmed
---

# Transport schemes: which one, when

A time-dependent solve in Underworld3 is a residual plus a history: the
history (`DuDt`, `DFDt`) says where the quantity was at the previous levels
and how it got there. The scheme is the choice of where that history lives
and how it is carried. The classes describe themselves
(`uw.capabilities("histories")`); this page says which to choose.

## The choices

| scheme | history lives on | carried by | use for |
|---|---|---|---|
| `Eulerian` | mesh nodes | nothing moves; the transport term is in the residual | diffusion-dominated fields, or with SUPG below |
| `EulerianSUPG` | mesh nodes | implicit advection with streamline-upwind stabilisation, assembled in the residual | the momentum equation of `NavierStokes`; a field advected by a resolved velocity |
| `SemiLagrangian` | mesh nodes | traced back along the flow to a departure point and interpolated there | advection-diffusion at moderate Courant number; the transport the adjoint can differentiate |
| `IntegrationPointSemiLagrangian` | integration points | the same trace, from the quadrature points | stress and other flux histories that must not be smoothed through the nodes |
| `Lagrangian_Swarm` | particles | the particles move; the mesh reads a proxy | material identity, and any history that must follow the material exactly |
| `Symbolic` | nowhere | the user's own expression | a history the script supplies itself |

## What the tests established

**Advection of a step (Waters and King, 2026-09-16, #749).** With the
same time integrator, a nodal history converges as the timestep falls; an
integration-point history degrades as the timestep falls, ringing and then
diverging; a grid history diverges at every timestep. The peak of the
profile is set by the integrator, the settled state by the transport. Rule:
for a scalar field carried by the flow, put the history on the nodes.

**Stress histories (2026-09-11, #735).** For a viscoelastic flux history
the grid and integration-point histories converge together with
resolution; the nodal history converges to a different answer, with about
half again too much stress in the first two cells off a no-slip wall,
because the nodal projection smooths the history through the wall. Rule:
a flux history lives on the integration points (or on the grid, with DEVSS
opted in); a scalar field's history lives on the nodes. The two rules are
not in conflict: they are different quantities.

**Courant number for the integration-point trace (#703, #737).** The
integration-point semi-Lagrangian scheme runs at Courant number about
one, not below it. Damping it to run at smaller steps was tried and
disliked; the memory term amplifies the low-Courant mode. Refine the mesh
and the timestep together.

**The momentum equation (#687).** `NavierStokes` carries its momentum
transport as `EulerianSUPG`: implicit advection in the residual, with a
partition-independent cell size in the stabilisation.

**What the adjoint can differentiate.** A semi-Lagrangian trace is
differentiable in the velocity, and the interpolation at the departure
points is materialised, so a run built on it is adjointable end to end. A
particle step is adjointable exactly when the particle set is fixed across
it, which `swarm.advection` checks by counting.

## The time integrator is a separate choice

`order` sets the depth of the history and the scheme's order in time;
`theta` sets the weighting (`0.5` is Crank-Nicolson, `1` backward Euler).
`model.step(dt)` carries the clock, and the coefficients of the scheme are
exact rationals in the residual, so the transcript's key shows the
integrator a part used. A fixed timestep keeps an objective from depending
on the control through the schedule; an adaptive one records its decisions
through `model.rewind(reason=...)`.

## Rulings still open

- Whether a nodal history should ever be offered for a flux quantity, or
  refused.
- A default Courant target for the nodal semi-Lagrangian scheme, and
  whether `estimate_dt()` should report it.
