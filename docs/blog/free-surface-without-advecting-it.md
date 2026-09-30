# A free surface you never advect

*Draft blog post — Underworld development team. Status: held pending a possible
method paper (the scheme is plausibly novel; see the
[method note](../developer/design/FREESLIP_DYNAMIC_TOPOGRAPHY_FREESURFACE.md)
and its cited novelty assessment). Not wired into the published docs toctree
yet — release at the team's discretion.*

---

## The mystery: convection that quietly gives up

We were running a very ordinary experiment: isoviscous convection in an annulus,
Rayleigh number 10⁵, with a deforming free surface on top. Nothing exotic. The
kind of thing that should just work.

It didn't. The Nusselt number — our measure of how vigorously the system carries
heat — started at a healthy 13.6 and then *decayed*, drifting down toward
sub-critical values as if the convection were running out of steam. The interior
slowly cooled. Swap in a different free-surface scheme and the same thing
happened. Relax-to-equilibrium, kinematic finite-element, kinematic-exact — every
way we knew to move a free surface produced the same slow death.

The frustrating part: by every conservation check we threw at it, the simulation
was *fine*. Mass was conserved. The surface amplitude was reasonable. There was
no obvious blow-up, no NaN, no tangled mesh. The model just... lost interest in
convecting.

## The diagnosis: it's a lag, and the lag is a cold pump

The breakthrough was realising the failure had nothing to do with how much the
surface moved, and everything to do with *when* it moved relative to the flow.

Here is the mechanism. We advect temperature semi-Lagrangian: to find the new
temperature at a point, we trace backwards along the velocity to where that
parcel came from, and read the old temperature there. At a downwelling — cold
material plunging into the interior — the flow at the surface is pushing
*inward*. If the surface itself hasn't kept up (because the integrator
under-moved it, or smoothed it, or relaxed it too slowly), then the backward
trace lands *outside* the lagging surface — in the thin layer the surface should
have occupied but doesn't quite, where the only thing to read is the cold `T = 0`
boundary condition.

So every step, at every downwelling, the scheme reaches past its own surface and
pulls in a little bit of cold. A **cold pump**, running on the lag between where
the surface *is* and where the velocity *thinks* it is. It is utterly silent —
mass is conserved, nothing diverges — but it steadily drains the thermal energy
that drives the convection. Hence the slow death.

Any surface lag does it. Fourier truncation of the surface shape: lag. Smoothing
to keep the mesh clean: lag. Relaxation that under-moves: lag. There was no
setting that made the lag zero, because the lag is structural.

## The trap inside the trap

Our first instinct was to make the surface track the velocity *better* — reduce
the lag rather than live with it. That is when we found the second, deeper
problem.

We isolated the surface normal velocity — the `v·n` that a kinematic scheme
integrates to move the surface — and watched it while holding the surface
essentially at its equilibrium shape. It should have been tiny: if the surface is
already where it wants to be, it shouldn't be moving. Instead it was *enormous* —
around 65 in our non-dimensional units — and it barely decayed.

That number isn't the surface trying to move. It's **convective throughflow**:
downwellings slamming into the top boundary and the return flow sweeping back.
The fluid is moving vigorously *past* the surface, not *carrying* it. Reading
`v·n` as a rate-of-change-of-topography is reading the wrong quantity entirely.
Any scheme that integrates it is integrating noise a hundred times larger than
the signal, and it will overshoot no matter how carefully you step it.

So the kinematic free surface is built on a contradiction: the very velocity it
uses to move the surface is dominated by flow that isn't moving the surface at
all.

## The fix: stop advecting the surface

If `v·n` at the surface is mostly throughflow, the cleanest thing to do is *not
let there be any throughflow*. Make the top boundary **free-slip**: `v·n = 0` by
construction. The fluid slides along the boundary but never crosses it. The
backward trace can no longer reach past the surface, because there is no normal
velocity carrying material across it. The cold pump is gone.

But now the surface can't move kinematically — we just forbade the normal
velocity that would move it. So where does topography come from?

From the **stress**. A free-slip lid still feels a normal stress pushing up under
upwellings and down under downwellings. Geodynamicists have computed *dynamic
topography* from exactly this normal stress for decades — it's the height the
surface would take if it were free to deform and compensate that stress
instantaneously. The twist is that this quantity has always been a **diagnostic**:
something you compute and plot *after* the fact, on a surface you keep flat. We
use it as a **prognostic** variable — we let it actually move the surface.

So the surface is no longer carried by the flow. It *relaxes toward* the
stress-derived equilibrium topography. The flow solve stays clean and free-slip;
the surface evolves on its own, driven by the stress the flow produces.

## Three numbers and a guarantee

How do you step a relaxing surface? The naive answer — height plus rate, stepped
forward — is exactly the scheme that gives geodynamics its infamous "drunken
sailor" instability: the surface overshoots its target, the overshoot drives a
stress that pushes it back too hard, and it sloshes itself to pieces unless you
take agonizingly small time steps.

The cure is a third number. Track the height `h`, its rate `ḣ`, *and* the
equilibrium it's heading for, `h_∞`. Those three pin a simple local model — the
surface decays exponentially toward equilibrium — with a rate

> γ = ḣ / (h_∞ − h)

and an exact update over the step:

> h ← h_∞ + (h − h_∞) · e^(−γ Δt)

This is **L-stable by construction**. The update always lands *between* the
current height and the equilibrium — it physically cannot overshoot, for any time
step, ever. The drunken sailor can't stagger because the scheme won't let him
past the target. And it's robust even when `ḣ` is corrupted by that throughflow
noise, because it only needs the *direction* of relaxation (`γ > 0`), not an
accurate rate.

We'll be honest about lineage here: this exponential step is not something we
invented. It's first-order *exponential time differencing*, well known in applied
mathematics (Cox & Matthews, 2002; the no-overshoot property is proven cleanly by
Aursand et al., 2014). What appears to be new is bringing it to a geodynamic free
surface, and pairing it with a stress-derived equilibrium instead of a
kinematically-advected one.

## One solve, two answers

There's a small elegance that makes the whole thing cheap. You might expect to
need two Stokes solves per step: one to get the velocity for advecting
temperature, and another, with a held lid, to get the normal stress for `h_∞`.

You don't. Under free-slip, the lithostatic part of the body force is a pure
gradient — it produces hydrostatic pressure and no flow. So a single *free-slip,
driving-only* solve gives you the advection velocity directly, and its normal
stress is exactly the `h_∞` you need, with no double-counting of the topographic
load. **One solve supplies both.**

## What it buys

With free-slip on top and the surface relaxing exponentially toward its
stress-equilibrium, the convection stops dying. Nusselt stays vigorous and
steady. The temperature field stays clean in [0, 1] — no cold injection, no
speckle. The surface tracks `h_∞` and oscillates *physically* with the flow
instead of marching off in a monotone runaway. The residual normal velocity at
the boundary drops from ~65 to ~0.4 — a hundred-fold — which is the throughflow
finally gone.

It is worth being clear about where this lives. The scheme assumes surface
evolution is relaxation toward instantaneous isostatic compensation — exact in
the long-wavelength limit that matters for whole-mantle dynamic topography, where
the surface is stiff (buoyancy small compared to the restoring force) and
deflections are a few percent. Push it into a genuinely soft regime, where
buoyancy overwhelms the surface, and there simply isn't a smooth surface to track
— that's physics, not a bug.

## So is it new?

We ran a careful literature check, including a targeted sweep of the major
geodynamics codes' manuals and changelogs. The verdict: a **novel combination of
established parts**. Computing free-slip dynamic topography from stress is old
(Zhong, Gurnis & Moresi, 1996; Crameri et al., 2012). The exponential integrator
is older still, just from a different field. The closest relative — the implicit
free surface in Fluidity / G-ADOPT (Kramer, Wilson & Davies, 2012) — co-solves
the surface height with no time-step penalty, but it uses a stress boundary
condition rather than free-slip, and the height *is* the evolving surface rather
than an equilibrium target it relaxes toward.

What we couldn't find anywhere was the synthesis: keep the lid free-slip, treat
the stress-derived dynamic topography as the equilibrium a separate surface
variable relaxes toward, and step that relaxation exponentially. Each ingredient
is published; the recipe doesn't appear to be.

That's why this post is a draft. If the novelty holds up to a proper paper
search, the right venue for this story is a methods paper, not a blog. But the
core idea is too clean not to write down: **the best way to move a free surface
might be to stop advecting it.**

---

*Technical details, the stability argument, the two deformed-geometry bugs we had
to fix first, and full citations are in the
[developer method note](../developer/design/FREESLIP_DYNAMIC_TOPOGRAPHY_FREESURFACE.md).*
