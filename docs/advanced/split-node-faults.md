---
title: "Split-Node Faults: Zero-Thickness Frictional Surfaces"
---

# Split-Node Faults

A fault, mechanically, is a surface across which the velocity field jumps:
slip, not strain. Underworld3 can represent a fault exactly that way — the
mesh nodes along a conforming fault surface are duplicated so the two
sides share no degrees of freedom, the two coincident copies are tied by a
strong no-opening constraint, and the tangential jump (the slip rate)
either emerges freely or is governed by a friction law. There is no weak
layer, no viscosity contrast, and no band width to resolve: the fault has
genuinely zero thickness, and the surrounding mesh can be uniform.

This is one of two fault representations in Underworld3, and they are
complementary rather than competing:

- **Split-node surface** (this page) — the fault is a sharp interface.
  The right tool when slip is the quantity of interest: earthquake-cycle
  models, stress transfer, plate-boundary-style discontinuities.
- **[Finite-width TI weak zone](vep-transverse-isotropy-faults.md)** —
  the fault is a thin volume with anisotropic rheology. The right tool
  when the fault-zone width is itself constitutive (rate-and-state
  linked to a physical gouge width), or when the same fault must appear
  as a damage zone for another equation (e.g. porous flow).

## Quick start (2-D)

A fault is a polyline with both tips strictly inside the domain. One call
places its points onto mesh vertices, cuts the mesh so the fault becomes
a conforming facet chain, and splits it:

```python
import numpy as np
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.05)
fault_points = np.array([[0.3, 0.4], [0.7, 0.6]])

child = mesh.add_fault(("Fault", fault_points))
```

`child` is a new, standalone mesh carrying boundaries `FaultPlus` and
`FaultMinus` — two geometrically coincident copies of the fault — plus
the record of which degree of freedom pairs with which. The source mesh
is untouched: when the fault moves, call `add_fault` again on the base
mesh at the new position (nothing is cumulative).

A frictionless (perfectly slippery) fault is then a one-line boundary
condition, and the solve is an ordinary `solve()`:

```python
v = uw.discretisation.MeshVariable("V", child, child.dim, degree=2)
p = uw.discretisation.MeshVariable("P", child, 1, degree=0,
                                   continuous=False)
stokes = uw.systems.Stokes(child, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

x, y = child.X
for wall in ("Bottom", "Top", "Left", "Right"):
    stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)   # far-field shear

stokes.add_fault_bc(0, boundary="Fault")            # frictionless
stokes.solve()
```

The no-opening constraint $[\mathbf v]\cdot\hat n = 0$ is always imposed
strongly (the measured normal jump is machine zero), and with `conds = 0`
the shear traction on the fault is exactly zero, so slip develops freely
— the stress-driven crack. Read the slip through the pairing:

```python
from underworld3.utilities import fault_contact

s, V, leak = fault_contact.fault_slip(
    stokes, "Fault", stokes._rotated_freeslip_info)
# s: along-fault coordinate; V: slip rate; leak: normal jump (~1e-17)
```

```{warning}
Always read fault quantities through the pairing-based helpers
(`fault_slip`, `fault_pair_jumps`, `fault_normal_traction`). The two
sides of the fault are *geometrically coincident*, so any query by
coordinate — `uw.function.evaluate` included — sees one side only and
silently averages or picks arbitrarily.
```

## Quick start (3-D)

In 3-D the fault is a triangulated patch whose rim stays strictly inside
the domain. `uw.meshing.BoxInternalPatch` embeds a planar polygon patch
conformingly in a simplex box (a disc is a many-sided polygon), and
`split_fault` does the rest:

```python
from underworld3.utilities.fault_split import split_fault

patch = np.array([[0.5, 0.3, 0.3], [0.5, 0.7, 0.3],
                  [0.5, 0.7, 0.7], [0.5, 0.3, 0.7]])
mesh = uw.meshing.BoxInternalPatch(cellSize=0.05, patch_points=patch,
                                   patch_name="Fault")
child = split_fault(mesh, "Fault")
```

Everything downstream is identical to 2-D — `add_fault_bc`, `solve()`,
and the law functions below all work unchanged. In 3-D the slip is an
in-plane vector; read it with `fault_pair_jumps`:

```python
coords, jumps, normals = fault_contact.fault_pair_jumps(
    stokes, "Fault", stokes._rotated_freeslip_info)
leak = np.einsum("ij,ij->i", jumps, normals)        # ~ machine zero
slip_vec = jumps - leak[:, None] * normals           # the slip vectors
```

## The fault laws

The tangential condition on the fault is an interface constitutive law
$\tau(V, \sigma_n, \theta)$ relating the shear traction to the slip rate
$V$, the effective normal stress $\sigma_n$, and (for rate-and-state) the
state variable $\theta$. Laws are sympy expressions; their consistent
Newton tangents are derived symbolically, so every law converges with
full Newton — none of them is Picard-limited.

| law | call | condition |
|-----|------|-----------|
| frictionless | `stokes.add_fault_bc(0, boundary="Fault")` | $\tau = 0$ |
| viscous | `stokes.add_fault_bc(eta_f, boundary="Fault")` | $\tau = \eta_f V$ |
| Coulomb | `fault_contact.add_coulomb_fault_bc(stokes, mu, "Fault", sigma_n="reaction", V0=1e-5)` | $\tau = \mu\,\sigma_n\,\tfrac{2}{\pi}\arctan(V/V_0)$ |
| rate-and-state | `fault_contact.add_rate_state_fault_bc(stokes, f0, "Fault", a=..., b=..., V0=..., Dc=...)` | regularised arcsinh form, ageing law |

Notes on each:

- **Viscous**: `eta_f` is a viscosity per unit length — the
  zero-thickness limit of a shear band is $\eta_f = \eta_{\rm band}/w$
  with the *band's own* (weak-zone) viscosity over its width. Small
  `eta_f` approaches the frictionless crack; large `eta_f` *welds* the
  fault — and welding recovers the uncut continuum exactly, because the
  law penalises only the jump. The natural scale is $\eta/a$ (bulk
  viscosity over fault half-length), at which the fault slips at roughly
  half its free rate.
- **Coulomb**: `sigma_n` may be a prescribed number or `"reaction"`, in
  which case the effective normal stress is recovered from the
  no-opening constraint's own reaction — no auxiliary projection, and
  compressive stress generated by the flow feeds straight into the
  frictional strength. The reaction-fed value is SIGNED: where it turns
  tensile the strength is zero, and a real fault would OPEN — the
  bilateral no-opening constraint holds it shut instead (a tensile
  reaction), so check the sign of `fault_normal_traction` before
  trusting results on faults that see tension. `V0` is the regularisation velocity: choose it
  well below the slip rates the flow produces. Below it the fault sticks
  (creep $\sim V_0$); above it the traction saturates at
  $\mu\,\sigma_n$ and the fault slides at constant stress drop.
- **Rate-and-state**: the state $\theta$ advances *between* solves by
  the ageing law $\dot\theta = 1 - V\theta/D_c$, integrated exactly over
  the interval. Nonlinear fault solves go through `solve_with_fault`,
  and a time loop alternates solve and state update:

```python
fault_contact.add_rate_state_fault_bc(
    stokes, 0.6, "Fault", a=0.015, b=0.010, V0=1e-6, Dc=1e-2)

for step in range(n_steps):
    result = fault_contact.solve_with_fault(stokes, picard=2)
    monitor = fault_contact.update_fault_state(
        stokes, "Fault", dt, solve_result=result)
    # monitor -> 1.0 as the fault approaches steady state (theta V = Dc)
```

The recovered normal traction along the fault (also the quantity a
Coulomb law reads when `sigma_n="reaction"`):

```python
s, sigma_nn = fault_contact.fault_normal_traction(
    stokes, "Fault", stokes._rotated_freeslip_info)
# negative in compression; in 3-D the first return is pair coordinates
```

## Prescribed (kinematic) slip

Because the two sides are ordinary named boundaries, a *prescribed* slip
distribution is just Dirichlet data — no law involved. The slip must
taper to zero at the tips (2-D) or rim (3-D), which stay unsplit:

```python
import sympy

x, y = child.X
u = ((x - cx) * tx + (y - cy) * ty) / half_length     # along-fault coord
taper = sympy.sqrt(sympy.Max(1 - u**2, 0))            # elliptical profile

stokes.add_dirichlet_bc((vx_bg + s0 / 2 * taper * tx,
                         vy_bg + s0 / 2 * taper * ty), "FaultPlus")
stokes.add_dirichlet_bc((vx_bg - s0 / 2 * taper * tx,
                         vy_bg - s0 / 2 * taper * ty), "FaultMinus")
```

The elliptical taper is the constant-stress-drop crack profile; because
it vanishes at the tips, the shared tip vertex receives the same value
from both sides and no special treatment is needed.

## Practical notes

- **Pressure space**: use P2 velocity with P0 *discontinuous* pressure
  on split meshes. A continuous pressure space smears the pressure jump
  across the fault and measurably pollutes the near-fault stress.
- **Solvers**: split meshes carry no geometric-multigrid tail (the
  coarse levels never contain the fault), so the velocity block takes
  its algebraic-multigrid default. Set accuracy with
  `stokes.tolerance = ...` — never raw `ksp_rtol`, which the solver
  configuration overrides silently.
- **Conditioning**: the contact formulation is dramatically better
  conditioned than a thin weak inclusion of the same fault — there is
  no thin feature and no viscosity contrast for the Schur complement
  to fight (measured: 10 outer iterations vs 147 for a $10^{-4}$
  contrast band on the same mesh).
- **Curved faults need their analytic normal**: a curved trace is
  sampled as a polyline, and by default each fault node's normal is the
  average of its adjacent facet normals — exact on a straight fault, but
  zig-zagging at the sampling kinks of a curve. The no-opening
  constraint then forbids smooth slip past each kink, producing slip
  notches and normal-traction sawteeth that *grow* under mesh
  refinement. Pass the smooth curve's normal instead:

  ```python
  x, y = mesh.X
  # e.g. a circular arc about (cx, cy): the radial direction
  stokes.add_fault_bc(0, boundary="Arc",
                      normal=sympy.Matrix([[x - cx, y - cy]]))
  ```

  Same conventions as `add_rotated_freeslip_bc`: a sympy `1×dim` matrix
  in `mesh.X` (need not be unit length; it is normalised per node and
  sign-aligned to the split's Plus→Minus orientation), or a constant
  array. For a **digitized trace with no analytic formula** — real
  mapped faults — pass `normal="trace"`: the smoothed normal is built
  from the fault's own control polyline (central-difference tangents at
  the control points, tangent angle interpolated along each segment),
  which `add_fault` stores on the mesh. Every fault-law variant
  (`add_coulomb_fault_bc`, `add_rate_state_fault_bc`, ...) accepts the
  same `normal=` argument, and the slip/traction diagnostics read the
  same frame automatically. Measured on a sampled circular arc: the
  smooth normal reduces the kink sawtooth by an order of magnitude and
  restores convergence under refinement. On straight faults it changes
  nothing — omit it. And a deliberately *kinked* fault should not be
  smoothed: there the kink response is the physics.
- **Moving faults**: re-derive, don't update. Cut and split again from
  the static base mesh at the new fault position; transfer fields with
  the standard re-adaptation machinery.
- **Networks**: pass a list of faults to `add_fault`. Segments must not
  share vertices — represent a branch or crossing as offset segments
  separated by a ligament of one or two cell sizes.
- **Parallel**: in 2-D a fault may cross a partition seam through a
  pinned crossing vertex (handled automatically by `add_fault`); a
  fault running *along* a seam is refused. In 3-D, `split_fault`
  automatically REDISTRIBUTES first: the patch's cell star (plus one
  growth layer — a thin skin, not the refined band) is gathered onto
  the rank that already owns most of it, everything else stays with
  the load-balanced partition, and the split then works at any rank
  count. The cost is a bounded imbalance on the fault-owning rank
  (measured ~1.8x at np = 8 on a graded box). One current exception:
  several 3-D faults on one mesh in parallel are refused (the pairing
  does not yet migrate through the redistribution) — split networks in
  serial for now. All refusals are collective and name the actual
  problem.

## Current limitations

Refused loudly rather than mishandled: closed-loop faults (rings,
spheres); faults that reach the domain boundary (daylighting); junctions
sharing vertices; 3-D faults touching a partition seam. The design
documents in `docs/developer/design/` (`SPLIT_NODE_FAULT_METHOD_2026-08`
and `FAULT_CONTACT_DEPLOYMENT_2026-08`) record the method, the
validation benchmarks, and the roadmap for these extensions.
