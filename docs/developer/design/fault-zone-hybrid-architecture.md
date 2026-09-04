# Hybrid fault zones: contacts along the fault, TI zones at the junctions

Underworld3 represents a fault two ways, and neither is right everywhere.
This note records the design that uses both in one model, with the handover
between them chosen from the geometry rather than by the modeller.

Status: the zone assembly change described under
[Fusing the zone](#fusing-the-zone) is implemented. Everything under
[Choosing the handover](#choosing-the-handover-from-the-geometry) is proposed
and not yet built. Numbers quoted as measured come from the exploratory
campaign in `~/+Simulations/listric_extension/`.

## The two representations

A **zero-thickness contact** ([split-node](../../advanced/split-node-faults.md))
is a surface across which the velocity jumps. It is cheap, and it matches the
fault assumptions exactly: a fault of no width, carrying a slip discontinuity
and an interface law.

A **finite-width zone** ([`place_thin_volume`](../subsystems/conforming-surfaces-and-fault-zones.md))
is a band of weak material of a width you choose. It is more expensive, and it
applies where the contact assumptions fail — at junctions, at merges, anywhere
the fault has a width or the geometry has no unique centreline.

On a single straight fault the two agree to 3.5%. Their cost per solve, on the
same mesh, was measured at 3.2 s for the contact, 13.5 s for a
transversely-isotropic (TI) band, and about 296 s for an isotropic band of the
same width. The isotropic band is the one to avoid; the TI band buys back most
of the cost by making the weakness directional instead of resolving a
viscosity contrast in every direction.

## The seam defect

Abutting the two representations end to end does not work. A hybrid built that
way returned 74.6% of the continuous-contact reference — worse than either
pure representation on its own.

The mechanism is **tip pinning**. A contact that stops at the edge of a zone
terminates as a free crack tip, and slip is forced to zero there. The zone
then has to re-accelerate the material it receives, and the model loses the
slip that the pinned tip refused to carry.

A tip that ends *inside* weak material is not pinned. Running each contact one
cell **into** the zone recovers 96.3% of the reference, saturating near 99%
with further penetration. One mechanism accounts for the 74.6%, for the cure,
and for the residual seen in the earlier junction work.

This is the finding that makes a hybrid worth building: the handover is not
inherently lossy, it was being built at the wrong place.

### Two ways to slice, and only one of them makes a seam

Measured on the listric pair at a trace separation of 0.30, four settings on
one mesh (5110 cells in every case; slicing adds two, from duplication at the
split):

| setting | contacts | weak zone | solve | slip against control |
|---|---|---|---|---|
| control | none | everywhere | 20.4 s | — |
| sliced | to the handover | everywhere | 4.7 s | 99.1% / 99.2% |
| stripped | to the handover | only across the merge | 2.7 s | 92.0% / 75.4% |
| stripped, penetrating | one cell past it | only across the merge | 2.5 s | 92.6% / 86.7% |

The three sliced rows are one mechanism seen from three sides. Where the zone
is left intact, a contact's tip ends *inside* weak material by construction,
so it is never pinned and the answer is the control's. Where the zone is
stripped back to the handover, the tip ends exactly at the weak material's
edge — the abutting case — and slip falls to 75%. Running the tip one cell
further in recovers part of it.

This matters for what the slice criterion IS. If the zone is kept, slicing
cannot produce a wrong answer: slicing less is slower, slicing more is faster,
and slicing where the split machinery cannot go is an explicit refusal. The
criterion is a performance knob with a hard stop. Only the stripped variant,
which buys a further factor of about two, needs the handover to be
geometrically right, and needs penetration.

```{note}
Keeping the contacts *and* the zone is roughly three times faster than the
zone alone, and we do not yet know why. The natural explanation — that the
contact supplies the velocity discontinuity the solver would otherwise build
out of the viscosity contrast — is measurably wrong: the advantage is flat
across three decades of contrast (2.8x at 1e-3, 3.1x at 1e-5) and the
zone-only cost barely moves with it. Transverse isotropy had already removed
that conditioning penalty; the ill-conditioned form was the *isotropic* band,
at 296 s against the TI band's 13.5 s. The likely candidate is that the two
configurations take different solver paths with different outer iteration
counts. Settling it needs the rotated solve's own KSP instrumented —
``snes.getLinearSolveIterations()`` reads zero for it, because the rotated
path builds its own prefixed KSP.
```

## Fusing the zone

When a network's zones are thickened and resolved against one another in CAD,
the boolean that resolves them decides what the mesher has to honour.
`fragment` keeps every overlap piece as its own region, so the mesh conforms to
the boundary of the overlap. Where two faults converge tangentially — a splay
soling into a detachment, the listric case — that boundary is a lens closing
at the convergence angle, and the mesh resolves it as a chain of slivers. On
one measured pair the fragmented assembly meshed to a minimum angle of 0.13
degrees with 52 cells under 5 degrees, and the Stokes SNES diverged on it.

`fuse` returns the union as one region with no internal seam. The same
geometry meshed at 26.5 degrees minimum with no cell under 15, in fewer
triangles, and solved.

Nothing downstream needs those seams. A zone carries one label, and a cell's
fault properties are read from the `Surface` objects by proximity, not from
the CAD piece the cell was meshed in. The zone mesh never has to know which
branch a cell came from, and its not knowing is what lets the connection
region select itself.

`place_thin_volume` therefore takes `assembly={"fuse", "fragment"}` and
defaults to `"fuse"`, in both 2-D and 3-D.

```{note}
The CAD area gate did not need changing, contrary to the expectation this
work started from. A fused Y does have less area than the sum of its ribbons,
but `cad_area` is computed from the faces that *survive* the boolean, and
those are the union under either choice. The measured CAD-against-meshed
relative difference is `1e-16` for both.
```

## Choosing the handover from the geometry

After fusing, the zone is one region and there is no marker saying which part
of it is a well-defined fault and which part is a junction. We propose to read
that off the geometry.

For a point on a fault trace, let $d(x)$ be the distance from the trace to the
**zone boundary**, and let $w$ be the zone width.

- $d \approx w/2$ — the trace is the local medial axis of a single band. The
  slip surface is well defined, so **cut a contact here**.
- $d > w/2 + \text{tol}$ — the stem is wider than one band, so two or more
  faults overlap and no unique centreline exists. **Leave it to the TI zone.**

The rule imposes no hierarchy. It does not need to know which fault is senior,
it is symmetric in the branches, and it is computable directly from the fused
outline with machinery `Surface` already has in 2-D. It is the formal version
of the observation that a fat Y's centreline departs from each trace somewhere
along the stem.

Two other formulations are worth testing against it: the true medial axis of
the fused polygon, with the criterion becoming "the medial axis departs from
the trace by more than tol"; and the local zone width measured normal to the
trace. We prefer the distance-to-boundary form unless it misbehaves at the
tangency corner.

Cutting the contact inside the zone needs no new meshing primitive. `add_fault`
is a *cut*, not a placement, so it slices the zone's own cells and the zone
survives — cell counts went from 72 to 74 in the measured case, the increase
being duplication at the split.

### What owns a cell where two zones overlap

The criterion above says where to stop cutting contacts. It does not say what
the rheology should be in the stem, and the two representations answer that
question in opposite ways today.

A **contact** cannot overlap another contact. `fault_split` refuses a fault
that terminates on an already-split fault's slit, and gives the reason: a
shared point would clamp every arm's slip to zero, which is stiffer than a
true junction. So no node ever carries two interface laws — the network forces
the faults apart into the offset form first, and pays the gap. That gap is the
tip pinning described above.

A **zone** may overlap freely, and carries a single label, so the question
moves into the property lookup. `SurfaceCollection.compute_nearest_fields`
gives each node the normal and identifier of the nearest surface **vertex**,
so the nearer trace wins node by node. Two consequences worth deciding on
before the criterion lands:

- the partition boundary is the medial axis between the traces, which carries
  no physics, and the director flips discontinuously across it;
- it is nearest *vertex* rather than nearest *surface*, so the boundary moves
  when a trace is re-sampled without being moved (issue #544).

It is benign for a near-tangential merge, where both directors nearly agree.
It is not benign at a high-angle junction, where two faults produce a region
weak in **two** directions. That is orthotropic rather than transversely
isotropic, and no single director represents it, so choosing a nearest fault
chooses which of the two weaknesses to discard.

### The intended API

```python
net = uw.meshing.FaultNetwork([...])              # Surface objects
mesh = net.prepare(h=...).build(zone="fuse", slice="auto")
constitutive_model, director = net.ti_rheology(v, eta_1=...)
net.apply_contact(stokes)                          # only on the sliced pieces
```

`slice="auto"` applies the criterion above; `slice=None` gives the pure
finite-width model. The two share one mesh, which is what makes the comparison
between them clean.

### What landed (2026-08)

The whole-network end of this arrived first, in a slightly different
shape. `build(width=..., realisation="split"|"ti")` places one ribbon
band along every prepared piece and either cuts it or leaves it whole,
so the two representations share one mesh as intended; `apply(solver,
...)` imposes whichever was built, and `ti_fields` paints the weak-plane
viscosity and director. The realisation is a property of the whole
network, not yet of an individual piece — there is no per-fault
`slice="auto"` criterion, so a model that is sliced *here* and finite-width
*there* still has to be assembled by hand.

The director question moved rather than closed. Within one strand's
footprint the director is the unit normal of the nearest **segment** of
that strand's trace, so it no longer moves when a trace is re-sampled.
Which strand *owns* a cell is still nearest-sample over the concatenated
spines, so the partition boundary and the high-angle-junction objection
above are unchanged.

## What still has to be built

In dependency order, for 2-D:

1. Zone-boundary distance exposed so the criterion can be evaluated.
2. ~~The nearest-fault director~~ — landed as `FaultNetwork.ti_fields`
   (nearest segment within a footprint); the ownership question (issue
   #544) is untouched.
3. The criterion itself, and the `slice="auto"` wiring — i.e. a
   per-piece rather than per-network realisation.

Placing a contact tip inside a zone works: `add_fault` puts a vertex on every
control point of the trace, so a truncated trace terminates cleanly. That path
was blocked by a defect in which the placed vertex could be taken from a
neighbouring cell, leaving the cell that held the tip without a corner on the
line (issue #542, fixed). It appeared inside zones rather than on plain meshes
because a ribbon's vertex rows sit at exactly the half-width and manufacture
near-ties.

3-D is the same argument with more mechanical work. The `fuse` change already
applies. The centreline becomes a medial *surface*, and the criterion is
unchanged — distance from the patch to the zone boundary against $w/2$.
Cutting is `place_sheet` plus `split_fault` rather than `cut_along_lines`, and
the tip condition becomes a rim condition, which is likely harder.
`SurfaceCollection.transfer_normals` already works in 3-D, since triangulated
surfaces have genuine cell normals, so the director may come free there.

We do 2-D end to end before starting 3-D.

## How we will know it works

- **Straight-fault control.** One fault, a meshed zone in the middle,
  contacts either side. The answer is known: 100% of the continuous-contact
  reference, with no notch at either seam. Any implementation must reproduce
  it.
- **Self-selection sweep.** Vary the fault separation so that the correct
  partition of slip between the two faults changes, and require one fixed
  geometric rule to track a gmsh-union control across the range. Agreement at
  a single separation proves little, because that geometry may admit only one
  sensible partition.
- **Traction continuity.** $\tau_{nt}$ and $\sigma_{nn}$ are continuous across
  the zone boundary; $\tau_{tt}$ and $p$ may jump. Checking the jump at
  stations along the fault is an oracle that does not depend on the
  implementation.

```{warning}
The ratio of $\tau_{nt}$ inside the zone to outside it is **not** a valid
metric, and cost real time in the exploratory work. Traction is continuous
across the zone boundary, so that ratio compares different regions of the
domain and resolves onto a meaningless plane away from the trace. It also
changes with the discretisation: the same ratio read 10x at cell centroids and
2.7x projected to P1, because continuity blends the value across one to two
cells. Quote cell values for metrics and P1 for figures.
```

## Related

- [Conforming surfaces and fault zones](../subsystems/conforming-surfaces-and-fault-zones.md)
  — the placement family, including `place_thin_volume` and the `assembly`
  argument.
- [Fault contact deployment](FAULT_CONTACT_DEPLOYMENT_2026-08.md) — the
  offset-junction convention this design replaces at junctions.
- Issue #539 — the TI compliance tensor (the inverse has the same structure
  with $\eta \to 1/(4\eta)$), needed for plasticity's compliance-to-stiffness
  map and to infer strain rate from recovered stress.
- Issue #540 — the nearest-fault director in 2-D.
