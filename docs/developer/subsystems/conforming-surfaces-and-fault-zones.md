# Conforming surfaces and fault zones

An internal surface — a fault, a material interface, the base of a sticky-air
layer — can be added to a mesh *after* the mesh exists, by splitting every edge
the surface crosses at the crossing point. The surface becomes a chain of element
edges, so no element straddles it and every element lies cleanly on one side.

```python
fault = uw.meshing.Surface("Fault", mesh, trace_points)
cut = mesh.add_conforming_surface(fault)

zone = cut.cells_supporting("Fault")            # the fault zone, per cell
eta = uw.discretisation.MeshVariable("eta", cut, 1, degree=0)
eta.array[:, 0, 0] = np.where(zone, 1.0e-3, 1.0)
```

## Why straddling elements are not a resolution problem

Stress is $\tau = 2\eta\dot\varepsilon$. Inside a linear element the discrete
stress is the *interpolated* viscosity times the *interpolated* strain rate, whose
cell average differs from the honest one by

$$-2\,\mathrm{Cov}(\eta, \dot\varepsilon)$$

per cell. That covariance is zero for any element lying wholly inside or wholly
outside the weak zone, and non-zero only for elements that **straddle** it. So
refining shrinks the straddling band but never empties it: the artefact is a
*representation* problem, not a resolution one. The cure is to stop straddling.

Measured on a 1/16 box with a viscosity step of $1 \to 10^4$ across a slanted
line:

| mesh | straddling cells | leak, continuous P1 $\eta$ | leak, cell-wise $\eta$ |
|---|---|---|---|
| uncut | 37 | 239.7 | 285.4 |
| cut | 0 | 298.7 | **0.0 exactly** |

Two things follow, and both matter:

* **the cut alone is not enough.** A continuous P1 viscosity leaks *more* on the
  cut mesh, because the nodes ON the interface are shared by both sides and a
  continuous field has to take one value there;
* **a cell-wise viscosity alone is not enough either.** On the uncut mesh it
  leaks 285. The cut is what makes a per-cell assignment *correct*.

The pairing is "cut **and** assign per cell".

:::{note}
Assign the contrast to a `degree=0` (P0) variable. `cells_supporting` returns
plex cell order, which is exactly the DOF order of a P0 variable, so it can be
assigned straight across. When rendering such a field, draw it as **cell** data —
interpolating between centroid DOFs fakes a smear across the sharp interface you
just went to the trouble of resolving.
:::

## The fault zone is the facet support

`cells_supporting(name)` returns the cells in the **support of the labelled
facets** — one element each side of the surface. It is not a geometrically
bounded region, and that is deliberate.

For a fault one element wide, the end cap (2-D) or edge band (3-D) that would
close a bounded region has extent equal to the **thickness**, so resolving it
would need $h \ll h$. You cannot have one-element thickness *and* an
independently specified geometric boundary. Deriving the zone from the facets
sidesteps the whole question: no cap, no band, no rim, no creases, and the
definition says nothing about dimension.

Measured properties:

* the zone is **exactly $2\times$ the facet count**. A cell carrying two labelled
  edges would have been cut in two, so no cell is double-counted and every facet
  contributes both its neighbours;
* it terminates automatically where the chain of facets ends — a fault tip needs
  no special treatment;
* the zone of a **network** is the union of its branches' zones, with no geometry
  to reconcile where they meet.

### The price, and why it is not a price

Thickness is no longer a physical parameter — it tracks $h$. Measured as the mean
distance from a zone cell's centroid to the trace, over the local cell size:

| mesh | thickness / $h$ |
|---|---|
| uniform 1/8, 1/16, 1/32 | 0.189, 0.183, 0.184 |
| adapted, `h_near` = 0.03, 0.02, 0.015 | 0.195, 0.202, 0.198 |

Constant across a 4× refinement and across the adapt metric. Under adapt-on-top
that is the *point*: put the surface at the finest adapted level and the fault
width becomes a **refinement parameter**, set locally by the metric and at
bounded cost.

```python
fault = uw.meshing.Surface("Fault", base, trace)
fault.discretize()
child = base.adapt(fault.refinement_metric_function(h_near=0.015, h_far=0.09,
                                                    width=0.06), max_levels=3)
cut = child.add_conforming_surface(uw.meshing.Surface("Fault", child, trace))
```

Adapt **then** cut. The child keeps its multigrid tail.

If the fault width matters physically *and* can be made much larger than $h$, that
is a different regime — bound it geometrically with two offset surfaces and an
explicit cap, which works today by chaining `add_conforming_surface`.

## Nothing below the child is cut

The surface exists on the **finest level only**. The mesh it is added to, and
every coarse multigrid level beneath it, are untouched and are reused as the
child's coarse tail.

That is the point of the stack-on formulation: fault geometry is a design
variable in an outer optimisation, so the surface has to be able to move and be
re-added against a base and a hierarchy that never change.

A coarse cut would buy nothing anyway. The custom-P hierarchy sets
`pc_mg_galerkin=both`, so every coarse operator is $P^\mathsf{T} A P$ formed from
the **fine** operator and inherits the material contrast whatever the coarse mesh
looks like. Measured on SolCx at contrasts of $10^2$ and $10^6$, cutting the
coarse levels changed the error in the fifth significant figure and the solve time
not at all.

## Snap or cut

A crossing landing close to an existing vertex would leave a sliver — worst case
measured, a cell of area $10^{-24}$ with a zero interior angle. So a crossing
within `snap_frac` of an edge's end, **measured along that edge**, moves the
vertex onto the surface instead of splitting beside it.

The along-edge measure is the one that matters: it is exactly the short side of
the sliver that would otherwise be created, and it carries no length scale, so the
same tolerance works on any mesh. The surface stays exactly where it was specified
either way — a snapped vertex moves *onto* it, never the other way about — so
what a larger tolerance costs is displacement of the surrounding mesh, not
accuracy of the interface.

What the slivers cost, measured with GAMG on a Poisson solve (CG iterations to
`rtol=1e-10`, 5,432 cells), alongside the worst angle of the cut:

| `snap_frac` | worst angle | CG iterations |
|---|---|---|
| uncut | 43.7° | 20 |
| 0.00 | 1.6° | 32 |
| 0.05 | 3.9° | 28 |
| 0.10 (default) | 6.6° | 23 |
| 0.20 | 13.9° | 21 |

Cutting without snapping costs 60 % more iterations; snapping buys it back. A
Lawson flip pass helps less (32 → 29 at `snap_frac=0`), so snapping is the lever
and repair is a second-order touch-up.

:::{warning}
Snapping is a **discrete** switch. As a surface sweeps across the mesh the
topology changes in jumps, which anything optimising over the surface's position
has to live with.
:::

## Tips and junctions

Both are the same problem: a distinguished point of the geometry that has to
coincide with a mesh vertex. Once it does, every branch arriving there hits the
already-legal case of "one crossed edge, one on-surface corner", and the cut
terminates cleanly.

```python
from underworld3.utilities.line_cut import pull_vertex_onto, cut_along_lines

dm = pull_vertex_onto(mesh.dm, junction)        # collective
for k, branch in enumerate(branches):
    dm, info = cut_along_lines(dm, [branch], label=f"F{k}", label_value=20 + k)
```

Prefer **pulling a vertex onto the tip** to snapping the tip to the nearest
vertex: measured on a 1/12 box, tip error 0.0000 against 0.0306, and a better
worst angle (8.79° against 5.29°). It costs mesh displacement rather than
geometric accuracy — the same trade as `snap_frac` — and the tip is where the
stress concentrates.

Y (branching), T (abutting) and X (crossing) networks all work, in serial and at
np=2/3/4. The union-of-cells zone **bulges** where branches meet, because the fan
of cells around the shared vertex is picked up by each branch. That is accepted:
intersecting faults are transient — if they slip they change the geometry — so
junction volume need not be resolved exactly, and a widened damage zone at a
junction is not physically unreasonable.

:::{warning}
Do **not** test a tip by asking whether cells straddle the infinite line. A fault
ending at a vertex deliberately does not separate the material there — you can
walk around the tip through the fan of cells — so those cells legitimately span
the line while the fault crosses none of their interiors. Assert instead that
consecutive on-fault vertices are joined by a **labelled** mesh edge, and that
nothing beyond the tip is labelled.
:::

## What is refused

Each of these is a case the cut cannot handle correctly, and each would otherwise
give a mesh that looks plausible and still leaks stress:

* an edge crossed more than once — it can only be split at one point;
* a triangle entered but not left, which means a surface ends inside the mesh
  without a vertex at the tip;
* a triangle crossed three times;
* nothing to cut at all;
* snapping that inverts a cell, or that will not settle.

Every one of these is raised **collectively**. A rank-local refusal aborts one
rank while its peers walk on into the next collective and block there, turning a
clear error into a hang.

## The other way: place the surface, do not cut for it

Everything above describes **cutting** — splitting the edges the surface crosses.
There is a second implementation,
`underworld3.utilities.place_surface.place_along_lines`, which reaches the same
end by the opposite move: it asserts the surface's **own points** as mesh
vertices, deletes the mesh vertices in the way, and retriangulates the cavity so
that the placed segments survive as element edges.

Both produce a chain of labelled facets with no straddling cell, and both leave
the base mesh untouched. What differs is which restrictions come with the method.
Every one of the cut's refusals above is a consequence of *splitting*, and none of
them applies to placing:

| | cut | place |
|---|---|---|
| a surface ending **inside** the mesh | refused — a triangle entered and not left has no split that represents it | the tip is a placed vertex like any other, and the cavity closes round it |
| surface **finer than the local `h`** | impossible — the surface's vertices *are* the mesh's crossings | `spacing` is a parameter |
| two surfaces **closer than one element** | refused (inherent) | refused for zero-thickness pairs; a close PAIR is one finite-width zone — use `place_thin_volume` |
| what it does to the mesh | moves vertices onto the surface (`snap_frac`) and splits edges | deletes vertices near the surface and refills the hole |
| parallel | yes, and partition-independent | 3-D yes (gather-first, partition-independent); 2-D serial |

```{warning}
An earlier version of this section claimed placement carried two parallel
surfaces down to a tenth of an element. That was wrong. The second surface's
cavity was consuming the first, and the identity being asserted summed each
placement's own facet counts, so it could not see the damage. Read such counts
back off the RESULT mesh.

Measured on a 1/16 box, both surfaces checked intact afterwards: placing one at
a time accepts 1.5 `h` separation and refuses 1.0 `h`; the cut accepts 1.0 `h`
and refuses 0.5 `h`. The cut's limit is inherent — converging flanks cross the
same edge and an edge splits at one point. Placement's was an implementation
limit, and it is lifted the way that paragraph predicted: two flanks closer
than a cell are one **finite-width zone**, placed into one cavity by
`place_thin_volume` below.
```

The construction is the same operation in every dimension — place, delete,
refill — which matters because cutting tetrahedra along a surface is an
unsolved pattern problem while filling a cavity is standard meshing practice.
All three forms exist: the 2-D curve (`place_along_lines`), the 3-D sheet
(`place_sheet`), and the thin volume in both dimensions (`place_thin_volume`).

### How the cavity is filled

Clearing the vertices in the way leaves one hole, bounded by the cavity ring
(2-D) or shell (3-D). The fill is delegated to **gmsh** and gated per call,
never trusted: the ring or shell goes in as a discrete entity carrying its
exact segmentation, the surface as a second discrete entity — embedded, its
ends or rim free where they lie inside the volume — and the call is refused
unless every constrained node comes back bit-identical, every input segment
or triangle survives in the fill, and nothing is inverted. On top of that the
caller gates conformity, the global Euler number, exact volume conservation,
and every previously embedded surface's facet count, re-read off the result.

A hand-rolled walk filled the 2-D cavity for one development generation — an
arc-length parameter around the surface's boundary, an angle-interpolated fan
at each tip, an ear-clipping third move. It was retired when the 3-D sheet
proved the fill could be delegated under gates: one fill mechanism for every
dimension, and the tip — the walk's hardest case — is gmsh's ordinary
free-end embed. The gmsh fill is also better shaped: the graded-mesh fixture
that reliably gave the walk cells under 15° comes out of the delegated fill
with none.

### What it costs

Flipping (`reconnect.flip_to_reduce_max_angle`) and deleting refuse to touch
a labelled edge, so a placed surface comes through any later repair with the
facets it was placed with. The fill itself rarely needs repair (see above),
but the composition remains sound.

The 3-D forms are **parallel** by gather-first: the surface's region is
redistributed onto one rank, the serial carve-and-fill runs there, and every
rank rebuilds its chart collectively (uninterpolate + `DMPlexInterpolate`).
The gathered region is rank-interior, so nothing the surgery deletes or adds
is shared, and the result is partition-independent by construction — measured
bit-identical over np = 1..5. Every refusal is collective: all ranks raise
the same error, or none does. The 2-D forms are serial; a parallel call is
refused rather than returning a mesh whose star-forest is silently wrong.

What the gather moves, and why, is base mesh — never the surface. The
surgery deletes base cells around the surface and creates new ones in the
cavity, and the rebuild carries the old star forest over by renumbering, so
no point the surgery deletes or creates may be shared. That needs exactly
three layers of base cells on one rank: the cells the carve drops, their
vertex star (the ring the fill attaches to), and one more layer so the
ring's own points are unshared. The mark covers the dropped cells (the
victims within the clearance plus the crossed cells' vertices, within one
cell diameter of the surface) and `_gather_region` grows the star and the
layer from it. The result is a shell about three cells thick around the
surface; a region whose shell is already interior to one rank is placed
there with nothing moved. The moved count is reported as
`info["n_gathered"]`, and `ptest_0855` bounds it by the cells within three
median cell diameters of the zone.

The gather is one-way: the moved cells stay on the surgery rank, together
with the cells the fill creates, so that rank carries the shell plus the
band as extra load. That is the accepted trade (extra load on one rank in
exchange for no communication during the solve), and it is proportional to
the surface, not the domain.

A network is gathered per region, not as a whole. The assembly's connected
components (zones fused through shared faces are one component; zones a
domain apart are separate ones) are marked as separate regions, regions
whose shells touch are merged, and each region goes to the rank that
already holds most of it, or stays where it is when its shell is already
interior to one. The surgeries then run concurrently, each owning rank
carving and filling its own components; the collective rebuild sews them
all at once. Two patches a domain apart at np=2 move 304 cells this way
where one region for the pair moved 8451, and `info["n_regions"]` and
`info["n_moved"]` report it. The split's own redistribution follows the
same regions (`split_faults(..., groups=...)`, which the network passes
from `info["embedded_regions"]`), so what the placement kept apart is not
gathered together afterwards. The outcrop and ladder paths keep one
region: their bowl, cap and extrusion machinery is single-rank.

One limit remains: a small domain cannot hold a shell at all (three base
cells reaching the walls is the whole box, which is what the
crossing-patches test fixture does), so that fixture measures correctness
only, never balance. The measurements, the design decision on cutting long
faults, and the layout throughput test are recorded in
`../design/fault-parallel-placement-2026-09.md`, the governing note for
parallel placement.

## The thin volume: finite-width zones, junctions in the volume

`place_surface.place_thin_volume(dm, patches, width)` embeds a layer of real
thickness rather than a zero-thickness surface — the finite-width fault
representation, where the width `w` is a mesh parameter and constitutive
(`V = 2 ε̇ w`), not something that tracks `h`. Sub-`h` widths are supported
and measured (`w = h/4` passes every gate).

The construction is "mesh the whole lot, then embed", and the two stages are
forced by kernel: the OCC booleans — the only operations that resolve
fault–fault intersections — see only CAD entities, and the cavity fill
honours only discrete ones. So the network's patches are thickened by
`±w/2` and resolved against one another **together** in OCC (a junction
becomes ordinary volumes of the union — no geometric junction treatment, the
rheology decides), the assembly is meshed standalone at layer scale, its
boundary skin is extracted, and the meshed assembly is embedded whole: a
cavity is carved around it and gmsh fills the annular gap with the skin as an
interior **hole** in the fill volume, both surfaces verbatim.

Which boolean resolves the overlaps is the `assembly` argument, and the
default is `"fuse"` — the union as **one** region. The alternative,
`"fragment"`, keeps every overlap piece as its own region, so the mesh must
conform to the boundary of the overlap; where two zones converge
tangentially that boundary is a lens closing at the convergence angle, and
the mesh resolves it as a chain of slivers (measured on a ribbon soling into
another: minimum angle 0°, 22 cells under 5°, against 37° and none for the
fused union). Nothing downstream needs those internal boundaries: the zone
carries a single label, and a cell's fault properties are read from the
`Surface` objects by proximity, not from the piece it was meshed in. Ask for
`"fragment"` only when the boundaries between overlapping zones are
themselves the object of interest.

In the result the layer's **cells** carry `(label, value)` — the zone exists
to hand cells to the rheology — and the skin's faces carry
`(label + "_skin", value)`. The split is deliberate: a label stratum
containing cells is a *volume* label and invisible to the interface
machinery, so it is the skin label that makes later surgery hold its cavity
clear of the zone.

In 2-D the same call takes polylines and produces ribbons — one
**mitre-joined outline polygon** per polyline. (Per-segment quads fragmented
together were measured to sliver at every kink: the overlap lens on the
kink's inner side meshes at ~2°; a single outline has no internal seam to
lens.) 3-D patches must currently be planar polygons; curved manifolds
arrive with the fault-preparation lifecycle work.

Two lessons of the carve, both dimensions: the reach is a **length**,
`max(clearance·h, 0.6·width)`, so the cavity covers the layer's own
half-width however sub-`h` the layer is; and the cavity around a *fat*
object can **pinch** — the drop set is grown at every non-manifold shell
edge or pinch vertex until the boundary is simple, which only enlarges the
fill.

## Outcropping faults: meeting the free surface

Plate-boundary faults intersect the upper surface — the science case this
machinery serves — and both representations support it under the
**specify-long contract**: define the fault generously PAST the domain, and
prep clips it. A sheet is clipped triangle-by-triangle (cut points exactly
on the plane); a zone's assembly is intersected with the box in OCC. The
trace left on the wall becomes the **outcrop** (a sheet's chain, a zone's
band), the cavity opens onto that wall as a bowl, and the patch of wall
over the cavity — the **cap** — is remeshed to conform: pre-meshed by the
2-D fill (rim verbatim; the outcrop chain embedded, or the band outline as
a hole) and handed to the 3-D fill as a *discrete* surface. Two measured
rules make this sound: a geo surface bounded by a discrete rim RESAMPLES
the rim (so caps must arrive discrete), and the cap's new wall faces are
relabelled *explicitly* with whatever the replaced faces carried, full
closures included — joins cannot recover new points, and an unlabelled
boundary patch silently loses its Dirichlet conditions. Once placed,
conformity is topological: the outcrop's nodes ARE surface nodes and ride a
deforming surface; precision matters only at (re)placement.

Refusals, stated: one flat wall per object (box-edge outcrops and
multi-wall contact refuse); curved or deformed tops refuse (the cap's
manifold-prep slot); removal of an outcropping object is not yet wired.

## Limitations
* **An essential boundary condition on the surface is not sound** under the
  geometric multigrid hierarchy. The coarse levels do not carry the surface, so
  the condition constrains the fine level and *zero* coarse degrees of freedom,
  and the coarse operator is singular where custom-P needs it not to be. A
  **material contrast** across the surface — the case this is built for — needs no
  condition on the facets at all and is unaffected.
* **Surface integrals do not work on an embedded surface**, however it was
  created, so flux and traction recovery on one is not yet available.

## See also

* {doc}`meshing` — mesh construction and `Surface`
* {doc}`mesh-metric-redistribution` — the adapt metric that sets the local `h`
* `underworld3.utilities.line_cut` — the cutting mechanism
* `underworld3.utilities.place_surface` — the placing mechanism
* `underworld3.utilities.reconnect` — the repair passes, and `rebuild_cavities`,
  the rebuild both deletion and placement go through
