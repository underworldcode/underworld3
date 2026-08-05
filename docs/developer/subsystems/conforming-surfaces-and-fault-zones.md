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
| two surfaces **closer than one element** | refused (inherent) | refused (implementation) — see below |
| what it does to the mesh | moves vertices onto the surface (`snap_frac`) and splits edges | deletes vertices near the surface and refills the hole |
| parallel | yes, and partition-independent | **serial only** — see below |

```{warning}
An earlier version of this section claimed placement carried two parallel
surfaces down to a tenth of an element. That was wrong. The second surface's
cavity was consuming the first, and the identity being asserted summed each
placement's own facet counts, so it could not see the damage. Read such counts
back off the RESULT mesh.

Measured on a 1/16 box, both surfaces checked intact afterwards: placing one at
a time accepts 1.5 `h` separation and refuses 1.0 `h`; the cut accepts 1.0 `h`
and refuses 0.5 `h`. **For closely spaced surfaces the cut is currently the more
capable of the two.**

The two limits are not the same kind. The cut's is inherent — converging flanks
cross the same edge and an edge can be split at one point, so no implementation
removes it. Placement's is an implementation limit: a cavity is cleared and
filled for one surface, and the cells carrying an earlier surface's facets are
held back so that surface survives, which eventually leaves no room. Lifting it
means placing both surfaces into a single cavity — the finite-width ribbon,
which is not built.
```

The construction is the same operation in 2-D on a curve and in 3-D on a sheet —
place, delete, refill — which matters because cutting tetrahedra along a surface
is an unsolved pattern problem while filling a cavity is standard meshing
practice. Only the 2-D half exists.

### How the cavity is filled

Clearing the vertices in the way leaves one hole, and what has to be triangulated
inside it depends only on how many ends of the surface reach the domain boundary:
none leaves an annulus, one leaves a disc, two leave a disc per flank. All three
are the same **walk** between two chains — the cavity ring outside, the surface
traversed *out and back* inside — advancing whichever chain is further behind in a
shared parameter.

The parameter is arc length around the surface's own boundary: down one flank,
around the tip, back up the other. Two cheaper choices were tried and both fail.
Arc length along each chain measures the cavity ring's *wiggle* rather than its
progress, and the two rings drifted four surface widths apart. Position along
strike is discontinuous at the tips, which is exactly where the difficulty lives.

At a zero-thickness tip the two flanks meet at a point, so the parameter would go
flat over the whole turn through 180 degrees and the walk would have nothing left
to order by. The turn is therefore given a window of the parameter to itself, one
point spacing wide, interpolated by **angle about the tip** — which turns the tip
into a fan of one placed vertex against many cavity vertices.

The walk has a third move, and without it it wedges. A cavity ring is not convex,
so a corner of it can protrude into the region; clipping such a corner off as an
**ear** consumes both of its ring edges and lets the walk carry on. Where even
that fails the ring vertex is a *spike* of surviving mesh poking into the cavity —
the only triangle that would fill the notch uses a vertex the walk has already
passed — and the answer is to swallow that vertex and re-clear, which is what the
routine does, up to eight times. Measured over 100 random traces on a uniform mesh
and 100 on a graded adapt-on-top mesh: 2 and 8 failures respectively without the
growth step, none with it, and the total area exact to 2e-16 throughout.

### What it costs

The walk fills the cavity by parameter, not by shape, so the cells it leaves are
worse than the cut's before repair. Flipping (`reconnect.flip_to_reduce_max_angle`)
fixes that, and both repair passes refuse to touch a labelled edge, so the surface
itself comes through with the same facets. This is the normal composition, not a
workaround.

Placement is **serial**. It adds points, and a point added across a partition seam
needs the star-forest's leaf set extended — the chart-expansion rebuild, which does
not exist. A parallel call is refused rather than returning a mesh whose
star-forest is silently wrong. The cut remains the parallel path.

## Limitations

* **Two dimensions.** The 3-D mechanism is validated on single tets and small
  boxes but is not wired up.
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
