# Reconnection and Delaunay adapt-on-top

Status: **investigation, 2-D prototype measured (2026-07-29).** Prototype and
raw numbers in `~/+Simulations/mesh_reconnection_study/`. No `src/` change yet.

## Why look at this

`mesh.adapt()` refines by **subdivision** — newest-vertex bisection
(`engine="nvb"`) or PETSc longest-edge (`engine="sbr"`). A subdivision engine
decides *where the new point goes* and *how the cells reconnect* with a single
rule and may never re-wire an existing simplex. Three limits follow: the
conforming closure refines cells the metric never asked for (measured halo 45.8 %
on the 3-D fault band); element shape is inherited from the base and can never be
improved; and there is no coarsening and no anisotropy.

The goal of adding reconnection is **usability, not mesh quality**. MMG/ParMmg
(`mesh.remesh()`) already produces better-shaped elements than any local operator
will, and we are not trying to beat it. What it costs is control: it repartitions
the whole mesh, so every call destroys the decomposition, the point-SF, the MG
hierarchy and the parent/child lineage, at a cost that scales with the whole mesh
rather than the adapted region. A local, rank-respecting operator that carries
some load imbalance is the better trade inside a running model.

Two constraints that used to forbid non-nested adaptation are already lifted:
custom-P MG transfers are built from **coordinates**, not nesting
(`utilities/custom_mg.py:54,156,515`), and any conforming simplex mesh keeps the
`exact` point-location capability. And `(coords, cells) → DMPlex → Mesh` is
already a production path (`utilities/nvb.py:673`, driven from
`discretisation_mesh.py:7296`).

## Finding 1 — a flip cannot be a `DMPlexTransform` (confirmed)

This was the assumption the whole parallel design rested on, so it was checked
first. It holds.

`DMPlexTransformGetCone_Internal`
(`petsc/src/dm/impls/plex/transform/interface/plextransform.c:1443-1497`) finds
every cone point of every produced point by **descending from the single source
point `p`** (line 1463 `pp = p`; the loop at 1474-1490 walks `pcone[pcp]`), and
identifies the new point as `(parent point, replica)` via
`DMPlexTransformGetTargetPoint` at line 1495. A child's cone can therefore only
reference points in the **source point's own transitive closure**. A 2↔3 flip
produces tets using the apex of the *other* parent tet, outside that closure.
The `celltransform` op signature (`dmplextransformimpl.h:22`) and the
`offset[ct/rt][ctNew]` numbering scheme say the same thing: every new point
belongs to exactly one old point. `DMPlexTransform` is structurally a
**subdivision** framework.

**Consequence.** The NVB Route-B precedent (`nvb_transform.c`, which inherits SF
propagation and the parallel closure from PETSc) does not carry over. There is no
C-transform fallback. The parallel route for reconnection must be
**freeze the seam** — forbid any cavity containing a cell incident on a shared
plex point, so every shared point is untouched, the point-SF is inherited
verbatim, and each rank rebuilds its local DM alone. No cross-rank closure, no SF
reconciliation, no collective fixpoint.

## Finding 2 — reconnection repairs shape but not a bad point set

The study set out to test a specific proposal: centroid (Alfeld) placement is
cheap, local and closure-free, and was ruled a shallow tool only because of
shape (3-D: max dihedral 179.6°, 70.7 % of cells below `q = 0.1`, manufactured
Poisson error stalling at 0.119→0.122→0.124→0.124). The premise was that the
1→3 star split is a bad *connectivity* choice, not a property of centroid
*placement*, and that deciding connectivity afterwards by a Delaunay criterion
would free the placement.

**The premise is wrong.** Reconnection helps a great deal and is still not
enough. 2-D, same size field, same P1 solver, only the engine differs; in-band
interpolation error (no solve involved):

| engine | h=0.08 | 0.04 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|---|
| nvb (bisection) | 0.2819 | 0.1750 | 0.1029 | 0.0651 | **0.0524** |
| centroid raw | 0.2819 | 0.2285 | 0.1843 | 0.1549 | **0.1469** |
| centroid + flip | 0.2819 | 0.1827 | 0.1358 | 0.1171 | **0.1166** |

Flips take centroid refinement from 8.06 % to **0.00 %** of cells below `q=0.1`,
and the 99th-percentile max angle from 175.5° to 132.4°. The in-band FE error
goes from *diverging* (0.898 → 1.374 with increasing DOFs) to flat (~0.78). But
the plateau is still **2.2×** bisection's.

### The structural reason, measured

A centroid star split leaves the parent's three edges untouched, so an original
edge inside a refined region can never be shortened by further centroid
refinement. Survival of the 178 base edges inside the refinement band:

| engine | surviving |
|---|---|
| centroid raw | **178/178 (100 %)** |
| centroid + flip | 81/178 (46 %) |
| nvb (bisection) | 42/178 (24 %) |

A flip *can* remove such an edge — that is exactly why flips help — but only
about half are removable, because a flip needs a convex quad and only exchanges
one diagonal for another. Delaunay optimises connectivity **given the points**;
centroid points are simply the wrong points.

### A live measurement trap this exposed

The production marking criterion is `h = sqrt(2A)` (`NVBMesh.centroids_h`). It is
the right proxy for bisection, which shrinks area and diameter together, and a
**misleading** one for any area-reducing split. On the centroid mesh the area
proxy reads `h = 0.0102` while the median in-band **diameter is 0.0331** — a 3.2×
overstatement. The size field is satisfied and the mesh is not resolved. Any
future engine that is not pure bisection must mark on the diameter.

## Finding 3 — the corrected design, and it is competitive

Keep the part of the proposal that was right (closure-free placement) and fix the
part that was wrong (place points that shorten diameters):

> **`edge-split + flip`** — mark on the cell diameter; insert the midpoint of the
> longest edge, splitting that edge in **both** incident cells; then run the
> Delaunay flip pass.

Splitting the shared edge in both incident cells is conforming by construction —
no hanging node, so **no conforming closure and no LEPP chain**. Unlike bisection
we never demand that the neighbour split its *own* preferred edge. The green
cells this leaves are badly shaped, and repairing them is precisely the flip
pass's job. In-band interpolation error:

| engine | h=0.04 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|
| nvb (bisection) | 0.1750 | 0.1029 | 0.0651 | 0.0524 |
| **edge-split + flip** | **0.1136** | **0.0677** | 0.0626 | 0.0610 |

and in-band FE error 0.089–0.18 against bisection's 0.23–0.58. Base-edge survival
in the band is 22 %, matching bisection's 24 %.

Marginal value of the flip pass on this engine, at the same size field:

| h_near | cells no-flip → flip | in-band error no-flip → flip |
|---|---|---|
| 0.01 | 1935 → 1592 (**−18 %**) | 0.0670 → 0.0626 (**−7 %**) |
| 0.005 | 4083 → 3376 (**−17 %**) | 0.0658 → 0.0610 (**−7 %**) |

Fewer cells *and* lower error.

## Correction — the earlier 2-D comparison was confounded

The tables above compare engines at the same nominal `h_near`. **That is not a
valid comparison** and the first version of this note drew a conclusion from it
that does not survive.

The engines mark on different quantities: NVB on `h = sqrt(2A)` / `(6V)^(1/3)`
(its own `centroids_h`), edge-split on the **diameter**, which is always the
larger number. The same nominal target therefore asks edge-split for a finer
mesh. In 3-D the effect is severe — 36 569 tets against NVB's 9 780 for the same
`h_near`. Error must be compared **at matched DOF**, not at matched nominal
target.

Re-run in 2-D with each engine's `h_near` bisected to land on ~1700 cells:

| engine | h_near | cells | q_med | q<0.1 | ang p99 | in-band interp | in-band FE |
|---|---|---|---|---|---|---|---|
| nvb bisection | 0.0101 | 1718 | 0.862 | 0.00 % | 118.8 | 0.0666 | 0.2674 |
| centroid raw | 0.0089 | 1704 | 0.592 | 11.09 % | 176.2 | 0.1525 | 1.2607 |
| centroid + flip | 0.0089 | 1704 | 0.874 | 0.00 % | 132.9 | 0.1170 | 0.7858 |
| edge-split, no flip | 0.0166 | 1716 | 0.855 | 0.00 % | 136.2 | 0.0725 | 0.1318 |
| **edge-split + flip** | 0.0152 | 1676 | **0.890** | 0.00 % | **115.1** | **0.0620** | **0.0826** |

The conclusions hold at matched DOF — edge-split + flip is 7 % better than NVB
on interpolation error and **3.2× better** on in-band FE error — but they now
rest on a fair comparison. Note also that in 2-D the flip pass is worth having:
it improves interpolation error 14 % and FE error 37 % at equal cell count.

## Finding 5 — 3-D: placement is the whole story, reconnection is not

The 3-D case (unit box, `cellSize=0.4`, `refinement=1`, dipping fault at 60°,
matching `~/+Simulations/nvb_3d_adapt_evaluation/`). Work-precision sweep,
in-band FE error against cell count:

| engine | observed rate | (P1 ideal in 3-D: `N^-0.67`) |
|---|---|---|
| nvb (bisection) | `N^-0.55` | suboptimal |
| centroid | `N^-0.31` | badly suboptimal — this is the stall |
| edge-split | **`N^-0.67`** | **optimal** |
| edge-split + flip | **`N^-0.69`** | **optimal** |

At matched DOF (~7 000 cells) edge-split gives 0.193 against NVB's ~0.35 —
**1.8× better**, and it is the only engine achieving the optimal P1 rate.

**But reconnection buys almost nothing in 3-D.** On the good engine, at
`h_near = 0.07`:

| | edge-split | + flip |
|---|---|---|
| cells | 26 632 | 26 557 |
| in-band FE error | 0.0862 | 0.0851 (−1.3 %) |
| cells with q<0.1 | 0.96 % | 0.94 % |
| runtime | 3 s | **116 s (39×)** |

And on the centroid engine, flips move `q<0.1` 41.2 % → 27.8 % but leave the
error unchanged (0.6154 → 0.6192) and the max dihedral identical at 178.7°.

**Why the 2-D result does not carry over.** 2-D Lawson flips reach the *unique*
Delaunay triangulation — a global optimum for the given points. The 3-D 2↔3 /
3↔2 flip set is a weak local search that cannot escape slivers, and a Delaunay
tetrahedralisation contains them anyway. The kernel test shows this directly: a
Delaunay tet mesh of a random cloud has `q_min = 1.9e-3` with 10 % of cells
below `q = 0.1`, and 125 quality-gated flips left `q_min` **exactly unchanged**.

This is the outcome flagged as the real risk before the port, and it is
confirmed. The literature agrees: clearing slivers needs flips *plus* smoothing
*plus* insertion/deletion (Klingner & Shewchuk), not flips alone.

## Finding 4 — the price of preserving the decomposition

Freezing the seam (the synthetic partition at `x = 0.5` deliberately **crosses**
the refinement band — the worst case):

- 13.2 % of base cells frozen;
- on `edge-split + flip` the cell-count benefit survives (3395 vs 3376) but
  accuracy costs ~10 % (0.0673 vs 0.0610) — freezing eats roughly the whole
  *accuracy* benefit of flipping while keeping the *cell-count* benefit;
- on the centroid engine the freeze is far more damaging (in-band FE 1.115 vs
  0.802), because that engine depends on flips for basic shape repair.

**Design rule that falls out: reconnection must be a polish, never load-bearing.**
An engine that needs flips to be correct will suffer at a partition seam; an
engine that uses flips to be *better* degrades gracefully. This is a stronger
argument for `edge-split + flip` than its error numbers.

## What survives reconnection downstream

| consumer | pure flips (vertex set fixed) | point insertion |
|---|---|---|
| `child._adapt_prolongation` (exact ½,½, `nvb.py:249`) | survives — matched by *vertex coordinate identity*, which flips do not touch; must be **re-captured** after the DM rebuild, not re-indexed | invalid for new vertices → geometric builder |
| `child._adapt_parent_cells` (any-degree, #425) | invalid — a flipped cell can straddle two coarse cells | invalid |
| `barycentric` / `rbf` custom-P | fine (coordinate-based) | fine |
| `_location_capability` | stays `exact` | stays `exact` |
| boundary / region labels | preserved iff labelled facets are locked | same |
| co-partitioning invariant | preserved iff the seam layer is frozen | same |

Secondary payoff worth measuring later: `custom_mg.py:60-89` records
Delaunay-vs-mesh cell agreement at 58.8 % (3-D uniform) and 17.1 % (3-D adapt
child). A Delaunay mesh would push that toward 100 % on a convex domain, making
the geometric P1 builder near-exact and attacking the root cause of the #424
zero-column failure rather than the symptom.

## Non-negotiables for any implementation

- **Exact predicates.** Orientation and in-circle/in-sphere must be exact in
  sign. Naive float determinants give inconsistent flip decisions and a
  non-conforming mesh. The prototype uses a float filter with a `Fraction`
  fallback; production wants Shewchuk's adaptive expansions.
- **Locked facets.** Any facet carrying a boundary label, a region interface or
  a registered `Surface` must never be flipped and no cavity may swallow one.
  This is the constrained-Delaunay part and it is what protects faults and
  material interfaces.
- **Orientation guard.** Reject any modification producing non-positive
  area/volume (precedent: the snap guard at `discretisation_mesh.py:7217`).
  `to_dm` also requires CCW winding (`nvb.py:687`).
- **Mark on the diameter, not `sqrt(2A)`** — see Finding 2.
- **Never mutate a live mesh's topology in place.** Go `arrays → to_dm → new
  Mesh`, as `adapt()` does; the in-place route hits the known `_nav_coords` /
  face-control-point staleness traps (issues #286, #135).

## Finding 6 — end-to-end: TI weak-plane Stokes on an edge-split child

The engine carries a real solve. Same shear box, fault, and constitutive model as
the validated reference (`~/+Simulations/shear_box_fault_study/shear_box_fault_ti.py`);
only the refinement engine differs. Irregular base, 1056 cells, `max_levels=3`.

| | cells | q_med | q<0.1 | in-band slip (TI − uniform) |
|---|---|---|---|---|
| nvb child | 2081 | 0.944 | 0.00 % | 0.193 |
| edge-split child | 2868 | 0.959 | 0.00 % | 0.215 |

Both converge; the weak plane localises slip as it should
(`eta_1` verified to dip to exactly 1e-3 at the fault and recover by `|d| = 0.05`,
director = (0.866, 0.5) = the exact fault normal).

**Multigrid via the geometric route works, with one wiring step.** Measured
velocity-block preconditioner:

| child | velocity-block PC |
|---|---|
| nvb (from `base.adapt`) | `mg` — automatic, 8-mesh custom-P tail |
| edge-split, as built | `gamg` — falls back |
| edge-split + `set_custom_fmg(s, base._coarse_level_meshes(), field_id=0)` | **`mg`** |

The gap is only that `_custom_mg_coarse_meshes` is attached by `_adapt_nested`,
not by `Mesh` construction, so a child built from arrays has no tail until one is
attached. One line when this is wired in as a real engine — the transfers
themselves need no work, since every inserted vertex is an exact edge midpoint.

**What is NOT shown here.** Iteration counts. `getLinearSolveIterations()` returns
1–2 for every configuration, which cannot be right for a saddle-point solve and
means the counter is not capturing the nested KSP work. No MG-vs-GAMG performance
claim should be read off this run; the run establishes that the path *works*, not
that it is fast.

**Measurement errors made and fixed along the way** (both would have produced a
confident wrong answer):
- Slip was first sampled at ±3·`w_mech`, *outside* the weak zone, where the two
  sample points differ in `y` and the imposed simple shear dominates. The fix is
  to sample at `w_mech` and subtract a uniform-viscosity solve on the same mesh,
  so what is reported is the fault's contribution alone.
- The first run used `regular=True`, whose right-isoceles cells are a single
  similarity class that *both* engines preserve — every mesh scored q = 0.866 and
  the comparison could not discriminate. An irregular base is required.

## Finding 7 — `relax()` is the cheapest win, and it composes with flips

Everything above was measured **unrelaxed**. `mesh.relax()` (MMPDE in the ideal
reference frame) moves nodes without changing topology or the size distribution,
and its own docstring names the cause this study reached independently:
*"refinement chooses where new nodes go from combinatorics … never from geometry,
so a refined mesh carries needles and slivers that reflect the base mesh's
arbitrary choices"*. Flips answer that from the connectivity side; relax answers
it from the position side.

Relax-at-end, same base and size field, 2-D:

| engine / state | cells | q_med | q<0.1 | ang p99 | diam band | in-band interp |
|---|---|---|---|---|---|---|
| nvb | 1732 | 0.862 | 0.00 % | 118.8 | 0.0168 | 0.0651 |
| + relax | 1732 | 0.908 | 0.00 % | 108.1 | 0.0163 | **0.0535** |
| centroid | 1564 | 0.658 | 8.06 % | 175.5 | 0.0331 | 0.1549 |
| + relax | 1564 | 0.696 | 2.37 % | 170.6 | 0.0332 | **0.1791 (worse)** |
| centroid + flip | 1546 | 0.874 | 0.00 % | 132.4 | 0.0137 | 0.1171 |
| + relax | 1546 | 0.929 | 0.00 % | 113.7 | 0.0145 | **0.0869** |
| edge-split | 2580 | 0.835 | 0.00 % | 142.0 | 0.0101 | 0.0670 |
| + relax | 2580 | 0.881 | 0.00 % | 115.8 | 0.0104 | **0.0533** |
| edge-split + flip | 2238 | 0.894 | 0.00 % | 114.8 | 0.0109 | 0.0626 |
| + relax | 2238 | **0.941** | 0.00 % | **100.9** | 0.0108 | **0.0521** |

Three things follow.

**It is free and it is the largest single accuracy gain measured here.** Cell
counts are identical and band resolution is preserved to ~1 %, yet in-band
interpolation error drops **17–20 %** on every healthy configuration — more than
the flip pass buys (7–14 %).

**Flips and relax compose rather than overlap.** On the centroid child, flips take
0.155 → 0.117 and relax then takes it to 0.087; each gains where the other could
not. `edge-split + flip + relax` is the best mesh measured anywhere in this study
(q_med 0.941, ang p99 100.9°, interp 0.0521) and beats relaxed NVB (0.0535).

**Relax makes centroid refinement WORSE** — the only regression in the table
(0.1549 → 0.1791). Shape improves (q<0.1 8.06 % → 2.37 %) while accuracy
degrades: relax equalises shape and in doing so pulls nodes off where the feature
needs them, because the centroid point *set* was wrong to begin with. That is the
same conclusion as Findings 2 and 5, reached from a third independent direction —
**bad placement cannot be rescued, by connectivity or by position**.

## Recommended next steps

The investigation set out to add **reconnection**. What it actually found is a
better **placement** rule, twice over — and that reconnection matters in 2-D and
essentially not at all in 3-D. The recommendation follows that, not the original
premise.

1. **Do not pursue centroid placement.** Finding 2 closes it: the limitation is
   structural, and reconnection recovers less than half of it.
2. **`edge-split` is the candidate engine, and the flip pass is optional.** Mark
   on the diameter, split the longest edge in both incident cells. It is
   closure-free (the property that motivated centroid refinement), dimension-
   general with no pattern tables, and the only engine measured at the optimal P1
   rate in 3-D. Ship the flip pass as an **opt-in polish**: worth it in 2-D
   (−14 % interpolation error at equal cells), not worth it in 3-D (−1.3 % for
   39× the runtime).
3. **The 3-D sliver question is open and is not answered by flips.** If element
   quality in 3-D needs to improve further, the lever is smoothing between
   sweeps (`mesh.relax` exists) or insertion/deletion — not a better flip set.
   Worth knowing that `edge-split` already reaches 0.96 % of cells below q=0.1
   against NVB's 1.92 %, so this may not need solving at all.
4. **Then** wire `engine="delaunay"` (better named `engine="edge-split"`) into
   `_adapt_nested` alongside `"nvb"`/`"sbr"`. The MG hierarchy needs no new work:
   every inserted vertex is an exact edge midpoint, so the recorded ½,½
   prolongation applies unchanged, and the geometric route already handles
   everything else.

Deferred, explicitly not blocking: anisotropic (metric) predicate; edge collapse
for coarsening.

## Finding 8 — the repair pass, and three corrections to the findings above

Landed 2026-07-30 as `mesh.adapt(engine="edge_split", repair=True)`
(`utilities/reconnect.py`). Full record in
`~/.claude/plans/parallel-mesh-reconnection-flips.md`; raw numbers in
`~/+Simulations/mesh_reconnection_study/results_production_repair.txt`.

**Delaunay is the wrong acceptance criterion — in 2-D as well as 3-D.** Finding 3
and the recommendation above treat "flip to Delaunay" as settled in 2-D because
Lawson flips reach the unique Delaunay triangulation. The *operator* question is
settled; the *criterion* question was not. Delaunay maximises the **minimum**
angle and says nothing about the maximum, while the P1 interpolation bound depends
on the **maximum** angle (Babuška–Aziz). Measured: flipping a gmsh-refined mesh
towards Delaunay **raised** the 99th-percentile maximum angle from 126.8° to
129.3°, because gmsh optimises element shape rather than the empty-circle property
and its triangulation is locally non-Delaunay exactly where it chose a
better-shaped configuration. Since every UW3 mesh starts from gmsh, the production
pass gates on the angle directly, which makes it monotone — it can decline, but it
cannot degrade a mesh.

**The "−14 % interpolation error at equal cells" in step 2 above was a placement
effect, not a connectivity effect.** In the prototype the flip pass ran *inside*
the refinement loop, so the repaired arm had a different point set (a flip changes
which edge is longest, hence where the next vertex lands), and the cell-count
matching bisected the size field separately per arm. Isolated properly — repair
after refinement is cell-count neutral, since two cells become two cells and no
vertex is inserted — connectivity alone is worth **≤3 %** of core error. Run
between passes, the ~20 % is real and belongs to **placement**. That is Finding 2's
conclusion restated in the opposite direction, and it applies to reconnection's own
benefit as much as to centroid refinement's failure.

**A flip preserves the point chart, which collapses the parallel design.** Finding
1 stands — a flip is not a `DMPlexTransform`, so the DM must be rebuilt — but the
rebuild keeps the **identical point numbering**, because a 2-D flip adds and
removes no points: the quad keeps its four vertices, five edges and two cells, and
only the diagonal edge's cone and the two cell cones change. The point star-forest
therefore transfers verbatim, labels transfer by point id and coordinates transfer
unchanged. The "reconstruct the star-forest by matching untouched seam
coordinates" stage in *Non-negotiables* is unnecessary. Two things not to
re-derive: surgery on the source DM is impossible (`DMPlexSymmetrize` refuses to
run on a plex that already has supports, and nothing outside `DMDestroy` frees
them); and a triangle's cone convention is that closure vertex order is
anticlockwise, cone entry `i` is the edge joining closure vertices `i` and `i+1`
mod 3, and its orientation is `0` when the edge's own cone runs that way and `-1`
when reversed — getting it wrong does not raise, it silently yields wrong geometry.

What the pass is actually for is **shape on a poor base**: 99th-percentile maximum
angle 156.0° → 115.1° on an aspect-ratio-4 grid and 175.5° → 118.0° on a
non-Delaunay one, with slivers below q=0.1 going 3.84 % → 0.00 %; on a gmsh base,
124.7° → 120.5° and little else. The aspect-ratio-4 case is the argument for
building it at all: that base has a maximum angle of **90°**, ideal for P1, and
edge-split refinement *degrades* it to 156°, because bisecting the longest edge of
a stretched right triangle repeatedly manufactures obtuse cells. Refinement creates
the problem; only reconnection removes it.

Two further corrections. **Tier 0 — Rivara terminal-edge selection — was measured
and rejected**: strict terminal-only selection stalls (a marked cell's
longest-edge-propagation path walks towards *longer* edges, where the size field
asks for less, so the terminal edge it reaches is nominated by nobody), and
completing it with a LEPP walk gives core error identical to the existing veto rule
while reintroducing propagation. Its apparent 33 % win was an artefact of the wedge
size field, whose error window is far wider than the region it refines — use a
flat-core field and a core-only window. And the **seam cost is small and shrinks**:
frozen repair sites are 3.5 % at np=8 and 56k cells, halving with every halving of
the target size, because repair sites scale with the refined band while seam
crossings stay O(1). The 99th-percentile maximum angle recovers fully under a
frozen seam; the absolute maximum does not.

`repair=True` is opt-in because it gives up the one property `edge_split` has and
it does not: the refined mesh is no longer **partition-independent**, since which
cavities may be flipped depends on where the partitioner drew the seam. Conformity,
orientation, volume, labels and the star-forest stay exact at every rank count.

## Finding 9 — what the mesh is actually for: stress leaked across an interface

The reconnection work above optimises element *shape*. For a fault problem the
quantity that matters is narrower, and it turns out to rank the options
differently, so it is recorded here rather than left in a results file.

**The metric.** Stress is `τ = 2ηε̇`, and a P1 element forms it from the
interpolated viscosity times the interpolated strain rate, independently. So the
cell carries `mean(η)·mean(ε̇)` while the honest cell average is `mean(η ε̇)`. The
difference is

```
leak = 2[mean(η)mean(ε̇) − mean(η ε̇)] = −2 Cov(η, ε̇)
```

per cell: **zero** for any element lying wholly inside or wholly outside the weak
zone, positive only where an element straddles the transition with high strain
rate at one end and high viscosity at the other. It converges (falls monotonically
with resolution), which is the check that it measures the transition and not
something else. Note it lives strictly *inside* elements — plotting nodal `2ηε̇`
cannot show it, because at a node the two fields are sampled at the same point.

**A material-based marking rule loses to the plain distance size field.** Marking
cells by their internal η variation is the intuitive response and is measurably
worse per degree of freedom: N^-0.37 for the absolute jump, a complete stall for
the log ratio, against **N^-1.04** for the size field. The leak is spread across
the whole transition rather than concentrated in a few identifiable cells, so
there is nothing for a targeting rule to target, and uniform refinement of a
correctly sized band is the efficient answer. The log ratio additionally refines
the wrong end — it is largest where η is *smallest*, i.e. in the fault core, while
the leak lives on the outer flank where η runs 0.5 → 1.

**The optimal band width depends on which quantity you minimise**, and the
objectives disagree. Total leak: narrower is better. Leak *into the matrix*: an
optimum at a core half-width equal to the **influence width**, 2.6× better than a
narrow band. Straddling-cell count: wider is monotonically better. State the
objective before choosing the band.

**A step-edged margin confines the artefact.** `influence_function(profile="step")`
plus marking on the distance level set puts ~0 % of the leak beyond d = 0.03
against 11.4 % for a smooth blend, and converges slightly faster (N^-1.32 — the
1-D refinement buys more h per cell than the naive h-scaling argument suggests).
The price is concentration: total leak 2.5× higher and the worst single cell 20×
worse, welded into a one-cell collar on the interface. Good for a viscous solve,
awkward for a yielding model.

**Two exact fixes.** An element-wise constant (P0) viscosity makes `Cov(η, ε̇) ≡ 0`
on any mesh at any resolution — not reduced, zero. Aligning the interface with
element boundaries does the same. Both relocate the error from *inside* elements
to *where the element boundaries fall*, which makes node placement, not shape
repair, the lever — and hence `relax(pin_bands=...)` (Finding 10).

## Finding 10 — relaxation and interface tracking fight; pin the band

`relax()` on a mesh refined onto an interface makes it worse: manufactured stress
+77 %, and it stops being confined to the fault. The MMPDE mover optimises element
shape against an equilateral reference and knows nothing about where the material
changes, so it slides the small cells refinement placed on the interface off it.
It even *reduces* the straddling-cell count (1343 → 965) while making things
worse, because the survivors are larger — leak per straddling cell up 2.5×.

`mesh.relax(pin_bands=[surface])` (or `[(surface, offset)]` for a weak zone of
half-width `offset`) labels the cells the interface cuts and holds them fixed:
leak unchanged to five decimal places, confinement preserved, straddling count
identical, and the mover still reshapes the rest of the domain. `pin_halo`
defaults to 1 because pinning only the cut cells lets the mover pull on them from
outside.

Two implementation notes that are easy to get wrong and fail silently:
`pin_bands` must **merge** with `pinned_labels` rather than replace it, since the
default is "pin every named boundary"; and the band test uses the **signed**
distance at offset zero and the **unsigned** distance at a non-zero offset — the
unsigned distance is never negative, so a straddle test against it at offset zero
labels nothing, and the resulting empty `DMLabel` hard-crashes `getStratumIS`
rather than raising.

## Open questions / caveats

- **Depth.** The 2-D figures sit at ~3–3.7 levels (log2 of base/finest diameter)
  and the sweeps reach ~4.7. Nothing here tests 6–8 levels, where a real fault
  model would sit. `edge-split` has no similarity-class bound — the guarantee
  newest-vertex bisection gives up front — so its quality at depth is unmeasured.
- **Halo.** `edge-split` shows a much larger "refined finer than asked" fraction
  than NVB (90 % vs 42 % in 3-D). Part of that is the diameter-vs-volume marking
  mismatch rather than genuine waste, and the metric was not re-tuned for the
  diameter criterion. Worth separating before quoting it as leakage.
- The flip pass is O(cells) per round in pure Python and is the runtime cost in
  3-D. A production version would be incremental.

## Related

- `NVB_GRADED_ADAPT.md` — the current engine and why SBR cannot grade.
- `nested-vs-geometric-mg-transfers.md` — the coordinate-vs-topological transfer
  trade and issue #424.
- `mesh-shape-relaxation.md:179` — the leakage convention used here (per-cell
  `log2(h/h_asked)`, explicitly not a single scalar).
- memory `project_centroid_vs_bisection_refinement` — the 3-D centroid ruling this
  study was testing.
