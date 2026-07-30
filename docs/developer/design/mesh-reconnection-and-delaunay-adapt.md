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

The **repair pass** is a separate job with its own handoff plan:
`~/.claude/plans/parallel-mesh-reconnection-flips.md`. Scoped as
*bisection-artefact repair* rather than general mesh improvement: the only badly
shaped cells are those split at an edge they did not nominate, so they all lie in
the star of a newly inserted vertex and are known without search. The plan tiers
the response — strengthen the edge **selection** first (no new operator, and the
existing parallel machinery already covers it), then 2-D Lawson restricted to
new-vertex stars, then 3-D edge removal on the same stars only if a deficit
remains.

Two corrections it carries, which matter for anyone reading Findings 5 and 4
above: the 3-D flip verdict is **provisional**, because only 2↔3/3↔2 were tested
— the weakest operators in the family; and the seam-cost measurement froze a
fraction of *all* cells rather than of repair sites, so it is pessimistic for the
wrong reason.

Deferred, explicitly not blocking: anisotropic (metric) predicate; edge collapse
for coarsening.

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
