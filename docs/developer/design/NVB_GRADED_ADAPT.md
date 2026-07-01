# Newest-Vertex Bisection (NVB) for graded adapt-on-top

Status: **implemented — serial Route A (2026-06-30) + parallel Route B native
transform, bit-confluent (2026-07-01)**. Serial engine in
`src/underworld3/utilities/nvb.py` + `custom_mg.nvb_refine`; wired into
`Mesh.adapt(engine="nvb")`; validated in `tests/test_0836_nvb_graded_adapt.py`.
The native `uwnvb` `DMPlexTransform` (`src/underworld3/utilities/nvb_transform.c`,
`tests/test_083{7,8}`) grades in parallel and produces a mesh identical to the
serial one at any communicator size. Remaining: Stage 2c integration
(`_adapt_nested(engine="nvb")` at np>1) + FMG tail. Follows the Layer-2
investigation in
[`LAYER2_SBR_ADAPT_ON_TOP.md`](LAYER2_SBR_ADAPT_ON_TOP.md). Goal: replace the
refinement *engine* under `mesh.adapt(adapter="sbr")` so that successive levels
produce a **graded** mesh (a level+1 ring, a level+2 sub-ring dividing *some*
of those cells, …) instead of a uniform-finest patch — keeping simplices,
without p4est/DMForest.

## Problem (why `refine_sbr` cannot grade)

PETSc's `refine_sbr` is **longest-edge bisection (LEB)**. Its conforming closure
is *unbounded for region marking* — the "longest-edge propagation path" (LEPP)
chains across a quasi-uniform mesh. We measured this directly (study scripts in
`~/+Simulations/layer2_adapt_on_top_study/`):

- **One triangle** → local on a structured base (refined cells stay within ~1
  ring), but on an unstructured base a single mark sends a 1-cell-wide chain
  clear across the domain (centre cell → refined cells out to r=0.583).
- **A dense marked region** cascades to a *constant* extent regardless of how
  much is marked (r<0.05 disk and r<0.30 disk both refined out to r=0.583 in one
  pass).
- **Cumulative multi-level** marking (nested disks / shrinking fault bands)
  **refills to uniform finest** with only single-cell level+1/level+2 transition
  rings at the patch boundary. Bullseye level histogram: `216 / 104 / 240 / 7408`
  — 95 % of refined cells at the finest level; level-3 cells spread to r=0.337
  when only r<0.12 was marked.
- A **structured base** helps the *first* level (closure is local) but each
  refinement destroys the local longest-edge structure, so deeper levels cascade
  again; the finest band fills. Unstructured is strictly worse (more cells,
  ragged transition, and *detached* refined cells where a LEPP chain jumped the
  gap).

Conclusion: the uniform-patch behaviour is a property of the **engine**, not of
the metric, mesh type, or marking. No tuning fixes it.

### The interior-interface intuition (why creating a patch works but grading inside it doesn't)

A revealing experiment: build an embedded uniformly-refined patch (refine the
whole disk r<0.25), then refine **one** cell at its centre. That single cell
drains the entire patch — +3592 cells, refined out to r=0.28 ≈ the patch edge —
on *both* structured and unstructured bases (whereas one cell on the *pristine*
base is local, +10).

The resolution of the apparent paradox:

- The **graded mesh you want is valid and constructible** (a level+1 ring around a
  level+2 sub-ring around a level+3 core is a fine 2:1-conforming mesh — gmsh
  could build it directly from a size field). Graded meshes are not impossible.
- **Creating the patch works because you refine the *whole* region at once** —
  every cell steps up together, so the *only* coarse/fine interface created is at
  the patch's *outer* edge, closed locally in one clean step. No interior cell
  ever neighbours a different resolution.
- **Grading needs a *new* interface in the *interior*** (the finer core must sit
  inside the coarser ring). Longest-edge bisection has **no local termination in a
  uniform bisected patch** — all edges are ~equal, so the closure's propagation
  path chains until it meets an edge-length contrast, i.e. the patch's *existing*
  outer interface. It can only push an interface outward, never carve one inside.
- **NVB's combinatorial marked edge restores local termination**: refine one cell
  deep in a patch → O(1) cells added with a small local transition, *which is*
  an interior interface. Same mesh space; only NVB can reach the graded states
  with bounded local moves.

Important scope note: this is purely a **DOF-efficiency / grading** problem. The
uniform-patch SBR meshes are perfectly valid for the *MG hierarchy* — the
custom-P FMG converges on them (4–6 iters). NVB is about spending DOFs where they
are needed, not about solver correctness.

## Why NVB

**Newest-vertex bisection** has a *provably bounded* closure: the number of
elements added by conforming completion is bounded by a constant times the number
of marked elements (Binev–Dahmen–DeVore 2004; Stevenson 2008), given a compatible
initial edge labelling. It produces conforming, 2:1-balanced, shape-regular
graded meshes — exactly the level+1-ring / level+2-divides-some staircase — and
in 2D yields at most **4 similarity classes** of triangle from each parent, so no
element degenerates under arbitrarily deep refinement. It is the standard engine
of adaptive FEM for this reason.

## NVB primer

Each triangle carries a **refinement edge** (equivalently a "newest vertex" — the
refinement edge is the edge *opposite* the newest vertex). Refining a triangle:

1. add the **midpoint M** of its refinement edge;
2. connect M to the opposite vertex, splitting the triangle into two children;
3. **M becomes the newest vertex of both children** — so each child's refinement
   edge is the parent edge incident to the apex (the two edges that shared the
   opposite vertex).

```
        v3 (apex)                      v3
        /\                             /|\
       /  \                           / | \
      /    \          refine         /  |  \
     /      \        ───────▶       /   |   \
    v1──────v2                     v1───M───v2
  (refinement edge v1–v2)     children: (v1,M,v3) ref-edge v1–v3
                                        (M,v2,v3) ref-edge v2–v3
```

Cycling the refinement edge through the parent's edges this way is what bounds
the recursion (4 similarity classes), unlike LEB which re-picks the geometric
longest edge each time and can chain.

**Conforming completion.** Bisecting refinement edge `e = v1–v2` of `T` adds M on
`e`. The neighbour `T'` sharing `e` must also split at M:
- if `e` is *also* `T'`'s refinement edge → both bisect at M → conforming in one
  step (a *compatible* edge);
- if not → first refine `T'` (bisect *its* refinement edge), then revisit. This
  recursion is the bounded part: it terminates because each step makes progress
  in the marked-edge structure, and globally the added work is O(#marked).

## Algorithm

```
refine(mesh, marked_cells):
    queue = marked_cells
    while queue:
        T = queue.pop()
        e = refinement_edge(T)
        T' = neighbour_across(T, e)            # None if boundary
        if T' is not None and e is not refinement_edge(T'):
            queue.push(T)                      # defer T
            queue.push(T')                     # make e compatible first
            continue
        bisect_pair(T, T')                     # add midpoint, 4 (or 2) children
        # children inherit refinement edges by the newest-vertex rule
```

**Initial labelling.** Assign each base-mesh triangle a refinement edge so the
mesh is *compatibly divisible* (every interior edge is the refinement edge of
both its triangles, or the labelling admits a compatible completion). Two
standard choices:
- **Longest-edge initial labelling** (label each triangle's longest edge): always
  conforming-terminating in 2D; simple; the labelling is only the *seed* — NVB's
  bounded propagation then takes over (this is *not* the same as running LEB,
  which re-picks longest edges every step).
- **Compatible (paired) labelling** via an edge ordering / matching — needed for
  the sharp O(#marked) complexity bound; can be a follow-up.

For a first cut the longest-edge seed is sufficient and robust.

## Representation and DMPlex construction

Two implementation routes:

**Route A — Python on the triangulation (proposed first).** Maintain the mesh as
numpy arrays: `cells (N,3) int` vertex indices, `coords (V,2)`, and a per-cell
**refinement-edge** encoded by *vertex ordering convention* (newest vertex =
local index 0 ⇒ refinement edge = local vertices (1,2)). Refinement and closure
are pure-numpy on these arrays; after a refinement pass, build the PETSc DMPlex
from the cell list (`DMPlexCreateFromCellListPetsc` / the UW wrapper used for
imported meshes) and wrap as a `uw.discretisation.Mesh` — exactly how
`custom_mg.sbr_refine` returns a DM today. Serial first. Pros: tractable,
debuggable, no PETSc-C; integrates with the existing custom-P hierarchy unchanged
(transfers are built from *coordinates*, so a graded mesh is fine). Cons: rebuilds
the DM each pass; serial; we own the boundary-label transfer.

**Route B — a PETSc `DMPlexTransform`.** Implement NVB as a transform type
alongside `refine_sbr`. Pros: native, parallel, incremental. Cons: deep PETSc-C,
marked-edge state must live in the transform, long lead time. Defer until Route A
proves the grading + hierarchy end-to-end.

The marked-edge bookkeeping is the crux either way: children's refinement edges
must be set by the newest-vertex rule, and boundary labels / region labels must
be carried onto the children (Route A: re-tag by edge midpoint membership of the
parent's labelled faces).

## Integration with Layer-2

Minimal surface change — NVB slots in where SBR is today:

- `custom_mg.nvb_refine(dm_or_mesh, marked_cells) -> refined` mirroring
  `sbr_refine`, plus a stateful `NVBMesh` carrying the marked-edge labelling
  across levels (needed because the labelling propagates parent→child).
- `_adapt_sbr` (rename → `_adapt_nested`) gains `engine="nvb"|"sbr"`: same
  metric → marking loop, but each level calls `nvb_refine` and keeps every
  intermediate level for the custom-P tail (already implemented — one MG level
  per refinement step). The **finest level is now graded**, so the marked count
  per level genuinely shrinks toward the feature.
- The mesh-owned custom-P hierarchy, REMAP transfer, parent/child lineage,
  copy_into prolong/restrict, no-redistribute guard — **all unchanged** (they are
  engine-agnostic; transfers come from coordinates).

## Parallel — preserving the inherited decomposition (decides Route A vs B)

The hard parallel requirement is **not** load balance — it is preserving the
parent's decomposition: custom-P's parallel path needs the finest co-partitioned
with the coarse tail (rank *r* owns the refinements of rank *r*'s base cells) so
rank-local point location holds. A re-partition *breaks* it, it does not merely
slow it. This is the same no-redistribute invariant the SBR adapt-on-top path
already satisfies.

**This requirement disqualifies Route A for parallel.** `createFromCellList`
builds a serial DM (rank 0) or, after `DMPlexDistribute`, a *freshly* partitioned
one — the parent's **point-SF** (shared/owned-vertex star-forest) is lost.
Preserving the decomposition via Route A would require manually rebuilding the SF
(ownership of every new rank-boundary midpoint), guaranteeing both ranks bisect
each shared edge *identically* (consistent marked-edge → consistent midpoint), and
a cross-rank closure-completion exchange. Fragile.

**How SBR preserves it today (the model to match):** `sbr_refine` calls
`adaptLabel` on the *already-distributed* DM — an in-place transform, so the
partition and SF are preserved by construction (why the SBR np=2 path works). The
transform framework propagates the SF.

Conclusion — the serial/parallel split is therefore:
- **Route A is serial-only**: a stepping stone to prove the NVB logic, label
  transfer, and custom-P integration (no decomposition to preserve at np=1).
- **Route B (native `DMPlexTransform`) is the real parallel implementation**,
  justified specifically by decomposition preservation — it inherits the same
  SF-propagation machinery as `refine_sbr`, not just speed.

**Acceptance test (np>1):** after an NVB adapt, every child cell's centroid lies
in a *locally-owned* base cell, and the coarse-tail partition is bit-identical
before/after — the same invariant the SBR path checks.

### Parallel-readiness review (2026-06-30) — serial Route A is non-blocking

A deliberate audit of the shipped serial engine against the parallel path, with
the requirement that *the serial implementation must not block parallel*.

**Is the co-partitioning invariant real, and does NVB preserve it?** Yes, and the
custom-P parallel transfer (`custom_mg._build_parallel_transfer` /
`_level_dof_layout`) is what *consumes* it: a fine **owned** DOF's coarse
contributions must come from coarse coords that are **local (incl. ghosts) on the
same rank** — i.e. the fine node must fall in a coarse simplex owned/ghosted by its
rank. That holds iff rank *r*'s fine cells are refinements of rank *r*'s base
cells. NVB preserves this *structurally*: bisecting a cell replaces it with two
children **of the same owner**, and the conforming closure only ever triggers more
*same-owner* bisections on a neighbour's rank — so any **in-place** NVB transform
on the distributed DM keeps the child co-partitioned with the coarse tail, exactly
as `dm.refine()` and `refine_sbr`'s `adaptLabel` do. The *only* thing that breaks
it is Route A's rank-0 `createFromCellList` rebuild — a property of the
*construction route*, not of NVB. **Measured bar:** the custom-P FMG already solves
on a locally-refined SBR adapt-on-top child at np=2 (pc=mg, 4 levels, converged,
exact err 2e-11) — the exact behaviour the NVB parallel path must reproduce, with
the parallel transfer machinery already built and proven.

**Does anything in the serial code block Route B?** No. The integration is a
parallel-alongside layer:
- `engine="nvb"` dispatch in `_adapt_nested` — Route B slots in here (at np>1,
  dispatch to the native transform instead of raising `NotImplementedError`).
- the custom-P tail is **coordinate-based and engine-agnostic**, and its parallel
  path (`_build_parallel_transfer`) already exists and is proven — a Route-B child
  feeds it unchanged.
- parent/child lineage, `copy_into`, the per-generation intermediate-DM snapshots,
  and the tail assembly (`_wrap_coarse_level`) are all identical for A and B.
- the only Route-A-specific pieces — `NVBMesh`'s cell-list `to_dm` and the
  coordinate/vertex-pair label transfer — are simply *not on* the parallel path
  (an in-place transform preserves numbering + SF, so neither is needed). `NVBMesh`
  remains useful as the serial engine and as a per-rank/oracle reference.

**The irreducible hard kernel** (what Route B must actually build) is the
**parallel conforming-closure fixpoint**: computing, consistently across ranks, the
final set of edges to bisect so the result is conforming with bounded closure, then
bisecting those edges in-place with correct SF. PETSc already does this for
*longest-edge* (`refine_sbr` `adaptLabel`); NVB needs the same closure machinery
with the **newest-vertex rule + marked-edge labelling**. Reassuringly, NVB's
labelling is **cross-rank-consistent for free**: the initial longest-edge seed is
geometric (both ranks compute the same edge for any shared cell — and each cell is
owned by exactly one rank), and the propagation rule (newest vertex = the geometric
midpoint) is deterministic and local, so no labelling-reconciliation protocol is
needed — only the closure *exchange*.

**No free PETSc transform.** petsc4py exposes `DMPlexTransform` generically but no
Python hook for a custom cell-subdivision rule, and PETSc ships only
`refine_regular` / `refine_alfeld` / `refine_boundary_layer` / `refine_tobox`
(plus `refine_sbr`) — **none** bisects a *labelled/specified* edge. So the
attractive "Python computes the closure, an existing PETSc transform does the
SF-preserving surgery" split is **not** available out of the box: Route B requires
registering a new transform type in C (ideally reusing PETSc's bisection-closure
infrastructure, swapping the longest-edge pick for the newest-vertex rule).

**Options for Route B**, in increasing cost / decreasing fragility:
1. *Manual-SF Route A-parallel* (pure Python): per-rank `NVBMesh` on local cells +
   a ghost layer, cross-rank closure by iterative halo exchange to a fixpoint, then
   per-rank `createFromCellList(comm=SELF)` + a **hand-built point-SF** matching
   shared points by coordinate. No PETSc rebuild, but the manual SF + parallel
   fixpoint are the "fragile" parts the original note flagged.
2. *Native C transform* (`DMPLEXTRANSFORMNVB`): the robust path — reuses PETSc's
   tested SF propagation and parallel closure; the cost is PETSc-C and a build.

**Parallel closure-fixpoint — DE-RISKED (2026-06-30).** The genuinely novel part
(the cross-rank conforming closure, the hard kernel shared by both options) is now
validated in Python *separately from* the SF construction:
[`nvb_parallel_closure_prototype.py`](nvb_parallel_closure_prototype.py) runs the
distributed NVB closure under a faithful discipline — geometric state shared (a real
run's point-SF, modelled by a deterministic shared midpoint), but **control flow
distributed**: a cell is bisected only by its owner, and the only cross-rank
coupling is an explicit work queue (no rank touches another's cells). Measured on a
diagonal/vertical feature straddling 2-, 3-, and 4-way partitions:

| property | result |
|---|---|
| globally conforming (incl. across the partition) | **0 hanging nodes**, every case |
| identical to the serial mesh (confluence) | **EQUALS_SERIAL = True**, every partition |
| communication rounds to converge | **≤ 3**, and **flat as the mesh grows** n=8→32 (8740 cells) |

So the cross-rank closure converges in a *bounded, mesh-size-independent* number of
rounds, stays conforming, and is partition-independent — exactly the guarantees a
parallel NVB needs. **What remains for a real parallel engine is only the SF / point
construction**, which Route B's native `DMPlexTransform` inherits from PETSc (and on
which option 1's manual-SF path could now be built, the algorithm being proven).
This is the milestone that says Route B is worth the C investment.

## Route B implementation plan (native transform) — feasibility established 2026-06-30

A symbol/feasibility audit against the installed custom PETSc (`petsc-3.25`,
`libpetsc.3.25.0.dylib`) shows Route B is a **self-contained UW extension — no PETSc
rebuild, no PETSc fork** — and is a *small delta* from the shipped SBR transform.

**The key insight (NVB ≈ SBR + a different edge pick).** Reading PETSc's SBR
transform (`src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c`): the
topological surgery (`CellTransform`, `GetSubcellOrientation`, the `RT_*` refinement
types, `SBRGetTriangleSplit{Single,Double}`, barycentre coordinate placement) is
**pure simplex-bisection geometry — identical for NVB**. The parallel conforming
closure is **already generic and SF-correct**: `DMLabelPropagate{Begin,Push,End}`
over the point-SF (the loop that exchanges the `splitPoints` edge label between
ranks) *is* the cross-rank closure fixpoint the Python prototype validated. The
**only** NVB-specific change is the *edge selected to split* in `SetUp`: SBR picks
the geometric **longest** edge; NVB picks the **newest-vertex refinement edge**.

**The newest-vertex rule with no stored bisection tree.** The newest vertex of a
triangle is identifiable by a per-vertex **age** label (base vertices = 0; each
midpoint = its creation generation). The refinement edge is the edge *opposite* the
unique highest-age vertex; when all three tie (a base cell) fall back to the
longest-edge seed. This is local, label-only, and **cross-rank-consistent for free**
(age = generation count is deterministic, so a shared midpoint gets the same age on
both ranks). The transform updates the age label on the midpoints it creates; the
Python driver carries it across generations. This reproduces the serial `NVBMesh`
exactly (longest-edge seed on base cells, newest-vertex thereafter).

**Symbol audit — what an external extension can link (no rebuild).** Checked with
`dyld_info -exports` on the installed `libpetsc`:

| needed | exported? |
|---|---|
| `DMPlexTransformRegister` | **yes** |
| `DMPlexPointQueue{Create,Enqueue,Dequeue,Empty,EmptyCollective,Destroy}` | **yes** |
| `DMLabelPropagate{Begin,Push,End}` (the parallel closure) | **yes** |
| `DMPlexTransformCellTransformIdentity`, `…GetSubcellOrientationIdentity` | **yes** |
| all generic plex/label API (`DMPlexGetCone/Support/TransitiveClosure`, `DMLabel*`, `DMGetPointSF`, `DMPlexTransformGet{DM,Active}`, …) | **yes** |
| `DMPlexTransformCellRefine_Regular` | hidden |
| `DMPlexTransformGetSubcellOrientation_Regular` | hidden |
| `DMPlexTransformMapCoordinatesBarycenter_Internal` (~12 lines) | hidden |
| `DMPlexTransformSetDimensions_Internal` (~20 lines) | hidden |
| `DMPolytopeTypeGetArrangement` (triangle: a fixed 6-orientation table) | hidden |

So **the entire parallel-hard machinery (closure + SF + registration) is linkable**;
the only hidden symbols are five small **2D geometry/orientation** helpers, all
short and reimplementable locally (the `SBRGetTriangleSplit{Single,Double}` cone
tables are *static functions in the SBR `.c`* — copy them; the regular triangle
(1→4) / segment (1→2) cone tables are small fixed arrays; barycentre placement and
set-dimensions are a few lines each).

**Build wiring.** UW already compiles Cython+C extensions against PETSc with a
shared `conf` (PETSc + petsc4py includes, `libpetsc` link) — e.g. the
`underworld3.function._function` extension pairs a `.pyx` with a hand-written
`petsc_tools.c`. Add `underworld3.utilities._nvb_transform` the same way:
`_nvb_transform.pyx` (imports register the `nvb` transform via
`DMPlexTransformRegister` and exposes a thin `nvb_refine`) + `nvb_transform.c` (the
`SetUp`/`CellTransform`/helpers, including `#include <petsc/private/dmplextransformimpl.h>`).
`$PETSC_DIR/include` carries the private headers, so the compile needs no rebuilt
PETSc.

### Build model (settled + proven) — headers + binary, no PETSc rebuild

The extension **compiles against PETSc headers and links the PETSc binary**; it does
not compile/rebuild any PETSc `.c` and does not fork PETSc. Three tiers:

1. **Exported symbols → linked from `libpetsc`.** All parallel-critical API is
   exported (verified `dyld_info -exports`): `DMPlexTransformRegister`,
   `DMPlexPointQueue*`, `DMLabelPropagate{Begin,Push,End}` (the cross-rank closure),
   `DMPlexTransformCellTransformIdentity`, `…GetSubcellOrientationIdentity`, and all
   generic plex/label API.
2. **Private headers → `#include <petsc/private/dmplextransformimpl.h>`** for the
   `_p_DMPlexTransform` struct (`tr->ops->…`, `tr->data`, `tr->trType`). Ships in
   `$PETSC_DIR/include`. This is how PETSc's own in-tree transforms are written — the
   supported internal API, used from outside the tree.
3. **Non-exported helpers → reimplemented locally** (short, 2D): `SetDimensions`
   (~4 lines), `MapCoordinatesBarycenter` (~6 lines) — both already in the Stage-1
   file — plus the regular **triangle 1→4** and **segment 1→2** cone/orientation
   tables (from `plexrefregular.c`'s `DMPlexTransformCellRefine_Regular` /
   `…GetSubcellOrientation_Regular`). NOTE: `DMPolytopeTypeGetArrangement` is a
   **`static inline` in the public `petscdm.h`** — usable by inclusion, *not* a
   reimplement (so the copied `SBRGetTriangleSplit{Single,Double}` work as-is).

**Proven:** the Stage-1 infra gate did exactly this (private header + reimplemented
helpers + linked `libpetsc`) and compiled, registered `uwnvb`, and applied on a
distributed DM at np=2 with a valid point-SF. **Caveat to record + assert at build:**
tiers 2/3 couple us to PETSc's internal struct/cone ABI for the pinned build
(3.25); re-verify on any PETSc upgrade.

**Source to copy/reimplement (exact locations):**
- `SetUp` + `CellTransform` + `GetSubcellOrientation` + `SBRGetTriangleSplit{Single,Double}`
  + the `RT_*` enum: `…/transform/impls/refine/sbr/plexrefsbr.c` (already read in full).
- `CellRefine_Regular` tri(1→4)/seg(1→2) cone tables + `GetSubcellOrientation_Regular`:
  `…/transform/impls/refine/regular/plexrefregular.c` (`DMPlexTransformCellRefine_Regular`, ~line 645+).
- `DMPolytopeTypeGetArrangement`: inline in `include/petscdm.h:609` (include, don't copy).

**Plan of record (validation-gated):**
1. *Infra gate* — **DONE (2026-06-30).** scaffold `_nvb_transform` that registers
   `uwnvb`; identity transform; proved build/registration/private-header/parallel-SF
   stack (np=2, valid point-SF).
2. *SBR-equivalent base (Stage 2a)* — **DONE (2026-07-01).** self-contained clone of
   SBR (SBR's `SetUp`/`CellTransform`/`GetSubcellOrientation`/`SBRGetTriangleSplit{,}`
   + the 4 reimplemented helpers) reproduces `refine_sbr` **byte-for-byte** — identical
   triangulation for every marking pattern incl. full uniform refine. `tests/
   test_0837_nvb_native_transform.py` (tier_b). Longest-edge edge pick; the newest-
   vertex choice layers on top.
3. *Newest-vertex grading (Stage 2b)* — **DONE, serial and parallel (2026-07-01)**
   via the single-bisection multi-pass driver (see below). Matches serial `NVBMesh`
   refinement edges exactly over repeated uniform refine; a deep mark is bounded
   (+2 vs SBR's +4824); conforming, deterministic, graded bullseye (805 vs SBR's
   102 868 cells). **Bit-confluent in parallel**: the refined mesh is identical to
   the serial one at any communicator size — uniform refine 159/352/760 and graded
   bullseye 215 cells agree exactly at np=1/2/3/4, 0 hanging nodes.
   `tests/test_0838_nvb_graded_native.py` (`test_parallel_confluence`).
4. *Integration + acceptance* — `_adapt_nested(engine="nvb")` at np>1 via the
   driver; Poisson + SolCx FMG match GAMG; result matches the serial NVB mesh.

### Stage 2b as built — single-bisection multi-pass driver (WORKS)

`_nvb_transform.refine(dm, want_label)` → C entry `UWNVBRefine`:
- each triangle carries a **`uwnvb_refedge`** cone-slot (0/1/2) label; `refedge =
  cone[slot]`, longest-edge seed on the base. No generation label is needed —
  single bisection makes each child's peak the new midpoint, so its refinement
  edge is just the edge opposite that midpoint (purely geometric).
- **closure**: grow the set `C` of cells that must bisect by adding, for each
  `C`-cell whose refinement edge is blocked by a neighbour with a different
  refinement edge, that neighbour (LEPP).
- **drain**: repeat — mark the *compatible* refinement edges (an edge that is the
  refinement edge of **all** its incident cells; reconciled across ranks by an
  `MPI_LAND` `PetscSFReduce`), single-split exactly those via a dumb
  **`uwnvb_bisect`** transform (reuses the Stage-2a cell-transform/orientation/
  coordinate ops), then rebuild the slot label (split child → edge opposite the
  midpoint; unsplit cell → map the parent refinement *edge* to its child edge —
  copying the slot *index* is wrong because the identity transform may renumber
  the cone) and the `C` set (bisected cells leave; blocked cells stay). Loop until
  `C` is globally empty. Every sub-pass touches each cell at most once ⇒ pure
  single-splits, always conforming, child slots trivial.

Gotchas fixed: the transform copies the *adaptation* label onto its output, so it
accumulates stale marks across calls (filter `want` to triangle cells + strip the
label from the output); and the drain loop condition must use the **global** `|C|`
(`MPIU_Allreduce`) or a rank with no local marks skips the collective loop and
deadlocks.

The parallel path needs **two** SF reconciliations, and both are now in place:

1. **Cross-rank closure.** `C`'s refinement edges are marked in a `req` edge label
   and propagated across the point SF with `DMLabelPropagate` (exactly as SBR's
   `SetUp` does — edges are shared SF points, cells are not), so a `C`-cell blocked
   by an off-rank neighbour marks that neighbour on its owning rank.
2. **Bisection-set consistency (`uwnvb_sf_lor`).** The per-pass set of edges to
   split must be *identical across ranks* for every shared edge. Marking it purely
   from the local `C` cells is not enough: a shared edge can be the agreed
   refinement edge of a `C`-cell on rank A while rank B's copy of the incident cell
   is not (yet) in its local `C`. Rank A then splits the shared edge (midpoint + two
   child edges) while rank B keeps it whole, so `DMPlexTransformCreateSF` maps rank
   A's child *edge* onto rank B's *vertex* — a **stratum-crossing point SF**. On the
   next refine, `DMLabelPropagate` reduces an edge's request onto that vertex, whose
   support is edges (not cells) → out-of-range read → **segfault**. The fix is an
   `MPI_LOR` `PetscSFReduce`+`Bcast` of the bisect-mark array over the point SF: an
   edge chosen on *any* rank is split on *every* rank owning a copy. Because the
   `agree` set (already `MPI_LAND`-reconciled) guarantees such an edge is the
   refinement edge of the incident cell on **all** sharers, splitting it everywhere
   is exactly the conforming closure — so the OR both fixes the SF and completes the
   cross-partition closure.

**Confluence (2026-07-01): achieved.** With both reconciliations, the refined mesh
is **bit-identical to the serial mesh** at any communicator size (uniform
159/352/760, graded bullseye 215, deep 5-level nesting 96/180/288/390/472 — all
equal at np=1/2/3/4; 0 hanging nodes; the output point SF is stratum-consistent).
The root cause of the earlier non-confluence *and* the nested-distributed segfault
was the single missing `uwnvb_sf_lor` — one bug, both symptoms. Next: Stage 2c —
wire `_adapt_nested(engine="nvb")` at np>1 to the driver.

### Stage 2b finding (2026-07-01): grading needs single-bisection, multi-pass

Two attempts to add the newest-vertex edge choice on top of the Stage-2a base both
**failed to grade** (a mark deep in a refined patch drained the whole patch, exactly
like SBR):

- **Age-derived edge (opposite highest-age vertex).** `age = max(endpoint ages)+1`
  produces **ties** — every vertex born in the same pass shares an age — so after any
  uniform refinement "the newest vertex" is ambiguous and the longest-edge tie-break
  silently reverts to longest-edge bisection. Pure vertex-age cannot reconstruct the
  true NVB refinement edge; this is why the serial `NVBMesh` tracks `(peak,b0,b1)`
  **explicitly** per cell.
- **Explicit per-cell refinement-edge slot** (a cone-slot 0/1/2 label, stored at child
  creation, read verbatim thereafter). Mechanically sound (slot populated, survives
  clone, read by `SetUp`; Stage 2a still exact) but **still drains.** The deeper
  obstacle: a single `DMPlexTransform` pass must stay **conforming**, so a cell caught
  in the closure across a *non*-refinement edge is forced into a **double/triple
  (green/blue) split** in the same pass. These one-pass multi-edge splits do **not**
  reproduce the compatible refinement-edge structure that `NVBMesh` builds via
  **sequential single bisections** (bisect the ref edge, *replace* the cell, recurse) —
  so the ref edges come out incompatible and the closure cascades.

**Redesign (the sound path): single-bisection-only, multi-pass.** Each transform pass
splits **only compatible refinement edges** — an edge that is the refinement edge of
*every* cell in its support. That is always a clean conforming bisection (both cells
single-split, share the midpoint), with a **trivial** child slot (each child's peak is
the new midpoint, so its ref edge is the edge opposite the midpoint). A driver loop
iterates passes, pulling in the neighbour across an incompatible ref edge (it must
refine its own ref edge first), until the marked set is satisfied — exactly
`NVBMesh`'s recursion, batched over the DM. This keeps every intermediate mesh
conforming, needs no green/blue child tables, and preserves co-partitioning (each pass
is an in-place transform). The Stage-2a clone stays the base; Stage 2b changes the
`SetUp` to mark only compatible ref edges + adds the driver loop.

## Checkpointing

NVB is deterministic given the **initial labelling + the marked-cell set per
level**. Store those (tiny) — replay reconstructs every level bit-identically,
consistent with the marker-sidecar scheme already designed for SBR. The
coordinate-built custom-P sidesteps the canonical-numbering fragility (same as
SBR).

**FMG-hierarchy checkpoint — reuse vs new structures (L.M. question, investigated
2026-07-01).** Finding: the *existing* FMG-hierarchy checkpoint
(`fmg-checkpoint-hierarchy.md`) does **not** carry a reusable per-node level label.
It stores the **coarsest** level as a one-DM sidecar (`mymesh.hierarchy.L0.h5`) plus
a refine count (`hierarchy_coarse_levels`) and `setCoarseDM`-links on reload; the
per-node "level" label reconstruction was **explicitly rejected** there because a
`createFromCellList` coarse DM loses the canonical `refine()` numbering PETSc's
*nested* interpolator requires (err 77). So there is no shared label to reuse for
option (a).

**That err-77 objection does not apply to NVB/adapt-on-top:** the Layer-2 custom-P
transfers are built from **coordinates**, not parent-child numbering — so
label-/replay-based reconstruction *is* viable here (the same reason SBR
adapt-on-top uses a marker-sidecar). ⇒ **Option (b) is the clean path:** checkpoint
the NVB portion of the FMG tail by persisting its **replay data** — the
per-generation marked-cell sets (deterministic ⇒ replay reproduces every level
bit-identically) — and carry the `uwnvb_refedge` slot + per-vertex generation as
labels so a reload can resume/extend adaptation without recomputing. The base
uniform hierarchy keeps its existing coarsest-sidecar mechanism; the NVB top levels
add a marker-sidecar. **Implication for this implementation:** keep the slot +
generation as ordinary DMLabels (they already survive `DMClone`/label-transfer and
checkpoint via the normal label I/O), and have the driver accept/emit the
per-generation marked-cell set as the replay record. No new bespoke label format;
the two mechanisms stay decoupled (base = sidecar+count, NVB top = marker replay),
which is correct since they reconstruct fundamentally differently (nested-numbering
vs coordinate-built).

## 3D (tetrahedra)

NVB is the right engine in 3D too, and the design generalises with no change of
approach:

- Each **tetrahedron** carries a *refinement edge*; bisecting adds its midpoint
  and splits the tet into two children, with the marked-edge propagation rule
  carried over (Bänsch 1991; Maubach 1995; Arnold–Mukherjee–Pouly). Stevenson
  (2008) proved **bounded closure + termination in n dimensions** given a
  compatible initial labelling, with a finite number of similarity classes (no
  sliver degeneration) — the same guarantees we rely on in 2D.
- **Longest-edge bisection is worse in 3D** (each refinement edge is shared by
  more tets, so the drain-to-interface chaining is more severe) — additional
  motivation to switch rather than patch `refine_sbr`.
- **Route A carries over directly**: cells become `(N,4)` vertex indices, the
  refinement edge is encoded by vertex ordering, bisection splits one edge into
  two children, and the bounded-closure completion is the same loop. DMPlex
  build-from-cells works for tets unchanged.
- Octree (DMForest/p4est) in 3D is **hexes** — ruled out (UW3 design change),
  same as 2D.

The marked-edge labelling and similarity-class bookkeeping are the only parts
that carry extra 3D subtlety (the compatible initial labelling is more involved);
the integration seam, custom-P hierarchy, and DM construction are
dimension-agnostic.

## Complexity assessment

| Piece | Route A (Python) | Risk |
|---|---|---|
| Marked-edge data model + newest-vertex rule | small | low (well-specified) |
| `refine` + bounded conforming completion | medium | medium (termination/closure correctness — needs careful tests) |
| Initial labelling (longest-edge seed) | small | low |
| DMPlex build-from-cells + boundary/region label transfer | medium | medium (label re-tagging, coordinate DM, UW Mesh wrap) |
| Wire into `_adapt_nested` + custom-P | small | low (engine-agnostic seam exists) |
| Parallel completion across ranks | — (serial first) | high (defer to Route B) |
| Route B transform (native, parallel) | large | high (PETSc-C) |

A serial Route-A prototype that produces a graded fault funnel + drives the
custom-P FMG is the milestone that tells us whether to invest in Route B.

## Prototype result (Phase 1 core — DONE, 2026-06-30)

The serial NVB core is implemented and validated in ~135 lines of numpy:
[`nvb_prototype_2d.py`](nvb_prototype_2d.py) (`NVBMesh`: newest-vertex data model,
recursive compatible-bisection `bisect`, `refine`, conformity check). Measured
against the SBR pathology, same nested-disk bullseye:

| | NVB | SBR (`refine_sbr`) |
|---|---|---|
| one cell deep in a uniform patch | **+2 cells, local** (r 0.016–0.030) | +3592, drained to r=0.28 |
| finest band (gen 6) | confined **r < 0.126** | refilled to r=0.337 |
| conformity (hanging nodes / over-shared edges) | **0 / 0** every step | — |
| total cells (same bullseye target) | **3024** | 7968 |

The bullseye renders as clean concentric generation rings (not a uniform core).
So the core algorithm — the genuinely hard part — works. Remaining Phase-1 work
is the DMPlex wrap + label transfer (engineering, not algorithm risk). NVB bisects
1→2 (area halves); a full isotropic SBR "level" (1→4) = two generations.

## Phasing

1. **Serial NVB core** (numpy) — **DONE** (`nvb_prototype_2d.py`): marked-edge
   model, `refine`, recursive compatible-closure; conformity (no hanging nodes,
   0 over-shared edges) and bounded-closure (1 cell → +2, local) verified; graded
   bullseye rendered.
2. **DMPlex wrap** — **DONE** (`src/underworld3/utilities/nvb.py`,
   `custom_mg.nvb_refine`): `NVBMesh.from_dm` / `to_dm` build the interpolated
   DMPlex from the cell list and transfer boundary-edge + region-cell labels.
   Gotchas resolved: `createFromCellList` needs consistent **CCW winding** (the
   `(peak,b0,b1)` refinement-edge order is geometry-agnostic, so reorient the
   *exported* cell list only); labels matched by **vertex pair** (vertex identified
   by coordinate — midpoints are exact float averages, so the match is exact —
   never by array index); each boundary edge also labels its two vertices so
   `Mesh()` derives `UW_Boundaries`/`Null_Boundary`, and `All_Boundaries` is the
   geometric outer boundary. De-risk: a Dirichlet Poisson on the injected DM solves
   to 4e-10. Graded bullseye / fault funnel coloured by generation rendered
   (`~/+Simulations/layer2_adapt_on_top_study/nvb_src_{bullseye,fault_funnel}.png`,
   `nvb_vs_sbr.png` — concentric rings, **not** a uniform core; 1800 vs 3616 cells).
3. **Layer-2 wiring** — **DONE**: `Mesh._adapt_sbr` renamed `_adapt_nested` with
   `engine="sbr"|"nvb"` (default sbr, unchanged); the NVB path drives a *persistent*
   `NVBMesh` across `2·max_levels` generations (1→2 area halving) so the
   refinement-edge labelling — hence the similarity-class bound — propagates
   parent→child, snapshotting one DM per generation into the engine-agnostic
   coordinate-based custom-P FMG tail. Validated serial (`tests/test_0836`,
   tier_b): conformity, bounded closure (single-cell local + ≤C·#marked),
   shape-regularity (similarity classes bounded; =1 on a right-isoceles structured
   base — the ideal), graded bullseye + fault funnel, **Poisson FMG** (5 levels)
   matches GAMG bit-identically, **SolCx Stokes** (η jump 1e6) velocity-block FMG
   matches GAMG iter-for-iter (8). `engine="nvb"` raises `NotImplementedError` at
   np>1 (verified clean under `mpirun -n 2`; SBR unaffected) and for dim≠2.
4. **Decide on Route B** (native transform) for parallel + incremental — **NEXT**,
   informed by 1–3. The seam is ready: `engine="nvb"` would dispatch to a native
   `DMPlexTransform` at np>1, and the custom-P tail is already coordinate-based
   (engine-agnostic), so Route B needs no change to the hierarchy or the wiring.

## Alternatives considered

- **Longest-edge bisection (`refine_sbr`)** — current; unbounded closure, no
  grading (this whole investigation).
- **Red–green refinement** — bounded but introduces "green" closure elements that
  must be removed/re-refined between passes (green cells aren't themselves
  refined), complicating a multi-level hierarchy; NVB's uniform treatment of all
  cells is cleaner for our level-per-refinement custom-P tail.
- **p4est / DMForest octree** — natural 2:1 grading, but quad/hex: a complete UW3
  design change. **Ruled out** by L.M.

## References

- E. Bänsch, *Local mesh refinement in 2 and 3 dimensions* (1991).
- W. F. Mitchell, *Adaptive refinement by bisection* — newest-vertex rule.
- P. Binev, W. Dahmen, R. DeVore, *Adaptive FEM with convergence rates* (2004) —
  bounded-closure complexity.
- R. Stevenson, *The completion of locally refined simplicial partitions created
  by bisection* (2008) — initial-labelling compatibility + O(#marked) bound.
