# Newest-Vertex Bisection (NVB) for graded adapt-on-top

Status: **implemented (serial Route A), 2026-06-30**. Engine in
`src/underworld3/utilities/nvb.py` + `custom_mg.nvb_refine`; wired into
`Mesh.adapt(engine="nvb")`; validated in `tests/test_0836_nvb_graded_adapt.py`.
Parallel (Route B, native transform) is the next step. Follows the Layer-2
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

**Recommended de-risk before committing to C:** prototype the parallel
closure-fixpoint algorithm (option 1's hard kernel) in Python on a partitioned
mesh — prove the cross-rank closure converges and stays bounded — *separately* from
the SF construction. That validates the genuinely novel part cheaply and tells us
whether option 1 is viable or we go straight to option 2.

## Checkpointing

NVB is deterministic given the **initial labelling + the marked-cell set per
level**. Store those (tiny) — replay reconstructs every level bit-identically,
consistent with the marker-sidecar scheme already designed for SBR. The
coordinate-built custom-P sidesteps the canonical-numbering fragility (same as
SBR).

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
