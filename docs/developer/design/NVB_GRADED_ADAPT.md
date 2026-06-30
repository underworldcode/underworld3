# Newest-Vertex Bisection (NVB) for graded adapt-on-top

Status: design (2026-06-30). Follows the Layer-2 investigation in
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

## Parallel

Route A is serial (rebuild-from-cell-list, on-rank). The no-redistribute
correctness requirement (custom-P needs the finest co-partitioned with the coarse
tail) is satisfied if NVB runs on each rank's owned cells with a halo for closure
across rank boundaries — the same on-rank story as SBR, but closure that crosses a
partition boundary needs a parallel completion step (exchange marked edges on
shared faces until no new marks). This is the main parallel risk and argues for
Route B (a transform handles it natively) once the serial design is proven.

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

## Phasing

1. **Serial NVB core** (numpy): marked-edge model, `refine`, closure; unit-test
   conformity (no hanging nodes), 2:1 balance, bounded closure (#added ≤ C·#marked),
   shape regularity (similarity-class count bounded) on stress cases.
2. **DMPlex wrap**: build-from-cells + label transfer; `nvb_refine` mirroring
   `sbr_refine`; render the graded bullseye + fault funnel coloured by level
   (the acceptance picture: distributed rings, *not* a uniform core).
3. **Layer-2 wiring**: `engine="nvb"` in `_adapt_nested`; custom-P FMG on the
   graded child (Poisson + SolCx Stokes), serial.
4. **Decide on Route B** (native transform) for parallel + incremental, informed
   by 1–3.

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
