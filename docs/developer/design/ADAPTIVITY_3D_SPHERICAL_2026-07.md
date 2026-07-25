# Adaptivity in 3D and spherical geometry — the MMPDE + NVB capstone

Status: **RULED — round 1 = NVB-3D** (2026-07-17). Maintainer ruling on the
phase-0 draft: *"NVB is the key piece here — existence at all in 3D is
important to implement… just figuring out how to do this in 3D is probably
the big task. Let's put that in the first round. Checking that it produces an
mg-viable hierarchy at every stage."* Curved / internal boundaries are
explicitly classed as later clean-up items. The phasing below reflects the
ruling: **round 1 = NVB-3D with an MG-viability gate at every stage**,
round 2 = MMPDE-3D, round 3 = composition + the boundary clean-ups +
spherical geometry.

## Scope

This capstone replaces exactly two honest `NotImplementedError`s with
implementations. The user-facing API does **not** change (the purposeful-naming
ruling of 2026-07-16 already fixed the API shape; this work fills it in):

1. **`mesh.redistribute_nodes(metric)` / `uw.meshing.node_redistribution` on 3D
   simplex meshes** — the d=3 tetrahedral discretization of the Huang–Kamenski
   variational MMPDE mover
   (`discretisation_mesh.py` gate; mover in `meshing/smoothing/mmpde.py`).
   This was never implemented: the old d=3 branch called a `_signed_volumes`
   that never existed anywhere (audit READ-01/BF-09; the guard is decision D8's
   flagged placeholder).
2. **Engine-less `mesh.adapt(metric, max_levels=...)` on a 3D mesh** — the NVB
   refinement engine generalised to tetrahedra
   (Bänsch / Maubach / Traxler / Arnold–Mukherjee–Pouly / Stevenson bisection),
   parallel via the native `uwnvb` `DMPlexTransform`, with the same np1/2/4
   bit-confluence contract the 2D engine has
   (`discretisation_mesh.py` engine default; engine in
   `utilities/nvb.py` + `utilities/nvb_transform.c`).

Then the two are composed (unified adapt + redistribute workflow) and taken to
spherical geometry (curved boundaries, rotated free-slip normals,
BoundingSurface snapping, deformed-mesh evaluate).

**A sharpening discovery (2026-07-17):** the `engine="sbr"` escape hatch that
the 3D `adapt()` error message offers almost certainly does not work. PETSc's
SBR transform (`plexrefsbr.c`) handles only POINT / SEGMENT / TRIANGLE and
raises `PETSC_ERR_SUP "Cannot handle points of type %s"` for tetrahedra. Until
NVB-3D lands there is **no adaptive refinement engine at all in 3D** — only
uniform `refine_regular` hierarchies. (Verify empirically in phase 0/1 and fix
the error message if confirmed.)

## What the phase-0 reading established

### The MMPDE mover core is already dimension-general

`_mmpde_mover` (`meshing/smoothing/mmpde.py:86`) implements the direct simplex
discretization of Huang–Kamenski (JCP 301 (2015) 322) in batched
`(N, d, d)` einsum algebra with `d = mesh.cdim` throughout: the edge-matrix
Jacobians `J = Ê E⁻¹`, the functional
`G = θ√detM·Sᑫ + (1−2θ)·dᑫ·rᵖ·detM^((1−p)/2)` with `q = d·p/2`, the analytic
vertex-velocity rows `V = −G E⁻¹ + E⁻¹ (∂G/∂𝕁) Ê E⁻¹ + (∂G/∂r) r E⁻¹`, the
metric-variation term `tr(∂G/∂M · ∂M/∂x)`, the Shepard/RBF metric bake, the
per-node step cap, and the energy + min-volume line search are all written for
general `d`. The non-folding property (`G → ∞` as `det𝕁 → 0`) is
dimension-independent.

The 2D locks are exactly three, all at the rim of the mover:

| lock | where | 3D replacement |
|---|---|---|
| cell extraction `_tri_cells` | `mmpde.py:236` | `_tet_cells` — **already exists** (`graph.py:441`, currently used only by `_boundary_facets`) |
| `signed_vol = _signed_areas` | `mmpde.py:237` | `_signed_volumes(coords, tets) = det(E)/6` (batched; ~10 lines, new) |
| `fact = 2.0` (= d!) | `mmpde.py:240` | `math.factorial(cdim)` |

plus the guard itself (`mmpde.py:162-170`) and the `redistribute_nodes` gate
(`discretisation_mesh.py:6148`).

### The support infrastructure is already 3D-complete and live

Verified in the phase-0 read (file:line refs in the audit trail):

- **Boundary slip**: `mesh.boundary_slip` is dimension-general — in 3D the
  tangential slide is a genuine per-vertex tangent-plane projection using the
  `Gamma_P1` projected normals (themselves built per-`cdim` and documented as
  designed for 3D). Unregistered slip labels build a transient
  `BoundingSurface(kind="facet")` whose `restore` dispatches to
  `_nearest_on_facets_3d` (exact closest-point on boundary triangles).
  `_boundary_facets` has an explicit `cdim==3` branch (boundary triangles via
  `_tet_cells`).
- **`_deform_mesh`**, `_owned_cell_mask`, `_owned_vertex_mask`,
  `_min_incident_edge_nd`, `_slip_normals`, `_resolve_slip`,
  `_auto_pinned_labels`: all dimension-general.
- **`_spd_sanitise`** already handles 3×3 batches, hardened for the batched-eigh
  LAPACK failure modes specifically "for the 3D capstone path" (#352 → PR #368),
  with 3×3 tests in `test_0850_mesh_smoothing.py`.
- **`metric_density_from_gradient`** is dimension-general
  (`coarsening = refinement**(1/cdim)` etc.).
- **Deformed-mesh evaluate** (the holes root cause, `8a9d2ff2`) and the
  parallel coordinate-section fixes (#360 → PR #363) are landed and
  dimension-general.
- **3D test meshes exist**: `UnstructuredSimplexBox` with 3-tuples (tets,
  `refinement>=1` supported), `SphericalShell` (tets, `refinement` + boundary
  snapping callback), `CubedSphere(simplex=True)`, `BoxInternalBoundary` 3D.

The remaining 2D locks *outside* the mover core:

- `mesh_metric_mismatch` (`metrics.py:66`) — the `skip_threshold` engine —
  requires triangles (`_tri_cells` + `_signed_areas`). Needs the tet analogue
  or `skip_threshold` must refuse honestly in 3D.
- `follow_metric`'s polish step uses the 2D quality formula
  `q = 4√3·A/Σe²` (`api.py:799-809`). Needs a tet quality
  (e.g. `q = c·V/ (Σe²)^{3/2}` normalised to 1 for the regular tet) or the
  polish is skipped in 3D.
- `_pinned_mask` closes boundary labels through vertices/edges only — a 3D
  mesh whose label tags faces alone leaves boundary vertices unpinned
  (pre-existing `TODO(follow-up)`, `discretisation_mesh.py:3083-3087`).
  Gmsh-built meshes label through the closure, so this only bites exotic
  hand-labelled DMs.

### The NVB architecture generalises; the tet production rules are the new work

The 2D engine's two-layer architecture is the right one for 3D and its hard
lessons carry over directly (`NVB_GRADED_ADAPT.md`):

- **Single-bisection multi-pass is non-negotiable.** The 2D Stage-2b finding:
  a one-pass conforming transform is forced into green/blue multi-edge splits
  that corrupt the refinement-edge structure and drain like SBR. Only repeated
  conforming single bisections of *compatible* edges preserve the marked-edge
  invariants. In 3D this simplification is even more valuable — it removes any
  need for the (large) 3D green-closure child tables. The per-pass production
  rules reduce to: SEGMENT 1→2 (exists), TRIANGLE split on one edge (exists),
  and **TETRAHEDRON split on one edge** (new: 2 child tets + 1 interior
  triangle; the two faces incident to the split edge each split 1→2).
- **The parallel closure machinery is dimension-independent in mechanism.**
  `DMLabelPropagate` over the point SF (cross-rank closure requests), the
  `MPI_LAND` agree-reduction (an edge is split only when it is the refinement
  edge of *every* incident cell) and the `MPI_LOR` bisect-mark reduction (an
  edge split anywhere is split on every rank owning a copy — the one-bug
  root cause of both 2D non-confluence and the stratum-crossing SF segfault)
  all operate on labels over SF points. 3D adds the **face stratum** to the
  reconciliation (split faces are shared SF points too) — the direct analogue
  of the 2D edge/vertex reasoning, and the most likely place for a
  parallel bug of the same family.
- **The drain loop, slot-label maintenance pattern, stale-label stripping,
  collective termination** (`UWNVBRefine`, `nvb_transform.c:929-1198`) carry
  over structurally.

What is genuinely new in 3D:

1. **The per-cell refinement state is richer than one cone slot.** A triangle's
   refinement edge is one of its 3 cone points (slot label 0/1/2); a tet's cone
   is 4 *faces* — its 6 edges live deeper in the closure, and the child's
   refinement edge is **not** determined by the new midpoint alone. The
   established formulations both amount to explicit per-cell state:
   *Maubach/Traxler* (ordered vertex tuple + level mod 3; children's tuples by
   fixed rule) or *Arnold–Mukherjee–Pouly* (refinement edge + a marked edge per
   face + a flag; children's marks by fixed rule). Either packs into a single
   small-integer DMLabel per cell (e.g. vertex-permutation index 0–23 ×
   level-mod-3, or refedge 0–5 × face-mark state), maintained across passes
   exactly like the 2D slot label. **The serial oracle (below) decides which
   encoding yields the cleaner child tables** — they are mathematically
   equivalent (Traxler 1997).
2. **Initial labelling / termination — SOLVED by the DGS initialization
   (stage-1a finding, 2026-07-17).** In 2D the longest-edge seed always
   conforming-terminates; in 3D arbitrary seeds can cycle and the classical
   remedies (Kossaczký/Stevenson pre-refinement — alters the user's base
   mesh, which is also the MG tail; Bänsch/AMP marked types — weaker closure
   theory) all cost something. Diening–Gehring–Storn (arXiv:2306.02674)
   removes the problem: a **greedy vertex coloring** of the base mesh (in a
   deterministic order), the vertices of each cell sorted by color (global-
   max color first, tag = n), then Maubach's plain bisection rule. For **any
   conforming initial triangulation in any dimension** this is proven to
   terminate, keep ≤ n!·n·2^(n−2) = 36 similarity classes per base tet,
   preserve shape regularity with an explicit constant, and satisfy the full
   Binev–Dahmen–DeVore closure estimate — the *sharp* bound, not a weakened
   one, with **no pre-refinement**. Parallel note for stage 1c: the coloring
   must be partition-independent — compute it once on the (small) serial
   base at construction, or greedy-color in coordinate-lexicographic order;
   it then travels as an ordinary vertex label.
   **Does DGS change the 2D engine? Yes — by unification, not by necessity
   (maintainer tech-debt ruling, 2026-07-17).** On correctness the 2D engine
   needs nothing: in n=2 the BDV closure estimate holds for *arbitrary*
   initializations (Karkulik–Pavlicek–Praetorius, cited in DGS §3.1) — the
   theorem that does not exist in 3D — so the longest-edge-seeded 2D engine
   already has termination, conformity, 4 classes per base triangle, and the
   BDV bound; DGS would only improve the closure constant against an
   adversarial base-mesh chain pathology not seen on quality gmsh meshes.
   But the Maubach/DGS formulation is **dimension-general**, and the
   maintainer's priority is servicing the tech debt by combining algorithms:
   two cell-rule layers (2D slot label + opposite-the-midpoint child rule vs
   3D tagged tuple + Maubach rule) and two seeding schemes are debt with no
   compensating capability. **Plan: build stages 1b/1c as a
   dimension-general tagged-simplex engine** (one data model
   `(ordered tuple, tag)`, one DGS coloring seed, one closure driver; only
   the per-polytope production/orientation tables dispatch on dimension —
   those are irreducibly per-polytope). Validate 3D first (the round-1
   ruling), then **switch the 2D dispatch onto the tagged engine and retire
   the slot-based path** in its own gated PR. Migration cost is bounded and
   the window is open: the marker-replay checkpoint was never implemented,
   so no persisted refinement state exists to stay compatible with — the
   cost is regenerating the hard-coded 2D confluence integers in
   `test_083x` (mechanical asserts) plus one re-validation of the 2D FMG
   parity gates. The refined 2D meshes change node-for-node but stay in the
   same quality class (both schemes are 4-classes-per-base-triangle NVB).

3. **The tet subcell-orientation tables.** `GetSubcellOrientation` for
   TETRAHEDRON under its 24 arrangements (`DMPolytopeTypeGetArrangement` is a
   `static inline` in public `petscdm.h` — usable by inclusion). This is the
   grind of the C work. PETSc's `plexrefregular.c` tet tables are the
   structural template; there is no SBR-3D to clone.
4. **No PETSc-side blocker is expected**: the 2D extension already proved the
   headers+binary build model (private `dmplextransformimpl.h`, exported
   closure/queue/label API, no PETSc rebuild), and the transform framework
   itself is dimension-agnostic (identity cases for tets already pass through).
   The custom-P MG tail is coordinate-based and engine/dimension-agnostic; its
   invariants (co-partitioned levels, label-preserving output, consistent
   sections) are properties the 3D transform must preserve, not new code.

## Plan of record (phased, maintainer in the loop at each boundary)

Worktree per work package; `rm -rf build/ && ./uw build` after source changes;
suites strictly serial; parallel gates as `tests/parallel/` scripts; style
gates without allowlist additions; adversarial review before merge; no
`pixi.toml`/`pixi.lock` changes. No new user-facing API anywhere below.

**Round order (maintainer ruling, 2026-07-17): NVB-3D is round 1** — existence
of 3D adaptive refinement is the big task and the key deliverable; MMPDE-3D is
round 2; composition, curved/internal-boundary clean-ups, and spherical
geometry are round 3. Within round 1, **every stage carries an MG-viability
gate**: it is not enough that a stage's mesh refines and conforms — the
refined generations must feed the coordinate-based custom-P tail and drive
FMG. Concretely, per stage:

- *oracle stage*: the structural prerequisites the tail depends on — every
  generation conforming (0 hanging faces/edges — `createDS` needs it),
  children geometrically nested in parents (fine-node point location in
  coarse cells is what the barycentric P builder does), similarity classes
  bounded (shape-regular coarse problems);
- *DMPlex stages*: build the `[base … child]` tail for the actual generations
  and run Poisson (then Stokes velocity-block) **FMG vs GAMG iteration
  parity** on the graded child — the same acceptance the 2D engine passed
  (`test_0836` / `test_0839`), at np=1 first, then np=1/2/4 bit-confluent.

### Round 1 — 3D NVB (`feature/nvb-3d`; the major work package)

Strictly serial-oracle-first, replaying the 2D de-risking sequence:

- **1a. Serial oracle** — `nvb_prototype_3d.py` (pure numpy, mirroring the 2D
  `nvb_prototype_2d.py`): marked-tet data model (decide Maubach vs AMP
  encoding here), recursive compatible-star-bisection closure (an edge is
  bisectable when it is the refinement edge of *every* tet in its star),
  conformity / bounded-closure / similarity-class diagnostics, plus the
  oracle-stage MG-structural checks above. Acceptance mirrors the 2D
  prototype: one tet deep in a uniform patch refines O(1)-locally; graded 3D
  bullseye; 0 hanging faces/edges; classes bounded over deep refinement.
  **This is the go/no-go gate for the C investment.**

  **DONE — PASS (2026-07-17).** Maubach rule + DGS coloring initialization,
  ~330 lines of numpy. Measured, on a structured Kuhn 2×2×2 base *and* an
  arbitrary unstructured Delaunay base (361 tets):

  | property | Kuhn (structured) | Delaunay (arbitrary) |
  |---|---|---|
  | conformity, every sweep (9 uniform sweeps, to 124k / 2.87M cells) | 0 hanging, 0 overshared | 0 hanging, 0 overshared |
  | per-base-tet similarity classes (theorem bound **36**) | ≤ 36, plateaued | **exactly 36**, plateaued from sweep 4 |
  | one deep mark inside a 2-level uniform patch | **+6 cells**, local | **+4 cells**, local |
  | graded bullseye (4 shrinking radii × 3 sweeps) | gens 2–18, finest confined r<0.094 | gens 0–20, finest confined r<0.093 |
  | volume conservation | exact (4e-16) | exact (0.0) |
  | child-in-base-ancestor nesting (custom-P prerequisite) | exact | 9e-15 |

  Hitting the similarity-class bound exactly and plateauing there is the
  sharpest available signature that the child rule is implemented correctly.
  The per-cell state for stage 1c is confirmed as **(ordered vertex 4-tuple,
  tag γ ∈ {1,2,3})** — encodable as one small-integer DMLabel value
  (permutation index 0–23 × tag), maintained across passes exactly like the
  2D `uwnvb_refedge` slot. **Go for stages 1b/1c.**
- **1b. DMPlex wrap + serial engine** — `from_dm`/`to_dm` with boundary-face
  and region label transfer; wire as the np=1 `engine="nvb"` path for tets;
  Poisson + FMG-vs-GAMG parity on the graded child (the MG gate, per
  generation).

  **DONE — PASS (2026-07-17).** `TaggedBisectionMesh` in `utilities/nvb.py`
  (dimension-general: the Maubach rule + DGS coloring from stage 1a, plus
  facet-label carrying — labelled boundary/interface facets split with their
  bisected edges — and the same `from_dm`/`refine`/`to_dm` interface as the
  2D `NVBMesh`). `_adapt_nested` dispatches 3D tets to it at np=1; the
  engine-less 3D `adapt()` guard is lifted (np>1 raises cleanly, verified
  under `mpirun -n 2`); `max_levels` now runs **dim** bisection generations
  per isotropic level (2D unchanged). The advertised `engine="sbr"` 3D
  fallback was confirmed broken (PETSc error 56, `DMPlexTransformSetUp_SBR`
  cannot handle tetrahedra) and the message/docstring corrected — NVB-3D is
  the only adaptive 3D refinement. Gates: `test_0840` 5/5 (engine
  conformity/closure/shape + **the MG gate**: Poisson FMG on the graded 3D
  child, `pc=mg` with one MG level per generation, matches GAMG to 1e-4,
  exact linear solution to 1e-8 — which also proves the 3D facet-label
  transfer); 2D adapt-family regressions `test_0830/0834–0838` 40/40;
  style gates clean, no allowlist additions.
- **1c. Native transform** — extend `nvb_transform.c`: the TETRAHEDRON
  single-edge-split production rule + orientation tables; per-cell state label
  (from 1a) maintenance across passes; agree/bisect SF reconciliation extended
  over the face stratum; the same drain-loop driver. Bit-confluence integers
  at np1/2/4 mirroring `test_0839`; FMG parity (Poisson + 3D Stokes velocity
  block) — the MG gate, parallel.

  **How the 3D refinement works, in plain terms.** To refine a
  tetrahedron we cut it in half: pick one of its six edges (its
  *refinement edge*), put a new vertex at that edge's midpoint, and slice
  the tet into two smaller tets through that midpoint. The cut surface is
  a new triangle inside the old tet, and any face of the tet that
  contained the cut edge is split into two triangles — that last part is
  exactly the operation the 2D engine already performs on triangles, so
  the 2D code is reused for it.

  Several tets share each edge, so cutting an edge means every tet around
  it must be cut too, or the mesh would have hanging nodes. What keeps
  this from cascading across the whole mesh is *which* edge each cell
  nominates as its refinement edge. We use Maubach's bookkeeping
  (Maubach 1995; equivalently Traxler 1997): each cell carries a vertex
  ordering and a small counter (the "tag"), the refinement edge is read
  off from them, and the two children inherit their ordering and tag by a
  fixed recipe. This guarantees the mesh stays conforming, cells never
  degenerate (at most 36 distinct shapes ever arise from one starting
  tet), and the total amount of extra cutting is proportional to what was
  asked for (Stevenson 2008).

  To start the bookkeeping on an arbitrary gmsh mesh, we colour the
  vertices so that neighbouring vertices get different colours (a greedy
  pass; a handful of colours suffice) and order each cell's vertices by
  colour. This initialisation is the contribution of Diening, Gehring &
  Storn (arXiv:2306.02674): it works for any conforming mesh, needs no
  pre-processing of the user's mesh, and preserves all the guarantees
  above. The stage-1a prototype demonstrates the whole scheme in ~300
  lines of plain numpy (`nvb_prototype_3d.py`) and is kept as the
  reference implementation the native code is tested against.

  Inside PETSc, "cut this tet along that edge" must be spelled out as
  static tables listing which new cells appear and how their faces attach
  to the neighbours' faces (the `DMPlexTransform` framework; see the
  `DMPlexTransformCellTransform` documentation in PETSc). These tables
  are easy to get subtly wrong — a face is viewed from opposite sides by
  its two neighbouring tets, and the tables must agree with both views —
  so we do not write them by hand: a generator script derives them from
  the conventions and checks them against the numpy reference before they
  ever compile (`tet_bisection_tables_generator.py`).

  **Sub-plan (started 2026-07-17), each step with its own gate:**

  * **1c-i — production tables.** The tet single-edge split: 6 refine types
    (one per closure-edge position), each producing `{1 interior TRIANGLE +
    2 child TETRAHEDRA}` — the exact 3D analogue of the 2D triangle rule's
    `{1 SEGMENT + 2 TRIANGLES}`. Note the driver's single-bisection design
    means NO green/blue multi-edge tet tables are needed — only
    1-marked-edge cases (two agreed edges can never share a cell, since a
    cell has one bisection edge; hence faces split at most 1→2 per pass).

    **Spec findings (PETSc 3.25 source extraction, 2026-07-17) that shrink
    the job:** (a) a tetrahedron is a CELL — it appears in no cone, so its
    `GetSubcellOrientation` source orientation is always identity: the
    feared 24-arrangement tet reconciliation tables are unreachable and
    reduce to the `so==0` fast path. The orientation reconciliation that
    IS load-bearing (shared faces seen differently by their two tets) runs
    through the TRIANGLE single-split tables **already present** in
    `nvb_transform.c` (Stage-2a SBR clone). (b) The only new vertex is the
    edge midpoint, produced by the SEGMENT rule — the default barycenter
    coordinate map is exactly the midpoint; no custom coordinate op.
    (c) Canonical conventions pinned from `plexrefregular.c:713-738` and
    `DMPlexGetRawFaces_Internal`: tet faces `f0=[v0,v1,v2], f1=[v0,v3,v1],
    f2=[v0,v2,v3], f3=[v2,v1,v3]`; closure edges `e0=[v0,v1] …
    e5=[v2,v3]`; cone-entry grammar `[type, Npath, d_1..d_Npath, r]` with
    one `ornt` per cone point. Remaining new C: the 6 tet cell-transform
    tables, the 3D `SetUp` (edge → supporting faces → supporting cells,
    refine-type assignment), and the identity-only tet
    `GetSubcellOrientation` case.

    **Method:** the 6 static tables are emitted by a Python generator
    (committed as a design experiment) that walks the canonical
    conventions and self-checks the produced complex against the stage-1a
    oracle symbolically before any C compiles — killing sign/path errors
    mechanically. Gate: `DMPlexCheckSymmetry/Skeleton/Faces` + volume
    conservation + oracle equality on a single tet (all 6 edges), the
    two-tets-sharing-a-face configuration (the shared-face orientation
    test), and Kuhn/Delaunay meshes.

    **DONE — PASS (2026-07-17, `test_0841`).** The generated tables landed
    in `nvb_transform.c`, together with the set-up code that finds which
    edge of each marked cell to cut, and the piece that tells PETSc the
    two child tets are only ever referenced by their own parent. Every
    case passes PETSc's full mesh-consistency battery
    (`-dm_plex_check_all`), conserves volume exactly, and reproduces the
    reference (numpy) split: all 6 edges of a single tet, six
    configurations of two tets sharing a face, and multi-edge batches on
    structured (Kuhn, 64-cell) and unstructured (Delaunay, 281-cell)
    meshes.

    **The gates caught two real, pre-existing bugs — worth reading as
    they affect more than this feature:**

    1. *Mistranslated face views.* When two tets look at their shared
       face from opposite sides, the transform must translate "child 0 of
       the split face, rotated this way" between the two viewpoints. The
       existing code used a shortcut for that translation which happens
       to be correct for the only two viewpoints a 2D mesh can produce —
       and wrong for the extra viewpoints that occur in 3D. It now calls
       PETSc's exact orientation-composition routine
       (`DMPolytopeTypeComposeOrientation`); behaviour in 2D is
       unchanged, bit for bit (the exact-mesh regression tests
       `test_0836/0837/0838` all pass untouched).
    2. *Tetrahedra listed the wrong way round.* The convention for the
       vertex order of a tetrahedron is the opposite handedness to the
       2D counter-clockwise rule (PETSc's reference tet is "negative" by
       the 2D intuition), and the serial engine was emitting cells
       inverted. Solvers never noticed — finite-element assembly uses the
       absolute volume — but visualisation winding, outward normals and
       boundary integrals all read the sign. The convention is now
       correct in both dimensions and locked into the adapt gate
       (`test_0840` runs the full check battery, including inversion, on
       every child mesh).
  * **1c-ii — tagged state + serial driver.** Per-cell DMLabel packing the
    Maubach state (vertex permutation 0–23 relative to the cell's closure
    order × tag 1–3 → 72 values). DGS coloring seed computed once per base
    mesh: gather the base-finest edge graph, deterministic
    coordinate-lexicographic greedy — identical on every rank by
    construction, and bounded by the BASE size (the coarse end of
    adapt-on-top), not the child. Driver decodes state → bisection edge,
    runs the same agree/drain loop, re-encodes children by the Maubach
    child rule and unsplit cells via the transform's point mapping (never
    by copying label values — the 2D lesson). Gate: native np=1 output
    equals the serial `TaggedBisectionMesh` oracle over multi-pass graded
    refinement.
  * **1c-ii status: DONE — PASS (2026-07-17, commit 6f46535e).** The native
    driver refines tetrahedral meshes at np=1 and produces the same mesh,
    cell for cell, as the serial engine over multi-pass graded refinement
    (they share one seed: `write_tagged_state_label` runs the serial
    engine's own initialization). `mesh.adapt()` on a 3D mesh now prefers
    the native driver, exactly as 2D does, and all the stage-1b gates
    (FMG parity, label transfer, orientation) pass through it.

    **One structural finding worth knowing:** the driver's conforming
    closure — the step that finds which neighbours must split first — had
    to move from "compute once up front" to "re-grow at the top of every
    pass". In 2D, splitting a blocking neighbour once always makes the
    blocked edge available next pass, so a single up-front closure was
    sufficient. In 3D, a waiting cell can find itself blocked by cells
    *created during the drain*, over several generations, and the
    original structure deadlocked. Re-running the closure is idempotent,
    so 2D behaviour is unchanged bit for bit (the exact-mesh regressions
    all pass untouched).

  * **1c-iii — parallel. DONE — PASS (2026-07-17, commit 3acdd992).**
    `mesh.adapt()` on tetrahedral meshes runs in parallel: the refinement
    state is seeded identically on every rank from geometry alone (the
    base cell list gathered by vertex coordinates, the same deterministic
    colouring run everywhere — cost bounded by the BASE mesh, the coarse
    end of adapt-on-top), and the cross-rank splitting rules the 2D engine
    already used needed **no changes** for the extra 3D mesh strata.
    Gates (`test_0842`, the 3D mirror of `test_0839`): identical global
    cell count (5198) at np=1/2/4; boundary labels survive (proven by the
    exact Poisson solution); geometric FMG matches GAMG at every
    communicator size. Full adapt-family sweep 57/57.

    **ROUND 1 (NVB-3D) IS COMPLETE.** 3D adaptive mesh refinement exists,
    serial and parallel, engine-less at the user API, with the MG gate
    green at every stage. Remaining round-1 loose end deferred to the PR
    review: a plain-language PR body per the 2026-07-17 wording ruling.

    **Mesh-evaluation findings (2026-07-18, maintainer review of rendered
    meshes; artefacts in `~/+Simulations/nvb_3d_adapt_evaluation/`):**

    * *No misses:* sampling 1720 points on a dipping fault plane, the
      containing-cell size is min 0.012 / median 0.019 / max 0.034
      against a target of 0.02 — every point of the fault surface sits in
      a cell at or near the finest size (bisection quantises h in steps
      of 2^(1/3), so under 2x target means on-target everywhere).
    * *The 3D refinement halo is wider than 2D's, and that is real, not a
      rendering artefact.* On the identical problem (the fault plane is
      invariant along strike, so the box's front wall is exactly a 2D
      fault problem), the fraction of cells finer than half the locally
      demanded size is ~73% in 3D vs ~57% in true 2D — measured by the
      driver's own volume-based h, so the face-vs-volume section effect
      accounts for almost none of it. The mechanism is the conforming
      closure: in 2D a bisection drags in at most the one neighbour
      across the refinement edge, while in 3D it must split every tet in
      the edge's star (typically 15–25), so each on-fault cut forces a
      thicker shell of neighbour splits. Bounded (the theory's closure
      constant grows with dimension) and centred on the fault; the funnel
      still beats a uniform fine mesh by an order of magnitude in cells.
    * *Marking-loop cost:* the per-generation marking in `_adapt_nested`
      computes centroids/sizes with a per-cell Python geometry call — at
      depth 3 (nine generations, 69k cells) marking dominates wall time
      over the C refinement. Fix before the PR: vectorise the
      centroid/size computation from the cell list (three lines, done in
      the evaluation scripts already).

    **PR close-out (2026-07-22).** All items above are landed:

    * *Parallel validation of the composition slice is complete.* The
      hyperbolic-profile 3D adapt gives the identical cell count (36,040)
      at np=1/2/4. The 2D redistribute-then-adapt combination runs clean
      in parallel — cell counts within the mover's documented partition
      drift of the 607-cell serial reference (604 at np=2, 583 at np=4)
      and geometric MG on the combined child converges in 2 iterations
      with the exact linear solution recovered at every communicator
      size.
    * *That validation flushed out a real parallel bug* (issue #376,
      fixed here): the canonical PETSc-sync callback on variable arrays
      also fired for fancy-indexed *copies* (`data[mask] /= s`), trying
      to pack the subset into the full-size vec. Serial: a swallowed
      warning. Parallel: the mask is partition-dependent, so some ranks
      abandoned the pack *before* its collective sync while others
      entered it — mismatched collectives that hung the MMPDE mover at
      np=2. The callback now only ever syncs the canonical storage it
      was registered on (mesh and swarm variables both); a regression
      test locks redistribute_nodes warning-free.
    * *Marking loop vectorised:* centroid = vertex mean, size from one
      vectorised determinant over the cell list (h = |det|^(1/dim), Gram
      determinant on manifolds). The full serial adapt-family sweep and
      the parallel confluence integers are unchanged.
    * *`profile="hyperbolic"` added* to the surface metric helpers with
      the evaluation's guidance in the docstrings: hyperbolic for the
      best transition fidelity per cell, gaussian for a uniform-size
      corridor, linear for the minimum cell count.
    * *Adversarial review (independent agent) found two real defects in
      this branch's own additions, both fixed and regression-tested:*
      the moved-coordinate carry wrote through DMClone's *shared*
      coordinates Vec (silently moving the parent's static hierarchy —
      self-masking, since the next moved-check compared moved against
      moved; the clone now installs a duplicated Vec), and the #376
      guard's `np.may_share_memory` probe is False for any zero-size
      array, so a rank with an empty local slice would have skipped the
      collective sync other ranks entered — the same asymmetry class
      the fix targets; view-vs-copy is now decided by identity in
      numpy's base chain, which follows the indexing statement and is
      rank-independent. The distributed seed also gained a loud guard
      against near-duplicate vertex coordinates collapsing under the
      rounding key.
- **1d. Integration** — lift the `_adapt_nested` dim guard and the engine-less
  `adapt()` 3D refusal; callable exact-distance metrics via the existing 3D
  `Surface` distance primitives; correct the `engine="sbr"` 3D claim in docs
  if the phase-0 suspicion is confirmed.
- **1e. 2D unification (tech-debt ruling; breaking changes sanctioned)** —
  the maintainer confirmed (2026-07-17) that NVB adapt-on-top has never been
  in production beyond its tests, so **now is the time for breaking
  changes**: the unified tagged-simplex engine *replaces* the 2D path
  outright and the slot-based 2D cell rule (and, once the native path
  covers it, the serial slot bookkeeping) is **deleted, not deprecated**.
  The 2D confluence integers in `test_083x` are regenerated as part of
  round 1's gates rather than in a compatibility-careful follow-up. Stages
  1b/1c are written dimension-generally from the start (the tagged data
  model and DGS seed are n-general; only the polytope tables are
  per-dimension).

The 2D marker-replay checkpoint design (deterministic replay from
per-generation marked sets + state labels) carries over unchanged and stays
out of scope here, as in 2D.

### Round 2a — 3D MMPDE, serial (`feature/mmpde-3d`)

- `_signed_volumes` in `graph.py`; dimension dispatch in `_mmpde_mover`
  (`_tet_cells` / `_signed_volumes` / `fact = d!`); lift the mover guard and
  the `redistribute_nodes` gate for `cdim==3` simplex meshes.
- Tet analogue of `mesh_metric_mismatch` (skip logic) — small: tet volumes +
  edge lengths in place of areas.
- `follow_metric` 3D: normalised tet quality for the polish step, or skip the
  polish with a documented warning (recommendation: implement the quality —
  it is a formula, not a subsystem).
- **Validation** (the actual substance of this phase — the discretization is
  believed correct by construction, the 3D *behaviour* is unproven):
  - Gaussian-bump isotropic equidistribution on `UnstructuredSimplexBox` 3D:
    energy monotone, folded=0, NN-spacing ratio follows the metric.
  - Anisotropic plane-feature metric `M = I + (R²−1)·exp(−(d/w)²)·n nᵀ`
    (the fault-band form): clusters AND aligns, non-tangling at step_frac up
    to the 2D-validated overdrive, similarity of behaviour to the 2D
    equivalent slice.
  - Boundary slip on `SphericalShell` (curved 3D boundary): nodes slide
    tangentially, shell radii preserved to the facet-restore tolerance.
  - Contract tests mirroring `test_0764` / `test_0850` on 3D fixtures; flip
    the negative 3D-raises tests.
- Cost note: per-outer-iteration algebra is batched 3×3 (fine); the FD
  metric-variation term is 2·d analytic metric evaluations per iteration —
  measure, and if hot, evaluate the Shepard-baked metric's gradient instead.

**Round-2a status: DONE (2026-07-23, worktree `mmpde-3d`).** The rim
locks fell as predicted (`_tet_cells` / new `_signed_volumes` /
`fact = d!`; guards flipped; `mesh_metric_mismatch` and the
`follow_metric` polish gained their tet forms). The one REAL blocker
was not on the list: `_pinned_mask` stopped its label closure at
edges, and 3D gmsh labels tag *faces only*, so on any 3D mesh **no
boundary vertex was classified at all** — nothing pinned, nothing
slipped, the whole boundary drifted freely into the interior. Every
tagged non-vertex point now closes down to its vertices (bit-identical
in 2D, where an edge's closure is exactly its endpoints). With that
fixed, first light on the dipping-fault-plane box: monotone energy
descent, zero folds, boundary nodes held **exactly** on their faces
under slip, and h-grading ratio 1.12 vs 1.30 for the identical 2D
problem — same family, compressed by the dimension exponent (the
ideal equidistribution ratio for this metric is 4× in 2D but only
2.5× in 3D, h ∝ ρ^(−1/d)); repeated calls hold a stable equilibrium,
matching the 2D "maintains, does not compound" behaviour.
`follow_metric` runs end-to-end on 3D. Full mover/adapt family sweep:
215 green serially; the negative 3D-raises contract tests flipped to
capability tests (`test_0764`, `test_0850_mesh_smoothing`). Spherical
slip validation deferred to round 3b with the rest of the curved-
boundary work.

### Round 2b — 3D MMPDE, parallel (same worktree, gated separately)

The mover's parallel machinery (coordinate-DM `localToGlobal(ADD_VALUES)`
velocity assembly, collective line-search predicates, halo sync) is
dimension-general, so this phase is expected to be *gates, not code*:
np2/np4 parity tests mirroring the 2D contract (velocity assembly
bit-identical; the known ~1e-4%-level step-cap partition drift documented in
`mmpde.py:487-495` applies unchanged). Any 3D-specific divergence is a bug to
fix, not a new mechanism to build.

**Round-2b status: DONE (2026-07-23).** Gates, not code, exactly as
predicted — zero 3D-specific changes were needed. The dipping-plane
box gate at np=1/2/4: runs clean at every rank count (exercising the
post-#379 collective sync machinery in 3D), zero folds everywhere,
boundary nodes held exactly on their faces, and the graded
near/far h medians agree across partitions to ≤2.5% each (grading
ratio 1.121 / 1.088 / 1.077) — the same few-percent equilibrium
drift the rank-local step cap produces in the 2D combination gate.

### Round 3a — unified adapt + redistribute workflow

Composition semantics need one design commitment: **redistribute-then-adapt**
is the safe order. `adapt()` re-marks from the static base each call, so
moving *base* nodes first is coherent; moving a *child's* nodes after adapt
would invalidate the coordinate-built custom-P transfers (the MG tail would
need a rebuild) and the child is discarded on re-adapt anyway. Deliverable: a
worked example (3D box or annulus fault: metric-driven redistribution of the
base + NVB band on the fault + Stokes FMG + advection-diffusion with field
transfer across re-adaptation), plus a short how-to in `docs/advanced/`.

**Round-3a status: 3D combination WORKS (2026-07-24, worktree
`adaptivity-round3`).** First run stalled the refinement drain — and the
cause was a PETSc label-semantics trap, not the driver: `DMLabelSetValue`
does not remove a point from its previous stratum, so RE-seeding the
tagged refinement state on the moved base left every cell in two strata
and readers got the *old* (unmoved-order) state back — the driver was
silently running the moved geometry with a stale seed. (The driver
itself was first cleared by exhaustive oracle sweeps: all 72 single-tet
states × 3 passes, and all 3,456 engine-valid two-tet state pairs with
single-cell marking — native vs Python engine, all equal.)
`write_tagged_state_label` now destroys and recreates the label;
regression test in `test_0840`. Result on the dipping-fault box (ml=2):
mover+adapt gives **33,570 cells vs 36,040 adapt-only**, slightly finer
on-fault (med h 0.0282 vs 0.0292) with the over-refinement fraction
down from 46% to 39% — the same budget-spends-better behaviour as 2D.
MG on the combined child: 3 iterations, exact solution to 9e-10.
Renders: `~/+Simulations/nvb_3d_adapt_evaluation/combo3d_*.png`. The
worked example + `docs/advanced/` how-to remain the landing
deliverable.

### Round 3b — spherical geometry + curved/internal-boundary clean-ups

- **MMPDE on shells**: `SphericalShell` (solid, `dim==cdim==3`) is a valid
  redistribution target once phase 1 lands; slip on inner/outer spheres runs
  through the already-3D facet-restore chain. Validation: stagnant-lid-style
  metric on the shell, folded=0, radii preserved.
- **NVB on curved boundaries — the one real design question.** Bisection
  midpoints are chord midpoints: new boundary vertices of an adapted annulus /
  shell child lie on chords of the *base* polygon/polyhedron, so boundary
  geometry stays frozen at base resolution no matter how deep the refinement.
  Options: (a) accept chords (status quo — the 2D annulus fault studies ran
  this way); (b) **snap new boundary vertices to the analytic surface**
  (the mechanism exists: `SphericalShell`'s refinement callback /
  `BoundingSurface`), making geometry converge with refinement. Snapping
  perturbs the custom-P transfer geometry (fine nodes marginally outside
  coarse cells — the barycentric builder must tolerate slight extrapolation,
  the RBF builder is indifferent) and the confluence integers (snap must be
  partition-independent — it is, being coordinate-driven). Recommendation:
  probe (b) on the 2D annulus first (cheap, the machinery is live), rule, then
  apply the ruling to 3D.
- **Rotated free-slip on adapted spherical children**: expected to compose
  already (per-node analytic normals `X/|X|`; the reaction/dynamic-topography
  path is boundary-label-driven). Validation: shell response benchmark
  (test_1064 pattern) on an NVB child.

**Round-3b probe results (2026-07-24):**

* *MMPDE on `SphericalShell`: validated, zero new code.* With the
  registered radial `BoundingSurface`s, a boundary-layer metric toward
  the inner sphere gives zero folds and holds BOTH spheres to machine
  precision (radius drift ≤ 2e-16) while the radial cell-size profile
  grades to ratio ~1.32 (better than the box's 1.12 — the boundary-layer
  metric shape suits the mover well).
* *The chord error is real and measured*: adapted annulus children carry
  an outer-boundary radius error of 1.8e-3 at base h≈0.125
  (the h²/8R chord sag), frozen at base resolution regardless of depth.
* *Snap is safe on every gate*: projecting the child's boundary vertices
  onto the analytic circles gives zero folds, radius error → 2e-16, and
  the custom-P MG on the snapped child is untouched (2 iterations,
  solution identical to the GAMG reference) — the coordinate-built
  transfers tolerate the slightly non-nested snapped boundary exactly as
  designed. **The chord-vs-snap ruling is now a pure decision** — no
  technical blocker either way.
* *A real MG gap found and fixed on the way*: the mesh-owned FMG
  auto-pickup rejected any square finest transfer as "no coarsening" —
  but a generation whose new vertices all land on a Dirichlet boundary
  adds only constrained dofs, making the free-dof counts of the last two
  levels coincide. Boundary-focused metrics (what curved domains invite)
  hit this every time, so every annulus boundary-layer adapt was
  silently falling back to the default preconditioner. Guard relaxed to
  genuine row/operator mismatches; annulus children now solve with
  custom MG in 2 iterations.
* *Internal interfaces resolved (maintainer discussion, 2026-07-24):*
  refinement always preserved embedded interfaces topologically
  (conforming bisection + label carry), but they were chord-frozen and
  invisible to the snap. The Internal circle/sphere of the
  *InternalBoundary meshes now registers as a radial surface flagged
  `interior=True`: adapt() snaps refinement onto the true interface
  radius, while the movers keep interface nodes FULLY pinned (no normal
  or tangential motion, even with slip_surfaces=True) — interface
  motion is physics-owned. Tangential slide along an interface is a
  one-flag change if ever wanted.

## Refinement-mechanism study: centroid (Alfeld) vs bisection (2026-07-25)

Maintainer proposal: split marked tets at the CENTROID instead of
bisecting an edge. A centroid split leaves the parent's four faces
untouched, so a refined cell stays conforming with unrefined neighbours
at ANY depth difference — no closure, and therefore none of the
refinement "escape" (the halo of forced neighbour splits) that bisection
pays. Probes in `~/+Simulations/nvb_3d_adapt_evaluation/`
(`centroid_refine_probe.py`, `centroid_solver_quality_probe.py`).

**The geometry side confirms the proposal exactly.** At matched on-fault
cell size the centroid child needs HALF the cells (19,136 vs 37,446),
the over-refinement fraction collapses from 45.8% to 8.9% (the residue
is just the metric's own quantisation), the mesh is conforming by
construction (zero non-conforming faces; passes the full `DMPlexCheck`
battery), and in parallel it would need no cross-rank reconciliation at
all — every new point is interior to its cell.

**The solver side kills it as a DEEP mechanism.** Solving a manufactured
Poisson problem on both meshes (P1, relative L2 error, in-band and
global), the bisection error falls monotonically with refinement while
the centroid error STALLS and creeps up — ending slightly worse than the
unrefined base while spending more dofs:

| on-fault target | bisection in-band err | centroid in-band err | max dihedral (bis / cen) |
|---|---|---|---|
| base | 0.114 | 0.114 | 148° |
| 0.12 | 0.111 | 0.119 | 167° / 170° |
| 0.09 | 0.100 | 0.122 | 173° / 176° |
| 0.06 | 0.086 | 0.124 | 172° / **179.0°** |
| 0.045 | 0.079 | 0.124 | 173° / **179.6°** |

The mechanism is the maximum-angle (Babuska-Aziz) condition, not the
minimum angle: at 179.6 degrees the elements are flat and P1
interpolation on them is no better than on the coarse parent, however
many are added. Jacobi-CG iterations also double (230 vs 113). Element
quality halves per generation (q_med 0.62 → 0.50 → 0.31 → 0.14 → 0.08),
so a well-shaped starting mesh buys a constant, not a slope.

**Ruling: centroid refinement is a SHALLOW tool.** At one-to-two
generations the two methods are at parity (1.07x the error, 1.02x the
iterations, 109% of the dofs) and centroid gets there with no closure —
which is also how Alfeld splits are used in practice (one split of a
good mesh, for Scott-Vogelius stability). The open follow-up worth
testing: ONE centroid pass as the FINAL generation on top of a bisection
hierarchy, where the halo costs the most cells and only the first
quality step is paid. PETSc already ships the uniform Alfeld transform
(`DMPLEXREFINEALFELD`); it does not consult the framework's active
label, so a selective version would be our own — but a far simpler
transform than the bisection one (no closure, no state label, no
cross-rank fixed point).

## Base-mesh provenance: where the starting quality goes (2026-07-25)

Prompted by the 148-degree max dihedral of the standard base. Findings:

* PETSc's uniform 1:8 tet refinement splits the interior octahedron
  along a FIXED reference diagonal (`plexrefregular.c`, all four inner
  children share the 3-7 pair), not the shortest physical one. It costs
  a ONE-TIME hit — identical statistics at refinement levels 1 and 2
  (148.4 degrees, q_min 0.275 both) — and the degraded cells are
  INTERIOR, not boundary slivers (at level 2 the worst 50 cells are 32%
  boundary-incident against 55% for the mesh as a whole).
* **But gmsh is not a 120-degree mesher.** The 120.6 degrees measured on
  the cellSize=0.4 base is a 184-cell artefact where every cell is
  boundary-fitted; at a realistic 699 cells (cellSize 0.2) gmsh's own
  worst dihedral is 155 degrees — worse than the uniform split's 148.
  Median quality is the honest discriminator: 0.74 (raw gmsh) vs 0.62
  (uniformly refined).
* Adapting from a raw gmsh base of the same resolution (no uniform
  refinement — admissible because the refinement generations are
  themselves MG levels and the custom-P transfers accept non-nested
  pairs) gives **~11% lower solution error at equal dofs** (8.0e-2 vs
  9.0e-2 at ~2,100 dofs), with better q_med (0.407 vs 0.379) and q_min.
  Real but modest; worth offering as an option rather than a default.
* Worth noting for both studies: a bisection child carries a max
  dihedral of 171-174 degrees and still converges cleanly. The
  discriminator is not the worst angle but the FRACTION of
  near-degenerate cells (bisection 1.9% below q=0.1; centroid 70.7%).

## Effort and risk, honestly

| round | new-code size | risk | notes |
|---|---|---|---|
| 1. NVB-3D | large (oracle ~300 py; C: tables + face-stratum SF) | **high** | the tet orientation tables and the face-stratum SF reconciliation are the two dragons; serial oracle de-risks the algorithm before any C |
| 2a. MMPDE-3D serial | small (~100 lines + tests) | **low-medium** | core is dim-general; risk is *behavioural* (3D tangling resistance, tet quality under strong metrics) — precisely what the validation ladder measures |
| 2b. MMPDE-3D parallel | tiny (gates) | low | machinery dim-general |
| 3a. unified workflow | small-medium | low | one ordering commitment + an example |
| 3b. spherical + boundary clean-ups | medium | medium | one geometry ruling (snapping); rest is validation |

**Pause points.** This is an ambitious integration of parts that individually
work; the phasing is designed so every boundary is a clean stop:

- Stage 1a (the serial oracle) is deliberately cheap and *decides* whether the
  C investment proceeds — the same gate that worked for 2D (the 2D oracle was
  ~135 lines and settled the grading question before any C was written).
- If 1c stalls (the C tables/SF prove worse than expected), the serial 1b
  engine still gives np=1 3D adapt-on-top with FMG, which is scientifically
  usable while the parallel path waits.
- Round 2 (3D node redistribution) is independent of round 1 and ships on its
  own — a complete, useful capability (and the only mover there is; the
  retired movers never did 3D either).

## Maintainer rulings and open decisions

**RULED (2026-07-17):**

1. **Round order** — NVB-3D first ("existence at all in 3D is important to
   implement… the big task"), with the MG-viability gate at every stage.
   Curved / internal boundaries are later clean-up items, not round-1 blockers.

**Open (will proceed as recommended unless overruled):**

2. **3D NVB first-landing guarantee level — RESOLVED by a better option
   (stage-1a, 2026-07-17).** The DGS coloring initialization delivers the
   *full* guarantee set (termination + conformity + shape regularity + the
   sharp BDV closure bound) on arbitrary conforming base meshes at the cost
   of a trivial greedy coloring — strictly better than the face-consistent
   longest-edge compromise this decision originally weighed. Adopted;
   validated by the oracle (see stage 1a).
3. **Curved-boundary vertices under NVB refinement** (round 3b, but the ruling
   shapes 1c's coordinate hook): chord midpoints (geometry frozen at base
   resolution) vs snap-to-analytic-surface (geometry converges; recommended,
   probed on the 2D annulus first). Round 1 proceeds with chord midpoints —
   the 2D engine's current behaviour — so the snap decision stays a clean
   later layer.
4. **Small defaults**: `mesh_metric_mismatch` and the `follow_metric` tet
   quality implemented with round 2a; the `_pinned_mask` face-only-label TODO
   left as-is (gmsh meshes are unaffected); the `engine="sbr"`-on-3D
   documentation corrected once verified; design-note and internal names keep
   the algorithm names (NVB, MMPDE, Maubach/AMP) while user-facing surfaces
   stay purposeful.

## References

- W. Huang, L. Kamenski, *A geometric discretization and a simple
  implementation for variational mesh generation and adaptation*, JCP 301
  (2015) 322 (arXiv:1410.7872) — the mover; dimension-general.
- E. Bänsch, *Local mesh refinement in 2 and 3 dimensions*, IMPACT Comput.
  Sci. Eng. 3 (1991) 181.
- J. Maubach, *Local bisection refinement for n-simplicial grids generated by
  reflection*, SISC 16 (1995) 210; C. Traxler, *An algorithm for adaptive mesh
  refinement in n dimensions*, Computing 59 (1997) 115.
- D. Arnold, A. Mukherjee, L. Pouly, *Locally adapted tetrahedral meshes using
  bisection*, SISC 22 (2000) 431 — the marked-tet formulation.
- R. Stevenson, *The completion of locally refined simplicial partitions
  created by bisection*, Math. Comp. 77 (2008) 227 — compatible labelling,
  O(#marked) closure.
- L. Diening, L. Gehring, J. Storn, *Adaptive mesh refinement for arbitrary
  initial triangulations* (arXiv:2306.02674; FoCM 2025) — the coloring
  initialization adopted here: Maubach's routine on ANY conforming initial
  mesh, with termination, 36 similarity classes per base tet, shape
  regularity, and the sharp BDV closure estimate.
- `NVB_GRADED_ADAPT.md`, `LAYER2_SBR_ADAPT_ON_TOP.md` (this repo) — the 2D
  engine, the single-bisection finding, the parallel confluence machinery.
