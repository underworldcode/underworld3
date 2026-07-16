# Adaptivity in 3D and spherical geometry — the MMPDE + NVB capstone

Status: **PROPOSED** (phase-0 design note, 2026-07-17). Awaiting maintainer
rulings on the decisions in the final section before implementation starts.

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
2. **Initial labelling / termination.** In 2D the longest-edge seed always
   conforming-terminates. In 3D, arbitrary seeds can cycle; the standard
   practical choice is **face-consistent marks** — each face's marked edge is
   its longest, with a *geometric, partition-independent* tie-break (both tets
   sharing a face then agree by construction, on any rank). This guarantees
   termination, conformity and boundedly many similarity classes
   (Bänsch 1991; AMP 2000); the *sharp* O(#marked) closure constant needs
   Stevenson's (2008) compatible initial labelling, which — exactly as the 2D
   note ruled — is a follow-up, not a first-landing requirement. (Kossaczký
   pre-refinement is rejected: it would alter the user's base mesh, which is
   also the MG tail.)
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

### Phase 1 — 3D MMPDE, serial (`feature/mmpde-3d`)

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

### Phase 2 — 3D MMPDE, parallel (`same worktree, gated separately`)

The mover's parallel machinery (coordinate-DM `localToGlobal(ADD_VALUES)`
velocity assembly, collective line-search predicates, halo sync) is
dimension-general, so this phase is expected to be *gates, not code*:
np2/np4 parity tests mirroring the 2D contract (velocity assembly
bit-identical; the known ~1e-4%-level step-cap partition drift documented in
`mmpde.py:487-495` applies unchanged). Any 3D-specific divergence is a bug to
fix, not a new mechanism to build.

### Phase 3 — 3D NVB (`feature/nvb-3d`; the major work package)

Strictly serial-oracle-first, replaying the 2D de-risking sequence:

- **3a. Serial oracle** — `NVBMesh3D` (pure numpy, mirroring `NVBMesh`):
  marked-tet data model (decide Maubach vs AMP encoding here), recursive
  compatible-bisection closure, conformity / bounded-closure / similarity-class
  diagnostics. Acceptance mirrors the 2D prototype: one tet deep in a uniform
  patch refines O(1)-locally; graded 3D bullseye; 0 hanging faces/edges.
- **3b. DMPlex wrap + serial engine** — `from_dm`/`to_dm` with boundary-face
  and region label transfer; wire as the np=1 `engine="nvb"` path for tets;
  Poisson + FMG-vs-GAMG parity on the graded child.
- **3c. Native transform** — extend `nvb_transform.c`: the TETRAHEDRON
  single-edge-split production rule + orientation tables; per-cell state label
  (from 3a) maintenance across passes; agree/bisect SF reconciliation extended
  over the face stratum; the same drain-loop driver. Bit-confluence integers
  at np1/2/4 mirroring `test_0839`; FMG parity (Poisson + 3D Stokes velocity
  block).
- **3d. Integration** — lift the `_adapt_nested` dim guard and the engine-less
  `adapt()` 3D refusal; callable exact-distance metrics via the existing 3D
  `Surface` distance primitives; correct the `engine="sbr"` 3D claim in docs
  if the phase-0 suspicion is confirmed.

The 2D marker-replay checkpoint design (deterministic replay from
per-generation marked sets + state labels) carries over unchanged and stays
out of scope here, as in 2D.

### Phase 4 — unified adapt + redistribute workflow

Composition semantics need one design commitment: **redistribute-then-adapt**
is the safe order. `adapt()` re-marks from the static base each call, so
moving *base* nodes first is coherent; moving a *child's* nodes after adapt
would invalidate the coordinate-built custom-P transfers (the MG tail would
need a rebuild) and the child is discarded on re-adapt anyway. Deliverable: a
worked example (3D box or annulus fault: metric-driven redistribution of the
base + NVB band on the fault + Stokes FMG + advection-diffusion with field
transfer across re-adaptation), plus a short how-to in `docs/advanced/`.

### Phase 5 — spherical geometry

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

## Effort and risk, honestly

| phase | new-code size | risk | notes |
|---|---|---|---|
| 1. MMPDE-3D serial | small (~100 lines + tests) | **low-medium** | core is dim-general; risk is *behavioural* (3D tangling resistance, tet quality under strong metrics) — precisely what the validation ladder measures |
| 2. MMPDE-3D parallel | tiny (gates) | low | machinery dim-general |
| 3. NVB-3D | large (oracle ~300 py; C: tables + face-stratum SF) | **high** | the tet orientation tables and the face-stratum SF reconciliation are the two dragons; serial oracle de-risks the algorithm before any C |
| 4. unified workflow | small-medium | low | one ordering commitment + an example |
| 5. spherical | medium | medium | one geometry ruling (snapping); rest is validation |

**Pause points.** This is an ambitious integration of parts that individually
work; the phasing is designed so every boundary is a clean stop:

- After phase 2, **3D node redistribution ships on its own** — a complete,
  useful capability (and the only mover there is; the retired movers never
  did 3D either).
- Phase 3a (the serial oracle) is deliberately cheap and *decides* whether the
  C investment proceeds — the same gate that worked for 2D (the 2D oracle was
  ~135 lines and settled the grading question before any C was written).
- If 3c stalls (the C tables/SF prove worse than expected), the serial 3b
  engine still gives np=1 3D adapt-on-top with FMG, which is scientifically
  usable while the parallel path waits.

## Decisions needed from the maintainer

1. **Commit / pause structure.** Proceed with phases 1–2 (3D MMPDE) now, with
   an explicit re-assessment at the phase-2 → 3 boundary before the C
   investment? (Recommended: yes — 1–2 are contained; 3 is where the effort
   concentrates and 3a is the cheap go/no-go probe.)
2. **3D NVB first-landing guarantee level.** Accept
   termination + conformity + shape-regularity via face-consistent
   longest-edge seeding (geometric tie-breaks), with Stevenson's compatible
   initial labelling (the sharp O(#marked) constant) as a follow-up — matching
   the 2D precedent? The alternative (compatibility from day one) adds
   substantial combinatorial work for a constant-factor guarantee.
3. **Curved-boundary vertices under NVB refinement** (phase 5, but the ruling
   shapes 3c's coordinate hook): chord midpoints (geometry frozen at base
   resolution) vs snap-to-analytic-surface (geometry converges; recommended,
   probed on the 2D annulus first)?
4. **Small defaults** (will proceed as stated unless overruled):
   `mesh_metric_mismatch` and the `follow_metric` tet quality implemented in
   phase 1; the `_pinned_mask` face-only-label TODO left as-is (gmsh meshes
   are unaffected); the `engine="sbr"`-on-3D documentation corrected once
   verified; design-note and internal names keep the algorithm names
   (NVB, MMPDE, Maubach/AMP) while user-facing surfaces stay purposeful.

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
- `NVB_GRADED_ADAPT.md`, `LAYER2_SBR_ADAPT_ON_TOP.md` (this repo) — the 2D
  engine, the single-bisection finding, the parallel confluence machinery.
