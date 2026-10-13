# Point-location capability and the evaluation fallback ladder

*2026-07-23 — design record for issues #390 / #392 (PRs #391 and the
capability-ladder follow-up). The full discussion, probe numbers, and
acceptance criteria live on issue #392.*

## The problem

PETSc's `DMLocatePoints` uses a half-open cell convention: a query point
sitting exactly on the domain's closed upper boundary faces belongs to the
cell "above", which does not exist, so the point is silently dropped. The
evaluator used to zero-fill dropped points — a plausible-looking field value
that propagates silently. Semi-Lagrangian stress-history trace-backs slide
departure points along boundaries, so on quad meshes every step read a
corrupted history there (the VEP stability blow-up of issue #390).

Underworld's own cell-wall estimator (`_test_if_points_in_cells_internal`,
a half-space intersection over face control-point planes) handles on-face
points correctly — but it is only *authoritative* on some geometries.

## When the estimator is authoritative

Its authority is governed by exactly two measurable properties: **face
planarity** (sagitta — max vertex deviation from the face's plane, relative
to face diameter) and **cell convexity** (no cell vertex outside another
face's plane). Measured regimes:

| Capability | Meshes | Authority |
|------------|--------|-----------|
| `exact` | simplex; manifold (dim ≠ cdim); **any valid 2-D quad mesh, deformed included** (straight edges are always planar, convexity ⇔ untangled); planar-faced hexes (rectilinear boxes, affine images) | all field types — bypass `DMLocatePoints` |
| `continuous` | warped hexes within sagitta ≤ 5e-2 (cubed-sphere class: lateral faces exactly planar, spherical faces ≤ 1e-2) | continuous fields only — a misclassified point lands in the face-adjacent neighbour where continuous FE interpolants agree to O(sagitta × gradient); a face-aligned jump (e.g. layered viscosity on a spherical shell) would see O(jump) errors |
| `none` | badly warped or non-convex cells | none — PETSc locates; dropped points take the RBF fallback |

The capability is a **measured mesh property**, not a type flag:
`Mesh._location_capability()` runs the geometry audit
(`_audit_cell_face_geometry`) and caches against
`(_mesh_version, _topology_version)`, so `deform()` and adaptation refresh
it automatically. It is per-rank (the evaluator runs on `COMM_SELF`; a
collective reduction here could deadlock since `petsc_interpolate` is
reached only by ranks holding points).

## The fallback ladder

1. **Authoritative** (`exact`, or `continuous` with all-continuous fields):
   the estimator's owner is passed as an authoritative hint and
   `DMLocatePoints` is bypassed (`petsc_tools.c`), with reference-coordinate
   clamping for on-face queries.
2. **Not authoritative**: `DMLocatePoints` decides. Points it drops get
   **NaN** in the interpolant (never zeros) and are reported in
   `unlocated_mask`; the caller fills them per-variable via
   `rbf_interpolate` — bounded, topology-free, honest. NaN survives only if
   that plumbing is bypassed, which is exactly when it should be visible.
3. Genuinely out-of-domain points never reach this machinery — they are
   classified out by `points_in_domain` and take the existing RBF
   extrapolation path.

The policy decision lives in ONE place —
`Mesh._hint_is_authoritative(all_fields_continuous)` — consumed by
`petsc_interpolate` (which also folds the policy into the DMInterpolation
cache key, since the same coords with a different field-continuity mix can
need both structures).

## Sentinel tests

- `tests/test_0503_evaluate.py::test_evaluate_on_domain_boundary_faces` —
  boundary-face evaluation exact on quad and simplex boxes (the #390 class).
- `test_location_capability_measured` — capability grading of regular /
  deformed quad and hex meshes.
- `test_evaluate_deformed_quad_boundary_exact` — the 2-D graduation.
- `test_evaluate_warped_hex_rbf_fallback_no_silent_values` — the RBF rung:
  no NaN escapes, no silent zeros, values bounded by the data range.
