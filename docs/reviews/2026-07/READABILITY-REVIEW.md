# Readability Review — July 2026 Quality Campaign (Dimension 4)

**Base**: `development` @ `1d003481` (audit worktree `.claude/worktrees/quality-audit-2026-07`)
**Date**: 2026-07-03
**Scope**: readability of the June-2026 change hotspots — `meshing/smoothing.py`,
`cython/petsc_generic_snes_solvers.pyx`, `discretisation/discretisation_mesh.py`,
`systems/ddt.py`, `systems/solvers.py`, `utilities/rotated_bc.py`,
`utilities/custom_mg.py`, `discretisation/remesh.py`.
**Standard applied**: the founding rule — *anyone should be able to read the code and
understand it* — assessed for a working geodynamicist reading each file cold.

---

## Overview

June 2026 added ~132 commits / +12.6k lines to `src/`, largely AI-assisted across many
non-overlapping sessions. This audit read the eight hotspot files and produced
**118 findings**: 14 adversarially verified (every `file:line` re-read in the worktree,
several with corrections recorded below) and 104 unverified observations queued for
verification before remediation. No refuted findings survived to the tables; nine
refuted *sub-claims* are recorded in the appendix so they are not re-found.

The consistent picture: **micro-readability is good to excellent, macro-structure is
poor**. Individual June additions carry intent-stating comments a geodynamicist can
follow (Nitsche local-h, FMG sidecar splice, rotated-BC lifecycle notes, DDt class
docstrings, remesh REMAP/CARRY/REINIT narrative). But each session added its feature by
copy-paste rather than extraction, so the same machinery now exists in 2–6 diverging
copies; dead code and stale docstrings from superseded designs were left in place; and
two files (`smoothing.py` at 4,518 lines, `petsc_generic_snes_solvers.pyx` at 8,150
lines) have grown past the point where any reader can hold them.

Three findings are severe enough to flag individually:

1. **READ-01** — `_winslow_mmpde`'s 3D branch references an undefined function
   (`_signed_volumes`, `smoothing.py:3124`); the docstring's "Dimension-general
   (d = 2, 3)" claim is false and any 3D MMPDE call raises `NameError`.
2. **READ-02** — `_winslow_anisotropic` contains a ~100-line inline duplicate
   (`smoothing.py:2152–2252`) of its own `_build_M_tensor()` closure (2060–2150),
   diverged only by `_`-suffixed variable names.
3. **READ-03** — the four solver classes in `petsc_generic_snes_solvers.pyx`
   quadruplicate their setup/BC/solve machinery, and the copies have **silently
   diverged in BC-label semantics** (2-vs-2: `str(boundary)` vs `"UW_Boundaries"`),
   so any shared-helper extraction must preserve the divergence deliberately.

---

## Changes Made

None. This is an audit-only deliverable; no source files were modified. This document
is the only artifact.

---

## System Architecture

What this dimension's reading revealed, per subsystem, for the maintainer.

### The mesh-mover module (`meshing/smoothing.py`, 4,518 lines) — *not graded; worst structural debt of the set*

The file is an accumulation of graft events, and its own banners say so
(`# ===== grafted from feature/elliptic-ma =====`, line 2930). It contains **six
movers**, five carrying a `_winslow_` prefix that is wrong for four of them:
`_winslow_spring` (506) is a truss-energy nonlinear-CG spring solver,
`_winslow_elliptic` (1162) is a Benamou–Froese–Oberman Monge–Ampère Picard,
`_winslow_equidistribute` (1453) is an OT-improvement step, `_winslow_mmpde` (2992) is
Huang–Kamenski MMPDE; only `_winslow_anisotropic` (1718) is genuinely Winslow. The
sixth mover (graph-Laplacian Jacobi, `metric=None`) is inline in the dispatch after
line 3657, unprefixed. The public API strings are already honest (`"spring"`, `"ma"`,
`"ot"`, `"anisotropic"`, `"mmpde"`) — the lie is internal-only, which makes the rename
cheap.

The module docstring (lines 1–59) describes a superseded design: it calls the spring
path Jacobi relaxation, "Status: under development", weak grading 1.03 (lines 33–37) —
directly contradicted by `_winslow_spring`'s own docstring, which explains it replaced
Jacobi sweeps *because* they stalled. The docstring never mentions `mmpde`, yet the
file's own `DeprecationWarning` (3627–3633) tells users to "Prefer 'mmpde' for
production adaptive meshing". Note the API **default** is still `method="spring"`
(2623, 3525) and in-tree callers use `"ma"`/`"ot"` — mmpde is *recommended*, not
default.

Duplication is the dominant failure mode: ~130 lines of verbatim ≥8-line blocks
(~260 counting both copies) plus substantial near-duplication — the signed-area
backtrack exists three times (only one copy got the sliver-floor improvement), the
per-vertex step cap twice, the `_build_M_tensor` body twice (READ-02), the
displacement reweighting twice, mean-edge-length twice, and ~19 inline
`from mpi4py import MPI as _MPI` imports guard 43 allreduce sites. Eight module-level
caches coexist. The long-term fix is the package split (READ-05), but note that
private names (`_edge_pairs`, `_tri_cells`, `_pinned_mask`, …) are imported by
production code in `meshing/surfaces.py`, `meshing/_ot_adapt.py`,
`systems/solvers.py`, `discretisation/discretisation_mesh.py` and by tests
0750/0760/0762/0763/0855 — a split must re-export those, not just the public API.

One finding here crosses from readability into defect: the 3D MMPDE branch is
un-executable (READ-01).

### The solver core (`cython/petsc_generic_snes_solvers.pyx`, 8,150 lines) — *not graded; under the no-numerics constraint*

Four classes (`SNES_Scalar`, `SNES_Vector`, `SNES_MultiComponent`,
`SNES_Stokes_SaddlePt`) each carry a private copy of: the xxhash
coordinate-change preamble, the `PetscDSAddBoundary_UW` BC-registration loops, the
`_setup_solver` copyFields/copyDS/createClosureIndex/SNES tail, and the solve()
epilogue (whose identical "sync `_gvec` / invalidate cached views" stanza proves bug
fixes are already being applied four times). Critically, the copies are **not**
verbatim: essential BCs register under `str(boundary)` in Scalar (2648) and SaddlePt
(6895) but under `"UW_Boundaries"` in Vector (3570, comment `# was: str(boundary)`) and
MultiComponent (4402), and SaddlePt's versions interleave multiplier-constraint and
fieldsplit logic. Any helper extraction (READ-03) must parameterize this 2-vs-2
divergence or it will silently change BC behaviour — hence the campaign rule that this
file gets naming/docs/dead-code changes only, with structural extraction in a
separately benchmarked wave.

Beyond duplication, the file reads as a fossil record: `from xmlrpc.client import
Boolean` is line 1; four error paths `raise("string")` (a `TypeError` at runtime, not
the intended message — 1383/1386/1466/2022); two `if True: #  c_label and label_val
!= -1:` guards wrap entire BC-wiring blocks (3884, 7140); 40-line commented-out
rebuild paths and unresolved design musings ("LM: this is probably not something we
need…") interrupt the narrative; and the hedging name `_maybe_install_snes_update`
(325) violates the maintainer's twice-stated naming rule.

### The mesh god-module (`discretisation/discretisation_mesh.py`, 6,616 lines) — **grade C+**

Individual June methods (boundary_normal, cell_size, boundary_slip, the FMG sidecar
splice) are excellently narrated, but `__init__` alone spans ~830 lines mixing eight
separable concerns, and the load-bearing coordinate-update callback exists in **three
silently diverged copies** (834 in `__init__` with the identity gate and teardown
guard; 2027 without the identity gate; 6086 in `adapt()` without either). Four dead
methods, a dead-and-broken `meshVariable_mask_from_label` (bare `MeshVariable`
NameError), `if False:` chains, and a duplicated boundary-label extraction where one
copy uses the exact pattern the other copy's comment warns is a hard-abort hazard.

### Time-derivative machinery (`systems/ddt.py`, 3,676 lines) — **grade C+**

Class-level docstrings are the best in the audit (BDF/AM pairing, SLCN vs SL-BDF2
with citations), but `SemiLagrangian.update_pre_solve` is a 515-line method
interleaving six concerns, the UnitAwareArray→ND unwrap dance is copy-pasted six times
with accidental branch-order differences, ~250 lines of per-class boilerplate are
quintuplicated across the five DDt flavors, `preserve_moments` is documented but dead
(`if 0 and …`), and two `_object_viewer`s reference attributes that don't exist
(AttributeError on view).

### Solver front-end (`systems/solvers.py`, 5,067 lines) — **grade B**

Excellent physics narrative throughout; graded down for triplicated
`smoothing_length` properties, duplicated 55-line `delta_t` setters opening with bare
`except: pass`, four copies of the diffusivity-max block in `estimate_dt`,
`_apply_unit_aware_scaling` swallowing all exceptions around two redundant branches,
another banned `_maybe_` name (2264), and
`SNES_Vector_Projection.projection_problem_description` double-counting smoothing and
penalty terms its own F1 Template already contains (2985–2991) — a reader cannot tell
what weak form is assembled.

### Rotated free-slip (`utilities/rotated_bc.py`, 795 lines) — **grade B**

Genuinely well-narrated (rotate→constrain→solve→rotate-back→gauge→σ_nn is followable),
marked down for compression style: the module docstring still calls itself a
"Development version … productizes the validated prototypes" (it *is* the module),
`_zero_rows_local` exists precisely to encapsulate the np>1 overflow subtlety yet its
body is hand-inlined three more times, and pervasive semicolon-chained
create/use/destroy lines obscure the PETSc object lifecycle the file's own comments
call the trickiest part.

### Custom multigrid (`utilities/custom_mg.py`, 653 lines) — **grade A-**

Exemplary narrative: design rationale in the module docstring, the BC-per-level
invariant explicitly marked load-bearing, every PETSc quirk explained. Remaining debt
is small: mode-dependent `maps` element types read via magic index `lay[3]` (557–559),
duplicated DM-prep and zero-column guards, semicolon packing, terse parallel-path
parameter names.

### Remesh transfer (`discretisation/remesh.py`, 494 lines) — **grade B**

REMAP/CARRY/REINIT intent and the sagitta/parallel-leak mechanism are clearly stated;
loses marks for two dead functions (`_remap_one_var`, `_new_coord_cache`), the
`REMESH_MONOTONE` env knob buried inside a per-variable loop inside a try block, and
a thrice-repeated guarded write-back `except Exception: pass` whose sanctioned failure
mode is stated only once.

### Cross-cutting patterns (for the Style Charter follow-on)

- **Copy-paste is the reuse mechanism**: every multi-session file shows 2–6 copies of
  shared machinery, several already diverged. Wave D should prioritize extraction
  where divergence is accidental and *document* it where deliberate.
- **Silent `except Exception: pass`** appears in every file, almost never stating the
  sanctioned failure mode. The remesh/model-registration blocks that *do* document
  their swallow are the template to mandate.
- **Stale self-description**: docstrings and comments describing deleted designs
  (smoothing module docstring, rotated_bc "prototype" framing, ddt theta comment,
  drifted line-number references). Mechanical CI gates can't catch these; review can.
- **Hedging names** (`_maybe_*`) re-appeared twice in June despite the standing rule —
  a lint-able pattern.

---

## Findings

Ranked most-severe-first. **Wave** column maps to the campaign's remediation waves
(A = deletion/dead code, D = readability rewrite, D-doc = comment/docstring-only).
`S/M/L` = effort.

### Verified findings (adversarially verified; every line re-read in the audit worktree)

| ID | Sev | Eff | Wave | Location | Finding | Fix |
|----|-----|-----|------|----------|---------|-----|
| READ-01 | High | S | A | `meshing/smoothing.py:3124` | `_winslow_mmpde` docstring claims "Dimension-general (d = 2, 3)" but the cdim==3 branch assigns `signed_vol = _signed_volumes`, which is **defined nowhere** in the package (src-wide grep) — any 3D MMPDE call raises `NameError`; the 3D branch is dead code contradicting the docstring. | Implement `_signed_volumes(coords, tets)` (tet analogue of `_signed_areas`; `_tet_cells` at 2931 exists to feed it) **or** raise `NotImplementedError` for cdim==3 and make the docstring 2D-only; add a level_1 test for the chosen behaviour. |
| READ-02 | High | M | D | `meshing/smoothing.py:2152–2252` | `_winslow_anisotropic` inlines a ~100-line density/eigen-clamped tensor-assembly block that is line-for-line identical (modulo `_`-suffixed names) to the density/assembly portion of its own `_build_M_tensor()` closure at 2080–2150; the closure additionally refreshes `Dcoords`/`gproj` (2067–2078), so calling it once pre-loop repeats a deterministic re-solve and makes the pre-loop `gvec/gn/gmax` at 1996–2019 redundant (`old0`, `h0` must stay). | Delete the inline block; call `_build_M_tensor()` once before the outer loop (it already reads/writes the same state via `nonlocal`); remove the now-redundant pre-loop `gvec/gn/gmax` computation. |
| READ-03 | High | L | D (benchmarked wave) | `cython/petsc_generic_snes_solvers.pyx:2473ff` | Four-way duplication across `SNES_Scalar`/`Vector`/`MultiComponent`/`Stokes_SaddlePt`: xxhash coord preamble (2479, 3456, 4299, 6697), BC-registration loops (2584, 3507, 4346, 6785), `_setup_solver` tail (2892, 3943, 4721, 7326), solve epilogue (3054, 4088, 4791, 8072) — the identical "sync `_gvec`/invalidate cached views" stanza in all four epilogues proves fixes are applied ×4. **Caution**: copies are NOT verbatim — essential-BC labels diverge 2-vs-2 (`str(boundary)` at 2648/6895 vs `"UW_Boundaries"` at 3570/4402), and SaddlePt interleaves multiplier/fieldsplit logic. | Extract base-class helpers (`_coords_unchanged_since_setup`, `_register_essential_bcs`/`_register_natural_bcs`, `_finalise_dm_and_snes`, `_copy_solution_to_fields`) that **parameterize or deliberately preserve** the label divergence and give SaddlePt hook points. No-behavior-change, benchmarked wave only. |
| READ-04 | Med | L | D | `meshing/smoothing.py:794` (whole file) | 4,518-line graft-accumulation module: 6 movers, MA solver wiring, Hessian recovery, metric builders, adapters, 8 module-level caches, explicit `grafted from feature/elliptic-ma` banner (2930). | Split into a `meshing/smoothing/` package (`graph.py`, `spring.py`, `monge_ampere.py`, `mmpde.py`, `metrics.py`, `api.py`). **Must also re-export the private names** imported cross-module and by tests (`_edge_pairs`, `_tri_cells`/`_tet_cells`, `_pinned_mask`, `_auto_pinned_labels`, `_owned_vertex_mask`, `_signed_areas`, …) and keep the smoothing↔`_ot_adapt` mutual imports lazy. |
| READ-05 | Med | M | D | `meshing/smoothing.py:2403` | The coherent global signed-area backtrack exists ×3 (1412–1437, 1665–1691, 2403–2444); only the anisotropic copy got the sliver-floor improvement (`a_min_floor = 0.01·median`, 2422/2435) — the other two remain sliver-blind (`a1min > 0.0`). Per-vertex step cap duplicated at 1403–1410 / 1656–1663. **Do not** fold in mmpde's structurally different cap (3424–3430) or the spring/mmpde line searches. | Extract `_backtracked_move(..., area_floor=0.0)` and `_cap_step_to_edge_fraction(...)`; `area_floor=0.0` default keeps the two older movers bit-identical. |
| READ-06 | Med | M | D | `meshing/smoothing.py:506` | Misleading `_winslow_` prefix on 4 of 5 prefixed movers: spring equilibrium (506), Monge–Ampère (1162), OT step (1453), MMPDE (2992); only `_winslow_anisotropic` (1718) is Winslow. Public method strings are already honest, so the lie is internal — but the blast radius includes docstring references in `remesh.py:229`, `_ot_adapt.py:8`, `smoothing.py:805`, 37 files under `scripts/`, and design docs; `_WINSLOW_CACHE` shares the prefix. | Rename to `_spring_equilibrium_mover`, `_monge_ampere_mover`, `_ot_improvement_step`, `_mmpde_mover` (keep `_winslow_anisotropic`); sweep scripts/docs or keep aliases for one cycle. |
| READ-07 | Med | M | D | `meshing/smoothing.py:1362` | ~19 inline `from mpi4py import MPI as _MPI` imports (e.g. 1361–1364, 2075–2078) guarding 43 allreduce sites and 29 `if uw.mpi.size > 1:` blocks; no module-top import despite mpi4py being a hard dependency. | Import `_MPI` at module top; add `_global_min/_global_max/_global_sum` (and `_global_mean` for the `allreduce/size` sites at 1612/2014/3238) that no-op in serial; each mover shrinks ~30 lines. |
| READ-08 | Med | S | D-doc | `meshing/smoothing.py:1–59` | Module docstring describes the deleted design: spring = Jacobi relaxation "Status: under development", weak grading 1.03, "can stall against the tangle guard" (33–37) — contradicted by `_winslow_spring`'s own docstring (nonlinear CG adopted *because* Jacobi stalled). `mmpde` never mentioned though the file's own DeprecationWarning (3627–3633) recommends it for production. Corrections recorded: 5 (not 6) movers carry the prefix; mmpde is *recommended*, not the default (API default is `spring`, 2623/3525); duplication is ~130–260 lines, not ~300. | Rewrite: one paragraph per mover (what it IS, status, when to use); delete stale spring status; state mmpde is recommended-for-production; move the 2026-05-16 MA note to the section banner at 794 where it is already duplicated. |
| READ-09 | Med | S | D-doc | `meshing/smoothing.py:3931–3932` | Dead amp-inversion: `amp = refinement**(cdim/power) − 1` is never used — every `refinement is not None` path enters the envelope branch (4063) which exhaustively returns/raises before `amp`'s only consumer (4155). The `refinement` param docstring (3827–3840) and comment (3925–3930) still document the superseded override semantics; actual envelope behaviour lives only in an inline comment (4025–4062). `'arc-length'` is accepted (4124–4133) but missing from the option lists (4045–4053, 4294). | Delete 3925–3932; rewrite the `refinement` doc to describe the envelope branch and which params it ignores; add `'arc-length'` to both option lists. |
| READ-10 | Med | S | D-doc | `meshing/smoothing.py:2693, 3646–3654` | `smooth_mesh_interior` docstring says `method : {"spring", "ma"}`; the ValueError (3646–3654) lists spring/ma/ot/anisotropic but **not** `mmpde` — the mover the same function's DeprecationWarning recommends. | Docstring → `{"spring", "ma", "anisotropic", "mmpde"}` with mmpde described as recommended production mover and `'ot'` noted deprecated-only; add mmpde to the ValueError text. |
| READ-11 | Med | S | D | `meshing/smoothing.py:2999` | `_winslow_mmpde(..., **_ignored)` silently swallows typo'd `method_kwargs` (no other mover has a catch-all). **But** it is not a pure hedge: the strategy path legitimately forwards `resolution_ratio` (injected 3565–3566, consumed pre-dispatch 3581–3583, forwarded via `**mk` 3643–3644) — bare removal breaks `strategy=` + mmpde with TypeError. | Accept `resolution_ratio=None` explicitly (or pop it in dispatch), then remove `**_ignored` and warn on any remaining unknown kwargs. |
| READ-12 | Med | S | D-doc | `meshing/smoothing.py:408` | `mesh_metric_mismatch` docstring (380, 408–411) documents a 3-key return dict; the actual return (476–480) has 5 keys — and the undocumented `misalignment` key is exactly what the `skip_threshold` machinery consumes (3595). Precision: misalignment = `sqrt(1 − max(0, r)²)` (negative correlation clamps to 0 → misalignment 1.0). | Document the full 5-key dict, one line each for `alignment` (Pearson r, globally reduced) and `misalignment` (the skip criterion, with the clamp). |
| READ-13 | Med | S | D | `cython/petsc_generic_snes_solvers.pyx:325` | Hedging name `_maybe_install_snes_update` (introduced 54b815c3, 2026-06-18) violates the twice-enforced no-`maybe_`/`try_` naming rule; docstring already states the condition ("iff callbacks are registered"). One call site (1095), no external references. | Rename to `_attach_snes_update_dispatcher` (or similar does-what-it-says name); update the call site. |
| READ-14 | Med | S | A | `cython/petsc_generic_snes_solvers.pyx:3884, 7140` | `if True: #  c_label and label_val != -1:` wraps the entire natural-BC DS-wiring block in SNES_Vector and SNES_Stokes_SaddlePt — a constant-literal guard adding a spurious indent level and implying a condition that no longer exists (no else/elif; Cython constant-folds it). | Delete both lines and dedent — byte-identical semantics. |

### Unverified findings (anchors sampled, not adversarially verified — verify before remediation)

I personally re-read the anchor lines for a sample of these in the worktree
(READ-15, READ-24, READ-27/28/33/35/37/39/41/43/47/49/50/52/54/56/58/60/62/64/66/89/97,
and the import blocks) and all sampled anchors match; the full adversarial pass
(callers, reachability, already-fixed-at-HEAD checks) has **not** been run on this
table.

#### High severity

| ID | Sev | Eff | Wave | Location | Finding | Fix |
|----|-----|-----|------|----------|---------|-----|
| READ-15 | High | L | D | `discretisation/discretisation_mesh.py:255` | FILE GRADE C+. `Mesh.__init__` (255–1084, ~830 lines) mixes unit-scale derivation, file-format dispatch, boundary-enum patching, FMG sidecar splice, refinement/coarsening, coordinate-callback wiring, and sympy setup in one flat block. | Extract 8 named private methods called in sequence (`_derive_length_scale_from_model`, `_load_dm_from_file`, `_patch_boundary_enum`, `_splice_hierarchy_from_sidecar`, `_build_refined_hierarchy`, `_build_coarsened_hierarchy`, `_install_coordinate_array`, `_setup_symbolic_coordinates`); pure code motion. |
| READ-16 | High | M | D | `discretisation/discretisation_mesh.py:834` | The `mesh_update_callback` closure exists ×3 (834, 2027, 6086) and has silently diverged: only the `__init__` copy has both the teardown guard and the load-bearing `array is not mesh._coords` identity gate; `_re_extract_from_parent` drops the gate; `adapt` drops both. | One module-level `_mesh_coords_update_callback` (the `__init__` version, comments included) + a `Mesh._install_coords_array` helper; call from all three sites. |
| READ-17 | High | L | D | `systems/ddt.py:2286` | FILE GRADE C+. `SemiLagrangian.update_pre_solve` (2286–2801, 515 lines) interleaves ≥6 concerns; the RK2 midpoint characteristic trace is buried under unit bookkeeping. | Extract intention-named helpers (`_shift_history_with_blend`, `_record_current_field_into_history`, `_trace_departure_points`, `_sample_history_at_departure`, …) so the method reads shift → record → trace → sample. |
| READ-18 | High | M | D | `systems/ddt.py:2593` | The UnitAwareArray/`.magnitude`/non-dimensionalise unwrap block is copy-pasted ≥6× (2069, 2400, 2593, 2657, 2904, 2916) with slightly different branch ordering that is not intentional. | One `_to_nondim_ndarray(value, ...)` helper next to `_as_float`, docstring stating the ND-space invariant (issue #267); replace all six sites. |
| READ-19 | High | M | D | `systems/ddt.py:2630` | Node-velocity (2560–2613) and midpoint-velocity (2630–2675) blocks in the trace-back loop are near-identical (evaluate vs global_evaluate + coord array) — DRY violation on the hottest path; unit-handling fixes must be applied twice. | Extract `_velocity_nd_at(coords, use_global, subtract_v_mesh)`; combines with READ-18. |
| READ-20 | High | L | D | `systems/ddt.py:3453` | ~250 lines of verbatim per-class boilerplate quintuplicated across the five DDt flavors: model-registration try/except (incl. the identical 7-line "#195" comment ×5), coefficient init, `effective_order`, `bdf_coefficients`, `bdf()`, `adams_moulton_flux()`, state-setter validation. | Shared `_DDtBase` (or module helpers as step one); each flavor keeps only its storage. |

#### Medium severity

| ID | Sev | Eff | Wave | Location | Finding | Fix |
|----|-----|-----|------|----------|---------|-----|
| READ-21 | Med | S | A | `…snes_solvers.pyx:1383` | Four error paths `raise("message")` (1383, 1386, 1466, 2022) → runtime `TypeError: exceptions must derive from BaseException` instead of the diagnostic. | `raise ValueError(...)`/`TypeError(...)` keeping the message text (error-path only). |
| READ-22 | Med | S | D | `…snes_solvers.pyx:2278` | `SNES_Scalar.__init__` creates a default var then unconditionally `self.Unknowns.u = u_Field` (possibly None) — works only because the setter silently ignores None; SNES_Vector uses truthiness `if not u_Field:`; the default scalar var uses `num_components=mesh.dim`. | Explicit if/else in both constructors; `num_components=1` (or comment if load-bearing). |
| READ-23 | Med | M | A | `…snes_solvers.pyx:5620` | Stokes `strategy` setter: 'robust'/'fast' branches are `pass` "(Reserved)"; remainder re-duplicates the `__init__` option bundle with one silent divergence (`pc_mg_type` kaskade vs additive); `is_setup = False` commented out so post-setup calls silently no-op. | Delete empty branches (ValueError on unknown); extract shared option bundle; comment on the kaskade/additive difference — do not change values without benchmarking. |
| READ-24 | Med | S | A | `…snes_solvers.pyx:4026` | Fossilised commented-out blocks: 40-line dead rebuild path (4026–4066), whole `add_essential_p_bc` (5148–5168), dead flux-BC block (2766–2778), stale alternates (2684, 3624, 5119, 6355, 6470). | Delete (git keeps them); keep 2–3-line prose notes for documented pitfalls (e.g. the clearDS/createDS note). |
| READ-25 | Med | M | D | `…snes_solvers.pyx:4500` | Explicit-index Jacobian construction (G0–G3 flat-layout loops) duplicated between SNES_Vector (3684–3714, BC 3746–3790) and SNES_MultiComponent (4500–4531, BC 4556–4596); the valuable layout commentary exists only in the MultiComponent copy. | Extract `_petsc_pointwise_jacobians` / `_petsc_bd_jacobians` module helpers carrying the comment once. |
| READ-26 | Med | M | D | `…snes_solvers.pyx:3888` | Bd-residual/jacobian registration written as three parallel `_has_f1` if/else pairs, duplicated Vector (3888–3931) vs MultiComponent (4664–4710), ~120 lines varying only in pointer-vs-NULL slots. | Compute optional pointers once (small cdef helper) or extract one shared `_wire_bd_terms`. |
| READ-27 | Med | M | D | `…snes_solvers.pyx:7659` | `compute_volume_residual_fields` / `compute_boundary_residual_fields` duplicate ~70 lines each (preamble 7469–7498 vs 7597–7636; per-field IS copy-out 7534–7569 vs 7659–7694); `_assemble_volume_reaction` (2086–2163) is a third overlapping variant. | Extract `_gather_state_for_residual` and `_split_local_residual_by_field`; the public methods become ~20-line wrappers. |
| READ-28 | Med | S | D-doc | `…snes_solvers.pyx:4617` | `SNES_MultiComponent._setup_pointwise_functions` omits `self._current_jit_cache_key = …` (its three siblings set it at 2815/3826/6682), so `_build`'s constants-only fast path (1175) can never fire for MultiComponent; nothing says whether that is intentional. | Document-first: comment stating the fast path is disabled here and why; align with siblings in the benchmarked wave. |
| READ-29 | Med | S | D | `…snes_solvers.pyx:8025` | Stokes solve: picard and else branches end with a byte-identical 12-line tail (8026–8039 vs 8043–8056); the `# Now go back to the original plan` comment at 8025 sits at the wrong indent, misleading about control flow. | Dedent the common tail to run once after the optional Picard warmup (verbatim code motion). |
| READ-30 | Med | S | D-doc | `…snes_solvers.pyx:7973` | Magic `snes_max_it = 50` silently pushed to petsc_options each solve (8029/8046), clobbering user settings; neighbouring options are read from current state, this one hardcoded, uncommented. | Comment the intent now; reading the current option is a behavior change → queue for benchmarked wave. |
| READ-31 | Med | S | A | `…discretisation_mesh.py:4492` | Four dead methods, zero callers: `_build_kd_tree_index_DS` (4492), `_build_kd_tree_index_PIC` (4606), `get_min_radius_old` (5584), `_get_mesh_centroids` (5544, self-labelled deprecated). | Delete; note in historical-notes.md if archaeologically valuable. |
| READ-32 | Med | S | A | `…discretisation_mesh.py:5699` | `meshVariable_mask_from_label` is dead AND broken: bare `MeshVariable` (5703) never imported → NameError on any call; no callers repo-wide. | Delete (or fix + test if the capability is wanted). |
| READ-33 | Med | S | A | `…discretisation_mesh.py:4548` | ~30-line commented-out face-normal block (4548–4578) duplicating `_mark_faces_inside_and_out` logic, plus commented `build_index()` scratch at 4510/4592/4645/4652 — the dead block is longer than the live loop. | Delete; one line pointing at git history if needed. |
| READ-34 | Med | S | A | `…discretisation_mesh.py:1063` | Dead `if False:`/`elif False:` chain (1063–1078) for removed native coordinate-system calculus; commented scratch at 527–535, 695, 711. | Replace with the single live line + one note; delete scratch. |
| READ-35 | Med | S | D | `…discretisation_mesh.py:316` | Length-scale derivation duplicates an identical 12-line try/except for `domain_depth` and `length` (only the key differs), both with bare `except:` silently swallowing `to_base_units()` failures. | Collapse to a loop over the two keys; narrow the except; comment why fallback-not-raise. |
| READ-36 | Med | M | D | `…discretisation_mesh.py:1668` | Surviving-boundary-label enum construction duplicated between `extract_region` (1668–1689) and `extract_surface` (1836–1866) with **different safety idioms** — extract_region uses the direct `getStratumIS` pattern the newer copy's comment warns can hard-abort on submesh DMs. | Extract `_surviving_labels(...)` implementing the safe getValueIS-first idiom; use in both. |
| READ-37 | Med | S | D | `…discretisation_mesh.py:1888` | `extract_surface` inlines a KDTree vertex map (1893–1901) with a stale comment claiming `_build_vertex_map` is broken (issue #197) — the method directly below (1908–1938) was already fixed (071c5636) with the same code. | Delete the inline block + stale comment; call `_build_vertex_map()` as extract_region does (verify tuple ordering). |
| READ-38 | Med | M | D | `…discretisation_mesh.py:2052` | `_re_extract_from_parent` duplicates `adapt()`'s variable teardown/reinit/transfer machinery (2052–2107 vs 6107–6174) with small differences (IDW vs `uw.function.evaluate`) a reader cannot tell are intentional. | Extract shared teardown/reinit/invalidate helpers; keep the two transfer strategies as small named functions so the difference is visible. |
| READ-39 | Med | M | D | `…discretisation_mesh.py:4734` | Four-way facet outward-normal dispatch duplicated with near-identical code+comments in `_mark_faces_inside_and_out` (4734–4766) and `_mark_local_boundary_faces_inside_and_out` (5015–5047). | Extract `_facet_outward_unit_normal(...)` carrying the dimension-case comments once. |
| READ-40 | Med | M | D | `…discretisation_mesh.py:1445` | `view()` level 0 computes `gather_data` three times and never uses the result (collective calls, dead results; 1449 gathers a loop-leftover `i`); variable/boundary tables triplicated across view level 0/level 1/view_parallel. | Delete dead gathers; extract `_print_variable_table`/`_print_boundary_table`; `uw.pprint` at 1469. |
| READ-41 | Med | S | D | `…discretisation_mesh.py:537` | `all_edges_IS_dm` assigned only inside `if all_edges_label_dm:` yet referenced after (NameError if no 'depth' label); name says "edges" but `getStratumIS(0)` fetches depth-0 **vertices**. | Single guarded block; rename to `vertex_stratum_is`; one intent comment (Null_Boundary = every vertex, value 666). |
| READ-42 | Med | S | D-doc | `…discretisation_mesh.py:3247` | `_legacy_access` docstring example is UW2 copy-paste that cannot run (`FeMesh_Cartesian`, `with someMesh._deform_mesh():` as a context manager — which the file spends 100 lines saying you must not do). | Replace with a real UW3 snippet or delete the example. |
| READ-43 | Med | S | D-doc | `…discretisation_mesh.py:3689` | Deprecated `points` setter ends with `self._coords = model_coords` — rebinding to a plain ndarray, silently discarding the NDArray_With_Callback wrapper: no deform callback, PETSc coords never updated; contradicts its docstring. | `# TODO(BUG)` per project convention (working path is `self._coords[...] = …`); follow-up fixes or removes the setter. |
| READ-44 | Med | S | A | `systems/ddt.py:2750` | Dead moment-preservation: two `if 0 and self.preserve_moments …` blocks (2750–2785), yet the parameter is accepted, stored (1480), documented "experimental" (1390), and `self.I` (1774) exists solely to serve the dead blocks. | Delete the blocks + `self.I`; remove the param or make it raise NotImplementedError; fix docstring. |
| READ-45 | Med | S | D-doc | `systems/ddt.py:1933` | Comment contradicts code: state setter says SemiLagrangian "doesn't take a theta argument" and hardcodes `_update_am_values(..., 0.5)` — but `__init__` takes theta (1467, PR #187) and update_pre_solve uses `self.theta` (2351); restore on a theta=1.0 instance silently re-derives CN coefficients. | Delete stale comment; pass `self.theta` (flag the one-line value change to the maintainer — next update_pre_solve overwrites anyway). |
| READ-46 | Med | S | D | `systems/ddt.py:3167` | Broken display code ×2: `Lagrangian._object_viewer` (3167–3174) and `Lagrangian_Swarm._object_viewer` (3519–3526) reference `self.psi`/`self.dt_physical` which are never set → AttributeError on view; Eulerian carries the same lines commented out (1023–1030). | `self.psi_fn`, drop/guard the dt_physical line, delete Eulerian's dead copy (or reduce all three to the working line). |
| READ-47 | Med | S | D | `systems/ddt.py:1116` | Nested bare `except:` clauses as dispatch (copy → evaluate → projection) with no types or comment; a genuine evaluate() bug silently becomes an expensive projection solve. | Explicit `if self._psi_meshVar is not None:` first; narrow the remaining except; one-line intent comment. |
| READ-48 | Med | S | D | `systems/solvers.py:2264` | FILE GRADE B. `_maybe_install_auto_gauge` (2026-06-24) uses the banned `maybe_` prefix (rule enforced twice before). | Rename `_install_auto_gauge_if_eligible`; update call site (2230) and cross-references (2132, 2385). |
| READ-49 | Med | M | D | `systems/solvers.py:2760` | `smoothing_length` getter/setter + `smoothing` property triplicated verbatim across the three projection classes (2760–2853, 3018–3071, 3376–3423) incl. `_smoothing_is_dimensional` bookkeeping. | `_SmoothingLengthMixin` or module helpers; keep per-class docstrings as thin wrappers. |
| READ-50 | Med | M | D | `systems/solvers.py:4338` | 55-line `delta_t` setter duplicated verbatim in SNES_AdvectionDiffusion (3832–3884) and SNES_Diffusion (4338–4390); both open with bare `except: pass` (swallows KeyboardInterrupt). | Extract `_nondimensionalise_timestep(value)`; narrow the except to `(TypeError, ValueError)`. |
| READ-51 | Med | M | D | `systems/solvers.py:4414` | Diffusivity-max block in `estimate_dt` duplicated ×4 (TransientDarcy 736, AdvDiff 3926, Diffusion 4414, NavierStokes 4992); centroid-velocity block ×3 (Stokes 1885, AdvDiff 3963, NS 5029). | Extract `_global_max_diffusivity` and `_centroid_velocities_nd`; the `.magnitude` rationale lives once. |
| READ-52 | Med | M | D | `systems/solvers.py:100` | `_apply_unit_aware_scaling`: one try/except silently discards every exception (147–149, unused `e`); its two branches are redundant — both multiply by `fundamental_scales['time']`; the field-units inspection changes nothing. | Collapse to a single path; delete or justify the dead branch; specific exceptions; drop redundant `import sympy` (103). |
| READ-53 | Med | S | D | `systems/solvers.py:2985` | `SNES_Vector_Projection.projection_problem_description` adds smoothing+penalty to `self._f1` that the F1 Template (2963–2970) already contains — doubling the terms; unmarked deprecated while its siblings carry deprecation notes. | Delete if unused (grep first); else mark deprecated and assign `self._f1 = self.F1.sym` only; at minimum `# TODO(BUG)`. |
| READ-54 | Med | M | D | `systems/solvers.py:4119` | Transient solve() choreography repeated near-identically ×3 (TransientDarcy 799–839, AdvDiff 4101–4138, Diffusion 4490–4526); the 3-line cache-invalidation idiom verbatim ×3. | Extract `_invalidate_solution_cache`; consider a `_transient_solve_template` with hooks. |
| READ-55 | Med | S | D-doc | `utilities/rotated_bc.py:1` | FILE GRADE B. Module docstring opens "Development version of underworld3.utilities.rotated_bc — … Productizes the validated prototypes" — but this file IS that module on the integration branch; stale provenance framing misleads about status. | Rewrite the first two lines to describe what the module IS. |
| READ-56 | Med | S | D | `utilities/rotated_bc.py:317` | `_zero_rows_local` exists to encapsulate the ownership-relative zeroing (np>1 overflow subtlety) yet its exact body is hand-inlined ×3 more (244–246, 610–612, 652–655). | Call the helper at all three sites; move its definition above first textual use. |
| READ-57 | Med | S | D | `utilities/rotated_bc.py:736` | Boundary-spec normalization written two different ways: nested-comprehension dict one-liner (736–737) vs build_rotation's loop (148). | Extract `_boundary_spec(spec) -> (name, normal)`; use in both. |
| READ-58 | Med | M | D | `utilities/rotated_bc.py:479` | `solve_rotated_freeslip_nonlinear` is 177 lines (338–514); the backtracking line search + interleaved manual `destroy()` bookkeeping forces the reader to track 6 live Vecs/Mats across three exit paths. | Extract the line search into a helper owning its temporaries' destroys; group per-iteration destroys at one point. |
| READ-59 | Med | M | D | `utilities/custom_mg.py:557` | `maps` element type is mode-dependent (parallel: 4-tuples read via magic `lay[3]` at 559; serial: bare index arrays) — reader must trace two return shapes. | `LevelLayout` NamedTuple from `_level_dof_layout`/`_coarse_dof_layout`; `lay.n_full`; optionally split serial/parallel loop bodies into named helpers. |
| READ-60 | Med | S | A | `discretisation/remesh.py:190` | FILE GRADE B. `_remap_one_var` is dead: body is `raise NotImplementedError`, docstring admits "not currently used", no callers. | Delete; its pointer text already lives in the module docstring. |
| READ-61 | Med | S | A | `discretisation/remesh.py:200` | `_new_coord_cache` has zero callers and duplicates the new-DOF-coordinate capture inline in `_remap_var_set` (411–416) — dead code doubling as a which-copy-is-live trap. | Delete; if the inline block wants a name, extract it there as the single implementation. |
| READ-62 | Med | S | D | `discretisation/remesh.py:447` | `REMESH_MONOTONE` env knob read via `import os as _os` **inside the per-variable loop inside a try block** (447–450) — a module behaviour switch invisible from the module top, re-parsed per variable. | Hoist to a documented module-level constant with the falsy-string normalisation done once. |
| READ-63 | Med | M | D | `discretisation/remesh.py:421` | Guarded write-back `try: …[...] = X / except Exception: pass` ×3 (349–353, 422–426, 467–471) + the twin in `_snapshot_remap_data` (177–187); each silently drops a variable's transfer with no stated failure mode. | One `_write_var_data(var, values)` helper with the legitimate failure mode named once (unallocated/size-0 storage); preserve the swallow (behaviour unchanged). |
| READ-64 | Med | S | D-doc | `discretisation/remesh.py:272` | Re-entrancy comment says it "surfaces the outer scratch dict" but the code does nothing of the sort (the outer wrapper set `_remesh_pending_scratch` at 280); the nested branch returns True unconditionally, contradicting the docstring contract. | Rewrite the comment to state what is true; note the return value is meaningless for nested calls. Comment-only. |

#### Low severity

| ID | Sev | Eff | Wave | Location | Finding | Fix |
|----|-----|-----|------|----------|---------|-----|
| READ-65 | Low | S | D | `meshing/smoothing.py:507` | Dead params `relax=None, step_frac=None` ("kept only for signature stability"); `n_sweeps` misnames the nonlinear-CG iteration cap (741). | Drop the dead params; rename to `max_cg_iters` with one deprecation cycle for `n_sweeps`. |
| READ-66 | Low | S | D | `meshing/smoothing.py:1342` | `_zig` computed identically ×3 (1342, 1584, 1968) each with a 5-line re-explanation; `_wire` closure pair duplicated ×3 (1228, 1545, 1888). | `_warm_start_krylov(...)` + `_solver_wiring(...)` helpers; rename `_zig` → `zero_init_guess`. |
| READ-67 | Low | S | D | `meshing/smoothing.py:2305` | `move_anisotropy` radial/tangential reweighting duplicated verbatim (1387–1400 vs 2305–2318). | Extract `_reweight_displacement_radial_tangential(...)`. |
| READ-68 | Low | S | A | `meshing/smoothing.py:1277` | Dead locals: `_cdim` ×3 (1277, 1593, 2271); `Lbar`/`L0`/`L0_mean` triple-name one constant in `_winslow_spring` (606–608). | Delete `_cdim`s; use `Lbar` directly in the verbose diagnostic. |
| READ-69 | Low | S | D | `meshing/smoothing.py:4405` | Global mean-edge-length h0 implemented twice (2004–2014, 4405–4416), both with matching compounding-refinement commentary. | Extract `_mean_edge_length(dm, coords)`; keep the warning once at the cache declaration (73–83). |
| READ-70 | Low | S | A | `…snes_solvers.pyx:1` | Line 1 of the flagship solver file is `from xmlrpc.client import Boolean` (unused); `sympify`, `TypeAlias`, `class_or_instance_method` also unused. | Delete the four imports. |
| READ-71 | Low | S | D | `…snes_solvers.pyx:1691` | F1 guard property's RuntimeError says "F0 is being used" (copy-paste of the F0 property at 1687) — points a developer at the wrong term. | Fix the message to F1. |
| READ-72 | Low | S | D | `…snes_solvers.pyx:1022` | Two divergent SNES convergence-reason tables in one class (866–888: 11 codes; 1022–1035: 8 codes; −9/−10/−11 already missing from the compact map). | One class-level table `code -> (NAME, explanation)`; both consumers format from it. |
| READ-73 | Low | S | D | `…snes_solvers.pyx:2115` | Time non-dimensionalisation snippet copy-pasted ×6 (2115, 3005, 7471, 7611, 7929, 7962). | One `_nondimensional_time(time)` base-class helper. |
| READ-74 | Low | S | D-doc | `…snes_solvers.pyx:3130` | Unresolved editorial musings as comments (3130 "LM: probably not something we need", 5508 "Why is this here??", 6220 "uf0, uF1 are redundant", 5029). | Delete or convert to `# TODO(DESIGN):` per CLAUDE.md policy. |
| READ-75 | Low | M | D | `…snes_solvers.pyx:2317` | Verbose-monitor option toggle copy-pasted ×4 (2317, 3250, 4254, 5035); GAMG default bundle ×3 (2304, 3241, 4245; incl. a literally repeated line at 2315). | Extract `_set_monitor_options(verbose)` / `_set_gamg_defaults()`. |
| READ-76 | Low | S | D | `…snes_solvers.pyx:3614` | Local named `dim` assigned `mesh.cdim` (3614–3615, 4439) — embedding dim, not topological; plus a stale "~line 173" comment reference. | Rename to `gdim`/`cdim`; replace the line reference with the method name. |
| READ-77 | Low | S | A | `…discretisation_mesh.py:8` | Dead imports: `mpi4py.MPI.Info` (8), the baffling `blockmatrix.bc_dist` (11), unused `gather_data` (18); CoordSys3D imported at 28 and re-imported at 878. | Delete all four. |
| READ-78 | Low | S | D-doc | `…discretisation_mesh.py:5236` | Self-contradictory stale docstring: two consecutive sentences state opposite semantics for `tol > 0` (first should describe `on_boundary=True`). | Fix the first sentence. |
| READ-79 | Low | S | D | `…discretisation_mesh.py:4894` | `_test_if_points_in_cells_internal` repeats the control-point loop ×3 with only the final comparison differing. | Compute once; select the threshold; one comment on strict/non-strict. |
| READ-80 | Low | M | D | `…discretisation_mesh.py:1274` | `quality()` is single-letter soup (`a, b, cl_, A, q, et, jr, rel`) under a superb docstring. | Rename locals to what they are; one comment on the edge-to-tris build; formulas unchanged. |
| READ-81 | Low | S | D-doc | `…discretisation_mesh.py:2649` | Blanket `except Exception: pass` in the refresh paths (2649, 3133–3145, 6156) — a failed boundary-normal refresh silently leaves a Nitsche BC on stale geometry, the exact bug class these features prevent. | Comment the sanctioned reason at each swallow or route through debug-level pprint; narrow types in a follow-up. |
| READ-82 | Low | S | D | `…discretisation_mesh.py:4170` | `Dict` used in an annotation but never imported (latent-NameError smell; harmless at runtime). | `dict[str, …]` builtin generic or add to the typing import. |
| READ-83 | Low | S | D | `…discretisation_mesh.py:3855` | Inverted no-op guards: `if os.path.exists(...): pass else: raise` ×2. | `if not …: raise`, hoist `abs_dir` once. |
| READ-84 | Low | S | D | `…discretisation_mesh.py:193` | `_hierarchy_sidecar_name` keeps a `level` arg "for forward-compatibility"; every call site passes 0 and the design persists only the coarsest level. | Drop the param (or delete the forward-compat sentence and state "always 0 today"). |
| READ-85 | Low | S | A | `systems/ddt.py:2508` | Dead diagnostic: `coords_template`/`has_units` (2508–2509, "# DIAGNOSTIC") computed, never read — leftover from the #267 rework. | Delete 2507–2509. |
| READ-86 | Low | S | A | `systems/ddt.py:1690` | `self._nswarm_psi = None` kept alive with an eight-line apology that itself says nothing reads it. | Delete; one-line note or rely on git history. |
| READ-87 | Low | S | D | `systems/ddt.py:1660` | Work variable named `f"W_{instance}_{i}"` where `i` is a **leaked loop index** from a loop ending at 1614 — the suffix looks meaningful but is accidental; `W` says nothing. | Rename e.g. `psi_work_sl_{instance}`; note why it exists (different degree/continuity than psi_star). |
| READ-88 | Low | S | A | `systems/ddt.py:3022` | Dead class attr `instances = 0` ("to create unique … ids") in Lagrangian (3022) and Lagrangian_Swarm (3404) — never incremented or read; ids come from `uw_object.instance_number`; the comment misleads. | Delete both. |
| READ-89 | Low | S | D-doc | `systems/ddt.py:2488` | Comments cite drifted hard-coded line numbers ("~line 1540" → actually ~2092; "lines 703-709" → ~2419–2424). | Method-name references instead of line numbers. |
| READ-90 | Low | S | D | `systems/ddt.py:2173` | Redundant inline imports of module-scope names: sympy ×4, numpy, RemeshPolicy ×4, UnitAwareArray ×5 — falsely suggesting circularity. | Hoist to module level (both are leaf modules); comment any import that genuinely must stay local. |
| READ-91 | Low | S | D-doc | `systems/ddt.py:534` | `Symbolic` accepts/stores/documents `bcs`, `smoothing` (and threads ignored `evalf`/`verbose`) that only make sense for projection-backed flavors. | Remove, or state "accepted for interface parity; unused here" in one comment. |
| READ-92 | Low | S | D | `systems/ddt.py:1801` | `try: register_remesh_hook … except Exception: pass` ×2 with no stated tolerated failure — reads as fear, contrasting with the well-documented guard just above. | Narrow to `except AttributeError` + one-line comment ("older Mesh without the hook registry"). |
| READ-93 | Low | S | A | `systems/solvers.py:4273` | Five stale commented-out blocks (4273–4289 "??? unable to solve after n timesteps", 556–561, 1849–1856, 3200–3203, 549–550). | Delete; replace 4273-block with a one-line known-limitation note if worth keeping. |
| READ-94 | Low | S | D-doc | `systems/solvers.py:3887` | `estimate_dt` takes `percentile: float = 0.0` but the docstring documents only `direction_aware`; the semantics live in an inline comment invisible to help()/Sphinx. | Lift the inline explanation into the Parameters section. |
| READ-95 | Low | S | A | `systems/solvers.py:4043` | `np.maximum(h_per_element, 0.0)` is a no-op (h = max−min ≥ 0 by construction) and its comment describes the *following* `np.where`. | Delete the line; move/reword the comment onto the np.where. |
| READ-96 | Low | S | D | `systems/solvers.py:76` | Module-level `expression = lambda …` shadows a common name, hides why `_unique_name_generation=True` is needed, and gives useless tracebacks. | `def expression(...)` with a one-line docstring citing SYMBOL_DISAMBIGUATION_2025-12.md. |
| READ-97 | Low | S | D | `systems/solvers.py:1313` | `_prev_effective_order` lazily created via hasattr-guard inside solve() instead of `__init__` — solver state invisible from `__init__`. | Initialise to None in `__init__` with a one-line comment; drop the guard. |
| READ-98 | Low | S | D | `systems/solvers.py:284` | Six shadow imports of module-scope names (`uw` ×2, `sympy` ×3, `np` ×1) inside methods. | Delete (disappears with READ-49 for three of them). |
| READ-99 | Low | S | D | `systems/solvers.py:335` | `CM_is_setup` abbreviated, defined twice identically (335, 1485), and raises AttributeError (rather than False) before a constitutive model is assigned. | Define once on the base as `constitutive_model_is_setup` (+ alias); document/handle the not-yet-assigned case. |
| READ-100 | Low | S | D | `utilities/rotated_bc.py:33` | `_velocity_field_id(solver)` ignores its argument and returns 0, while the pressure id is a bare inline `PRE = 1` (230) — inconsistent ceremony for two constants. | Module constants `_VELOCITY_FIELD = 0`, `_PRESSURE_FIELD = 1` with one comment; use at all call sites. |
| READ-101 | Low | S | D | `utilities/rotated_bc.py:229` | Pressure-datum search runs unconditionally but `pin` is consumed only in the opt-in LU branch (252–258); its parallel-unsafety is noted in prose, not `TODO(BUG)`. | Move the search inside the LU branch (pure motion); convert the note to `# TODO(BUG):`. |
| READ-102 | Low | S | D | `utilities/rotated_bc.py:89` | `try: import sympy … except Exception: sym_fn = None` guards an impossible state (sympy is a hard dep) and would silently misinterpret a lambdify error as a constant-array normal. | Import sympy plainly; drop the try/except. |
| READ-103 | Low | S | D-doc | `utilities/rotated_bc.py:692` | `info` parameter is the untyped result dict whose keys differ between linear and nonlinear paths; the contract is learnable only by diffing two return statements (272–274 vs 510–514). | Rename to `solve_result`; add Returns key listings to both solve functions' docstrings. |
| READ-104 | Low | M | D | `utilities/rotated_bc.py:436` | Pervasive semicolon-chained lines compress create/use/destroy of PETSc objects (215–220, 230, 259–261, 386, 422, 433, 436, 450, 461–462, 473, 486–488, 582), obscuring the lifecycle the file's own comments call trickiest. | Mechanically split to one statement per line; keep lifecycle comments attached. |
| READ-105 | Low | S | D | `utilities/rotated_bc.py:444` | `from underworld3 import mpi` repeated inline ×4 (444, 457, 501, 614) just for `mpi.pprint` — reader cannot tell if load-bearing. | Import once at module top (uw.mpi is a leaf) or one commented inline import per function. |
| READ-106 | Low | S | D | `utilities/custom_mg.py:268` | FILE GRADE A-. 4-line DM-prep block (clone/copyFields/copyDS/createDS) duplicated between `_coarse_reduced_map` (268–271) and `_coarse_dof_layout` (318–321). | Extract `_clone_dm_with_solver_discretisation(...)`; move the copyDS 'trick' explanation onto it. |
| READ-107 | Low | S | D | `utilities/custom_mg.py:586` | Zero-column singular-coarse-operator guard exists twice (inline scipy 586–591; `_assert_no_zero_columns_parallel` 362–375); the physics rationale documented only on one. | Extract `_assert_no_zero_columns_serial` next to the parallel one; share the wording. |
| READ-108 | Low | S | D | `utilities/custom_mg.py:228` | Bare `except Exception: return dm` around `getNumFields()` with no stated raising state — silently falling back to the whole DM would mask a multi-field mistake. | Narrow to `PETSc.Error` + one-line comment, or drop if it cannot fail on a built DM. |
| READ-109 | Low | S | D | `utilities/custom_mg.py:325` | `_build_parallel_transfer` uses `cc, fc, lay_c, lay_f` while the serial `_reduced_transfer` (275) spells out full names — reader re-derives the convention. | Rename to match the serial vocabulary (call site at 578 updates in the same commit). |
| READ-110 | Low | S | D-doc | `utilities/custom_mg.py:87` | `r = np.where(r == 0.0, 1e-30, r)` — uncommented magic clamp; the intent (r²log r → 0; clamp only keeps log finite) unstated. | One-line comment. |
| READ-111 | Low | S | D | `utilities/custom_mg.py:66` | Semicolon-packed multi-statement lines (66, 73, 196, 366, 569). | Split; pure formatting. |
| READ-112 | Low | S | D-doc | `utilities/custom_mg.py:639` | Legacy finest-only path in `inject_custom_mg` is live (set_custom_mg, test_1015) but duplicates degree/continuity extraction from `build` and carries no lifespan marker. | `# TODO(deprecate): remove with SolverBaseClass.set_custom_mg …` + state the one case where legacy differs. |
| READ-113 | Low | S | D | `utilities/custom_mg.py:489` | `sub` holds an options-prefix string but collides with `sub` = sub-DM elsewhere in the file. | Rename `vel_prefix`; move inside the `if verbose:` block. |
| READ-114 | Low | S | D | `discretisation/remesh.py:173` | `_snapshot_remap_data` is also called on the managed-vars bucket (304) — the name lies about half its uses. | Rename `_snapshot_var_data`; two call sites. |
| READ-115 | Low | S | A | `discretisation/remesh.py:168` | Unreachable `else: buckets["remap"].append(var)` (policy already normalised; all members matched) and an impossible `if var is None:` guard; the deliberate ValueError→REMAP fail-safe (159–161) doesn't say so. | Delete both; half-line comment pointing at the fail-safe rationale in the RemeshPolicy docstring. |
| READ-116 | Low | S | D | `discretisation/remesh.py:474` | `remap_var_set = _remap_var_set` alias 90 lines after the private definition; grep for the public name lands on an assignment (external caller: ddt.py:1840). | Rename the function itself to `remap_var_set`; delete the alias. |
| READ-117 | Low | S | D | `discretisation/remesh.py:380` | Redundant `if live else []` tail on a comprehension; `RemeshContext.scratch` docstring says "keys are by convention" without listing the exactly-two conventions. | Drop the tail; enumerate `ale_opt_out` and `v_mesh` with producer/consumer. |
| READ-118 | Low | S | D-doc | `discretisation/remesh.py:329` | Operator `on_remesh` hook exceptions swallowed, reported only under `verbose` — a failing ALE history update vanishes silently, with no stated rationale (unlike the file's other guards). | Comment the intended contract (best-effort hooks; REMAP already secured the fields) — or Wave-D decision to warn unconditionally. |

---

## Testing Instructions

How to validate the eventual fixes, per the campaign ground rules
(`tier_a` green pre/post each wave; `tier_a or tier_b` before merge; np2/np4 for
anything touching partition-sensitive code).

1. **Baseline before any wave**: from the remediation worktree, `./uw build` then
   `pytest -m "level_1 and tier_a"`; capture the pass list as the reference.
2. **Pure code-motion / dedent / rename claims** (READ-02, -05, -13, -14, -16, -29,
   and most Wave-D items): assert no behavioural diff mechanically —
   - For `.py` files: `python -m py_compile` + run the owning test files
     (smoothing/mover: `test_0750, test_0760, test_0762, test_0763, test_0855`).
   - For the `.pyx`: rebuild with `rm -rf build/` first (stale `build/src/*.c` is a
     known trap — confirm the change landed via `strings <.so> | grep <marker>`),
     then run `test_1010, test_1015, test_1016, test_1018, test_1053, test_1064`.
3. **READ-01 (`_signed_volumes`)**: whichever option is chosen, add a level_1 test
   that calls `smooth_mesh_interior(method="mmpde")` on a small 3D mesh and asserts
   either successful movement or a clean `NotImplementedError` — today it must
   raise `NameError`, which the new test should first reproduce (test-before-fix per
   project policy).
4. **READ-02**: after deleting the inline block, run the anisotropic mover on a
   deterministic metric with `metric_refresh_per_iter` both True and False and
   diff final coordinates against pre-change output (expect bit-identical for the
   refresh path; identical-to-tolerance for the once-before-loop path since the
   closure re-runs a deterministic `gproj.solve()`).
5. **READ-03 and all `.pyx` structural extractions**: benchmarked wave only. Beyond
   tests, verify BC-label behaviour explicitly: solve one Scalar/Poisson and one
   Vector problem with a Dirichlet BC pre- and post-change and compare solutions
   bit-for-bit; run np2 and np4. The 2-vs-2 label divergence must survive
   extraction unchanged unless a separate decision unifies it.
6. **Mesh-mover findings (READ-04..-12, -65..-69)**: run at np1/np2/np4 — many
   duplicated blocks are MPI-reduction code where a wrong extraction deadlocks
   (rank-local early return before a collective). Any hang = failure.
7. **Package split (READ-04)**: after the move, verify every cross-module private
   import still resolves: `python -c "from underworld3.meshing.smoothing import
   _edge_pairs, _tri_cells, _tet_cells, _pinned_mask, _auto_pinned_labels,
   _owned_vertex_mask, _signed_areas"` and run the five mover test files plus
   `surfaces`/`_ot_adapt` consumers.
8. **Doc-only fixes** (D-doc rows): `pixi run docs-build` clean; no test impact
   expected, but run the owning test file once as a smoke check.
9. **Unverified findings**: each must pass the same adversarial verification as the
   verified table (re-read lines at the remediation base, confirm callers/
   reachability, confirm not already fixed) **before** its fix is applied.

---

## Known Limitations

- **104 of 118 findings are unverified.** Their anchors were sampled (all sampled
  anchors matched at `1d003481`) but the full adversarial pass — caller analysis,
  reachability, fixed-at-HEAD checks — was performed only for READ-01..READ-14.
  Treat the unverified tables as a triage queue, not a worklist.
- **Line numbers are pinned to `development@1d003481`.** Any commit to these files
  invalidates them; remediation PRs should re-verify against their own base.
- **READ-01 is a functional defect, not just readability** (3D MMPDE cannot run).
  It is reported here because the audit found it via a docstring contradiction, but
  it may deserve routing to the bugs track / external planning file.
- **READ-45 and READ-53 include one-line behavioural corrections** (restore-time
  theta; double-counted smoothing/penalty terms). Both are flagged for explicit
  maintainer sign-off rather than being folded silently into readability waves.
- `petsc_generic_snes_solvers.pyx` findings respect the campaign constraint
  (naming/docs/dead-code only); the structural extractions (READ-03, -25, -26, -27,
  -75) are queued for the separately benchmarked wave and are **not** Wave-D-safe.
- Two hotspot files (`smoothing.py`, the `.pyx`) were not assigned letter grades by
  the grading pass; their state is characterized narratively instead.
- The dimension briefing document was not available at its expected path; campaign
  context was taken from `docs/reviews/2026-07/README.md` (ground rules, waves,
  constraints) and the existing Dimension-3 review's section conventions.

---

## Appendix — Refuted claims (do not re-find)

No whole findings were refuted, but nine sub-claims were refuted or corrected during
adversarial verification. Recorded so later audits do not resurrect them:

1. **"mmpde is the production default"** — refuted. The API default is
   `method="spring"` (`smoothing.py:2623, 3525`); in-tree callers use `"ma"`/`"ot"`.
   mmpde is *recommended* by the file's own DeprecationWarning, not the default.
2. **"Six movers carry the `_winslow_` prefix"** — refuted; five do. The sixth
   (graph-Laplacian Jacobi) is inline in the dispatch and unprefixed.
3. **"~300 lines of literal duplicates in smoothing.py"** — overstated; ~130 lines
   of verbatim ≥8-line blocks (~260 counting both copies) plus near-duplication.
4. **"The `_build_M_tensor` closure exactly matches the inline block"** — refuted;
   the closure is a superset (additionally refreshes `Dcoords`/`gproj`, 2067–2078).
   The line-for-line match is the density/assembly portion only.
5. **"Renaming the `_winslow_*` movers touches only the dispatch plus scripts"** —
   refuted; docstrings in `remesh.py`/`_ot_adapt.py`/`smoothing.py:805`, 37 script
   files, and design docs also reference the names.
6. **"`**_ignored` in `_winslow_mmpde` is a pure hedge; removing it is safe"** —
   refuted; the strategy path legitimately forwards `resolution_ratio`
   (3565–3566 → 3643–3644). Bare removal breaks `strategy=` + mmpde.
7. **"The four solver classes' helpers are verbatim-shared modulo the class-name
   string"** — refuted; essential-BC label semantics diverge 2-vs-2
   (`str(boundary)` vs `"UW_Boundaries"`), and SaddlePt interleaves
   multiplier/fieldsplit logic. Extraction must preserve this deliberately.
8. **"The four-times-applied fix stanza in the .pyx came from PR #216
   (af537d56)"** — refuted attribution; that commit touched `swarm.py` /
   `_function.pyx`. The evidence is the identical stanza itself, whatever PR
   introduced it.
9. **"Re-exporting only the public names suffices for the smoothing package
   split"** — refuted; private helpers are imported by production modules and by
   tests 0750/0760/0762/0763/0855 and must keep resolving under
   `underworld3.meshing.smoothing.<name>`.

---

## Sign-Off
| Role | Name | Date | Status |
|------|------|------|--------|
| Maintainer | Louis Moresi | 2026-07-05 | Pending review |
| Author | Claude (audit session) | 2026-07-03 | Complete |



- Audit dimension: **4 — Readability of change hotspots**
- Base audited: `development` @ `1d003481`
- Findings: **118** total — 14 verified (3 high / 11 medium), 104 unverified
  (6 high / 44 medium / 54 low); 0 refuted findings, 9 refuted sub-claims recorded.
- All verified-table `file:line` anchors personally re-read in the audit worktree
  on 2026-07-03; unverified-table anchors sampled.
- Prepared for the July 2026 quality campaign, Phase 1.

*Underworld development team with AI support from [Claude Code](https://claude.com/claude-code)*