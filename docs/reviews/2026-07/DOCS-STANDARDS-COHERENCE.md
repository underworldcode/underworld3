# Docs & Standards Coherence Review — July 2026 Quality Campaign (Dimension 6)

**Status**: audit complete; findings adversarially verified 2026-07-03
**Base**: `development` @ `1d003481` (audit worktree, campaign index at `e848d131`)
**Scope**: coherence of the documentation system itself — which documents claim
authority, whether they agree with the code and with each other, the state of
the docstring backlog and its triage queue, the changelog as the quarterly
record, and hygiene of the design-doc directory.

Abbreviation used throughout: `pyx` = `src/underworld3/cython/petsc_generic_snes_solvers.pyx`.

## Overview

The June 2026 development burst (~95 first-parent commits May–June, +12.6k
lines to `src/`) moved the *code* forward much faster than the documents that
govern it. The result is not missing documentation so much as **contradictory
authority**: the document CLAUDE.md names as the "Authoritative Reference" for
data access (`UW3_Style_and_Patterns_Guide.md`) teaches a pattern the code
deprecates at runtime, prescribes a docstring format (markdown-for-pdoc) that
the settled NumPy/Sphinx standard replaced, and mandates a file format (Quarto
`.qmd`) of which zero instances exist in the repository. Meanwhile the
machinery meant to drive documentation remediation is itself stale: the
docstring review queue predates every June feature, and the changelog's
"Q2 (April – June)" section ends in April.

Seven findings were adversarially verified (DOC-01 … DOC-07); one
lower-severity finding was evidence-checked by the author — every cited line
was read directly in this worktree — but did not go through the adversarial
pass (DOC-08). All fixes are docs-only (plus two docstring edits); none
touches solver numerics or public API, consistent with the campaign ground
rules. Details refuted or corrected during verification are recorded in the
appendix so the same leads are not re-chased.

### Verified findings (adversarially checked)

| ID | Sev | Effort | File | Finding |
|----|-----|--------|------|---------|
| DOC-01 | high | M | `docs/developer/UW3_Style_and_Patterns_Guide.md` | The CLAUDE.md-declared authoritative style guide is stale on four normative topics: docstring format, doc file format, data-access examples, and Quarto front matter |
| DOC-02 | high | S | `docs/docstrings/review_queue.md` | Docstring triage queue is six months stale — flags now-complete items as missing and omits the entire June 2026 public API |
| DOC-03 | medium | M | `docs/developer/CHANGELOG.md` | Quarterly changelog (CIG/stakeholder record) has no entry after April 2026 despite ~95 May–June first-parent commits and a dozen headline PRs |
| DOC-04 | medium | S | `CLAUDE.md:324` | Authority pointer for data access names the stale guide; four overlapping documents cover one topic with no governing-doc map |
| DOC-05 | medium | M | `src/underworld3/swarm.py:4559` | The genuinely-undocumented public API (`uw.function.evaluate` params, `Swarm.advection`, the checkpoint trio) is not what the stale queue would have a docstring wave work on |
| DOC-06 | medium | S | `docs/developer/design/` | 30 tracked experiment artifacts (24 `_*.py`, 6 `_*.trace.txt`) plus 5 result PNGs sit beside ~36 genuine design docs |
| DOC-07 | medium | M | `docs/developer/design/jacobian-consistent-tangent.md` | Merged/parked/superseded design docs lack the status headers the directory's own convention establishes |

### Author-verified findings (not adversarially checked)

| ID | Sev | Effort | File | Finding |
|----|-----|--------|------|---------|
| DOC-08 | low | S | `docs/advanced/mesh-adaptation.md:247` | One deprecated `mesh.data` survives in user-facing docs, and the `mesh.adapt()` docstring's own Examples use `with mesh.access(...)` + `mesh.data` |

## Changes Made

None — audit only. Proposed fixes are recorded per finding below and are
scheduled for **Wave E (docs alignment)**, with DOC-05 feeding the docstring
wave and DOC-06 a candidate for **Wave A (deletions)**.

## System Architecture

What this dimension's audit established about how the documentation system is
put together, for the maintainer:

**Three layers claim normativity, and they disagree.** (1) `CLAUDE.md` is the
operational contract for AI-assisted sessions — its "Documentation Requests"
section correctly mandates MyST `.md`/Sphinx, and its Data Access quick table
correctly marks `mesh.data` and the access-context deprecated. (2) The
document CLAUDE.md itself points to as "Authoritative Reference"
(`UW3_Style_and_Patterns_Guide.md`, also routed to contributors via
`docs/developer/guides/contributing.md:57`) contradicts layer 1 on docstring
format, file format, and coordinate access. (3) The code is the ground truth:
`mesh.data` emits a `DeprecationWarning` directing to `mesh.X.coords`
(`discretisation_mesh.py:3563-3578`), and the solver docstrings that exist
follow NumPy/Sphinx style (e.g. `solve` at `pyx:2923`, `SNES_Scalar` at
`pyx:2197`). A reader who follows the declared chain of authority ends at the
one document that is wrong. Where a topic has a single dated design doc with a
supersession banner — units, via `UNITS_SIMPLIFIED_DESIGN_2025-11.md:3` ("This
document supersedes all previous units planning documents") — coherence is
fine; the failures cluster exactly where multiple undated documents share a
topic (data access has four: the Style Guide, `subsystems/data-access.md`
["Current Implementation (2025+)"], `UW3_Developers_NDArrays.md`, and
`design/ARCHITECTURE_ANALYSIS.md`).

**The remediation machinery is itself documentation, and it rotted the same
way.** The docstring backlog is managed through a generated queue
(`docs/docstrings/review_queue.md`, from `scripts/docstring_sweep.py`) last
committed 2026-01-13 (`cdf5bb21`); the quarterly changelog is hand-maintained
and last touched 2026-05-04 (content ending April). Both are "pull" documents
with no per-release regeneration step, so a six-month development burst
invalidated them silently. The design directory has the opposite dynamic:
things get *added* (experiment scripts, solver traces, result PNGs beside the
write-up that cites them) and never re-filed once the investigation closes.
Some closed docs do carry status markers (`multilevel-nonlinear-stokes-strategy.md`
frontmatter `status: PARKED (2026-06-16)`; `EXPONENTIAL_VE_INTEGRATOR.md` has
a `**Status**` line) — roughly 20 of the 36 design docs — so the convention
exists and merely needs completing, not inventing.

**Consequence for the campaign**: Wave E should fix the *authority map* first
(DOC-04), then the documents it points at (DOC-01), then regenerate the queue
(DOC-02) before any docstring-wave slot is allocated (DOC-05) — otherwise the
wave will be triaged from January data and re-document what June already
documented.

## Findings in detail

### DOC-01 — Style Guide: the declared authority is stale on four normative topics

**Severity: high · Effort: M · Category: docs-standards**
**File**: `docs/developer/UW3_Style_and_Patterns_Guide.md`

The guide carries Quarto YAML front matter (L1–21: `format: html/pdf`,
`code-fold`, `theme: cosmo`) and is named "Authoritative Reference"
(`CLAUDE.md:324`) and the contributor style reference
(`docs/developer/guides/contributing.md:57`). Four of its normative sections
contradict the current standards:

1. **Docstring format** — "## Markdown Docstrings for pdoc/pdoc3" (L113, with
   a full worked markdown-headers example, and summary item 4 at L459)
   contradicts the settled NumPy/Sphinx RST standard: the conversion plan
   exists at `docs/plans/docstring-conversion-plan.md`, and shipped docstrings
   follow it (e.g. `solve` at `pyx:2923` — Parameters/Returns/Examples/Notes;
   `SNES_Scalar` at `pyx:2197`).
2. **Doc file format** — "**Format**: Quarto markdown (`.qmd`)" (L422–423) and
   the migration-table row "Plain markdown → Quarto markdown" (L470)
   contradict CLAUDE.md's Documentation Requests section (MyST `.md` for
   Sphinx). Zero `.qmd` files exist in the repository.
3. **Data access** — `mesh.data[0] = new_position` is presented under
   "# Preferred: Direct array access" (L205) and `mesh.data += displacement`
   appears in the delay-callback example (L270), while the code deprecates
   `mesh.data` at runtime (`discretisation_mesh.py:3563` property; warning at
   3576–3578, "use mesh.X.coords instead"). The "Coordinate System
   Transformations" migration block (L217–221) directs readers to *private*
   attributes `swarm._particle_coordinates` (`swarm.py:3039`) and
   `mesh._deform_mesh` (`discretisation_mesh.py:3148`) as the "NEW" pattern.
4. **Front matter** — the Quarto YAML header itself (L1–21) is dead weight in
   a Sphinx/MyST build.

**Scope guard**: `swarm.data` (L206, L271) is **not** stale — it remains the
current documented pattern for swarm variables per CLAUDE.md's module-boundary
table and must be kept. Likewise the `with mesh.access(var)` example at L251
is correctly filed under a deprecated-legacy heading — it is documentation
*of* the deprecation, not an instance of staleness.

**Proposed fix**: rewrite the four sections — (a) replace the pdoc section
with the NumPy/Sphinx RST standard (one worked example with `:math:` and
Parameters/Returns, referencing `docs/plans/docstring-conversion-plan.md`);
(b) replace the `.qmd` file-convention section and migration-table row with
MyST `.md`/Sphinx guidance matching CLAUDE.md; (c) fix the coordinate examples
to `mesh.X.coords` and delete the private-attribute migration advice; (d)
replace the Quarto front matter with a plain MyST title. Keep as-is (verified
still consistent with current code): Naming Conventions (~L55–60),
NDArray_With_Callback patterns, array-vs-data shapes (matches
`subsystems/data-access.md`), MPI patterns, callback patterns, testing
patterns, and test-file naming (L425–429).

### DOC-02 — Docstring review queue is six months stale in both directions

**Severity: high · Effort: S · Category: docstrings**
**File**: `docs/docstrings/review_queue.md` (last commit `cdf5bb21`, 2026-01-13)

The queue that would drive the docstring-backlog wave misrepresents the
codebase both ways:

- **False negatives (already fixed)**: the queue flags `❌ solve [method]
  L1264` and `❌ SNES_Scalar [class] L807` as needing overview/parameters;
  both now carry full NumPy docstrings (`pyx:2923` and `pyx:2197`
  respectively — read directly).
- **Missing the newest API entirely**: `grep -cE "add_nitsche_bc|
  add_rotated_freeslip_bc|boundary_flux|set_custom_fmg|consistent_jacobian"`
  over the queue returns **0**, while all five exist in current source
  (`pyx:3317`, `pyx:5170`, `pyx:2165`, `utilities/custom_mg.py:605`,
  `pyx:92/2165ff`).

A wave triaged from this file would spend slots re-fixing fixed items and skip
the least-documented, newest API.

**Proposed fix**: rerun `scripts/docstring_sweep.py` (exists, verified)
against `development@1d003481` to regenerate `review_queue.md` (and
`inventory.json`) **before** allocating any docstring-wave slots; add the
sweep to the release checklist so the queue cannot go stale unnoticed again.

### DOC-03 — Changelog stops in April; ~95 May–June commits unrecorded

**Severity: medium · Effort: M · Category: changelog**
**File**: `docs/developer/CHANGELOG.md:7`

The changelog declares itself the source for quarterly CIG/stakeholder
reporting (line 3). Its "2026 Q2 (April – June)" section contains exactly
three entries, all April (DDt.set_initial_history, Multi-Component Projection,
PETSc Jacobian layout fix); the file's last commit is `aed517f6` (committer
date 2026-05-04). Measured in this worktree: **95** first-parent commits with
commit dates 2026-05-01 … 2026-06-30 on `development@1d003481`, and zero of
the headline PRs appear anywhere in the file (grep for #216, #250, #251,
#258, #259, #264, #265, #266, #275, #276, #290, #293, #294, #297, #298 → 0
hits).

One scoping nuance (verified against `git log --first-parent`): several
headline features merged just *after* Q2 closed — #298 on 2026-07-03; #297,
#216, #258, #293, #294, #290 all on 2026-07-02 — while #275, #265 (06-24),
#266/#264 (06-22), #259 (06-20), #251 (06-19), #250 (06-18) are genuinely
June. The backfill should therefore read **"May – early July"**, either as an
extended Q2 section or a Q2 section plus the first Q3 entries.

**Proposed fix**: backfill at the changelog's existing conceptual granularity
(roughly 10–14 grouped entries; the PR descriptions already contain the
prose). Add a release-checklist item in
`docs/developer/guides/release-process.md` requiring a changelog sweep of
`git log --first-parent` since the last entry.

### DOC-04 — Authority map: repoint CLAUDE.md and adopt one-governing-doc-per-topic

**Severity: medium · Effort: S · Category: docs-authority**
**File**: `CLAUDE.md:324`

`CLAUDE.md:324` reads `**Authoritative Reference**:
\`docs/developer/UW3_Style_and_Patterns_Guide.md\`` under "## Data Access
Patterns" — yet CLAUDE.md's own quick table eight lines below correctly marks
`mesh.data` deprecated, and the guide it crowns teaches `mesh.data` as
"Preferred" (DOC-01). Four documents cover data access:
the Style Guide, `subsystems/data-access.md` (heading "## Current
Implementation (2025+)", accurate on the `array`-property /
NDArray_With_Callback model), `UW3_Developers_NDArrays.md` (internals), and
`design/ARCHITECTURE_ANALYSIS.md` (2025 analysis of the same machinery). By
contrast, units has a clean single authority
(`design/UNITS_SIMPLIFIED_DESIGN_2025-11.md:3`, explicit supersession banner)
— that is the pattern to replicate.

**Proposed fix**: adopt a one-governing-doc-per-topic map and repoint
`CLAUDE.md:324` accordingly — data access → `subsystems/data-access.md`
(governing), `UW3_Developers_NDArrays.md` demoted to "internals reference",
`ARCHITECTURE_ANALYSIS.md` given a "historical analysis (2025), see
data-access.md" banner; docstring style → the corrected Style Guide section
(with `docs/plans/docstring-conversion-plan.md` as implementation plan); doc
file format → CLAUDE.md Documentation Requests (Style Guide `.qmd` section
superseded); units → `UNITS_SIMPLIFIED_DESIGN_2025-11.md` (add subordination
notes to `ai-notes/units-system-guide.md` and
`ai-notes/MESHVARIABLE_UNITS_GUIDE.md`); branching →
`guides/branching-strategy.md` (no conflict found); testing tiers →
`TESTING-RELIABILITY-SYSTEM.md` with `ai-notes/TEST-CLASSIFICATION-2025-11-15.md`
marked as a dated snapshot. Record the table in `docs/developer/index.md` as
the master authority index.

### DOC-05 — The real docstring gaps are not where the stale queue points

**Severity: medium · Effort: M · Category: docstrings**
**File**: `src/underworld3/swarm.py:4559` (and companions below)

Verified undocumented public API, in priority order for the wave:

1. **`uw.function.evaluate`** (`function/functions_unit_system.py:789`,
   confirmed as the `uw.function` export) — 14 parameters; the docstring
   documents only `monotone` and defers the other ~13 to the *private*
   `_evaluate_impl`. Inline the full parameter documentation (or hoist
   `_evaluate_impl`'s docs into the public wrapper).
2. **`Swarm.advection`** (`swarm.py:4559`) and **`NodalPointSwarm.advection`**
   (`swarm.py:4961`) — no docstrings. NumPy docstrings should cover `V_fn`,
   `delta_t` units handling (`delta_t` is non-dimensionalised internally,
   `swarm.py:4573`), `order`, `corrector`, and `step_limit` — noting the
   defaults differ (`Swarm.advection` `step_limit=False` at 4567;
   `NodalPointSwarm.advection` `step_limit=True` at 4969).
3. **The checkpoint trio** — `Swarm.read_timestep` (`swarm.py:3879`,
   signature `base_filename/swarm_id`), `SwarmVariable.read_timestep`
   (`swarm.py:1971`, signature `data_filename/swarmID/data_name`), and
   `SwarmVariable.write_proxy` (`swarm.py:1960`) — all undocstringed; the two
   `read_timestep` signatures genuinely disagree and each constructs
   undocumented filename conventions that only the source reveals.

**Do not** spend slots on items the January queue flags that are already
complete: `solve`, `add_essential/natural/dirichlet_bc`, `add_nitsche_bc`,
`add_rotated_freeslip_bc`, `estimate_dt`, `boundary_flux` (all in `pyx`) and
`set_custom_fmg` (`utilities/custom_mg.py:605`) all have docstrings.

### DOC-06 — 35 experiment artifacts tracked inside the design-doc directory

**Severity: medium · Effort: S · Category: docs-hygiene**
**File**: `docs/developer/design/` (e.g. `_exp_integrator_phase_a.py`)

Counted directly in the worktree: **24** underscore-prefixed `.py` experiment
scripts (8 matching `_exp_integrator_*` plus `_exp_jury_rig_minimal.py` and
15 others incl. `_repro_dminterp_multifield_bug.py`), **6** `_*.trace.txt` raw
solver logs, and **5** `exp_integrator_*.png` result images — 35 tracked
artifacts beside the ~36 genuine design `.md` docs.
`EXPONENTIAL_VE_INTEGRATOR.md` cites the scripts and PNGs at ~10 points
(verified at lines 50, 89, 218, 279, 375 and further), so they are provenance
and cannot be blindly deleted; grep across `src/`, `tests/`, and all docs
finds **no other consumer** of any artifact, and nothing at all references the
6 trace files.

**Proposed fix**: `git mv` the 24 scripts and 5 PNGs to
`docs/developer/design/experiments/exp-integrator/` (one batch;
`_exp_jury_rig_minimal.py` moves with them since `EXPONENTIAL_VE_INTEGRATOR.md`
cites it) and update the ~10 relative references; delete the 6 `.trace.txt`
logs outright (reproducible from the scripts, preserved in git history);
relocate `_repro_dminterp_multifield_bug.py` next to the test suite or attach
it to the corresponding issue — it is a bug reproduction, not a design
artifact. Candidate for Wave A.

### DOC-07 — Design docs missing status headers the directory convention expects

**Severity: medium · Effort: M · Category: docs-hygiene**
**File**: `docs/developer/design/jacobian-consistent-tangent.md:1`

The directory holds 36 `.md` docs; roughly 20 carry a status-style marker (the
convention is real: `multilevel-nonlinear-stokes-strategy.md` has frontmatter
`status: PARKED (2026-06-16)`; `EXPONENTIAL_VE_INTEGRATOR.md` a `**Status**`
line). Verified missing and misleading without one:

- `jacobian-consistent-tangent.md` — no status header, never mentions that it
  merged as PR #258 (`c63cd707`, opt-in `solver.consistent_jacobian`).
- `snesfas-feasibility.md` and `snesfas-vanka-feasibility-study.md` — open
  with GO verdicts, preserved via PR #245, but contain zero reference to
  PR #290, whose custom-prolongation FMG is the shipped geometric-MG path.
- `MATHEMATICAL_MIXIN_DESIGN.md:4` — "**Status**: Design Phase" (2025-10-26)
  although `utilities/mathematical_mixin.py` ships and CLAUDE.md cites the doc
  as its internals reference.
- No doc-level status: `COORDINATE_MIGRATION_GUIDE.md`,
  `ARCHITECTURE_ANALYSIS.md`, `fmg-checkpoint-hierarchy`,
  `fault-refinement-simplification`, `mesh-adaptation-formulation`,
  `petsc-dmplex-checkpoint-reload-plan`, `ND_UNITS_BOUNDARY_CONTRACT`.

**Proposed fix**: add a one-to-three-line `**Status:**` header per doc
following the existing convention — e.g. jacobian doc → "Implemented — PR #258
(opt-in `solver.consistent_jacobian`, default off)"; snesfas docs →
"Investigation record (preserved via PR #245); production geometric-MG path is
custom prolongation, PR #290"; MATHEMATICAL_MIXIN → "Implemented
(`utilities/mathematical_mixin.py`)"; COORDINATE_MIGRATION_GUIDE and
ARCHITECTURE_ANALYSIS → historical/2025 snapshots. Requires a short
verification pass per doc against git history before stamping.

### DOC-08 — Last deprecated coordinate patterns in user-facing docs (author-verified)

**Severity: low · Effort: S · Category: docs-standards**
**Files**: `docs/advanced/mesh-adaptation.md:247`,
`src/underworld3/discretisation/discretisation_mesh.py:5940-5942`

Read directly in this worktree: the mesh-adaptation guide's complete worked
example prints `mesh.data.shape[0]` (line 247) — `mesh.data` raises a
`DeprecationWarning`; current API is `mesh.X.coords`. Worse, the
`mesh.adapt()` docstring's own Examples section (`discretisation_mesh.py`,
`with mesh.access(metric):` at 5940 and `fault.distance_from(mesh.data)` at
5942) demonstrates both deprecated patterns — the API teaching what its own
deprecation warnings forbid. Beginner tutorials spot-checked clean: no
`mesh.access`/`swarm.access` in any of the 18 notebooks (see appendix).

**Proposed fix**: replace `mesh.data.shape[0]` with `mesh.X.coords.shape[0]`
in `mesh-adaptation.md:247`; rewrite the adapt-docstring example to direct
`metric.data[:, 0] = ...` plus `mesh.X.coords` per the current pattern table.
(The docstring edit is a comment-only change to `discretisation_mesh.py`, not
`pyx`; no numerics.)

## Testing Instructions

Wave E fixes are docs and docstrings only; validation is mechanical:

1. **Docs build** — `pixi run docs-build` must complete without new warnings
   after every batch (Style Guide rewrite, banner additions, artifact moves).
   Broken relative links from the DOC-06 `git mv` batch will surface here;
   additionally `grep -rn "_exp_integrator\|_exp_jury_rig\|.trace.txt"
   docs/` must show only paths under `design/experiments/exp-integrator/`.
2. **Pattern regression** — run `/check-patterns` (deprecated-pattern scanner)
   over `docs/` after DOC-01/DOC-08 land: zero hits for `mesh.data` as a
   coordinate accessor and for `with mesh.access(` outside explicitly-labelled
   legacy/deprecation sections. Confirm no over-correction: `swarm.data` usages
   must remain.
3. **Docstring queue regeneration (DOC-02)** — run
   `scripts/docstring_sweep.py`; assert the regenerated queue (a) no longer
   flags `solve`/`SNES_Scalar` in `pyx`, and (b) contains entries for
   `Swarm.advection`, `read_timestep`, `write_proxy`, and
   `uw.function.evaluate`. That double-check validates DOC-05's target list.
4. **Docstring wave (DOC-05, DOC-08)** — docstrings render via the docs build;
   spot-check `help(uw.function.evaluate)` and `help(uw.swarm.Swarm.advection)`
   in `pixi run -e default python`. Run `pytest -m "level_1 and tier_a"`
   before/after as the campaign gate (changes are comment-only; this guards
   against accidental code edits, especially for the `discretisation_mesh.py`
   and any `pyx` docstring touches, which require a rebuild via `./uw build`).
5. **Changelog (DOC-03)** — review-only; verify each backfilled entry cites a
   real merged PR (`git log --first-parent --oneline` since `aed517f6`).
6. **Authority map (DOC-04)** — after repointing, grep for the old claim:
   `grep -n "Authoritative Reference" CLAUDE.md` should point at
   `subsystems/data-access.md`; confirm `docs/developer/index.md` carries the
   authority table and builds into the toctree.

## Known Limitations

- **Line numbers are exact at `1d003481`/`e848d131` only** and will drift as
  waves land; findings identify sections by heading/content as well for that
  reason.
- **DOC-08 was not adversarially verified** (author-read only). Its evidence
  is small and was re-read directly for this document, but it kept its
  original tier per campaign rules.
- **Coverage**: this audit checked the documents that claim authority, the
  remediation machinery, the design directory, and spot-checked the 18
  beginner tutorial notebooks for deprecated patterns. It did **not**
  line-audit `docs/advanced/` beyond mesh-adaptation.md, the ~695 example
  `.py` files under `docs/`, or notebook prose accuracy — the API-consistency
  review (dimension 3) covers the BC-argument-order sweep of those files.
- **DOC-07 requires per-doc git-history verification before stamping** status
  headers; the suggested wordings above are grounded in the PRs cited but each
  stamp should be confirmed at fix time.
- **Commit-count convention**: the "95 commits" figure counts first-parent
  commits by commit date 2026-05-01…2026-06-30 at `1d003481` (measured both
  via explicit `--since/--until` datetime bounds and via `%cs` date strings —
  both give 95). A bare `--until=2026-07-01` gives 93 (boundary/timezone
  artifact); including early-July merges gives ~105. Use 95.

## Sign-Off

- **Auditor**: Claude (docs dimension subagent), 2026-07-03. All `file:line`
  evidence above read directly in the audit worktree at `e848d131` (content
  identical to `development@1d003481` for every cited file).
- **Adversarial verification**: DOC-01 … DOC-07 passed an independent
  adversarial pass; corrections from that pass are incorporated and logged in
  the appendix. DOC-08 author-verified only.
- **Maintainer sign-off (L. Moresi)**: ☐ pending — required before Wave E
  scheduling; DOC-06 deletions additionally require the campaign's
  per-batch deletion sign-off (Wave A rules).

---

## Appendix — Refuted and corrected claims (do not re-find)

No finding was refuted outright, but the following details from earlier drafts
were corrected during verification, and several suspects were checked clean.
Recording them here so later waves don't rediscover the same false leads:

1. **"Only 54 commits May–June"** — wrong; measured 95 first-parent commits
   (commit dates 2026-05-01…2026-06-30) at `1d003481`. The changelog gap is
   *larger* than first claimed (DOC-03).
2. **"Headline June features"** — several actually merged 2026-07-02/03
   (#298, #297, #216, #258, #290, plus unlisted #293, #294); genuinely-June
   are #275, #265, #266, #264, #259, #251, #250. Backfill scope is
   "May – early July", not "May–June" (DOC-03).
3. **"23 of 38 design docs have status markers"** — not reproducible; the
   directory holds 36 `.md` docs, ~20 with a status-style marker. The
   convention-completion argument stands; the precise count does not (DOC-07).
4. **"9 `_exp_integrator_*.py` files"** — actually 8, plus
   `_exp_jury_rig_minimal.py` which is cited from `EXPONENTIAL_VE_INTEGRATOR.md`
   and must move with the batch. Totals (24 scripts / 6 traces / 5 PNGs / 30
   underscore-tracked artifacts) are exact (DOC-06).
5. **Style Guide line drift** — the pdoc heading is at L113 (not 111);
   `mesh.data += displacement` at L270 (L271 is `swarm.data`); the `mesh.data`
   deprecation property is at `discretisation_mesh.py:3563` with the
   `warnings.warn` call at 3576–3578; `solve` is defined at `pyx:2923`
   (DOC-01, DOC-02).
6. **Checked clean — not staleness**: the Style Guide's `with mesh.access`
   example (L251) is correctly filed under a deprecated-legacy heading;
   `swarm.data` (L206/L271) is the *current* swarm pattern and must not be
   "fixed" (DOC-01).
7. **Checked clean — beginner tutorials**: no `mesh.access`/`swarm.access` in
   any of the 18 notebooks; `mesh.points` grep hits there are PyVista
   `pvmesh.points`, not UW3; `uw.non_dimensionalise` in tutorials 12/15 is the
   current public `units.py` function, not the deprecated `model.py` methods
   (DOC-08 scope check).
8. **Checked clean — already documented**: `solve`, the BC family
   (`add_essential/natural/dirichlet_bc`, `add_nitsche_bc`,
   `add_rotated_freeslip_bc`), `estimate_dt`, `boundary_flux` (all `pyx`) and
   `set_custom_fmg` (`utilities/custom_mg.py:605`) have docstrings — despite
   what the January queue says (DOC-02, DOC-05).
9. **`evaluate` parameter count** — 14 parameters total, so the public wrapper
   defers ~13 (not ~14) to `_evaluate_impl` (DOC-05).

*Underworld development team with AI support from Claude Code.*
