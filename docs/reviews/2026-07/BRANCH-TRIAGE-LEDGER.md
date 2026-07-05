# Branch Triage Ledger — Quality Audit 2026-07

## Overview

This ledger triages every local branch and worktree of the underworld3 repository as part of
the July 2026 quality audit. The June 2026 development surge left ~40 investigated branches
plus ~40 merged-and-clean worktrees behind. Each investigated branch was checked against
`development` (audit baseline `1d003481`; development tip at verification time `9bd6c8ee`)
for: unmerged commits (`git merge-base --is-ancestor`, `git cherry`, content diffs against
squash merges), unpushed commits (local vs `origin`), and uncommitted work in the attached
worktree (`git status --porcelain`, cross-branch grep for the dirty content).

Verdict taxonomy:

- **LAND** — complete, validated, unmerged work that should be pushed and PR'd to development.
- **EXTRACT** — the branch as a whole is dead, but specific commits, uncommitted files, or
  documents exist nowhere else and must be rescued before deletion.
- **KEEP_ACTIVE** — in-flight work; do not touch.
- **ARCHIVE_DELETE** — fully superseded by merged PRs; tag `archive/<branch>` then delete
  branch (and worktree, if any).
- **REMOVE_WORKTREE_ONLY** — branch fully merged and worktree clean; remove the worktree and
  delete the branch after the archive tag.

Verdict counts (investigated branches): **LAND 4, EXTRACT 11, KEEP_ACTIVE 2,
ARCHIVE_DELETE 15, REMOVE_WORKTREE_ONLY 8**, plus a bulk list of 40 merged+clean
REMOVE_WORKTREE_ONLY candidates. **14 rows carry risk_of_loss = HIGH** — every one of them
holds work that exists in exactly one place (unpushed commits or uncommitted worktree files).

## Changes Made

None — audit only; proposed changes are listed as findings (verdicts and the execution
protocol below). No branch, worktree, or file outside `docs/reviews/2026-07/` was modified.

## System Architecture

The repository uses a hub of worktrees under `.claude/worktrees/`, one per work stream, each
with its own pixi environment. `development` is the integration trunk; `main` is the release
branch. Session isolation means abandoned worktrees accumulate: some are pure leftovers of
squash-merged PRs, some hide the only copy of uncommitted work. Three worktrees are
**misnamed relative to the branch they hold** (a hazard for any cleanup script that assumes
worktree name == branch name):

| Worktree directory | Actual branch checked out |
|---|---|
| `.claude/worktrees/custom-mg-prolongation` | `feature/rotated-freeslip-bc` |
| `.claude/worktrees/vep-loading-unloading` | `feature/exp-integrator-investigation` |
| `.claude/worktrees/integrate-surface-submesh` | `feature/fs-surface-smoother-driver` |
| `.claude/worktrees/in-memory-checkpoint` | `docs/snapshot-toolkit-changelog` |
| `.claude/worktrees/winslow-mesh-smoother` | `feature/anisotropic-metric-mover` |
| `.claude/worktrees/snesfas-spike` | `docs/snesfas-investigation` |

All cleanup actions must therefore key on the **branch**, verified with `git worktree list`,
never on the directory name.

---

## HIGH-RISK ROWS — read these first

These 14 rows hold work that exists in **exactly one place**. Deleting the branch or worktree
without the listed rescue step loses it permanently.

| # | Branch | Verdict | What is at risk |
|---|--------|---------|-----------------|
| 1 | `feature/adapt-on-top` | LAND | Entire Layer-2 NVB/SBR adapt-on-top engine — local-only, **no origin remote** |
| 2 | `bugfix/custom-mg-parallel` | KEEP_ACTIVE | Continuation of the above + today's np>1 cross-partition transfer, **unpushed**, dirty design docs |
| 3 | `bugfix/yield-homotopy` | LAND | The planned PR-2 after #258 (yield homotopy + FMG monitor-leak fix) — **local-only, unpushed** |
| 4 | `worktree-product-system` | LAND | Canonical workflows package + only copy of its tests/examples/guides; uncommitted doc polish |
| 5 | `feature/adaptive-convection` | KEEP_ACTIVE | 13 commits ahead of origin (kinematic fault, budget metric) — **not pushed** |
| 6 | `docs/blog-posts` (worktree) | EXTRACT | Uncommitted 2026-06-04 revision of finding-particles post + figure set in no branch |
| 7 | `feature/cetz-figures` (worktree) | EXTRACT | Uncommitted element-location blog figure set + two doc/figure polish edits |
| 8 | `feature/elliptic-ma` (worktree) | EXTRACT | Uncommitted `mover=` selector for `Mesh.OT_adapt`, parallel RBF-cloud allgather fix, MA study scripts |
| 9 | `feature/exp-integrator-freesurface` | EXTRACT | Unmerged free-surface paper draft + integrator-zoo supplementary (only copy) |
| 10 | `feature/fault-convection` (worktree) | EXTRACT | Uncommitted `adaptive-fault-convection-reimplementation.md` spec — exists nowhere else |
| 11 | `feature/gradient-plasticity` (worktree) | EXTRACT | Entirely-uncommitted `add_smoothing_field()` gradient-plasticity spike (solver pyx + solvers.py) |
| 12 | `feature/petsc-cell-hint` | EXTRACT | Unmerged tip (cell-plane projection), uncommitted `build-petsc.sh` macOS SDK fix, FE-vs-RBF demo |
| 13 | `feature/snes-update-callbacks` | EXTRACT | Unmerged tip commit `b82acea7` (all-solver final-iterate dispatch, post-#250 review fix) |
| 14 | `feature/vep-two-stokes` | EXTRACT | Unmerged, validated `ViscoPlasticExplicitElastic` constitutive class + Phase G post-mortem |

---

## Triage Ledger — Investigated Branches

### LAND — push and PR to development

| Branch | Worktree | Risk of loss | One-liner |
|--------|----------|--------------|-----------|
| `feature/adapt-on-top` | `adapt-on-top` | **HIGH** | Layer-2 NVB/SBR adapt-on-top engine (`mesh.adapt` child meshes, native parallel newest-vertex bisection, `Surface.remap_to`, post-#290 custom_mg fixes) — complete and validated, local-only with no remote backup, and required by the already-shipped adapt-on-top-faults skill; push and PR to development. |
| `bugfix/yield-homotopy` | `yield-homotopy` | **HIGH** | The planned follow-up PR to #258: unified delta-soft-min/power-mean yield law + residual-paced yield homotopy + FMG monitor-leak fix, with tests and design docs — complete, local-only, unpushed. |
| `worktree-product-system` | `product-system` | **HIGH** | The canonical workflows/product-system package (api 0.2) with its only copy of the tests, convection+H2Ex examples, guides and scaffold command — unmerged, plus uncommitted doc/config polish that exists nowhere else; finish (commit dirty polish, drop .baks, fold in adaptive-convection's mesh_updates tweak, re-verify ddt fix against development) and PR to development so feature/adaptive-convection can rebase onto it. |
| `feature/parallel-point-eval` | `parallel-point-eval` | low | Manifold-PDE (dim != cdim) campaign, 6/7 commits already in development; only the tip — cell-plane projection enabling SLCN advection on SphericalManifold — remains to PR, and the petsc-cell-hint follow-up branch builds directly on it. |

**Evidence (LAND):**

- **`feature/adapt-on-top`** — 49 commits ahead of development, not merged (merge-base
  check), local-only branch with NO origin remote and no PR; worktree clean. Holds the
  Layer-2 adapt-on-top system: `mesh.adapt(engine="nvb"/"sbr")` returning a refined child
  (`discretisation_mesh.py` +506, engine param verified at branch lines 6015/6122), NVB
  graded-bisection engine incl. native uwnvb DMPlexTransform (`utilities/nvb.py` 341 lines,
  `nvb_transform.c` 1207 lines, `_nvb_transform.pyx`), parallel Route B with SF cross-rank
  closure, `Surface.remap_to`/director (`surfaces.py` +266), custom_mg fixes on top of merged
  Layer-1 (scalar-solver auto-inject guard `a4add4b7`; maybe_->auto_inject rename `1e1fabdd`),
  tests test_0835–0839 (~1150 lines), design docs LAYER2_SBR_ADAPT_ON_TOP.md +
  NVB_GRADED_ADAPT.md. NOT superseded: PR #290 merged only Layer-1 custom-P FMG; `git grep`
  shows zero NVB code in development. Dependency inversion: the adapt-on-top-faults skill
  already shipped to development (`.claude/skills/adapt-on-top-faults` via #299) documents
  `mesh.adapt(engine="nvb")`, which exists only here. Design doc marks Stages 2a/2b/2c DONE
  (2026-07-01) with parallel acceptance tests; tip commit dated 2026-07-03 is naming polish.
  Only recorded open item: marker-replay checkpoint.
- **`bugfix/yield-homotopy`** — 32 commits ahead of development (merge-base `25a4388e`,
  #262); worktree clean (`git status --porcelain` empty). Holds the "PR-2 after #258" VEP
  convergence work: unified delta-soft-min yield law + power-mean smoother
  (`constitutive_models.py` +599; power-mean at branch line 977, absent from development),
  residual-paced `enable_yield_homotopy`/`yield_smoother` API (absent from development src),
  FMG monitor-leak fix (`snes.monitorCancel` at `petsc_generic_snes_solvers.pyx`
  :7461/:7490/:7501/:7512, absent from development), flux_jacobian surrogate-tangent hook,
  `tests/test_1053_yield_homotopy.py` (+189 lines, new), 3 design docs
  (VISCOPLASTIC_YIELD_HOMOTOPY.md 386L, yield-homotopy-convergence-study.md 233L,
  jacobian-consistent-tangent.md 99L) and Spiegelman benchmark upgrades. NOT superseded:
  PR #258 (`c63cd707`) merged only the consistent-Jacobian half (carried here via merged
  sub-branch bugfix/jacobian-unwrap-to-constants); no other merged PR touches
  homotopy/power-mean/monitorCancel. Branch exists ONLY locally: no
  origin/bugfix/yield-homotopy, `git branch -r --contains` tip is empty. Last commit
  2026-06-30 wraps up "THE RECIPE" — study looks complete and PR-ready modulo a development
  catch-up merge (base predates #265/#290/#293/#294/#298/#304, so expect conflicts in the
  solver pyx and constitutive_models.py).
- **`worktree-product-system`** — Branch holds the ORIGIN of the `underworld3.workflows`
  product system: ~27 real commits (rest of the 47 are development merges +
  feature/fault-system-workflow ancestry) adding `src/underworld3/workflows/` (10 modules
  ~3.0k lines: _base/_products/_run/_runner/_cache/_cli/_diagram/scaffold, api 0.2 with
  cache_key freshness), `./uw workflow scaffold`, `tests/test_0810_workflow_runner.py`
  (248 lines), full convection-sweep + H2Ex example workflows (~4k lines under
  `docs/examples/workflows/`) and user+developer guides (tip `bb054b77`, 2026-05-07); plus
  cleanup commits (`model.py` −130 demo ThermalConvectionConfig classes;
  constitutive_models_new.py removal — the latter independently done on development,
  `995ec767`). NOT superseded: `git log development` shows no workflow-package merge and
  `src/underworld3/workflows/` is absent from development. feature/adaptive-convection
  carries a committed near-identical copy of the src package (deliberate local port per
  session records, unmerged, plus one improvement: `_run.py` mesh_updates kwarg) but NOT the
  tests, examples, guides, or scaffold — those exist only here. Worktree is DIRTY (14 files,
  post-tip polish): NEW docs found nowhere else (`docs/developer/guides/workflow-concepts.md`,
  `docs/api/workflows.md`, `docs/examples/workflows/index.md`), real config improvements
  (`convection_config.py` qdegree model_validator; `simulate.py` hidden-field CLI), Sphinx
  wiring (`docs/conf.py`, `api/index.md`), 2 junk .bak files, and a `ddt.py` SemiLagrangian
  `_clamp_to_domain` fix whose core intent (clamp SL back-trace to domain) development
  already has at `ddt.py:2627-2628/2689-2692` via return_coords_to_bounds — only its kd-tree
  fallback for meshes lacking that hook is unique.
- **`feature/parallel-point-eval`** — 7 commits ahead of development; 6 are patch-equivalent
  to commits already in development (`2d421388`, `6f4d64b0`, `fbf3df3e`, `2f892ca3`,
  `fd6070d3`, `6423c73f`, landed ~2026-05-21 — `git cherry` flags `9f7a1fa0` but its patch
  differs from merged `fd6070d3` only in hunk offsets). Only tip `7232bedb` (2026-05-24) is
  unmerged: `Mesh._project_to_nearest_cell_plane` (+139 lines `discretisation_mesh.py`) +
  `SphericalManifold.return_coords_to_bounds` cell-plane composition (+26 lines
  `spherical.py`) — both absent from development per `git grep` — plus 86-line
  INVESTIGATION.md diagnosis of the PETSc DMInterpolation 2-manifold cell-misroute, 12 probe
  scripts, and an advection animation. This commit is what makes SLCN advection run on the
  sphere (validated, tier-A unchanged per commit message). Worktree is clean. Tip is pushed
  to origin at the same SHA and is also the base of feature/petsc-cell-hint (dirty sibling
  worktree carrying the planned DMInterpolation cell-hint follow-up), so landing must
  coordinate with that successor: PR the single tip commit (rebase onto current development,
  ~5 weeks behind), then rebase petsc-cell-hint onto development and this branch/worktree can
  be removed.

### EXTRACT — rescue named content, then delete

| Branch | Worktree | Risk of loss | One-liner |
|--------|----------|--------------|-----------|
| `docs/blog-posts` | `blog-posts` | **HIGH** | Merged blog-posts branch whose worktree hides an unsaved 4-June revision of the finding-particles post plus its new element-location figure set — commit/PR the dirty files, then delete branch and worktree. |
| `feature/cetz-figures` | `cetz-figures` | **HIGH** | Superseded skill-shipping commit (merged as #299), but the dirty worktree uniquely holds a finished element-location blog figure set plus two small doc/figure polish edits worth rescuing before deleting. |
| `feature/elliptic-ma` | `elliptic-ma` | **HIGH** | The MMPDE/Monge-Ampere mover branch — fully merged into development, but its worktree holds uncommitted, unique follow-on work: the `mover=` selector for `Mesh.OT_adapt` (dropped from dev in a merge), a parallel RBF-cloud allgather fix for `_winslow_mmpde`, and the MA+arc-length study driver/renderer scripts. |
| `feature/exp-integrator-freesurface` | `exp-integrator-freesurface` | **HIGH** | Free-surface integrator-zoo campaign (May 2026): unique unmerged paper draft + benchmark supplementary + 1.5k-line investigation record, docs-only; dirty src prototypes already upstreamed to development, throwaway scripts should not land as-is. |
| `feature/fault-convection` | `fault-convection` | **HIGH** | Merged/never-diverged branch whose only value is one uncommitted design doc (the fault+adaptive-convection re-implementation spec, found nowhere else) plus 5 named reference scripts — commit the doc (and optionally those scripts) to development, then remove worktree and delete branch. |
| `feature/gradient-plasticity` | `gradient-plasticity` | **HIGH** | Gradient-plasticity research spike living entirely as uncommitted changes: a monolithically-coupled implicit-gradient (screened-Poisson) smoothing field for Stokes_Constrained, found nowhere else — commit it to preserve before removing the worktree; the accompanying Jacobian-bug handoff doc is superseded by PR #258. |
| `feature/petsc-cell-hint` | `petsc-cell-hint` | **HIGH** | Manifold/cdim stack mostly landed in development; extract the unmerged tip (cell-plane projection into return_coords_to_bounds), the uncommitted build-petsc.sh SDK-fallback fix, and the FE-vs-RBF demo, then archive — the dirty petsc_tools.c bypass prototype is superseded by feature/dminterp-bypass-element-check. |
| `feature/snes-update-callbacks` | `snes-update-callbacks` | **HIGH** | SNES per-iteration update callbacks — merged as #250 except the tip review-address commit (all-solver final-iterate dispatch + hook rename + test), a small genuine improvement to cherry-pick before archiving. |
| `feature/vep-two-stokes` | `vep-two-stokes` | **HIGH** | VEP integrator campaign mostly merged via PR #161; uniquely holds the unmerged, validated ViscoPlasticExplicitElastic constitutive class (operator-split yield-on-total, 7x SNES speed-up) plus the Phase G post-mortem design doc — extract those, discard the rest. |
| `bugfix/deform-cache-invalidation` | `deform-cache-invalidation` | low | Cache-invalidation fix for direct `_deform_mesh` calls; two of three commits already landed on development, and the remaining 31-line `_mesh_version` bump addresses a still-live stale-kdtree gap in `mesh.deform()` but must be reworked around the snapshot version-gate (e.g. a separate geometry-version counter) — extract that finding as an issue/small PR, then archive-tag and delete branch + remove the clean worktree. |
| `feature/fault-system-workflow` | `fault-system-workflow` | low | Origin of the workflows/product-system idea plus the H2Ex geographic-fault pipeline examples; package superseded by worktree-product-system and src fixes by PR #131, but the W1–W5/Step5–6 example suite (~3.5k lines) exists only here — cherry-pick it onto product-system, then archive. |

**Evidence (EXTRACT):**

- **`docs/blog-posts`** — Branch tip `29a2029d` is fully merged into development
  (`merge-base --is-ancestor` confirms; `development..docs/blog-posts` is empty, 0 ahead) and
  development has not touched `publications/blog-posts/` since, so the BRANCH itself is dead
  weight. But the WORKTREE holds a coherent uncommitted editing session dated 2026-06-04: a
  145-line prose revision of `publications/blog-posts/finding-particles.md` (KDTree
  control-point cell location + `dm.migrate()` negotiation write-up), 33 lines in
  particles-as-symbols.md, a README tweak, and 6 untracked figure assets
  (element-location-demo.png/.typ, 3 data JSONs, generate-element-location-data.py) that
  appear in NO branch (`git log --all` on those paths is empty). The revised text embeds the
  new figure, so markdown + figures belong together. Extraction = commit the 9 dirty paths to
  a fresh docs branch and PR to development; then the worktree can be removed and the merged
  branch deleted.
- **`feature/cetz-figures`** — 1 commit ahead (`a8f87f87`: track .claude/skills +
  cetz-figures skill) — byte-identical to merged PR #299 (`aa3e383e` on development,
  2026-07-03), so the branch itself is fully superseded. But the worktree is dirty with 18
  files, and 3 items exist NOWHERE else: (a) a complete new element-location blog figure set
  (typ + rendered png + generator .py + 3 data JSONs, dated June 3-4) for
  `publications/blog-posts/finding-particles.md`, which the merged blog post does not yet
  reference; (b) an unmerged edit to `docs/advanced/curved-boundary-conditions.md` swapping
  the ASCII-art normals sketch for the already-committed
  `figures/curved-bc/facet-vs-true-normals.png`; (c) cosmetic polish to
  `docs/advanced/figures/cuboid-3d/cuboid-faces.typ` (+regenerated png). The other 12 dirty
  files are byte-identical duplicates of content already on development (blog figures /
  cuboid-3d generators).
- **`feature/elliptic-ma`** — Branch tip `4dccafb7` is 0 ahead and a verified ancestor of
  development (`git merge-base --is-ancestor` passes; listed in `--merged`). Its work —
  anisotropic MMPDE mover, elliptic Monge-Ampere mover, slip surfaces, RBF metric eval — is
  live in development (`smoothing.py:2992` `_winslow_mmpde`, `:3617` "ma", `:3642` "mmpde";
  `_ot_adapt.py:250` says "grafted from feature/elliptic-ma"). Note: the branch's committed
  `mover=` kwarg on `_ot_adapt_step` (`c977e485`) was silently dropped from current
  development by merge `b7da8f29`, so dev's OT_adapt hardcodes the "ot" mover. The worktree
  is DIRTY with 5 files, none junk: (1) pyx +8 `_pre_solve_hook` — redundant, duplicated on
  feature/gmg-geometric-interp and superseded by custom_mg/set_custom_fmg (#290);
  (2) `discretisation_mesh.py` +2 exposing `mover=` through `Mesh.OT_adapt` — exists in no
  current branch; (3) `smoothing.py` +12 parallel allgather of the RBF reference cloud in
  `_winslow_mmpde` (fixes rank-local KDTree → non-SPD metric at partition boundaries) — no
  equivalent anywhere; (4) `scripts/stagnant_lid_adapt_loop.py` +94 study flags
  (--adapt-method ot-reset-ma, --metric-choice arc-length, --dt-basis mean) plus an OT-reset
  resume-reference correctness fix; (5) untracked `scripts/render_ma_arclen_frames.py`
  (167 lines, P3-faithful MA/arc-length animation renderer). Items 2–5 found nowhere else by
  cross-branch grep.
- **`feature/exp-integrator-freesurface`** — 6 commits ahead of development, +12,399 lines,
  ALL docs/publications, zero src changes. Holds: publication-track paper draft
  `publications/free-surface-paper/draft.md` (494 lines: semi-Lagrangian kinematic update +
  amplitude-invariant relaxation CFL for free-surface viscous flow) and
  integrator_zoo_supplementary.md (819 lines, FE/RK2/RK4/AB2/BDF2-SL + Cathles benchmark);
  investigation record `docs/developer/design/EXPONENTIAL_FREE_SURFACE.md` (1496 lines) +
  2 handoff docs + deformable_surface_metronome_design_note.md (Model.advance timekeeper
  architecture sketch); plus ~20 throwaway `_phase_i_*`/`_plot_*` scripts (~9k lines) of
  exactly the docs/design contamination class the audit flags. NOT superseded:
  `git log --all -- publications/free-surface-paper/` shows the draft exists only on this
  branch; no merged PR contains any of it. Dirty worktree (25 files): the 3 modified src
  files (`discretisation_mesh.py` +25, `ddt.py` +93, `solvers.py` +3) are prototypes of fixes
  that ARE now on development — SL monotone limiter + theta landed via `d09f1a3b`/`75bf61af`
  (`ddt.py:1466-1507` on development), `_deform_mesh` cache invalidation via `93d2501f`
  (`discretisation_mesh.py:3213-3225`) — so the src diffs are safely discardable. The 22
  untracked files are post-tip debugging/viz scripts plus
  free-surface-convection-session-2026-05-13.md (session note, exists nowhere else; its
  conclusions were upstreamed as the above commits). Extract = land the paper draft,
  supplementary, EXPONENTIAL_FREE_SURFACE.md and metronome note (relocating throwaway scripts
  per audit policy), optionally rescue the session note, then archive-tag and delete; the
  free-surface effort has since pivoted to the 3-number held-lid integrator, so the branch
  itself is not in-flight.
- **`feature/fault-convection`** — Branch tip `25a4388e` equals the merge-base with
  development (zero ahead commits; `git log development..feature/fault-convection` is empty)
  — the branch holds no unique commits and the real fixes it explored landed via PRs
  #259/#264/#266 already on development. The worktree is dirty with 58 untracked files: 57
  exploratory diagnostic scripts (`scripts/fault_*.py` etc.) whose quantitative results are
  explicitly declared suspect/superseded, plus ONE valuable file —
  `docs/developer/design/adaptive-fault-convection-reimplementation.md` (73-line spec dated
  2026-06-22: validated mmpde recipe, anisotropic fault-metric formula
  M = rho·I + (Rf²−1)·exp(−(d/w)²)·n nᵀ, gmsh refine_lines + carrier approach, diagnostics
  inventory, 4-step landing plan). Verified this doc exists nowhere else: no git history on
  any branch, absent from the main checkout and ~/+Simulations; the adapt-on-top-faults and
  adaptive-meshing skills cover different/partial recipes. The doc cites 5 of the scratch
  scripts by name as its reference material (fault_convection_adapt_loop.py,
  mmpde_metric_proof.py, adapt_vs_uniform_compare.py, bc_leak_check.py,
  multi_history_plot.py). The previously recorded uncommitted smoothing.py dup is no longer
  present (no modified tracked files).
- **`feature/gradient-plasticity`** — Branch has 0 unique commits (tip `7696d89f` =
  development ancestor, PR #240; trivially merged). ALL value is uncommitted in the worktree:
  (1) `solvers.py` +51 — new `SNES_Stokes_Constrained.add_smoothing_field()`:
  screened-Poisson implicit-gradient smoothing field coupled monolithically into the saddle
  solve (gradient-plasticity regularization); (2) `petsc_generic_snes_solvers.pyx` +105/−9 —
  its assembly (Helmholtz F0/F1, G0/G3, su/us cross-coupling Jacobian blocks, fieldsplit
  grouping of all multipliers). `git grep` across every branch head:
  `add_smoothing_field`/`_is_smoothing` exist NOWHERE else. (3) Untracked
  `docs/developer/design/jacobian-unwrap-constants-bug.md` (190 lines, "fix not yet started")
  IS superseded — the fix landed as PR #258 (`c63cd707`) and development has
  jacobian-consistent-tangent.md — but the smoothing-field spike is a distinct capability NOT
  covered by #258. Spike has no tests and changes solver-pyx numerics, so
  commit-to-preserve (branch or archive tag), don't PR as-is.
- **`feature/petsc-cell-hint`** — Holds the cdim/manifold stack (SphericalManifold,
  embedded-coord plumbing, parallel point-eval on 2-manifolds). 6 of 7 ahead commits already
  landed in development via cherry-picks (`git cherry`:
  `3a09fa02`/`107de9b6`/`fff550eb`/`166cf5af`/`09138cc0` patch-equivalent; `9f7a1fa0`
  identical to dev `fd6070d3` modulo hunk offsets). UNMERGED: tip `7232bedb` —
  `Mesh._project_to_nearest_cell_plane` (139 lines) +
  `SphericalManifold.return_coords_to_bounds` cell-plane composition (26 lines) +
  INVESTIGATION.md DMInterpolation-misroute diagnosis + 12 probe scripts; absent from
  development and from successor branch feature/dminterp-bypass-element-check (which has the
  real bypass fix, `17a5a8d3`+`7168a0ae`, off an older base without the manifold files).
  Dirty worktree: petsc_tools.c/_function.pyx = earlier prototype of the bypass, superseded
  by dminterp-bypass branch commits; `build-petsc.sh` 18-line macOS SDK xcrun-fallback fix
  exists NOWHERE else (`git log --all -S` empty); untracked compare_fe_rbf_lonlat.py
  FE-vs-RBF validation demo + rendered results exist nowhere else.
- **`feature/snes-update-callbacks`** — 4 commits ahead of development (+589/−56 across
  `petsc_generic_snes_solvers.pyx`, `systems/solvers.py`, test_1016,
  `docs/advanced/solver-iteration-callbacks.md`). First three commits were squash-merged as
  PR #250 (development `54b815c3`, 2026-06-18 20:26). Tip commit `b82acea7` (20:54, 28 min
  AFTER the merge) is UNMERGED and exists only on this branch (local + origin): it
  centralises the final-iterate callback dispatch in `_snes_solve_with_retries` so non-Stokes
  solvers get it too (verified: development pyx:8070 still Stokes-only, pyx:325 still
  `_maybe_install_snes_update`), removes the spurious `_needs_function_rewire` on callback
  registration, renames `_maybe_install_snes_update` to `_attach_snes_update_hook` (the
  no-maybe_-prefix rule), and adds test_final_iterate_dispatch_on_scalar_solver. Worktree is
  clean (`git status --porcelain` empty).
- **`feature/vep-two-stokes`** — 91 ahead of development, but `git cherry` shows 73/91
  commits patch-equivalent — landed via PR #161 (feature/exp-integrator-investigation, merged
  2026-05-05): Phase A–F ETD work, integrator='bdf'/'etd' API, DDt.set_initial_history,
  test_1052_* regression tests (verified in development tree). The 17 unique commits
  (Phase G, 2026-04-29..05-01) are NOT superseded: two-Stokes operator split rejected with
  post-mortem, then the working v5b result — class ViscoPlasticExplicitElastic at
  `constitutive_models.py:2008` (~260 net src lines, bdf_blend/etd_blend damping, 4-sig-fig
  baseline match, ~7x fewer SNES iters on first-order integrators), a 316-line completed
  VEP_TWO_STOKES_OPERATOR_SPLIT.md (development only has the plan stub), a VE convergence
  study, plus ~15 throwaway `_phase_g_*` scripts/traces/pngs in docs/developer/design/.
  `git grep` confirms ViscoPlasticExplicitElastic absent from development; June's VEP work
  (PR #258 consistent Jacobian + yield-homotopy) is a different route, not a replacement.
  Worktree clean (`status --porcelain` empty), branch pushed to origin. Extract = cherry-pick
  the class + design-doc post-mortem + convergence study onto a fresh branch (skip the
  `_phase_g_` contamination), then archive-tag and delete.
- **`bugfix/deform-cache-invalidation`** — 3 substantive commits ahead. Two (lambdify
  `_expr_hash` fix closing #194 + cache-contract test rewrite) already on development as
  cherry-picks `ca5c3efb`/`16103071` — net unique diff is one 31-line hunk (`c641dfe8`): bump
  `_mesh_version` and clear `_restore_kdt`/`_restore_coords_id` in `_deform_mesh`. PR #191
  was CLOSED 2026-07-02 as superseded, but the closure claim ("deform() bumps the version,
  lines 863-865") is wrong — verified those lines are the mesh.X.coords callback, while
  public `deform()` (`discretisation_mesh.py:3092`) calls `_deform_mesh` directly
  (`:3123-3127`) which bumps only `_topology_version` (`:3224`). So `_mesh_version`-keyed
  caches (`_get_domain_kdtree` `:5571-5582`, `_owned_cells_mask_cache` `:4794`, variable
  `_get_kdtree` used by rbf_interpolate, `discretisation_mesh_variables.py:998-1006`) stay
  frozen after `mesh.deform()` on development@1d003481 — the gap looks still live. The fix
  cannot cherry-pick as-is: it collides with the snapshot `_mesh_version` gate
  (`:4191-4219`; PR #191 CI failed with SnapshotInvalidatedError), and the `_restore_kdt`
  half targets attributes appearing nowhere in src/. Worktree clean; all commits pushed to
  origin and preserved in closed PR #191.
- **`feature/fault-system-workflow`** — 31 ahead of development, last commit 2026-04-22,
  worktree clean, tip `c567b311` == origin/feature/fault-system-workflow (fully pushed).
  Holds three things: (1) the original underworld3.workflows "make-like product system"
  package — SUPERSEDED: worktree-product-system branched from these very commits
  (`af3778ef`/`f9970c1e` are ancestors, verified with `merge-base --is-ancestor`) and
  extended it to api 0.2 (+1895 lines _cache/_cli/_diagram/_run/_runner/scaffold, last commit
  2026-05-07); (2) src fixes (ellipsoid quantities, checkpoint round-trip, Darcy projection
  sign, commit `5668edb4`) — SUPERSEDED by merged PR #131 (`3829b271`/`62bebb3f` on
  development; `_checkpoint_ellipsoid_pending` present in dev coordinates.py); (3) UNIQUE
  unmerged work found nowhere else except its own origin remote:
  `docs/advanced/h2ex-workflow.md` + ~3.5k lines of H2Ex geographic fault pipeline examples
  in `docs/examples/WIP/Geographical/` (W1–W5b, W5-Interactive trame visualiser,
  Step5-RunPipeline, Step6-Visualise), Louis-authored, absent from both development and
  worktree-product-system (`git cat-file` checked). Those scripts import
  underworld3.workflows (WorkflowProducts), so they must be carried onto product-system (or
  land with it), then tag archive/feature/fault-system-workflow and delete.

### KEEP_ACTIVE — in-flight, do not touch

| Branch | Worktree | Risk of loss | One-liner |
|--------|----------|--------------|-----------|
| `bugfix/custom-mg-parallel` | `custom-mg-parallel` | **HIGH** | In-flight continuation of the custom-MG program past merged PR #290: the whole Layer-2 NVB/adapt-on-top engine (native uwnvb transform, mesh.adapt child, 5 test files) plus today's non-nested np>1 cross-partition transfer — none of it merged or pushed, with uncommitted design-doc updates for today's work; keep and finish (should be committed/pushed and PR'd soon). |
| `feature/adaptive-convection` | `adaptive-convection` | **HIGH** | Active adaptive-convection-with-faults study: workflows-package port (+adaptive-mesh checkpoint fix), Annulus refine_lines, and the kinematic-fault example — in-flight, 13 commits not yet pushed to origin. |

Per the execution protocol, `feature/numpy2-support` (main-checkout work stream) and the
quality-audit branch/worktree (`feature/quality-audit-2026-07`) are also KEEP_ACTIVE.

**Evidence (KEEP_ACTIVE):**

- **`bugfix/custom-mg-parallel`** — 52 commits ahead of development (merge-base `417d18b5`,
  2026-07-02; tip `7311fe6d` dated 2026-07-03), +5274/−115 lines across 25 files. Despite the
  branch name, only a small fraction is "custom-mg parallel bugfix": the branch has absorbed
  feature/adapt-on-top (merge `29890cd0`) and carries the ENTIRE Layer-2 program that is NOT
  in development: (1) NVB graded adapt engine — new `src/underworld3/utilities/nvb.py`
  (341 lines), `_nvb_transform.pyx` + `nvb_transform.c` (native uwnvb DMPlexTransform, 1249
  lines), mesh.adapt SBR/NVB adapt-on-top child in `discretisation_mesh.py` (+506),
  `Surface.remap_to` + director in `meshing/surfaces.py` (+266); (2) five new test files
  test_0835–0839 (~1144 lines) + test_1017 parallel additions; (3) two brand-new commits
  (`5ac985ef` operator-faithful finest reduced map on adapt children; `7311fe6d`
  cross-partition transfer for non-nested coarse tails at np>1) extending `custom_mg.py`
  (+262 over development's copy). Supersession check: PR #290 (`f6a75ef0`) merged only
  Layer-1 custom-P FMG; PR #297 merged the FMG lockout — nothing in development contains
  nvb.py, the NVB tests, adapt-on-top, or the cross-partition transfer (`git ls-tree
  development` confirms nvb.py absent; grep of development log finds no NVB/adapt-on-top
  merges). No remote copy exists (origin has only feature/custom-mg-prolongation); the tip
  commits live only on this branch and local feature/adapt-on-top. Worktree dirty: 2 files,
  both docs/developer/design/ notes (GENERALIZED_FMG_HIERARCHY_AND_ADAPT.md +24/−8,
  LAYER2_SBR_ADAPT_ON_TOP.md 1-line rename fix) — real, uncommitted documentation of the
  step-4 work (non-nested np>1 DONE) that exists nowhere else. The NVB adapt engine is also
  load-bearing for the active adapt-on-top-faults workflow/skill and the annulus fault
  studies.
- **`feature/adaptive-convection`** — 17 commits ahead of development, +6,224 lines, worktree
  clean. Holds: (1) local port of the underworld3.workflows package from
  worktree-product-system plus a unique 9-line mesh_updates fix in `workflows/_run.py`
  (per-step mesh geometry checkpointing for adaptive meshes) — diffed the two branches, only
  `_run.py` differs, so product-system does NOT have this fix; (2) `meshing/annulus.py`
  refine_lines gmsh line-refinement (+56 lines, absent from development);
  (3) `docs/examples/workflows/adaptive_convection/` study (~3,900 lines: fault_config.py,
  kinematic fault/ridge advection, budget multi-monitor metric, diagnostics/render scripts).
  NOT superseded by any merged PR: development has no workflows package, no refine_lines, no
  adaptive_convection example (verified via `git ls-tree`/grep and `log --grep`). CAUTION:
  local tip `22deae75` is 13 commits ahead of origin/feature/adaptive-convection (remote
  stuck at `4686b86e`) — the kinematic-fault and budget-metric commits exist only in the
  local repo and should be pushed. Project records mark this as in-flight (Task 2 = fault
  friction is next); workflows duplication with worktree-product-system is deliberate,
  coordinate before PR.

### ARCHIVE_DELETE — superseded; tag `archive/<branch>` then delete

| Branch | Worktree | Risk of loss | One-liner |
|--------|----------|--------------|-----------|
| `feature/boundary-flux` | `boundary-flux` | none | CBF boundary_flux primitive (heat flux/Nusselt/traction) — merged verbatim as PR #294 and since improved on development; branch and clean worktree are safe to archive-tag and delete. |
| `feature/rotated-freeslip-bc` | `custom-mg-prolongation` (misnamed) | none | Source branch of merged PR #293 (rotated strong free-slip + sigma_nn dynamic topography); tip byte-identical to the squash, worktree clean — archive-tag and delete both branch and the misnamed custom-mg-prolongation worktree. |
| `feature/dminterp-bypass-element-check` | `dminterp-bypass-element-check` | low | The DMLocatePoints-bypass prototype that development already absorbed (`c52201b8` cites it by name); only residue is a cosmetic one-line sentinel-cast guard in petsc_tools.c that is a no-op on real platforms. |
| `feature/fs-stress-equilibrium` | `fs-stress-equilibrium` | none | Docs-only snapshot of the ETD stress-equilibrium free-surface investigation (design notes + phase-I experiment scripts), identically contained in and superseded by feature/fs-surface-smoother-driver. |
| `feature/fs-surface-smoother-driver` | `integrate-surface-submesh` (misnamed) | low | Mid-June free-surface etd_topo exploration record (12.6k lines of design docs/scripts, method since superseded); every src/test change already merged via PRs #237/#238/#246/#249/#251 — archive-tag (after committing 3 dirty exploration files) and delete. |
| `feature/projection-smoothing-length` | `projection-smoothing-length` | none | v1 of the projection smoothing_length API — fully superseded by the v2 rewrite merged as PR #234; clean worktree, nothing unique left. |
| `feature/snes-constant-nullspace` | `snes-constant-nullspace` | none | Original standalone constant-nullspace-for-singular-Poisson branch; its feature, guard, and test all landed on development (PR #236 + dedup `5bb72c4c`) in stronger form — safe to tag archive/ and delete with its worktree. |
| `feature/exp-integrator-investigation` | `vep-loading-unloading` (misnamed) | low | Merged (PR #161) exponential-integrator branch whose single unmerged commit is a superseded Phase I kinematic-ETD free-surface investigation doc + handoff — the handoff was fulfilled by the merged FREESLIP_DYNAMIC_TOPOGRAPHY_FREESURFACE.md scheme; tag archive/ and delete. |
| `api/surface-quantity-support` | — | none | Feb-2026 unit-aware Surface API branch (quantity args for refinement_metric/influence_function + dimensionalised pv_mesh) whose both commits were cherry-picked verbatim into development in March — fully superseded, safe to tag archive/ and delete. |
| `feature/custom-mg-prolongation` | — | none | The generalized custom-P FMG hierarchy work (Phase 1 complete), squash-merged to development as PR #290 — every commit's content verified present in development, safe to tag archive/ and delete. |
| `feature/snesfas-spike` | — | none | Raw commit trail of the SNESFAS/Vanka multilevel nonlinear-Stokes feasibility spike — its final docs (plus curated prototypes) already merged via PR #245. |
| `feature/vep-loading-unloading` | — | low | April-2026 VEP loading/unloading investigation whose real fixes (divergence_retries, mixin recursion) already re-landed on development; leftover is an A/B evidence trail (test-only plastic-basis option, benchmark scripts, figures) preserved by the archive tag. |
| `fix/darcy-sign-convention` | — | none | The rejected PR #243 flipped-flux Darcy sign fix — same bug fixed correctly (with f!=0 tests) by merged #255; approach explicitly superseded, nothing to salvage. |
| `ss2098/fix-thermal-convection-units-tutorial` | — | none | External contributor's units-tutorial fix, fully squash-merged (and slightly improved) as PR #263 — the branch is now redundant. |
| `test-darcy-rebase` | — | none | Stale local rebase-test of the Darcy/Richards feature branch, fully merged to development in April (`4895c433`); safe to tag archive/test-darcy-rebase and delete. |

**Evidence (ARCHIVE_DELETE):**

- **`feature/boundary-flux`** — Holds the Consistent Boundary Flux (CBF) primitive: 3 commits
  ahead (`4dbd30bd` feature, `40e86996` np4 partition-cut reaction fix, `787c0bde` Copilot
  review fixes) adding `utilities/boundary_flux.py` (+255), solver pyx methods (+108), tests
  test_1019 + parallel test_1065 (+144). Fully superseded by merged PR #294 (squash
  `ae647025` on development): `git diff feature/boundary-flux ae647025` shows ZERO
  differences in any boundary-flux file (only pre-existing #290 custom_mg files differ). The
  prior session record's "OPEN np4 boundary-cut issue" was fixed in `40e86996` and IS in the
  merge. Development has since evolved the module further (partial_reaction mode +
  write_boundary_scalar_field factored out by #293). Worktree `.claude/worktrees/boundary-flux`
  is clean (`git status --porcelain` empty).
- **`feature/rotated-freeslip-bc`** — Branch holds 11 commits (`41c64819`..`f4d217f9`):
  rotated strong free-slip BC with sigma_nn reaction, geometric-FMG solve path,
  parallel-safety fixes, CBF sigma_nn topography, dynamic_topography hand-off, plus tests
  test_1018/test_1064. Fully superseded by merged PR #293 (squash `51b19182` "Rotated strong
  free-slip BC + sigma_nn dynamic-topography (parallel), free-surface hand-off") — verified:
  `git diff 51b19182 f4d217f9` on all 5 touched files (rotated_bc.py, boundary_flux.py,
  petsc_generic_snes_solvers.pyx, both test files) is EMPTY, and merge-base `ae647025` is the
  commit directly before the squash. Development has since moved further on these files
  (#298 rotated-SNES). Worktree
  `/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/custom-mg-prolongation` is
  CLEAN (empty `status --porcelain`). Worktree name is a leftover from the #290 custom-MG
  work; the directory was reused for this branch.
- **`feature/dminterp-bypass-element-check`** — 2 commits ahead of merge-base `44c2f472`:
  `17a5a8d3` (DMLocatePoints bypass on simplex/manifold meshes given a kdtree cell hint) and
  `7168a0ae` ((size_t)-1 sentinel guard before PetscInt cast). Commit 1 is EXPLICITLY
  superseded: development's `petsc_tools.c:73-74` says "ported from
  feature/dminterp-bypass-element-check, 17a5a8d", brought in by `c52201b8` (2026-05-28,
  "remesh field-transfer redesign + parallel locator hardening"), and development has since
  evolved past the branch (PR #203 serial bypass, hint policy centralized in
  `mesh._eval_use_robust_location`, out-of-domain best-claim block in `_function.pyx`).
  Commit 2's one-line guard is the only unported residual, and it is practically inert:
  `(PetscInt)(size_t)-1 == -1` on all supported compilers and development's recovery path
  (`petsc_tools.c:142`) already rejects negatives via `recovery_cells[k] >= 0` — a
  portability nicety, not a live bug. Worktree is clean (`git status --porcelain` empty; root
  vep_*.png are tracked at merge-base, not stray work).
- **`feature/fs-stress-equilibrium`** — 1 commit ahead of development (`9a5b5037`,
  2026-06-14): docs-only, +12,406 lines in docs/developer/design/ — 5 design/session notes
  (EXPONENTIAL_FREE_SURFACE.md, EXPONENTIAL_FREE_SURFACE_HANDOFF.md,
  STRESS_EQUILIBRIUM_FREESURFACE.md, deformable_surface_metronome_design_note.md,
  free-surface-convection-session-2026-05-13.md) plus ~20 throwaway
  `_phase_i_fs_*`/`_plot_*`/`_probe_*` experiment scripts. No src/ changes. Not merged to
  development, but SUPERSEDED by branch feature/fs-surface-smoother-driver: `git cherry`
  marks `9a5b5037` as already-applied there, and `git range-diff` confirms it equals
  `a0ef4df8` byte-for-byte; that branch adds 4 further commits (surface-smoother driver
  wiring, deform identity-gate fix, gmsh spacedim-leak fix, zoo port with tests). No merged
  PR contains it; #293's free-surface hand-off on development is different content. Worktree
  `.claude/worktrees/fs-stress-equilibrium` is clean (`git status --porcelain` empty).
  Caveat: "no loss" depends on feature/fs-surface-smoother-driver being kept or archived — it
  is the only other holder of this patch.
- **`feature/fs-surface-smoother-driver`** — 5 commits ahead of development (tip `b72c6521`,
  2026-06-15). The two src fixes (`c383aa9d` gmsh dm_plex_gmsh_spacedim leak; `ff2bddab`
  coord identity-gate) merged to development via PRs #238 (`84294f2c`) and #237 (`0ed93123`)
  — `_clear_gmsh_import_options` and the gate are present in dev's discretisation_mesh.py.
  The remaining 3 commits (etd_topo stress-equilibrium free-surface integrator + zoo driver)
  are ~12.6k lines entirely in docs/developer/design/ (EXPONENTIAL_FREE_SURFACE.md 1496
  lines, HANDOFF, STRESS_EQUILIBRIUM_FREESURFACE.md, ~19 `_phase_i_`/`_plot_`/`_probe_`
  exploration scripts) — exploration whose method was superseded by the free-slip +
  three-number topography integrator line (PR #293 hand-off, Crameri-benchmarked write-up).
  Worktree dirty (13 files): ALL src/test modifications verified present in development
  (capability gate + ephemeral_coords via #246/#249; old_frame_traceback via
  #251/`a708e8c4`; topography(reference="mean") + parallel-safe Stokes_Constrained; SL-BDF2
  docstrings); untracked test_0826/test_0855/SL docs byte-identical to dev. Unique
  uncommitted leftovers are exploration-only: +232-line zoo script extension,
  `_phase_i_fs_deform_demo.py`, and a ~45-line prototype-provenance section in
  lagged-clone-sl-history.md (mechanism already merged+documented in dev ddt.py). Recommend
  committing the 3 dirty exploration files onto the branch before tagging
  archive/feature/fs-surface-smoother-driver so the tag captures everything, then delete
  branch + worktree.
- **`feature/projection-smoothing-length`** — Holds 2 commits (`16cb46ef`, `1c5ef1c7`):
  unit-aware smoothing_length property on the four SNES Projection classes plus a Pint
  round-trip fix, with `tests/test_0505_projection_smoothing_length.py`; touches only
  systems/solvers.py and that test. Superseded by merged PR #234 (development merge
  `da4573a0`, from feature/projection-smoothing-length-v2, squash `31e7fd2f`) — the
  CI-passing rewrite of this v1 branch (its own PR #200 had 5 CI failures). Test file is
  byte-identical to what #234 merged; current development has smoothing_length live (27 refs
  in solvers.py). Worktree is clean (`git status --porcelain` empty). Sibling worktree
  projection-smoothing-length-v2 sits at the merged commit (bulk list).
- **`feature/snes-constant-nullspace`** — 2 commits ahead (`d99e8723` adds
  SNES_Scalar.petsc_use_constant_nullspace; `f5b53ad4` warns on nullspace+Dirichlet, adds
  test_1055, 129 lines); worktree clean (`status --porcelain` empty). Fully superseded by PR
  #236 (merge `5c790acd` of feature/manifest-constant-nullspace, commit `dc6dacd0`) plus
  dedup commit `5bb72c4c` on development: development's petsc_generic_snes_solvers.pyx
  carries the canonical constant_nullspace with a STRONGER guard (raises ValueError on
  Dirichlet BCs vs this branch's warning), cached nullspace object, near-nullspace for GAMG,
  and keeps petsc_use_constant_nullspace as a documented back-compat alias (pyx ~2429-2442).
  The branch's test survives as `tests/test_1056_constant_nullspace.py` (same file ported to
  canonical flag name, plus an extra setter test). `d99e8723`'s own message anticipated this:
  "once that merges to development the duplicate will be resolved on rebase."
- **`feature/exp-integrator-investigation`** — Main work (Phase A–I VE/VEP exponential
  integrator, ETD-1/ETD-2 API, EXPONENTIAL_VE_INTEGRATOR.md) merged to development via PR
  #161 (`03473c62`). Sole ahead-commit `2d7205bb` (2026-05-06) adds only
  docs/developer/design/ material: EXPONENTIAL_FREE_SURFACE.md (899 lines, self-marked
  "investigation in progress"), a 284-line session HANDOFF doc with open problems (large-dt
  drift, gamma estimation), and 5 throwaway `_phase_i_*`/`_plot_*` scripts (2,619 lines
  total, no src/ changes). That handoff was consumed: its open questions are resolved by the
  merged, Crameri-benchmarked three-number h-infinity scheme in
  docs/developer/design/FREESLIP_DYNAMIC_TOPOGRAPHY_FREESURFACE.md on development (landed
  with PR #293/#264 work), which rejects the kinematic-ETD approach the Phase I doc explores.
  Ahead-commit exists on no other branch; worktree vep-loading-unloading is completely clean
  (empty `git status --porcelain`). Archive tag preserves the investigation narrative for the
  record.
- **`api/surface-quantity-support`** — Holds 2 commits touching only
  `src/underworld3/meshing/surfaces.py` (+86/−17): `9ffff618`
  "Surface.refinement_metric()/influence_function() accept uw.quantity values" (adds
  `_to_nd_length` helper) and `38fb30b9` "Surface.pv_mesh returns dimensionalised coordinates
  for visualisation". Both are EXACT patch-id matches to commits already in development —
  `9ffff618` ≡ `4c16f5a5` and `38fb30b9` ≡ `d37a76b0` (both cherry-picked to development
  2026-03-20; verified with `git patch-id`, identical hashes). Current development
  surfaces.py confirmed to contain `_to_nd_length` (line 58, now used at ~15 call sites) and
  the dimensionalised pv_mesh property (lines 921-935). No worktree exists for this branch;
  nothing dirty. Zero unique content remains.
- **`feature/custom-mg-prolongation`** — 16 commits ahead of merge-base `a7f0b11f`: custom-P
  geometric-MG prolongation (`utilities/custom_mg.py` 653 lines, set_custom_mg hook in
  petsc_generic_snes_solvers.pyx, tests 1015/1016/1017 + parallel, design doc). Fully
  superseded by merged PR #290 (squash `f6a75ef0`, 2026-07-02 12:00, includes branch tip
  `b7d1c4fd` from 11:28): all 6 non-pyx files byte-identical to development, pyx hooks
  confirmed present in development (lines 156-158/259/3046). Local tip == origin tip; no
  worktree on this branch (the `.claude/worktrees/custom-mg-prolongation` directory is
  checked out on feature/rotated-freeslip-bc, a different branch).
- **`feature/snesfas-spike`** — 16 docs-only commits ahead of development, netting to 3
  design docs (+847 lines): multilevel-nonlinear-stokes-strategy.md, snesfas-feasibility.md,
  snesfas-vanka-feasibility-study.md (SNESFAS/Vanka nonlinear-multigrid feasibility spike, no
  src/ changes). Fully superseded by merged PR #245 (`34a9dd46`, "docs(solvers): preserve
  SNESFAS / Vanka / grid-sequencing investigation") via sibling branch
  docs/snesfas-investigation, which contains these 16 commits in its history: two of the
  three docs are byte-identical on development, the third is a strict superset there (adds
  pointer to in-repo prototypes docs/examples/snesfas_investigation/ that the branch lacks).
  No worktree on this branch; nothing uncommitted.
- **`feature/vep-loading-unloading`** — 14 commits ahead of development (last 2026-04-24), no
  worktree attached (the vep-loading-unloading worktree dir is now on
  feature/exp-integrator-investigation). All functional src changes are already on
  development: divergence_retries re-landed as `ee667376` (present at
  `petsc_generic_snes_solvers.pyx:1068` `_snes_solve_with_retries`), MathematicalMixin
  recursion fix and timestep restore are patch-equivalent per `git cherry` (dev
  `c09f6281`/`20c2cbec`). Remaining unique content is exploratory evidence only: a test-only
  `plastic_strain_rate_basis="effective"|"instantaneous"` option in constitutive_models.py
  (notes file states "Default is unchanged — read-only evidence, not a proposal"), 3 A/B
  benchmark scripts + notes, and 6 figure-only commits (retake/reactive-dt strategy PNGs; the
  retake code itself was never committed). Development later replaced this investigation with
  its own vardt benchmark suite (docs/advanced/benchmarks/bench_ve_square_vardt*.py) and
  solved variable-dt VEP drift via snapshot projection (`5b3548c8`/`beb4f2e3`), superseding
  the branch's line of inquiry.
- **`fix/darcy-sign-convention`** — 2 commits ahead of development (`a807b552`, `c50aaac4`;
  last 2026-06-16, author J.C. Graciosa), no worktree, clean. They flip DarcyFlowModel.flux
  to −q and re-plumb SNES_Darcy/TransientDarcy velocity projection and DFDt around the
  flipped sign; the branch's own F1 docstring concedes the f≠0 source term is left wrong.
  Explicitly superseded by merged PR #255 (development commit `46b8f16c`, 2026-06-19,
  "Closes #214. Supersedes #243 (which flipped the flux instead, introducing an untested
  f != 0 source-sign regression)") which fixes the same transient-velocity sign bug the
  opposite way (assembly flux unchanged, velocity = −darcy_flux) with regression tests
  test_1004b including f≠0. Current development `solvers.py:804` confirms the branch's
  approach was rejected, not merely unlanded.
- **`ss2098/fix-thermal-convection-units-tutorial`** — One commit ahead (`374d1e68` "Fix
  thermal convection units tutorial", sshukla2@alaska.edu, 2026-06-19) touching only
  `docs/examples/Tutorial_Thermal_Convection_Units.py` (+238/−232). Superseded by merged PR
  #263 (development commit `4185a768`, same author, merged 2026-07-01): the merged file
  version equals the branch version plus 18 extra lines (kelvin_to_celsius helper, unit-aware
  dt cap) — `git diff` branch→`4185a768` on the tutorial file shows only additions on the
  merged side, so nothing unique remains on the branch. No worktree, no dirty files.
- **`test-darcy-rebase`** — Local rebase-test of origin/feature/darcy-richards-solvers: 7
  commits (2026-02/03) adding TransientDarcy + Richards solvers (solvers.py +465),
  `utilities/retention_curves.py` (Haverkamp/van Genuchten retention curves), tutorials
  16/17, Tracy (2006) benchmark, porous-flow.md docs, tests 1005/1006. Fully superseded by
  merge `4895c433` "Merge TransientDarcy and Richards solvers" (development, 2026-04-08):
  `git cherry` marks 6/7 commits patch-equivalent; the 7th (`aa45e0dc`) differs from merged
  `1dfa914f` by only 2 lines of conflict resolution; every branch-created file
  (retention_curves.py, both notebooks, both tests, porous-flow.md, Tracy benchmark) is
  byte-identical in current development. Remaining solvers.py delta is development having
  advanced past the branch (June docstring/descriptor-pattern refactors, Darcy sign fix #255)
  — branch holds nothing development lacks. No worktree, no dirty files.

### REMOVE_WORKTREE_ONLY — investigated: merged, worktree clean (or holding only junk)

| Branch | Worktree | Risk of loss | One-liner |
|--------|----------|--------------|-----------|
| `bugfix/fault-influence-edge` | `fault-influence-edge` | none | Finite-edge fix for Surface.influence_function, fully merged to development as PR #241 — safe to remove the worktree and delete the branch. |
| `bugfix/gamma-p1-deformed-normal` | `gamma-p1-deformed-normal` | none | Source branch of merged PR #264 (deformed-mesh normals/membership fix); 100% contained in development, worktree clean — remove worktree, delete branch. |
| `docs/snapshot-toolkit-changelog` | `in-memory-checkpoint` (misnamed) | none | Merged docs branch (PR #199) for the snapshot toolkit changelog/API docs; worktree contains only stale demo PNGs — remove worktree, branch safe to delete. |
| `feature/rotated-snes` | `rotated-snes` | none | Rotated free-slip SNES-integration branch, fully merged as PR #298 — remove clean worktree and delete branch; nothing unmerged or uncommitted. |
| `docs/ship-claude-skills` | `ship-claude-skills` | none | Ships the UW3 Claude skills in-repo — already landed verbatim via squash-merge PR #299; branch and clean worktree are pure leftovers. |
| `worktree-sl-field-carry-on-deform` | `sl-field-carry-on-deform` | none | Spent staging area for the merged coord-mutation gate + mesh.deform() work (PR #246): every staged byte already in development, untracked fault scripts superseded by newer copies in the fault-convection worktree — remove worktree and delete the merged branch. |
| `docs/snesfas-investigation` | `snesfas-spike` (misnamed) | none | Docs-only preservation of the SNESFAS/Vanka/grid-sequencing feasibility spike, already merged verbatim to development as PR #245 — safe to remove worktree and delete the branch. |
| `feature/anisotropic-metric-mover` | `winslow-mesh-smoother` (misnamed) | low | The merged (PR #209) Winslow/anisotropic mesh-smoother + OT_adapt branch; worktree dirt is May-2026 exploratory scripts plus two tracked edits already superseded in development — remove worktree, branch safe to delete. |

**Evidence (REMOVE_WORKTREE_ONLY, investigated):**

- **`bugfix/fault-influence-edge`** — One ahead commit `6291cab6` (Surface.influence_function
  finite-edge fix: +75 lines `src/underworld3/meshing/surfaces.py`, +103-line
  `tests/test_0851_surface_influence_edge.py`). Superseded by merged PR #241 (squash
  `44aff945` on development, identical title); `git cherry` marks the commit
  patch-equivalent ("−") and both files are byte-identical between branch tip and current
  development. Worktree is clean (`status --porcelain` empty).
- **`bugfix/gamma-p1-deformed-normal`** — Holds the deformed-mesh boundary-normals /
  domain-membership fix: 2 ahead commits (`8a9d2ff2` core fix, `e086e72a` Copilot-review
  follow-up) touching `discretisation_mesh.py` (+177), `petsc_generic_snes_solvers.pyx`
  (26 lines), and new tests test_0056_projected_normals_deform.py /
  test_0057_deformed_domain_membership.py. Fully superseded by merged PR #264 (development
  squash commit `51ecd292`, same title): the merge-base..tip diff md5 is byte-identical to
  the squash commit's diff, and branch-tip file trees match development's merged state
  exactly. Worktree is clean (`git status --porcelain` empty).
- **`docs/snapshot-toolkit-changelog`** — Branch holds the snapshot-toolkit documentation
  commit (CHANGES entry, current API names, toctree wiring) on top of the snapshot-disk
  feature (PR #198). It is 0 commits ahead of development and was merged via PR #199 (merge
  commit `44c2f472` on development). Worktree
  `/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/in-memory-checkpoint` is
  dirty only with 2 untracked demo render PNGs (snapshot_backstepping_demo.png,
  snapshot_backstepping_spatial.png, dated 2026-05-12/13) plus a few gitignored vep_*.png
  renders — no code or doc changes. Nothing exists only here except throwaway figures.
- **`feature/rotated-snes`** — Holds the rotated strong free-slip SNES integration: 9 commits
  ahead of development (merge-base `af537d56`), 5 files (+605/−54): utilities/rotated_bc.py,
  cython/petsc_generic_snes_solvers.pyx, systems/solvers.py,
  tests/test_1018_rotated_freeslip.py, tests/parallel/test_1064_rotated_freeslip_parallel.py.
  Fully superseded by merged PR #298 (squash commit `1d003481` on development, 2026-07-03,
  "Rotated strong free-slip inside the nonlinear Stokes solve"), whose message enumerates the
  branch's commit subjects. Content containment verified directly (`git cherry` '+' is a
  squash-merge artifact): rotated_bc.py, solvers.py, and both test files are identical
  between branch tip and development; the only .pyx difference is development-side additions
  from the later PR #297 FMG guard. Worktree `.claude/worktrees/rotated-snes` is clean
  (`git status --porcelain` empty).
- **`docs/ship-claude-skills`** — Single ahead commit `2349ed12` adds `.claude/skills/`
  (6 skills, 24 files, 9,766 lines incl. cetz-figures examples/PNGs) plus a .gitignore
  allow-list. Squash-merged to development as `aa3e383e` (#299, 2026-07-03,
  ancestor-verified); `git diff 2349ed12..development` over .claude/skills is empty (only an
  unrelated later Expt_Grains/ gitignore line differs). Worktree at
  `.claude/worktrees/ship-claude-skills` is clean (`status --porcelain` empty). Pending
  follow-up outside this repo: delete duplicate ~/.claude/skills copies now that #299 merged.
- **`worktree-sl-field-carry-on-deform`** — Branch is 0 ahead and a verified ancestor of
  development (`git merge-base --is-ancestor` passes); its intended work merged as PR #246
  (merge `865e7027`, commit `f99c8aa2` "Foolproof mesh-coordinate mutation: capability gate +
  public deform() + SL-field CARRY transfer", from bugfix/sl-field-carry-on-deform), and the
  code is live in development (`discretisation_mesh.py:3022-3092`, `solvers.py:3711-3712`).
  Worktree dirty state: 5 STAGED files whose blobs hash byte-identical to `f99c8aa2`'s
  versions (nothing unmerged), plus 2 UNTRACKED scripts (fault_convection_adapt_loop.py,
  fault_render.py) that are older June-16 snapshots of the larger June-21 copies living in
  the fault-convection worktree (newer copies add
  --fault-passive/--old-frame/--fault-base-smin/--sim-dir; sl-field copies hold nothing
  unique).
- **`docs/snesfas-investigation`** — Branch holds exactly 1 commit (`fcb9692b`, 2026-06-16):
  9 new docs-only files, +1451 lines — 3 design notes
  (docs/developer/design/snesfas-feasibility.md, snesfas-vanka-feasibility-study.md,
  multilevel-nonlinear-stokes-strategy.md) plus docs/examples/snesfas_investigation/
  (README + 5 prototype scripts). No src/ changes. Fully superseded by merged PR #245
  (development commit `34a9dd46`, 2026-06-18, identical title): `git cherry` reports the
  commit patch-equivalent ("− fcb9692b"), and `git diff development docs/snesfas-investigation`
  over all 9 files is empty — every file is byte-identical on development. Worktree
  `.claude/worktrees/snesfas-spike` is clean (`git status --porcelain` empty). The spike's
  run scripts live outside the repo (~/+Simulations/snesfas_spike/) and are unaffected.
- **`feature/anisotropic-metric-mover`** — Branch holds the anisotropic/Winslow MMPDE
  mesh-smoother rewrite (meshing/smoothing.py) plus mesh.OT_adapt() (commit `17b98a5d`) and
  the constant-nullspace guard (tip `d4e1cbf6`). Fully merged: 0 commits ahead of
  development; PR #209 merge commit `989d69ae` is in development, and the work was further
  evolved by successor branch feature/anisotropic-mover-adapt-transfer (PR #228, merge
  `59775f97`). Worktree is dirty with 112 files: ~108 are untracked throwaway
  OT-investigation scripts (scripts/_ot_*, _sl_*, plot_*, launch/watch shells). The two
  tracked modifications are superseded: (1) `discretisation_mesh.py` +28 adds
  get_mean_radius() — development already has the refined version at
  `discretisation_mesh.py:5600-5661`; (2) stagnant_lid_adapt_loop.py +270/−100 is a CLI reorg
  whose substance (mesh.OT_adapt call, mover dispatch) exists in development's evolved
  harness (script line 388, PR #228 commits `b6cbe0f1`/`40be67b7`); only cosmetic argparse
  grouping is unique, sitting on a stale pre-#228 base (blob `238dda87` never committed
  anywhere).

---

## Bulk REMOVE_WORKTREE_ONLY — merged branches with clean worktrees

The following 40 branches were pre-screened as fully merged into development with clean
worktrees (where a worktree exists). Spot verification performed 2026-07-03 on three
randomly chosen entries before writing this ledger:

- `git merge-base --is-ancestor bugfix/swarm-empty-partition-read development` → **passes** (merged)
- `git merge-base --is-ancestor feature/advdiff-theta-exposure development` → **passes** (merged)
- `git merge-base --is-ancestor bugfix/jit-c-cache development` → **passes** (merged)
- Worktrees `advdiff-estimate-dt-percentile`, `advdiff-theta-exposure`, `jit-c-cache`:
  `git status --porcelain` empty (0 dirty files) in each.

Before executing the batch, the same two checks (`merge-base --is-ancestor` + clean
`status --porcelain`) MUST be run on every entry — do not trust this list blind.

| Worktree (`.claude/worktrees/…`) | Branch |
|---|---|
| `advdiff-estimate-dt-percentile` | `bugfix/swarm-empty-partition-read` |
| `advdiff-monotone-mode-kwarg` | `feature/advdiff-monotone-mode-kwarg` |
| `advdiff-theta-exposure` | `feature/advdiff-theta-exposure` |
| `analytic-solcx-stress` | `bugfix/analytic-solcx-stress` |
| `boundary-slip-surfaces` | `feature/boundary-slip-surfaces` |
| `constant-nullspace-test` | `feature/manifest-constant-nullspace` |
| `crameri-benchmark` | `feature/crameri-benchmark` |
| `freesurface-material-advection` | `bugfix/freesurface-material-advection` |
| `global-evaluate-parallel-extrapolation` | `bugfix/global-evaluate-parallel-extrapolation` |
| `gmg-geometric-interp` | `feature/anisotropic-mover-adapt-transfer` |
| `in-cell-test-loose-semantics` | `bugfix/in-cell-test-loose-semantics` |
| `integrals-revert` | `bugfix/integrals-revert` |
| `jit-c-cache` | `bugfix/jit-c-cache` |
| `jit-constant-recompilation` | `worktree-jit-constant-recompilation` |
| `lambdify-caching-cachehit` | `bugfix/lambdify-caching-cachehit` |
| `memprobe-diagnostics` | `bugfix/memprobe-diagnostics` |
| `mesh-deform-cache-invalidation` | `feature/mesh-deform-cache-invalidation` |
| `multi-component-projection` | `feature/multi-component-projection` |
| `ns-ddt-projection-source` | `bugfix/ns-ddt-projection-source` |
| `parallel-singular-corruption` | `bugfix/parallel-singular-corruption` |
| `project-work-var-vtype` | `bugfix/project-work-var-vtype` |
| `projection-smoothing-length-v2` | `feature/projection-smoothing-length-v2` |
| `refined-submesh-pair` | `feature/refined-submesh-pair` |
| `region-ds-cell-labels` | `bugfix/region-ds-cell-labels-quarantine` |
| `sl-traceback-monotone-limiter` | `feature/sl-traceback-monotone-limiter` |
| `surface-submesh` | `feature/surface-submesh` |
| `swarm-advection-migrate` | `bugfix/swarm-advection-migrate` |
| `swarm-routed-point-eval` | `feature/swarm-mesh-adapt-transfer` |
| `ve-stokes-hang-130` | `bugfix/ve-stokes-hang-130` |
| `worktree-policy` | `docs/worktree-policy` |
| — (no worktree) | `archive/docs-legacy-quarto` |
| — (no worktree) | `bugfix/dminterp-vector-multifield` |
| — (no worktree) | `bugfix/vep-investigation-fixes` |
| — (no worktree) | `docs/maturity-gated-release` |
| — (no worktree) | `feature/constant-nullspace-test` |
| — (no worktree) | `feature/gmg-geometric-interp` |
| — (no worktree) | `feature/integrate-surface-submesh` |
| — (no worktree) | `investigate/issue-151` |
| — (no worktree) | `test-multicomp` |

---

## Execution Protocol

1. **Archive tag before any delete.** For every branch slated for deletion (ARCHIVE_DELETE,
   REMOVE_WORKTREE_ONLY, EXTRACT-after-rescue, and every bulk entry), first:
   `git tag archive/<branch-name> <branch-tip-sha>` — tags are cheap and make every deletion
   reversible. Push tags (`git push origin --tags`) so the archive survives local cleanup.
2. **Re-verify at execution time.** For each branch in a batch: confirm
   `git merge-base --is-ancestor <branch> development` (or, for squash merges, re-run the
   content-diff check recorded in the evidence above) and confirm the attached worktree is
   clean (`git -C <worktree> status --porcelain` empty). Any mismatch → pull the branch out
   of the batch and escalate.
3. **Maintainer signs off deletions per batch.** Louis Moresi approves each deletion batch
   before it runs. Suggested batching: (a) bulk REMOVE_WORKTREE_ONLY list; (b) investigated
   REMOVE_WORKTREE_ONLY; (c) ARCHIVE_DELETE with risk "none"; (d) ARCHIVE_DELETE with risk
   "low"; (e) EXTRACT branches only AFTER their rescue commits/PRs have landed and been
   verified on development.
4. **EXTRACT rescues come first.** No EXTRACT-verdict branch or worktree may be touched until
   the specific files/commits named in its evidence are committed somewhere durable (a fresh
   branch pushed to origin, or a merged PR). High-risk EXTRACT rows hold uncommitted work —
   removing the worktree destroys it unrecoverably.
5. **LAND branches are not cleanup targets.** They go through the normal PR flow
   (push to origin first for the local-only ones: feature/adapt-on-top,
   bugfix/yield-homotopy) and their worktrees are removed only after merge.
6. **KEEP_ACTIVE — do not touch:** `feature/numpy2-support` (the main-checkout work stream;
   per campaign instructions), the quality-audit branch/worktree
   (`feature/quality-audit-2026-07` at `.claude/worktrees/quality-audit-2026-07`),
   `bugfix/custom-mg-parallel`, and `feature/adaptive-convection`. Both of the latter two
   should be pushed to origin promptly as a safety measure (they contain unpushed,
   locally-unique commits) — pushing is not "touching".
7. **Worktree removal command:** use `./uw worktree remove <name>` only where the branch
   should also die; where the branch must survive (LAND before merge), use
   `git worktree remove <path>` alone. Never key on worktree directory names — six worktrees
   are misnamed relative to their branch (see System Architecture).

## Testing Instructions

This is an audit document; there is no code to test. To re-verify any row:

```bash
cd /Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/quality-audit-2026-07

# merged-by-ancestry?
git merge-base --is-ancestor <branch> development && echo merged

# merged-by-squash? (content check against the squash commit named in the evidence)
git diff <branch> <squash-sha> -- <files-named-in-evidence>

# unique commits
git log --oneline development..<branch>

# unpushed?
git log --oneline origin/<branch>..<branch>   # (or: git branch -r --contains <tip>)

# worktree dirty?
git -C /Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/<worktree> status --porcelain
```

The bulk-list spot checks recorded above were run on 2026-07-03 against development tip
`9bd6c8ee` from the audit worktree.

## Known Limitations

- **Not every branch was triaged.** Two items known from prior records were NOT investigated
  in this pass and are absent from the ledger: `bugfix/rotated-freeslip-schur-pc` (worktree
  `.claude/worktrees/rotated-freeslip-schur-pc`, clean at verification time, tip `b8e4a053`)
  and `origin/bugfix/stokes-constrained-parallel` (remote-only; prior records flag an OPEN
  release-blocking item — 0.4% 3D velocity discrepancy). Both need a follow-up triage before
  any repo-wide cleanup claims completeness. Other remote-only branches
  (origin/feature/stokes-constrained, origin/bugfix/numpy2-followup-cross-and-artifact, …)
  were out of scope.
- **State drift.** Verdicts were formed against development@`1d003481`..`9bd6c8ee` on
  2026-07-03. Any merge, push, or worktree edit after that date can invalidate a row — hence
  the mandatory re-verification step in the execution protocol. Notably, at final
  verification the main checkout was on `development` (clean) and no local
  `feature/numpy2-support` branch existed (it exists on origin); the KEEP_ACTIVE instruction
  for numpy2-support is carried from the campaign brief regardless.
- **Bulk list verified by sampling.** Only 3 of 40 bulk entries were individually re-verified
  (all passed); the remaining 37 rely on the pre-screening and MUST be re-checked at
  execution time (protocol step 2).
- **Evidence line numbers** refer to the branch/commit named in each evidence block, not
  necessarily to current development; development moves daily.
- Dirty-worktree inventories describe file-level content, not semantic completeness — an
  EXTRACT rescue should be reviewed by someone who knows the sub-project before landing.

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Maintainer | Louis Moresi | 2026-07-05 | Pending review |
| Author | Claude (audit session, Dimension 2 — branch triage) | 2026-07-03 | Complete |