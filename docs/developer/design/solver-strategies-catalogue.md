---
title: "Solver strategies catalogue"
---

# Solver strategies — switches, dials, and when to reach for them

**Scope:** the index + picking guide for solver knowledge across
**all UW3 PDE families** — Stokes (linear, variable viscosity,
nonlinear / strain-rate-dependent, yield / viscoplastic), Darcy,
Poisson, Navier–Stokes — plus time-integration *order* for
visco-elastic (VE) and visco-elasto-plastic (VEP) problems,
boundary-treatment / pressure-space / parallel-correctness
choices, and the diagnostic tooling that supports them.

**Status:** working notes / catalogue, ahead of full documentation.
**Consult and contribute as standard PDE work** (see
`memory/feedback_solver_strategies_catalogue.md`): start here when
hitting a solver wall, and add findings back when settled. Each
entry: what it does, the mechanism, the evidence, when to reach
for it, and the caveats. The catalogue is the *aggregation point*
— individual deep findings live in sibling design notes in this
directory, linked from here.

**Current body of content (2026-05):** the adaptive-mesh + Stokes
warm-start investigation populated the catalogue with its first
batch of entries (V,P remap, `snes_atol`, cold-restart, SNES
line-search variants, GAMG anisotropy tuning, direct inner solve,
`mesh.quality()`, the error-estimator and geometric-MG design
arcs). **Intended growth:** entries for the other PDE families
(Darcy, Poisson, Navier–Stokes), time-order guidance for VE/VEP
(consolidating the existing project-memory findings on BDF
order, yield-coupling, two-Stokes split, dt-yield interactions),
variable-viscosity / viscosity-contrast pressure-space choices,
and viscoplastic flow strategies. Extend as those threads land
or as referenced project memories are touched.

The investigation's mental model: solver fragility on an adaptive
problem has several *independent* failure classes, each with its
own appropriate cure. Reaching for the wrong cure for a given
failure can give the right answer for the wrong reason and mask
the real cause — so it's worth being explicit about which cure
addresses which class.

## Failure classes — quick reference

| symptom | underlying class | indicated cure |
|---|---|---|
| Re-solving a near-solved state fails (`DIVERGED_LINE_SEARCH` from a tiny initial residual) | guess-relative-only convergence (`snes_atol` unset) | **snes_atol** absolute path |
| Warm-start from a stale guess on a just-moved mesh fails | V,P not remapped across the mesh move | **V,P remap** (mirror T) |
| Warm-start fails through a violent transient *despite* a fresh, correct previous solution | inner KSP gives an inexact Newton step that `bt` line search rejects on an anisotropic operator | **accurate inner solve** (best PC for the operator), or **bypass the line search** (l2 / direct) |
| Failures recur in same-mesh bursts after a single failure | corrupted V,P propagates as next warm start | **cold-restart fallback** |
| Adaptation degrades element regularity → AMG aggregation degrades | mesh-quality side of the coupled mesh⇄solver problem | **mesh.quality()** monitoring + a less-aggressive grading dial (the equidist `resolution_ratio` is the user-facing one; legacy `coarsen_cap` / `aniso_cap` are demoted overrides) |
| Refinement bunches even where it isn't needed; can't say "add nodes" | percentile metric is relative-not-absolute | **error-estimator-driven metric** (design arc) |

## Diagnostics (harness)

The harness (`scripts/adaptive_saturation.py`) carries the flags
the investigation accumulated; they belong as durable diagnostic
tooling, not just one-off probes:

* `--snes-debug` — after each adv/Stokes solve, query
  `snes.getConvergedReason()` + `getIterationNumber()` and tag
  which physics solver diverged + reason code + iter count.
  Replaces the solver-anonymous PETSc retry message. **Does not**
  set global PETSc viewers (they leak into the mover's `ksponly`
  sub-solves and spam phantom `DIVERGED_MAX_IT iterations 0`).
* `--resume-from N` + `--src-tag SRC` — restart from a specific
  checkpoint of another model, write outputs under the current
  `--model` tag. Enables the *clean-restart probe* pattern: a
  reproducible failure window from a known state without re-running
  the entire trajectory.
* `--stokes-cold-recover N` — see "cold-restart fallback" below.
* `--no-vp-remap` — A/B disable the V,P remap; see "V,P remap".
* `--stokes-snes-opt {default,basic,l2,tr,ksponly,direct,
  gamg-n1,gamg-thr,gamg-noagr,gamg-sor,gamg-full,...}` — selects a
  preset bundle of PETSc options on the Stokes solver; see the
  SNES-line-search and GAMG sections.
* `--stokes-snes-atol-auto` — captures cold ‖F₀‖ and sets a fixed
  `snes_atol`; see "snes_atol" / `snes-atol-convergence-scale.md`.

The **`mesh.quality()` API** + the `view()` summary line is the
mesh-side diagnostic — shape quality `q = 4√3·A/Σℓ²` (min /
percentiles), max interior angle, aspect ratio, neighbour
size-jump, and the joint "large-AND-stretched" cell count. The
relevant *tail* metrics for FE conditioning are what `minA/meanA`
hides.

## V,P remap on mesh move

**Class:** correctness, esp. for *nonlinear* solves where Newton
has a small convergence basin.

**What:** when the mesh moves, evaluate the previous V,P at the
new DOF coords (the same FE-evaluate-at-new-coords that T already
gets) and write the results onto the new mesh — *do not* leave the
old nodal values on moved nodes.

**Evidence:** without it, warm-start at adapt steps takes a
spatially-scrambled guess → `DIVERGED_LINE_SEARCH` on adapt steps
specifically (a16r15v cleared all adapt-step failures by remapping;
24 → 24 (no adapt) vs failures concentrated on non-adapt steps).

**Status:** implemented in the harness adapt functions
(`adapt_local_fe_interp`, `adapt_pristine`). **Production version
belongs in UW3's adaptation/deform path** (gated). The pristine /
local subtlety: V,P live in `X_prev` geometry (not the pristine
X0c geometry that T transfers through) — that asymmetry matters.

## `snes_atol` — guess-independent convergence

**Class:** near-converged-guess re-solve (steady-state continuation,
restarts, lightly-evolving). The PETSc default `snes_atol ~ 1e-50`
makes the absolute convergence path effectively dead; UW3 sets
`snes_rtol` but not `snes_atol` ⇒ only the guess-relative
`rtol·‖F(x₀)‖` criterion is live.

**What:** set `snes_atol` to the problem's natural residual scale
(e.g. `rtol · ‖F(x=0)‖_current`, recomputed per warm solve,
temporarily applied and restored), so `SNES_CONVERGED_FNORM_ABS`
fires at it==0 when the warm guess is already good — zero Newton
iterations.

**Evidence:** PETSc 3.25 `SNESConvergedDefault` source-verified;
confirmation experiment showed exactly the predicted behaviour
(works for the near-converged class; *did not* fix the violent-
transient class — that's a different mechanism).

**Status:** design note in `snes-atol-convergence-scale.md`,
gated on sign-off + benchmarking. Internal/automatic, **no user
API** (user never sets `atol` directly).

## Cold-restart fallback

**Class:** operational safety net for any divergence that survives
the other fixes (genuine nonlinear divergence from a bad guess
where no line-search config rescues; or the transient case
described next, before the inner-solve fix is in place).

**What:** on a Stokes `DIVERGED_LINE_SEARCH` (or any negative
reason), discard the (now corrupted) warm V,P and re-solve cold
(`zero_init_guess=True`) on the *same mesh, same T* before
advancing. Warm-first, cold-on-failure — standard robust
nonlinear-solver practice.

**Evidence:** harness `--stokes-cold-recover N`. a16r15r:
31/31 recoveries succeeded; the run settled cleanly. Important
nuance: in a violent transient, cold-restart fires on *runs* of
consecutive steps (not isolated events) — every step warm-fails
because the previous step's true solution is itself a poor Newton
start for the next step. Cold-restart guarantees correctness, but
in the violent transient regime is *not* cheap (one cold solve
per step in the danger window).

**Status:** harness flag; production = port to UW3's SNES
solve() path.

## SNES line search / type variants

| `--stokes-snes-opt` | mechanism | takeaway |
|---|---|---|
| `default` (`newtonls`+`bt`) | full backtracking | the existing default; brittle to inexact Newton steps |
| `basic` | full step, no backtracking | works on *linear* problems; **removes globalisation → unsafe nonlinear** — diagnostic only |
| `l2` | minimises ‖F‖ along the Newton direction | clean *general* line-search variant (legitimate fallback), but **slow** (extra residual evaluations); fixes the bt-rejection symptom, not the cause |
| `tr` (`newtontr`) | trust region | **hopeless on the Stokes saddle point** (indefinite Jacobian, TR quadratic model ill-posed); 98 fails at *step 1* — do not use |
| `ksponly` | one linear KSP solve, no Newton/line-search | works only because Stokes is *linear* here; **invalid for nonlinear rheology** |
| `direct` | MUMPS LU on the full Stokes Jacobian | exact inner solve; 24→0 warm divergences. **Gold standard at small/2D scale; not feasible at scale** |

The cleanest pattern (from the GAMG sweep, see below): **none of
these is the production cure.** The principled fix is to make the
*existing* default `newtonls`+`bt` work, by giving it an *accurate
enough Newton step* — i.e., fix the inner KSP/PC, not the outer
line search.

## GAMG anisotropy tuning

**Class:** AMG aggregation defaults degrade on anisotropic
operators (stretched / graded cells from adaptive refinement),
producing aggregates that span the weak direction. The inner KSP
under-converges the Newton correction; `bt` line search rejects
the step; SNES reports `DIVERGED_LINE_SEARCH`.

UW3's default Stokes PC is **GAMG (aggregation AMG)** with
`pc_gamg_type=agg`, `pc_gamg_agg_nsmooths=2` (PETSc default is 1),
`pc_mg_type=additive`. Smoother defaults: Chebyshev + Jacobi.

**CRITICAL — option scope (corrected 2026-05-20).** UW3 Stokes
nests its GAMG inside the velocity Schur sub-block at prefix
``fieldsplit_velocity_pc_gamg_*`` (see
``cython/petsc_generic_snes_solvers.pyx`` ~L4199-4205). Setting
``pc_gamg_*`` at the bare/global scope ⇒ silent no-op — PETSc
reads the option key at the velocity sub-block prefix and never
inherits from the bare prefix. Verified bit-identical KSP
residuals to default on a static one-shot probe
(``scripts/_sl_preset_verify.py``), and bit-identical warm-fail
signature to default on the dynamic 40-step probe (both gave
4 fails at steps 61-64 with iter counts [4,6,4,1] —
indistinguishable).

The **earlier GAMG sweep in this catalogue used the WRONG
scope** and therefore "validated" a string of no-op presets
against each other. Re-run with the correct
``fieldsplit_velocity_pc_gamg_*`` prefix gives a very different
table — including one preset that **actively breaks** the
solver:

**Corrected sweep (restart-from-50 testbed, 40 steps, baseline
4 warm DIVERGED at steps 61–64, all options at the proper
``fieldsplit_velocity_*`` prefix):**

| `--stokes-snes-opt` | option(s) (at `fieldsplit_velocity_*` prefix) | warm fails | mechanism |
|---|---|---|---|
| `gamg-n1-corr` | `pc_gamg_agg_nsmooths=1` (PETSc default) | **0** ✓ | revert UW3's `=2` override; smoothed aggregates of degree 2 on graded mesh hurt |
| `gamg-thr-corr` | `pc_gamg_threshold=0.02`, `threshold_scale=0.5` | **23** ✗ DANGEROUS | aggressive thresholding prunes the weak-direction connections AMG actually needs on adapted velocity operator — *worse* than default |
| `gamg-noagr-corr` | `pc_gamg_aggressive_coarsening=0` | **0** ✓ | suppress finest-level MIS-2 aggressive coarsening |
| `gamg-sor-corr` | `mg_levels_ksp_type=richardson`, `pc_type=sor`, `ksp_max_it=2` | **0** ✓ | stronger smoother absorbs sub-optimal aggregates |
| `gamg-full-corr` | combined | **0** ✓ | no improvement over single fixes |
| `gamg-noagrsor-corr` | noagr + sor | **0** ✓ | no improvement over either alone |

**Findings:**

1. Five of six correct-scope variants close the failure window
   independently. They produce indistinguishable wall times
   (≈5 min for 40 steps at res-16) → no clear performance winner
   on this small problem. Any of them can serve as the
   surgical fix.
2. **`gamg-thr-corr` is dangerous** — 23 fails vs 4 baseline.
   The threshold+threshold_scale pair at the velocity sub-block
   removes structure GAMG needs. Do not use. (Was silently a
   no-op at the wrong scope, masking this danger.)
3. The mechanistic story (Cheb+Jac × poor aggregates →
   divergence; fix either side and it works) survives — the
   evidence base just shrunk to noagr/n1/sor/full/noagrsor.

**Recommended UW3 default change (corrected):**
``fieldsplit_velocity_pc_gamg_aggressive_coarsening = 0`` on the
Stokes solver. Single integer; surgical; closes the failure
window; preserves Cheb+Jac for HPC parallel scalability. **Note
the scope** — bare ``pc_gamg_aggressive_coarsening = 0`` does
nothing.

**Verification methodology (mandatory for future GAMG-tuning
claims):** before claiming a tuning helps, verify the option is
actually applied to the GAMG instance it targets. Static probe:
run the SAME problem twice with and without the option, on a
fixed T snapshot, with ``snes_monitor`` and ``ksp_monitor``
enabled. If the KSP residual values are bit-identical between
the two runs, the option is a no-op (wrong scope) and any
"benefit" elsewhere is illusory. See
``scripts/_sl_preset_verify.py`` for the verification harness.

**Caveats:**
- The 40-step restart probe is a narrow window (4 failure
  opportunities). Closing it does *not* prove a candidate
  survives a full settled trajectory or harder problems.
- These tests are on a *simple* PDE (constant-viscosity
  Stokes, T-fixed buoyancy). The story may differ with
  nonlinear rheology / yield / temperature- and strain-rate-
  dependent viscosity. The next stress test is the harder
  PDE family, not more aggressive Ra=1e6 of the same simple
  problem.

## Direct inner solve (MUMPS LU)

**Class:** the gold-standard *demonstration* of the
"accurate-inner-Newton-step → bt accepts λ=1 → robust" mechanism.
At small/2D scale (e.g. res-16 annulus), MUMPS LU on the full
Stokes Jacobian is cheap and exact.

**What:** `pc_type=lu`, `ksp_type=preonly`,
`pc_factor_mat_solver_type=mumps`, `mat_mumps_icntl_24=1`.

**Evidence:** a16r15d (warm, default `bt`, no recover) → 0 warm
DIVERGED (vs 24 baseline). The cleanest single-experiment proof
that the failure is inner-step accuracy, not the outer solver
type.

**Status:** keep as a diagnostic / sanity tool. Generalise as
"solve the inner Newton correction accurately on the adapted
operator" — implemented in production via tight KSP or strong PC
(see GAMG-tuning above), *not* by always-direct.

## Error-estimator-driven metric (design arc)

**Class:** the absolute, resolution-aware refinement criterion —
the principled successor to the percentile metric. The
percentile is purely relative (always bunches the top X% of
*whatever* distribution; can't say "this needs more nodes than
redistribution can give"; can't recognise "the uniform mesh is
already fine"). This is the *adaptation analogue* of the missing
`snes_atol`: in both cases the fix is "judge against the problem,
not the distribution."

**Routes:**
- *(a) Recovery-based (ZZ) — cheap first cut:* recovered ∇u minus
  FE ∇u as a per-cell error indicator. Reuses the existing
  projected-gradient machinery; no hierarchy needed.
- *(b) Hierarchical / τ two-grid estimator (richer):* leverage
  UW3's `dm_hierarchy` for both the error estimator *and* a
  **geometric multigrid preconditioner** that sidesteps
  AMG-anisotropy entirely. Two birds from one structure.

**Status:** scoped, not started. To be written up as a design
note (cf. `snes-atol-convergence-scale.md`) before implementation.

## Geometric MG via `dm_hierarchy`

**Class:** the alternative to AMG that is *inherently*
anisotropy-robust (the hierarchy is built geometrically, not from
the operator's connection graph).

**Status:** **Landed.** A mesh built with `refinement >= 1` carries a
`dm_hierarchy`, and the solvers now switch to geometric Full Multigrid
on it automatically. The user-facing control is the `preconditioner`
property (`"auto"` | `"fmg"` | `"gamg"`) on the Stokes / scalar / vector
solvers; `"auto"` (the default) selects FMG when a hierarchy is present
and falls back to GAMG otherwise.

Implementation: `SolverBaseClass._apply_preconditioner_options()` in
`petsc_generic_snes_solvers.pyx`, invoked from `_build` so `"auto"`
re-resolves against the current mesh (a true remesh collapses the
hierarchy → automatic GAMG fallback). The auto path is deliberately
conservative — it only *adds* geometric MG on top of an untouched
default and never rewrites pc options a solver/user configured directly
(e.g. the tuned GAMG in the OT/φ-Poisson smoother). User guide:
`docs/advanced/multigrid-preconditioning.md`. Tests:
`tests/test_1014_stokes_multigrid.py`.

Benchmark-validated on a deforming adaptive mesh (annulus res32, R=8,
mode-1, np=5, MMPDE mover, 50 steps): the 3-level hierarchy survives
every step and the inner velocity-block KSP stays flat at ~5 iters under
FMG where GAMG is a volatile ~64-131 (~23×) without cliffing at this
anisotropy; wall-clock gap only ~1.8× (the cold-start Stokes solve, common
to both, dominates the time). The value is predictability/mesh-independence,
not raw speed. Figure + data in `docs/advanced/multigrid-preconditioning.md`.

Still pairs naturally with the error-estimator design arc — the same
multi-level structure yields both the anisotropy-robust PC and the
absolute error indicator.

## Mesh-quality / `mesh.quality()` API

**Class:** the diagnostic on the *mesh* side of the coupled
mesh⇄solver problem.

**What:** `mesh.quality()` returns per-mesh aggregate + tail
metrics — shape quality `q = 4√3·A/Σℓ²` (min, percentiles,
mean), max interior angle, aspect ratio (max, p99), neighbour
size-jump, joint "large-AND-stretched" count, plus the dimension-
agnostic `vol_min_over_mean`. `mesh.view()` prints a one-line
summary with a hazard flag for `q<0.2` cells.

**Why it matters here:** bulk `minA/meanA` hid the equidist mover's
poor-cell problem; the tail metrics exposed it. AMG aggregation
degrades on poor cells (the GAMG anisotropy section above) —
mesh-quality monitoring is therefore not aesthetic, it directly
predicts solver robustness.

## Failure-class → strategy map (the picking guide)

```
Symptom                           First-line cure         Backup
--------------------------------  ----------------------  ----------------------
"Re-solve = no fewer iterations"  snes_atol               cold-restart
than a fresh solve                                        
                                                          
Warm-start fails at adapt step    V,P remap               cold-restart
                                                          
Warm-start fails in violent       Accurate inner          cold-restart
transient (non-adapt)             solve (GAMG tuning /    + l2 (slow but safe)
                                  direct at small scale)  
                                                          
Adaptive metric bunches a smooth  (design arc) error-     reduce R / use coarsen
solution / can't signal "more     estimator metric        cap; mesh.quality()
nodes needed"                                             monitors regularity
                                                          
AMG diverges on adapted mesh      pc_gamg_aggressive_     gamg-thr; gamg-sor
                                  coarsening=0            geometric MG (long-term)
```

## Open follow-ups

- Fresh full-settled validation of `pc_gamg_aggressive_coarsening=0`
  alone on a16r15-equivalent (verify 24→0 on the full trajectory,
  not just the 40-step probe).
- Combined `gamg-noagrsor` discriminator run (in flight as of the
  catalogue's first draft).
- Test the strategies on a harder PDE family (nonlinear /
  temperature- or strain-rate-dependent viscosity, yield) — the
  current evidence is on simple Stokes only.
- Design notes: error-estimator metric; geometric-MG via
  `dm_hierarchy`.
- Port the harness-side fixes (V,P remap, cold-restart) into the
  UW3 core (adaptation/deform path + SNES `solve()`).

## Related artefacts

- `docs/developer/design/snes-atol-convergence-scale.md` —
  full design note for the snes_atol fix.
- `docs/developer/design/mesh-adaptation-formulation.md` —
  the equidistribution mover formulation + single-knob
  `resolution_ratio` API.
- `scripts/adaptive_saturation.py` — the diagnostic harness
  (the flags listed under "Diagnostics" above).
- `scripts/_cellquality.py`, `_dial_quality_compare.py`,
  `_pctl_parallel_check.py`, `_equidist_probe.py` — focused
  validation / sweep scripts kept for reproducibility.
