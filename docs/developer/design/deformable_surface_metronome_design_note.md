# Deformable surface module: timekeeper / metronome architecture

Design note from the integrator-zoo characterisation session
(2026-05-10 / 11). The substantive work is the supplementary in
`publications/free-surface-paper/integrator_zoo_supplementary.md`;
this note captures the architectural discussion that emerged at the
end and parks it for a dedicated future session.

## Context

The integrator-zoo work characterised several time-integration
schemes for a single physics subsystem (free-surface kinematics in
a deforming mesh): FE-SL, RK2-SL, RK4-SL, AB2-SL, BDF2-SL plus
the surface-load shortcut variants. RK4-SL with the relaxation
CFL is the production-quality scheme. The load-shortcut variants
are cheaper per step but regime-restricted and *single-physics-cohort*.

The conversation generalised toward a model-wide time-integration
abstraction — one centralised "metronome" / `Model.advance(Δt)`
call that orchestrates all registered time-evolving objects in a
coordinated way. The integrator-zoo work is the prototype of what
one cohort within that framework would look like.

## Core abstractions (sketch)

1. **Time-evolving object** — uniform interface for any field that
   participates in time integration:
   ```
   save_state() -> token
   restore_state(token)
   apply_increment(stage_dt, rate)
   suggest_dt(state) -> float | None
   rate_at(state) -> rate
   ```

2. **Stepper** — knows a time-integration scheme (RK2, RK4, BDF1,
   BDF2, AB2, ...). Composes save/restore/advance calls on cohort
   members to advance them all by Δt according to its scheme.

3. **Cohort** — a group of time-evolving objects that share
   integrity invariants and must be advanced in lockstep. Single
   Stepper per cohort. Inter-cohort coupling is operator-split.

4. **Metronome / TimeKeeper** — top-level orchestrator. Sequentially
   advances cohorts. Aggregates `suggest_dt` across cohorts. Runs
   end-of-step hooks (migration, remeshing, checkpointing) after
   all cohorts have committed.

## Cohort = integrity-invariant group, not accuracy preference

A cohort is determined by *correctness*, not numerical accuracy.
Mesh + swarm must be in one cohort because:
- Swarm particles must remain inside the deforming mesh (geometric).
- Property projection at Stokes integration points uses swarm data;
  desync produces silently-wrong velocities even when no particles
  are lost (correctness).

The framework cannot eliminate ALE-mesh-vs-Lagrangian-swarm drift
(it's structural to the formulation), but it ensures the drift is
visible via per-stage `assert_invariants()` checks and recoverable
via user-declared policies (rollback, re-seed, accept loss).

## Composition policy (recommended for v1)

**Policy A** — uniform Stepper inside a cohort; operator-split between
cohorts. Documented in detail in the supplementary's "Architecture
sketch" section. Future versions could relax to Policy B (lockstep
stages with per-variable subscription), but the API must be designed
not to *prevent* the upgrade.

The single primitive that makes the upgrade possible:
**`apply_increment(stage_dt, rate)` must be commutative-with-itself
across stages of the same step.** Apply (Δt/2, k₁), rewind, apply
(Δt/2, k₂) must give a consistent state. The mesh `_deform_mesh`
primitive satisfies this; the swarm position update does too. New
fields joining the framework must satisfy it explicitly.

## Two points to revisit in the dedicated session

These came up at the end of the integrator-zoo session and warrant
careful handling in the metronome design:

### 1. Stale-marking, not push-updating

The current direction in the codebase is for objects to *mark
dependants stale* rather than directly *trigger updates*. The
metronome / TimeKeeper is the right place to schedule the
recomputation of stale objects in a coordinated, dependency-ordered
way. This decouples "who needs to know about a change" from "when
does the actual recomputation happen".

The `.data` cache validation (`id(self._lvec)` tracking) is already
a working example of this pattern. Extending it to a model-wide
dependency graph is the natural generalisation. The Stepper inside
a cohort would mark mesh-state-dependent caches stale on each
apply_increment; the next read triggers the recomputation.

### 2. RK stages create "hidden" intermediate states with side-effects

Multi-stage schemes (RK2, RK4) rewind and re-evaluate during the
metronome's single Δt tick. These intermediate states are *off the
master clock* — they're transient configurations the metronome
exposes only to the cohort's Stepper. Any downstream machinery
registered against the affected objects (cache invalidation, JIT
recompilation, swarm migration, property re-projection) will fire
once per stage if naively connected.

This is a real architectural cost that didn't show up in the
per-Stokes-solve cost analysis of the integrator zoo. It's also
a separate argument *for* the load-shortcut variants beyond the
numerical/cost story: the load shortcut isolates between-stage
change to a single Neumann BC value, leaving the broader
dependency graph (mesh state, swarm membership, JIT-tied caches)
untouched. Even where the load shortcut's regime is borderline,
this freedom-from-side-effects can be the deciding factor for
problems with expensive dependent recomputations.

The dedicated session needs to think carefully about whether the
metronome / cohort framework should:
- Suppress side-effect firing during stages (only commit at end of
  cohort step), then have a final "all stale things now recompute"
  pass — efficient but requires every cache to be deferrable.
- Fire eagerly per stage — simpler semantics but expensive.
- Distinguish "between-stage" vs "end-of-step" stale, allowing
  caches to declare which they care about.

The third option is probably right but adds complexity.

## Other items for the dedicated session

- How does `Model.advance(Δt)` interact with adaptive Δt (Stepper
  signals retry-with-smaller-dt)? Per-cohort retry, or global?
- Integrity-invariant recovery policies (rollback / re-seed /
  accept loss): declared per-cohort or globally? Where does the
  user-facing API surface this?
- Default cohorts and default Steppers — what does the user who
  doesn't think about cohorts see? Probably: one cohort containing
  all DDt-tracked variables, with a BDF1 Stepper. Free-surface
  solver installation auto-creates a free-surface cohort with an
  RK4 Stepper if surface-relevant variables are registered.
- Backwards compatibility with the existing per-variable DDt
  history mechanism. A "cohort of one with BDF Stepper" should be
  behaviour-identical to today's DDt.
- How AB2's history requirement composes with the cohort framework
  (history is per-Stepper, not per-variable; cohort-of-one for AB2
  is fine, but multi-variable AB2 cohorts need shared history state).
- Failure-mode tests: a deforming-mesh + swarm + VEP combination
  where the operator-split between cohorts creates the largest
  commutator error. Need a regression benchmark.

## Update from 2026-05-11 design session

This session scoped a v1 FreeSurface module and turned up two
architectural conclusions that should drive the next session.

**Conclusion 1: there is no useful single-effective-step scheme.** The
integrator-zoo characterisation showed that FE-SL at the Δt where it's
accurate enough is structurally as expensive as the multi-stage schemes
(~10× safe Δt of RK4-SL c=1 → ~10× the step count); all other schemes
are multi-stage, multi-step, implicit, biased, or unstable.
"Use FE-SL when you have swarms" is not a viable fallback.

**Conclusion 2: multi-stage SL with property-carrying swarms is
correctness-unsafe without cohort co-evolution.** The load-shortcut
empirical data (4× over-damping in pure relaxation; supplementary
"Cathles relaxation-benchmark results") proves that "small" geometric
inconsistencies between mesh and rate-source state do not degrade
gracefully. The same logic applies if a swarm carrying body-force
properties is held frozen while the mesh deforms through RK stages.

**Implication for v1 scope: two viable paths, decision pending.**

- (a) **Constrained v1.** `FreeSurface` ships with a clear refusal for
  any swarm-carried body-force source. Serves the relaxation /
  forced-equilibrium / mesh-variable-density audience (thermal
  convection, isostasy without varying materials) but excludes
  tectonics. Ships fast — days of work, no architectural prerequisites.
- (b) **Cohort framework v1.** Full Stepper / Cohort / Model machinery
  built so multi-stage SL + swarm work correctly. Bigger investment;
  resulting `FreeSurface` serves the broader audience including
  tectonics. Gated on the snapshot toolkit (see spin-off below) and
  on the cohort framework being built on top of it.

The decision between (a) and (b) depends on who needs `FreeSurface`
most urgently. Open for the next session.

## Spin-off: snapshot toolkit

The save/restore primitive needed by the cohort framework — and
identified in this session as a general UW3 capability with multiple
consumers (backtrack-on-failure, adaptive Δt, predictor-corrector,
crash recovery, bisection) — was spun off as a separate activity:

- Worktree: `feature/in-memory-checkpoint`
- Design note: `docs/developer/design/in_memory_checkpoint_design.md`
- Scope: ~four weeks; in-memory + on-disk-full-state backends sharing a
  serialisation contract for both PETScSection state and Python-side
  solver-internal state (state-as-dataclass pattern for new code,
  retrofit for existing classes including DDt)
- Status: design alignment complete; awaiting decision to commit +
  start implementation

The snapshot toolkit is a hard prerequisite for cohort-framework v1
(path b) but irrelevant to constrained v1 (path a).

**Status of the two original "open issues" from earlier in this note:**

- *Stale-marking vs push-updating* — partially resolved by the
  snapshot toolkit's mesh-change-event vocabulary (separate `deform`,
  `snapshot`, `restore` events with different propagation rules). The
  remaining work is wiring this into the cohort framework when it's
  built.
- *RK stages with hidden side-effects* — same status. The
  within-process snapshot/restore primitive plus forward-going deform
  events give the right vocabulary; per-stage side-effect classification
  (substage-visible vs step-visible) lives at the cohort level.

## Status

Architectural design captured (sessions 2026-05-10 / 11). Two viable
v1 paths identified (constrained-no-swarm vs cohort-framework), pending
decision on which audience to serve first. Snapshot toolkit spun off as
a separate, prerequisite activity (`feature/in-memory-checkpoint`
worktree). The cohort/Stepper framework itself remains parked; will be
revisited once the snapshot toolkit lands and a v1 scope is chosen.
