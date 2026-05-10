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

## Status

Parked. The integrator-zoo characterisation is the immediate
deliverable; this architectural discussion is the natural follow-up
that's too large to fold in. The supplementary mentions the
deformable-surface module as a future destination, with a brief
"architecture sketch" pointer to this design note.
