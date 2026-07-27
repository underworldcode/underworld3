# Free-surface roadmap (July 2026)

Where the free-surface capability stands after the hardening campaign (PR #435), and
the plan for the remaining work: what the pieces are, what order they go in, and who
is best placed to do each one.

## Where we are

The 2D free surface is complete and validated for linear (isoviscous) rheology:

- The three-solve manager (`uw.systems.FreeSurface`) holds temperature bounded to at
  least 17% surface deformation, keeps the surface a material boundary to machine-level
  net flux (serial *and* parallel), and supports the full-density (Boussinesq)
  formulation in which topography carries its own restoring force — verified against
  the Cathles relaxation rate and an isostatic column referee.
- Nonlinear rheology works everywhere except one spot: the *consistent* solve's strong
  material-boundary constraint (`u·n = ũ_n`) is implemented on the linear solve path
  only. Nonlinear models run today with the weak penalty fallback
  (`consistent_constraint="penalty"`), which leaks ~4% of net volume flux through the
  surface — usable for exploration, not for production.
- 3D raises `NotImplementedError` (deliberate: without the guard it would run with
  meaningless surface machinery and be silently wrong).

## The three workstreams

### A. Land the 2D toolkit (now)

**Owner: Louis (review/merge), Claude (support).**

1. Review and merge PR #435 to `development`. Squash-merge recommended — the history
   contains a corrected mis-attribution and the squashed message carries the right
   narrative. Close #421 by hand on merge (dev-targeted PRs do not auto-close).
2. Post-merge follow-ups, in order of value:
   - Rebuild the teaching parameter table on the full-density formulation, with the
     αΔT = (thermal buoyancy)/ρ₀g ≲ 0.3 consistency constraint respected (the old
     "soft surface" parameter sets were unphysical fluids).
   - One long, checkpointed run to statistical stationarity on the corrected
     formulation — the teaching endpoint and the reference state for later work.
   - (Cheap, optional) measure how the σ_nn recovery's 2.4% residual (#431) scales
     with deformation and resolution; decide whether it ever needs fixing.

### B. Nonlinear datum + rotated-path unification (joint session)

**Owner: Louis + Claude together (agreed: joint session only). Tracking: #438, with
#403 items 2 and 4 as the primitive-level checklist.**

The single mathematical gap and a design smell, best fixed as one piece of work
because they touch the same dispatch code:

1. Carry the prescribed-normal datum through the SNES path: per Newton iteration, set
   the rotated-normal DOF of the iterate to the datum and zero its residual row — the
   machinery the `u·n = 0` nonlinear path already has, generalised to a field-valued
   datum. Includes the pyx `add_rotated_freeslip_bc` datum argument (Cython rebuild).
2. While in that code: unify the linear/nonlinear rotated dispatch (the split itself
   is the smell — one constraint, one path, linear as the ksponly special case).
3. Small follow-up: a scalar viscosity proxy for the Schur mass preconditioner under
   anisotropic (TI) viscosity in the derived solves.

Deliverable that proves it: power-law annulus free-surface convection through the
strong datum, with material-boundary error at the strong-constraint level; a TI
fault-bearing smoke test that converges. The fault-at-the-surface physics questions
(ring filter across a weak-zone trace) wait until such a model actually runs.

### C. 3D free surface (structural, delegable)

**Suggested owner: Thyagi — it builds directly on his PR #404 (3D boundary-flux
de-smear), and it is assembly work, not new mathematics.** The integrator, the datum
constraint, the deform carrier and the full-density h∞ are all dimension-free already.
The 2D-only pieces and their 3D replacements:

| 2D piece | 3D replacement | source |
|---|---|---|
| P2 line-mass de-smear for σ_nn | triangle boundary mass | PR #404 (P1 path is sound; decide whether to land P1-only while the P2 vertex-integral question rests) |
| θ-ordered ring (gather/scatter) | boundary-node gather keyed by position; **each physical node exactly once** (the #421 seam-dedup lesson carries over verbatim) | new, small |
| ring Taubin filter | existing surface Taubin smoother (already parallel-tested) or none initially | existing utility |
| arc-length datum gauge | area-weighted mean over boundary triangles | new, small |
| tangential transport | disable initially (it is an optional flag; assess need on real 3D runs) | — |
| `dim != 2` guard | capability check per piece | trivial |

Acceptance: the spherical-shell topographic relaxation benchmark (the 3D analogue of
the annulus Cathles test — decay of an imposed Y_lm topography at the analytic rate),
then a low-Ra spherical convection case with bounded T.

Sequencing note: C does not depend on B (different code areas: boundary utilities vs
pyx dispatch) and can run in parallel once #404's P1 question is decided. Both depend
on A being merged so they branch from `development`.

## Deferred, tracked, deliberately not scheduled

- #423 — *why* the old-frame reach-back amplifies (retired from the manager; the
  facility still exists for other users, so the question stays open).
- #402 — public API for the composition transport (temperature still wired through a
  private attribute; small, can ride along with any of the above).
- Fault-at-surface filter interaction — recorded in #438, revisit with a running model.

## Issue map

| issue | role |
|---|---|
| PR #435 | the 2D linear-complete toolkit (workstream A) |
| #438 | nonlinear/anisotropic tracking (workstream B) |
| #403 | rotated-datum primitive checklist (items 2, 4 feed B) |
| #404 | 3D de-smear — the seed of workstream C |
| #421 | fixed in #435; close on merge |
| #423, #431, #402 | deferred, scoped, no action required |
