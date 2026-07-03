# July 2026 — Post-AI-Development Quality Campaign

**Status**: Phase 1 (audit) in progress
**Base**: `development` @ `1d003481`
**Campaign plan**: approved 2026-07-03 (L. Moresi)

## Why

June 2026 brought ~132 commits and +12.6k lines to `src/`, largely AI-assisted across
many non-overlapping sessions. This campaign audits the result against the project's
founding quality rule — *anyone should be able to read the code and understand it* —
and remediates in reviewed, tested waves.

## Audit dimensions

| # | Dimension | Review document | Status |
|---|-----------|-----------------|--------|
| 1 | Loose ends (TODOs, stubs, skipped tests, disabled logic) | `LOOSE-ENDS-AUDIT.md` | pending |
| 2 | Branch & worktree triage ledger | `BRANCH-TRIAGE-LEDGER.md` | pending |
| 3 | API consistency & convention proposal | `API-CONSISTENCY-REVIEW.md` | pending |
| 4 | Readability of change hotspots | `READABILITY-REVIEW.md` | pending |
| 5 | Swarm/particle subsystem architecture | `SWARM-SUBSYSTEM-REVIEW.md` | pending |
| 6 | Docs & standards coherence | `DOCS-STANDARDS-COHERENCE.md` | pending |

Cross-dimension synthesis and the ranked remediation worklist: `REMEDIATION-WORKLIST.md`.

## Remediation waves (post-audit, each its own PR)

- **Wave A** — deletions & dead code
- **Wave B** — internal migration off deprecated access patterns (~41 sites)
- **Wave C** — API harmonization (deprecation shims, no hard break)
- **Wave D** — readability rewrites of hotspot files
- **Wave E** — docs alignment
- **Branch-triage execution** (parallel track; deletions signed off per batch)

Follow-ons: swarm modernization refactor (design doc from dimension 5);
guardrails — UW3 Style Charter + mechanical CI gates.

## Ground rules

- Every finding carries `file:line` evidence and is adversarially verified before
  entering the worklist.
- Every remediation PR cites the finding(s) it resolves.
- `petsc_generic_snes_solvers.pyx`: naming/docs/dead-code changes only — no numerics
  without separate benchmarking.
- Gate per wave: `tier_a` tests green pre/post; `tier_a or tier_b` before merge;
  parallel (np2/np4) tests for anything touching swarm/migration code.
