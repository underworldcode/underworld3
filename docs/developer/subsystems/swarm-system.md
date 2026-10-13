---
title: "Swarm System"
---

# Swarm System Documentation

```{warning} This page is a placeholder — do not rely on it
An earlier version of this stub described the swarm subsystem as
"well documented" and low priority. That assessment was wrong: the July 2026
quality audit found the swarm subsystem to be the densest cluster of
correctness risk in the codebase (findings SWARM-01…24), and the module it
cited (`swarm/`, "4,484 lines") never existed.

Until the modernization refactor delivers the real subsystem documentation,
the current authorities are:

- **Audit** — `docs/reviews/2026-07/SWARM-SUBSYSTEM-REVIEW.md` (repository
  file; not rendered in this documentation build). Note that many of its
  findings have since been fixed — check the status table in the design
  document before treating any finding as open.
- **Design and current-behaviour reference** —
  {doc}`../design/SWARM_MODERNIZATION_DESIGN_2026-07`, which records the
  verified data pipeline, the migration trigger matrix, the cache and
  checkpoint contracts as they stand on `development`, and the target design.

For day-to-day usage patterns (direct `.data`/`.array` access, proxy
variables, migration contexts), see {doc}`data-access` and the
`Swarm`/`SwarmVariable` docstrings, which are maintained.
```

This page will be replaced by the full subsystem documentation (pipeline,
migration trigger matrix, cache contract, checkpoint/restore contract, fossil
contract) when the swarm modernization phases land — see the phasing plan in
{doc}`../design/SWARM_MODERNIZATION_DESIGN_2026-07`.
