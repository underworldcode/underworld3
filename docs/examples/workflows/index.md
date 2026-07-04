---
title: "Workflow Examples"
---

# Workflow Examples

Worked examples that exercise the `underworld3.workflows` package.
Each example is a small set of Python modules plus a pair of
markdown guides — one for users (running the workflow), one for
developers (extending it).

## Rayleigh-Bénard convection

A 2-D thermal convection workflow: time-loop solve to steady state,
checkpointing, sweep over `(Ra × aspect_ratio)`, warm-start
recipes, optional movie rendering.

```{toctree}
:maxdepth: 1

convection-user
convection-developer
```

## H2Ex / fault-controlled flow

A product-graph workflow — mesh adaptation, transverse-isotropic
stress, permeability, Darcy flow, surface accumulation.  Different
shape from convection (no time loop; one-shot pipeline) but
exercises the same `WorkflowConfig` / `@workflow_step` / `WorkflowRunner`
mechanics via `WorkflowProducts`.

(User and developer guides for h2ex are not yet written — see open
items in
[`convection-developer.md`](convection-developer.md#open-work).)

## Source code

The workflow modules live in this directory:

- `convection_config.py`, `convection_sweep.py`,
  `convection_visualise.py`, `convection_notebook.py`
- `simulate.py` — auto-derived single-run CLI
- `warm_start.py`, `ramp.py` — composition recipes
- `h2ex_config.py`, `h2ex_notebook.py`
