---
title: "Convection Workflow — Developer Guide"
---

# Convection Workflow — Developer Guide

How the Rayleigh-Bénard convection workflow is built on top of
`underworld3.workflows`.  Companion to the
[user guide](convection-user.md), which covers what the workflow
*does* rather than how it's wired.

The convection workflow is the *worked-example consumer* of the
`underworld3.workflows` package.  Reading this guide alongside the
source files (~1500 lines, listed below) shows how each abstraction
in `uw.workflows` lands in a real workflow.  A new workflow author
would copy this directory and rewrite the physics; the surrounding
mechanics — config class, step decorators, runner cascade, recipes,
CLI driver — stay essentially the same.

## Module map

| File | What lives there |
|------|------------------|
| `convection_config.py` | `ConvectionConfig` (Pydantic), the workflow-step DAG (`create_mesh` → `create_solvers` → `evolve` → `summarise_run`), diagnostics, steady-state test |
| `convection_sweep.py` | `SweepConfig` and the `(rayleigh × aspect_ratio)` aggregation cascade (`run_sweep` → `tabulate_*` → `plot_*`) |
| `convection_visualise.py` | Frame rendering and movie encoding (not yet a workflow_step — see "Open work" below) |
| `simulate.py` | CLI driver, auto-derived from `ConvectionConfig` via `cli_from_config` |
| `warm_start.py` | Recipe: project T from a prior run onto a new mesh / degree as a seeded IC |
| `ramp.py` | Recipe: chain warm-starts through a sequence of Ra values |

## Workflow DAGs

### Per-case workflow (`convection_config`)

A "case" is one model run: one `ConvectionConfig` instance, one
output directory, one steady-state termination.  In the sweep
context, every (rayleigh, aspect_ratio) combination is a case.

```
predict_bl_thickness ──▶ bl_thickness ──┐
predict_mean_T ──────▶ mean_T_predicted │
                                        │
create_mesh ─▶ mesh ──┬──▶ create_solvers ──▶ stokes, adv_diff, T, v, p
                      │                                    │
                      └────────────────────────────────────┤
                                                           ▼
                                                        evolve ──▶ run_directory, evolution_log
                                                                              │
                                                                              ▼
                                                                       summarise_run ──▶ run_summary
```

Auto-generated DOT source from `runner.diagram()`:

```python
from underworld3.workflows import diagram
import convection_config as cc
print(diagram(cc))
# → render with `dot -Tpng - -o convection_dag.png`
```

### Sweep workflow (`convection_sweep`)

```
run_sweep ──▶ all_cells_completed ──┬──▶ tabulate_nu_vs_ra ──▶ nu_vs_ra_csv ──▶ plot_nu_vs_ra ──▶ nu_vs_ra_plot
                                    │
                                    └──▶ tabulate_vrms_vs_ra ──▶ vrms_vs_ra_csv ──▶ plot_vrms_vs_ra ──▶ vrms_vs_ra_plot
```

`run_sweep` is the bridge between the two workflows.  This
particular sweep grid is `(rayleigh × aspect_ratio)` because those
are the diagnostics canonically swept in Rayleigh-Bénard work; the
framework imposes no specific axes.  For each case in the grid,
`run_sweep` instantiates a nested
`WorkflowRunner(convection_config, case_config, products=...)` and
calls `runner.build("run_summary")`.  Each case's products land in
its own `<case_dir>/products/manifest.yaml`; the sweep's outer
products (CSV tables, plots) land in `<output_dir>/products/manifest.yaml`.

Note: the product name `all_cells_completed` and the helper
`_cell_key()` in the source use "cell" in the spreadsheet sense
(one entry in the parameter grid).  Prose in this doc uses "case"
to avoid collision with "convection cell" in the physics literature.

## Step reference

### Per-case steps (in `convection_config`)

| Step | Produces | Requires | Notes |
|------|----------|----------|-------|
| `predict_bl_thickness` | `bl_thickness` | — | high-Ra 2D scaling, `δ ≈ 1/(2·Nu_pred)`, `Nu_pred ≈ 0.27·Ra^(1/3)` |
| `predict_mean_T` | `mean_T_predicted` | — | `0.5·(T_top + T_bottom)` |
| `create_mesh` | `mesh` | — | builds an `UnstructuredSimplexBox`, applies `restart_policy` on identity-hash mismatch |
| `create_solvers` | `stokes, adv_diff, T, v, p` | `mesh` | Taylor-Hood (v=2, p=1), T at `config.T_degree`, free-slip walls, isothermal top/bottom, multigrid PETSc options |
| `evolve` | `run_directory, evolution_log` | `mesh, stokes, adv_diff, T, v, bl_thickness` | idempotent: short-circuits on `run_summary.yaml` `status="steady"`, otherwise resumes from latest h5 step or starts fresh |
| `summarise_run` | `run_summary` | `run_directory` | averages over the steady window, writes `run_summary.yaml` only when `status="steady"` |

### Sweep steps (in `convection_sweep`)

| Step | Produces | Requires | Notes |
|------|----------|----------|-------|
| `run_sweep` | `all_cells_completed` | — | iterates `(rayleigh × aspect_ratio)`, runs each case to steady state via a nested per-case runner.  Product name retains the spreadsheet-cell convention. |
| `tabulate_nu_vs_ra` | `nu_vs_ra_csv` | `all_cells_completed` | tidy CSV columns: `aspect, Ra, status, n_steps, Nu_mean, Nu_std, Nu_top_mean, Nu_bot_mean` |
| `tabulate_vrms_vs_ra` | `vrms_vs_ra_csv` | `all_cells_completed` | same shape for Vrms |
| `plot_nu_vs_ra` | `nu_vs_ra_plot` | `nu_vs_ra_csv, all_cells_completed` | log-log Nu(Ra), one curve per aspect_ratio, with `0.27·Ra^(1/3)` reference |
| `plot_vrms_vs_ra` | `vrms_vs_ra_plot` | `vrms_vs_ra_csv, all_cells_completed` | log-log Vrms(Ra), one curve per aspect_ratio |

## Config classes

### `ConvectionConfig` — per-case parameters

Identity fields (declared in `_identity_fields`, fold into `cache_key`):

`aspect_ratio, cellsize, qdegree, regular, T_degree, rayleigh,
viscosity, diffusivity, T_top, T_bottom`

The non-dimensional Boussinesq formulation pins `viscosity = 1`,
`diffusivity = 1`, `T_top = 0`, `T_bottom = 1` — Ra absorbs the
viscous + thermal scales, and the boundary temperatures set the
temperature contrast.  These four fields stay in the config (so a
developer who really wants to tweak them — e.g. shift the T scale
to `[-0.5, 0.5]` to mask AdvDiff drift, per the in-code comment) and
in `_identity_fields` (so the tweak invalidates the cache).
`simulate.py` suppresses them from the CLI's `--help` so they don't
present as user knobs; the user-facing doc table omits them.

`qdegree` is also pinned by a `@model_validator(mode='after')` —
it's structurally a function of the highest variable degree
(currently `T_degree`, since v=2 and p=1 are dominated by T at
default `T_degree=3`).  Even if a caller passes `qdegree` to the
constructor, the validator overrides it to `max(2, T_degree)`.
Field stays in the config so it lands in the manifest snapshot;
not user-tunable.  Hidden from the CLI for the same reason as the
non-dim quartet above.

Operational fields (changeable between invocations without
invalidating cached products):

| Field | Default | Purpose |
|-------|---------|---------|
| `steady_window` | `0.3` | Fraction of timeseries used for stationarity test |
| `steady_tol_mean` | `0.02` | Tolerance on between-halves mean drift |
| `steady_tol_cv` | `0.05` | Tolerance on coefficient of variation |
| `steady_min_window` | `50` | Minimum sample count for steady test |
| `batch_steps` | `200` | (currently informational; not used in `evolve`) |
| `max_steps` | `5000` | Hard step cap |
| `save_every` | `10` | h5 checkpoint cadence |
| `dt_factor` | `2.0` | Multiplier on `adv_diff.estimate_dt()` |
| `diag_every` | `0` | Heavy-diagnostic cadence (0 = match `save_every`) |
| `clip_T_range` | `False` | Clip T to `[T_top, T_bottom]` after each AdvDiff solve (high-Ra safety) |
| `output_dir` | `"output/convection/run"` | Per-case run directory |
| `restart_policy` | `"error"` | One of `error / fresh / seed_from_old` |

Timeseries column schema: `step, t, dt, Nu_top, Nu_bot, Vrms, Vmax, mean_T`.

### `SweepConfig` — sweep grid + per-case defaults

The sweep axes are workflow-specific.  This convection workflow
sweeps over Rayleigh number and aspect ratio (`rayleigh_values`
and `aspect_ratios` lists).  Other workflows would expose
different lists.

Identity fields (the grid axes plus per-case mesh + physics):

`rayleigh_values, aspect_ratios, cellsize, qdegree, regular,
viscosity, diffusivity, T_top, T_bottom`

Note: `_IDENTITY_FIELDS` does **not** include `T_degree` — current
`SweepConfig` doesn't expose `T_degree` as a sweep field.  Adding it
would mean exposing `T_degree` as a `SweepConfig` attribute *and*
forwarding it in `_per_run_config`.

## How each `uw.workflows` primitive lands

| Primitive | Where convection uses it |
|-----------|--------------------------|
| `WorkflowConfig` (Pydantic base) | `ConvectionConfig`, `SweepConfig`, `VisualiseConfig` |
| `_identity_fields` (class-level tuple) | declared on both configs; aliases `_IDENTITY_FIELDS` module constant for `ConvectionConfig` |
| `WorkflowConfig.cache_key()` | called by the runner during `_save_quietly` and `_expected_cache_key` — convection doesn't call it directly |
| `@workflow_step(produces=, requires=)` | every step in `convection_config` and `convection_sweep` |
| `WorkflowRunner` | `simulate.py` for single runs, `convection_sweep.run_sweep` for nested per-case runners, `convection_notebook.py` for the sweep cascade |
| `WorkflowProducts` | per-case at `<output_dir>/products/`, plus the sweep's outer products dir |
| `Run` | wraps every per-case run directory; `evolve` produces a `Run`, `summarise_run` consumes one |
| `Manifest` | the case's `manifest.yaml` (different from the products manifest); accessed via `Run.manifest` |
| `RUN_NAME` | the `"run"` filename stem for the h5/xdmf chain |
| `Run.append_step` | `evolve`'s step-0 fresh-start path; `warm_start`'s seeded step-0 |
| `Run.append_timeseries_row` | `evolve`'s inner loop (csv every step, h5 every save_every) |
| `Run.load_field` | `warm_start` (read source T into target_T); `evolve`'s seed-from-old path uses `var.read_timestep` directly |
| `Run.steps` / `Run.timeseries` / `Run.summary` / `Run.archive` | `evolve` (resume), `summarise_run`, `_check_compatibility` |
| `cli_from_config` | `simulate.py:build_parser()` |
| `diagram` / `WorkflowRunner.diagram` | available; documentation generator could embed |
| `WorkflowRunner.observe` / `what_invalidates` | available; not yet wired into the example notebook (UI hooks for future widgets) |

## On-disk layout (per-case)

```
<output_dir>/
  manifest.yaml                # Run-directory manifest (workflow + config_hash + workflow_api)
  timeseries.csv               # Append-only per-step diagnostics
  run_summary.yaml             # Only when status="steady"
  run.mesh.NNNNN.{h5,xdmf}     # Mesh + per-variable h5 chain
  run.mesh.T.NNNNN.h5
  run.mesh.U.NNNNN.h5
  products/
    manifest.yaml              # WorkflowProducts manifest (per-product cache_key + inputs audit)
    mesh.mesh.00000.h5         # Mesh product (separate from run.mesh)
    mesh.mesh.00000.xdmf
    bl_thickness.yaml          # Float, yaml-fallback persistence
    evolution_log.yaml         # List-of-dicts, yaml-fallback
    run_summary.yaml           # Dict, yaml-fallback (mirrors the run-dir's summary)
```

The two `manifest.yaml` files in the same tree look similar but
serve different roles:

- The **outer** `manifest.yaml` (next to the h5 chain) is the
  Run-directory's identity card — workflow name, config_hash,
  config_snapshot, `workflow_api`.  Exists for any run.
- The **inner** `products/manifest.yaml` is the registered
  workflow-product graph for this case — `cache_key` + `inputs` per
  product.  Created when `WorkflowRunner(..., products=...)` is
  used.

When a case's identity changes (e.g. `T_degree` bump), the outer
manifest's `config_hash` mismatch triggers `_check_compatibility`'s
archive-or-error logic.  The products manifest's `cache_key`s
independently mismatch, triggering the runner's per-product rebuild.

## Sweep-level layout

```
<sweep_output_dir>/
  aspect_1x1/Ra1e3/...         # Per-case run directory (see above)
  aspect_1x1/Ra1e4/...
  aspect_4x1/Ra1e3/...
  ...
  tables/
    nu_vs_ra.csv               # Tidy aggregation
    vrms_vs_ra.csv
  figures/
    nu_vs_ra.png               # log-log Nu(Ra)
    vrms_vs_ra.png
  products/
    manifest.yaml              # Sweep-level products: all_cells_completed, *_csv, *_plot
    all_cells_completed.yaml
```

## Recipes

`warm_start` and `ramp` live in this directory but are **not** part
of the `uw.workflows` API.  They are example scripts that compose
the public primitives (`Run.open`, `Run.load_field`, `Run.create`,
`Run.append_step`, `WorkflowRunner.build`).  Promotion to API
happens "if 3+ apps end up writing nearly-identical versions" — the
discipline keeps the public surface small while letting per-app
patterns live where they belong.

### `warm_start(source_dir, target_dir, **target_overrides)`

1. Open the source `Run`, find its latest checkpoint.
2. Build a fresh target mesh + MeshVariables at the target's
   degrees.
3. Project source's T into target's T via `Run.load_field`
   (kd-tree interpolation handles different mesh / different FE
   space).
4. Solve Stokes once for a consistent v.
5. Persist the seeded state as step 0 of the target run via
   `Run.append_step`, write the manifest with `warm_start_source`
   metadata.

### `ramp_rayleigh(rayleigh_values, *, base_dir, **shared)`

Walks a sequence of Ra values, each leg warm-starting from the
previous converged leg.  Stalled legs break the chain; subsequent
legs cold-start instead.  ~50 lines, mostly orchestration.

## Open work

- `convection_visualise.render_temperature_frames` /
  `render_tracer_frames` / `encode_movie` are *not* yet
  `@workflow_step`-decorated.  Decorating them would add
  `temperature_frames`, `tracer_frames`, `temperature_movie`,
  `tracer_movie` as products with cache_keys derived from the
  case's `run_directory` cache_key.  Re-running the notebook would
  then short-circuit movie rendering when the underlying h5 chain
  hasn't grown.

- The visualisation-side `(run_dir / "run_summary.yaml").exists()`
  check in `convection_notebook.make_movies_for` could become
  `Run(run_dir).summary is not None`.

- `T_degree` is a `ConvectionConfig` field but not a `SweepConfig`
  field, so the sweep can't currently sweep over discretisation
  orders.

- `H2ExConfig` doesn't declare `_identity_fields`, so its products
  fall back to existence-based caching.  Bringing it up to par is
  about an hour of work.

## Rebuilding this guide

The structural content of this guide (steps, fields, identity
tuples, on-disk layout) can be regenerated by introspecting
`convection_config` and `convection_sweep`.  A future
`generate_developer_doc.py` would emit this file modulo prose
sections.  For now, hand-maintained — bump the relevant table when
adding/removing a step or field.
