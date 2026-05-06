---
title: "Convection Workflow — User Guide"
---

# Convection Workflow — User Guide

How to run a Rayleigh-Bénard convection simulation, sweep over
Rayleigh numbers, and read the results.  Companion to the
[developer guide](convection-developer.md), which covers how the
workflow is built; this page covers what it does and how to use it.

## What this workflow simulates

A 2-D rectangular box of fluid heated from below and cooled from
above, with free-slip side walls.  The contrast between hot fluid
rising and cold fluid sinking sets up convection cells whose
strength grows with the **Rayleigh number** (a dimensionless ratio
of buoyancy to viscous + thermal-diffusion effects).  The same
equations describe mantle convection, atmospheric circulation, and
many other geophysical flows.

The workflow drives a coupled Stokes (momentum) + advection-diffusion
(temperature) solve forward in time until the system settles into a
**steady state** — a flow pattern that looks the same from one snapshot
to the next.  At that point it records average values of Nu, Vrms, and
mean T over the steady window, and writes a "done" marker so re-running
the workflow short-circuits.

## Quickstart

### Run a single case

```bash
pixi run -e default python simulate.py
```

That uses defaults (Ra=10⁶, square box, output to
`output/convection/run/`).  Watch the per-step printout — once Nu and
Vrms stop drifting, the run flags itself as steady, writes a summary,
and exits.

To override any parameter:

```bash
python simulate.py --rayleigh=1e5 --T-degree=5 \
                   --output-dir=output/Ra1e5_T5
```

`python simulate.py --help` lists every available knob.

### Run a sweep over (Ra × aspect ratio) and produce Nu(Ra) plots

Open `convection_notebook.py` (it's a Jupytext-style script — runs
as a regular Python file, opens as a notebook in JupyterLab):

```python
import convection_sweep as sweep
from underworld3.workflows import WorkflowRunner, WorkflowProducts

config = sweep.SweepConfig(
    rayleigh_values=[1e4, 1e5, 1e6],
    aspect_ratios=[1.0, 4.0],
    cellsize=1.0/24,
    max_steps=4000,
    output_dir="output/convection_sweep",
)
runner = WorkflowRunner(sweep, config, products=WorkflowProducts(config))
runner.build("nu_vs_ra_plot")    # drives every cell to steady state, then plots
runner.build("vrms_vs_ra_plot")
```

Each `(Ra, aspect)` cell runs in its own subdirectory.  Cells that
were already steady on a previous invocation skip immediately;
stalled cells extend on the next run.

### Render movies

After a cell has reached steady state:

```python
import convection_visualise as viz
cfg = viz.VisualiseConfig(run_dir="output/convection_sweep/aspect_1x1/Ra1e6")
viz.render_temperature_frames(cfg)
viz.encode_movie(cfg, kind="temperature")
```

Movies land in `<run_dir>/movies/` as MP4s.  An `ffmpeg` binary
needs to be on PATH.

## Configuration knobs

These are the parameters you'll want to set on
`ConvectionConfig` (single-run) or `SweepConfig` (sweep).  Values
are non-dimensional unless otherwise noted.  The setup is normalised
so that the layer thickness is 1, the temperature contrast is 1
(top → 0, bottom → 1 by default), and time is in units of thermal
diffusion time across the layer.

### What the simulation is solving

| Knob | Default | Meaning |
|------|---------|---------|
| `rayleigh` | `1e6` | Buoyancy strength.  ~10³ is barely-convective, 10⁴ steady cells, 10⁵–10⁶ vigorous, 10⁷+ time-dependent.  Cubed in the parameter space — every factor of 10 makes the boundary layers thinner by ~3×. |
| `aspect_ratio` | `1.0` | Box width / height.  1×1 gives a single roll; longer boxes give multiple rolls.  4×1 is a common diagnostic geometry. |
| `viscosity` | `1.0` | Constant fluid viscosity (non-dim).  Changing this is equivalent to changing Ra. |
| `diffusivity` | `1.0` | Thermal diffusivity (non-dim).  Same. |
| `T_top`, `T_bottom` | `0.0`, `1.0` | Boundary temperatures.  The defaults give Nu ≈ heat flux directly. |

### Mesh + numerics

| Knob | Default | Meaning |
|------|---------|---------|
| `cellsize` | `1/16` | Target element size.  At Ra=10⁶ you want at least 1/24 to resolve the boundary layers; 1/48 for safety.  Bigger → faster but under-resolved. |
| `qdegree` | `3` | Quadrature order.  Match to the highest-degree variable. |
| `T_degree` | `3` | Polynomial order of the temperature field.  Higher → smoother T at the boundaries (so Nu is more accurate) but more DOFs.  Increasing this on a converged run is a good test of mesh-independence (use the warm-start recipe — see below). |
| `regular` | `False` | If True, builds a regular mesh; otherwise unstructured. |

### When the run stops

| Knob | Default | Meaning |
|------|---------|---------|
| `max_steps` | `5000` | Hard cap.  If reached without convergence, the run is "stalled" — re-run with a higher cap to extend. |
| `steady_window` | `0.3` | Test stationarity over the last 30% of the timeseries. |
| `steady_tol_mean` | `0.02` | The mean of the first half vs the second half of the test window must match within 2%. |
| `steady_tol_cv` | `0.05` | Coefficient of variation (std / mean) within the window must be below 5%. |
| `steady_min_window` | `50` | Need at least 50 saved samples before declaring steady. |

If a run isn't converging, loosen the tolerances before changing
the physics.  At low Ra the flow is quasi-static and `0.02` /
`0.05` are easily met; at high Ra you may need `0.05` / `0.1`.

### Saving + diagnostics

| Knob | Default | Meaning |
|------|---------|---------|
| `save_every` | `10` | Write an h5 checkpoint every N steps.  Smaller → more frames for movies but bigger output. |
| `dt_factor` | `2.0` | Multiplier on the solver's CFL-derived dt estimate.  Reduce if the run blows up at high Ra. |
| `diag_every` | `0` | How often to compute the heavy volume integrals (Vrms, mean T).  `0` means match `save_every`; `1` means every step (low-Ra short runs); `>1` means every N steps. |
| `clip_T_range` | `False` | Clip T to `[T_top, T_bottom]` after each AdvDiff solve.  Useful as a safety belt during the early high-Ra transient where overshoots can pump runaway buoyancy. |

### Where output goes

| Knob | Default | Meaning |
|------|---------|---------|
| `output_dir` | `"output/convection/run"` | Where this run's files live.  Each `(Ra, aspect)` cell of a sweep gets its own subdir under the sweep's `output_dir`. |
| `restart_policy` | `"error"` | What to do when re-running with a *different* mesh or physics in the same `output_dir`: <br>• `"error"` (default) — stop with a list of changed fields. <br>• `"fresh"` — archive the old directory to `<name>.archive-<timestamp>` and start over. <br>• `"seed_from_old"` — same archive, but use the old run's last T as the new run's IC. |

## Output: what you get

Every per-cell run dir contains:

- **`manifest.yaml`** — what the run is for: workflow name,
  parameter snapshot, identity hash, version stamps.  Reading it
  tells you exactly what was running.
- **`timeseries.csv`** — one row per simulation step: `step, t, dt`
  plus the diagnostics (`Nu_top, Nu_bot, Vrms, Vmax, mean_T`).
  Cells where `Vrms` or `mean_T` show `---` are steps where the
  heavy volume integrals were skipped to save time (controlled by
  `diag_every`).
- **`run.mesh.NNNNN.h5`** + `run.mesh.{T,U}.NNNNN.h5` +
  **`run.mesh.NNNNN.xdmf`** — the checkpoint chain.  ParaView opens
  the `.xdmf` files directly.
- **`run_summary.yaml`** — *only present once the run reached
  steady state*.  Contains the steady-window averages (Nu_mean,
  Vrms_mean, mean_T_mean, plus standard deviations) and `n_steps`,
  the step at which steady state was declared.
- **`products/manifest.yaml`** — the workflow's bookkeeping (see
  glossary below).  Safe to ignore unless you're debugging.

A sweep additionally produces:

- **`tables/nu_vs_ra.csv`**, **`tables/vrms_vs_ra.csv`** — tidy CSVs
  with one row per (cell, status) suitable for pandas / excel.
- **`figures/nu_vs_ra.png`**, **`figures/vrms_vs_ra.png`** — log-log
  plots, error bars from the steady-window standard deviation, with
  the `0.27·Ra^(1/3)` reference line on the Nu plot.
- **`<cell>/movies/temperature.mp4`** etc. (optional) — rendered
  movies.

## Recipes

The workflow ships with two example recipes for common operations.
They live as Python files alongside the workflow code so you can
copy and adapt them.

### Warm-start a high-resolution run from a low-resolution one

`warm_start.py` takes a source run dir and projects its converged T
field onto a new mesh, optionally with different settings:

```python
from warm_start import warm_start
from underworld3.workflows import WorkflowRunner
import convection_config as cc

# Source: the converged T_degree=3 run
target_cfg = warm_start(
    "output/Ra1e5",
    "output/Ra1e5_T5",
    T_degree=5,           # bump the polynomial order
)

# Drive the new run forward; transient is short because the seed is good
WorkflowRunner(cc, target_cfg).build("run_summary")
```

Useful for: refining `T_degree` or `qdegree` after a coarse run
has settled, branching ensembles from a single seed, or moving a
case to a finer mesh.

### Ramp through Rayleigh numbers

`ramp.py` chains warm-starts through a sequence of Ra values, each
leg seeded from the previous one's converged state:

```python
from ramp import ramp_rayleigh

ramp_rayleigh(
    rayleigh_values=[1e4, 1e5, 1e6],
    base_dir="output/ramp",
    cellsize=1.0/24,
    max_steps=4000,
)
```

This is much cheaper than cold-starting at high Ra: each leg
inherits an already-organised circulation pattern from below.  If a
leg stalls, subsequent legs cold-start instead.

## Customising

### Sweep over a different parameter

`SweepConfig` currently exposes `(Ra × aspect_ratio)`.  To sweep
over, say, `cellsize` instead, copy `convection_sweep.py` and:

1. Add `cellsize_values: list[float]` to your subclass (and remove
   the scalar `cellsize` from `_identity_fields`).
2. In `_per_run_config`, iterate over `cellsize_values` and forward
   each to the per-cell config.
3. Mirror the change in `tabulate_*` and `plot_*`.

### Add a new diagnostic

Two places to touch:

1. `_compute_diagnostics` in `convection_config.py` — extend the
   returned dict with your new value.
2. `_TS_FIELDS` at the top of `convection_config.py` — add the
   column name.  The CSV will auto-migrate on first append.

### Add a new aggregation product

Decorate the function with `@workflow_step(produces=...,
requires=...)` and add the product name to whatever notebook calls
`runner.build(...)`.  The framework will figure out the rest.

## Glossary

A small vocabulary used throughout the workflow.  Worth learning
once because it generalises across UW3 workflows.

**Workflow**
A named computation broken into discrete steps with declared
inputs and outputs.  The convection workflow is one; the sweep
workflow is another.  Each is a Python module of decorated
functions.

**Step**
A single function decorated with `@workflow_step`, declaring
what it `produces` (one or more named outputs) and `requires`
(named outputs of upstream steps).  Together the steps form a
DAG (directed acyclic graph).

**Product**
A named, cached output.  In the convection workflow,
`run_directory`, `run_summary`, `nu_vs_ra_plot` are products.
The runner builds, caches, and freshness-checks them.

**Cache key**
A short hex digest derived from a product's inputs (the
relevant config fields plus upstream products' cache keys).
Two products with the same cache key are guaranteed to be
equivalent under deterministic producers; mismatched cache
keys mean "the inputs changed, the cached version is stale,
rebuild".  Stored next to each product in
`products/manifest.yaml`.

**Freshness**
The property of a cached product whose cache key still matches
what the current config expects.  When you ask the workflow for
something, it walks the dependency graph; products that are
fresh hit the cache, products that are stale rebuild.

**Recipe**
An example script that composes the workflow's primitives —
e.g. `warm_start`, `ramp`.  Recipes are not part of the
underlying API; they're patterns you can copy and adapt.

**Run directory**
The on-disk folder for one simulation: manifest, h5 chain,
timeseries.csv, summary.  Corresponds to one
`ConvectionConfig`.

**Steady state**
The condition under which the workflow declares the run "done"
— Nu and Vrms are statistically stationary over the last 30% of
the timeseries (default).  At that point it writes
`run_summary.yaml` as a "done marker" so future invocations
short-circuit.

**Identity fields**
The subset of a config whose change should *invalidate* cached
products.  Mesh and physics fields are identity; step caps and
tolerances aren't.

**Restart policy**
What the workflow does when it finds an existing run dir whose
identity differs from the current config.  Default is `error`
— it stops and lists the differences.  `fresh` archives and
restarts; `seed_from_old` uses the old final T as the new IC.

## Where to ask questions

- Per-workflow questions: open an issue in the workflow's
  repository (this example currently lives in the UW3 main repo
  under `docs/examples/workflows/`).
- Underlying-machinery questions (`Run`, `WorkflowConfig`,
  `WorkflowRunner`, etc.): see the [developer guide](convection-developer.md)
  or the UW3 workflow-packages guide
  (`docs/developer/guides/workflow-packages.md`).
