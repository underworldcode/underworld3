# %% [markdown]
r"""
# Thermal Convection — Idempotent Workflow

**PHYSICS:** Rayleigh–Bénard convection in 2-D, Cartesian box(es)

**DIFFICULTY:** intermediate

## What this notebook does

Drives a small parameter sweep of Rayleigh number × box aspect ratio,
and produces:

1. A **Nu(Ra) table** and figure — Nusselt number vs Rayleigh number for
   each aspect ratio.
2. A **V_rms(Ra) table** and figure — RMS velocity vs Rayleigh number.
3. **Movies** for each cell: a temperature-only movie and a
   tracer-overlay movie that shows mixing / circulation patterns.

## Idempotency

Re-running this notebook is safe.  Each per-run cell of the sweep keeps
its own state in ``output/convection_sweep/aspect_<w>x1/Ra<r>/``.  When a
cell already has ``run_summary.yaml`` with ``status="steady"``, no
solver work is done.  Otherwise the run is **resumed** from the latest
on-disk h5 checkpoint until either steady state or ``max_steps`` is hit.

The notebook never deletes prior data.  Changing a mesh- or physics-
relevant config field on a cell that already has on-disk state will
either raise (``restart_policy="error"``, default) or archive the old
directory to ``Ra<r>.archive-<timestamp>/`` and start fresh.

## What's quietly using boundary-layer theory

The initial-temperature step seeds a ``tanh``-profile field with the
predicted thermal-boundary-layer thickness ``δ ≈ 1/(2·Nu_pred)`` and
``Nu_pred ≈ 0.27·Ra^(1/3)``.  This shaves the conductive transient
considerably at high Ra.

## Output layout

```
output/convection_sweep/
    aspect_1x1/Ra1e4/        per-run dir: mesh, h5 chain, timeseries.csv,
    aspect_1x1/Ra1e5/        run_summary.yaml (when steady)
    aspect_1x1/Ra1e6/
    aspect_4x1/Ra1e4/
    ...
    tables/nu_vs_ra.csv
    tables/vrms_vs_ra.csv
    figures/nu_vs_ra.png
    figures/vrms_vs_ra.png
    aspect_1x1/Ra1e6/frames/temperature/frame_*.png
    aspect_1x1/Ra1e6/frames/tracers/frame_*.png
    aspect_1x1/Ra1e6/movies/temperature.mp4
    aspect_1x1/Ra1e6/movies/tracers.mp4
```
"""

# %%
#| echo: false
import nest_asyncio
nest_asyncio.apply()

# %%
import os
import underworld3 as uw

import convection_config as convection
import convection_sweep as sweep
import convection_visualise as viz

# %% [markdown]
"""
## 1. Sweep configuration

Set the (Ra, aspect) grid and the per-run defaults (mesh resolution,
steady-state tolerances, step caps).  Operational fields (tolerances,
batch size, ``max_steps``) can change between invocations without
flagging a config mismatch — only mesh and physics fields do.
"""

# %%
sweep_config = sweep.SweepConfig(
    rayleigh_values=[1e4, 1e5, 1e6],
    aspect_ratios=[1.0, 4.0],
    cellsize=1.0 / 24,
    qdegree=3,
    # Steady-state detection
    steady_window=0.3,
    steady_tol_mean=0.02,
    steady_tol_cv=0.05,
    steady_min_window=80,
    # Step budget per cell
    save_every=10,
    max_steps=4000,
    output_dir="output/convection_sweep",
)
sweep_config.view()

# %% [markdown]
"""
## 2. Per-run workflow steps

These are the workflow steps that will execute (in dependency order)
for each cell of the sweep.  ``view()`` shows the DAG with declared
``produces`` and ``requires`` lists.
"""

# %%
convection.view()

# %% [markdown]
"""
## 3. Drive the sweep

For each ``(Ra, aspect)`` cell:

* If ``run_summary.yaml`` exists with ``status="steady"`` — short-circuit.
* Else — resume (or fresh-start) and extend until steady state or
  ``max_steps``.  Stalled cells leave no summary, so a future invocation
  with a larger ``max_steps`` continues seamlessly.

Per-step diagnostics are printed during long runs so progress is
visible (and the run is interruptible with the partial chain intact).
"""

# %%
results = sweep.run_sweep(sweep_config)

uw.pprint("\nSweep status summary:")
for (ra, aspect), summary in results.items():
    if summary is None:
        status = "no summary"
    else:
        status = (
            f"{summary.get('status', '?'):8}  "
            f"n_steps={summary.get('n_steps', 0):4d}  "
            f"Nu_mean={summary.get('Nu_mean', float('nan')):.3f}  "
            f"Vrms_mean={summary.get('Vrms_mean', float('nan')):.3f}"
        )
    uw.pprint(f"  aspect={aspect}  Ra={ra:.0e}  {status}")

# %% [markdown]
"""
## 4. Aggregate Nu(Ra) and V_rms(Ra) tables and figures

These read each cell's ``run_summary.yaml``; any cell that hasn't yet
reached steady state is left out of the table.
"""

# %%
nu_csv = sweep.tabulate_nu_vs_ra(sweep_config)
vrms_csv = sweep.tabulate_vrms_vs_ra(sweep_config)
uw.pprint(f"Wrote {nu_csv}")
uw.pprint(f"Wrote {vrms_csv}")

# %%
fig_nu = sweep.plot_nu_vs_ra(sweep_config)
fig_vrms = sweep.plot_vrms_vs_ra(sweep_config)
uw.pprint(f"Wrote {fig_nu}")
uw.pprint(f"Wrote {fig_vrms}")

# %% [markdown]
"""
## 5. Movies

For each (Ra, aspect) cell that produced an h5 chain, render

* a **temperature-only** movie, and
* a **temperature + tracers** movie (passive markers advected through the
  velocity history) showing circulation patterns.

These read only h5 — no solver code, fully decoupled from step 3.
"""

# %%
def make_movies_for(ra, aspect):
    run_dir = sweep._run_dir(sweep_config, ra, aspect)
    if not (run_dir / "run_summary.yaml").exists():
        uw.pprint(f"  skip {run_dir.name} (no summary yet)")
        return
    cfg = viz.VisualiseConfig(
        run_dir=str(run_dir),
        every=1,
        n_tracers_per_dim=40,
        fps=12,
    )
    viz.render_temperature_frames(cfg)
    viz.encode_movie(cfg, kind="temperature")
    viz.render_tracer_frames(cfg)
    viz.encode_movie(cfg, kind="tracers")
    uw.pprint(f"  done {run_dir.name}")

# %%
for aspect in sweep_config.aspect_ratios:
    for ra in sweep_config.rayleigh_values:
        make_movies_for(ra, aspect)

# %% [markdown]
"""
## Summary

This notebook orchestrates a parameter study end-to-end:

* **Per-run idempotency** — each cell either short-circuits or extends
  from its on-disk checkpoint chain.  Re-execution is always safe.
* **Boundary-layer-scaled IC** — `predict_bl_thickness` quietly seeds
  the IC with the predicted thermal-BL thickness, accelerating onset.
* **Strict separation** — the solver workflow writes h5; the
  visualisation workflow reads h5.  Either can run without the other.
* **DAG metadata** — ``produces`` / ``requires`` annotations let
  ``WorkflowRunner`` resolve dependencies automatically when building
  any product.

To extend the study: add Ra values or aspect ratios to
``sweep_config.rayleigh_values`` / ``aspect_ratios`` and re-run.
Existing cells short-circuit; new cells run fresh.

To **invalidate** an existing run (e.g. on a new mesh resolution),
either bump ``sweep_config.cellsize`` and set
``sweep_config.restart_policy="fresh"`` (archives old, starts clean),
or move the offending directory aside manually.
"""
