"""Drive a (Ra × aspect_ratio) sweep of the per-run convection workflow.

The sweep is itself a **workflow** — its steps (``run_sweep``,
``tabulate_nu_vs_ra``, ``plot_nu_vs_ra``, ``tabulate_vrms_vs_ra``,
``plot_vrms_vs_ra``) are :func:`@workflow_step`-decorated so the whole
graph can be driven by ``WorkflowRunner.build("nu_vs_ra_plot")``,
walking the cascade and short-circuiting cached steps.  Each
``(Ra, aspect_ratio)`` cell still uses its own nested
``WorkflowRunner`` with the per-run convection module, with its own
``output_dir`` and its own idempotent steady-state termination.

Idempotency layers:

- **Per cell**: each cell's ``run_summary.yaml`` short-circuits its
  inner runner once steady; stalled cells extend on the next sweep
  invocation.
- **Sweep level**: the aggregation products (CSV tables, PNG plots)
  are cached under ``<sweep output_dir>/products/`` with cache keys
  derived from the sweep config's identity.  Re-asking for
  ``nu_vs_ra_plot`` after nothing has changed hits the cache; if the
  sweep config changes (e.g. a Ra value added) the affected products
  rebuild automatically.

Usage::

    import convection_sweep as sweep
    from underworld3.workflows import WorkflowRunner, WorkflowProducts

    config = sweep.SweepConfig(
        rayleigh_values=[1e3, 1e4, 1e5, 1e6],
        aspect_ratios=[1.0, 4.0],
        cellsize=1/16,
    )

    # Single cascade — runs cells, tabulates, plots; caches everything.
    products = WorkflowProducts(config)
    runner = WorkflowRunner(sweep, config, products=products)
    runner.build("nu_vs_ra_plot")
    runner.build("vrms_vs_ra_plot")

The plain functions are still callable directly when a workflow
runner is overkill::

    sweep.run_sweep(config)              # drives every cell
    sweep.tabulate_nu_vs_ra(config=config)   # reads each cell's summary

Output layout::

    <output_dir>/
        aspect_1x1/Ra1e3/...        per-cell run directory
        aspect_1x1/Ra1e4/...
        ...
        aspect_4x1/Ra1e3/...
        ...
        tables/
            nu_vs_ra.csv            tidy: aspect, Ra, Nu_mean, Nu_std, n_steps
            vrms_vs_ra.csv          tidy: aspect, Ra, Vrms_mean, Vrms_std, n_steps
        figures/
            nu_vs_ra.png            log-log Nu(Ra) with reference Ra^(1/3) line
            vrms_vs_ra.png
        products/
            manifest.yaml           workflow products: cache_keys for the
                                    aggregation outputs.
"""

import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import Field

from underworld3.workflows import Run, WorkflowConfig, WorkflowRunner, workflow_step

# We share the per-run definitions but hide the "single-run" config under
# a different name to avoid confusion with this module's SweepConfig.
import convection_config as _convection


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


class SweepConfig(WorkflowConfig):
    """Drive a (Ra × aspect_ratio) sweep of the convection workflow.

    Mesh/physics defaults are forwarded to each per-run config along
    with operational fields (steady-state tolerances, batch sizing).
    """

    workflow_name: str = "rayleigh_benard_sweep"
    description: str = "Sweep convection over (Ra, aspect_ratio) to steady state"

    # Sweep grid
    rayleigh_values: list[float] = Field(
        default_factory=lambda: [1e3, 1e4, 1e5, 1e6]
    )
    aspect_ratios: list[float] = Field(default_factory=lambda: [1.0, 4.0])

    # Per-run mesh/physics defaults (forwarded)
    cellsize: float = Field(default=1.0 / 16, gt=0)
    qdegree: int = Field(default=3, ge=1)
    regular: bool = False
    viscosity: float = Field(default=1.0, gt=0)
    diffusivity: float = Field(default=1.0, gt=0)
    T_top: float = 0.0
    T_bottom: float = 1.0

    # Per-run operational defaults (forwarded)
    steady_window: float = Field(default=0.3, gt=0, le=1.0)
    steady_tol_mean: float = Field(default=0.02, gt=0)
    steady_tol_cv: float = Field(default=0.05, gt=0)
    steady_min_window: int = Field(default=50, ge=10)
    batch_steps: int = Field(default=200, gt=0)
    max_steps: int = Field(default=5000, gt=0)
    save_every: int = Field(default=10, gt=0)
    dt_factor: float = Field(default=2.0, gt=0)
    restart_policy: str = "error"

    # Output
    output_dir: str = "output/convection_sweep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aspect_dirname(aspect: float) -> str:
    """``1.0 -> 'aspect_1x1'``, ``4.0 -> 'aspect_4x1'``."""
    if aspect == int(aspect):
        return f"aspect_{int(aspect)}x1"
    return f"aspect_{aspect:g}x1"


def _ra_dirname(ra: float) -> str:
    """``1e3 -> 'Ra1e3'``, ``2.5e6 -> 'Ra2.5e6'``."""
    exponent = int(np.floor(np.log10(ra)))
    mantissa = ra / 10**exponent
    if abs(mantissa - 1.0) < 1e-9:
        return f"Ra1e{exponent}"
    return f"Ra{mantissa:g}e{exponent}"


def _run_dir(sweep_config: SweepConfig, ra: float, aspect: float) -> Path:
    return (
        Path(sweep_config.output_dir)
        / _aspect_dirname(aspect)
        / _ra_dirname(ra)
    )


def _per_run_config(sweep_config: SweepConfig, ra: float, aspect: float):
    """Build a per-run ConvectionConfig for this (Ra, aspect) cell."""
    return _convection.ConvectionConfig(
        rayleigh=ra,
        aspect_ratio=aspect,
        cellsize=sweep_config.cellsize,
        qdegree=sweep_config.qdegree,
        regular=sweep_config.regular,
        viscosity=sweep_config.viscosity,
        diffusivity=sweep_config.diffusivity,
        T_top=sweep_config.T_top,
        T_bottom=sweep_config.T_bottom,
        steady_window=sweep_config.steady_window,
        steady_tol_mean=sweep_config.steady_tol_mean,
        steady_tol_cv=sweep_config.steady_tol_cv,
        steady_min_window=sweep_config.steady_min_window,
        batch_steps=sweep_config.batch_steps,
        max_steps=sweep_config.max_steps,
        save_every=sweep_config.save_every,
        dt_factor=sweep_config.dt_factor,
        restart_policy=sweep_config.restart_policy,
        output_dir=str(_run_dir(sweep_config, ra, aspect)),
    )


def _read_run_summary(run_dir: Path) -> Optional[dict]:
    return Run(run_dir).summary


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _cell_key(ra: float, aspect: float) -> str:
    """String key used to identify a (Ra, aspect) cell in dict products.

    YAML-serialisable (no tuple keys).  Stable under
    ``json.dumps(..., sort_keys=True)`` so the cache key for
    ``all_cells_completed`` is reproducible.
    """
    return f"aspect_{aspect:g}_Ra_{ra:g}"


@workflow_step(
    description="Drive every (Ra, aspect) cell to steady state",
    produces=["all_cells_completed"],
)
def run_sweep(config: SweepConfig) -> dict:
    """Drive every (Ra, aspect) cell to steady state.

    Each cell uses its own nested ``WorkflowRunner`` against the
    convection workflow; that runner reads the cell's on-disk state
    and either short-circuits (already steady) or extends the run.

    Returns a dict ``{cell_key: summary_or_None}``.  Cells that have
    not yet converged have summary ``None`` (no ``run_summary.yaml``).
    """
    import underworld3 as uw

    results = {}
    for aspect in config.aspect_ratios:
        for ra in config.rayleigh_values:
            run_dir = _run_dir(config, ra, aspect)
            run_dir.mkdir(parents=True, exist_ok=True)

            uw.pprint(
                f"\n=== aspect={aspect}  Ra={ra:.0e}  -> {run_dir} ===",
                flush=True,
            )

            cell_config = _per_run_config(config, ra, aspect)
            runner = WorkflowRunner(_convection, cell_config, products=None)
            summary = runner.build("run_summary")
            results[_cell_key(ra, aspect)] = summary

    return results


@workflow_step(
    description="Tabulate Nu(Ra) from each cell's run summary into a tidy CSV",
    produces=["nu_vs_ra_csv"],
    requires=["all_cells_completed"],
)
def tabulate_nu_vs_ra(all_cells_completed: dict, config: SweepConfig) -> Path:
    """Aggregate per-cell summaries into a tidy CSV.

    Columns: ``aspect, Ra, Nu_mean, Nu_std, Nu_top_mean, Nu_bot_mean,
    n_steps, status``.  Skips cells whose summary is ``None`` (no
    ``run_summary.yaml`` yet — i.e. not steady).
    """
    out_dir = Path(config.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nu_vs_ra.csv"

    rows = []
    for aspect in config.aspect_ratios:
        for ra in config.rayleigh_values:
            summary = all_cells_completed.get(_cell_key(ra, aspect))
            if summary is None:
                continue
            rows.append({
                "aspect": aspect,
                "Ra": ra,
                "status": summary.get("status", ""),
                "n_steps": summary.get("n_steps", 0),
                "Nu_mean": summary.get("Nu_mean"),
                "Nu_std": summary.get("Nu_std"),
                "Nu_top_mean": summary.get("Nu_top_mean"),
                "Nu_bot_mean": summary.get("Nu_bot_mean"),
            })

    fieldnames = [
        "aspect", "Ra", "status", "n_steps",
        "Nu_mean", "Nu_std", "Nu_top_mean", "Nu_bot_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


@workflow_step(
    description="Tabulate Vrms(Ra) from each cell's run summary into a tidy CSV",
    produces=["vrms_vs_ra_csv"],
    requires=["all_cells_completed"],
)
def tabulate_vrms_vs_ra(all_cells_completed: dict, config: SweepConfig) -> Path:
    """Aggregate per-cell summaries into Vrms(Ra) tidy CSV."""
    out_dir = Path(config.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vrms_vs_ra.csv"

    rows = []
    for aspect in config.aspect_ratios:
        for ra in config.rayleigh_values:
            summary = all_cells_completed.get(_cell_key(ra, aspect))
            if summary is None:
                continue
            rows.append({
                "aspect": aspect,
                "Ra": ra,
                "status": summary.get("status", ""),
                "n_steps": summary.get("n_steps", 0),
                "Vrms_mean": summary.get("Vrms_mean"),
                "Vrms_std": summary.get("Vrms_std"),
                "mean_T_mean": summary.get("mean_T_mean"),
            })

    fieldnames = [
        "aspect", "Ra", "status", "n_steps",
        "Vrms_mean", "Vrms_std", "mean_T_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


@workflow_step(
    description="Plot Nu(Ra) log-log per aspect ratio, with Ra^(1/3) reference",
    produces=["nu_vs_ra_plot"],
    requires=["nu_vs_ra_csv", "all_cells_completed"],
)
def plot_nu_vs_ra(
    nu_vs_ra_csv: Path,
    all_cells_completed: dict,
    config: SweepConfig,
) -> Optional[Path]:
    """Log-log Nu(Ra) per aspect ratio, with the Ra^(1/3) reference line.

    Reads from *all_cells_completed* (numerically authoritative) but
    declares ``nu_vs_ra_csv`` in ``requires=`` so the runner knows the
    plot post-dates the table when both are part of the same cascade.

    Returns the figure path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(config.output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "nu_vs_ra.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    for aspect in config.aspect_ratios:
        ras, nus, errs = [], [], []
        for ra in config.rayleigh_values:
            summary = all_cells_completed.get(_cell_key(ra, aspect))
            if summary is None or summary.get("Nu_mean") is None:
                continue
            ras.append(ra)
            nus.append(summary["Nu_mean"])
            errs.append(summary.get("Nu_std", 0.0) or 0.0)
        if ras:
            ax.errorbar(
                ras, nus, yerr=errs,
                marker="o", capsize=3,
                label=f"aspect = {aspect}",
            )

    # Reference scaling: Nu = 0.27 Ra^(1/3) (high-Ra 2D)
    ras_ref = np.logspace(
        np.log10(min(config.rayleigh_values)),
        np.log10(max(config.rayleigh_values)),
        50,
    )
    ax.plot(
        ras_ref, 0.27 * ras_ref ** (1 / 3),
        ":", color="grey",
        label=r"$0.27\,\mathrm{Ra}^{1/3}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rayleigh number")
    ax.set_ylabel(r"$\mathrm{Nu}$ (steady-state mean)")
    ax.set_title("Nusselt number vs Rayleigh number")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


@workflow_step(
    description="Plot Vrms(Ra) log-log per aspect ratio",
    produces=["vrms_vs_ra_plot"],
    requires=["vrms_vs_ra_csv", "all_cells_completed"],
)
def plot_vrms_vs_ra(
    vrms_vs_ra_csv: Path,
    all_cells_completed: dict,
    config: SweepConfig,
) -> Optional[Path]:
    """Log-log V_rms(Ra) per aspect ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(config.output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "vrms_vs_ra.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    for aspect in config.aspect_ratios:
        ras, vrs, errs = [], [], []
        for ra in config.rayleigh_values:
            summary = all_cells_completed.get(_cell_key(ra, aspect))
            if summary is None or summary.get("Vrms_mean") is None:
                continue
            ras.append(ra)
            vrs.append(summary["Vrms_mean"])
            errs.append(summary.get("Vrms_std", 0.0) or 0.0)
        if ras:
            ax.errorbar(
                ras, vrs, yerr=errs,
                marker="s", capsize=3,
                label=f"aspect = {aspect}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rayleigh number")
    ax.set_ylabel(r"$V_{\mathrm{rms}}$ (steady-state mean)")
    ax.set_title("RMS velocity vs Rayleigh number")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def status(sweep_config: SweepConfig) -> dict:
    """Summarise which cells are steady, stalled, or not yet started."""
    out = {}
    for aspect in sweep_config.aspect_ratios:
        for ra in sweep_config.rayleigh_values:
            run_dir = _run_dir(sweep_config, ra, aspect)
            summary = _read_run_summary(run_dir)
            if summary is not None:
                out[(ra, aspect)] = summary.get("status", "unknown")
            elif Run(run_dir).manifest is not None:
                # Has work but no summary -> stalled or in-progress
                out[(ra, aspect)] = "in_progress"
            else:
                out[(ra, aspect)] = "not_started"
    return out


def view():
    """Display this module's workflow steps and config classes."""
    from underworld3.workflows import view as _wf_view
    _wf_view(sys.modules[__name__])
