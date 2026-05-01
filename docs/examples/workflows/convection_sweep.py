"""Drive a (Ra × aspect_ratio) sweep of the per-run convection workflow.

This module is an **imperative driver** rather than a pure workflow:
each ``(Ra, aspect_ratio)`` cell of the sweep is a separate per-run
workflow with its own ``WorkflowRunner``, its own ``output_dir``, and
its own idempotent steady-state termination.  Aggregation steps then
read each cell's ``run_summary.yaml`` and produce Nu(Ra) / Vrms(Ra)
tables and matplotlib figures.

Idempotency carries over from the per-run workflow: re-running the
sweep does no work for cells that have already reached steady state.
A cell that's stalled (hit max_steps without converging) extends on
the next sweep invocation.

Usage::

    import convection_sweep as sweep

    config = sweep.SweepConfig(
        rayleigh_values=[1e3, 1e4, 1e5, 1e6],
        aspect_ratios=[1.0, 4.0],
        cellsize=1/16,
    )
    sweep.run_sweep(config)        # drive every cell to steady state
    sweep.tabulate_nu_vs_ra(config)
    sweep.tabulate_vrms_vs_ra(config)
    sweep.plot_nu_vs_ra(config)
    sweep.plot_vrms_vs_ra(config)

Output layout::

    <output_dir>/
        aspect_1x1/Ra1e3/...        per-run directory
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
"""

import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pydantic import Field

from underworld3.workflows import WorkflowConfig, WorkflowRunner

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
    p = run_dir / "run_summary.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_sweep(sweep_config: SweepConfig) -> dict:
    """Drive every (Ra, aspect) cell to steady state.

    Each cell uses its own ``WorkflowRunner`` with the shared per-run
    convection module; the runner reads on-disk state and either
    short-circuits (already steady) or extends the run.

    Returns a dict ``{(ra, aspect): summary_or_None}``.  Cells that
    have not yet converged have summary ``None`` (no run_summary.yaml).
    """
    import underworld3 as uw

    results = {}
    for aspect in sweep_config.aspect_ratios:
        for ra in sweep_config.rayleigh_values:
            run_dir = _run_dir(sweep_config, ra, aspect)
            run_dir.mkdir(parents=True, exist_ok=True)

            uw.pprint(
                f"\n=== aspect={aspect}  Ra={ra:.0e}  -> {run_dir} ===",
                flush=True,
            )

            cell_config = _per_run_config(sweep_config, ra, aspect)
            runner = WorkflowRunner(_convection, cell_config, products=None)
            summary = runner.build("run_summary")
            results[(ra, aspect)] = summary

    return results


def tabulate_nu_vs_ra(sweep_config: SweepConfig) -> Path:
    """Aggregate per-run summaries into a tidy CSV.

    Columns: ``aspect, Ra, Nu_mean, Nu_std, Nu_top_mean, Nu_bot_mean,
    n_steps, status``.  Only writes rows for cells whose
    ``run_summary.yaml`` exists (steady or summarised stalled runs).
    """
    out_dir = Path(sweep_config.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nu_vs_ra.csv"

    rows = []
    for aspect in sweep_config.aspect_ratios:
        for ra in sweep_config.rayleigh_values:
            summary = _read_run_summary(_run_dir(sweep_config, ra, aspect))
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


def tabulate_vrms_vs_ra(sweep_config: SweepConfig) -> Path:
    """Aggregate per-run summaries into Vrms(Ra) tidy CSV."""
    out_dir = Path(sweep_config.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vrms_vs_ra.csv"

    rows = []
    for aspect in sweep_config.aspect_ratios:
        for ra in sweep_config.rayleigh_values:
            summary = _read_run_summary(_run_dir(sweep_config, ra, aspect))
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


def plot_nu_vs_ra(sweep_config: SweepConfig) -> Optional[Path]:
    """Log-log Nu(Ra) per aspect ratio, with the Ra^(1/3) reference line.

    Returns the figure path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(sweep_config.output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "nu_vs_ra.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    for aspect in sweep_config.aspect_ratios:
        ras, nus, errs = [], [], []
        for ra in sweep_config.rayleigh_values:
            summary = _read_run_summary(_run_dir(sweep_config, ra, aspect))
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
        np.log10(min(sweep_config.rayleigh_values)),
        np.log10(max(sweep_config.rayleigh_values)),
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


def plot_vrms_vs_ra(sweep_config: SweepConfig) -> Optional[Path]:
    """Log-log V_rms(Ra) per aspect ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(sweep_config.output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "vrms_vs_ra.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    for aspect in sweep_config.aspect_ratios:
        ras, vrs, errs = [], [], []
        for ra in sweep_config.rayleigh_values:
            summary = _read_run_summary(_run_dir(sweep_config, ra, aspect))
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
            elif (run_dir / "manifest.yaml").exists():
                # Has work but no summary -> stalled or in-progress
                out[(ra, aspect)] = "in_progress"
            else:
                out[(ra, aspect)] = "not_started"
    return out


def view():
    """Display this module's workflow steps and config classes."""
    from underworld3.workflows import view as _wf_view
    _wf_view(sys.modules[__name__])
