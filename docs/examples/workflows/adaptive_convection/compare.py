"""Compare adaptive-convection runs at MATCHED PHYSICAL TIME.

Runs on different meshes take different dt, so they must be compared by
physical time ``t``, never by step number (see
``feedback_debug_adaptive_solver_method``). This reads each run's
``timeseries.csv`` (written by ``underworld3.workflows.Run``) and plots
Nu(t), vrms(t) and the mesh-quality time series (folded count,
area-ratio) on a shared time axis.

Use it to validate an adaptive run against a *resolved arbiter* (a
uniform mesh finer in BOTH space and time): if the adaptive Nu(t)/vrms(t)
track the arbiter, the adaptation is faithful.

Usage:
  python compare.py --sim-dir ~/+Simulations/AdaptiveConvection \
      --runs wf_adapt_res24_R5 wf_arbiter_res48_uniform \
      --labels "adaptive R5" "arbiter res48" --out compare.png
"""
from __future__ import annotations

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_timeseries(path):
    """Read a workflows ``timeseries.csv`` into a dict of column arrays."""
    cols: dict[str, list] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(
                    float("nan") if v in ("", "---", None) else float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default="~/+Simulations/AdaptiveConvection")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run sub-directory names (each holds timeseries.csv).")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args(argv)

    sim = os.path.expanduser(args.sim_dir)
    labels = args.labels or args.runs
    data = []
    for run in args.runs:
        p = os.path.join(sim, run, "timeseries.csv")
        if not os.path.exists(p):
            print(f"  [skip] no timeseries at {p}")
            continue
        data.append((run, read_timeseries(p)))

    panels = [
        ("Nu", "Nusselt number  Nu(t)"),
        ("vrms", "RMS velocity  vrms(t)"),
        ("area_ratio", "mesh cell-area ratio (max/min)"),
        ("folded", "folded elements"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), panels):
        for (run, d), lab in zip(data, labels):
            if col in d and "t" in d:
                ax.plot(d["t"], d[col], marker=".", ms=3, lw=1.2, label=lab)
        ax.set_title(title)
        ax.set_xlabel("physical time  t")
        ax.grid(alpha=0.3)
        if col == "folded":
            ax.set_ylim(-0.5, max(1.0, ax.get_ylim()[1]))
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle("Adaptive convection — matched physical time", fontsize=13)
    fig.tight_layout()
    out = os.path.join(sim, args.out)
    fig.savefig(out, dpi=110)
    print("->", out)

    # Quick numeric summary: late-time means (last 25% of each series).
    print("\nLate-time means (last 25% by t):")
    for (run, d), lab in zip(data, labels):
        if "t" not in d:
            continue
        t = d["t"]
        late = t >= (t.min() + 0.75 * (t.max() - t.min()))
        nu = np.nanmean(d["Nu"][late]) if "Nu" in d else float("nan")
        vr = np.nanmean(d["vrms"][late]) if "vrms" in d else float("nan")
        fold = int(np.nanmax(d["folded"])) if "folded" in d else -1
        print(f"  {lab:<22} t_end={t.max():.4f}  Nu={nu:+.3f}  "
              f"vrms={vr:.3f}  max_folded={fold}")


if __name__ == "__main__":
    main()
