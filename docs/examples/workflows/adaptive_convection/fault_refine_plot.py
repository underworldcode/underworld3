"""Plot achieved fault refinement under live convection.

Reads a fault run's ``timeseries.csv`` and plots the **fault/bulk
nearest-neighbour spacing ratio** over physical time alongside the
convection vigour (vrms, Nu) and node count near the fault. The fault
ratio is the diagnostic of record: < 1 means the mesh is finer at the
fault than in the bulk; the equilibrium value is the "creation cap" the
mmpde mover reaches from a uniform base with the anisotropic tensor
metric (a resolved fault needs a gmsh refine_lines base — follow-up).

Usage:
  python fault_refine_plot.py --sim-dir ~/+Simulations/FaultConvection \
      --runs wf_fault_res24_Rf8 --labels "Rf=8" --out fault_refine.png
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
    cols: dict[str, list] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(
                    float("nan") if v in ("", "---", None) else float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default="~/+Simulations/FaultConvection")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="fault_refine.png")
    args = ap.parse_args(argv)

    sim = os.path.expanduser(args.sim_dir)
    labels = args.labels or args.runs
    data = []
    for run in args.runs:
        p = os.path.join(sim, run, "timeseries.csv")
        if os.path.exists(p):
            data.append(read_timeseries(p))
        else:
            print(f"  [skip] {p}")
            labels = [l for l, r in zip(labels, args.runs) if r != run]

    panels = [
        ("fault_ratio", "fault/bulk NN-spacing ratio  (<1 = finer at fault)"),
        ("n_fault", "nodes within 1.5·width of the fault"),
        ("vrms", "RMS velocity  vrms(t)"),
        ("folded", "folded elements"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), panels):
        for d, lab in zip(data, labels):
            if col in d and "t" in d:
                ax.plot(d["t"], d[col], marker=".", ms=3, lw=1.2, label=lab)
        ax.set_title(title)
        ax.set_xlabel("physical time  t")
        ax.grid(alpha=0.3)
        if col == "fault_ratio":
            ax.axhline(1.0, color="grey", ls="--", lw=0.8)
            ax.set_ylim(0.0, 1.1)
        if col == "folded":
            ax.set_ylim(-0.5, max(1.0, ax.get_ylim()[1]))
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle("Anisotropic-tensor fault refinement under live convection "
                 "(uniform base)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(sim, args.out)
    fig.savefig(out, dpi=110)
    print("->", out)

    print("\nEquilibrium fault refinement (last 25% by t):")
    for d, lab in zip(data, labels):
        if "t" not in d:
            continue
        t = d["t"]
        late = t >= (t.min() + 0.75 * (t.max() - t.min()))
        fr = np.nanmean(d["fault_ratio"][late]) if "fault_ratio" in d else float("nan")
        nf = np.nanmean(d["n_fault"][late]) if "n_fault" in d else float("nan")
        fold = int(np.nanmax(d["folded"])) if "folded" in d else -1
        print(f"  {lab:<14} fault/bulk={fr:.3f} (~{1/fr:.2f}x finer)  "
              f"n_fault~{nf:.0f}  max_folded={fold}")


if __name__ == "__main__":
    main()
