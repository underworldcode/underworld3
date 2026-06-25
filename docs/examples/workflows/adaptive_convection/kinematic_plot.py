"""Diagnostic figure for a kinematic feature-motion run.

Reads ``timeseries.csv`` and plots the migration of the fault (θ_f) and ridge
(θ_MOR) surface azimuths against time, the velocity each feature experiences
(v·n̂ for the fault, surface-tangential v for the ridge), and the mesh-following
diagnostics (fault/bulk nearest-neighbour spacing ratio, folded count) — the
three things that together say "the feature migrated with the flow AND the
refinement followed it on a clean mesh".

Usage:
  python kinematic_plot.py --run <dir>
"""
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(D):
    rows = []
    with open(os.path.join(D, "timeseries.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def col(name):
        out = []
        for r in rows:
            v = r.get(name)
            try:
                out.append(float(v) if v not in (None, "", "---") else np.nan)
            except ValueError:
                out.append(np.nan)
        return np.array(out)

    return {k: col(k) for k in
            ("step", "t", "theta_f", "theta_MOR", "v_n_fault", "v_t_MOR",
             "fault_ratio", "folded", "vrms", "Nu")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    D = os.path.expanduser(args.run)
    d = _read(D)
    t = d["t"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    a = ax[0, 0]
    a.plot(t, d["theta_f"], "-o", ms=3, color="C3", label=r"$\theta_f$ (fault)")
    a.plot(t, d["theta_MOR"], "-s", ms=3, color="m", label=r"$\theta_{MOR}$ (ridge)")
    a.set_xlabel("time"); a.set_ylabel("surface azimuth (deg)")
    a.set_title("Feature migration"); a.legend(); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(t, d["v_n_fault"], "-o", ms=3, color="C3", label=r"$\langle v\cdot\hat n\rangle$ fault")
    a.plot(t, d["v_t_MOR"], "-s", ms=3, color="m", label=r"$\langle v\cdot\hat t\rangle$ ridge")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("time"); a.set_ylabel("experienced velocity")
    a.set_title("Velocity driving the motion"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(t, d["fault_ratio"], "-o", ms=3, color="C0")
    a.set_xlabel("time"); a.set_ylabel("fault/bulk NN-spacing ratio")
    a.set_title("Refinement following the moving fault (<1 = finer)")
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(t, d["folded"], "-x", ms=4, color="C1")
    a2.set_ylabel("folded elements", color="C1")
    a2.set_ylim(-0.5, max(1.5, np.nanmax(d["folded"]) + 0.5))

    a = ax[1, 1]
    a.plot(t, d["vrms"], "-o", ms=3, color="C2", label="vrms")
    a.set_xlabel("time"); a.set_ylabel("vrms", color="C2")
    a.set_title("Convective vigour"); a.grid(alpha=0.3)
    a3 = a.twinx()
    a3.plot(t, d["Nu"], "-s", ms=3, color="C4", label="Nu")
    a3.set_ylabel("Nu", color="C4")

    df = np.nanmax(d["theta_f"]) - np.nanmin(d["theta_f"])
    dm = np.nanmax(d["theta_MOR"]) - np.nanmin(d["theta_MOR"])
    fig.suptitle(f"Kinematic feature motion — fault swept {df:.1f}°, "
                 f"ridge swept {dm:.1f}°, max folded={int(np.nanmax(d['folded']))}",
                 fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join(D, "kinematic_motion.png")
    fig.savefig(out, dpi=130)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
