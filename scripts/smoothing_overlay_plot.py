"""Overlay scatter plot of vertex positions from serial and parallel
runs of ``smooth_mesh_interior``. Look for artefacts visually.

Reads /tmp/winslow_overlay_np{1,2,4}.npz produced by
``smoothing_overlay_check.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def load(np_size: int):
    path = f"/tmp/winslow_overlay_np{np_size}.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["coords"], d["is_bnd"], d["rank"]


def main():
    serial = load(1)
    if serial is None:
        print("Missing /tmp/winslow_overlay_np1.npz — run serial first")
        sys.exit(1)
    coords_s, bnd_s, _ = serial

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: serial alone vs np=2
    par2 = load(2)
    ax = axes[0]
    ax.scatter(coords_s[:, 0], coords_s[:, 1],
               s=20, c="tab:blue", marker="o",
               label=f"serial ({coords_s.shape[0]} verts)",
               edgecolor="none", alpha=0.8)
    if par2 is not None:
        coords_p2, bnd_p2, rank_p2 = par2
        # Colour parallel points by rank, plot as crosses on top
        for r in np.unique(rank_p2):
            sel = rank_p2 == r
            ax.scatter(coords_p2[sel, 0], coords_p2[sel, 1],
                       s=35, marker="x",
                       c=["tab:red", "tab:orange"][r % 2],
                       label=f"np=2 rank {r} ({int(sel.sum())})",
                       linewidth=1.0, alpha=0.9)
    ax.set_aspect("equal")
    ax.set_title("serial (o) vs np=2 (x by rank)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Right: serial vs np=4 if available, else zoom on a corner
    par4 = load(4)
    ax = axes[1]
    ax.scatter(coords_s[:, 0], coords_s[:, 1],
               s=20, c="tab:blue", marker="o",
               label=f"serial", edgecolor="none", alpha=0.8)
    if par4 is not None:
        coords_p4, bnd_p4, rank_p4 = par4
        cmap = plt.get_cmap("tab10")
        for r in np.unique(rank_p4):
            sel = rank_p4 == r
            ax.scatter(coords_p4[sel, 0], coords_p4[sel, 1],
                       s=35, marker="x", c=[cmap(r)],
                       label=f"np=4 rank {r} ({int(sel.sum())})",
                       linewidth=1.0, alpha=0.9)
        ax.set_title("serial (o) vs np=4 (x by rank)")
    else:
        # No np=4 available — zoom into a corner of the np=2 view
        ax.set_title("(np=4 results missing)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    out = "/tmp/winslow_serial_vs_parallel.png"
    fig.suptitle(
        "Final mesh vertices after smooth_mesh_interior — "
        "perfect overlap ⇒ no parallel artefacts",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
