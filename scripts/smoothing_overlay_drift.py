"""Diagnostic plot: where exactly does the parallel run disagree
with serial?

For each parallel vertex, find the nearest-neighbour serial vertex
(initial coords are bit-identical across runs, so the nearest match
is the same physical vertex). Plot scatter coloured by drift
magnitude, overlaid with the rank-partition seam.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def panel(ax, parallel, serial, tree_serial_pre, size, with_seam=True):
    coords_p, pre_p, rank_p = (
        parallel["coords"], parallel["pre"], parallel["rank"])
    # Match by initial coord — they're identical between runs.
    _, idx = tree_serial_pre.query(pre_p, k=1)
    matched_serial = serial["coords"][idx]
    drift = np.linalg.norm(coords_p - matched_serial, axis=1)
    sc = ax.scatter(coords_p[:, 0], coords_p[:, 1],
                    c=drift, s=30, cmap="magma",
                    vmin=0, vmax=max(drift.max(), 1e-3))
    plt.colorbar(sc, ax=ax, label="drift |Δx_parallel − Δx_serial|")
    if with_seam:
        # Overlay rank-cut: which vertices are adjacent in cKDTree
        # to a vertex owned by a different rank?
        tree_p = cKDTree(coords_p)
        _, nbrs = tree_p.query(coords_p, k=7)
        on_seam = np.zeros(len(rank_p), dtype=bool)
        for i in range(len(rank_p)):
            if (rank_p[nbrs[i, 1:]] != rank_p[i]).any():
                on_seam[i] = True
        ax.scatter(coords_p[on_seam, 0], coords_p[on_seam, 1],
                   facecolors="none", edgecolors="cyan",
                   s=120, linewidth=0.8, label="near rank cut")
    ax.set_aspect("equal")
    ax.set_title(f"np={size}  max drift {drift.max():.2e}  "
                 f"mean {drift.mean():.2e}")
    ax.legend(loc="upper right", fontsize=8)
    return drift


def main():
    d1 = np.load("/tmp/winslow_overlay_np1.npz")
    d2 = np.load("/tmp/winslow_overlay_np2.npz")
    d4 = np.load("/tmp/winslow_overlay_np4.npz")
    tree1 = cKDTree(d1["pre"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    panel(axes[0], d2, d1, tree1, 2)
    panel(axes[1], d4, d1, tree1, 4)
    fig.suptitle(
        "Parallel-vs-serial drift after smooth_mesh_interior "
        "(cyan rings = vertices adjacent to a different rank)",
        fontsize=12)
    fig.tight_layout()
    out = "/tmp/winslow_drift_map.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
