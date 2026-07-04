"""Mesh evolution under fixed-pinning uw smoother on a longer run.

Whole-annulus view at sparse step intervals + a zoom on the top.
Surface deformation curve (the actual outer rim) is drawn in red.
"""
from __future__ import annotations
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

D = "/tmp/mdump_uw_long"
OUT = "/tmp/mdump_uw_long_evolution.png"

dumps = sorted(glob.glob(os.path.join(D, "step_*.npz")))
if not dumps:
    print(f"No dumps in {D}")
    sys.exit(1)
max_step = int(os.path.basename(dumps[-1]).split("_")[1].split(".")[0])
print(f"Found {len(dumps)} dumps, last step = {max_step}")

# Build reference triangulation from step 1 coords; topology is
# preserved across the run.
ref = np.load(os.path.join(D, "step_0001.npz"))
ref_coords = ref["coords"]
tri = Delaunay(ref_coords)
triangles = tri.simplices
centroids = ref_coords[triangles].mean(axis=1)
keep = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2) > 0.5 + 1e-3
triangles = triangles[keep]

# Outer-rim vertex indices (by initial radius)
r0 = np.sqrt(ref_coords[:, 0] ** 2 + ref_coords[:, 1] ** 2)
outer_idx = np.where(r0 > 0.99)[0]
ang0 = np.arctan2(ref_coords[outer_idx, 1], ref_coords[outer_idx, 0])
outer_order = outer_idx[np.argsort(ang0)]

# Pick 6 steps roughly evenly spaced including first + last
picks = sorted(set([1, max(1, max_step // 5),
                    max(1, 2 * max_step // 5),
                    max(1, 3 * max_step // 5),
                    max(1, 4 * max_step // 5),
                    max_step]))
picks = picks[:6]
print(f"Plotting steps: {picks}")

fig, axes = plt.subplots(2, len(picks),
                         figsize=(3.0 * len(picks), 6.5))
for col, s in enumerate(picks):
    data = np.load(os.path.join(D, f"step_{s:04d}.npz"))
    c = data["coords"]
    # Full annulus view
    ax = axes[0, col]
    ax.triplot(c[:, 0], c[:, 1], triangles,
               color="black", lw=0.3)
    bnd = c[outer_order]
    bnd = np.vstack([bnd, bnd[:1]])
    ax.plot(bnd[:, 0], bnd[:, 1], color="tab:red", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {s}", fontsize=10)
    # Zoom on the top
    ax = axes[1, col]
    ax.triplot(c[:, 0], c[:, 1], triangles,
               color="black", lw=0.5)
    ax.plot(bnd[:, 0], bnd[:, 1], color="tab:red", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0.78, 1.06)
    ax.set_xticks([]); ax.set_yticks([])
    # Compute h_pole at the top vertex (θ closest to π/2)
    bnd_r = np.sqrt(c[outer_order, 0] ** 2 + c[outer_order, 1] ** 2)
    bnd_th = np.arctan2(c[outer_order, 1], c[outer_order, 0])
    pole_i = int(np.argmin(np.abs(bnd_th - np.pi / 2)))
    h_pole = bnd_r[pole_i] - 1.0
    ax.set_title(f"h_pole = {h_pole:+.3e}", fontsize=9)

fig.suptitle(
    f"Mesh evolution with uw.meshing.smooth_mesh_interior "
    f"(edge-closure pinning)\n"
    f"rk4 + θ=1.0 (BE) + monotone=clamp + smoother every 2 steps  "
    f"|  res=16, Ra=1e4, {max_step} steps total\n"
    f"top: full annulus  •  bottom: zoom on deforming surface "
    f"(red curve)",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved {OUT}")
