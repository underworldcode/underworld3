"""Side-by-side mesh evolution: in-script winslow_smooth_interior
vs uw.meshing.smooth_mesh_interior over the same 6-step rk4 +
free-surface convection run.

Top row: in-script smoother.
Bottom row: uw smoother.
Smoother fires at step 2, 4, 6 (--winslow-every-n-steps 2).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

IN_DIR = "/tmp/mdump_inscript"
UW_DIR = "/tmp/mdump_uw"
OUT_PNG = "/tmp/smoother_mesh_evolution.png"

# Load step 1 (pre-smoothing) from in-script run and use the
# resulting Delaunay triangulation as the topology reference.
# (Topology is preserved across steps; only coords move.)
ref = np.load(os.path.join(IN_DIR, "step_0001.npz"))
ref_coords = ref["coords"]
# Use a thin annular trim to avoid Delaunay's convex-hull artefact
# spanning the inner hole.
r_initial = np.sqrt(ref_coords[:, 0] ** 2 + ref_coords[:, 1] ** 2)
# Build triangulation against the undeformed coords.
tri = Delaunay(ref_coords)
# Drop triangles that cross the inner hole (centroid radius too small).
triangles = tri.simplices
centroids = ref_coords[triangles].mean(axis=1)
cent_r = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
keep = cent_r > 0.5 + 1e-3
triangles = triangles[keep]


def load_step(d, s):
    return np.load(os.path.join(d, f"step_{s:04d}.npz"))


steps = [1, 2, 3, 4, 5, 6]
fig, axes = plt.subplots(2, len(steps), figsize=(3.4 * len(steps), 7.2))

# Zoom window — top of annulus, where the deformed surface lives.
# (r ∈ [0.85, 1.05], θ ∈ [60°, 120°]). Outer radius is 1.0;
# h_pole ~ -3e-2 by step 6, so r at the pole reaches ~0.97.
ZX = (-0.55, 0.55)
ZY = (0.78, 1.05)

for col, s in enumerate(steps):
    for row, (lbl, d) in enumerate(
            [("in-script", IN_DIR), ("uw.meshing", UW_DIR)]):
        ax = axes[row, col]
        data = load_step(d, s)
        c = data["coords"]
        ax.triplot(c[:, 0], c[:, 1], triangles,
                   color="black", lw=0.5)
        # Highlight the actual outer boundary (vertices closest to r=1
        # before deformation, traced in run order).
        r0 = np.sqrt(ref_coords[:, 0] ** 2 + ref_coords[:, 1] ** 2)
        bnd_mask = r0 > 0.99
        bnd_idx = np.where(bnd_mask)[0]
        ang = np.arctan2(ref_coords[bnd_idx, 1],
                         ref_coords[bnd_idx, 0])
        order = np.argsort(ang)
        bnd_curve = c[bnd_idx[order]]
        # Close the curve
        bnd_curve = np.vstack([bnd_curve, bnd_curve[:1]])
        ax.plot(bnd_curve[:, 0], bnd_curve[:, 1],
                color="tab:red", lw=1.2, alpha=0.8)
        smoother_fired = (s % 2) == 0
        ttl = f"step {s}"
        if smoother_fired:
            ttl += "  (smoother fired)"
        if col == 0:
            ttl = f"[{lbl}]  " + ttl
        ax.set_title(ttl, fontsize=10,
                     color="tab:red" if smoother_fired else "black")
        ax.set_aspect("equal")
        ax.set_xlim(*ZX)
        ax.set_ylim(*ZY)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(
    "Top-of-annulus zoom — mesh evolution under rk4 + θ=1.0 + "
    "monotone=clamp + smooth_every=2 steps\n"
    "(top row: in-script Winslow / bottom row: "
    "uw.meshing.smooth_mesh_interior; red curve: deformed surface)",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved {OUT_PNG}")

# Also compute per-step max coord-diff between the two smoothers
# (serial; in-script is correct in serial too, so this is just a
# sanity diff)
print("\nPer-step max |Δcoord| in-script vs uw:")
for s in steps:
    c1 = load_step(IN_DIR, s)["coords"]
    c2 = load_step(UW_DIR, s)["coords"]
    d = np.linalg.norm(c1 - c2, axis=1)
    print(f"  step {s}: max={d.max():.3e}  mean={d.mean():.3e}")
