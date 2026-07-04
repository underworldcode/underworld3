"""Mesh + T-field evolution from /tmp/mdump_uw_long. Rebuild the
mesh once with the same topology, deform it to each step's vertex
coords, and read t_soln.coords to plot T at the P3 lagrange nodes.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay

import underworld3 as uw

D = "/tmp/mdump_uw_long"
OUT = "/tmp/mdump_uw_long_evolution_T.png"

dumps = sorted(glob.glob(os.path.join(D, "step_*.npz")))
max_step = int(os.path.basename(dumps[-1]).split("_")[1].split(".")[0])
print(f"Found {len(dumps)} dumps, last step = {max_step}")

# Build same mesh + t_soln as the zoo script (res=16, P3 T).
mesh = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 16, qdegree=3)
t_soln = uw.discretisation.MeshVariable(
    "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")

# Triangulation for the mesh wireframe (built from initial mesh
# vertices once and kept fixed; topology is preserved across the run).
ref = np.load(os.path.join(D, "step_0001.npz"))
ref_v = ref["coords"]
tri_v = Delaunay(ref_v).simplices
keep_v = np.sqrt(ref_v[tri_v].mean(axis=1).sum(axis=1) ** 2 - 0) > 0
# Actually drop triangles whose centroid is in the hole.
centroids_v = ref_v[tri_v].mean(axis=1)
keep_v = np.sqrt(centroids_v[:, 0] ** 2 + centroids_v[:, 1] ** 2) > 0.5 + 1e-3
tri_v = tri_v[keep_v]

# T-DOF triangulation: built once from the initial deformed-to-ref
# t_soln.coords; cell connectivity for P3 lagrange nodes is fixed.
mesh._deform_mesh(ref_v)
tcoords_ref = np.asarray(t_soln.coords).copy()
tri_t = Delaunay(tcoords_ref).simplices
centroids_t = tcoords_ref[tri_t].mean(axis=1)
keep_t = np.sqrt(centroids_t[:, 0] ** 2 + centroids_t[:, 1] ** 2) > 0.5 + 1e-3
tri_t = tri_t[keep_t]

# Pick 6 steps roughly evenly spaced (same as before)
picks = [1, max(1, max_step // 5), max(1, 2 * max_step // 5),
         max(1, 3 * max_step // 5), max(1, 4 * max_step // 5), max_step]
picks = sorted(set(picks))[:6]
print(f"Plotting steps: {picks}")

fig, axes = plt.subplots(2, len(picks),
                         figsize=(3.0 * len(picks), 6.5))

T_LEVELS = np.linspace(0.0, 1.0, 21)

for col, s in enumerate(picks):
    data = np.load(os.path.join(D, f"step_{s:04d}.npz"))
    v_coords = data["coords"]
    T = data["T"]
    # Deform mesh to step's vertex coords -> read updated T-DOF coords.
    mesh._deform_mesh(v_coords)
    t_coords = np.asarray(t_soln.coords).copy()

    # Outer rim path (red).
    r0 = np.sqrt(ref_v[:, 0] ** 2 + ref_v[:, 1] ** 2)
    outer_idx = np.where(r0 > 0.99)[0]
    outer_idx = outer_idx[np.argsort(
        np.arctan2(ref_v[outer_idx, 1], ref_v[outer_idx, 0]))]
    bnd = v_coords[outer_idx]
    bnd = np.vstack([bnd, bnd[:1]])

    for row, (zoom, lbl) in enumerate(
            [(False, "full"), (True, "zoom")]):
        ax = axes[row, col]
        # T field via tricontourf at P3 DOFs.
        triang = Triangulation(t_coords[:, 0], t_coords[:, 1], tri_t)
        cf = ax.tricontourf(triang, T, levels=T_LEVELS,
                            cmap="RdBu_r", extend="both")
        # Mesh wireframe on top (light).
        ax.triplot(v_coords[:, 0], v_coords[:, 1], tri_v,
                   color="black", lw=0.25, alpha=0.45)
        ax.plot(bnd[:, 0], bnd[:, 1], color="black", lw=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if zoom:
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(0.78, 1.06)
            bnd_r = np.sqrt(v_coords[outer_idx, 0] ** 2
                            + v_coords[outer_idx, 1] ** 2)
            bnd_th = np.arctan2(v_coords[outer_idx, 1],
                                v_coords[outer_idx, 0])
            pole_i = int(np.argmin(np.abs(bnd_th - np.pi / 2)))
            h_pole = bnd_r[pole_i] - 1.0
            ax.set_title(f"h_pole = {h_pole:+.3e}", fontsize=9)
        else:
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.08)
            ax.set_title(f"step {s}", fontsize=10)

# Single colorbar at the right.
cbar = fig.colorbar(cf, ax=axes.ravel().tolist(),
                    shrink=0.75, pad=0.02, label="T")
fig.suptitle(
    "T field + mesh evolution — uw.meshing.smooth_mesh_interior, "
    "edge-closure pinning\n"
    "rk4 + θ=1.0 (BE) + monotone=clamp + smoother every 2 steps  "
    f"|  res=16, Ra=1e4, {max_step} steps\n"
    "top: full annulus  •  bottom: zoom on deforming surface "
    "(black curve = outer rim)",
    fontsize=11)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved {OUT}")
