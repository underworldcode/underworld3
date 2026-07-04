"""BE (θ=1.0) vs CN (θ=0.5) — mesh + T evolution at the same step
numbers. Two rows are the two runs; columns are 6 sample steps.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay

import underworld3 as uw

D_BE = "/tmp/mdump_uw_long"
D_CN = "/tmp/mdump_cn_long"
OUT = "/tmp/be_vs_cn.png"

dumps = sorted(glob.glob(os.path.join(D_BE, "step_*.npz")))
max_step = int(os.path.basename(dumps[-1]).split("_")[1].split(".")[0])

# Same mesh + t_soln for both runs.
mesh = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 16, qdegree=3)
t_soln = uw.discretisation.MeshVariable(
    "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")

# Reference vertex triangulation (initial state)
ref_v = np.load(os.path.join(D_BE, "step_0001.npz"))["coords"]
tri_v = Delaunay(ref_v).simplices
cv = ref_v[tri_v].mean(axis=1)
tri_v = tri_v[np.sqrt(cv[:, 0] ** 2 + cv[:, 1] ** 2) > 0.5 + 1e-3]

# Reference DOF triangulation
mesh._deform_mesh(ref_v)
tref = np.asarray(t_soln.coords).copy()
tri_t = Delaunay(tref).simplices
ct = tref[tri_t].mean(axis=1)
tri_t = tri_t[np.sqrt(ct[:, 0] ** 2 + ct[:, 1] ** 2) > 0.5 + 1e-3]

# Outer-rim indices (closed curve in θ order)
r0 = np.sqrt(ref_v[:, 0] ** 2 + ref_v[:, 1] ** 2)
outer_idx = np.where(r0 > 0.99)[0]
outer_idx = outer_idx[np.argsort(
    np.arctan2(ref_v[outer_idx, 1], ref_v[outer_idx, 0]))]

picks = [1, 8, 16, 24, 32, 40]
T_LEVELS = np.linspace(0.0, 1.0, 21)

fig, axes = plt.subplots(2, len(picks),
                         figsize=(3.0 * len(picks), 6.5))

for col, s in enumerate(picks):
    for row, (label, d) in enumerate(
            [("BE θ=1.0", D_BE), ("CN θ=0.5", D_CN)]):
        ax = axes[row, col]
        data = np.load(os.path.join(d, f"step_{s:04d}.npz"))
        v_coords = data["coords"]
        T = data["T"]
        mesh._deform_mesh(v_coords)
        t_coords = np.asarray(t_soln.coords).copy()
        triang = Triangulation(t_coords[:, 0], t_coords[:, 1], tri_t)
        cf = ax.tricontourf(triang, T, levels=T_LEVELS,
                            cmap="RdBu_r", extend="both")
        ax.triplot(v_coords[:, 0], v_coords[:, 1], tri_v,
                   color="black", lw=0.25, alpha=0.45)
        bnd = v_coords[outer_idx]
        bnd = np.vstack([bnd, bnd[:1]])
        ax.plot(bnd[:, 0], bnd[:, 1], color="black", lw=0.9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xticks([]); ax.set_yticks([])
        bnd_r = np.sqrt(v_coords[outer_idx, 0] ** 2
                        + v_coords[outer_idx, 1] ** 2)
        h_pole = bnd_r.max() - 1.0
        h_pole_min = bnd_r.min() - 1.0
        ttl = f"step {s}"
        if col == 0:
            ttl = f"[{label}]  " + ttl
        ax.set_title(
            f"{ttl}  |h|_max={max(abs(h_pole), abs(h_pole_min)):.3f}",
            fontsize=9)

cbar = fig.colorbar(cf, ax=axes.ravel().tolist(),
                    shrink=0.75, pad=0.02, label="T")
fig.suptitle(
    "BE vs CN diffusion at identical settings — uw.meshing."
    "smooth_mesh_interior every 2 steps, monotone=clamp\n"
    "res=16 Annulus, Ra=1e4, rk4 free surface, 40 steps  "
    "|  trajectories diverge slowly; CN rings a bit more by step 40",
    fontsize=11)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved {OUT}")
