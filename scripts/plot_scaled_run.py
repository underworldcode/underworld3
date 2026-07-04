"""Plot the scaled Ra=ρg=1e5 run (diffuser only, no Winslow)."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay

import underworld3 as uw

D = "/tmp/mdump_scaled"
OUT = "/tmp/scaled_run_evolution.png"

mesh = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5, cellSize=1.0 / 16, qdegree=3)
t_soln = uw.discretisation.MeshVariable(
    "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")

ref_v = np.load(os.path.join(D, "step_0001.npz"))["coords"]


def cell_triangles_from_dm(dm):
    """Real cell-vertex connectivity from DMPlex (not Delaunay)."""
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    tri = []
    for c in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [p - pStart for p in closure
                 if pStart <= p < pEnd]
        if len(verts) == 3:
            tri.append(verts)
    return np.asarray(tri, dtype=np.int64)


mesh._deform_mesh(ref_v)
tri_v = cell_triangles_from_dm(mesh.dm)
print(f"Real cell triangles: {tri_v.shape[0]}")

# T-DOF triangulation still uses Delaunay (P3 DOF connectivity is
# more complex than vertex cells); filter by max edge length.
tref = np.asarray(t_soln.coords).copy()
tri_t = Delaunay(tref).simplices
ct = tref[tri_t].mean(axis=1)
keep = np.sqrt(ct[:, 0] ** 2 + ct[:, 1] ** 2) > 0.5 + 1e-3
pts = tref[tri_t]
emax = np.maximum.reduce([
    np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1),
    np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1),
    np.linalg.norm(pts[:, 0] - pts[:, 2], axis=1),
])
keep &= emax < 0.04  # P3 DOF spacing is ~1/(3·16) ≈ 0.021
tri_t = tri_t[keep]
print(f"T-DOF triangles after filter: {tri_t.shape[0]}")

r0 = np.sqrt(ref_v[:, 0] ** 2 + ref_v[:, 1] ** 2)
outer_idx = np.where(r0 > 0.99)[0]
outer_idx = outer_idx[np.argsort(
    np.arctan2(ref_v[outer_idx, 1], ref_v[outer_idx, 0]))]

picks = [1, 8, 16, 24, 32, 40]
T_LEVELS = np.linspace(0.0, 1.0, 21)

fig, axes = plt.subplots(2, len(picks),
                         figsize=(3.0 * len(picks), 6.5))

for col, s in enumerate(picks):
    data = np.load(os.path.join(D, f"step_{s:04d}.npz"))
    v_coords = data["coords"]
    T = data["T"]
    mesh._deform_mesh(v_coords)
    t_coords = np.asarray(t_soln.coords).copy()
    triang = Triangulation(t_coords[:, 0], t_coords[:, 1], tri_t)

    # Full annulus
    ax = axes[0, col]
    cf = ax.tricontourf(triang, T, levels=T_LEVELS,
                        cmap="RdBu_r", extend="both")
    ax.triplot(v_coords[:, 0], v_coords[:, 1], tri_v,
               color="black", lw=0.25, alpha=0.45)
    bnd = v_coords[outer_idx]; bnd = np.vstack([bnd, bnd[:1]])
    ax.plot(bnd[:, 0], bnd[:, 1], color="black", lw=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([]); ax.set_yticks([])
    bnd_r = np.sqrt(v_coords[outer_idx, 0] ** 2
                    + v_coords[outer_idx, 1] ** 2)
    h_max = np.abs(bnd_r - 1.0).max()
    ax.set_title(f"step {s}  |h|max={h_max:.3f}", fontsize=10)
    # Zoom on top
    ax = axes[1, col]
    cf = ax.tricontourf(triang, T, levels=T_LEVELS,
                        cmap="RdBu_r", extend="both")
    ax.triplot(v_coords[:, 0], v_coords[:, 1], tri_v,
               color="black", lw=0.5, alpha=0.5)
    ax.plot(bnd[:, 0], bnd[:, 1], color="black", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0.78, 1.06)
    ax.set_xticks([]); ax.set_yticks([])
    bnd_th = np.arctan2(v_coords[outer_idx, 1],
                        v_coords[outer_idx, 0])
    pole_i = int(np.argmin(np.abs(bnd_th - np.pi / 2)))
    h_pole = bnd_r[pole_i] - 1.0
    ax.set_title(f"h_pole={h_pole:+.3e}", fontsize=9)

cbar = fig.colorbar(cf, ax=axes.ravel().tolist(),
                    shrink=0.75, pad=0.02, label="T")
fig.suptitle(
    "Free-surface convection with PROPER scaling Ra=ρg=1e5 "
    "(no Winslow, CN+clamp)\n"
    "Restoring force from -ρg·r̂ body force keeps |h|max≈0.13 "
    "(4× smaller than unscaled). Mass drift ΔA/A=-25% remains "
    "(separate issue).",
    fontsize=11)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved {OUT}")
