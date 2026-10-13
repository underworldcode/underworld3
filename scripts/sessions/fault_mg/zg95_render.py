"""Render a zg95 checkpoint offline: T on the base from the global h5 (P2 cells
subdivided into four), with the trench and the surface profiles of that step
when the driver wrote them.

python zg95_render.py <run dir> <index> [out.png]
"""
import glob
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

run, idx = sys.argv[1], int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run, f"render_{idx:04d}.png")
with h5py.File(os.path.join(run, f"zg95.mesh.T_b.{idx:05d}.h5")) as h:
    T = np.asarray(h["fields/T_b"]).ravel()
    X = np.asarray(h["fields/coordinates"])
    C = np.asarray(h["fields/cells"]).astype(int)
# P2 triangle node order: 3 vertices then 3 edge midpoints (opposite order per PETSc: 
# midpoints follow the vertex order of the edges 01, 12, 20); subdivide into 4
v0, v1, v2, m01, m12, m20 = (C[:, k] for k in range(6))
tris = np.vstack([np.column_stack([v0, m01, m20]), np.column_stack([m01, v1, m12]),
                  np.column_stack([m20, m12, v2]), np.column_stack([m01, m12, m20])])
state = {}
sf = os.path.join(run, f"state_{idx:04d}.json")
if os.path.exists(sf):
    state = json.load(open(sf))
surf = os.path.join(run, f"surface_{idx:04d}.npz")
has_surf = os.path.exists(surf)
fig, axes = plt.subplots(1, 3 if has_surf else 1, figsize=(18 if has_surf else 7, 5.8), squeeze=False)
ax = axes[0, 0]
tc = ax.tripcolor(mtri.Triangulation(X[:, 0], X[:, 1], tris), T, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1)
if has_surf:
    tr = np.load(surf)["trace"]
    ax.plot(tr[:, 0], tr[:, 1], color="k", lw=1.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"step {idx}, t = {state.get('t', float('nan')):.4f}: T on the base", fontsize=9, loc="left")
fig.colorbar(tc, ax=ax, shrink=0.8)
if has_surf:
    d = np.load(surf)
    th0 = np.degrees(state.get("theta_t", 0.0))
    ax = axes[0, 1]
    ax.plot(np.degrees(d["theta"]), d["vt"], lw=1); ax.axvline(th0, color="#c0392b", lw=0.8, ls="--")
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("surface tangential velocity"); ax.grid(alpha=0.3)
    ax = axes[0, 2]
    ax.plot(np.degrees(d["theta_h"]), d["h"], lw=1); ax.axvline(th0, color="#c0392b", lw=0.8, ls="--")
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("dynamic topography"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
