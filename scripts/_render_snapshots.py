"""Render all snapshots from a stagnant_lid_adapt_loop run as a
grid of mesh-on-T panels.
"""
from __future__ import annotations
import os
import re
import glob
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

import underworld3 as uw


p = argparse.ArgumentParser()
p.add_argument("--out-dir", type=str, required=True)
p.add_argument("--label", type=str, default="snapshots")
args = p.parse_args()

SRC = args.out_dir
DIAG = os.path.join(SRC, "diagnostics")
os.makedirs(DIAG, exist_ok=True)

# Discover snapshots
init_path = os.path.join(SRC, "init.mesh.00000.h5")
step_files = sorted(
    glob.glob(os.path.join(SRC, "step*.mesh.00000.h5")))
snapshots = []
if os.path.exists(init_path):
    snapshots.append(("init", 0))
for f in step_files:
    m = re.search(r"step(\d+)\.mesh\.00000\.h5$",
                   os.path.basename(f))
    if m:
        snapshots.append((f"step{int(m.group(1)):04d}",
                          int(m.group(1))))
snapshots.sort(key=lambda x: x[1])
print(f"  snapshots: {[s[0] for s in snapshots]}", flush=True)


def cell_quality(mesh):
    coords = np.asarray(mesh.X.coords)
    dm = mesh.dm
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, _ = dm.getDepthStratum(0)
    areas = np.empty(cEnd - cStart, dtype=float)
    for k, c in enumerate(range(cStart, cEnd)):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [pp for pp in closure
                 if pStart <= pp < pStart + coords.shape[0]]
        v = [coords[pp - pStart] for pp in verts[:3]]
        v0, v1, v2 = v
        areas[k] = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1])
                              - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    return float(areas.min() / areas.mean())


def mesh_edges(mesh):
    dm = mesh.dm
    coords = np.asarray(mesh.X.coords)
    pStart, _ = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    segs = np.empty((eEnd - eStart, 2, 2), dtype=float)
    for k, e in enumerate(range(eStart, eEnd)):
        cone = dm.getCone(e)
        segs[k, 0] = coords[cone[0] - pStart]
        segs[k, 1] = coords[cone[1] - pStart]
    return segs


def render_one(ax, label):
    mesh = uw.discretisation.Mesh(
        os.path.join(SRC, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    T.read_timestep(label, "T_v2p1", 0, outputPath=SRC)
    q = cell_quality(mesh)
    Xc = np.asarray(T.coords)
    Tv = np.asarray(T.data[:, 0])
    tri = Triangulation(Xc[:, 0], Xc[:, 1])
    cx = Xc[tri.triangles, 0].mean(axis=1)
    cy = Xc[tri.triangles, 1].mean(axis=1)
    rcen = np.sqrt(cx**2 + cy**2)
    mask = (rcen > 1.0 + 1e-6) | (rcen < 0.5 - 1e-6)
    tri.set_mask(mask)
    ax.tripcolor(tri, Tv, cmap="RdBu_r", shading="gouraud",
                 vmin=0, vmax=1)
    ax.add_collection(LineCollection(mesh_edges(mesh),
                                       colors="#202020",
                                       linewidths=0.3,
                                       alpha=0.7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}  q={q:.3f}", fontsize=10)
    return q


N = len(snapshots)
ncols = min(5, N)
nrows = (N + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 4.2, nrows * 4.2),
                          constrained_layout=True,
                          squeeze=False)
qs = []
for i, (label, _) in enumerate(snapshots):
    r, c = divmod(i, ncols)
    qs.append((label, render_one(axes[r, c], label)))
    print(f"  rendered {label}  q={qs[-1][1]:.3f}", flush=True)
# blank unused axes
for j in range(N, nrows * ncols):
    r, c = divmod(j, ncols)
    axes[r, c].axis("off")

fig.suptitle(
    f"snapshots from {os.path.basename(SRC)}  "
    f"(RdBu_r, q = min-cell / mean-cell area)",
    fontsize=12)
out = os.path.join(DIAG, f"{args.label}_grid.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"  → {out}", flush=True)
