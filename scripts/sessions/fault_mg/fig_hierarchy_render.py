"""Render the actual split-level hierarchy near the fault: one panel per level,
coarsest to finest, mesh edges with the Plus and Minus facets in two colours
(the slit opened by a small offset so the two copies are visible).

python fig_hierarchy_render.py <k> <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import underworld3 as uw

k = int(sys.argv[1])
out = sys.argv[2]
A_, B_ = np.array([0.30, 0.45]), np.array([0.70, 0.55])
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1,
                                         refinement=1, qdegree=2)
pts3 = np.column_stack([np.linspace(A_[0], B_[0], 9), np.linspace(A_[1], B_[1], 9), np.zeros(9)])
surf = uw.meshing.Surface("g", base, pts3)
surf.discretize()
child = base.adapt(surf.refinement_metric_function(h_near=0.05 / 2**k, h_far=0.05, width=0.08,
                                                   profile="linear"), max_levels=k)
mesh = child.add_fault([("Flt", np.vstack([A_, B_]))])
levels = list(mesh._custom_mg_coarse_meshes) + [mesh]


def edges_of(m, offset):
    """all edges as segments; fault-side facets returned separately, shifted
    by ``offset`` along the fault normal so the two copies separate"""
    dm = m.dm
    eS, eE = dm.getDepthStratum(1)
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    n = np.array([-(B_ - A_)[1], (B_ - A_)[0]])
    n /= np.linalg.norm(n)
    sides = {}
    for side, sgn in (("Plus", 1.0), ("Minus", -1.0)):
        name = f"Flt{side}"
        if name in [b.name for b in m.boundaries]:
            lab = dm.getLabel(name)
            value = m.boundaries[name].value
            if lab and lab.getStratumSize(value) > 0:
                sides[side] = (set(int(e) for e in lab.getStratumIS(value).getIndices()
                                   if eS <= int(e) < eE), sgn)
    segs, fsegs = [], {"Plus": [], "Minus": []}
    for e in range(eS, eE):
        a, b = (int(q) - vS for q in dm.getCone(e))
        seg = np.array([X[a], X[b]])
        placed = False
        for side, (es, sgn) in sides.items():
            if e in es:
                fsegs[side].append(seg + sgn * offset * n)
                placed = True
        if not placed:
            segs.append(seg)
    return segs, fsegs


fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.6))
for ax, m, li in zip(np.atleast_1d(axes), levels, range(len(levels))):
    segs, fsegs = edges_of(m, offset=0.004)
    ax.add_collection(LineCollection(segs, colors="#777", linewidths=0.5))
    for side, col in (("Plus", "#c0392b"), ("Minus", "#1a5276")):
        if fsegs[side]:
            ax.add_collection(LineCollection(fsegs[side], colors=col, linewidths=1.8))
    static = getattr(m, "_mg_static", False)
    split = bool(getattr(m, "_fault_point_pairs", None))
    cells = m.dm.getHeightStratum(0)[1]
    kind = "background" if static else ("finest" if m is mesh else "add-on")
    ax.set_title(f"level {li}: {kind}\n{cells} cells, {'split' if split else 'welded'}",
                 fontsize=9)
    ax.set_xlim(0.22, 0.78)
    ax.set_ylim(0.32, 0.68)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Split-level hierarchy (C), k = {k}: red Plus and blue Minus facets drawn "
             "0.004 apart; the background is never cut", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=170)
print("wrote", out)
