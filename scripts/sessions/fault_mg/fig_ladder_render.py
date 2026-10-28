"""Render the nested ladder hierarchy of ladder_c.py near the band: the two
background levels and the placed ribbons at spacing 0.04, 0.02, 0.01, each
spine split (Plus red, Minus blue, drawn apart), band cells shaded.

python fig_ladder_render.py <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

out = sys.argv[1]
H_BG = 1.0 / 8
WIDTH = 0.06
TRACE = np.array([[0.15, 0.42], [0.85, 0.58]])
TIP = 0.04
FAULT = "Main"
a, b = TRACE[0], TRACE[-1]
t_hat = (b - a) / np.linalg.norm(b - a)
L = float(np.linalg.norm(b - a))
n_hat = np.array([-t_hat[1], t_hat[0]])


def banded(base, size):
    for clearance in (0.55, 0.7, 0.9):
        try:
            dm, _info = place_thin_volume(
                base.dm, [TRACE], width=WIDTH, label="Zone", label_value=31,
                size=size, clearance=clearance, assembly="fuse", mesher="ladder")
            return uw.discretisation.Mesh(
                dm, simplex=True, qdegree=2, boundaries=base.boundaries,
                coordinate_system_type=base.CoordinateSystem.coordinate_type)
        except RuntimeError:
            continue
    raise RuntimeError(f"band build failed at size {size}")


def spine_line(size):
    n_along = max(2, int(round(L / size)))
    step = L / n_along
    k0 = int(round(TIP / step))
    return a + (np.arange(k0, n_along - k0 + 1) * step)[:, None] * t_hat


def edges_of(m, offset):
    dm = m.dm
    eS, eE = dm.getDepthStratum(1)
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    sides = {}
    for side, sgn in (("Plus", 1.0), ("Minus", -1.0)):
        name = f"{FAULT}{side}"
        if name in [bd.name for bd in m.boundaries]:
            lab = dm.getLabel(name)
            value = m.boundaries[name].value
            if lab and lab.getStratumSize(value) > 0:
                sides[side] = (set(int(e) for e in lab.getStratumIS(value).getIndices()
                                   if eS <= int(e) < eE), sgn)
    segs, fsegs = [], {"Plus": [], "Minus": []}
    for e in range(eS, eE):
        p, q = (int(r) - vS for r in dm.getCone(e))
        seg = np.array([X[p], X[q]])
        placed = False
        for side, (es, sgn) in sides.items():
            if e in es:
                fsegs[side].append(seg + sgn * offset * n_hat)
                placed = True
        if not placed:
            segs.append(seg)
    return segs, fsegs


def band_polys(m):
    dm = m.dm
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    try:
        mask = m.cells_labelled("Zone", 31)
    except Exception:
        return []
    polys = []
    for c in np.nonzero(mask)[0]:
        cl = dm.getTransitiveClosure(int(c) + cS)[0]
        vs = [int(p) - vS for p in cl if vS <= p < vE]
        polys.append(X[vs])
    return polys


base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=H_BG,
                                         regular=False, qdegree=2, refinement=1)
levels = base._coarse_level_meshes()[:-1] + [base]
titles = ["background 0", "background 1"]
for size in (0.04, 0.02, 0.01):
    m = banded(base, size).add_fault((FAULT, spine_line(size)), split_levels=False)
    levels.append(m)
    titles.append(f"ladder {size}, split")

fig, axes = plt.subplots(1, len(levels), figsize=(3.3 * len(levels), 3.5))
for ax, m, t in zip(axes, levels, titles):
    polys = band_polys(m)
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors="#dff0d8", edgecolors="none"))
    segs, fsegs = edges_of(m, offset=0.003)
    ax.add_collection(LineCollection(segs, colors="#777", linewidths=0.5))
    for side, col in (("Plus", "#c0392b"), ("Minus", "#1a5276")):
        if fsegs[side]:
            ax.add_collection(LineCollection(fsegs[side], colors=col, linewidths=1.6))
    cells = m.dm.getHeightStratum(0)[1]
    ax.set_title(f"{t}\n{cells} cells", fontsize=9)
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.37, 0.63)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("The nested ladder hierarchy (test_1022 rig): base levels welded and untouched; "
             "the band (green) placed at 0.04, 0.02, 0.01 and its spine split on each",
             fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=170)
print("wrote", out)
