"""The multigrid stack under a split fault on an add-on mesh (UWTN 2026-022).

One panel per level near a straight fault in a unit box: the background
levels on the left (never cut), the add-on levels and the finest on the
right, every add-on level split by ``add_fault``. On a split level the two
copies of the fault are drawn a little apart, Plus in red and Minus in blue;
the tips are single vertices (black). On a welded level the trace is drawn
in red. The box and fault are those of
docs/developer/subsystems/fault-split-multigrid.md.

    python fig_stack.py -uw_depth 3 -uw_out stack.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import underworld3 as uw

params = uw.Params(
    uw_depth=3,              # add-on levels k toward the fault
    uw_base_cell=0.1,        # background cell size; refinement 1 halves it
    uw_band_width=0.08,      # width of the refinement band about the trace
    uw_gap=0.01,             # drawn separation of the two fault copies
    uw_out="stack.png",
)

TIP_A = np.array([0.30, 0.45])          # the fault, end to end
TIP_B = np.array([0.70, 0.55])
VIEW_X, VIEW_Y = (0.22, 0.78), (0.32, 0.68)
EDGE, PLUS, MINUS = "#8c8c8c", "#c0392b", "#1a5276"

base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=params.uw_base_cell, refinement=1, qdegree=2)
h_base = params.uw_base_cell / 2         # the background's finest spacing

# the add-on part: refine toward the trace, one level per halving of h
samples = np.column_stack([np.linspace(TIP_A[0], TIP_B[0], 9),
                           np.linspace(TIP_A[1], TIP_B[1], 9), np.zeros(9)])
guide = uw.meshing.Surface("guide", base, samples)
guide.discretize()
metric = guide.refinement_metric_function(
    h_near=h_base / 2 ** int(params.uw_depth), h_far=h_base,
    width=params.uw_band_width, profile="linear")
child = base.adapt(metric, max_levels=int(params.uw_depth))

# the fault is cut and split on the finest AND on every add-on level
mesh = child.add_fault([("Flt", np.vstack([TIP_A, TIP_B]))])
levels = list(mesh._custom_mg_coarse_meshes) + [mesh]

normal = np.array([TIP_A[1] - TIP_B[1], TIP_B[0] - TIP_A[0]])
normal /= np.linalg.norm(normal)


def level_edges(m):
    """The level's edges, with the edges on each fault copy set apart."""
    dm = m.dm
    e_start, e_end = dm.getDepthStratum(1)
    v_start = dm.getDepthStratum(0)[0]
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    copies = {}
    for side in ("Plus", "Minus"):
        name = f"Flt{side}"
        if name in [b.name for b in m.boundaries]:
            label = dm.getLabel(name)
            value = m.boundaries[name].value
            if label and label.getStratumSize(value):
                copies[side] = {int(p) for p in
                                label.getStratumIS(value).getIndices()}
    plain, sides = [], {"Plus": [], "Minus": []}
    for e in range(e_start, e_end):
        a, b = (int(q) - v_start for q in dm.getCone(e))
        seg = np.array([X[a], X[b]])
        for side, shift in (("Plus", 1.0), ("Minus", -1.0)):
            if e in copies.get(side, ()):
                sides[side].append(seg + shift * params.uw_gap / 2 * normal)
                break
        else:
            plain.append(seg)
    return plain, sides


static = [m for m in levels if getattr(m, "_mg_static", False)]
addon = [m for m in levels if not getattr(m, "_mg_static", False)]
columns = len(static) + len(addon)
fig, axes = plt.subplots(1, columns, figsize=(2.6 * columns, 2.2))
for ax, m in zip(axes, static + addon):
    plain, sides = level_edges(m)
    ax.add_collection(LineCollection(plain, colors=EDGE, linewidths=0.4))
    split = any(sides.values())
    if split:
        for side, colour in (("Plus", PLUS), ("Minus", MINUS)):
            ax.add_collection(LineCollection(sides[side], colors=colour,
                                             linewidths=1.6))
    else:
        ax.plot(*np.vstack([TIP_A, TIP_B]).T, color=PLUS, lw=1.6)
    ax.plot(*np.vstack([TIP_A, TIP_B]).T, "o", color="k", ms=3)
    cells = m.dm.getHeightStratum(0)[1]
    role = ("background" if m in static
            else "finest" if m is mesh else "add-on")
    ax.set_title(f"{role}, {cells} cells\n{'split' if split else 'welded'}",
                 fontsize=8)
    ax.set_xlim(*VIEW_X)
    ax.set_ylim(*VIEW_Y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(params.uw_out, dpi=200)
uw.pprint(f"wrote {params.uw_out}: levels "
          + ", ".join(f"{m.dm.getHeightStratum(0)[1]}"
                      f"{'s' if getattr(m, '_fault_point_pairs', None) else ''}"
                      for m in static + addon))
