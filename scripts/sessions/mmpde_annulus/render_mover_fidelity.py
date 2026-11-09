"""Figures for the mover fix: the starting mesh, the development mover's mesh and
#1024's, coloured by T, full annulus and a zoom on the layer; and cell size against
radius for the three."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation

src = sys.argv[1]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = {"start": "#2a78d6", "development": "#eb6834", "pr1024": "#1baf7a"}
LABEL = {"start": "starting mesh", "development": "development: mover stops after 5 steps",
         "pr1024": "#1024: mover converges (72 steps)"}
blues = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

data = {k: np.load(f"{src}/{k}.npz") for k in SERIES}

fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=SURFACE)
for col, key in enumerate(SERIES):
    d = data[key]
    tri = Triangulation(d["xy"][:, 0], d["xy"][:, 1], d["cells"])
    for row in range(2):
        ax = axes[row, col]
        ax.set_facecolor(SURFACE)
        ax.tripcolor(tri, d["T"], shading="gouraud", cmap=blues, vmin=0.0, vmax=1.3)
        ax.triplot(tri, color=INK2, lw=0.12 if row == 0 else 0.45)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if row == 1:
            ax.set_xlim(0.78, 1.01); ax.set_ylim(-0.115, 0.115)
    axes[0, col].set_title(LABEL[key], color=INK, fontsize=12)
axes[1, 0].set_ylabel("zoom: the thermal layer at r = 0.9", color=INK2)
fig.suptitle("One MMPDE move on an annulus with a sharp thermal layer (T, light to dark)",
             color=INK, fontsize=14)
fig.tight_layout()
fig.savefig(f"{src}/mover_meshes.png", dpi=200, facecolor=SURFACE)

chart_src = sys.argv[2] if len(sys.argv) > 2 else src
chart = {k: np.load(f"{chart_src}/{k}.npz") for k in SERIES}
fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
bins = np.linspace(0.5, 1.0, 15)
centres = 0.5 * (bins[1:] + bins[:-1])
ends = {}
for key, colour in SERIES.items():
    d = chart[key]
    xy, cells = d["xy"], d["cells"]
    edges = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    length = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    radius = np.linalg.norm(0.5 * (xy[edges[:, 0]] + xy[edges[:, 1]]), axis=1)
    idx = np.clip(np.digitize(radius, bins) - 1, 0, len(centres) - 1)
    median = np.array([np.median(length[idx == i]) for i in range(len(centres))])
    ax.plot(centres, median, color=colour, lw=2, label=LABEL[key])
    ends[key] = median[-1]
# Direct labels at the right end, pushed apart where the lines meet.
order = sorted(ends, key=ends.get)
span = max(ends.values()) - min(ends.values())
gap = 0.0012
placed = []
for key in order:
    y = ends[key] if not placed else max(ends[key], placed[-1] + gap)
    placed.append(y)
    ax.annotate(LABEL[key].split(":")[0], (centres[-1], ends[key]), xytext=(centres[-1] + 0.012, y),
                textcoords="data", color=INK2, va="center", fontsize=10)
ax.axvline(0.9, color=INK2, lw=0.8, ls=":")
ax.text(0.902, ax.get_ylim()[1], "layer", color=INK2, va="top", fontsize=10)
ax.set_xlabel("radius")
ax.set_ylabel("median edge length")
ax.set_xlim(0.5, 1.1)
ax.grid(axis="y", color="#e6e5e1", lw=0.6)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=10)
ax.set_title("Cell size across the annulus after one move (6496 cells)", color=INK)
fig.tight_layout()
fig.savefig(f"{src}/mover_cell_size.png", dpi=200, facecolor=SURFACE)
print("written", {k: round(float(v), 4) for k, v in ends.items()})
