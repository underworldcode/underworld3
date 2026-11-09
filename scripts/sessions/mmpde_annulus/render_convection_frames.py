"""Frames of an adaptive convection run: T on a blue-neutral-red map, the mesh in black,
the full annulus at several steps and a zoom on the top boundary layer at the last."""
import glob
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation

run, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
temperature = LinearSegmentedColormap.from_list(
    "blue_neutral_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f5a3a3", "#e34948", "#a42f2f"])


def frame(step):
    stem = f"{run}/step{step:04d}.mesh"
    with h5py.File(f"{stem}.00000.h5", "r") as h:
        xy = h["geometry/vertices"][...]
        cells = h["viz/topology/cells"][...]
    with h5py.File(f"{stem}.T_v2p1.00000.h5", "r") as h:
        T = h["visualization/T_v2p1"][...]
    return Triangulation(xy[:, 0], xy[:, 1], cells), T


steps = sorted(int(m.group(1)) for f in glob.glob(f"{run}/step*.mesh.00000.h5")
               if (m := re.search(r"step(\d+)\.mesh", f)))
wanted = [s for s in (5, 50, 100) if s in steps] + [steps[-1]]
wanted = sorted(set(wanted))
zooms = (("top boundary layer", (-0.22, 0.22), (0.76, 1.01)),
         ("inner hot layer", (-0.18, 0.18), (0.47, 0.66)))
fig, axes = plt.subplots(2, len(wanted), figsize=(4.6 * len(wanted), 9.6), facecolor=SURFACE)
for ax, step in zip(axes[0], wanted):
    tri, T = frame(step)
    ax.tripcolor(tri, T, shading="gouraud", cmap=temperature, vmin=0.0, vmax=1.0)
    ax.triplot(tri, color=INK, lw=0.05)
    ax.set_title(f"step {step}", color=INK)
    ax.set_aspect("equal"); ax.axis("off")
tri, T = frame(wanted[-1])
for ax, (name, xlim, ylim) in zip(axes[1], zooms):
    ax.tripcolor(tri, T, shading="gouraud", cmap=temperature, vmin=0.0, vmax=1.0)
    ax.triplot(tri, color=INK, lw=0.45)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"step {wanted[-1]}: {name}", color=INK)
for ax in axes[1][len(zooms):]:
    ax.axis("off")
fig.suptitle("Adaptive stagnant-lid convection, Ra 1e7, Δη 1e3, mover every step "
             "(T: blue cold, red hot)", color=INK, fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig(out, dpi=170, facecolor=SURFACE)
print("written", out, "steps", wanted)
