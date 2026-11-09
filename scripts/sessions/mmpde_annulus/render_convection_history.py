"""History of an adaptive convection run: vrms, Nu and the cost of each part of a step,
against model time, as small multiples sharing the time axis."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
h = np.load(f"{run}/history.npz")
t = h["t"]

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].plot(t, h["vrms"], color=BLUE, lw=2)
axes[0].set_ylabel("vrms")
axes[1].plot(t, h["Nu"], color=BLUE, lw=2)
axes[1].set_ylabel("Nu (surface)")
for key, colour, label in (("t_stokes", BLUE, "Stokes solve"),
                           ("t_adapt", ORANGE, "mover and remap"),
                           ("t_advdiff", AQUA, "transport")):
    axes[2].plot(t, h[key], color=colour, lw=2, label=label)
    axes[2].annotate(label, (t[-1], h[key][-1]), xytext=(6, 0), textcoords="offset points",
                     color=INK2, va="center", fontsize=10)
axes[2].set_ylabel("seconds per step (np 5)")
axes[2].set_xlabel("model time")
axes[2].legend(frameon=False, loc="upper left", fontsize=10)
axes[2].set_xlim(t[0], t[-1] * 1.18)
fig.suptitle("Adaptive stagnant-lid convection, Ra 1e7, Δη 1e3, res 32, 3 FMG levels, "
             "mover every step", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=170, facecolor=SURFACE)
for key in ("vrms", "Nu", "t_stokes", "t_adapt", "t_advdiff", "wall"):
    late = h[key][len(t) // 2:]
    print(f"{key}: median second half {np.median(late):.3g}, range {late.min():.3g}-{late.max():.3g}")
print("stokes_ksp_its", np.unique(h["stokes_ksp_its"]), "snes", np.unique(h["stokes_snes_its"]))
