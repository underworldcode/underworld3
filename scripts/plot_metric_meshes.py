"""Plot-only (loads the saved case_*.npz from show_metric_mesh.py
— no re-solving). Produces large, clear mesh figures to judge the
metric-driven grading visually:

  /tmp/metric_mesh/meshes_big.png   full annulus, Spring vs MA
  /tmp/metric_mesh/meshes_zoom.png  zoomed outer-band wedge
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/tmp/metric_mesh"
R_O, R_I, WIDTH = 1.0, 0.5, 0.12
AMPS = [0, 2, 8, 20]
th = np.linspace(0, 2 * np.pi, 360)


def load(method, amp):
    d = np.load(os.path.join(OUT, f"case_{method}_amp{amp}.npz"))
    return d["coords1"], d["tri"]


# ---------- 1. big full-annulus grid ----------
fig, axes = plt.subplots(2, 4, figsize=(22, 11.5),
                         facecolor="white")
for ri, method in enumerate(("spring", "ma")):
    for ci, amp in enumerate(AMPS):
        c1, TRI = load(method, amp)
        ax = axes[ri, ci]
        ax.set_facecolor("white")
        ax.triplot(c1[:, 0], c1[:, 1], TRI, color="black", lw=0.6)
        ax.plot(R_O * np.cos(th), R_O * np.sin(th),
                color="tab:red", lw=1.4)
        ax.plot(R_I * np.cos(th), R_I * np.sin(th),
                color="tab:blue", lw=1.4)
        ax.set_title(("uniform AMP=0" if amp == 0
                      else f"AMP={amp}"), fontsize=14)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    axes[ri, 0].set_ylabel(
        "ELASTIC-SPRING" if method == "spring"
        else "MONGE–AMPÈRE", fontsize=15)
fig.suptitle("Metric-driven mesh grading — full annulus "
             "(red = outer surface where the metric peaks)",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
p1 = os.path.join(OUT, "meshes_big.png")
fig.savefig(p1, dpi=130, bbox_inches="tight")
print("Saved", p1)

# ---------- 2. zoomed outer-band wedge (AMP 8 & 20) ----------
fig, axes = plt.subplots(2, 4, figsize=(22, 11.5),
                         facecolor="white")
cols = [("spring", 8), ("ma", 8), ("spring", 20), ("ma", 20)]
# top row: full; bottom row: zoom into a 70° wedge near r=R_O
for ci, (method, amp) in enumerate(cols):
    c1, TRI = load(method, amp)
    label = ("Spring" if method == "spring" else "MA")
    # full
    ax = axes[0, ci]
    ax.triplot(c1[:, 0], c1[:, 1], TRI, color="black", lw=0.6)
    ax.plot(R_O * np.cos(th), R_O * np.sin(th),
            color="tab:red", lw=1.4)
    ax.add_patch(plt.Rectangle((0.30, -0.05), 0.78, 0.95,
                 fill=False, ec="tab:green", lw=1.5, ls="--"))
    ax.set_title(f"{label}  AMP={amp}", fontsize=14)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    # zoom: a box on the right side spanning the outer band
    ax = axes[1, ci]
    ax.triplot(c1[:, 0], c1[:, 1], TRI, color="black", lw=0.9)
    ax.plot(R_O * np.cos(th), R_O * np.sin(th),
            color="tab:red", lw=1.8)
    ax.set_xlim(0.30, 1.08)
    ax.set_ylim(-0.05, 0.90)
    ax.set_title(f"{label}  AMP={amp}  (zoom: deep→surface)",
                 fontsize=13)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Zoom on the deep-interior → outer-surface "
             "transition (green dashed = zoom box). Look for the "
             "fine band hugging the red surface vs coarse interior.",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p2 = os.path.join(OUT, "meshes_zoom.png")
fig.savefig(p2, dpi=130, bbox_inches="tight")
print("Saved", p2)
