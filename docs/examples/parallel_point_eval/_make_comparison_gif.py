"""Render the FE | RBF | analytic equirectangular comparison as a GIF for
GitHub PR upload, from the saved data_{fe,rbf}.npz grids (no re-solve).

Run:
    pixi run -e amr-dev python _make_comparison_gif.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "_fe_vs_rbf_lonlat")

# Keep the GIF small for GitHub: subsample frames + modest size/dpi.
FRAME_STRIDE = 2          # use every 2nd frame (73 -> 37)
FPS = 10
FIGSIZE = (12.0, 2.9)
DPI = 80

fe = np.load(os.path.join(DATA, "data_fe.npz"))
rbf = np.load(os.path.join(DATA, "data_rbf.npz"))
fe_g, rbf_g, ana_g = fe["grids"], rbf["grids"], fe["ana_grids"]
angles = fe["angles_deg"]

frames = list(range(0, fe_g.shape[0], FRAME_STRIDE))
if frames[-1] != fe_g.shape[0] - 1:
    frames.append(fe_g.shape[0] - 1)  # always include the final 360 deg frame

cmap = LinearSegmentedColormap.from_list(
    "signed_warm",
    [
        (0.00, "#7e2e9d"), (0.15, "#bb91dd"), (0.231, "#ffffff"),
        (0.31, "#cfe5ff"), (0.46, "#4f9aff"), (0.65, "#3ec76b"),
        (0.85, "#ffa040"), (1.00, "#c41e1e"),
    ],
)
clim = (-0.3, 1.0)
extent = [-180, 180, -90, 90]

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True)
panels = [("FE trace-back", fe_g), ("RBF trace-back", rbf_g), ("analytic", ana_g)]
ims = []
for ax, (title, data) in zip(axes, panels):
    im = ax.imshow(data[0], extent=extent, origin="lower", aspect="auto",
                   cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-90, 0, 90])
    ax.tick_params(labelsize=8)
    ims.append(im)
axes[0].set_ylabel("lat", fontsize=9)
fig.colorbar(ims[-1], ax=axes, shrink=0.8, label="T")
suptitle = fig.suptitle("", fontsize=11)


def update(k):
    ims[0].set_data(fe_g[k])
    ims[1].set_data(rbf_g[k])
    ims[2].set_data(ana_g[k])
    suptitle.set_text(
        f"SLCN rotation on a sphere — rel-L2 vs analytic:  "
        f"FE {fe['l2'][k] * 100:.1f}%   RBF {rbf['l2'][k] * 100:.1f}%   "
        f"(rotation {angles[k]:.0f}°)"
    )
    return (*ims, suptitle)


anim = FuncAnimation(fig, update, frames=frames, blit=False)
gif = os.path.join(DATA, "fe_vs_rbf_lonlat.gif")
anim.save(gif, writer=PillowWriter(fps=FPS), dpi=DPI)
plt.close(fig)

size_mb = os.path.getsize(gif) / 1e6
print(f"wrote {gif}  ({size_mb:.2f} MB, {len(frames)} frames @ {FPS} fps)")
