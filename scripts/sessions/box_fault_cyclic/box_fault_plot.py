"""Figures for a box_fault_cyclic run.

Usage: python box_fault_plot.py <sim_dir> <run_name>

Reads <sim_dir>/<run_name>/records.json and cycle_*.npz; writes
<sim_dir>/boxfault_<run_name>.png with
  (a) the load components and the mean slip rate over time,
  (b) the net slip per cycle on the fault plane (x along strike, z depth),
  (c) the cumulative slip on the fault plane,
  (d) the surface net displacement over the last cycle, with the fault trace.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sim, run = sys.argv[1], sys.argv[2]
d = os.path.join(sim, run)
with open(os.path.join(d, "records.json")) as f:
    rec = json.load(f)
with open(os.path.join(d, "args.json")) as f:
    args = json.load(f)
cycles = sorted(glob.glob(os.path.join(d, "cycle_*.npz")))
if not cycles:
    sys.exit("no cycle files yet")
last = np.load(cycles[-1])

t = np.array([r["t"] for r in rec])
rate = np.array([r["slip_rate_mean"] for r in rec])
N = np.array([r.get("load_N", np.nan) for r in rec])
S = np.array([r.get("load_S", np.nan) for r in rec])
sig_min = np.array([r["sigma_n_min"] for r in rec])
sig_max = np.array([r["sigma_n_max"] for r in rec])

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
ax = axes[0, 0]
ax.plot(t, S, color="#059669", label=r"$\sigma_{xy}$ shear load")
ax.plot(t, N, color="#c2410c", ls="--", label=r"$\sigma_{xx}$ normal load")
ax.set_ylabel("traction on the loaded face")
ax.axhline(0, color="0.7", lw=0.6)
ax2 = ax.twinx()
ax2.fill_between(t, sig_min, sig_max, color="0.88", label=r"$\sigma_n$ range on the fault", zorder=0)
ax2.plot(t, rate, color="#1f3a6b", label="mean slip rate (x)", zorder=3)
ax2.set_ylabel("slip rate / normal traction", color="#1f3a6b")
ax.set_zorder(ax2.get_zorder() + 1)        # the load curves draw over the band
ax.patch.set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, frameon=False)
ax.set_xlabel("t / period")
ax.set_title(f"{run}: De {args['De']}, mu {args['fault_mu']}, Tb {args['fault_Tb']}, phi {args['phi']}", fontsize=10)

pc = last["pair_coords"]
for ax, key, title in ((axes[0, 1], "slip_cycle", "net slip over the last cycle"),
                       (axes[1, 0], "slip_cum", "cumulative slip")):
    val = last[key]
    vmax = np.abs(val).max() if len(val) else 1.0
    sc = ax.scatter(pc[:, 0], pc[:, 2], c=val, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=18)
    ax.set_aspect("equal")
    ax.set_xlabel("x (along strike)")
    ax.set_ylabel("z")
    ax.set_title(title, fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="signed slip, right-lateral +")

ax = axes[1, 1]
sp = last["surf_pts"]
dc = last["disp_cycle"]
nx = len(np.unique(sp[:, 0]))
ny = len(np.unique(sp[:, 1]))
X = sp[:, 0].reshape(nx, ny)
Y = sp[:, 1].reshape(nx, ny)
mag = np.linalg.norm(dc[:, :2], axis=1).reshape(nx, ny)
pcm = ax.pcolormesh(X, Y, mag, cmap="magma", shading="auto")
stride = max(1, nx // 16)
ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
          dc[:, 0].reshape(nx, ny)[::stride, ::stride], dc[:, 1].reshape(nx, ny)[::stride, ::stride],
          color="white", scale=None, width=0.004)
tr = last["trace"]
if len(tr):
    ax.plot(tr[:, 0], tr[:, 1], color="#c2410c", lw=2)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("surface net displacement over the last cycle", fontsize=10)
plt.colorbar(pcm, ax=ax, shrink=0.8, label="|u| net per cycle")

fig.tight_layout()
out = os.path.join(sim, f"boxfault_{run}.png")
fig.savefig(out, dpi=130)
print("wrote", out)
for c in cycles:
    z = np.load(c)
    sc_ = z["slip_cycle"]
    print(f"  t {float(z['t']):.2f}: net slip per cycle mean {sc_.mean():+.3e} (max {sc_.max():+.3e}, min {sc_.min():+.3e}); "
          f"cumulative mean {z['slip_cum'].mean():+.3e}; surface net |u| max {np.linalg.norm(z['disp_cycle'][:, :2], axis=1).max():.2e}")
