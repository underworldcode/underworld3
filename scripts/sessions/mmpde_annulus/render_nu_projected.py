"""Projected-flux Nu (outer and inner surfaces) against time for the Ra 1e6 talk runs, with
the logged raw-gradient outer Nu faint behind. Inputs: runs/mmpde/nu_proj/<run>.txt from
nu_projected.py, and each run's run_log.txt for the step -> t map. Usage: render_nu_projected.py OUT.png"""
import os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = os.path.expanduser("~/hyperion-disk/home/lmorese/hyperion-work/runs/mmpde/nu_proj")
S = os.path.expanduser("~/+Simulations/StagnantLid")

def log(r):
    d = {}
    for l in open(f"{S}/{r}/run_log.txt"):
        p = l.split()
        if p and p[0].isdigit():
            d[int(p[0])] = (float(p[1]), float(p[5]))
    return d

def proj(r):
    rows = []
    for l in open(f"{W}/{r}.txt"):
        m = re.search(r"step (\d+): projected Nu outer ([\d.]+) inner ([\d.]+)", l)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return np.array(sorted(rows))

runs = (("ra1e6_fixed", "0.35", "fixed mesh, res 32"),
        ("ra1e6_mmpde5", "#2a6fdb", "MMPDE every 5 steps, res 32, R 8"),
        ("ra1e6_fixed_res64", "#d1495b", "fixed mesh, res 64"))
fig, ax = plt.subplots(figsize=(10, 5.5))
for r, c, lab in runs:
    lg = log(r); p = proj(r)
    if not len(p):
        continue
    ks = sorted(lg); ax.plot([lg[k][0] for k in ks], [lg[k][1] for k in ks], color=c, lw=0.5, alpha=0.18)
    t = np.array([lg[int(s)][0] for s in p[:, 0]])
    ax.plot(t, p[:, 1], color=c, lw=1.6, label=f"{lab}: outer")
    ax.plot(t, p[:, 2], color=c, lw=1.6, ls="--", label=f"{lab}: inner")
ax.set_ylim(8, 12.5); ax.set_xlabel("t"); ax.set_ylabel("Nu")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right")
ax.set_title("Ra 1e6 on the depth, isoviscous SUPG: Nu from the projected total flux (uT - grad T),\n"
             "outer (solid) and inner (dashed) surfaces; faint: the logged raw-gradient outer Nu", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=140)
