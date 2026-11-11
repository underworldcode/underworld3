"""Nu and vrms against time for the Ra 1e6 talk runs: fixed res 32, MMPDE res 32, fixed res 64.
Usage: render_talk_history.py OUT.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load(r):
    d = {}
    for l in open(os.path.expanduser(f"~/+Simulations/StagnantLid/{r}/run_log.txt")):
        p = l.split()
        if p and p[0].isdigit():
            d[int(p[0])] = (float(p[1]), float(p[4]), float(p[5]))
    return np.array([d[k] for k in sorted(d)])

runs = (("ra1e6_fixed", "0.6", "fixed mesh, res 32", 1, 0.9),
        ("ra1e6_mmpde5", "#2a6fdb", "MMPDE every 5 steps, res 32, R 8", 3, 1.0),
        ("ra1e6_fixed_res64", "#d1495b", "fixed mesh, res 64", 2, 1.2))
fig, (x, y) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
for r, c, lab, zo, lw in runs:
    z = load(r)
    x.plot(z[:, 0], z[:, 2], color=c, lw=lw, label=f"{lab} (t = {z[-1, 0]:.3f})", zorder=zo)
    y.plot(z[:, 0], z[:, 1], color=c, lw=lw, zorder=zo)
x.set_ylabel("Nu (surface)"); y.set_ylabel("vrms (area integral)"); y.set_xlabel("t")
x.set_ylim(0, 25); y.set_ylim(0, 3000); x.legend(frameon=False, loc="upper right")
for z in (x, y):
    z.spines[["top", "right"]].set_visible(False)
fig.suptitle("Ra 1e6 on the depth, isoviscous SUPG, degree-7 start")
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=130)
