"""Recovered friction against the noise on the observations, from the data files."""
import glob
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["flat", "lower ramp", "upper ramp", "near surface"]
runs = []
for path in sorted(glob.glob("fault_friction_uplift+stress*_data.npz")):
    m = re.search(r"_noise([0-9.]+)", path)
    level = float(m.group(1)) if m else 0.0
    r = re.search(r"_prior([0-9.]+)", path)
    alpha = float(r.group(1)) if r else 0.0
    d = np.load(path)
    runs.append((level, alpha, d["history"][-1, 1:], d["true"]))
runs.sort(key=lambda r: (r[0], r[1]))
labels = [f"{r[0]:g}" + (f"\nprior σ={r[1]:g}" if r[1] > 0 else "") for r in runs]
true = runs[0][3]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(len(runs))
w = 0.18
for k in range(len(names)):
    ax.bar(x + (k - 1.5) * w, [r[2][k] for r in runs], width=w, color=f"C{k}", label=names[k])
    ax.hlines(true[k], -0.5, len(runs) - 0.5, colors=f"C{k}", linestyles="--", lw=0.8)
ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
ax.axhline(0.005, color="0.3", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("noise on the observed velocity, fraction of its rms")
ax.set_ylabel("recovered friction (dashed: true; dotted: bounds)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig("fault_friction_noise.png", dpi=180)
fig.savefig("fault_friction_noise.pdf")
print("wrote fault_friction_noise.png / .pdf")
