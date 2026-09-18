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
    d = np.load(path)
    runs.append((level, d["history"][-1, 1:], d["true"]))
runs.sort(key=lambda r: r[0])
levels = np.array([r[0] for r in runs])
true = runs[0][2]

fig, ax = plt.subplots(figsize=(6.2, 3.6))
x = np.arange(len(levels))
w = 0.18
for k in range(len(names)):
    ax.bar(x + (k - 1.5) * w, [r[1][k] for r in runs], width=w, color=f"C{k}", label=names[k])
    ax.hlines(true[k], -0.5, len(levels) - 0.5, colors=f"C{k}", linestyles="--", lw=0.8)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"{l:g}" for l in levels])
ax.set_xlabel("noise on the observed velocity, fraction of its rms")
ax.set_ylabel("recovered friction (dashed: true)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig("fault_friction_noise.png", dpi=180)
fig.savefig("fault_friction_noise.pdf")
print("wrote fault_friction_noise.png / .pdf")
