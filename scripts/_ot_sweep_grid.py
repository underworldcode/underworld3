"""Compose a 4-up panel of the sweep cases at a given step,
side by side so the variants can be visually compared.
"""
from __future__ import annotations
import os
import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


p = argparse.ArgumentParser()
p.add_argument("--root", type=str,
               default=os.path.expanduser(
                   "~/+Simulations/StagnantLid"))
p.add_argument("--step", type=int, required=True)
p.add_argument("--out", type=str, required=True)
args = p.parse_args()

CASES = [
    ("R=1.5  coar=1.0  ff",         "ot_sweep_R1.5_coar1.0_ff"),
    ("R=5.0  coar=1.0  ff",         "ot_sweep_R5.0_coar1.0_ff"),
    ("R=3.0  coar=auto ff",         "ot_sweep_R3.0_coarauto_ff"),
    ("R=3.0  coar=1.0  grad-unif",  "ot_sweep_R3.0_coar1.0_grad"),
]
label = f"step{args.step:04d}"
fig, axes = plt.subplots(2, 2, figsize=(11, 11),
                          constrained_layout=True)
for ax, (title, tag) in zip(axes.flat, CASES):
    path = os.path.join(args.root, tag,
                         "diagnostics", "frames",
                         f"{label}.png")
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"{title}\n(no {label})", fontsize=11)
    ax.axis("off")
fig.suptitle(f"OT sweep — {label}", fontsize=13)
fig.savefig(args.out, dpi=130)
plt.close(fig)
print(f"wrote {args.out}")
