"""3-up panel for the R=5 coar=auto sweep variants."""
from __future__ import annotations
import os
import argparse

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
    ("R=5 coar=auto ff  deg=1",  "ot_sweep_R5.0_coarauto_ff_deg1"),
    ("R=5 coar=auto ff  deg=2",  "ot_sweep_R5.0_coarauto_ff_deg2"),
    ("R=5 coar=auto grad deg=1", "ot_sweep_R5.0_coarauto_grad_deg1"),
]
label = f"step{args.step:04d}"
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                          constrained_layout=True)
for ax, (title, tag) in zip(axes, CASES):
    path = os.path.join(args.root, tag,
                         "diagnostics", "frames",
                         f"{label}.png")
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"{title}\n(no {label})", fontsize=11)
    ax.axis("off")
fig.suptitle(f"R=5 coar=auto sweep — {label}", fontsize=13)
fig.savefig(args.out, dpi=130)
plt.close(fig)
print(f"wrote {args.out}")
