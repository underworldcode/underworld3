"""4-up grid: res16 (gsmooth1, gsmooth2) at step Sx and res32
(gsmooth1, gsmooth2) at step 2*Sx (same physical time at
fixed_dt=3e-4 vs 1.5e-4)."""
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
p.add_argument("--res16-step", type=int, required=True,
               help="step in res16 runs (dt=3e-4)")
p.add_argument("--out", type=str, required=True)
args = p.parse_args()

s16 = args.res16_step
s32 = 2 * s16
t = s16 * 3.0e-4

CASES = [
    (f"res16  L=h0    step{s16:04d}",
     f"ot_sweep_R3.0_coarauto_ff_gsmooth1_fixdt",
     f"step{s16:04d}.png"),
    (f"res16  L=2·h0  step{s16:04d}",
     f"ot_sweep_R3.0_coarauto_ff_gsmooth2_fixdt",
     f"step{s16:04d}.png"),
    (f"res32  L=h0    step{s32:04d}",
     f"ot_sweep_R3.0_coarauto_ff_res32_gsmooth1_fixdt",
     f"step{s32:04d}.png"),
    (f"res32  L=2·h0  step{s32:04d}",
     f"ot_sweep_R3.0_coarauto_ff_res32_gsmooth2_fixdt",
     f"step{s32:04d}.png"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 11),
                          constrained_layout=True)
for ax, (title, tag, fname) in zip(axes.flat, CASES):
    path = os.path.join(args.root, tag,
                         "diagnostics", "frames", fname)
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"{title}\n(no {fname})", fontsize=11)
    ax.axis("off")
fig.suptitle(
    f"R=3 coar=auto ff   res16 vs res32   t = {t:.4f}",
    fontsize=13)
fig.savefig(args.out, dpi=130)
plt.close(fig)
print(f"wrote {args.out}")
