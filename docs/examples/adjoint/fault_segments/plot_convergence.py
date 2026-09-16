"""Convergence of the friction inversion under the three observation sets."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cases = [("uplift + stress", "fault_friction_data.npz"),
         ("orientation, points + surface", "fault_friction_orientation_data.npz"),
         ("orientation, surface only", "fault_friction_orientation_surface_data.npz")]
names = ["flat", "lower ramp", "upper ramp", "near surface"]

fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex="col",
                         gridspec_kw={"height_ratios": [2.2, 1]})
for col, (title, path) in enumerate(cases):
    d = np.load(path)
    history, true = d["history"], d["true"]
    its = np.arange(len(history))
    ax = axes[0, col]
    for k in range(len(true)):
        ax.semilogy(its, history[:, 1 + k], "o-", ms=3, lw=1, color=f"C{k}", label=names[k])
        ax.axhline(true[k], color=f"C{k}", lw=0.8, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.03, 0.6)
    if col == 0:
        ax.set_ylabel("friction coefficient")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax = axes[1, col]
    ax.semilogy(its, history[:, 0] / history[0, 0], "k.-", ms=4, lw=1)
    ax.set_xlabel("misfit evaluation")
    if col == 0:
        ax.set_ylabel("$J / J_0$")
    ax.set_ylim(1e-11, 2)
fig.tight_layout()
fig.savefig("fault_friction_convergence.png", dpi=180)
fig.savefig("fault_friction_convergence.pdf")
print("wrote fault_friction_convergence.png / .pdf")
