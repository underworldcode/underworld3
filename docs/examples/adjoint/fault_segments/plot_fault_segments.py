"""The figure for the fault-segments example, from fault_segments_data.npz."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("fault_segments_data.npz")
history, true = d["history"], d["true"]
n_seg = len(true)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [2.2, 1.6, 1.4]})

ax = axes[0]
ax.contourf(d["gx"], d["gy"], np.log10(d["eta_1"]), levels=np.linspace(-2.2, 0, 12), cmap="viridis")
ax.plot(d["points"][:, 0], d["points"][:, 1], "wx", ms=7, mew=1.5)
ax.set_aspect("equal")
ax.set_xlim(0, 2); ax.set_ylim(0, 1)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_title(r"$\log_{10}\eta_1$ at the true strengths; $\times$ stress points", fontsize=10)

ax = axes[1]
for label, style in (("true", "k-"), ("initial", "C3--"), ("recovered", "C0:")):
    ax.plot(d["xs"], d[f"uplift_{label}"], style, lw=1.6, label=label)
ax.set_xlabel("$x$ along the surface"); ax.set_ylabel("uplift rate $v_y$")
ax.set_title("surface uplift rate", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
its = np.arange(len(history))
for k in range(n_seg):
    ax.semilogy(its, history[:, 1 + k], "o-", ms=3, lw=1, color=f"C{k}", label=f"segment {k + 1}")
    ax.axhline(true[k], color=f"C{k}", lw=0.8, ls="--")
ax.set_xlabel("misfit evaluation"); ax.set_ylabel("weak-plane viscosity")
ax.set_title("strengths (dashed: true)", fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("fault_segments.png", dpi=180)
fig.savefig("fault_segments.pdf")
print("wrote fault_segments.png / .pdf")
