"""The figure for the fault-segments examples.

    python plot_fault_segments.py fault_segments_data.npz   # weak-plane viscosity
    python plot_fault_segments.py fault_friction_data.npz   # friction coefficient
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = sys.argv[1] if len(sys.argv) > 1 else "fault_segments_data.npz"
friction = "friction" in source
d = np.load(source)
history, true = d["history"], d["true"]
names = ["flat", "lower ramp", "upper ramp", "near surface"]
n_seg = len(true)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [2.2, 1.6, 1.4]})

ax = axes[0]
# The bulk sits at log10(eta_1) = 0 to round-off, so the top level is set a
# little below it and "extend" gives the bulk one flat colour.
ax.contourf(d["gx"], d["gy"], np.log10(d["eta_1"]), levels=np.linspace(-2.4, -0.1, 12),
            cmap="viridis", extend="both")
band = float(d["band"]) if "band" in d else 0.08
# the surface band under the top, where the uplift rate (or the orientation) is read
ax.axhspan(1 - 2 * band, 1.0, color="white", alpha=0.35, lw=0)
ax.plot(d["points"][:, 0], d["points"][:, 1], "wx", ms=7, mew=1.5)
ax.set_aspect("equal")
ax.set_xlim(0, 2); ax.set_ylim(0, 1)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_title(r"$\log_{10}\eta_1$ at the truth; $\times$ points, white band: surface observations", fontsize=10)

ax = axes[1]
for label, style in (("true", "k-"), ("initial", "C3--"), ("recovered", "C0:")):
    ax.plot(d["xs"], d[f"uplift_{label}"], style, lw=1.6, label=label)
ax.set_xlabel("$x$ along the surface"); ax.set_ylabel("uplift rate $v_y$")
ax.set_title("surface uplift rate", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
its = np.arange(len(history))
for k in range(n_seg):
    ax.semilogy(its, history[:, 1 + k], "o-", ms=3, lw=1, color=f"C{k}", label=names[k])
    ax.axhline(true[k], color=f"C{k}", lw=0.8, ls="--")
ax.set_xlabel("misfit evaluation"); ax.set_ylabel("friction coefficient" if friction else "weak-plane viscosity")
ax.set_title(("friction" if friction else "strengths") + " (dashed: true)", fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
stem = source.replace("_data.npz", "")
fig.savefig(stem + ".png", dpi=180)
fig.savefig(stem + ".pdf")
print(f"wrote {stem}.png / .pdf")
