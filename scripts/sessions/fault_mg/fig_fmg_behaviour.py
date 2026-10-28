"""FMG behaviour plots from the measured tables (exploration note and the library
runs of 2026-10-13): iterations against refinement for the four strategies, the
Coulomb Newton sequence, and the moving-fault cost breakdown.

python fig_fmg_behaviour.py <outdir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
k = np.array([1, 2, 3, 4, 5])
h = 0.05 / 2.0**k
# exploration (prototype drivers, FAULT_MULTIGRID_HIERARCHIES.md)
fric = {"A": [9, 13, 18, 23, 32], "B2": [10, 14, 22, 30, 46], "C": [10, 7, 7, 6, 5],
        "D": [14, 9, 6, 6, 5]}
visc = {"A": [9, 12, 17, 22, 30], "B2": [9, 19, 53, 200, 200], "C": [9, 7, 6, 6, 5],
        "D": [17, 9, 8, 7, 6]}
# library path (tests/test_1087, scripts/sessions/fault_mg/library_c.py), serial
lib = {"C": {2: 6, 3: 5, 4: 4}, "A": {2: 13, 3: 18, 4: 23}}
col = {"A": "#2c3e50", "B2": "#7f8c8d", "C": "#c0392b", "D": "#e67e22"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, data, title in ((axes[0], fric, "frictionless fault"), (axes[1], visc, "viscous fault")):
    for name, its in data.items():
        its = np.array(its, dtype=float)
        capped = its >= 200
        ax.plot(h, np.where(capped, np.nan, its), "o-", color=col[name], label=name)
        if capped.any():
            ax.plot(h[capped], np.full(capped.sum(), 60), "x", color=col[name], ms=9, mew=2)
            ax.text(h[capped][0], 64, "cap (200)", fontsize=7.5, color=col[name], ha="center")
    if title.startswith("friction"):
        for name in ("A", "C"):
            kk = np.array(sorted(lib[name]))
            ax.plot(0.05 / 2.0**kk, [lib[name][q] for q in kk], "s", mfc="none", ms=9,
                    color=col[name], label=f"{name} (library)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("near-fault cell size h")
    ax.set_ylabel("velocity iterations per solve")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(0, 70)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
ax = axes[2]
steps = np.arange(1, 5)
coul = {"A k=1": [5, 15, 17, 17], "C k=1": [5, 15, 17, 17], "A k=3": [9, 21, 24, 24],
        "C k=3": [4, 14, 15, 16], "A k=5": [15, 38, 45, 50], "C k=5": [3, 13, 15, 15]}
for name, its in coul.items():
    ax.plot(steps, its, "o-" if name.startswith("C") else "s--",
            color=col[name[0]], alpha={1: 0.4, 3: 0.7, 5: 1.0}[int(name[-1])], label=name)
ax.set_xlabel("Newton step")
ax.set_ylabel("velocity iterations")
ax.set_title("Coulomb fault, per Newton step", loc="left", fontsize=10)
ax.set_xticks(steps)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.suptitle("Iterations of the velocity block (FMG F-cycle preconditioner, FGMRES) as the fault is "
             "refined: box in simple shear, straight fault", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, "fmg_iterations_vs_h.png"), dpi=160)
plt.close(fig)

# moving fault, k = 5, seconds per step (exploration), and k = 4 library at np 1 / 8
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
stages = ["adapt", "split finest", "split levels", "level frames", "background TI", "solve"]
rows = {"A": [3.1, 1.9, 0, 0, 0, 137], "C": [2.7, 1.7, 2.8, 1.0, 0, 38],
        "D": [4.7, 2.9, 5.1, 1.8, 2.4, 61], "GAMG": [3.8, 2.4, 0, 0, 0, 127]}
cmap = plt.get_cmap("tab20c")
bottom = np.zeros(len(rows))
names = list(rows)
for i, st in enumerate(stages):
    vals = np.array([rows[n][i] for n in names])
    ax.bar(names, vals, bottom=bottom, color=cmap(i * 2), label=st)
    bottom += vals
for j, n in enumerate(names):
    ax.text(j, bottom[j] + 2, f"{bottom[j]:.0f} s", ha="center", fontsize=8.5)
ax.set_ylabel("seconds per step (prototype drivers, k = 5, 1 rank)")
ax.set_title("moving fault: cost per step by stage", loc="left", fontsize=10)
ax.legend(fontsize=8)
ax = axes[1]
lib_mov = {"1 rank": {"adapt": 2.9, "cut + split all levels": 4.6, "solver build": 0.5, "solve": 28.6},
           "8 ranks": {"adapt": 1.0, "cut + split all levels": 1.8, "solver build": 0.45, "solve": 12.4}}
bottom = np.zeros(2)
for i, st in enumerate(lib_mov["1 rank"]):
    vals = np.array([lib_mov[n][st] for n in lib_mov])
    ax.bar(list(lib_mov), vals, bottom=bottom, color=cmap(i * 2 + 1), label=st)
    bottom += vals
for j, n in enumerate(lib_mov):
    ax.text(j, bottom[j] + 0.5, f"{bottom[j]:.0f} s", ha="center", fontsize=8.5)
ax.set_ylabel("seconds per step (library, C, k = 4)")
ax.set_title("library path, mean of three positions", loc="left", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "moving_fault_cost.png"), dpi=160)
plt.close(fig)
print("ok")
