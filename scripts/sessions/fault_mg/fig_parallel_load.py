"""Parallel behaviour plots from parallel_load.py's records.

python fig_parallel_load.py <k> <outdir> [tag]   (reads ~/+Simulations/fault_split_mg/load_k<k><tag>_np*.json)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

k = int(sys.argv[1])
out = sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else ""
os.makedirs(out, exist_ok=True)
runs = {}
for f in glob.glob(os.path.expanduser(f"~/+Simulations/fault_split_mg/load_k{k}{tag}_np*.json")):
    rec = json.load(open(f))[0]
    runs[rec["np"]] = rec
nps = sorted(runs)
print("ranks:", nps)

# 1. owned cells per rank on every level, for the largest two rank counts
big = [n for n in nps if n > 1][-2:]
fig, axes = plt.subplots(1, len(big), figsize=(6.2 * len(big), 4.4), squeeze=False)
for ax, n in zip(axes[0], big):
    rec = runs[n]
    load = rec["load"]
    nlev = len(load)
    width = 0.8 / n
    for r in range(n):
        own = [lv["owned_cells"][r] for lv in load]
        star = [lv["fault_star"][r] for lv in load]
        x = np.arange(nlev) + (r - (n - 1) / 2) * width
        ax.bar(x, own, width, color=plt.get_cmap("tab10")(r), label=f"rank {r}")
        ax.bar(x, star, width, color="none", edgecolor="k", hatch="///", lw=0.5,
               label="fault star" if r == 0 else None)
    ideal = [sum(lv["owned_cells"]) / n for lv in load]
    ax.plot(np.arange(nlev), ideal, "k_", ms=14, label="perfect balance")
    labels = [("bg" if lv["static"] else ("finest" if i == nlev - 1 else "add-on"))
              + ("\nsplit" if lv["split"] else "") for i, lv in enumerate(load)]
    ax.set_xticks(np.arange(nlev))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("owned cells")
    ax.set_title(f"{n} ranks, k = {k}{tag}: cells per rank on each level", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(out, f"parallel_load_k{k}{tag}.png"), dpi=160)
plt.close(fig)

# 2. imbalance (max / mean owned cells) per level vs ranks; misses; iterations; times
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
for n in nps:
    if n == 1:
        continue
    load = runs[n]["load"]
    imb = [max(lv["owned_cells"]) / (sum(lv["owned_cells"]) / n) for lv in load]
    ax.plot(range(len(load)), imb, "o-", label=f"{n} ranks")
    child = runs[n]["load_unsplit_child"]
    ax.plot([len(load) - 1], [max(child) / (sum(child) / n)], "k_", ms=12)
ax.axhline(1.0, color="k", lw=0.6)
ax.set_xlabel("level (0 = coarsest background)")
ax.set_ylabel("max / mean owned cells")
ax.set_title("imbalance per level (dash: unsplit adapt child)", loc="left", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for n in nps:
    st = runs[n].get("transfer_stats") or []
    xs, ys = [], []
    for li, s in enumerate(st):
        if s:
            xs.append(li)
            ys.append(100.0 * s["misses"] / max(s["owned_nodes"], 1))
    if xs:
        ax.plot(xs, ys, "o-", label=f"{n} ranks")
ax.set_xlabel("level (fine side of the transfer)")
ax.set_ylabel("fine nodes not located locally (%)")
ax.set_title("transfer misses answered by other ranks", loc="left", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[2]
stages = ["adapt", "split_all_levels", "solver_build", "transfers", "solve_total"]
labels = ["adapt", "cut + split all levels", "solver build", "transfer build (in solve)",
          "solve total"]
for st, lab in zip(stages, labels):
    ax.plot(nps, [runs[n]["times"].get(st, np.nan) for n in nps], "o-", label=lab)
ax.set_xscale("log", base=2)
ax.set_xticks(nps)
ax.set_xticklabels([str(n) for n in nps])
ax.set_xlabel("ranks")
ax.set_ylabel("seconds")
its = [runs[n]["vel_its"][-1] for n in nps]
ax.set_title("stage times; velocity its " + ", ".join(f"{n}:{i}" for n, i in zip(nps, its)),
             loc="left", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out, f"parallel_behaviour_k{k}{tag}.png"), dpi=160)
plt.close(fig)
print("ok")
