"""Where the parallel imbalance comes from: the base levels, the adapt child
(partition inherited from the base), and the gathered fault star, from
parallel_load.py's records.

python fig_load_decomposition.py <outdir> [tag=_b0.025] [k=3]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "_b0.025"
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(out, exist_ok=True)
runs = {}
for n in (4, 8, 16):
    f = os.path.expanduser(f"~/+Simulations/fault_split_mg/load_k{k}{tag}_np{n}.json")
    if os.path.exists(f):
        runs[n] = json.load(open(f))[0]
nps = sorted(runs)


def imb(cells, n):
    return max(cells) / (sum(cells) / n)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
labels = ["base, coarsest", "base, finest", "adapt child\n(no fault)", "split finest\n(child + star)"]
cols = ["#9bb0c4", "#6d8aa6", "#e8c872", "#c0392b"]
w = 0.8 / len(labels)
for j, (lab, col) in enumerate(zip(labels, cols)):
    ys = []
    for n in nps:
        rec = runs[n]
        load = rec["load"]
        base = [lv for lv in load if lv["static"]]
        vals = [imb(base[0]["owned_cells"], n), imb(base[-1]["owned_cells"], n),
                imb(rec["load_unsplit_child"], n), imb(load[-1]["owned_cells"], n)]
        ys.append(vals[j])
    x = np.arange(len(nps)) + (j - (len(labels) - 1) / 2) * w
    ax.bar(x, ys, w, color=col, label=lab)
    for xi, yi in zip(x, ys):
        ax.text(xi, yi + 0.05, f"{yi:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(1.0, color="k", lw=0.6)
ax.set_xticks(np.arange(len(nps)))
ax.set_xticklabels([f"{n} ranks" for n in nps])
ax.set_ylabel("max / mean owned cells")
ax.set_title("the base is balanced; the child inherits the base's partition", loc="left", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
n = nps[-1]
rec = runs[n]
load = rec["load"]
base_f = np.array([lv for lv in load if lv["static"]][-1]["owned_cells"])
child = np.array(rec["load_unsplit_child"])
fin = np.array(load[-1]["owned_cells"])
star = np.array(load[-1]["fault_star"])
x = np.arange(n)
w = 0.27
ax.bar(x - w, base_f, w, color="#6d8aa6", label="base, finest level")
ax.bar(x, child, w, color="#e8c872", label="adapt child, no fault")
ax.bar(x + w, fin, w, color="#f2c2b0", edgecolor="#c0392b", label="split finest")
ax.bar(x + w, star, w, color="none", edgecolor="k", hatch="///", lw=0.5, label="of which fault star")
ax.axhline(fin.sum() / n, color="#c0392b", lw=0.8, ls="--", label="perfect balance, finest")
ax.axhline(base_f.sum() / n, color="#6d8aa6", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xlabel("rank")
ax.set_ylabel("owned cells")
ax.set_title(f"{n} ranks: cells per rank on the base's finest level, the adapt child and the split finest",
             loc="left", fontsize=10)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(out, f"load_decomposition_k{k}{tag}.png"), dpi=160)
print("ok", nps)
