"""Iterations of the single-level ladder and its tails as the band deepens:
from ladder_c.json, ladder_depth.json, ladder_depth_split*.json and
ladder_base_ref2.json (test_1022 rig).

python fig_ladder_depth.py <out.png>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
D = os.path.expanduser("~/+Simulations/fault_split_mg")
recs = []
for f in ("ladder_c.json", "ladder_depth.json", "ladder_depth_split.json",
          "ladder_depth_split4.json", "ladder_base_ref2.json"):
    recs += json.load(open(os.path.join(D, f)))
by = {r["arm"]: r["vel_its"] for r in recs}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
# left: contact
ax = axes[0]
ax.plot([0.01, 0.005, 0.0025], [by["fine 0.01: no mid"], by["fine 0.005: no mid"],
                                 by["fine 0.0025: no mid"]], "o-", color="#c0392b",
        label="single ladder on base h = 1/16 (no mid)")
ax.plot([0.005, 0.0025], [by["ref 2 fine 0.005: no mid"], by["ref 2 fine 0.0025: no mid"]],
        "s-", color="#1a5276", label="single ladder on base h = 1/32 (no mid)")
ax.plot([0.0025] * 3, [by["fine 0.0025: welded mid [0.005]"],
                        by["fine 0.0025: welded mid [0.01, 0.005]"],
                        by["fine 0.0025: welded mid [0.02, 0.01, 0.005]"]],
        "v", color="#7f8c8d", ms=9, label="h = 1/16 + 1, 2, 3 welded mids")
ax.plot([0.0025] * 4, [by["fine 0.0025: split mid [0.005]"],
                        by["fine 0.0025: split mid [0.01, 0.005]"],
                        by["fine 0.0025: split mid [0.02, 0.01, 0.005]"],
                        by["fine 0.0025: split mid [0.04, 0.02, 0.01, 0.005]"]],
        "^", color="#e67e22", ms=9, mfc="none", label="h = 1/16 + 1, 2, 3, 4 split mids")
ax.annotate("3 and 4 split mids\n(chain from 0.02)", (0.0025, 13), (0.0045, 15), fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#e67e22"))
ax.annotate("1, 2 split or any welded", (0.0025, 21), (0.0045, 22), fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#7f8c8d"))
ax.set_xscale("log", base=2)
ax.invert_xaxis()
ax.set_xticks([0.01, 0.005, 0.0025])
ax.set_xticklabels(["0.01\n(6 | 3)", "0.005\n(12 | 6)", "0.0025\n(25 | 12)"])
ax.set_xlabel("ladder spacing  (ratio to base finest: h = 1/16 | h = 1/32)")
ax.set_ylabel("velocity iterations")
ax.set_title("split-node contact on the ladder", loc="left", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
# right: TI
ax = axes[1]
ax.plot([0.01, 0.005, 0.0025], [by["TI fine 0.01: no mid"], by["TI fine 0.005: no mid"],
                                 by["TI fine 0.0025: no mid"]], "o-", color="#c0392b",
        label="single ladder on base h = 1/16 (no mid)")
ax.plot([0.005, 0.0025], [by["TI ref 2 fine 0.005: no mid"], by["TI ref 2 fine 0.0025: no mid"]],
        "s-", color="#1a5276", label="single ladder on base h = 1/32 (no mid)")
ax.plot([0.01, 0.005], [by["TI fine 0.01: welded mid [0.02]"],
                        by["TI fine 0.005: welded mid [0.02, 0.01]"]], "v--", color="#7f8c8d",
        label="h = 1/16 + all nested welded mids")
ax.plot([0.0025] * 3, [by["TI fine 0.0025: welded mid [0.005]"],
                        by["TI fine 0.0025: welded mid [0.01, 0.005]"],
                        by["TI fine 0.0025: welded mid [0.02, 0.01, 0.005]"]],
        "v", color="#7f8c8d", ms=9)
ax.annotate("1, 2, 3 welded mids", (0.0025, 5), (0.0045, 3.2), fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#7f8c8d"))
ax.set_xscale("log", base=2)
ax.invert_xaxis()
ax.set_xticks([0.01, 0.005, 0.0025])
ax.set_xticklabels(["0.01\n(6 | 3)", "0.005\n(12 | 6)", "0.0025\n(25 | 12)"])
ax.set_xlabel("ladder spacing  (ratio to base finest: h = 1/16 | h = 1/32)")
ax.set_title("TI band on the ladder (contrast 1e-3)", loc="left", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
ax.set_ylim(0, 10)
fig.suptitle("A single finest ladder over the welded base, and what the tail buys as the band deepens",
             fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=160)
print("wrote", out)
