"""Velocity iterations against add-on depth for four hierarchies (UWTN 2026-022).

The box in simple shear with a straight fault of fault-split-multigrid.md.
Left: frictionless fault, iterations per solve against the add-on depth k for
the four hierarchies of FAULT_MULTIGRID_HIERARCHIES.md:

  A   welded tail (library: ``add_fault(..., split_levels=False)``)
  B2  welded tail with a rediscretised transversely isotropic band
  C   every add-on level split (library: ``add_fault``)
  D   split add-on levels over a rediscretised weak-zone background

Right: the viscous fault at k = 3, A against C.

Every number is re-measured, not copied: A and C from the library driver
``library_c.py`` (one JSON record per run), B2 and D from the exploration
driver ``stage5_variants.py`` (B2 and D exist only there; its ``RESULT``
lines are read from the logs). See the README for the commit.

    python fig_iterations.py -uw_library ~/+Simulations/fault_split_mg \\
        -uw_logs /mnt/data/lmoresi/logs -uw_out iterations.png
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import underworld3 as uw

params = uw.Params(
    uw_library="~/+Simulations/fault_split_mg",   # library_c.py records
    uw_logs="/mnt/data/lmoresi/logs",             # stage5_variants.py logs
    uw_depths="2,3,4",
    uw_out="iterations.png",
)

DEPTHS = [int(k) for k in str(params.uw_depths).split(",")]
# label offsets (points) so equal counts of two series stay readable
OFFSET = {"A": (6, 4), "B2": (6, -3), "C": (-14, 4), "D": (6, -11)}
STYLE = {"A": ("#7f8c8d", "o", "A  welded tail"),
         "B2": ("#d68910", "s", "B2  welded, TI band"),
         "C": ("#c0392b", "D", "C  split add-on levels"),
         "D": ("#2e86c1", "^", "D  split, weak-zone background")}


def library_iterations(strategy, k, law):
    """Velocity iterations of one library run, or None if absent."""
    path = os.path.join(os.path.expanduser(str(params.uw_library)),
                        f"lib_{strategy}_k{k}_{law}_np1.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))[0]["vel_its"][0]


def exploration_iterations(variant, k):
    """Velocity iterations of one stage5_variants.py run, or None."""
    for path in glob.glob(os.path.join(str(params.uw_logs),
                                       f"uwtn-stage5-{variant}-k{k}.log")):
        for line in open(path):
            match = re.search(r"variant=(\w+) .*vel_its=\[(\d+)\]", line)
            if line.startswith("RESULT") and match:
                return int(match.group(2))
    return None


frictionless = {
    "A": [library_iterations("A", k, "frictionless") for k in DEPTHS],
    "B2": [exploration_iterations("B2", k) for k in DEPTHS],
    "C": [library_iterations("C", k, "frictionless") for k in DEPTHS],
    "D": [exploration_iterations("D", k) for k in DEPTHS],
}
viscous = {s: library_iterations(s, 3, "viscous") for s in ("A", "C")}

fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.4),
                                  gridspec_kw={"width_ratios": [2.2, 1.0]})
for name, its in frictionless.items():
    colour, marker, label = STYLE[name]
    points = [(k, n) for k, n in zip(DEPTHS, its) if n is not None]
    if points:
        left.plot(*zip(*points), marker=marker, color=colour, label=label)
        for k, n in points:
            left.annotate(f"{n}", (k, n), textcoords="offset points",
                          xytext=OFFSET[name], fontsize=7, color=colour)
left.set_xticks(DEPTHS)
left.set_xlabel("add-on depth k")
left.set_ylabel("velocity iterations per solve")
left.set_title("frictionless fault", fontsize=9)
left.set_ylim(bottom=0)
left.legend(fontsize=7, frameon=False)

names = [n for n in ("C", "A") if viscous[n] is not None]
right.bar(names, [viscous[n] for n in names],
          color=[STYLE[n][0] for n in names])
for i, n in enumerate(names):
    right.annotate(f"{viscous[n]}", (i, viscous[n]), ha="center",
                   textcoords="offset points", xytext=(0, 3), fontsize=8)
right.set_title("viscous fault, k = 3", fontsize=9)
right.set_ylabel("velocity iterations per solve")
fig.tight_layout()
fig.savefig(params.uw_out, dpi=200)
uw.pprint(f"wrote {params.uw_out}: frictionless {frictionless}, "
          f"viscous k=3 {viscous}")
