"""What runs where when the add-on is partitioned once (UWTN 2026-018).

The parallel pipeline of a split fault on an add-on mesh, redrawn for the
partition-once design (docs/developer/subsystems/fault-split-multigrid.md,
"Parallel"; docs/developer/design/parallel-addon-distribution-2026-10.md).
The add-on is partitioned once, before any surgery, with every fault region
placed on one rank and the rest balanced by diffusion over the seams; the cut,
the split and the level replay then run on rank-interior regions and move
nothing, and the transfers find every node locally. The earlier pipeline
(scripts/sessions/fault_mg/fig_schematics.py) gathered each fault's star at
the cut and had the transfers ask other ranks for the nodes they missed.

    python fig_pipeline.py -uw_out parallel-pipeline.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import underworld3 as uw

params = uw.Params(uw_out="parallel-pipeline.png")

BASE, ADDON, PART, SURGERY = "#dfe6ee", "#f6e3b4", "#fadbd8", "#f2c2b0"
TRANSFER, EXTRUDE = "#e8daef", "#d5f5e3"
WIDTH, HEIGHT = 2.6, 1.35

# (x, y, text, colour): one box per stage, left to right in time
BOXES = {
    "base": (0.2, 3.4, "base mesh\ndistributed once,\nnever moves", BASE),
    "adapt": (3.1, 3.4, "adapt: add-on levels\nrefined in place,\n"
              "co-partitioned with the base", ADDON),
    "part": (6.0, 3.4, "partition the add-on ONCE\nfault regions to one rank;\n"
             "balance by diffusion;\none assignment, every level", PART),
    "cut": (8.9, 3.4, "cut + split, finest and\nevery add-on level\n"
            "rank-interior: nothing moves", SURGERY),
    "transfer": (8.9, 1.3, "transfers: add-on pairs\nfound locally; base to\n"
                 "first add-on by provenance", TRANSFER),
    "frames": (6.0, 1.3, "per-level pair frames\nrank-local pairing\n"
               "(labels ride the move)", TRANSFER),
    "extrude": (3.1, 1.3, "extrude (thick twin)\nbuilt on one rank,\n"
                "per-fault move (unchanged)", EXTRUDE),
}
ARROWS = [("base", "adapt"), ("adapt", "part"), ("part", "cut"),
          ("cut", "transfer"), ("cut", "frames")]

fig, ax = plt.subplots(figsize=(12.0, 4.6))
ax.axis("off")
for x, y, text, colour in BOXES.values():
    ax.add_patch(FancyBboxPatch((x, y), WIDTH, HEIGHT,
                                boxstyle="round,pad=0.03", fc=colour,
                                ec="#555555", lw=0.8))
    ax.text(x + WIDTH / 2, y + HEIGHT / 2, text, ha="center", va="center",
            fontsize=8.5)


def edge_point(name, towards):
    """Where an arrow leaves box ``name`` heading for box ``towards``."""
    x, y = BOXES[name][:2]
    tx, ty = BOXES[towards][:2]
    if abs(ty - y) < 0.1:                       # same row: side to side
        return (x + WIDTH, y + HEIGHT / 2) if tx > x else (x, y + HEIGHT / 2)
    return (x + WIDTH / 2, y) if ty < y else (x + WIDTH / 2, y + HEIGHT)


for a, b in ARROWS:
    start, end = edge_point(a, b), edge_point(b, a)
    if BOXES[b][1] < BOXES[a][1] and BOXES[b][0] != BOXES[a][0]:
        end = (BOXES[b][0] + WIDTH, BOXES[b][1] + HEIGHT / 2)
    ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", lw=1.0))

ax.text(0.2, 0.75,
        "Nothing is gathered at the cut and nothing is asked of other ranks "
        "in the transfers: every add-on pair is co-partitioned exactly,\nand "
        "the one pair the partition separates (the base's finest level "
        "against the first add-on level) reads each cell's recorded parent\n"
        "through one star forest. The load imbalance is bounded by the fault "
        "regions, which are indivisible units.", fontsize=8.5, va="top")
ax.set_xlim(0, 11.8)
ax.set_ylim(-0.4, 5.0)
fig.tight_layout()
fig.savefig(params.uw_out, dpi=200)
uw.pprint(f"wrote {params.uw_out}")
