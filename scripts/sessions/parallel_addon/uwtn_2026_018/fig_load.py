"""Load per processor before and after partitioning the add-on (UWTN 2026-018).

The refined patch box, split (``BoxInternalPatch(refinement=2)`` then
``split_fault``, 19072 cells, P2 velocity): finest P2 nodes per rank at
np = 2, 4 and 8, before the add-on partition and after it.

The numbers are not computed here. BEFORE was measured at f21f3d85, ahead of
the partition, by ``../probe_scaling.py`` (recorded in
docs/developer/design/parallel-addon-distribution-2026-10.md); it cannot be
re-run on the current library. AFTER was re-measured on 2026-10-25 with
``../probe_load_3d.py`` and is identical, rank for rank, to the record. The
ratio in each title is the largest rank's load over the mean.

    python fig_load.py -uw_out load-per-processor.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import underworld3 as uw

params = uw.Params(uw_out="load-per-processor.png")

# finest P2 nodes per rank (see the docstring for where each set comes from)
BEFORE = {
    2: [17118, 12501],
    4: [9695, 7789, 7599, 5529],
    8: [7500, 3555, 4113, 3461, 3399, 3837, 3762, 2147],
}
AFTER = {
    2: [13683, 16090],
    4: [8170, 8470, 7766, 7697],
    8: [5899, 4062, 4007, 4165, 3772, 3642, 3949, 3907],
}
OLD, NEW = "#b3b6b7", "#2e86c1"


def imbalance(loads):
    """The largest rank's load over the mean."""
    return max(loads) / np.mean(loads)


fig, axes = plt.subplots(1, len(BEFORE), figsize=(10.0, 3.2),
                         gridspec_kw={"width_ratios": [2, 4, 8]})
for ax, ranks in zip(axes, BEFORE):
    index = np.arange(ranks)
    ax.bar(index - 0.2, BEFORE[ranks], 0.4, color=OLD, label="before")
    ax.bar(index + 0.2, AFTER[ranks], 0.4, color=NEW, label="after")
    ax.axhline(np.mean(AFTER[ranks]), color="k", lw=0.6, ls=":")
    ax.set_xticks(index)
    ax.set_xlabel("rank")
    ax.set_title(f"np {ranks}: {imbalance(BEFORE[ranks]):.2f} to "
                 f"{imbalance(AFTER[ranks]):.2f}", fontsize=9)
axes[0].set_ylabel("finest P2 nodes")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(params.uw_out, dpi=200)
uw.pprint(f"wrote {params.uw_out}: " + ", ".join(
    f"np {n} {imbalance(BEFORE[n]):.2f} -> {imbalance(AFTER[n]):.2f}"
    for n in BEFORE))
