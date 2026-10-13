"""Schematic figures for the fault multigrid write-up (wiki and technical note).

1. hierarchy_strategies.png  — the four hierarchies A, B2, C, D as level stacks
2. transfer_rule.png         — the side-aware cell-located transfer and the clamp
3. parallel_pipeline.png     — what moves where in parallel (split, levels, extrusion)

python fig_schematics.py <outdir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
BG, ADD, FINE = "#dfe6ee", "#f6e3b4", "#f2c2b0"
SPLIT, TI, WELD = "#c0392b", "#7f8c8d", "#2c3e50"


def level_box(ax, y, kind, label, fault, width=3.6, x0=0.2):
    """one multigrid level: a box, a fault symbol, a label"""
    face = {"bg": BG, "add": ADD, "fine": FINE}[kind]
    ax.add_patch(FancyBboxPatch((x0, y), width, 0.72, boxstyle="round,pad=0.02",
                                fc=face, ec="#555", lw=0.8))
    xc = x0 + width * 0.62
    if fault == "split":
        ax.plot([xc - 0.9, xc + 0.9], [y + 0.40, y + 0.40], color=SPLIT, lw=2.2)
        ax.plot([xc - 0.9, xc + 0.9], [y + 0.30, y + 0.30], color=SPLIT, lw=2.2)
        ax.plot([xc - 0.9, xc - 0.9], [y + 0.30, y + 0.40], color=SPLIT, lw=1.0)
        ax.plot([xc + 0.9, xc + 0.9], [y + 0.30, y + 0.40], color=SPLIT, lw=1.0)
    elif fault == "ti":
        ax.add_patch(Polygon([[xc - 0.95, y + 0.18], [xc + 0.95, y + 0.18],
                              [xc + 0.95, y + 0.54], [xc - 0.95, y + 0.54]],
                             closed=True, fc=TI, alpha=0.35, ec="none"))
        ax.plot([xc - 0.9, xc + 0.9], [y + 0.36, y + 0.36], color=TI, lw=1.2, ls="--")
    else:
        ax.plot([xc - 0.9, xc + 0.9], [y + 0.36, y + 0.36], color=WELD, lw=1.0, ls=":")
    ax.text(x0 + 0.12, y + 0.36, label, va="center", ha="left", fontsize=8.5)


def strategies():
    cols = [
        ("A  welded tail", ["split", "weld", "weld", "weld", "weld"]),
        ("B2  TI band on the add-on levels", ["split", "ti", "ti", "weld", "weld"]),
        ("C  split add-on levels", ["split", "split", "split", "weld", "weld"]),
        ("D  C + TI background", ["split", "split", "split", "ti", "ti"]),
    ]
    names = ["finest (add-on)", "add-on 2", "add-on 1", "background 1", "background 0"]
    kinds = ["fine", "add", "add", "bg", "bg"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.6))
    for ax, (title, faults) in zip(axes, cols):
        for i, (nm, kd, f) in enumerate(zip(names, kinds, faults)):
            level_box(ax, 4.0 - i * 0.95, kd, nm, f)
        ax.set_xlim(0, 4.0)
        ax.set_ylim(-0.2, 5.0)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.axis("off")
    # legend
    ax = axes[0]
    ax.plot([], [], color=SPLIT, lw=2.2, label="split slit (two coincident node sets)")
    ax.plot([], [], color=TI, lw=1.2, ls="--", label="transversely isotropic weak band")
    ax.plot([], [], color=WELD, lw=1.0, ls=":", label="welded (no fault)")
    fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.01, 0.97, "The background levels (grey) are the static base hierarchy and are "
             "never cut; the add-on levels (yellow) are refined from it toward the fault.",
             fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(out, "hierarchy_strategies.png"), dpi=160)
    plt.close(fig)


def transfer_rule():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # left: coarse split cells, a fine node on the slit, its own-side cell centroid
    ax = axes[0]
    # fault along y = 0 (slit drawn as a gap); coarse triangles above and below
    up = [np.array([[0, 0], [2, 0], [1, 1.6]]), np.array([[2, 0], [4, 0], [3, 1.6]]),
          np.array([[1, 1.6], [2, 0], [3, 1.6]])]
    dn = [np.array([[0, 0], [2, 0], [1, -1.6]]), np.array([[2, 0], [4, 0], [3, -1.6]]),
          np.array([[1, -1.6], [2, 0], [3, -1.6]])]
    gap = 0.08
    for t in up:
        ax.add_patch(Polygon(t + [0, gap], closed=True, fc="#eaf2fb", ec="#444", lw=0.9))
    for t in dn:
        ax.add_patch(Polygon(t - [0, gap], closed=True, fc="#fdf2e9", ec="#444", lw=0.9))
    ax.plot([0, 4], [gap, gap], color=SPLIT, lw=2.2)
    ax.plot([0, 4], [-gap, -gap], color=SPLIT, lw=2.2)
    # a fine cell on the upper side touching the slit, its centroid
    ft = np.array([[1.4, gap], [2.0, gap], [1.7, 0.55]])
    ax.add_patch(Polygon(ft, closed=True, fc="none", ec="#1a5276", lw=1.6, ls="-"))
    c = ft.mean(axis=0)
    ax.plot(*c, "o", color="#1a5276", ms=6)
    ax.plot(1.4, gap, "s", color="#1a5276", ms=7)
    ax.annotate("fine node on the slit\n(upper copy)", (1.4, gap), (0.2, 0.9), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("fine cell's centroid:\nnever on the fault,\nlocates the coarse cell\n"
                "on the node's own side", c, (2.6, 1.0), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(2.0, -1.0, "the lower copy of the same node\nis located from a lower-side cell",
            ha="center", fontsize=8.5, color="#7b3f00")
    ax.set_xlim(-0.2, 4.4)
    ax.set_ylim(-1.9, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("side-aware location", loc="left", fontsize=10.5)
    # right: the clamp
    ax = axes[1]
    T = np.array([[0.3, 0.2], [3.2, 0.4], [1.6, 2.3]])
    ax.add_patch(Polygon(T, closed=True, fc="#eef5ee", ec="#444", lw=1.0))
    nodes = np.vstack([T, (T[[0, 1]]).mean(0), (T[[1, 2]]).mean(0), (T[[2, 0]]).mean(0)])
    ax.plot(nodes[:, 0], nodes[:, 1], "o", color="#2e7d32", ms=5)
    p_out = np.array([3.0, 1.55])
    # nearest point of the triangle edge (1,2)
    a, b = T[1], T[2]
    t = np.clip(((p_out - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    q = a + t * (b - a)
    ax.plot(*p_out, "s", color="#1a5276", ms=7)
    ax.plot(*q, "x", color=SPLIT, ms=9, mew=2)
    ax.annotate("", q, p_out, arrowprops=dict(arrowstyle="->", color=SPLIT, lw=1.2))
    ax.text(3.05, 1.65, "fine node a little outside\nits located coarse cell\n"
            "(levels not nested near the trace)", fontsize=8.5, va="bottom")
    ax.text(1.0, -0.35, "evaluate the coarse Lagrange interpolant at the clamped point:\n"
            r"barycentric coordinates clipped to $[0,1]$ and renormalised; every weight stays"
            "\nwithin the basis on the simplex (no extrapolation, no 5.4 weights at a tip)",
            fontsize=8.5, va="top")
    ax.text(0.9, 1.9, "P2 nodes of the\ncoarse cell", fontsize=8.5, color="#2e7d32")
    ax.set_xlim(-0.2, 5.4)
    ax.set_ylim(-1.3, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the clamp", loc="left", fontsize=10.5)
    fig.suptitle(r"The transfer: $W = B\,M^{-1}$ — the coarse cell's own nodes determine its "
                 "interpolant; no node-cloud triangulation (which degenerates on a slit)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "transfer_rule.png"), dpi=160)
    plt.close(fig)


def parallel_pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.axis("off")
    boxes = [
        (0.3, 3.4, "base mesh\n(distributed, static)", BG),
        (3.2, 3.4, "adapt: add-on levels\n(in place, co-partitioned\nwith the base)", ADD),
        (6.1, 3.4, "cut + split, finest\n(fault star gathered to\none rank, per region)", FINE),
        (6.1, 1.4, "cut + split, each add-on level\n(its own star gathered;\npairing rank-local)",
         ADD),
        (9.0, 3.4, "transfers\n(rank-local location; misses\nasked of the other ranks)", "#e8daef"),
        (9.0, 1.4, "per-level pair frames\n(rank-local pair blocks)", "#e8daef"),
        (3.2, 1.4, "extrude (thick twin)\n(built on one rank, distributed,\nstars re-gathered, "
                   "pairing as labels)", "#d5f5e3"),
    ]
    for x, y, txt, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.6, 1.3, boxstyle="round,pad=0.03", fc=fc,
                                    ec="#555", lw=0.8))
        ax.text(x + 1.3, y + 0.65, txt, ha="center", va="center", fontsize=8.5)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.0))
    arrow(2.9, 4.05, 3.2, 4.05)
    arrow(5.8, 4.05, 6.1, 4.05)
    arrow(5.8, 3.6, 6.1, 2.3)
    arrow(8.7, 4.05, 9.0, 4.05)
    arrow(8.7, 2.05, 9.0, 2.05)
    arrow(8.7, 2.3, 9.0, 3.6)
    arrow(7.4, 3.4, 5.8, 2.7)
    ax.text(0.3, 0.55, "Load: everything stays load-balanced except the fault stars (a layer "
            "about two cells thick around each fault, on every level),\nwhich sit on the rank "
            "holding most of the star. No communication inside the contact solve; the "
            "imbalance is bounded by the stars, not by the refined band.",
            fontsize=9, va="top")
    ax.set_xlim(0, 12.2)
    ax.set_ylim(-0.2, 5.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parallel_pipeline.png"), dpi=160)
    plt.close(fig)


strategies()
transfer_rule()
parallel_pipeline()
print("figures in", out)


def embedding_routes():
    """The two embeddings of a fault and what each does for multigrid: the
    cut (where C lives) and the placed ladder band (welded tail + FAC patch),
    and the proposed meeting point: a 2:1 sub-sampled ladder split on every
    level."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    names = ["finest", "add-on 2", "add-on 1", "background 1", "background 0"]
    kinds = ["fine", "add", "add", "bg", "bg"]

    def column(ax, title, faults, bands, notes):
        for i, (nm, kd, f, b) in enumerate(zip(names, kinds, faults, bands)):
            y = 4.0 - i * 0.95
            level_box(ax, y, kd, nm, f)
            if b:
                xc = 0.2 + 3.6 * 0.62
                ax.add_patch(Polygon([[xc - 1.05, y + 0.12], [xc + 1.05, y + 0.12],
                                      [xc + 1.05, y + 0.60], [xc - 1.05, y + 0.60]],
                                     closed=True, fc="none", ec="#2e7d32", lw=1.3, ls="-"))
        ax.set_xlim(0, 4.0)
        ax.set_ylim(-1.3, 5.0)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.text(0.2, -0.25, notes, fontsize=8.5, va="top")
        ax.axis("off")

    column(axes[0], "the cut  (C, in the library)",
           ["split", "split", "split", "weld", "weld"], [False] * 5,
           "adapt refines toward the trace; the trace is\ncut into the existing edges "
           "(vertices pulled onto it)\nand split, on the finest AND on every add-on\nlevel. "
           "Whole-level smoothing. Welded background.")
    column(axes[1], "the ladder band  (#629 / #672, in the library)",
           ["split", "weld", "weld", "weld", "weld"], [True, False, False, False, False],
           "a structured band (rails + exact spine, green)\nis placed into the finest by "
           "carve-and-fill;\nthe spine is split without cutting (cut=False).\nWelded tail; "
           "the finest smoother gains an ASM\npatch (LU blocks) over the band's cells.")
    column(axes[2], "proposed: C on the ladder",
           ["split", "split", "split", "weld", "weld"], [True, True, True, False, False],
           "the ladder sub-sampled 2:1 shares every spine\nvertex between levels: placed "
           "and split on each\nadd-on level, the slits NEST (no clamp needed).\nThe FAC patch "
           "can stay on the finest. Not built;\nthe no-mid measurements had welded mids only.")
    axes[0].plot([], [], color=SPLIT, lw=2.2, label="split slit")
    axes[0].plot([], [], color="#2e7d32", lw=1.3, label="placed ladder band (cell label)")
    axes[0].plot([], [], color=WELD, lw=1.0, ls=":", label="welded")
    fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "embedding_routes.png"), dpi=160)
    plt.close(fig)


embedding_routes()
