"""A transversely isotropic band built through a partition seam (UWTN 2026-018).

A fault network with the weak-plane (TI) realisation, built at np = 2 with
``seams="conform"``: the band is meshed through the seam in place, nothing
gathered. The figure colours each rank's cells, outlines the band cells and
marks the band vertices both ranks hold (the seam inside the band).

STAND-IN FIXTURE: the note asks for the "S-fault rig", whose script is not on
this machine (only its measurements are cited in the library comments). The
fault here is an S-shaped trace of the same kind; replace ``trace()`` with
the rig's when it is to hand.

Two runs, because rank ownership is the thing drawn: the build at np = 2
writes each rank's cells to its own file, the serial render reads them.

    mpirun -np 2 python fig_band_seam.py -uw_stage build
    python fig_band_seam.py -uw_stage render -uw_out band-through-the-boundary.png
"""
import glob
import os

import numpy as np

import underworld3 as uw

params = uw.Params(
    uw_stage="build",            # "build" (parallel) or "render" (serial)
    uw_h=0.02,                   # the network's spacing along the trace
    uw_amplitude=0.04,           # the S trace: half its peak-to-peak height
                                 # (0.08, 0.15 fail at np 2: see README)
    uw_files="band_seam_rank",   # per-rank files: <stem><rank>.npz
    uw_out="band-through-the-boundary.png",
)

H = float(params.uw_h)
AMPLITUDE = float(params.uw_amplitude)
RANK_COLOURS = ("#d6eaf8", "#fdebd0", "#d5f5e3", "#f5eef8")
BAND_EDGE, SHARED = "#7d3c98", "#c0392b"


def trace():
    """An S-shaped fault across the box, sampled at the network spacing."""
    x = np.arange(0.2, 0.8 + 1e-12, H)
    y = 0.5 + AMPLITUDE * np.sin(2 * np.pi * (x - 0.2) / 0.6)
    return np.column_stack([x, y])


if str(params.uw_stage) == "build":
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=4 * H,
        regular=False, refinement=1, qdegree=2)
    network = uw.meshing.FaultNetwork([("S", trace())])
    network.prepare(h=H, verbose=False)
    network.build(base=base, width=2 * H, realisation="ti", max_levels=1,
                  seams="conform", band=2 * H, ramp=6 * H)
    mesh = network.mesh
    dm = mesh.dm
    start, end = dm.getHeightStratum(0)
    v_start, v_end = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    corners = np.array([[p - v_start for p in dm.getTransitiveClosure(c)[0]
                         if v_start <= p < v_end] for c in range(start, end)])
    band = np.asarray(mesh.cells_labelled("Band"), dtype=bool)
    # this rank's piece; the rank number is the quantity the figure draws
    np.savez(f"{params.uw_files}{uw.mpi.rank}.npz", X=X, corners=corners,
             band=band, trace=trace())
    uw.pprint(f"built: band cells this rank {int(band.sum())}, "
              f"cells this rank {end - start}")
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    pieces = [np.load(f) for f in sorted(glob.glob(f"{params.uw_files}*.npz"))]
    # a band vertex held by more than one rank is on the seam
    held = {}
    for r, piece in enumerate(pieces):
        band_vertices = np.unique(piece["corners"][piece["band"]])
        for xy in np.round(piece["X"][band_vertices], 10):
            held.setdefault(tuple(xy), set()).add(r)
    shared = np.array([xy for xy, ranks in held.items() if len(ranks) > 1])

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for r, piece in enumerate(pieces):
        cells = piece["X"][piece["corners"]]
        ax.add_collection(PolyCollection(cells, facecolors=RANK_COLOURS[r],
                                         edgecolors="#7f8c8d", lw=0.2))
        ax.add_collection(PolyCollection(cells[piece["band"]],
                                         facecolors="none",
                                         edgecolors=BAND_EDGE, lw=0.5))
    ax.plot(*pieces[0]["trace"].T, color="k", lw=0.8)
    if len(shared):
        ax.plot(shared[:, 0], shared[:, 1], "o", ms=2.5, color=SHARED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    counts = [int(p["band"].sum()) for p in pieces]
    ax.set_title(f"np {len(pieces)}: band cells per rank {counts}, "
                 f"{len(shared)} shared band vertices", fontsize=9)
    fig.tight_layout()
    fig.savefig(params.uw_out, dpi=200)
    print(f"wrote {params.uw_out}: cells per rank "
          f"{[len(p['corners']) for p in pieces]}, band {counts}, "
          f"shared band vertices {len(shared)}")
