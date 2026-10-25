"""A fault moved through three positions on one background (UWTN 2026-022).

The box and fault of docs/developer/subsystems/fault-split-multigrid.md, the
fault translated from x = 0.3 to 0.7 at add-on depth k = 4. The background
is built once and never cut; at each position the add-on part is rebuilt
from it (one ``adapt``), the fault is cut and split on every add-on level
(``add_fault``), and the frictionless Stokes problem is solved. Each run
writes the per-step wall times, velocity iterations and peak slip to JSON.

The figure: top row, the mesh and log10 of the deviatoric stress invariant
near the fault at each position; bottom row, stacked wall time per step for
the runs whose JSON is listed in ``-uw_bars``. Outside the refinement band
every cell is a background cell: the count of cells there that coincide
with the background's finest level is recorded per position (distance
``-uw_far`` from the trace; grading and conformity reach past the band).

    mpirun -np 8 python fig_moving_fault.py -uw_render 0 -uw_json np8.json
    python fig_moving_fault.py -uw_render 1 -uw_json np1.json \\
        -uw_bars np8.json,np1.json -uw_out moving-fault.png
"""
import json
import time

import numpy as np
import sympy

import underworld3 as uw

params = uw.Params(
    uw_depth=4,                 # add-on levels k toward the fault
    uw_positions=3,             # fault centres from x = 0.3 to 0.7
    uw_base_cell=0.1,           # background cell size; refinement 1 halves it
    uw_band_width=0.08,         # width of the refinement band about the trace
    uw_far=0.15,                # beyond this from the trace: background test
    uw_render=1,                # draw the figure (a serial run)
    uw_json="timings.json",     # this run's record
    uw_bars="",                 # records to draw as bars, comma separated
    uw_out="moving-fault.png",
)

HALF = np.array([0.20, 0.05])      # half the fault, centre to tip
VIEW = 0.30                         # half-width of the plotted window
CENTRES = np.linspace(0.3, 0.7, int(params.uw_positions))
STAGES = ("refine", "conform and split", "solve")

base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=params.uw_base_cell, refinement=1, qdegree=2)
h_base = params.uw_base_cell / 2
background_finest = base._coarse_level_meshes()[-1]


def centroids(m):
    """Cell centroids of a level, rounded so equal cells compare equal."""
    dm = m.dm
    start, end = dm.getHeightStratum(0)
    v_start, v_end = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    corners = [[p - v_start for p in dm.getTransitiveClosure(c)[0]
                if v_start <= p < v_end] for c in range(start, end)]
    return np.round(X[np.array(corners)].mean(axis=1), 10), np.array(corners), X


def distance_to(points, tip_a, tip_b):
    """Distance from each point to the segment tip_a - tip_b."""
    d = tip_b - tip_a
    t = np.clip((points - tip_a) @ d / (d @ d), 0.0, 1.0)
    return np.linalg.norm(points - (tip_a + t[:, None] * d), axis=1)


def solve_at(centre):
    """Rebuild the add-on part for a fault centred at ``centre``; solve."""
    tip_a = np.array([centre, 0.5]) - HALF
    tip_b = np.array([centre, 0.5]) + HALF
    times = {}

    start = time.perf_counter()
    samples = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                               np.linspace(tip_a[1], tip_b[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface(f"guide{centre:.2f}", base, samples)
    guide.discretize()
    metric = guide.refinement_metric_function(
        h_near=h_base / 2 ** int(params.uw_depth), h_far=h_base,
        width=params.uw_band_width, profile="linear")
    child = base.adapt(metric, max_levels=int(params.uw_depth))
    times["refine"] = time.perf_counter() - start

    start = time.perf_counter()
    mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))])
    times["conform and split"] = time.perf_counter() - start

    start = time.perf_counter()
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)      # simple shear
    stokes.add_fault_bc(0, "Flt")                           # frictionless
    stokes.solve()
    times["solve"] = time.perf_counter() - start

    info = stokes._rotated_freeslip_info
    from underworld3.utilities.fault_contact import fault_slip
    _s, slip, _leak = fault_slip(stokes, "Flt", info)
    # this rank's largest slip: the whole fault's in a serial run
    peak = float(np.abs(slip).max()) if len(slip) else 0.0
    record = {"centre": float(centre), "np": uw.mpi.size,
              "cells_this_rank": int(mesh.dm.getHeightStratum(0)[1]),
              "levels": len(mesh._custom_mg_coarse_meshes) + 1,
              "vel_its": info["vel_its_last"],
              "peak_slip_local_max": peak,
              "times": {k: round(v, 3) for k, v in times.items()}}
    return record, mesh, stokes, (tip_a, tip_b)


records, fields = [], []
for centre in CENTRES:
    record, mesh, stokes, tips = solve_at(centre)
    records.append(record)
    uw.pprint("STEP " + json.dumps(record))
    if int(params.uw_render):
        # tau_II = 2 eta sqrt(E:E / 2), eta = 1, at the cell centroids
        E = mesh.vector.strain_tensor(stokes.Unknowns.u.sym)
        tau = 2 * sympy.sqrt(sum(E[i, j] ** 2 for i in range(2)
                                 for j in range(2)) / 2)
        cent, corners, X = centroids(mesh)
        values = np.asarray(uw.function.evaluate(tau, cent)).reshape(-1)
        # cells beyond the band that coincide with background cells
        far = distance_to(cent, *tips) > params.uw_far
        base_cent = {tuple(c) for c in centroids(background_finest)[0]}
        same = sum(tuple(c) in base_cent for c in cent[far])
        record["far_cells"] = int(far.sum())
        record["far_cells_background"] = int(same)
        fields.append((X, corners, values, tips, record))

with uw.selective_ranks(0) as writer:
    if writer:
        with open(params.uw_json, "w") as f:
            json.dump(records, f, indent=1)

if int(params.uw_render):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    runs = [json.load(open(p)) for p in str(params.uw_bars).split(",") if p]
    columns = len(fields)
    fig, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 4.6),
                             gridspec_kw={"height_ratios": [1.0, 1.1],
                                          "hspace": 0.25})
    low = min(np.log10(f[2][f[2] > 0].min()) for f in fields)
    high = max(np.log10(f[2].max()) for f in fields)
    for ax, (X, corners, values, tips, record) in zip(axes[0], fields):
        tri = Triangulation(X[:, 0], X[:, 1], corners)
        image = ax.tripcolor(tri, np.log10(np.maximum(values, 10 ** low)),
                             cmap="viridis", vmin=low, vmax=high)
        ax.triplot(tri, color="k", lw=0.12, alpha=0.35)
        ax.plot(*np.vstack(tips).T, color="#c0392b", lw=1.2)
        cx = record["centre"]
        ax.set_xlim(cx - VIEW, cx + VIEW)
        ax.set_ylim(0.5 - VIEW * 0.6, 0.5 + VIEW * 0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"x = {cx:.1f}: {record['cells_this_rank']} cells, "
                     f"{record['vel_its'][0]} its", fontsize=8)
    fig.colorbar(image, ax=axes[0], shrink=0.8,
                 label=r"$\log_{10}\,\tau_{II}$")

    colours = ("#7f8c8d", "#2e86c1", "#c0392b")
    width = 0.8 / max(len(runs), 1)
    tallest = max(sum(run[step]["times"].values()) for run in runs
                  for step in range(columns))
    for ax, step in zip(axes[1], range(columns)):
        for r, run in enumerate(runs):
            bottom = 0.0
            for stage, colour in zip(STAGES, colours):
                t = run[step]["times"][stage]
                ax.bar(r * width, t, width * 0.9, bottom=bottom,
                       color=colour, label=stage if r == 0 else None)
                bottom += t
        ax.set_xticks([r * width for r in range(len(runs))])
        ax.set_xticklabels([f"np {run[step]['np']}" for run in runs])
        ax.set_ylim(0, 1.05 * tallest)
    axes[1][0].set_ylabel("wall time per step (s)")
    axes[1][-1].legend(fontsize=7, frameon=False, loc="upper left",
                       bbox_to_anchor=(1.02, 1.0))
    fig.savefig(params.uw_out, dpi=200, bbox_inches="tight")
    uw.pprint(f"wrote {params.uw_out}")
