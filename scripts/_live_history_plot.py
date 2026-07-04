"""Live-update history plot for a running stagnant_lid_adapt_loop
run. Polls <out_dir>/history.npz every N seconds and rewrites
<out_dir>/diagnostics/live_history.png. Also detects the newest
step snapshot and renders mesh-on-T to live_mesh.png.

Stop with Ctrl-C or TaskStop.
"""
from __future__ import annotations
import os
import re
import sys
import time
import glob
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation


p = argparse.ArgumentParser()
p.add_argument("--out-dir", type=str, required=True)
p.add_argument("--interval", type=float, default=15.0)
args = p.parse_args()

SRC = args.out_dir
DIAG = os.path.join(SRC, "diagnostics")
FRAMES_DIR = os.path.join(DIAG, "frames")
os.makedirs(DIAG, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
HIST = os.path.join(SRC, "history.npz")
PNG = os.path.join(DIAG, "live_history.png")
MESH_PNG = os.path.join(DIAG, "live_mesh.png")
print(f"live plot: {HIST} → {PNG} every {args.interval:.0f}s",
      flush=True)
print(f"  mesh plot → {MESH_PNG} on snapshot updates", flush=True)
print(f"  per-snapshot frames → {FRAMES_DIR}/step####.png",
      flush=True)


_last_mesh_step = -1
_uw_loaded = False
_uw = None
_vis = None


def latest_snapshot():
    files = glob.glob(os.path.join(SRC, "step*.mesh.00000.h5"))
    if not files:
        return None
    steps = []
    for f in files:
        m = re.search(r"step(\d+)\.mesh\.00000\.h5$",
                       os.path.basename(f))
        if m:
            steps.append((int(m.group(1)),
                          os.path.basename(f).replace(
                              ".mesh.00000.h5", "")))
    if not steps:
        return None
    return max(steps)


def _ensure_uw():
    global _uw_loaded, _uw, _vis
    if _uw_loaded:
        return
    import underworld3 as uw_
    _uw = uw_
    _uw_loaded = True


def plot_mesh(label):
    """Render mesh-on-T panel for the latest snapshot."""
    _ensure_uw()
    uw = _uw
    mesh = uw.discretisation.Mesh(
        os.path.join(SRC, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    try:
        T.read_timestep(label, "T_v2p1", 0, outputPath=SRC)
    except Exception as e:
        print(f"  T read fail at {label}: {e}", flush=True)
        return False
    # Build edges directly from the dm.
    dm = mesh.dm
    pStart, _ = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    coords = np.asarray(mesh.X.coords)
    segs = np.empty((eEnd - eStart, 2, 2), dtype=float)
    for k, e in enumerate(range(eStart, eEnd)):
        cone = dm.getCone(e)
        segs[k, 0] = coords[cone[0] - pStart]
        segs[k, 1] = coords[cone[1] - pStart]
    # Render T on the T-DOF cloud
    Xc = np.asarray(T.coords)
    Tv = np.asarray(T.data[:, 0])
    tri = Triangulation(Xc[:, 0], Xc[:, 1])
    cx = Xc[tri.triangles, 0].mean(axis=1)
    cy = Xc[tri.triangles, 1].mean(axis=1)
    rcen = np.sqrt(cx**2 + cy**2)
    mask = (rcen > 1.0 + 1e-6) | (rcen < 0.5 - 1e-6)
    tri.set_mask(mask)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                            constrained_layout=True)
    ax.tripcolor(tri, Tv, cmap="RdBu_r", shading="gouraud",
                 vmin=0, vmax=1)
    ax.add_collection(LineCollection(segs, colors="#202020",
                                       linewidths=0.4, alpha=0.7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}  ({os.path.basename(SRC)})")
    fig.savefig(MESH_PNG, dpi=120)
    plt.close(fig)
    return True


def plot_once():
    if not os.path.exists(HIST):
        return False
    try:
        h = np.load(HIST)
    except Exception as e:
        print(f"  load fail: {e}", flush=True)
        return False
    step = h["step"]
    if len(step) == 0:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                              constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(step, h["vrms"], color="tab:blue")
    for s in step[h["adapted"] > 0.5]:
        ax.axvline(s, color="0.7", lw=0.5, zorder=-1)
    ax.set_xlabel("step"); ax.set_ylabel("vrms")
    ax.set_title("vrms (grey lines = adapt events)")

    ax = axes[0, 1]
    ax.plot(step, h["Nu"], color="tab:orange")
    ax.set_xlabel("step"); ax.set_ylabel("Nu")
    ax.set_title("Nu_surface")

    ax = axes[1, 0]
    ax.plot(step, h["Tmin"], color="tab:green", label="T_min")
    ax.plot(step, h["Tmax"], color="tab:red", label="T_max")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axhline(1, color="0.5", lw=0.5)
    ax.axhline(-0.1, color="r", lw=0.5, ls=":",
                label="abort window")
    ax.axhline(1.1, color="r", lw=0.5, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("T extents")
    ax.set_title("T_min / T_max")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(step, h["dt"], color="tab:purple")
    ax.set_xlabel("step"); ax.set_ylabel("dt")
    ax.set_title("Δt")

    fig.suptitle(
        f"{os.path.basename(SRC)}  —  step={int(step[-1])}  "
        f"vrms={h['vrms'][-1]:.2e}  Nu={h['Nu'][-1]:.3f}  "
        f"T=[{h['Tmin'][-1]:+.3f},{h['Tmax'][-1]:+.3f}]  "
        f"adapts={int(h['adapted'].sum())}")
    fig.savefig(PNG, dpi=110)
    plt.close(fig)
    return True


# Keep updating
n = 0
while True:
    if plot_once():
        n += 1
        if n % 4 == 0:
            print(f"  refresh #{n}: step={int(np.load(HIST)['step'][-1])}",
                  flush=True)
    snap = latest_snapshot()
    if snap is not None:
        snap_step, snap_label = snap
        if snap_step != _last_mesh_step:
            try:
                if plot_mesh(snap_label):
                    _last_mesh_step = snap_step
                    # Keep a per-snapshot copy with the step in
                    # its name so the full evolution is retained.
                    import shutil
                    frame_path = os.path.join(
                        FRAMES_DIR, f"{snap_label}.png")
                    shutil.copyfile(MESH_PNG, frame_path)
                    print(f"  mesh refresh → {snap_label} "
                          f"(+ {frame_path})",
                          flush=True)
            except Exception as e:
                print(f"  mesh refresh fail at {snap_label}: {e}",
                      flush=True)
    time.sleep(args.interval)
