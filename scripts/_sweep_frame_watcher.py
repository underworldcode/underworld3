"""Watch a list of sweep run directories. When a new step snapshot
appears in any of them, render a mesh-on-T frame using the same
RdBu_r/edges style as the live-plot sidecar.

Frames go to <dir>/diagnostics/frames/step####.png and the
latest-rendered overall mesh goes to <dir>/diagnostics/live_mesh.png
so the snapshot history is preserved.
"""
from __future__ import annotations
import os
import re
import glob
import time
import argparse
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

import underworld3 as uw


p = argparse.ArgumentParser()
p.add_argument("--root", type=str,
               default=os.path.expanduser(
                   "~/+Simulations/StagnantLid"))
p.add_argument("--prefix", type=str, default="ot_sweep_",
               help="watch dirs matching <root>/<prefix>*")
p.add_argument("--interval", type=float, default=10.0)
args = p.parse_args()

R_OUTER, R_INNER = 1.0, 0.5
print(f"watcher: <{args.root}>/{args.prefix}*  every "
      f"{args.interval:.0f}s",
      flush=True)

_seen = {}  # dir -> set of already-rendered step ints


def list_dirs():
    return sorted(glob.glob(os.path.join(
        args.root, args.prefix + "*")))


def latest_steps(d):
    out = []
    for f in glob.glob(os.path.join(d, "step*.mesh.00000.h5")):
        m = re.search(r"step(\d+)\.mesh\.00000\.h5$",
                       os.path.basename(f))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def render(d, step):
    label = f"step{step:04d}"
    diag = os.path.join(d, "diagnostics")
    frames = os.path.join(diag, "frames")
    os.makedirs(frames, exist_ok=True)
    out_path = os.path.join(frames, f"{label}.png")
    live_path = os.path.join(diag, "live_mesh.png")
    if os.path.exists(out_path):
        return True
    try:
        mesh = uw.discretisation.Mesh(
            os.path.join(d, f"{label}.mesh.00000.h5"))
        T = uw.discretisation.MeshVariable(
            "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
            continuous=True, varsymbol="T")
        T.read_timestep(label, "T_v2p1", 0, outputPath=d)
    except Exception as e:
        print(f"  load fail {d} {label}: {e}", flush=True)
        return False
    # Edges
    dm = mesh.dm
    pStart, _ = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    coords = np.asarray(mesh.X.coords)
    segs = np.empty((eEnd - eStart, 2, 2), dtype=float)
    for k, e in enumerate(range(eStart, eEnd)):
        cone = dm.getCone(e)
        segs[k, 0] = coords[cone[0] - pStart]
        segs[k, 1] = coords[cone[1] - pStart]
    Xc = np.asarray(T.coords)
    Tv = np.asarray(T.data[:, 0])
    tri = Triangulation(Xc[:, 0], Xc[:, 1])
    cx = Xc[tri.triangles, 0].mean(axis=1)
    cy = Xc[tri.triangles, 1].mean(axis=1)
    rcen = np.sqrt(cx**2 + cy**2)
    mask = (rcen > R_OUTER + 1e-6) | (rcen < R_INNER - 1e-6)
    tri.set_mask(mask)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                            constrained_layout=True)
    ax.tripcolor(tri, Tv, cmap="RdBu_r", shading="gouraud",
                 vmin=0, vmax=1)
    ax.add_collection(LineCollection(segs, colors="#202020",
                                       linewidths=0.4, alpha=0.7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}  ({os.path.basename(d)})",
                  fontsize=9)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    try:
        shutil.copyfile(out_path, live_path)
    except Exception:
        pass
    return True


while True:
    for d in list_dirs():
        if d not in _seen:
            _seen[d] = set()
        for s in latest_steps(d):
            if s in _seen[d]:
                continue
            if render(d, s):
                _seen[d].add(s)
                print(f"  rendered {os.path.basename(d)} step{s:04d}",
                      flush=True)
    time.sleep(args.interval)
