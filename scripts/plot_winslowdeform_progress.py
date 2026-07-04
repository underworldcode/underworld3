"""Incremental compare: baseline (May 14) vs restored-fix run.
Plots whatever steps are currently captured in the restored-fix
snapshot dir. Top row = baseline, bottom = restored-fix.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

BASE = ("/Users/lmoresi/+Underworld/underworld3-pixi/.claude/"
        "worktrees/exp-integrator-freesurface/output/"
        "convection_zoo_snapshots_rk4_monotone_clamp")
RESTORED = "output/convection_zoo_snapshots_winslowdeform"
OUT = "/tmp/winslowdeform_progress.png"
SCHEME = "rk4"
TN = "T_conv_v2p1"
ALL_STEPS = [5, 10, 15, 20, 25, 30, 35]

# Only steps the restored run has actually written
steps = [s for s in ALL_STEPS
         if os.path.exists(
             f"{RESTORED}/uw_{SCHEME}_step{s:04d}.mesh.00000.h5")]
if not steps:
    print("No restored checkpoints yet")
    sys.exit(0)
print(f"Plotting steps: {steps}")

P3B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                [2/3, 1/3, 0], [1/3, 2/3, 0],
                [0, 2/3, 1/3], [0, 1/3, 2/3],
                [1/3, 0, 2/3], [2/3, 0, 1/3],
                [1/3, 1/3, 1/3]])
P3S = np.array([[0, 3, 8], [3, 9, 8], [3, 4, 9], [4, 5, 9],
                [4, 1, 5], [8, 9, 7], [9, 5, 6], [9, 6, 7],
                [7, 6, 2]])


def bp3(mesh, T):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    co = np.asarray(mesh.X.coords)
    tr = cKDTree(np.asarray(T.coords))
    ts = []
    for c in range(cS, cE):
        cl, _ = dm.getTransitiveClosure(c, useCone=True)
        vv = [p - pS for p in cl if pS <= p < pE]
        if len(vv) != 3:
            continue
        _, idx = tr.query(P3B @ co[vv], k=1)
        for s in P3S:
            ts.append(idx[s])
    return np.asarray(ts, dtype=np.int64)


pv.OFF_SCREEN = True
PNL = 900
pl = pv.Plotter(shape=(2, len(steps)),
                window_size=(PNL * len(steps), PNL * 2),
                off_screen=True, border=False)
pl.set_background("white")
pl.disable_anti_aliasing()

for r, (lab, sn) in enumerate(
        [("baseline", BASE), ("winslow-deform", RESTORED)]):
    for c, s in enumerate(steps):
        mp = f"{sn}/uw_{SCHEME}_step{s:04d}.mesh.00000.h5"
        if not os.path.exists(mp):
            continue
        m = uw.discretisation.Mesh(mp)
        T = uw.discretisation.MeshVariable(
            TN, m, vtype=uw.VarType.SCALAR,
            degree=3, continuous=True)
        T.read_timestep(f"uw_{SCHEME}_step{s:04d}", TN, 0,
                        outputPath=sn)
        ts = bp3(m, T)
        pts = np.zeros((T.coords.shape[0], 3))
        pts[:, :2] = T.coords
        fc = np.column_stack([
            np.full(ts.shape[0], 3, dtype=np.int64), ts]).ravel()
        pT = pv.PolyData(pts, faces=fc)
        pT.point_data["T"] = np.asarray(T.data[:, 0])
        eg = vis.mesh_to_pv_mesh(m).extract_all_edges()
        pl.subplot(r, c)
        pl.add_mesh(pT, scalars="T", cmap="RdBu_r", clim=(0, 1),
                    show_edges=False, lighting=False,
                    show_scalar_bar=False)
        pl.add_mesh(eg, color="black", line_width=0.5,
                    lighting=False)
        t = f"step {s}" if c else f"[{lab}] step {s}"
        pl.add_text(t, font_size=16, color="black",
                    position="upper_left")
        pl.view_xy()
        pl.camera.zoom(1.18)
pl.screenshot(OUT)
pl.close()
print(f"Saved {OUT} ({len(steps)} steps)")
