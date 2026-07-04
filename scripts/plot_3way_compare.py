"""Three-row compare: baseline (May 14) / pre-fix today / post-fix today.
Same CLI for all: rk4 + clamp, res=20, Ra=ρg=1e5, 35 steps.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

DIRS = [
    ("baseline (May 14)",
     "/Users/lmoresi/+Underworld/underworld3-pixi/.claude/"
     "worktrees/exp-integrator-freesurface/output/"
     "convection_zoo_snapshots_rk4_monotone_clamp"),
    ("pre-fix",
     "output/convection_zoo_snapshots_rk4_repro"),
    ("post-fix (_mesh_version bump)",
     "output/convection_zoo_snapshots_rk4_postfix"),
]
OUT = "/tmp/baseline_prefix_postfix.png"
STEPS = [5, 10, 15, 20, 25, 30, 35]
SCHEME = "rk4"
T_NAME = "T_conv_v2p1"

P3_BARY = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2/3, 1/3, 0], [1/3, 2/3, 0],
    [0, 2/3, 1/3], [0, 1/3, 2/3],
    [1/3, 0, 2/3], [2/3, 0, 1/3],
    [1/3, 1/3, 1/3],
])
P3_SUBTRIS = np.array([
    [0, 3, 8], [3, 9, 8], [3, 4, 9], [4, 5, 9], [4, 1, 5],
    [8, 9, 7], [9, 5, 6], [9, 6, 7], [7, 6, 2],
])


def build_p3(mesh, T):
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    coords = np.asarray(mesh.X.coords)
    tree = cKDTree(np.asarray(T.coords))
    tris = []
    for c in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [p - pStart for p in closure
                 if pStart <= p < pEnd]
        if len(verts) != 3:
            continue
        p3_pos = P3_BARY @ coords[verts]
        _, idx = tree.query(p3_pos, k=1)
        for sub in P3_SUBTRIS:
            tris.append(idx[sub])
    return np.asarray(tris, dtype=np.int64)


pv.OFF_SCREEN = True
PANEL = 900
plotter = pv.Plotter(shape=(len(DIRS), len(STEPS)),
                     window_size=(PANEL * len(STEPS),
                                  PANEL * len(DIRS)),
                     off_screen=True, border=False)
plotter.set_background("white")
plotter.disable_anti_aliasing()

for r, (label, snap) in enumerate(DIRS):
    for c, s in enumerate(STEPS):
        mesh_path = f"{snap}/uw_{SCHEME}_step{s:04d}.mesh.00000.h5"
        if not os.path.exists(mesh_path):
            continue
        mesh = uw.discretisation.Mesh(mesh_path)
        T = uw.discretisation.MeshVariable(
            T_NAME, mesh, vtype=uw.VarType.SCALAR,
            degree=3, continuous=True)
        T.read_timestep(f"uw_{SCHEME}_step{s:04d}", T_NAME, 0,
                        outputPath=snap)
        tris = build_p3(mesh, T)
        pts = np.zeros((T.coords.shape[0], 3))
        pts[:, :2] = T.coords
        faces = np.column_stack([
            np.full(tris.shape[0], 3, dtype=np.int64),
            tris]).ravel()
        pv_T = pv.PolyData(pts, faces=faces)
        pv_T.point_data["T"] = np.asarray(T.data[:, 0])
        edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
        plotter.subplot(r, c)
        plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                         clim=(0, 1), show_edges=False,
                         lighting=False, show_scalar_bar=False)
        plotter.add_mesh(edges, color="black", line_width=0.5,
                         lighting=False)
        ttl = f"step {s}"
        if c == 0:
            ttl = f"[{label}]  " + ttl
        plotter.add_text(ttl, font_size=16, color="black",
                         position="upper_left")
        plotter.view_xy()
        plotter.camera.zoom(1.18)
plotter.screenshot(OUT)
plotter.close()
print(f"Saved {OUT}")
