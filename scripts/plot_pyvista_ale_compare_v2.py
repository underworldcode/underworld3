"""rk4_sl ALE on vs off — native pyvista 2×5 subplot, no matplotlib
downsampling. T on P3 DOF cloud, deformed-mesh edges overlaid.
"""
from __future__ import annotations
import os
import numpy as np
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis

SNAP_ALE = "output/convection_zoo_snapshots_rk4sl"
SNAP_NOALE = "output/convection_zoo_snapshots_rk4sl_noale"
OUT_PNG = "/tmp/rk4sl_ale_compare_v2.png"
STEPS = [8, 16, 24, 32, 40]
T_NAME = "T_conv_v2p1"
SCHEME = "rk4_sl"

pv.OFF_SCREEN = True

# 2 rows × 5 cols at 800px per panel = 4000×1600.
plotter = pv.Plotter(
    shape=(2, len(STEPS)),
    window_size=(800 * len(STEPS), 800 * 2),
    off_screen=True, border=False)
plotter.set_background("white")

rows = [("ALE on", SNAP_ALE), ("ALE off", SNAP_NOALE)]
for r, (label, snap_dir) in enumerate(rows):
    for c, s in enumerate(STEPS):
        mesh_path = (
            f"{snap_dir}/uw_{SCHEME}_step{s:04d}.mesh.00000.h5")
        if not os.path.exists(mesh_path):
            continue
        mesh = uw.discretisation.Mesh(mesh_path)
        T = uw.discretisation.MeshVariable(
            T_NAME, mesh, vtype=uw.VarType.SCALAR,
            degree=3, continuous=True)
        T.read_timestep(f"uw_{SCHEME}_step{s:04d}", T_NAME, 0,
                        outputPath=snap_dir)
        pv_T = vis.meshVariable_to_pv_mesh_object(T)
        pv_T.point_data["T"] = np.asarray(T.data[:, 0])
        edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
        plotter.subplot(r, c)
        plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                         clim=(0.0, 1.0), show_edges=False,
                         lighting=False, show_scalar_bar=False)
        plotter.add_mesh(edges, color="black", line_width=0.6,
                         lighting=False)
        plotter.add_text(f"[{label}]  step {s}",
                         font_size=18, color="black",
                         position="upper_left")
        plotter.view_xy()
        plotter.camera.zoom(1.18)

plotter.screenshot(OUT_PNG)
plotter.close()
print(f"Saved {OUT_PNG}")
