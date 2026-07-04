"""rk4_sl WITH vs WITHOUT --ale-correction.

Top row: ALE on, bottom row: ALE off. Same other args.
Both use the proper UW3 pyvista pipeline (T on its P3 DOF cloud,
deformed-mesh edges overlaid, no VTU).
"""
from __future__ import annotations
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import underworld3 as uw
import underworld3.visualisation as vis

SNAP_ALE = "output/convection_zoo_snapshots_rk4sl"
SNAP_NOALE = "output/convection_zoo_snapshots_rk4sl_noale"
OUT_PNG = "/tmp/rk4sl_ale_compare.png"
STEPS = [8, 16, 24, 32, 40]
T_NAME = "T_conv_v2p1"
SCHEME = "rk4_sl"

pv.OFF_SCREEN = True
WINDOW = (900, 900)
ZOOM = 1.15


def render(snap_dir: str, label_short: str, out: str) -> bool:
    mesh_path = f"{snap_dir}/uw_{SCHEME}_{label_short}.mesh.00000.h5"
    if not os.path.exists(mesh_path):
        return False
    mesh = uw.discretisation.Mesh(mesh_path)
    T = uw.discretisation.MeshVariable(
        T_NAME, mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(f"uw_{SCHEME}_{label_short}", T_NAME, 0,
                    outputPath=snap_dir)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    plotter = pv.Plotter(window_size=WINDOW, off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                     clim=(0.0, 1.0), show_edges=False,
                     lighting=False, show_scalar_bar=False)
    plotter.add_mesh(edges, color="black", line_width=0.5,
                     lighting=False)
    plotter.view_xy()
    plotter.camera.zoom(ZOOM)
    plotter.screenshot(out)
    plotter.close()
    return True


tmp = "/tmp/pv_ale_compare"
os.makedirs(tmp, exist_ok=True)
tiles = {}  # (row_label, step) -> path
for row_label, snap_dir in [("ALE on", SNAP_ALE),
                            ("ALE off", SNAP_NOALE)]:
    for s in STEPS:
        out = os.path.join(
            tmp, f"{row_label.replace(' ', '_')}_step{s:04d}.png")
        if render(snap_dir, f"step{s:04d}", out):
            tiles[(row_label, s)] = out

fig, axes = plt.subplots(2, len(STEPS),
                         figsize=(2.8 * len(STEPS), 6.0))
for row, label in enumerate(["ALE on", "ALE off"]):
    for col, s in enumerate(STEPS):
        ax = axes[row, col]
        path = tiles.get((label, s))
        if path:
            ax.imshow(mpimg.imread(path))
        title = f"step {s}"
        if col == 0:
            title = f"[{label}]  " + title
        ax.set_title(title, fontsize=10)
        ax.axis("off")

fig.suptitle(
    "rk4_sl + Ra=ρg=1e5 + CN + clamp  (40 steps)\n"
    "top: --ale-correction on (ΔA/A=-3.0%, h_pole=-9.4e-3)  |  "
    "bottom: --ale-correction off (ΔA/A=-3.4%, h_pole=-9.9e-3)",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
