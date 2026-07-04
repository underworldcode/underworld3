"""Mesh + T-field evolution via the proper UW3 pyvista pipeline.

Per [[feedback_pyvista_viz_pattern]]:
- Load mesh from saved h5 (uw_step*.mesh.00000.h5)
- Create matching MeshVariable, read_timestep its data
- Render T on its own DOF cloud (P3 high-order, not vertex-piecewise)
- Overlay deformed-mesh edges from vis.mesh_to_pv_mesh + extract_all_edges
- white background, lighting=False, no VTU
"""
from __future__ import annotations
import os
import numpy as np
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis

SNAP_DIR = "output/convection_zoo_snapshots_rk4sl"
OUT_PNG = "/tmp/rk4sl_evolution_pv.png"
STEPS = [8, 16, 24, 32, 40]
pair_tag = "v2p1"
T_NAME = f"T_conv_{pair_tag}"

pv.OFF_SCREEN = True

# Capture each step at the same camera + zoom, then composite via PIL.
WINDOW = (1100, 1100)
ZOOM_FACTOR = 1.15


SCHEME = "rk4_sl"  # matches the suffix used by _zoo's _capture


def render_step(label_short: str, screenshot_path: str):
    mesh_path = (
        f"{SNAP_DIR}/uw_{SCHEME}_{label_short}.mesh.00000.h5")
    if not os.path.exists(mesh_path):
        print(f"Missing {mesh_path}; skipping")
        return False
    mesh = uw.discretisation.Mesh(mesh_path)
    T = uw.discretisation.MeshVariable(
        T_NAME, mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(f"uw_{SCHEME}_{label_short}", T_NAME, 0,
                    outputPath=SNAP_DIR)

    # T on its own P3-DOF cloud
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])

    # Deformed-mesh edges
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    plotter = pv.Plotter(window_size=WINDOW, off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                     clim=(0.0, 1.0), show_edges=False,
                     lighting=False,
                     scalar_bar_args={"title": "T",
                                      "vertical": True,
                                      "position_x": 0.92,
                                      "position_y": 0.10,
                                      "height": 0.80,
                                      "width": 0.04})
    plotter.add_mesh(edges, color="black", line_width=0.5,
                     lighting=False)
    plotter.add_text(label_short, font_size=14, color="black",
                     position="upper_left")
    plotter.view_xy()
    plotter.camera.zoom(ZOOM_FACTOR)
    plotter.screenshot(screenshot_path)
    plotter.close()
    return True


tmpdir = "/tmp/pv_tiles"
os.makedirs(tmpdir, exist_ok=True)
tiles = []
for s in STEPS:
    label = f"step{s:04d}"
    out = os.path.join(tmpdir, f"{label}.png")
    if render_step(label, out):
        tiles.append(out)
print(f"Rendered {len(tiles)} tiles -> compositing")

# Composite the tiles side-by-side with matplotlib (easier than
# manipulating pv multi-subplot which has its own quirks).
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, len(tiles),
                         figsize=(3.0 * len(tiles), 3.4))
if len(tiles) == 1:
    axes = [axes]
for ax, path, s in zip(axes, tiles, STEPS):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(f"step {s}", fontsize=11)
    ax.axis("off")
fig.suptitle(
    "rk4_sl + ALE + Ra=ρg=1e5 (CN, clamp) — T on P3 DOF cloud, "
    "deformed-mesh edges overlaid; ΔA/A=-3.0% at step 40",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
