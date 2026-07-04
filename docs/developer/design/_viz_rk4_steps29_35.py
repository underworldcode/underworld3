"""Pyvista grid of T (P3, on DOF cloud) + deformed mesh edges for
rk4-full run, steps 29 through 35 (approaching and through first ring).
"""
import os
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True

SNAP_DIR = "output/convection_zoo_snapshots_rk4_full"
STEPS = [29, 30, 31, 32, 33, 34, 35]
OUT_PNG = "output/rk4_T_meshes_steps29_35.png"


def load_T(snap_dir, step):
    root = f"uw_rk4_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


# 2 rows x 4 cols, last cell empty
plotter = pv.Plotter(shape=(2, 4), window_size=(2200, 1100),
                     border=False, off_screen=True)
plotter.set_background("white")

for i, step in enumerate(STEPS):
    row, col = divmod(i, 4)
    mesh, T = load_T(SNAP_DIR, step)
    Tmin, Tmax = float(T.data[:, 0].min()), float(T.data[:, 0].max())
    print(f"step {step:2d}: T=[{Tmin:+.3f}, {Tmax:+.3f}]")

    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    plotter.subplot(row, col)
    plotter.set_background("white")
    show_bar = (i == 0)
    plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                     clim=(0, 1), show_edges=False, lighting=False,
                     show_scalar_bar=show_bar,
                     scalar_bar_args={"title": "T",
                                      "vertical": False,
                                      "position_x": 0.15,
                                      "position_y": 0.04,
                                      "width": 0.7,
                                      "height": 0.04,
                                      "title_font_size": 16,
                                      "label_font_size": 14}
                     if show_bar else None)
    plotter.add_mesh(edges, color="black", line_width=0.5,
                     lighting=False)
    plotter.add_text(
        f"step {step}    T∈[{Tmin:+.2f}, {Tmax:+.2f}]",
        position="upper_edge", font_size=12, color="black")
    plotter.view_xy()

# Blank last cell
plotter.subplot(1, 3)
plotter.set_background("white")

plotter.screenshot(OUT_PNG, transparent_background=False,
                   window_size=(2200, 1100))
plotter.close()
print(f"wrote {OUT_PNG}")
