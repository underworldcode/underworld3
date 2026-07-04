"""Pyvista grid of T (P3 on DOF cloud) + deformed mesh edges for the
rk2-full run at selected steps spanning the thrashing regime.
"""
import os
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True

SNAP_DIR = "output/convection_zoo_snapshots_rk2_full"
STEPS = [5, 10, 15, 20, 25, 28, 30, 32]
OUT_PNG = "output/rk2_full_T_meshes.png"


def load_T(snap_dir, step):
    root = f"uw_rk2_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


plotter = pv.Plotter(shape=(2, 4), window_size=(2200, 1100),
                     border=False, off_screen=True)
plotter.set_background("white")

for i, step in enumerate(STEPS):
    row, col = divmod(i, 4)
    mesh, T = load_T(SNAP_DIR, step)
    Tmin, Tmax = float(T.data[:, 0].min()), float(T.data[:, 0].max())
    h_max = float(np.max(np.linalg.norm(mesh.X.coords, axis=1)) - 1.0)
    print(f"step {step:2d}: T=[{Tmin:+.3f}, {Tmax:+.3f}]  h_max={h_max:.3f}")

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
        f"step {step}    h_max={h_max:.3f}",
        position="upper_edge", font_size=12, color="black")
    plotter.view_xy()

plotter.screenshot(OUT_PNG, transparent_background=False,
                   window_size=(2200, 1100))
plotter.close()
print(f"wrote {OUT_PNG}")
