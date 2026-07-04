"""Zoom on the RK2 ringing onset: steps 32, 33, 34 (just-pre, just-onset,
ringing). Bracketed by 31 (clean) and 35 (catastrophic) for context.
"""
import os
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True

SNAP_DIR = "output/convection_zoo_snapshots_rk2_full"
STEPS = [31, 32, 33, 34, 35]
OUT_PNG = "output/rk2_full_onset_31_35.png"


def load_T(snap_dir, step):
    root = f"uw_rk2_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


plotter = pv.Plotter(shape=(1, 5), window_size=(2750, 700),
                     border=False, off_screen=True)
plotter.set_background("white")

for i, step in enumerate(STEPS):
    mesh, T = load_T(SNAP_DIR, step)
    Tmin, Tmax = float(T.data[:, 0].min()), float(T.data[:, 0].max())
    print(f"step {step:2d}: T=[{Tmin:+.3f}, {Tmax:+.3f}]")

    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    plotter.subplot(0, i)
    plotter.set_background("white")
    # Cap colour range to [-0.2, 1.2] so steps 34/35 ringing is visible
    # but earlier clean steps stay readable.
    plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                     clim=(-0.2, 1.2), show_edges=False, lighting=False,
                     show_scalar_bar=(i == 0),
                     scalar_bar_args={"title": "T",
                                      "vertical": False,
                                      "position_x": 0.1,
                                      "position_y": 0.04,
                                      "width": 0.8,
                                      "height": 0.05,
                                      "title_font_size": 18,
                                      "label_font_size": 14}
                     if i == 0 else None)
    plotter.add_mesh(edges, color="black", line_width=0.5,
                     lighting=False)
    plotter.add_text(
        f"step {step}\nT∈[{Tmin:+.2f}, {Tmax:+.2f}]",
        position="upper_edge", font_size=14, color="black")
    plotter.view_xy()

plotter.screenshot(OUT_PNG, transparent_background=False,
                   window_size=(2750, 700))
plotter.close()
print(f"wrote {OUT_PNG}")
