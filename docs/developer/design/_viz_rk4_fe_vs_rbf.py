"""Side-by-side RK4 FE-traceback (rings at step 35) vs RK4 RBF-traceback
(clean, but more diffusive) for steps 28-35.
"""
import os
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True

FE_DIR  = "output/convection_zoo_snapshots_rk4_full"
RBF_DIR = "output/convection_zoo_snapshots_rk4_rbf_traceback"
STEPS = [28, 31, 33, 34, 35]
OUT_PNG = "output/rk4_fe_vs_rbf_comparison.png"


def load_T(snap_dir, step):
    root = f"uw_rk4_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


plotter = pv.Plotter(shape=(2, len(STEPS)),
                     window_size=(550 * len(STEPS), 1100),
                     border=False, off_screen=True)
plotter.set_background("white")

for row, (snap_dir, label) in enumerate(
        [(FE_DIR, "RK4 + FE trace-back (default)"),
         (RBF_DIR, "RK4 + RBF trace-back (fix)")]):
    for col, step in enumerate(STEPS):
        mesh, T = load_T(snap_dir, step)
        Tmin = float(T.data[:, 0].min())
        Tmax = float(T.data[:, 0].max())
        print(f"{label}  step {step}: T=[{Tmin:+.3f}, {Tmax:+.3f}]")

        pv_T = vis.meshVariable_to_pv_mesh_object(T)
        pv_T.point_data["T"] = np.asarray(T.data[:, 0])
        edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

        plotter.subplot(row, col)
        plotter.set_background("white")
        show_bar = (row == 0 and col == 0)
        plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                         clim=(0, 1), show_edges=False,
                         lighting=False,
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
            f"{label}\nstep {step}    T∈[{Tmin:+.2f}, {Tmax:+.2f}]",
            position="upper_edge", font_size=14, color="black")
        plotter.view_xy()

plotter.screenshot(OUT_PNG, transparent_background=False,
                   window_size=(550 * len(STEPS), 1100))
plotter.close()
print(f"wrote {OUT_PNG}")
