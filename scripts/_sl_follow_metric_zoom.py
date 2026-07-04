"""Single-panel high-res render of the ref=3.0 follow_metric result —
the most-refined case, where slivers would show first."""
import os
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True

OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/follow_metric_compare')


def panel(label, out_png, window_size=(2200, 2200), zoom=1.25):
    d = label.replace(" ", "_").replace(",", "").replace(
        "=", "").replace("(", "").replace(")", "")
    out_dir = os.path.join(OUT, d)
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_view", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out_dir)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl = pv.Plotter(off_screen=True, window_size=window_size,
                    border=False)
    pl.set_background("white")
    pl.add_text(label, font_size=30, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.2,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(zoom)
    pl.screenshot(out_png)
    pl.close()
    print(f"wrote {out_png}")


# ref=3.0 — densest BL refinement
panel("ref=3.0, coar=auto, FF",
      os.path.join(OUT, "plot_ref3_singlepanel.png"))
# also a zoomed-in view to see cell-level detail at the BL
panel("ref=3.0, coar=auto, FF",
      os.path.join(OUT, "plot_ref3_zoom_BL.png"),
      window_size=(2200, 2200), zoom=2.5)
