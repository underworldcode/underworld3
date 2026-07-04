"""Side-by-side: target ρ field on a fine reference mesh +
the adapted meshes overlaid on the SAME ρ background. Shows
what the mover is being asked to resolve, and what it
actually did."""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_compare')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import shape_field


# Build a fine reference mesh + ρ for the background everywhere.
mesh_ref = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
    cellSize=0.012, qdegree=3)
T_ref = uw.discretisation.MeshVariable(
    "T_target", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_ref.data[:, 0] = shape_field(np.asarray(T_ref.coords))
rho_expr = uw.meshing.metric_density_from_gradient(
    mesh_ref, T_ref, refinement=3.0, name="target_rho")
rho_view = uw.discretisation.MeshVariable(
    "rho_view", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
import underworld3.function as fn
rho_view.data[:, 0] = np.asarray(
    fn.evaluate(rho_expr, rho_view.coords)).reshape(-1)

# pv object for the target ρ background
pv_rho = vis.meshVariable_to_pv_mesh_object(rho_view)
pv_rho.point_data["rho"] = np.asarray(rho_view.data[:, 0])
rho_clim = (0.3, 9.0)


# Adapted mesh sources (existing dirs)
SE = "/Users/lmoresi/+Simulations/StagnantLid/synthetic_shapes_eulerian"
SS = "/Users/lmoresi/+Simulations/StagnantLid/synthetic_shapes"
ADAPTED = [
    ("uniform (no adapt)", f"{SS}/uniform_no_adapt"),
    ("aniso + Lag (frozen M)",
     f"{SE}/aniso__Lag_frozen_M"),
    ("aniso + Eul (frozen M)",
     f"{SE}/aniso__Eul_frozen_M"),
    ("aniso + Lag (refresh M)",
     f"{SE}/aniso__Lag_refresh_M"),
]


# 2x2 grid
ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")
for i, (label, src_dir) in enumerate(ADAPTED):
    row, col = i // ncols, i % ncols
    pl.subplot(row, col)
    pl.add_text(label, font_size=24, color='black')
    # Background: target ρ on the reference mesh (made
    # slightly translucent so mesh edges show through)
    pl.add_mesh(pv_rho, scalars="rho", cmap="viridis",
                clim=rho_clim, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.55)
    # Overlay the adapted mesh edges (or initial for uniform)
    if os.path.exists(src_dir):
        try:
            m_ad = uw.discretisation.Mesh(
                os.path.join(src_dir, "adapted.mesh.00000.h5"))
            edges_ad = vis.mesh_to_pv_mesh(m_ad).extract_all_edges()
            # Render the deformed mesh slightly raised above
            # the ρ plane so it sits ON TOP of the background.
            pts = np.asarray(edges_ad.points)
            pts[:, 2] = 0.1
            edges_ad.points = pts
            pl.add_mesh(edges_ad, color="black", line_width=2.5,
                        lighting=False, opacity=0.95)
        except Exception as e:
            print(f"  skip {label}: {e}")
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_compare.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
