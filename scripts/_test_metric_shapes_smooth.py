"""Shapes test sweeping gradient_smoothing_length.

Hypothesis: with no smoothing, the metric is razor-sharp at the
shape boundaries; the mover then puts cells right at the edges
but with no neighbour-sharing across features it produces those
"corridor" artifacts. A wider gradient_smoothing_length spreads
the metric into a halo around each shape, so the mover sees a
locally-supported demand rather than a delta-function demand.

Fixed: refinement=3, no skip, default polish.
Swept: gradient_smoothing_length ∈ {None, 1·h0, 2·h0, 4·h0}.
"""
import os
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_smooth')
os.makedirs(OUT, exist_ok=True)

# Re-use the shape definitions from the earlier script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import (build_mesh_with_field,)


CELLSIZE = 0.04
H0 = CELLSIZE   # nominal mean edge for the smoothing-length scale

CASES = [
    ("L = 0 (no smoothing)",  None),
    ("L = 1.h0",              1.0 * H0),
    ("L = 2.h0",              2.0 * H0),
    ("L = 4.h0",              4.0 * H0),
]


for label, gL in CASES:
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace(".", "p").replace("(", "")
              .replace(")", ""))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: already adapted")
        continue
    print(f"{label}: gradient_smoothing_length={gL}")
    m, T = build_mesh_with_field()
    old_X = np.asarray(m.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    moved = uw.meshing.follow_metric(
        m, T, refinement=3.0,
        gradient_smoothing_length=gL,
        skip_threshold=None,
        verbose=False)
    if moved:
        new_X = np.asarray(m.X.coords).copy()
        new_Tx = np.asarray(T.coords).copy()
        m._deform_mesh(old_X)
        T.data[...] = old_T
        rT = np.asarray(uw.function.evaluate(
            T.sym[0], new_Tx)).reshape(-1)
        m._deform_mesh(new_X)
        T.data[:, 0] = rT
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)


# Render 2x2
ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")

for i, (label, gL) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace(".", "p").replace("(", "")
              .replace(")", ""))
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_shapes", 0, outputPath=out_dir)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"ref=3, {label}", font_size=26, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="Blues",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_smooth_sweep.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
