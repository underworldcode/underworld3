"""Synthetic-shapes test using the VINTAGE meshing path —
no spring, no polish, no h0 cache, no sliver-floor extras.
Pure `metric_density_from_gradient` + `smooth_mesh_interior`
with the OLD strategy='med' API. Used to isolate whether the
shape asymmetries (doughnut inner-vs-outer, triangle under-
resolved) come from the underlying mover, OR from the
scaffolding we added on top.
"""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.meshing import smoothing as _sm
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_vintage')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import build_mesh_with_field

# Sweep strategy and rho approach so we can see the family of
# legacy outputs. Each entry: (label, strategy, kwargs to
# metric_density_from_gradient, kwargs to smooth_mesh_interior)
CASES = [
    # 1) Pure strategy='med' default (R=1.4, mild)
    ("strategy=med (legacy default)",
     dict(mdg=dict(strategy="med"),
          smi=dict(method="anisotropic", strategy="med",
                   method_kwargs=dict(relax=0.2, n_outer=12)))),
    # 2) strategy='high' (R=1.5, more refinement)
    ("strategy=high",
     dict(mdg=dict(strategy="high"),
          smi=dict(method="anisotropic", strategy="high",
                   method_kwargs=dict(relax=0.2, n_outer=12)))),
    # 3) strategy='extreme' (R=2, power=1.5)
    ("strategy=extreme",
     dict(mdg=dict(strategy="extreme"),
          smi=dict(method="anisotropic", strategy="extreme",
                   method_kwargs=dict(relax=0.2, n_outer=12)))),
    # 4) explicit dials: amp=8, power=2 (gradient-uniform),
    #    R=1.5 — controlled mid-strength case
    ("explicit amp=8 power=2 R=1.5",
     dict(mdg=dict(strategy="med", amp=8.0, power=2.0,
                   lo_percentile=50.0, hi_percentile=97.0),
          smi=dict(method="anisotropic",
                   method_kwargs=dict(relax=0.2, n_outer=12,
                                       resolution_ratio=1.5)))),
]


def adapt_one(label, kw):
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace("(", "").replace(")", "").replace(".", "p"))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(
            os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: cached")
        return out_dir
    print(f"{label}: adapting")
    m, T = build_mesh_with_field()
    rho = uw.meshing.metric_density_from_gradient(
        m, T, name=f"vintage_{label}", **kw["mdg"])
    uw.meshing.smooth_mesh_interior(
        m, metric=rho, verbose=False, **kw["smi"])
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)
    return out_dir


for label, kw in CASES:
    adapt_one(label, kw)


ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace("(", "").replace(")", "").replace(".", "p"))
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
    pl.add_text(label, font_size=22, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="Blues",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_vintage_sweep.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
