"""Visual confirmation: 2×2 panel mesh comparison testing the
Lagrangian-vs-Eulerian metric hypothesis.

Panels:
  (A) Lagrangian ρ, default 12×0.2   — current production
  (B) Lagrangian ρ, 1×1.0            — user's "clean" reference
  (C) Analytic Eulerian ρ + refresh, 12×0.2  — truly Eulerian
  (D) Analytic Eulerian ρ + refresh, 1×1.0

If the Eulerian-D hypothesis were correct, panel (C) should
look CLEANER than (A). Per the disp trajectory in
_test_metric_shapes_analytic_disp.py, max|Δx| at 12×0.2 GROWS
with Eulerian refresh (positive feedback) — so we expect (C)
to look WORSE than (A).
"""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_analytic_iter2')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import build_mesh_with_field, shape_field
from _test_metric_shapes_analytic_disp import analytic_rho

CASES = [
    ("A. Lagrangian rho, 12 x 0.2 (default)",
     "lagrangian", dict(n_outer=12, relax=0.2,
                         metric_refresh_per_iter=False)),
    ("B. Lagrangian rho, 1 x 1.0 (clean ref)",
     "lagrangian", dict(n_outer=1, relax=1.0,
                         metric_refresh_per_iter=False)),
    ("C. Analytic Eul rho + refresh, 12 x 0.2",
     "analytic",   dict(n_outer=12, relax=0.2,
                         metric_refresh_per_iter=True)),
    ("D. Analytic Eul rho + refresh, 1 x 1.0",
     "analytic",   dict(n_outer=1, relax=1.0,
                         metric_refresh_per_iter=True)),
]


for label, kind, kw in CASES:
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace(",", "")
              .replace(".", "p"))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: cached")
        continue
    print(f"{label}: adapting")
    m, T = build_mesh_with_field()
    if kind == "lagrangian":
        rho = uw.meshing.metric_density_from_gradient(
            m, T, refinement=3.0, name=label.replace(" ", "_"))
    else:
        rho = analytic_rho(m)
    uw.meshing.smooth_mesh_interior(
        m, metric=rho, method="anisotropic", strategy="med",
        method_kwargs=kw, verbose=False)
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)


ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")

for i, (label, kind, kw) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace(",", "")
              .replace(".", "p"))
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    # Make a degree-3 viz var on the adapted mesh, fill it with
    # the ANALYTIC indicator field evaluated at the deformed
    # DOF positions. This shows the shapes at their TRUE physical
    # positions, with the adapted mesh overlaid — the only honest
    # way to see whether the mesh tracks the features.
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.data[:, 0] = shape_field(np.asarray(T.coords))
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(label, font_size=20, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="Blues",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_analytic_iter2.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
