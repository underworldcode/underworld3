"""MA convergence sweep: vary n_picard and n_outer with target-side
ρ + box slip. Probe whether the residual phase mismatch + bulk
under-redistribution come from incomplete Picard convergence
(more n_picard) or insufficient outer composition (more n_outer).

Panels:
  A. 1 outer × 25 picard  (current best)
  B. 1 outer × 100 picard (more Picard convergence)
  C. 3 outer × 50 picard  (more outer composition)
  D. 3 outer × 100 picard (both)
"""
import os
import sys
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_MA_sweep')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


CASES = [
    ("A. 1 outer x 25 picard",   1,  25),
    ("B. 1 outer x 100 picard",  1, 100),
    ("C. 3 outer x 50 picard",   3,  50),
    ("D. 3 outer x 100 picard",  3, 100),
]


# ρ background (same physical position for every panel).
m_bg = build_uniform_mesh()
rho_bg_sym = analytic_rho(m_bg)
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
    rho_bg_sym, np.asarray(T_bg.coords))).reshape(-1)
rho_max = float(T_bg.data[:, 0].max())
rho_clip = (1.0, rho_max)
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])


for label, n_outer, n_picard in CASES:
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace(".", "p"))
    if os.path.exists(os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: cached")
        continue
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {label} ===")
    m = build_uniform_mesh()
    rho = analytic_rho(m)
    try:
        uw.meshing.smooth_mesh_interior(
            m, metric=rho, method="ma", verbose=True,
            boundary_slip="box",
            method_kwargs=dict(n_outer=n_outer, n_picard=n_picard,
                               relax=1.0, target_side_rho=True))
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        continue
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[],
                     meshUpdates=True, create_xdmf=True)


ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, n_outer, n_picard) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace(".", "p"))
    mesh_path = os.path.join(out_dir, "adapted.mesh.00000.h5")
    if not os.path.exists(mesh_path):
        pl.subplot(row, col)
        pl.add_text(f"{label}\n(failed)", font_size=20,
                    color='red')
        continue
    m_viz = uw.discretisation.Mesh(mesh_path)
    edges = vis.mesh_to_pv_mesh(m_viz).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(label, font_size=22, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_MA_sweep.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
