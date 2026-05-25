"""OT improvement step (method='ot'): each call applies one
weighted-Poisson equidistribution flow step. Composable — the
input mesh has no special status. Test by chaining multiple
calls and watching the imbalance ratio drop.

4 panels:
  0. undeformed
  1. after 1 OT step
  3. after 3 OT steps
  8. after 8 OT steps
"""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_OT')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


m = build_uniform_mesh()
rho_sym = analytic_rho(m)

# ρ background (analytic, on a separate undeformed reference)
m_bg = build_uniform_mesh()
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
    analytic_rho(m_bg), np.asarray(T_bg.coords))).reshape(-1)
rho_clip = (1.0, float(T_bg.data[:, 0].max()))
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])

# capture coords across calls
captured = [np.asarray(m.X.coords).copy()]
STEPS_TO_CAPTURE = {0, 1, 5, 15, 40}

N_STEPS = 40
print("OT-improve sweep — each call is one improvement step:")
imb_traj = []
for k in range(1, N_STEPS + 1):
    uw.meshing.smooth_mesh_interior(
        m, metric=rho_sym, method="ot", verbose=True,
        boundary_slip="box",
        method_kwargs=dict(n_outer=1, relax=0.1))
    if k in STEPS_TO_CAPTURE:
        captured.append(np.asarray(m.X.coords).copy())

# Render: undeformed + each captured step
sorted_steps = sorted(STEPS_TO_CAPTURE)
ncols = min(len(sorted_steps), 4)
nrows = (len(sorted_steps) + ncols - 1) // ncols
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for panel_idx, step in enumerate(sorted_steps):
    row, col = panel_idx // ncols, panel_idx % ncols
    m_viz = build_uniform_mesh()
    cap_idx = sorted_steps.index(step)
    m_viz._deform_mesh(captured[cap_idx])
    edges = vis.mesh_to_pv_mesh(m_viz).extract_all_edges()
    pl.subplot(row, col)
    label = "undeformed" if step == 0 else f"after {step} OT step(s)"
    pl.add_text(label, font_size=24, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
