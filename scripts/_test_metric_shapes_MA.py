"""Smoke-test: drive the existing _winslow_elliptic (BFO
convex-branch Picard MA solver) on the synthetic shapes
with the analytic Eulerian ρ.

If the existing MA implementation just works on this box
geometry, we have our OT comparison without writing new
code. Per the pivot memory, the prior failure modes were
on Annulus + re-solve. Single-shot (n_outer=1) on a box
may be fine.
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
    '~/+Simulations/StagnantLid/synthetic_shapes_MA')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


m = build_uniform_mesh()
rho_sym = analytic_rho(m)

# ρ background: render via a high-degree MeshVariable on a
# separate UNDEFORMED reference mesh (uses the same pyvista
# path as the T plots). Fixed in physical space — does not
# move with the deforming mesh.
m_bg = build_uniform_mesh()
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(
    uw.function.evaluate(rho_sym, np.asarray(T_bg.coords))
).reshape(-1)
rho_max = float(T_bg.data[:, 0].max())
rho_clip = (1.0, rho_max)
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])

# Capture mesh states across the MA Picard outer iters.
captured = [np.asarray(m.X.coords).copy()]
orig_deform = m._deform_mesh


def deform_and_capture(new_coords, *args, **kwargs):
    res = orig_deform(new_coords, *args, **kwargs)
    captured.append(np.asarray(m.X.coords).copy())
    return res


m._deform_mesh = deform_and_capture

print("Running MA (target-side ρ + box slip) on shapes...")
try:
    uw.meshing.smooth_mesh_interior(
        m, metric=rho_sym, method="ma", verbose=True,
        boundary_slip="box",
        method_kwargs=dict(n_outer=1, n_picard=25, relax=1.0,
                            target_side_rho=True))
except Exception as e:
    print(f"\nMA SOLVE FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
m._deform_mesh = orig_deform

print(f"\ncaptured {len(captured)} mesh states")
for it, c in enumerate(captured):
    if it == 0:
        continue
    dx = np.linalg.norm(c - captured[it - 1], axis=1)
    cum = np.linalg.norm(c - captured[0], axis=1)
    print(f"  outer {it}: this-step max={dx.max():.3e}  "
          f"cumulative max={cum.max():.3e}")

# Render: undeformed vs MA result
panels = [(0, "iter 0 (undeformed)")] + [
    (i, f"after MA outer {i}") for i in range(1, len(captured))]
if len(panels) <= 1:
    print("WARN: MA didn't deform the mesh — nothing to render")
ncols = min(len(panels), 4)
nrows = (len(panels) + ncols - 1) // ncols if panels else 1
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for panel_idx, (it, label) in enumerate(panels):
    row, col = panel_idx // ncols, panel_idx % ncols
    m_viz = build_uniform_mesh()
    m_viz._deform_mesh(captured[it])
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

out_png = os.path.join(OUT, "plot_MA.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
