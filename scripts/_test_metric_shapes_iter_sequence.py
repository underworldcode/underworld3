"""Per-iteration mesh sequence with TRULY EULERIAN metric.

Take the analytic ρ(x,y) (pure sympy, frozen in physical space)
and step the mover one outer iteration at a time, relax=1.0,
refresh=True. Save the mesh + the proposed full displacement
(scale=1 *before* backtrack) at every step.

Render the sequence with ρ as the background (NOT the
Lagrangian T) — so the shapes stay where they really are and we
can see whether the mesh is being pulled toward the metric or
oscillating around it.

Question being answered: with an Eulerian target metric, does
iter 2 *correct* iter 1 (move smaller, toward equidistribution),
or does it apply another full displacement on top?
"""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_iter_sequence')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho

# Iterations to capture (0 = undeformed)
ITERS_TO_CAPTURE = [0, 1, 2, 3, 4, 6, 8, 12]


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


# Re-use one mesh, step it one outer iter at a time, recording.
m = build_uniform_mesh()
rho_sym = analytic_rho(m)

# Sample ρ on a viz lattice (independent of the mesh) for the
# background. ρ is a sympy expression — evaluate analytically.
import sympy
X = m.CoordinateSystem.X
rho_lam = sympy.lambdify((X[0], X[1]), rho_sym, "numpy")
xv = np.linspace(-1.0, 1.0, 401)
yv = np.linspace(-1.0, 1.0, 401)
XX, YY = np.meshgrid(xv, yv)
RHO = rho_lam(XX, YY)
rho_clip = (1.0, float(RHO.max()))  # ρ ∈ [1, ~9]

captured = {}
captured[0] = np.asarray(m.X.coords).copy()
print(f"iter 0 (undeformed) captured")

for it in range(1, max(ITERS_TO_CAPTURE) + 1):
    coords_before = np.asarray(m.X.coords).copy()
    uw.meshing.smooth_mesh_interior(
        m, metric=rho_sym, method="anisotropic", strategy="med",
        method_kwargs=dict(n_outer=1, relax=1.0,
                            metric_refresh_per_iter=True),
        verbose=False)
    coords_after = np.asarray(m.X.coords).copy()
    dx = np.linalg.norm(coords_after - coords_before, axis=1)
    print(f"iter {it}: max|Δx|={dx.max():.3e}  mean|Δx|={dx.mean():.3e}")
    if it in ITERS_TO_CAPTURE:
        captured[it] = coords_after


# Render: 2x4 grid showing ρ background + adapted mesh at each iter
ncols, nrows = 4, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

# Build a uniform reference mesh to deform back to each captured
# state, just for visualisation purposes.
m_viz_template = build_uniform_mesh()

for panel_idx, it in enumerate(ITERS_TO_CAPTURE):
    row, col = panel_idx // ncols, panel_idx % ncols
    m_viz = build_uniform_mesh()
    m_viz._deform_mesh(captured[it])

    edges = vis.mesh_to_pv_mesh(m_viz).extract_all_edges()

    # ρ background — a structured grid sampled analytically
    bg = pv.ImageData(
        dimensions=(401, 401, 1),
        spacing=(2.0 / 400, 2.0 / 400, 1.0),
        origin=(-1.0, -1.0, 0.0))
    bg.point_data["rho"] = RHO.ravel(order="F")

    pl.subplot(row, col)
    pl.add_text(f"iter {it}", font_size=24, color='black')
    pl.add_mesh(bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False, opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_iter_sequence.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
