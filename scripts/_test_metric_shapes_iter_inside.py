"""Per-outer-iteration mesh snapshots from INSIDE ONE
smooth_mesh_interior call with truly Eulerian metric, refresh
and relax=1.0.

Why this matters: calling smooth_mesh_interior multiple times
with n_outer=1 each does NOT replay the inner outer-iter loop
faithfully (a separate cache bug — second call gives u=0). To
see what iter 2 actually does, we need to snapshot mesh state
inside the running outer loop.

We do that by patching mesh._deform_mesh to capture old/new
coords on every call.

Question being answered: with an Eulerian target metric, does
iter 2 *correct* iter 1 (small move toward target) or apply
another full displacement (overshoot)?
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
    '~/+Simulations/StagnantLid/synthetic_shapes_iter_inside')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


m = build_uniform_mesh()
rho_sym = analytic_rho(m)

# ---- ρ background lattice (independent of mesh, analytic) ---
X = m.CoordinateSystem.X
rho_lam = sympy.lambdify((X[0], X[1]), rho_sym, "numpy")
xv = np.linspace(-1.0, 1.0, 401)
yv = np.linspace(-1.0, 1.0, 401)
XX, YY = np.meshgrid(xv, yv)
RHO = rho_lam(XX, YY)

bg = pv.ImageData(
    dimensions=(401, 401, 1),
    spacing=(2.0 / 400, 2.0 / 400, 1.0),
    origin=(-1.0, -1.0, 0.0))
bg.point_data["rho"] = RHO.ravel(order="F")
rho_clip = (1.0, float(RHO.max()))

# ---- snapshot patch ----------------------------------------
captured = [np.asarray(m.X.coords).copy()]
orig_deform = m._deform_mesh


def deform_and_capture(new_coords, *args, **kwargs):
    res = orig_deform(new_coords, *args, **kwargs)
    captured.append(np.asarray(m.X.coords).copy())
    return res


m._deform_mesh = deform_and_capture

# ---- one call, n_outer=12 ----------------------------------
N_OUTER = 12
RELAX = 1.0
uw.meshing.smooth_mesh_interior(
    m, metric=rho_sym, method="anisotropic", strategy="med",
    method_kwargs=dict(n_outer=N_OUTER, relax=RELAX,
                       metric_refresh_per_iter=True),
    verbose=True)

m._deform_mesh = orig_deform

print(f"\ncaptured {len(captured)} mesh states "
      f"(iter 0 + {len(captured)-1} outer iters)")
for it, c in enumerate(captured):
    if it == 0:
        continue
    dx = np.linalg.norm(c - captured[it - 1], axis=1)
    cum = np.linalg.norm(c - captured[0], axis=1)
    print(f"  iter {it:2d}: this-step max={dx.max():.3e}  "
          f"cumulative max={cum.max():.3e}")

# ---- render ------------------------------------------------
panels = [0, 1, 2, 3, 4, 6, 8, 12]
panels = [p for p in panels if p < len(captured)]
ncols, nrows = 4, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for panel_idx, it in enumerate(panels):
    row, col = panel_idx // ncols, panel_idx % ncols
    m_viz = build_uniform_mesh()
    m_viz._deform_mesh(captured[it])
    edges = vis.mesh_to_pv_mesh(m_viz).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"iter {it}", font_size=24, color='black')
    pl.add_mesh(bg, scalars="rho", cmap="Blues", clim=rho_clip,
                show_edges=False, lighting=False,
                show_scalar_bar=False, opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_iter_inside.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
