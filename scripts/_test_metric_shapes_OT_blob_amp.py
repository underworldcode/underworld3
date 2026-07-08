"""Coarse-blob OT — amplification sweep.

Tests whether OT under a filled-blob ρ does meaningful long-range
node transport into shape interiors, and whether amplifying the
blob makes OT pull harder. Every panel = 5 OT steps from the
uniform mesh, rendered against the SHARP ρ (so you can judge
whether the nodes ended up in the right places).
"""
import os
import sys
import io
import re
import contextlib
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_blob_amp')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
# Only import from the GUARDED analytic_disp module. The
# multires script has unguarded top-level work — avoid that
# import chain.
from _test_metric_shapes_analytic_disp import (
    sym_smax, sym_smin, sym_smin3, sym_sabs, sym_sech2, AMP)


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


def analytic_rho_eps(mesh, eps):
    """Sech²-band ρ with parametrised band width (sharp ρ)."""
    X = mesh.CoordinateSystem.X
    x, y = X[0], X[1]
    cx_sq, cy_sq, side = 0.55, 0.35, 0.4
    ang_rad = 30.0 * np.pi / 180.0
    ct, st = float(np.cos(ang_rad)), float(np.sin(ang_rad))
    dxs, dys = x - cx_sq, y - cy_sq
    xp = ct * dxs + st * dys
    yp = -st * dxs + ct * dys
    d_sq = side / 2 - sym_smax(sym_sabs(xp), sym_sabs(yp))
    cx_dh, cy_dh, r_in, r_out = -0.55, 0.45, 0.15, 0.30
    r = sympy.sqrt((x - cx_dh) ** 2 + (y - cy_dh) ** 2)
    d_dh = sym_smin(r - r_in, r_out - r)
    v0 = (sympy.Float(0.05), sympy.Float(-0.65))
    v1 = (sympy.Float(0.55), sympy.Float(-0.35))
    v2 = (sympy.Float(-0.30), sympy.Float(-0.30))

    def half_plane(a, b):
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = -ey, ex
        nl = sympy.sqrt(nx * nx + ny * ny)
        return ((x - a[0]) * nx + (y - a[1]) * ny) / nl

    d_tr = sym_smin3(
        half_plane(v0, v1), half_plane(v1, v2),
        half_plane(v2, v0))
    rho = sympy.Integer(1)
    for d in (d_sq, d_dh, d_tr):
        rho = rho + AMP * sym_sech2(d / eps)
    return rho


def shape_distances(mesh):
    """Return (d_sq, d_dh, d_tr): signed-distance sympy fields,
    POSITIVE inside each shape, smooth at the boundary."""
    X = mesh.CoordinateSystem.X
    x, y = X[0], X[1]
    cx_sq, cy_sq, side = 0.55, 0.35, 0.4
    ang_rad = 30.0 * np.pi / 180.0
    ct, st = float(np.cos(ang_rad)), float(np.sin(ang_rad))
    dxs, dys = x - cx_sq, y - cy_sq
    xp = ct * dxs + st * dys
    yp = -st * dxs + ct * dys
    d_sq = side / 2 - sym_smax(sym_sabs(xp), sym_sabs(yp))
    cx_dh, cy_dh, r_in, r_out = -0.55, 0.45, 0.15, 0.30
    r = sympy.sqrt((x - cx_dh) ** 2 + (y - cy_dh) ** 2)
    d_dh = sym_smin(r - r_in, r_out - r)
    v0 = (sympy.Float(0.05), sympy.Float(-0.65))
    v1 = (sympy.Float(0.55), sympy.Float(-0.35))
    v2 = (sympy.Float(-0.30), sympy.Float(-0.30))

    def half_plane(a, b):
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = -ey, ex
        nl = sympy.sqrt(nx * nx + ny * ny)
        return ((x - a[0]) * nx + (y - a[1]) * ny) / nl

    d_tr = sym_smin3(
        half_plane(v0, v1), half_plane(v1, v2),
        half_plane(v2, v0))
    return d_sq, d_dh, d_tr


def rho_blob_tanh(mesh, amp, eps_smooth):
    """ρ = 1 + amp · 0.5 · (1 + tanh(d / eps))   per shape."""
    d_sq, d_dh, d_tr = shape_distances(mesh)
    rho = sympy.Integer(1)
    for d in (d_sq, d_dh, d_tr):
        rho = rho + amp * sympy.Rational(1, 2) * (
            1 + sympy.tanh(d / eps_smooth))
    return rho


def rho_blob_gauss(mesh, amp, sigma):
    """ρ = 1 + amp · exp(-(d_centroid)^2 / sigma^2) per shape —
    radial blob centred on each shape's CENTROID (not distance
    to boundary). Ignores shape geometry; pure 2-D Gaussian."""
    X = mesh.CoordinateSystem.X
    x, y = X[0], X[1]
    centres = [(0.55, 0.35),     # rotated square centre
               (-0.55, 0.45),    # doughnut centre
               (0.10, -0.43)]    # triangle centroid
    rho = sympy.Integer(1)
    for cx, cy in centres:
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        rho = rho + amp * sympy.exp(-r2 / sigma ** 2)
    return rho


def build_bg(rho_fn, *args, label=""):
    m_bg = build_uniform_mesh()
    T_bg = uw.discretisation.MeshVariable(
        f"T_bg_{label}", m_bg, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
        rho_fn(m_bg, *args),
        np.asarray(T_bg.coords))).reshape(-1)
    clip = (1.0, float(T_bg.data[:, 0].max()))
    pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
    pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])
    return pv_bg, clip


# SHARP background for ALL panels (so we judge where nodes land)
pv_bg_sharp, clip_sharp = build_bg(
    analytic_rho_eps, 0.04, label="sharp")


def step_OT(mesh, rho, relax=0.1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ot", verbose=True,
            boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=relax,
                                step_frac=0.3))
    m = re.search(r"imb=([0-9.e+-]+)", buf.getvalue())
    return float(m.group(1)) if m else None


def measure_sharp_imb(mesh):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=analytic_rho_eps(mesh, eps=0.04),
            method="ot", verbose=True, boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=0.0,
                                step_frac=0.3))
    m = re.search(r"imb=([0-9.e+-]+)", buf.getvalue())
    return float(m.group(1)) if m else float("nan")


def run5(rho_fn, *args):
    m = build_uniform_mesh()
    rho = rho_fn(m, *args)
    for _ in range(5):
        step_OT(m, rho)
    return m, measure_sharp_imb(m)


# Six strategies, all 5 OT steps from uniform.
def make_rho_eps(*a):
    return lambda m: analytic_rho_eps(m, *a)


RECIPES = [
    ("A. sharp boundary band  AMP=8 (reference)",
     lambda: run5(analytic_rho_eps, 0.04)),
    ("B. tanh blob  AMP=8, eps=0.06",
     lambda: run5(rho_blob_tanh, 8.0, 0.06)),
    ("C. tanh blob  AMP=20, eps=0.06  (amplified)",
     lambda: run5(rho_blob_tanh, 20.0, 0.06)),
    ("D. tanh blob  AMP=50, eps=0.06  (HUGE amp)",
     lambda: run5(rho_blob_tanh, 50.0, 0.06)),
    ("E. Gaussian blob  AMP=8, sigma=0.18  (centred)",
     lambda: run5(rho_blob_gauss, 8.0, 0.18)),
    ("F. Gaussian blob  AMP=20, sigma=0.25",
     lambda: run5(rho_blob_gauss, 20.0, 0.25)),
]

results = {}
import time
for label, runfn in RECIPES:
    t0 = time.time()
    print(f"\n=== {label} ===")
    m, sharp_imb = runfn()
    print(f"  sharp imb={sharp_imb:.3f}  "
          f"({time.time()-t0:.1f}s)")
    results[label] = (m, sharp_imb)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(RECIPES):
    row, col = i // ncols, i % ncols
    m, sharp_imb = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nsharp imb={sharp_imb:.3f}",
                font_size=18, color='black')
    pl.add_mesh(pv_bg_sharp, scalars="rho", cmap="Blues",
                clim=clip_sharp, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_blob_amp.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
