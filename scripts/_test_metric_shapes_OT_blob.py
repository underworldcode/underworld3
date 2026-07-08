"""Multi-resolution OT with a coarse 'FILLED BLOB' metric:

  ρ_blob(x) = 1 + AMP · 0.5 · (1 + tanh(d_shape(x) / EPS_big))

This is a smoothed shape indicator: ρ ≈ 1+AMP INSIDE the
shape, ≈ 1 OUTSIDE, with smooth transition. Mass-transport
OT under this metric pulls nodes INTO each shape — the
long-range transport effect we want.

Then run the sharp boundary-band metric (the original
sech² ρ) to migrate those interior nodes to the boundary.

Panels:
  A. raw sharp OT × 10 (reference — no multi-res)
  B. blob OT × 5 only — see if nodes flow INTO shapes
  C. (B's result) + sharp OT × 5
  D. blob × 3, sharp × 7 (more sharp)
  E. blob × 7, sharp × 3 (more blob)
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
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_blob')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import (
    sym_smax, sym_smin, sym_smin3, sym_sabs, sym_sech2,
    AMP)
from _test_metric_shapes_OT_multires import (
    analytic_rho_eps, build_uniform_mesh)


def analytic_rho_blob(mesh, eps_smooth):
    """ρ = 1 + AMP * 0.5 * (1 + tanh(d / eps_smooth)) per shape.
    Filled-blob indicator. Use a smaller eps_smooth for sharper
    blob edges; large eps_smooth blurs the blob outline.
    """
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
        rho = rho + AMP * sympy.Rational(1, 2) * (
            1 + sympy.tanh(d / eps_smooth))
    return rho


def build_bg(rho_fn, eps_or_kw):
    m_bg = build_uniform_mesh()
    T_bg = uw.discretisation.MeshVariable(
        f"T_bg_{id(rho_fn)}_{int(eps_or_kw*1000)}",
        m_bg, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
        rho_fn(m_bg, eps_or_kw),
        np.asarray(T_bg.coords))).reshape(-1)
    clip = (1.0, float(T_bg.data[:, 0].max()))
    pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
    pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])
    return pv_bg, clip


# Two backgrounds for context — sharp (true target) and blob.
pv_bg_sharp, rho_clip_sharp = build_bg(analytic_rho_eps, 0.04)
pv_bg_blob, rho_clip_blob = build_bg(analytic_rho_blob, 0.06)


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
    """Imbalance of the current mesh against the sharp ρ."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=analytic_rho_eps(mesh, eps=0.04),
            method="ot", verbose=True, boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=0.0,
                                step_frac=0.3))
    m = re.search(r"imb=([0-9.e+-]+)", buf.getvalue())
    return float(m.group(1)) if m else float("nan")


def run_schedule(steps):
    """steps: list of ('blob'|'sharp', eps, n)"""
    m = build_uniform_mesh()
    traj_blob, traj_sharp = [], []
    for kind, eps, n in steps:
        if kind == "blob":
            rho = analytic_rho_blob(m, eps)
        else:
            rho = analytic_rho_eps(m, eps)
        for _ in range(n):
            ret = step_OT(m, rho)
            if ret is not None:
                if kind == "blob":
                    traj_blob.append(ret)
                else:
                    traj_sharp.append(ret)
    return m, traj_blob, traj_sharp


RECIPES = [
    ("A. sharp OT x 10  (reference)",
     [("sharp", 0.04, 10)],
     "sharp"),
    ("B. blob OT x 5    (does it pull nodes IN?)",
     [("blob", 0.06, 5)],
     "blob"),
    ("C. blob x 5 + sharp x 5",
     [("blob", 0.06, 5), ("sharp", 0.04, 5)],
     "sharp"),
    ("D. blob x 3 + sharp x 7",
     [("blob", 0.06, 3), ("sharp", 0.04, 7)],
     "sharp"),
    ("E. blob x 7 + sharp x 3",
     [("blob", 0.06, 7), ("sharp", 0.04, 3)],
     "sharp"),
    ("F. blob x 5 alone (different background)",
     [("blob", 0.06, 5)],
     "blob_alt"),
]


results = {}
for label, schedule, bg_kind in RECIPES:
    print(f"\n=== {label} ===")
    m, tb, ts = run_schedule(schedule)
    sharp_imb = measure_sharp_imb(m)
    print(f"  blob-imb traj : "
          f"{' '.join(f'{v:.3f}' for v in tb)}")
    print(f"  sharp-imb traj: "
          f"{' '.join(f'{v:.3f}' for v in ts)}")
    print(f"  FINAL sharp imb: {sharp_imb:.3f}")
    results[label] = (m, sharp_imb, bg_kind)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _, bg_kind) in enumerate(RECIPES):
    row, col = i // ncols, i % ncols
    m, sharp_imb, _ = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nsharp imb={sharp_imb:.3f}",
                font_size=18, color='black')
    if bg_kind == "blob":
        pl.add_mesh(pv_bg_blob, scalars="rho", cmap="Blues",
                    clim=rho_clip_blob, show_edges=False,
                    lighting=False, show_scalar_bar=False,
                    opacity=0.85)
    else:
        pl.add_mesh(pv_bg_sharp, scalars="rho", cmap="Blues",
                    clim=rho_clip_sharp, show_edges=False,
                    lighting=False, show_scalar_bar=False,
                    opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_blob.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
