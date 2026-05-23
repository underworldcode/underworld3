"""Multi-resolution OT (ρ-widening homotopy) on synthetic shapes.

Idea: start with a SMOOTHED ρ (broad sech² bumps, large EPS)
so the OT source has long support → nodes can transport across
the domain in early steps. Progressively narrow EPS until we
reach the true sharp ρ — annealing-style multi-scale OT.

Compares against raw OT × N at matched compute. The
`analytic_rho_eps` parametrised builder takes an EPS so we can
choose the bump width per level.
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
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_multires')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import (
    sym_smax, sym_smin, sym_smin3, sym_sabs, sym_sech2,
    AMP)


def analytic_rho_eps(mesh, eps):
    """Same shapes as analytic_rho but with parametrised band
    width EPS — wide EPS = smoothed metric for multi-res."""
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


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


# ρ background — true (sharp) ρ for visualisation
m_bg = build_uniform_mesh()
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
    analytic_rho_eps(m_bg, eps=0.04),
    np.asarray(T_bg.coords))).reshape(-1)
rho_clip = (1.0, float(T_bg.data[:, 0].max()))
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])


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


# Multi-res schedules: (eps, n_OT_at_this_level)
SCHEDULES = [
    ("A. raw OT x 10 (EPS=0.04)",
     [(0.04, 10)]),
    ("B. 2-level: EPS=0.16 x 5, EPS=0.04 x 5",
     [(0.16, 5), (0.04, 5)]),
    ("C. 3-level: EPS=0.16,0.08,0.04 x (3,3,4)",
     [(0.16, 3), (0.08, 3), (0.04, 4)]),
    ("D. 3-level wide: EPS=0.32,0.10,0.04 x (3,3,4)",
     [(0.32, 3), (0.10, 3), (0.04, 4)]),
    ("E. 4-level: EPS=0.32,0.16,0.08,0.04 x (2,2,2,4)",
     [(0.32, 2), (0.16, 2), (0.08, 2), (0.04, 4)]),
    ("F. 5-level: EPS=0.48,0.24,0.12,0.06,0.04 x (2,2,2,2,2)",
     [(0.48, 2), (0.24, 2), (0.12, 2), (0.06, 2), (0.04, 2)]),
]


results = {}
for label, schedule in SCHEDULES:
    n_ot = sum(n for _, n in schedule)
    print(f"\n=== {label} ({n_ot} OT total) ===")
    m = build_uniform_mesh()
    imb_traj = []
    for eps, n_steps in schedule:
        rho_lvl = analytic_rho_eps(m, eps)
        for k in range(n_steps):
            ret = step_OT(m, rho_lvl, relax=0.1)
            if ret is not None:
                imb_traj.append(ret)
    # Final imbalance is measured against the SHARP ρ — that's
    # what we actually care about. Do one zero-step measurement
    # using the true ρ.
    rho_sharp = analytic_rho_eps(m, eps=0.04)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            m, metric=rho_sharp, method="ot", verbose=True,
            boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=0.0,
                                step_frac=0.3))
    msr = re.search(r"imb=([0-9.e+-]+)", buf.getvalue())
    sharp_final = float(msr.group(1)) if msr else float("nan")
    print(f"  level-eps trajectory:")
    cursor = 0
    for eps, n_steps in schedule:
        segment = imb_traj[cursor:cursor + n_steps]
        cursor += n_steps
        print(f"    EPS={eps}: "
              f"{' '.join(f'{v:.3f}' for v in segment)}")
    print(f"  FINAL imb vs SHARP ρ (EPS=0.04): "
          f"{sharp_final:.3f}")
    results[label] = (m, imb_traj, sharp_final, n_ot)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(SCHEDULES):
    row, col = i // ncols, i % ncols
    m, imb, sharp_final, n_ot = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nsharp imb={sharp_final:.3f}  "
                f"({n_ot} OT)",
                font_size=18, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_multires.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
