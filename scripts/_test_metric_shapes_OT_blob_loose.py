"""Blob OT with the brakes off: larger relax and step_frac so
the OT can actually transport nodes long distances into the
shape interiors. Previous run was strangled by step_frac=0.3
and relax=0.1 (sharp-metric tuning) — total movement budget
was 0.006 over 5 steps, vs shapes ~0.5 wide.
"""
import os, sys, io, re, contextlib, time
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_blob_loose')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import (
    sym_smax, sym_smin, sym_smin3, sym_sabs, sym_sech2, AMP)


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


def shape_distances(mesh):
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


def rho_blob_tanh(mesh, amp, eps):
    d_sq, d_dh, d_tr = shape_distances(mesh)
    rho = sympy.Integer(1)
    for d in (d_sq, d_dh, d_tr):
        rho = rho + amp * sympy.Rational(1, 2) * (
            1 + sympy.tanh(d / eps))
    return rho


def rho_sharp(mesh, eps=0.04):
    d_sq, d_dh, d_tr = shape_distances(mesh)
    rho = sympy.Integer(1)
    for d in (d_sq, d_dh, d_tr):
        rho = rho + AMP * sym_sech2(d / eps)
    return rho


# Background = SHARP ρ (always)
m_bg = build_uniform_mesh()
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
    rho_sharp(m_bg), np.asarray(T_bg.coords))).reshape(-1)
clip = (1.0, float(T_bg.data[:, 0].max()))
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])


def step_OT(mesh, rho, relax, step_frac):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ot", verbose=True,
            boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=relax,
                                step_frac=step_frac))
    out = buf.getvalue()
    imb_m = re.search(r"imb=([0-9.e+-]+)", out)
    scl_m = re.search(r"scale=([0-9.e+-]+)", out)
    dx_m = re.search(r"max\|Δx\|=([0-9.e+-]+)", out)
    return (float(imb_m.group(1)) if imb_m else None,
            float(scl_m.group(1)) if scl_m else None,
            float(dx_m.group(1)) if dx_m else None)


def run(rho_fn, relax, step_frac, n=5):
    m = build_uniform_mesh()
    rho = rho_fn(m)
    history = []
    for k in range(n):
        imb, scl, dx = step_OT(m, rho, relax, step_frac)
        history.append((imb, scl, dx))
    # measure final imb against sharp ρ
    rho_s = rho_sharp(m)
    imb_s, _, _ = step_OT(m, rho_s, 0.0, 0.3)
    return m, history, imb_s


RECIPES = [
    # (label, rho factory, relax, step_frac)
    ("A. sharp, relax=0.1, sf=0.3  (baseline)",
     lambda m: rho_sharp(m), 0.1, 0.3),
    ("B. blob AMP=8 eps=0.06, relax=0.1, sf=0.3",
     lambda m: rho_blob_tanh(m, 8.0, 0.06), 0.1, 0.3),
    ("C. blob AMP=8 eps=0.06, relax=1.0, sf=1.0",
     lambda m: rho_blob_tanh(m, 8.0, 0.06), 1.0, 1.0),
    ("D. blob AMP=20 eps=0.06, relax=1.0, sf=1.0",
     lambda m: rho_blob_tanh(m, 20.0, 0.06), 1.0, 1.0),
    ("E. blob AMP=20 eps=0.06, relax=0.5, sf=0.6",
     lambda m: rho_blob_tanh(m, 20.0, 0.06), 0.5, 0.6),
    ("F. blob AMP=20 eps=0.15 (wide), relax=1.0, sf=1.0",
     lambda m: rho_blob_tanh(m, 20.0, 0.15), 1.0, 1.0),
]

results = {}
for label, rho_fn, relax, sf in RECIPES:
    t0 = time.time()
    print(f"\n=== {label} ===")
    m, hist, imb_s = run(rho_fn, relax, sf, n=5)
    # Print per-step diagnostic — see whether scale was capped
    # by the backtrack, and how big the actual move was.
    for k, (imb, scl, dx) in enumerate(hist):
        print(f"  step {k+1}: imb={imb}  scale={scl}  "
              f"max|Δx|={dx:.3e}" if dx else f"  step {k+1}: --")
    print(f"  final sharp imb={imb_s:.3f}  "
          f"({time.time()-t0:.1f}s)")
    results[label] = (m, imb_s)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, *_) in enumerate(RECIPES):
    row, col = i // ncols, i % ncols
    m, imb_s = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nsharp imb={imb_s:.3f}",
                font_size=18, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_blob_loose.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
