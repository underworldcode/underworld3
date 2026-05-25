"""Refined OT-interleave sweep: always end on OT; vary the
intermediate heuristic and its tuning. Apples-to-apples on
"how many OT calls" (5 OT each — same compute as the OT baseline).
"""
import os
import sys
import io
import re
import contextlib
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_interleave2')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


m_bg = build_uniform_mesh()
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
    analytic_rho(m_bg), np.asarray(T_bg.coords))).reshape(-1)
rho_clip = (1.0, float(T_bg.data[:, 0].max()))
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])


def step_OT(mesh, rho):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ot", verbose=True,
            boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=0.1,
                                step_frac=0.3))
    m = re.search(r"imb=([0-9.e+-]+)", buf.getvalue())
    return float(m.group(1)) if m else None


def step_jacobi(mesh, n_iters=3, alpha=0.5):
    with contextlib.redirect_stdout(io.StringIO()):
        uw.meshing.smooth_mesh_interior(
            mesh, n_iters=n_iters, alpha=alpha)


def step_spring(mesh, rho, size_w=8.0, shape_w=1.0, n_sweeps=300):
    with contextlib.redirect_stdout(io.StringIO()):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="spring",
            boundary_slip="box",
            method_kwargs=dict(size_w=size_w, shape_w=shape_w,
                                n_sweeps=n_sweeps))


# Each recipe: list of callables that take (mesh, rho).
def OT(m, r):
    return step_OT(m, r)


def jac(n=3):
    def _f(m, r):
        step_jacobi(m, n_iters=n)
    return _f


def spr(size_w=8.0, shape_w=1.0, n_sweeps=300):
    def _f(m, r):
        step_spring(m, r, size_w=size_w, shape_w=shape_w,
                    n_sweeps=n_sweeps)
    return _f


# All recipes end on OT. Each has 5 OT calls.
RECIPES = [
    ("A. OT x 5 (ref)",
     [OT] * 5),
    ("B. (OT, jac3) x 4 + OT",
     [OT, jac(3), OT, jac(3), OT, jac(3), OT, jac(3), OT]),
    ("C. (jac3, OT) x 5",
     [jac(3), OT, jac(3), OT, jac(3), OT, jac(3), OT, jac(3), OT]),
    ("D. OT, jac3, OT, jac3, OT (3 OT)",
     [OT, jac(3), OT, jac(3), OT]),
    ("E. (OT, spr_shape) x 4 + OT  (size_w=0)",
     [OT, spr(size_w=0.0, shape_w=1.0, n_sweeps=60),
      OT, spr(size_w=0.0, shape_w=1.0, n_sweeps=60),
      OT, spr(size_w=0.0, shape_w=1.0, n_sweeps=60),
      OT, spr(size_w=0.0, shape_w=1.0, n_sweeps=60),
      OT]),
    ("F. (OT, spr_light) x 4 + OT  (size_w=2)",
     [OT, spr(size_w=2.0, shape_w=1.0, n_sweeps=60),
      OT, spr(size_w=2.0, shape_w=1.0, n_sweeps=60),
      OT, spr(size_w=2.0, shape_w=1.0, n_sweeps=60),
      OT, spr(size_w=2.0, shape_w=1.0, n_sweeps=60),
      OT]),
]


results = {}
for label, actions in RECIPES:
    print(f"\n=== {label} ===")
    m = build_uniform_mesh()
    rho = analytic_rho(m)
    imb_traj = []
    for act in actions:
        ret = act(m, rho)
        if ret is not None:
            imb_traj.append(ret)
    final = imb_traj[-1] if imb_traj else float("nan")
    n_ot = sum(1 for a in actions if a is OT)
    print(f"  {n_ot} OT calls; imb {imb_traj[0]:.3f} → {final:.3f}")
    print(f"  traj: {' '.join(f'{v:.3f}' for v in imb_traj)}")
    results[label] = (m, imb_traj, final, n_ot)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(RECIPES):
    row, col = i // ncols, i % ncols
    m, imb, final, n_ot = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nimb={final:.3f}  ({n_ot} OT)",
                font_size=20, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_interleave2.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
