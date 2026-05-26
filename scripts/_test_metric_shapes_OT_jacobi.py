"""Jacobi-as-preconditioner sweep: test whether a light
Jacobi BEFORE each OT step enables OT to take larger relax
than it could safely take alone.

Hypothesis: pure OT at relax=0.1 is mesh-quality-limited
(backtrack engages once cells get anisotropic). If Jacobi
restores cell shape between OTs, OT can run at higher relax
and converge faster. Trade-off: Jacobi pulls toward centroid
(anti-OT redistribution) so a sweet spot exists.
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
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_jacobi')
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


def step_jacobi(mesh, n_iters=1, alpha=0.3):
    with contextlib.redirect_stdout(io.StringIO()):
        uw.meshing.smooth_mesh_interior(
            mesh, n_iters=n_iters, alpha=alpha)


def OT(relax=0.1):
    def _f(m, r):
        return step_OT(m, r, relax=relax)
    return _f


def jac(n=1, a=0.3):
    def _f(m, r):
        step_jacobi(m, n_iters=n, alpha=a)
    return _f


# All recipes end on OT. Compare matched OT-count (5 OT) vs
# matched-imbalance ("how fast does it converge?").
RECIPES = [
    ("A. OT(0.1) x 5 (baseline)",
     [OT(0.1)] * 5),
    ("B. OT(0.3) x 5 (raw bigger step)",
     [OT(0.3)] * 5),
    ("C. (jac1_a0.3, OT0.1) x 4 + OT0.1",
     [jac(1, 0.3), OT(0.1)] * 4 + [OT(0.1)]),
    ("D. (jac1_a0.3, OT0.3) x 4 + OT0.3",
     [jac(1, 0.3), OT(0.3)] * 4 + [OT(0.3)]),
    ("E. (jac3_a0.5, OT0.3) x 4 + OT0.3",
     [jac(3, 0.5), OT(0.3)] * 4 + [OT(0.3)]),
    ("F. (jac1_a0.3, OT0.6) x 4 + OT0.6",
     [jac(1, 0.3), OT(0.6)] * 4 + [OT(0.6)]),
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
    print(f"  imb {imb_traj[0]:.3f} → {final:.3f}")
    print(f"  traj: {' '.join(f'{v:.3f}' for v in imb_traj)}")
    results[label] = (m, imb_traj, final)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(RECIPES):
    row, col = i // ncols, i % ncols
    m, imb, final = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nimb={final:.3f}",
                font_size=20, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_jacobi.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
