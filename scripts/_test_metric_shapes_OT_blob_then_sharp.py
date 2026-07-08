"""Blob (wide, amplified, brakes off) → Sharp chain.

Hypothesis: wide AMP=20 blob with relax=1.0/sf=1.0 transports
nodes into broad halos around features (recipe F from
blob_loose). Then sharp OT can narrow those halos onto the
true band positions — finally seeing the multi-res benefit.

Compare to raw sharp × N at matched compute.
"""
import os, sys, io, re, contextlib, time
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_blob_then_sharp')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_OT_blob_loose import (
    build_uniform_mesh, rho_blob_tanh, rho_sharp)


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
    dx_m = re.search(r"max\|Δx\|=([0-9.e+-]+)", out)
    return (float(imb_m.group(1)) if imb_m else None,
            float(dx_m.group(1)) if dx_m else None)


def run_chain(schedule, label):
    print(f"\n=== {label} ===")
    t0 = time.time()
    m = build_uniform_mesh()
    history = []
    for kind, n in schedule:
        if kind == "narrow":
            # narrow conservative blob — uses sharp's brake
            # tuning (B from blob_loose: AMP=8 eps=0.06,
            # relax=0.1, sf=0.3). Gentle preconditioning.
            rho = rho_blob_tanh(m, 8.0, 0.06)
            relax, sf = 0.1, 0.3
        elif kind == "sharp":
            rho = rho_sharp(m)
            relax, sf = 0.1, 0.3
        for _ in range(n):
            imb, dx = step_OT(m, rho, relax, sf)
            history.append((kind, imb, dx))
            print(f"  {kind}: imb={imb}  max|Δx|={dx:.3e}"
                  if dx else f"  {kind}: --")
    # final sharp imb measurement
    imb_s, _ = step_OT(m, rho_sharp(m), 0.0, 0.3)
    print(f"  FINAL sharp imb={imb_s:.3f}  "
          f"({time.time()-t0:.1f}s)")
    return m, imb_s


# All recipes have 10 OT calls total (matched compute).
RECIPES = [
    ("A. sharp OT x 10 (raw reference)",
     [("sharp", 10)]),
    ("B. narrow x 5, sharp x 5",
     [("narrow", 5), ("sharp", 5)]),
    ("C. narrow x 3, sharp x 7",
     [("narrow", 3), ("sharp", 7)]),
    ("D. narrow x 7, sharp x 3",
     [("narrow", 7), ("sharp", 3)]),
    ("E. narrow x 2, sharp x 8",
     [("narrow", 2), ("sharp", 8)]),
    ("F. narrow x 10 alone (no sharp)",
     [("narrow", 10)]),
]

results = {}
for label, sched in RECIPES:
    m, imb_s = run_chain(sched, label)
    results[label] = (m, imb_s)


ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(RECIPES):
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

out_png = os.path.join(OUT, "plot_OT_blob_then_sharp.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
