"""Adapt the step-125 T field at multiple R values and render
all the resulting meshes side by side over the |∇T| field.
"""
from __future__ import annotations
import os
import time
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True


SRC_DIR = os.path.expanduser(
    '~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
SRC_STEM = "sl_uniform_res16_Ra1e7_dEta1e4_step00125"
OUT_BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_compare')
R_LIST = [1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]

os.makedirs(OUT_BASE, exist_ok=True)


def load_uniform():
    m = uw.discretisation.Mesh(os.path.join(
        SRC_DIR, f"{SRC_STEM}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(SRC_STEM, "T_v2p1", 0, outputPath=SRC_DIR)
    return m, T


def gradT_mag(mesh, T):
    X = mesh.CoordinateSystem.X
    return sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                      + T.sym[0].diff(X[1]) ** 2)


# ---- 1. Adapt + save snapshots for each R ----------------------

for R in R_LIST:
    out_dir = os.path.join(OUT_BASE, f"R{R}")
    os.makedirs(out_dir, exist_ok=True)
    snap = os.path.join(out_dir, "adapted.mesh.00000.h5")
    if os.path.exists(snap):
        print(f"R={R}: already adapted, skipping")
        continue
    print(f"R={R}: adapting...")
    m, T = load_uniform()
    if R > 1.0:
        rho = uw.meshing.metric_density_from_gradient(
            m, T, amp=8.0, lo_percentile=50.0,
            hi_percentile=97.0, name=f"R{R}")
        t0 = time.time()
        uw.meshing.smooth_mesh_interior(
            m, metric=rho, method="anisotropic",
            method_kwargs=dict(resolution_ratio=R,
                               relax=0.2, n_outer=12))
        print(f"  adapted in {time.time() - t0:.1f}s")
    m.write_timestep(
        filename="adapted", index=0, outputPath=out_dir,
        meshVars=[T], meshUpdates=True, create_xdmf=True)


# ---- 2. Render all R in one plot -------------------------------

# Pre-pass: shared |∇T| color range
g_max = 0.0
loaded = []
for R in R_LIST:
    m_path = os.path.join(OUT_BASE, f"R{R}",
                          "adapted.mesh.00000.h5")
    m = uw.discretisation.Mesh(m_path)
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0,
                    outputPath=os.path.join(OUT_BASE, f"R{R}"))
    loaded.append((R, m, T))
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    g = vis.scalar_fn_to_pv_points(pv_T, gradT_mag(m, T))
    g_max = max(g_max, float(np.nanmax(g)))
print(f"global |∇T|max = {g_max:.3e}")

ncols = len(R_LIST)
pl = pv.Plotter(shape=(1, ncols), off_screen=True,
                window_size=(900 * ncols, 900),
                border=False)
pl.set_background("white")

for col, (R, m, T) in enumerate(loaded):
    pv_g = vis.meshVariable_to_pv_mesh_object(T)
    pv_g.point_data["gradT"] = vis.scalar_fn_to_pv_points(
        pv_g, gradT_mag(m, T))
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()

    # Compute alignment for the title
    rho = uw.meshing.metric_density_from_gradient(
        m, T, amp=8.0, lo_percentile=50.0, hi_percentile=97.0,
        name=f"R{R}_plot")
    mm = uw.meshing.mesh_metric_mismatch(m, rho,
                                          resolution_ratio=R)

    pl.subplot(0, col)
    title = (f"R={R}\n"
             f"alignment r={mm['alignment']:+.2f}\n"
             f"misalign={mm['misalignment']:.2f}")
    pl.add_text(title, font_size=14, color="black")
    pl.add_mesh(pv_g, scalars="gradT", cmap="Greens",
                clim=(0.0, g_max), show_edges=False,
                lighting=False,
                show_scalar_bar=(col == ncols - 1),
                scalar_bar_args=dict(title="|∇T|",
                                     color="black"))
    pl.add_mesh(edges, color="black", line_width=0.8,
                lighting=False, opacity=0.65)
    pl.view_xy()
    pl.camera.zoom(1.3)

out = os.path.join(OUT_BASE, "plot_R_compare.png")
pl.screenshot(out)
pl.close()
print(f"wrote {out}")
