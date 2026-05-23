"""Compare the two halves of recipe B side-by-side:
  (i)  5 OT steps with SMOOTHED ρ (EPS=0.16) only
  (ii) 5 OT steps with SHARP ρ (EPS=0.04) only

This isolates what each level of the multi-res chain does on
its own — so we can see whether the smoothed level moves nodes
toward useful "coarse" positions or just settles into something
unrelated to the sharp metric.
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
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_levels')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_OT_multires import (
    analytic_rho_eps, build_uniform_mesh)


# Background = TRUE sharp ρ (always)
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

# Background = SMOOTH ρ for the smoothed panel
m_bg2 = build_uniform_mesh()
T_bg2 = uw.discretisation.MeshVariable(
    "T_bg_rho_smooth", m_bg2, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg2.data[:, 0] = np.asarray(uw.function.evaluate(
    analytic_rho_eps(m_bg2, eps=0.16),
    np.asarray(T_bg2.coords))).reshape(-1)
rho_clip2 = (1.0, float(T_bg2.data[:, 0].max()))
pv_bg2 = vis.meshVariable_to_pv_mesh_object(T_bg2)
pv_bg2.point_data["rho"] = np.asarray(T_bg2.data[:, 0])


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


def run_N_OT(eps, n_steps):
    m = build_uniform_mesh()
    rho = analytic_rho_eps(m, eps=eps)
    traj = []
    for _ in range(n_steps):
        ret = step_OT(m, rho, relax=0.1)
        if ret is not None:
            traj.append(ret)
    return m, traj


# (i) Smoothed-only: 5 OT × EPS=0.16
print("=== (i) 5 OT @ EPS=0.16 (smoothed ρ only) ===")
m_smooth, traj_smooth = run_N_OT(eps=0.16, n_steps=5)
print(f"  imb @ smoothed: "
      f"{' '.join(f'{v:.3f}' for v in traj_smooth)}")

# Measure final state's imbalance against SHARP ρ too — what
# does the smoothed-OT mesh look like to the real target?
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    uw.meshing.smooth_mesh_interior(
        m_smooth, metric=analytic_rho_eps(m_smooth, eps=0.04),
        method="ot", verbose=True, boundary_slip="box",
        method_kwargs=dict(n_outer=1, relax=0.0, step_frac=0.3))
m = re.search(r"imb=([0-9.e+-]+)", buf.getvalue())
sharp_imb_smooth = float(m.group(1)) if m else float("nan")
print(f"  imb of smoothed result vs SHARP ρ: {sharp_imb_smooth:.3f}")

# (ii) Sharp-only: 5 OT × EPS=0.04
print("\n=== (ii) 5 OT @ EPS=0.04 (sharp ρ only) ===")
m_sharp, traj_sharp = run_N_OT(eps=0.04, n_steps=5)
print(f"  imb @ sharp: "
      f"{' '.join(f'{v:.3f}' for v in traj_sharp)}")


# Two-panel render: smoothed-mesh on smoothed-ρ; sharp-mesh on
# sharp-ρ. Plus duplicate row showing both meshes against the
# SAME (sharp) ρ for direct mesh comparison.
ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1300 * ncols, 1300 * nrows),
                border=False)
pl.set_background("white")

# Top row: each panel against its own ρ
edges_smooth = vis.mesh_to_pv_mesh(m_smooth).extract_all_edges()
edges_sharp = vis.mesh_to_pv_mesh(m_sharp).extract_all_edges()

pl.subplot(0, 0)
pl.add_text(f"(i) 5 OT @ EPS=0.16  (smoothed rho)\n"
            f"imb={traj_smooth[-1]:.3f}",
            font_size=20, color='black')
pl.add_mesh(pv_bg2, scalars="rho", cmap="Blues",
            clim=rho_clip2, show_edges=False, lighting=False,
            show_scalar_bar=False, opacity=0.85)
pl.add_mesh(edges_smooth, color="black", line_width=1.0,
            lighting=False, opacity=0.85)
pl.view_xy(); pl.camera.zoom(1.15)

pl.subplot(0, 1)
pl.add_text(f"(ii) 5 OT @ EPS=0.04  (sharp rho)\n"
            f"imb={traj_sharp[-1]:.3f}",
            font_size=20, color='black')
pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
            clim=rho_clip, show_edges=False, lighting=False,
            show_scalar_bar=False, opacity=0.85)
pl.add_mesh(edges_sharp, color="black", line_width=1.0,
            lighting=False, opacity=0.85)
pl.view_xy(); pl.camera.zoom(1.15)

# Bottom row: both meshes against the SAME SHARP background
pl.subplot(1, 0)
pl.add_text(f"(i) smoothed-OT mesh on SHARP rho\n"
            f"sharp imb={sharp_imb_smooth:.3f}",
            font_size=20, color='black')
pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
            clim=rho_clip, show_edges=False, lighting=False,
            show_scalar_bar=False, opacity=0.85)
pl.add_mesh(edges_smooth, color="black", line_width=1.0,
            lighting=False, opacity=0.85)
pl.view_xy(); pl.camera.zoom(1.15)

pl.subplot(1, 1)
pl.add_text(f"(ii) sharp-OT mesh on SHARP rho\n"
            f"sharp imb={traj_sharp[-1]:.3f}",
            font_size=20, color='black')
pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
            clim=rho_clip, show_edges=False, lighting=False,
            show_scalar_bar=False, opacity=0.85)
pl.add_mesh(edges_sharp, color="black", line_width=1.0,
            lighting=False, opacity=0.85)
pl.view_xy(); pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_levels.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
