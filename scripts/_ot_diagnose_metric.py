"""Diagnose the OT anti-resolution issue on mode-1 step 75.

Renders three side-by-side panels of the ORIGINAL (undeformed)
mesh at step 75:
  (1) T field (RdBu_r) — what the convection looks like
  (2) projected |∇T| — what the gradient projector sees
  (3) metric density ρ from metric_density_from_gradient
      (front-following, refinement=3.0) — what the mover is
      asked to equidistribute

If (3) doesn't match (1)/(2)'s plume location, the metric build
is wrong. If (3) matches but the mover ignores it, the mover is
broken. Either way the diagnosis is unambiguous.
"""
from __future__ import annotations
import os
import numpy as np
import sympy
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis


SRC = os.path.expanduser("~/+Simulations/StagnantLid/ot_test_mode1")
LABEL = "step0075"
DIAG = os.path.join(SRC, "diagnostics")
os.makedirs(DIAG, exist_ok=True)

mesh = uw.discretisation.Mesh(
    os.path.join(SRC, f"{LABEL}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
T.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
print(f"Loaded {LABEL}: T degree={T.degree}, "
      f"mesh qdegree={mesh.qdegree}, n_verts={mesh.X.coords.shape[0]}",
      flush=True)

# Projected |∇T| at degree 2 (same projector pattern as the
# metric build uses internally).
gradT_var = uw.discretisation.MeshVariable(
    "gradT_proj", mesh, vtype=uw.VarType.VECTOR,
    degree=2, continuous=True)
gproj = uw.systems.Vector_Projection(mesh, gradT_var)
gproj.smoothing = 0.0
X = mesh.CoordinateSystem.X
gproj.uw_function = sympy.Matrix(
    [T.sym[0].diff(X[i]) for i in range(2)]).T
gproj.solve()
gmag = np.linalg.norm(gradT_var.data, axis=1)
print(f"|∇T| projected: min={gmag.min():.3e}  "
      f"max={gmag.max():.3e}  mean={gmag.mean():.3e}", flush=True)

# Metric density via metric_density_from_gradient (exactly what
# the OT/chain frames script uses).
rho_sym = uw.meshing.metric_density_from_gradient(
    mesh, T, refinement=3.0, name="diag_mode1")
# rho_sym is a sympy expression; evaluate on a degree-1 MV
rho_var = uw.discretisation.MeshVariable(
    "rho_diag", mesh, vtype=uw.VarType.SCALAR,
    degree=1, continuous=True)
rho_var.data[:, 0] = np.asarray(uw.function.evaluate(
    rho_sym, rho_var.coords)).reshape(-1)
print(f"ρ from metric_density_from_gradient: "
      f"min={rho_var.data.min():.3e}  "
      f"max={rho_var.data.max():.3e}  "
      f"geomean={np.exp(np.log(rho_var.data.clip(1e-12)).mean()):.3e}",
      flush=True)


# ----------------------------------------------------------------
# Render
pl = pv.Plotter(off_screen=True, shape=(1, 3),
                 window_size=(1800, 700))
pl.background_color = "white"
edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

# Panel 1: T
pl.subplot(0, 0)
pv_T = vis.meshVariable_to_pv_mesh_object(T)
pv_T.point_data["T"] = np.asarray(T.data[:, 0])
pl.add_text("T (RdBu_r)", font_size=14, color="black")
pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0.0, 1.0),
            show_edges=False, lighting=False, show_scalar_bar=False)
pl.add_mesh(edges, color="#202020", line_width=0.7,
            lighting=False, opacity=0.55)
pl.view_xy(); pl.camera.zoom(1.25)

# Panel 2: |∇T| projected
pl.subplot(0, 1)
pv_grad = vis.meshVariable_to_pv_mesh_object(gradT_var)
pv_grad.point_data["gmag"] = gmag
pl.add_text(f"|∇T| projected  (max={gmag.max():.1e})",
            font_size=14, color="black")
pl.add_mesh(pv_grad, scalars="gmag", cmap="viridis",
            show_edges=False, lighting=False, show_scalar_bar=False)
pl.add_mesh(edges, color="#202020", line_width=0.7,
            lighting=False, opacity=0.55)
pl.view_xy(); pl.camera.zoom(1.25)

# Panel 3: ρ
pl.subplot(0, 2)
pv_rho = vis.meshVariable_to_pv_mesh_object(rho_var)
pv_rho.point_data["rho"] = np.asarray(rho_var.data[:, 0])
pl.add_text(f"ρ metric (front-following, ref=3)  "
            f"[{rho_var.data.min():.2f},{rho_var.data.max():.2f}]",
            font_size=14, color="black")
pl.add_mesh(pv_rho, scalars="rho", cmap="plasma",
            show_edges=False, lighting=False, show_scalar_bar=False)
pl.add_mesh(edges, color="#202020", line_width=0.7,
            lighting=False, opacity=0.55)
pl.view_xy(); pl.camera.zoom(1.25)

out = os.path.join(DIAG, "metric_diagnosis_step0075.png")
pl.screenshot(out)
pl.close()
print(f"wrote {out}", flush=True)

# Also: a percentile histogram of ρ to see how it's distributed.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                       constrained_layout=True)
rho_flat = np.asarray(rho_var.data[:, 0])
ax[0].hist(rho_flat, bins=40, color="tab:purple", alpha=0.7)
ax[0].axvline(1.0, color="k", lw=0.5, label="ρ=1 (no refine/coarsen)")
ax[0].set_xlabel("ρ"); ax[0].set_ylabel("count")
ax[0].set_title(f"ρ distribution  "
                f"(geomean = {np.exp(np.log(rho_flat.clip(1e-12)).mean()):.3f})")
ax[0].legend()
ax[1].hist(gmag, bins=40, color="tab:green", alpha=0.7)
ax[1].set_xlabel("|∇T|"); ax[1].set_ylabel("count")
ax[1].set_title("|∇T| distribution (projected)")
fig.savefig(os.path.join(DIAG, "metric_histograms_step0075.png"), dpi=130)
plt.close(fig)
print(f"wrote {os.path.join(DIAG, 'metric_histograms_step0075.png')}",
      flush=True)
