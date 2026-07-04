"""Zoom-in view of one plume showing how OT refinement aligns
with ∇T bands. Builds the original mesh, the OT×5-improved mesh,
and renders a 2-panel side-by-side at ~3× zoom on the top plume.
"""
from __future__ import annotations
import os
import numpy as np
import sympy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

import underworld3 as uw


SRC = os.path.expanduser("~/+Simulations/StagnantLid/ot_test")
LABEL = "step0025"
R_INNER, R_OUTER = 0.5, 1.0


def load_state():
    mesh = uw.discretisation.Mesh(
        os.path.join(SRC, f"{LABEL}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    T.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
    return mesh, T


def mesh_edges_xy(mesh):
    dm = mesh.dm
    coords = np.asarray(mesh.X.coords)
    pStart, _ = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    segs = np.empty((eEnd - eStart, 2, 2), dtype=float)
    for k, e in enumerate(range(eStart, eEnd)):
        cone = dm.getCone(e)
        segs[k, 0] = coords[cone[0] - pStart]
        segs[k, 1] = coords[cone[1] - pStart]
    return segs


def fe_remap_T(mesh, T, old_X, old_T_data, new_X):
    new_Tx = np.asarray(T.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T_data
    remapped = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = remapped


def make_panel(ax, mesh, T_field, title, xlim, ylim):
    Xc = np.asarray(T_field.coords)
    Tv = np.asarray(T_field.data[:, 0])
    tri = Triangulation(Xc[:, 0], Xc[:, 1])
    cx = Xc[tri.triangles, 0].mean(axis=1)
    cy = Xc[tri.triangles, 1].mean(axis=1)
    rcen = np.sqrt(cx**2 + cy**2)
    mask = (rcen > R_OUTER + 1e-6) | (rcen < R_INNER - 1e-6)
    tri.set_mask(mask)
    ax.tripcolor(tri, Tv, cmap="inferno", shading="gouraud",
                 vmin=0, vmax=1)
    # ∇T contour overlay (medium-intensity bands at T=0.3, 0.5, 0.7)
    ax.tricontour(tri, Tv, levels=[0.2, 0.4, 0.6, 0.8],
                  colors="cyan", linewidths=0.6, alpha=0.6)
    segs = mesh_edges_xy(mesh)
    lc = LineCollection(segs, colors="white", linewidths=0.5,
                        alpha=0.85)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# --- panel A: original ---
mesh_a, T_a = load_state()

# --- panel B: OT × 5 ---
mesh_b, T_b = load_state()
old_X_b = np.asarray(mesh_b.X.coords).copy()
old_T_b = np.asarray(T_b.data).copy()
rho_b = uw.meshing.metric_density_from_gradient(
    mesh_b, T_b, refinement=3.0, name="ot_zoom")
uw.meshing.smooth_mesh_interior(
    mesh_b, metric=rho_b, method="ot",
    boundary_slip="box",
    method_kwargs=dict(n_outer=5, relax=0.1, step_frac=0.3),
    verbose=False)
new_X_b = np.asarray(mesh_b.X.coords).copy()
fe_remap_T(mesh_b, T_b, old_X_b, old_T_b, new_X_b)

# Top plume sits near (0, +0.8). Zoom on a wedge there.
ZOOM = ((-0.55, 0.55), (0.45, 1.05))

fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                          constrained_layout=True)
make_panel(axes[0], mesh_a, T_a,
           "(a) original snapshot mesh (after 2 anisotropic adapts)",
           *ZOOM)
make_panel(axes[1], mesh_b, T_b,
           "(b) + OT × 5 improvement step",
           *ZOOM)
fig.suptitle(
    "Top-plume zoom: mesh edges (white) over T (inferno), "
    "T isolines (cyan)\n"
    "step 25 — Ra=1e7, Δη=1e2, mode 5  —  refinement=3.0")
out = "/tmp/ot_test_logs/ot_zoom_plume_step0025.png"
fig.savefig(out, dpi=160)
plt.close(fig)
print(f"wrote {out}")
