"""Compare in-loop incremental OT mesh at step 60 vs a fresh
uniform Annulus mesh subjected to OT×5 of the SAME T field
projected onto it.

Three panels:
  (a) step 60 mesh+T from the in-loop incremental OT run
  (b) fresh uniform mesh with T field projected onto it (no
      adaptation) — baseline for what the metric sees
  (c) fresh uniform mesh + OT×5 of the same T (refinement=3,
      coarsening=1, ring slip) — what a clean OT call produces

If (c) is much cleaner than (a), the slivers are path-dependent;
if (c) shows similar slivers, the OT step is intrinsically
sliver-prone on this metric.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import sympy
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis


p = argparse.ArgumentParser()
p.add_argument("--snapshot-dir", type=str,
               default=os.path.expanduser(
                   "~/+Simulations/StagnantLid/"
                   "ot_invest_ot5_mode1_ringslip"))
p.add_argument("--step", type=int, default=60)
p.add_argument("--refinement", type=float, default=3.0)
p.add_argument("--out", type=str,
               default="ot_fresh_vs_incremental_step60.png")
args = p.parse_args()

SNAPSHOT_LABEL = f"step{args.step:04d}"
SRC = args.snapshot_dir
DIAG = os.path.join(SRC, "diagnostics")
os.makedirs(DIAG, exist_ok=True)
OUT_PATH = os.path.join(DIAG, args.out)
print(f"=== fresh-OT vs incremental-OT @ {SNAPSHOT_LABEL} ===",
      flush=True)


def cell_quality(mesh):
    coords = np.asarray(mesh.X.coords)
    dm = mesh.dm
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, _ = dm.getDepthStratum(0)
    areas = np.empty(cEnd - cStart, dtype=float)
    for k, c in enumerate(range(cStart, cEnd)):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [pp for pp in closure
                 if pStart <= pp < pStart + coords.shape[0]]
        v = [coords[pp - pStart] for pp in verts[:3]]
        v0, v1, v2 = v
        areas[k] = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1])
                              - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    return float(areas.min() / areas.mean())


def render_subplot(pl, slot, mesh, T, title):
    pl.subplot(0, slot)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    pl.add_text(title, font_size=11, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                 clim=(0.0, 1.0), show_edges=False,
                 lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="#202020", line_width=0.5,
                 lighting=False, opacity=0.75)
    pl.view_xy()
    pl.camera.zoom(1.25)


# ------------------------------------------------------------
# (a) incremental in-loop OT mesh at step 60
# ------------------------------------------------------------
print("\n--- (a) loading in-loop step 60 ---", flush=True)
mesh_a = uw.discretisation.Mesh(
    os.path.join(SRC, f"{SNAPSHOT_LABEL}.mesh.00000.h5"))
T_a = uw.discretisation.MeshVariable(
    "T_v2p1", mesh_a, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
T_a.read_timestep(SNAPSHOT_LABEL, "T_v2p1", 0, outputPath=SRC)
q_a = cell_quality(mesh_a)
print(f"  step 60 mesh: q={q_a:.3f}  n_verts={mesh_a.X.coords.shape[0]}",
      flush=True)

# ------------------------------------------------------------
# (b) fresh uniform mesh with T projected — baseline
# ------------------------------------------------------------
print("\n--- (b) fresh uniform mesh + projected T ---",
      flush=True)
mesh_b = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5,
    cellSize=1.0/16, qdegree=3)
T_b = uw.discretisation.MeshVariable(
    "T_v2p1", mesh_b, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
# Project T from mesh_a onto mesh_b's DOF coords
T_b.data[:, 0] = np.asarray(uw.function.evaluate(
    T_a.sym[0], T_b.coords)).reshape(-1)
q_b = cell_quality(mesh_b)
print(f"  fresh mesh: q={q_b:.3f}  n_verts={mesh_b.X.coords.shape[0]}",
      flush=True)

# ------------------------------------------------------------
# (c) fresh + OT×5
# ------------------------------------------------------------
print("\n--- (c) fresh + OT × 5 (ring slip, no coarsening) ---",
      flush=True)
mesh_c = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5,
    cellSize=1.0/16, qdegree=3)
T_c = uw.discretisation.MeshVariable(
    "T_v2p1", mesh_c, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
T_c.data[:, 0] = np.asarray(uw.function.evaluate(
    T_a.sym[0], T_c.coords)).reshape(-1)
old_X = np.asarray(mesh_c.X.coords).copy()
old_T = np.asarray(T_c.data).copy()
rho_c = uw.meshing.metric_density_from_gradient(
    mesh_c, T_c, refinement=args.refinement,
    coarsening=1.0, metric_choice="front-following",
    name="fresh_compare")
uw.meshing.smooth_mesh_interior(
    mesh_c, metric=rho_c, method="ot",
    boundary_slip="ring",
    method_kwargs=dict(n_outer=5, relax=0.1, step_frac=0.3),
    verbose=True)
# FE-remap T from original-state to new mesh DOF positions
new_X = np.asarray(mesh_c.X.coords).copy()
new_Tx = np.asarray(T_c.coords).copy()
mesh_c._deform_mesh(old_X)
T_c.data[...] = old_T
remapped = np.asarray(uw.function.evaluate(
    T_c.sym[0], new_Tx)).reshape(-1)
mesh_c._deform_mesh(new_X)
T_c.data[:, 0] = remapped
q_c = cell_quality(mesh_c)
print(f"  fresh+OT×5: q={q_c:.3f}  n_verts={mesh_c.X.coords.shape[0]}",
      flush=True)

# ------------------------------------------------------------
# Render
pl = pv.Plotter(off_screen=True, shape=(1, 3),
                 window_size=(1800, 700))
pl.background_color = "white"
render_subplot(pl, 0, mesh_a, T_a,
                f"(a) incremental in-loop @ step 60  q={q_a:.3f}")
render_subplot(pl, 1, mesh_b, T_b,
                f"(b) fresh uniform (T projected)  q={q_b:.3f}")
render_subplot(pl, 2, mesh_c, T_c,
                f"(c) fresh + OT × 5  q={q_c:.3f}")
pl.screenshot(OUT_PATH)
pl.close()
print(f"\n=== SUMMARY ===", flush=True)
print(f"  (a) incremental  q={q_a:.3f}", flush=True)
print(f"  (b) fresh        q={q_b:.3f}", flush=True)
print(f"  (c) fresh + OT×5 q={q_c:.3f}", flush=True)
print(f"  → {OUT_PATH}", flush=True)
