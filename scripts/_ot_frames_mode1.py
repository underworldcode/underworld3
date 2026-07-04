"""Render per-step frames of the OT-only and OT+spring chain
progressions on mode-1 step 75.

PyVista with the project's canonical Red-Blue palette
(``cmap='RdBu_r'``, ``clim=(0, 1)``, white background, no
lighting, dark edges).

Per variant we save numbered PNGs into
``~/+Simulations/StagnantLid/ot_test_mode1/diagnostics/frames/``:

  OT-only chain (n_outer=1 calls, repeated):
    frame_ot_00.png         original
    frame_ot_01.png         after OT × 1
    frame_ot_02.png         after OT × 2
    frame_ot_03.png         after OT × 3
    frame_ot_04.png         after OT × 4
    frame_ot_05.png         after OT × 5

  OT + spring chain (4 OT + spring + final OT):
    frame_chain_00.png      original
    frame_chain_01.png      after OT × 1
    frame_chain_02.png      after OT × 2
    frame_chain_03.png      after OT × 3
    frame_chain_04.png      after OT × 4
    frame_chain_05.png      + spring polish (size_w=2)
    frame_chain_06.png      + final OT × 1

Each frame is a single annulus view; the same camera + scale-bar
for every frame so the sequence can be flipped into an animation
later.
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import sympy
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis


p = argparse.ArgumentParser()
p.add_argument("--snapshot-dir", type=str,
               default=os.path.expanduser(
                   "~/+Simulations/StagnantLid/ot_test_mode1"))
p.add_argument("--step", type=int, default=75)
p.add_argument("--refinement", type=float, default=3.0)
p.add_argument("--out-subdir", type=str, default="frames")
args = p.parse_args()

SNAPSHOT_LABEL = f"step{args.step:04d}"
SRC = args.snapshot_dir
FRAMES_DIR = os.path.join(SRC, "diagnostics", args.out_subdir)
os.makedirs(FRAMES_DIR, exist_ok=True)
print(f"=== OT/chain frames on {SNAPSHOT_LABEL} ===", flush=True)
print(f"  frames dir: {FRAMES_DIR}", flush=True)


def load_state():
    mesh = uw.discretisation.Mesh(
        os.path.join(SRC, f"{SNAPSHOT_LABEL}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    T.read_timestep(SNAPSHOT_LABEL, "T_v2p1", 0, outputPath=SRC)
    return mesh, T


def fe_remap_T(mesh, T, old_X, old_T_data, new_X):
    new_Tx = np.asarray(T.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T_data
    remapped = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = remapped


def cell_quality(mesh):
    """Min/mean of triangle areas — simple sliver indicator."""
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


# Display-T MeshVariable on a fresh mesh; we re-render the deformed
# state on a separate display copy each frame (display_T tracks the
# T values; we re-create a pv mesh object each frame).
def render_frame(mesh, T_field, label, out_path):
    pv_T = vis.meshVariable_to_pv_mesh_object(T_field)
    pv_T.point_data["T"] = np.asarray(T_field.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    pl = pv.Plotter(off_screen=True, window_size=(900, 900))
    pl.background_color = "white"
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                 clim=(0.0, 1.0), show_edges=False,
                 lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="#202020", line_width=0.7,
                 lighting=False, opacity=0.55)
    pl.add_text(label, font_size=12, color="black",
                 position=(20, 850))
    pl.view_xy()
    pl.camera.zoom(1.25)
    pl.screenshot(out_path)
    pl.close()
    print(f"  → {out_path}", flush=True)


# ----------------------------------------------------------------
# OT-only progression
# ----------------------------------------------------------------
print("\n--- OT-only chain (frames) ---", flush=True)
mesh, T = load_state()
old_X = np.asarray(mesh.X.coords).copy()
old_T = np.asarray(T.data).copy()
rho = uw.meshing.metric_density_from_gradient(
    mesh, T, refinement=args.refinement, name="frames_ot")

# Frame 0 — original
q = cell_quality(mesh)
render_frame(mesh, T,
              f"original | min/mean={q:.3f}",
              os.path.join(FRAMES_DIR, "frame_ot_00.png"))

# Frames 1..5 — one OT outer step at a time. We keep the SAME mesh
# instance throughout, calling smooth_mesh_interior(n_outer=1) each
# step, and FE-remap T against the previous-frame mesh state each
# time so the rendered T continues to represent the original field
# at the new coordinates.
prev_X = old_X
prev_T = old_T
for k in range(1, 6):
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="ot",
        boundary_slip="box",
        method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
        verbose=False)
    cur_X = np.asarray(mesh.X.coords).copy()
    fe_remap_T(mesh, T, prev_X, prev_T, cur_X)
    q = cell_quality(mesh)
    render_frame(
        mesh, T,
        f"OT × {k} | min/mean={q:.3f}",
        os.path.join(FRAMES_DIR, f"frame_ot_{k:02d}.png"))
    prev_X = cur_X
    prev_T = np.asarray(T.data).copy()

# ----------------------------------------------------------------
# Chain progression: OT×4 (one outer at a time) → spring → OT×1
# ----------------------------------------------------------------
print("\n--- Chain (frames) ---", flush=True)
mesh, T = load_state()
old_X = np.asarray(mesh.X.coords).copy()
old_T = np.asarray(T.data).copy()
rho = uw.meshing.metric_density_from_gradient(
    mesh, T, refinement=args.refinement, name="frames_chain")

# Frame 0 — original
q = cell_quality(mesh)
render_frame(mesh, T,
              f"original | min/mean={q:.3f}",
              os.path.join(FRAMES_DIR, "frame_chain_00.png"))

prev_X = old_X
prev_T = old_T
# Frames 1..4 — OT × 1 each
for k in range(1, 5):
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="ot",
        boundary_slip="box",
        method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
        verbose=False)
    cur_X = np.asarray(mesh.X.coords).copy()
    fe_remap_T(mesh, T, prev_X, prev_T, cur_X)
    q = cell_quality(mesh)
    render_frame(
        mesh, T,
        f"OT × {k} | min/mean={q:.3f}",
        os.path.join(FRAMES_DIR, f"frame_chain_{k:02d}.png"))
    prev_X = cur_X
    prev_T = np.asarray(T.data).copy()

# Frame 5 — spring polish
uw.meshing.smooth_mesh_interior(
    mesh, metric=rho, method="spring",
    boundary_slip="box",
    method_kwargs=dict(size_w=2.0),
    verbose=False)
cur_X = np.asarray(mesh.X.coords).copy()
fe_remap_T(mesh, T, prev_X, prev_T, cur_X)
q = cell_quality(mesh)
render_frame(
    mesh, T,
    f"+ spring(size_w=2) | min/mean={q:.3f}",
    os.path.join(FRAMES_DIR, "frame_chain_05.png"))
prev_X = cur_X
prev_T = np.asarray(T.data).copy()

# Frame 6 — final OT × 1
uw.meshing.smooth_mesh_interior(
    mesh, metric=rho, method="ot",
    boundary_slip="box",
    method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
    verbose=False)
cur_X = np.asarray(mesh.X.coords).copy()
fe_remap_T(mesh, T, prev_X, prev_T, cur_X)
q = cell_quality(mesh)
render_frame(
    mesh, T,
    f"+ final OT × 1 | min/mean={q:.3f}",
    os.path.join(FRAMES_DIR, "frame_chain_06.png"))

print(f"\ndone — frames in {FRAMES_DIR}", flush=True)
