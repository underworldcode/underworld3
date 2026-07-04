"""Re-render mode-1 step-75 OT-only and chain progressions with
boundary_slip='ring' (the correct mode for an annulus) instead
of 'box' (which lets boundary nodes drift onto an axis-aligned
bounding box, distorting the actual circular boundary).

Same as _ot_frames_mode1.py but with a single boundary_slip
change and a different output subdir name to keep the two side
by side.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis


p = argparse.ArgumentParser()
p.add_argument("--snapshot-dir", type=str,
               default=os.path.expanduser(
                   "~/+Simulations/StagnantLid/ot_test_mode1"))
p.add_argument("--step", type=int, default=75)
p.add_argument("--refinement", type=float, default=3.0)
p.add_argument("--slip", type=str, default="ring",
               choices=["ring", "box", "false"],
               help="boundary_slip mode: 'ring' (correct for "
                    "annulus), 'box' (the wrong setting that "
                    "drifted the surface in the original run), "
                    "or 'false' for pinned boundaries.")
p.add_argument("--out-subdir", type=str, default="frames_ring")
args = p.parse_args()

SLIP = False if args.slip == "false" else args.slip
SNAPSHOT_LABEL = f"step{args.step:04d}"
SRC = args.snapshot_dir
FRAMES_DIR = os.path.join(SRC, "diagnostics", args.out_subdir)
os.makedirs(FRAMES_DIR, exist_ok=True)
print(f"=== OT/chain frames on {SNAPSHOT_LABEL} "
      f"(slip={SLIP!r}) ===", flush=True)
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
    mesh, T, refinement=args.refinement, name=f"frames_ot_{args.slip}")

q = cell_quality(mesh)
render_frame(mesh, T,
              f"original | min/mean={q:.3f}",
              os.path.join(FRAMES_DIR, "frame_ot_00.png"))

for k in range(1, 6):
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="ot",
        boundary_slip=SLIP,
        method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
        verbose=False)
    cur_X = np.asarray(mesh.X.coords).copy()
    # FE-remap from the ORIGINAL state to the current mesh — never
    # compound interpolation across frames. The displayed T is
    # always the true step-75 field, just sampled on the current
    # mesh.
    fe_remap_T(mesh, T, old_X, old_T, cur_X)
    q = cell_quality(mesh)
    render_frame(
        mesh, T,
        f"OT × {k} (slip={SLIP!r}) | min/mean={q:.3f}",
        os.path.join(FRAMES_DIR, f"frame_ot_{k:02d}.png"))

# ----------------------------------------------------------------
# Chain progression: OT×4 → spring → OT×1
# ----------------------------------------------------------------
print("\n--- Chain (frames) ---", flush=True)
mesh, T = load_state()
old_X = np.asarray(mesh.X.coords).copy()
old_T = np.asarray(T.data).copy()
rho = uw.meshing.metric_density_from_gradient(
    mesh, T, refinement=args.refinement,
    name=f"frames_chain_{args.slip}")

q = cell_quality(mesh)
render_frame(mesh, T,
              f"original | min/mean={q:.3f}",
              os.path.join(FRAMES_DIR, "frame_chain_00.png"))

for k in range(1, 5):
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="ot",
        boundary_slip=SLIP,
        method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
        verbose=False)
    cur_X = np.asarray(mesh.X.coords).copy()
    fe_remap_T(mesh, T, old_X, old_T, cur_X)
    q = cell_quality(mesh)
    render_frame(
        mesh, T,
        f"OT × {k} (slip={SLIP!r}) | min/mean={q:.3f}",
        os.path.join(FRAMES_DIR, f"frame_chain_{k:02d}.png"))

uw.meshing.smooth_mesh_interior(
    mesh, metric=rho, method="spring",
    boundary_slip=SLIP,
    method_kwargs=dict(size_w=2.0),
    verbose=False)
cur_X = np.asarray(mesh.X.coords).copy()
fe_remap_T(mesh, T, old_X, old_T, cur_X)
q = cell_quality(mesh)
render_frame(
    mesh, T,
    f"+ spring(size_w=2) (slip={SLIP!r}) | min/mean={q:.3f}",
    os.path.join(FRAMES_DIR, "frame_chain_05.png"))

uw.meshing.smooth_mesh_interior(
    mesh, metric=rho, method="ot",
    boundary_slip=SLIP,
    method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
    verbose=False)
cur_X = np.asarray(mesh.X.coords).copy()
fe_remap_T(mesh, T, old_X, old_T, cur_X)
q = cell_quality(mesh)
render_frame(
    mesh, T,
    f"+ final OT × 1 (slip={SLIP!r}) | min/mean={q:.3f}",
    os.path.join(FRAMES_DIR, "frame_chain_06.png"))

print(f"\ndone — frames in {FRAMES_DIR}", flush=True)
