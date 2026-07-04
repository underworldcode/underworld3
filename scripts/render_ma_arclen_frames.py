"""Render an animation from the MA + arc-length adaptive-convection
checkpoints written by stagnant_lid_adapt_loop.py.

Serial post-processor: the production run is parallel-blocked for the
reset-to-uniform MA mover, and rendering wants the whole field on one
rank anyway, so we just walk the per-step checkpoints (deformed mesh +
T written with meshUpdates=True) and draw each one.

Each frame draws the *high-order* (P3) T field the faithful way — the
``uw.visualisation`` "remesher" idiom from ``scripts/aniso_movie.py``:
``meshVariable_to_pv_mesh_object(T)`` triangulates through T's full P3
DOF cloud (not just the P1 vertices), and ``T.data`` is used directly
as the nodal point data, so the sub-cell cubic structure of thin plumes
survives instead of being smeared to a P1 / vertex-sampled view. The
actual deformed mesh edges are overlaid. ``RdBu_r`` / white bg /
lighting off (the free-surface viz convention). PNG frames go to
``<run>/frames/`` and (if ffmpeg is on PATH) are stitched into
``<run>/ma_arclen.mp4``.

Usage
-----
    pixi run -e amr-dev python -u scripts/render_ma_arclen_frames.py \
        --run ~/+Simulations/StagnantLid/ma_arclen_mode1_ra1e7_res24
"""
from __future__ import annotations
import os
import re
import glob
import argparse
import subprocess
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

# Headless off-screen rendering (the convention from scripts/aniso_movie.py).
pv.OFF_SCREEN = True


p = argparse.ArgumentParser()
p.add_argument("--run", type=str, required=True,
               help="Run output directory (contains step*.mesh.*.h5).")
p.add_argument("--cmap", type=str, default="RdBu_r")
p.add_argument("--dpi", type=int, default=170)
p.add_argument("--lw", type=float, default=0.5,
               help="Mesh-edge line width on the overlay.")
p.add_argument("--fps", type=int, default=15)
p.add_argument("--stride", type=int, default=1,
               help="Render every Nth available step (1 = all).")
p.add_argument("--force", action="store_true",
               help="Re-render frames even if the PNG already "
                    "exists (default: skip existing — incremental, "
                    "so the watcher only draws new steps).")
p.add_argument("--no-movie", action="store_true",
               help="Render frames only; skip the mp4/gif stitch.")
args = p.parse_args()

RUN = os.path.expanduser(args.run)
FRAME_DIR = os.path.join(RUN, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)


def _labels():
    """All checkpoint labels present, in step order (init first)."""
    out = []
    for f in glob.glob(os.path.join(RUN, "*.mesh.00000.h5")):
        b = os.path.basename(f)
        m = re.match(r"(init|step\d+)\.mesh\.00000\.h5$", b)
        if not m:
            continue
        lab = m.group(1)
        step = 0 if lab == "init" else int(lab[4:])
        out.append((step, lab))
    out.sort()
    return out


# History (for the t / Nu annotation), if present.
HIST = {}
hpath = os.path.join(RUN, "history.npz")
if os.path.exists(hpath):
    z = np.load(hpath)
    for i in range(len(z["step"])):
        HIST[int(z["step"][i])] = (float(z["t"][i]), float(z["Nu"][i]))


def render(step, label):
    mesh = uw.discretisation.Mesh(
        os.path.join(RUN, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    T.read_timestep(label, "T_v2p1", 0, outputPath=RUN)

    # Remesher: triangulate through T's P3 DOF cloud and use the
    # raw nodal data so the high-order field is shown faithfully.
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    t_sim, nu = HIST.get(step, (float("nan"), float("nan")))
    pl = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    pl.set_background("white")
    pl.add_text(
        f"MA + arc-length   step {step:>3d}   "
        f"t={t_sim:.4f}   Nu={nu:+.2f}",
        font_size=13, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap=args.cmap, clim=(0.0, 1.0),
                show_edges=False, lighting=False,
                show_scalar_bar=True,
                scalar_bar_args=dict(title="T", color="black"))
    pl.add_mesh(edges, color="#202020", line_width=args.lw,
                lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.35)
    out = os.path.join(FRAME_DIR, f"frame_{step:04d}.png")
    pl.screenshot(out)
    pl.close()
    return out


labels = _labels()
labels = labels[:: max(args.stride, 1)]
n_new = 0
print(f"{len(labels)} checkpoints in {RUN}", flush=True)
for step, label in labels:
    fp = os.path.join(FRAME_DIR, f"frame_{step:04d}.png")
    if os.path.exists(fp) and not args.force:
        continue
    render(step, label)
    n_new += 1
    print(f"  {label:>9s} -> {os.path.basename(fp)}", flush=True)
print(f"rendered {n_new} new frame(s)", flush=True)
if args.no_movie:
    raise SystemExit(0)

# Stitch into a movie. Prefer ffmpeg (mp4); fall back to a Pillow GIF.
have_ffmpeg = subprocess.run(["which", "ffmpeg"],
                             capture_output=True).returncode == 0
frame_paths = [os.path.join(FRAME_DIR, f"frame_{s:04d}.png")
               for s, _ in labels]
if have_ffmpeg and labels:
    mp4 = os.path.join(RUN, "ma_arclen.mp4")
    listfile = os.path.join(FRAME_DIR, "frames.txt")
    with open(listfile, "w") as fh:
        for step, _ in labels:
            fh.write(f"file 'frame_{step:04d}.png'\n")
            fh.write(f"duration {1.0/args.fps:.4f}\n")
        fh.write(f"file 'frame_{labels[-1][0]:04d}.png'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", listfile, "-vf",
           "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
           "-r", str(args.fps), mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"wrote {mp4}" if r.returncode == 0
          else "ffmpeg failed:\n" + r.stderr[-1500:], flush=True)
elif labels:
    from PIL import Image
    gif = os.path.join(RUN, "ma_arclen.gif")
    frames = [Image.open(fp).convert("P", palette=Image.ADAPTIVE)
              for fp in frame_paths]
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000.0 / args.fps), loop=0,
                   optimize=True)
    print(f"ffmpeg not found — wrote Pillow GIF {gif}", flush=True)
else:
    print("no frames to stitch.", flush=True)
