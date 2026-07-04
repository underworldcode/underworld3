"""Render frames from the t=0 → t=0.01 adapt-loop checkpoints
(snapshot every 5 steps) and assemble into an mp4.

Each frame: T field + mesh overlay on the annulus, with a
header showing step number and physical time.
"""
import os
import glob
import re
import subprocess
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True

OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapt_loop_movie_ref3')
FRAMES = os.path.join(OUT, 'frames')
os.makedirs(FRAMES, exist_ok=True)

# Discover checkpoints in step order
files = sorted(glob.glob(os.path.join(OUT, "step*.mesh.00000.h5")))
steps = [int(re.search(r"step(\d+)", f).group(1)) for f in files]
# Include init at step 0
if os.path.exists(os.path.join(OUT, "init.mesh.00000.h5")):
    steps = [0] + steps
    files = [os.path.join(OUT, "init.mesh.00000.h5")] + files
print(f"Found {len(steps)} checkpoints (step 0 to {max(steps)})")

# Load history for the per-step t value
hist = np.load(os.path.join(OUT, "history.npz"))
hist_step = np.asarray(hist['step']).astype(int)
hist_t = np.asarray(hist['t'])
step_to_t = {0: 0.0}
for i, s in enumerate(hist_step):
    step_to_t[int(s)] = float(hist_t[i])

# Render frames
for i, (s, mesh_file) in enumerate(zip(steps, files)):
    out_png = os.path.join(FRAMES, f"frame_{i:04d}.png")
    if os.path.exists(out_png):
        continue
    label = "init" if s == 0 else f"step{s:04d}"
    t_val = step_to_t.get(s, 0.0)
    m = uw.discretisation.Mesh(mesh_file)
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(label, "T_v2p1", 0, outputPath=OUT)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()

    pl = pv.Plotter(off_screen=True, window_size=(1500, 1500),
                    border=False)
    pl.set_background("white")
    pl.add_text(f"step {s:>4d}     t = {t_val:.5f}",
                font_size=22, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=0.9,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.25)
    pl.screenshot(out_png)
    pl.close()
    if i % 10 == 0:
        print(f"  rendered frame {i}/{len(steps)}")

print(f"\n{len(steps)} frames in {FRAMES}")

# Stitch with ffmpeg @ 12 fps (every 5 steps; this gives a
# ~10-second movie for 120 frames)
mp4 = os.path.join(OUT, "evolution.mp4")
cmd = [
    "ffmpeg", "-y", "-framerate", "12",
    "-i", os.path.join(FRAMES, "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    mp4,
]
print(f"ffmpeg ...")
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-1500:])
else:
    print(f"wrote {mp4}")
