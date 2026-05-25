"""Movie of the mesh adapting during a saturation run. One frame
per checkpoint (dense ckpt_every=5 ⇒ one per adaptation event):
P3 T on its own DOF cloud + the deformed-mesh edges, RdBu_r /
white bg / lighting off (the free-surface viz convention).
Reads whatever checkpoints exist (partial OK — re-run as more
land). Writes a GIF (always) + MP4 if ffmpeg is present.

  python scripts/aniso_movie.py [tag=a16x] [fps=8]
"""
import sys
import os
import glob
import re
import shutil
import subprocess
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

TAG = sys.argv[1] if len(sys.argv) > 1 else "a16x"
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
D = "/tmp/metric_mesh/sat"
OUT = f"{D}/aniso_{TAG}_movie"


def ckpts(tag):
    ix = []
    for f in glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5"):
        m = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
        if m:
            ix.append(int(m.group(1)))
    return sorted(ix)


idx = ckpts(TAG)
if not idx:
    print(f"no checkpoints for sat_{TAG} yet")
    sys.exit(0)
print(f"{TAG}: {len(idx)} checkpoints {idx[0]}..{idx[-1]}")

pv.OFF_SCREEN = True
FRD = f"{D}/_frames_{TAG}"
os.makedirs(FRD, exist_ok=True)
for old in glob.glob(f"{FRD}/f*.png"):
    os.remove(old)
nfr = 0
for fi, k in enumerate(idx):
    m = uw.discretisation.Mesh(f"{D}/sat_{TAG}.mesh.{k:05}.h5")
    Tv = uw.discretisation.MeshVariable(
        "T", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    Tv.read_timestep(f"sat_{TAG}", "T", k, outputPath=D)
    pv_T = vis.meshVariable_to_pv_mesh_object(Tv)
    pv_T.point_data["T"] = np.asarray(Tv.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    pl.set_background("white")
    pl.add_text(f"{TAG}   adaptation checkpoint {k}",
                font_size=14, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0.0, 1.0),
                show_edges=False, lighting=False,
                show_scalar_bar=True,
                scalar_bar_args=dict(title="T", color="black"))
    pl.add_mesh(edges, color="#202020", line_width=0.7,
                lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.35)
    pl.screenshot(f"{FRD}/f{fi:04d}.png")
    pl.close()
    nfr += 1
    print(f"  frame {fi} (ckpt {k})", flush=True)

if not shutil.which("ffmpeg"):
    print(f"frames in {FRD} ({nfr}); ffmpeg not found — "
          f"no video assembled")
    sys.exit(0)
mp4 = OUT + ".mp4"
subprocess.run(
    ["ffmpeg", "-y", "-framerate", str(FPS), "-i",
     f"{FRD}/f%04d.png", "-c:v", "libx264", "-pix_fmt",
     "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4],
    check=True, capture_output=True)
print(f"saved {mp4}  ({nfr} frames @ {FPS} fps)")
# also a moderate-size gif for quick inline viewing (ffmpeg
# palette: clean colours)
pal = f"{FRD}/_pal.png"
gif = OUT + ".gif"
subprocess.run(
    ["ffmpeg", "-y", "-framerate", str(FPS), "-i",
     f"{FRD}/f%04d.png", "-vf",
     f"fps={FPS},scale=720:-1:flags=lanczos,palettegen", pal],
    check=True, capture_output=True)
subprocess.run(
    ["ffmpeg", "-y", "-framerate", str(FPS), "-i",
     f"{FRD}/f%04d.png", "-i", pal, "-lavfi",
     f"fps={FPS},scale=720:-1:flags=lanczos[x];[x][1:v]"
     "paletteuse", gif],
    check=True, capture_output=True)
print(f"saved {gif}")
