"""Render EVERY snapshot of a run as a convection frame (documented pyvista
pattern: meshVariable_to_pv_mesh_object + T.data, RdBu_r, white bg,
lighting=False), in TWO styles — clean (no overlay) and with the mesh edges
(publication style) — and assemble a gif of each. Stepping through the clean
gif is the definitive transient-patch check; the mesh gif is for publication.
"""
import os, glob, re, argparse
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--fps", type=float, default=4.0)
ap.add_argument("--zoom", action="store_true", help="zoom on the 6 o'clock BL")
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
args = ap.parse_args()
D = os.path.expanduser(os.path.join(args.sim_dir, args.tag))
cands = sorted(glob.glob(os.path.join(D, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))


def frame(label, with_mesh):
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True, varsymbol="T")
    T.read_timestep(label, "T_v2p1", 0, outputPath=D)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pl.set_background("white")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0, 1),
                show_edges=False, lighting=False, show_scalar_bar=False)
    if with_mesh:
        edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
        pl.add_mesh(edges, color="black", line_width=0.5, lighting=False)
    pl.view_xy(); pl.enable_parallel_projection()
    if args.zoom:
        pl.camera.focal_point = (0.0, -0.72, 0.0); pl.camera.parallel_scale = 0.32
    suffix = "mesh" if with_mesh else "clean"
    out = os.path.join(D, f"f_{suffix}_{label}.png")
    pl.screenshot(out); pl.close()
    return out


for style in (False, True):
    frames = [frame(re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1), style)
              for c in cands]
    tag2 = "mesh" if style else "clean"
    gif = os.path.join(D, f"anim_{tag2}.gif")
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif, [imageio.imread(f) for f in frames],
                        duration=1.0 / args.fps, loop=0)
        print(f"-> {gif}  ({len(frames)} frames)", flush=True)
    except Exception as e:
        print(f"  gif assembly failed ({e}); frames in {D}", flush=True)
