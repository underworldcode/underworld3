"""Convection-style render using the DOCUMENTED UW3 pyvista pattern
(memory: PyVista visualisation pattern for UW3 fields). Faithful to the P3 T
field, clean RdBu_r, white background, mesh edges overlaid.

Key points (all matter — getting any wrong makes it look grey/patchy):
  * meshVariable_to_pv_mesh_object(T) + attach T.data DIRECTLY (P3-faithful;
    do NOT re-evaluate at vertices).
  * cmap="RdBu_r" (clean blue-white-red), NOT coolwarm (grey midtone).
  * pl.set_background("white") — else the grey bg bleeds through T=0.5 white.
  * lighting=False on every add_mesh — else the colormap darkens.
  * overlay mesh edges via mesh_to_pv_mesh(mesh).extract_all_edges().
"""
import os, glob, re, argparse
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--step", default="", help="stepNNNN; empty = latest")
ap.add_argument("--all", action="store_true", help="render every snapshot")
args = ap.parse_args()
D = os.path.expanduser(f"~/+Simulations/StagnantLid/{args.tag}")

cands = sorted(glob.glob(os.path.join(D, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
if args.all:
    labels = [re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1) for c in cands]
elif args.step:
    labels = [args.step]
else:
    labels = [re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)]


def render_T(label):
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True, varsymbol="T")
    T.read_timestep(label, "T_v2p1", 0, outputPath=D)

    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pl.set_background("white")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0, 1),
                show_edges=False, lighting=False, scalar_bar_args={"title": "T"})
    pl.add_mesh(edges, color="black", line_width=0.5, lighting=False)
    pl.view_xy(); pl.camera.zoom(1.3)
    out = os.path.join(D, f"T_{label}.png")
    pl.screenshot(out); pl.close()
    print("->", out, flush=True)


for lab in labels:
    render_T(lab)
