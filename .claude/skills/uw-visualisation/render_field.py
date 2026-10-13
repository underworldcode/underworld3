"""Render a UW3 scalar MeshVariable + mesh edges for one or all snapshots of a
run directory, following the canonical UW3 PyVista pattern (see SKILL.md).

Outputs <var>_<step>.png into the run directory. Outputs live under ~/+Simulations.

Usage:
  python render_field.py --tag <run> --sim-dir ~/+Simulations/<study> [--step stepNNNN | --all]
  python render_field.py --tag myrun --var T_v2p1 --degree 3 --clim 0 1 --all
"""
import os, glob, re, argparse
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
ap.add_argument("--var", default="T_v2p1", help="MeshVariable name in the checkpoint")
ap.add_argument("--degree", type=int, default=3)
ap.add_argument("--clim", type=float, nargs=2, default=[0.0, 1.0])
ap.add_argument("--cmap", default="RdBu_r")
ap.add_argument("--step", default="", help="stepNNNN; empty = latest")
ap.add_argument("--all", action="store_true", help="render every snapshot")
args = ap.parse_args()
D = os.path.expanduser(os.path.join(args.sim_dir, args.tag))

cands = sorted(glob.glob(os.path.join(D, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
if args.all:
    labels = [re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1) for c in cands]
elif args.step:
    labels = [args.step]
else:
    labels = [re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)]


def render(label):
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{label}.mesh.00000.h5"))
    var = uw.discretisation.MeshVariable(args.var, mesh, 1, degree=args.degree,
                                         continuous=True)
    var.read_timestep(label, args.var, 0, outputPath=D)
    pv_v = vis.meshVariable_to_pv_mesh_object(var)
    pv_v.point_data["f"] = np.asarray(var.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pl.set_background("white")
    pl.add_mesh(pv_v, scalars="f", cmap=args.cmap, clim=tuple(args.clim),
                show_edges=False, lighting=False,
                scalar_bar_args={"title": args.var})
    pl.add_mesh(edges, color="black", line_width=0.5, lighting=False)
    pl.view_xy(); pl.camera.zoom(1.3)
    out = os.path.join(D, f"{args.var}_{label}.png")
    pl.screenshot(out); pl.close()
    print("->", out, flush=True)


for lab in labels:
    render(lab)
