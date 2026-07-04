"""Zoomed side-by-side of the fault region (12 o'clock) showing T + the
adapted mesh for several runs at one step. The full-annulus render washes out
fault mesh detail (memory note); this crops to the fault neighbourhood so the
clustering (or lack of it) is visible.

Usage:
  python fault_zoom_compare.py --step step0060 \
      --tags rq_passive_uniform rq_passive_gmsh --labels "uniform 1.6x" "gmsh 4.7x"
"""
import os, argparse
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--step", default="step0060")
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", default=None)
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid+Fault")
ap.add_argument("--out", default="fault_zoom_compare.png")
# Fault is at theta=90 (top), dipping east. Crop a box around the trace.
ap.add_argument("--cx", type=float, default=0.12)
ap.add_argument("--cy", type=float, default=0.82)
ap.add_argument("--half", type=float, default=0.34)
args = ap.parse_args()

SIM = os.path.expanduser(args.sim_dir)
labels = args.labels or args.tags
n = len(args.tags)

pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(700 * n, 760),
                border=False)
pl.set_background("white")
for i, (tag, lab) in enumerate(zip(args.tags, labels)):
    D = os.path.join(SIM, tag)
    mp = os.path.join(D, f"{args.step}.mesh.00000.h5")
    if not os.path.exists(mp):
        continue
    mesh = uw.discretisation.Mesh(mp)
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3,
                                       continuous=True, varsymbol="T")
    T.read_timestep(args.step, "T_v2p1", 0, outputPath=D)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    pl.subplot(0, i)
    pl.add_text(lab, font_size=11, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0, 1),
                show_edges=False, lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=0.7, lighting=False)
    pl.view_xy()
    pl.camera.focal_point = (args.cx, args.cy, 0.0)
    pl.camera.position = (args.cx, args.cy, 10.0)
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = args.half

out = os.path.join(SIM, args.out)
pl.screenshot(out)
pl.close()
print("->", out)
