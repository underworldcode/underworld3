"""Render a UW3 run: T colormap + mesh edges + VELOCITY STREAMLINES.
Follows the canonical UW3 PyVista pattern (RdBu_r, white bg, lighting=False,
DOF-faithful) and overlays streamlines of V. Outputs <stem>_<step>.png.

Usage:
  python render_field_streamlines.py --tag <run> --sim-dir ~/+Simulations/<study> [--step stepNNNN | --all]
"""
import os, glob, re, argparse
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
pv.OFF_SCREEN = True

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
ap.add_argument("--tvar", default="T_v2p1"); ap.add_argument("--tdeg", type=int, default=3)
ap.add_argument("--vvar", default="V_v2p1"); ap.add_argument("--vdeg", type=int, default=2)
ap.add_argument("--clim", type=float, nargs=2, default=[0.0, 1.0])
ap.add_argument("--step", default=""); ap.add_argument("--all", action="store_true")
ap.add_argument("--rin", type=float, default=0.5); ap.add_argument("--rout", type=float, default=1.0)
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
    T = uw.discretisation.MeshVariable(args.tvar, mesh, 1, degree=args.tdeg, continuous=True)
    T.read_timestep(label, args.tvar, 0, outputPath=D)
    V = uw.discretisation.MeshVariable(args.vvar, mesh, mesh.dim, degree=args.vdeg, continuous=True)
    V.read_timestep(label, args.vvar, 0, outputPath=D)

    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    # velocity mesh + 3D vectors for streamlines
    pv_V = vis.meshVariable_to_pv_mesh_object(V)
    Vd = np.asarray(V.data)
    vec = np.zeros((pv_V.n_points, 3)); vec[:, 0] = Vd[:, 0]; vec[:, 1] = Vd[:, 1]
    pv_V["V"] = vec; pv_V.set_active_vectors("V")
    vmax = float(np.linalg.norm(Vd, axis=1).max())

    # seed a sparse disc of points across the annulus (sparse + short
    # integration so streamlines read as flow direction, not wound-up spirals
    # that black out the field — low-Ra cells are tight closed loops).
    rng = np.linspace(args.rin + 0.05, args.rout - 0.05, 4)
    th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    R, TH = np.meshgrid(rng, th)
    seed = pv.PolyData(np.c_[(R * np.cos(TH)).ravel(),
                             (R * np.sin(TH)).ravel(),
                             np.zeros(R.size)])
    strm = pv_V.streamlines_from_source(
        seed, vectors="V", integration_direction="both",
        max_step_length=0.5, initial_step_length=0.02,
        max_steps=300, terminal_speed=max(vmax * 1e-4, 1e-9))

    pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pl.set_background("white")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=tuple(args.clim),
                show_edges=False, lighting=False, scalar_bar_args={"title": "T"})
    pl.add_mesh(edges, color="grey", line_width=0.3, lighting=False, opacity=0.4)
    if strm.n_points > 0:
        pl.add_mesh(strm, color="black", line_width=1.4, lighting=False)
    pl.add_text(f"{label}  |v|max={vmax:.2f}", font_size=10, color="black")
    pl.view_xy(); pl.camera.zoom(1.3)
    out = os.path.join(D, f"TV_{label}.png")
    pl.screenshot(out); pl.close()
    print("->", out, flush=True)


for lab in labels:
    render(lab)
