"""Render an adaptive-convection workflow run: T + mesh + V streamlines.

Reads the ``underworld3.workflows.Run`` directory layout: a per-step mesh
geometry ``run.mesh.NNNNN.h5`` (written with ``meshUpdates=True`` so each
adapted mesh is captured) plus ``run.mesh.<var>.NNNNN.h5`` field data.
Follows the canonical UW3 PyVista recipe (RdBu_r, white background,
lighting=False, DOF-faithful point data — see the ``uw-visualisation``
skill). Writes ``TV_NNNNN.png`` into the run directory. Always render ALL
frames when judging a run.

Usage:
  python render.py --run ~/+Simulations/AdaptiveConvection/baseline --all
  python render.py --run <dir> --all --focus 0 0.85 0.35   # crop to a feature
"""
import os
import re
import glob
import argparse

import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True

RUN_NAME = "run"


def fault_polyline(theta_deg=90.0, dip_deg=30.0, depth=0.225, dip_dir="east",
                   n=25):
    """Fault trace polyline (N,3) in model coords — mirrors
    ``fault_config._fault_geometry`` so renders can overlay the fault."""
    delta, th0 = np.deg2rad(dip_deg), np.deg2rad(theta_deg)
    P0 = np.array([np.cos(th0), np.sin(th0)])
    e_hat = np.array([np.cos(th0), np.sin(th0)])
    t_hat = np.array([-np.sin(th0), np.cos(th0)])
    side = 1.0 if dip_dir == "east" else -1.0
    dhat = side * np.cos(delta) * t_hat - np.sin(delta) * e_hat
    L = depth / np.sin(delta)
    xy = P0[None, :] + np.linspace(0, L, n)[:, None] * dhat[None, :]
    return np.column_stack([xy, np.zeros(len(xy))])


def fault_from_manifest(D):
    """Read the fault geometry from a run's manifest config_snapshot.
    Returns the (N,3) polyline, its midpoint (cx,cy), or (None, None)."""
    import yaml
    p = os.path.join(D, "manifest.yaml")
    if not os.path.exists(p):
        return None, None
    snap = (yaml.safe_load(open(p)) or {}).get("config_snapshot", {})
    if "fault_dip_deg" not in snap:
        return None, None
    poly = fault_polyline(
        theta_deg=snap.get("fault_theta_deg", 90.0),
        dip_deg=snap.get("fault_dip_deg", 30.0),
        depth=snap.get("fault_depth", 0.225),
        dip_dir=snap.get("fault_dip_dir", "east"))
    mid = poly[len(poly) // 2, :2]
    return poly, (float(mid[0]), float(mid[1]))


def _add_fault(pl, fault_xyz, color="red", line_width=3.0):
    """Overlay the fault trace as a line on the plotter (default red; pass a
    contrasting colour for warm/sequential field colourmaps)."""
    n = len(fault_xyz)
    line = pv.PolyData(fault_xyz)
    line.lines = np.hstack([[n], np.arange(n)])
    pl.add_mesh(line, color=color, line_width=line_width, lighting=False)


def render(D, index, *, tvar="T", tdeg=3, vvar="v", vdeg=2,
           clim=(0.0, 1.0), r_in=0.5, r_out=1.0, focus=None, mesh_only=False,
           fault_xyz=None):
    """Render the frame at *index*. ``focus=(cx, cy, half_width)`` crops to
    a feature region with a parallel-projection camera and bolder edges —
    the full-annulus view washes out mesh grading (viz skill).
    ``mesh_only`` draws just the (black) mesh edges on white — the clearest
    view of the adaptation."""
    mesh_h5 = os.path.join(D, f"{RUN_NAME}.mesh.{index:05d}.h5")
    mesh = uw.discretisation.Mesh(mesh_h5)
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    if mesh_only:
        pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
        pl.set_background("white")
        ew = 0.8 if focus is not None else 0.4
        pl.add_mesh(edges, color="black", line_width=ew, lighting=False)
        if fault_xyz is not None:
            _add_fault(pl, fault_xyz)
        pl.add_text(f"step {index}  ({mesh.X.coords.shape[0]} vertices)",
                    font_size=10, color="black")
        pl.view_xy()
        if focus is not None:
            cx, cy, hw = focus
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = hw
            pl.camera.focal_point = (cx, cy, 0)
            pl.camera.position = (cx, cy, 1)
            out = os.path.join(D, f"MESHfocus_{index:05d}.png")
        else:
            pl.camera.zoom(1.3)
            out = os.path.join(D, f"MESH_{index:05d}.png")
        pl.screenshot(out)
        pl.close()
        print("->", out, flush=True)
        return

    T = uw.discretisation.MeshVariable(tvar, mesh, 1, degree=tdeg, continuous=True)
    T.read_timestep(data_filename=RUN_NAME, data_name=tvar, index=index, outputPath=D)
    V = uw.discretisation.MeshVariable(vvar, mesh, mesh.dim, degree=vdeg,
                                       continuous=True)
    V.read_timestep(data_filename=RUN_NAME, data_name=vvar, index=index, outputPath=D)

    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])

    pv_V = vis.meshVariable_to_pv_mesh_object(V)
    Vd = np.asarray(V.data)
    vec = np.zeros((pv_V.n_points, 3))
    vec[:, 0], vec[:, 1] = Vd[:, 0], Vd[:, 1]
    pv_V["V"] = vec
    pv_V.set_active_vectors("V")
    vmax = float(np.linalg.norm(Vd, axis=1).max())

    rng = np.linspace(r_in + 0.05, r_out - 0.05, 4)
    th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    R, TH = np.meshgrid(rng, th)
    seed = pv.PolyData(np.c_[(R * np.cos(TH)).ravel(),
                             (R * np.sin(TH)).ravel(), np.zeros(R.size)])
    strm = pv_V.streamlines_from_source(
        seed, vectors="V", integration_direction="both",
        max_step_length=0.5, initial_step_length=0.02,
        max_steps=300, terminal_speed=max(vmax * 1e-4, 1e-9))

    pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pl.set_background("white")
    edge_w = 0.7 if focus is not None else 0.3
    edge_op = 0.85 if focus is not None else 0.4
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=tuple(clim),
                show_edges=False, lighting=False, scalar_bar_args={"title": "T"})
    pl.add_mesh(edges, color="grey", line_width=edge_w, lighting=False, opacity=edge_op)
    if strm.n_points > 0 and focus is None:
        pl.add_mesh(strm, color="black", line_width=1.4, lighting=False)
    if fault_xyz is not None:
        _add_fault(pl, fault_xyz)
    pl.add_text(f"step {index}  |v|max={vmax:.2f}", font_size=10, color="black")
    pl.view_xy()
    if focus is not None:
        cx, cy, hw = focus
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = hw
        pl.camera.focal_point = (cx, cy, 0)
        pl.camera.position = (cx, cy, 1)
        out = os.path.join(D, f"TVfocus_{index:05d}.png")
    else:
        pl.camera.zoom(1.3)
        out = os.path.join(D, f"TV_{index:05d}.png")
    pl.screenshot(out)
    pl.close()
    print("->", out, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Path to the run directory (holds run.mesh.*.h5).")
    ap.add_argument("--tvar", default="T")
    ap.add_argument("--vvar", default="v")
    ap.add_argument("--tdeg", type=int, default=3)
    ap.add_argument("--vdeg", type=int, default=2)
    ap.add_argument("--clim", type=float, nargs=2, default=[0.0, 1.0])
    ap.add_argument("--rin", type=float, default=0.5)
    ap.add_argument("--rout", type=float, default=1.0)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--focus", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "HALFWIDTH"),
                    help="Crop to a feature region (parallel projection).")
    ap.add_argument("--mesh-only", action="store_true",
                    help="Draw only the mesh edges (clearest adaptation view).")
    ap.add_argument("--fault", action="store_true",
                    help="Overlay the fault trace (red) read from the manifest.")
    ap.add_argument("--focus-fault", type=float, default=None, metavar="HALFWIDTH",
                    help="Crop centred on the fault midpoint at this half-width.")
    args = ap.parse_args(argv)

    D = os.path.expanduser(args.run)
    fault_xyz, fault_mid = (fault_from_manifest(D)
                            if (args.fault or args.focus_fault) else (None, None))
    xdmfs = sorted(glob.glob(os.path.join(D, f"{RUN_NAME}.mesh.[0-9]*.xdmf")),
                   key=lambda c: int(re.search(r"mesh\.(\d+)\.xdmf", c).group(1)))
    idxs = [int(re.search(r"mesh\.(\d+)\.xdmf", os.path.basename(c)).group(1))
            for c in xdmfs]
    if args.all:
        sel = idxs
    elif args.index is not None:
        sel = [args.index]
    elif idxs:
        sel = [idxs[-1]]
    else:
        sel = []

    if args.focus_fault and fault_mid is not None:
        focus = (fault_mid[0], fault_mid[1], args.focus_fault)
    elif args.focus:
        focus = tuple(args.focus)
    else:
        focus = None
    for idx in sel:
        render(D, idx, tvar=args.tvar, tdeg=args.tdeg, vvar=args.vvar,
               vdeg=args.vdeg, clim=tuple(args.clim),
               r_in=args.rin, r_out=args.rout, focus=focus,
               mesh_only=args.mesh_only,
               fault_xyz=(fault_xyz if args.fault else None))


if __name__ == "__main__":
    main()
