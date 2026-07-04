"""Publication view of the fault working: T (RdBu_r) + velocity STREAMLINES +
the FAULT trace overlaid + fine mesh lines. High-res pyvista, documented pattern
(meshVariable_to_pv_mesh_object + .data, white bg, lighting=False).
"""
import os, glob, re, argparse
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--step", default="")
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
ap.add_argument("--fault-dip-deg", type=float, default=30.0)
ap.add_argument("--fault-theta-deg", type=float, default=90.0)
ap.add_argument("--fault-depth", type=float, default=0.225)
ap.add_argument("--nseed", type=int, default=110, help="streamline seed points")
ap.add_argument("--max-time", type=float, default=0.3,
                help="streamline integration time (small = short arcs, no looping)")
ap.add_argument("--all", action="store_true")
args = ap.parse_args()
D = os.path.expanduser(os.path.join(args.sim_dir, args.tag))
cands = sorted(glob.glob(os.path.join(D, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
labels = ([re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1) for c in cands]
          if args.all else [args.step or
          re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)])


def fault_trace():
    d = np.deg2rad(args.fault_dip_deg); th = np.deg2rad(args.fault_theta_deg)
    P0 = np.array([np.cos(th), np.sin(th)])
    e = np.array([np.cos(th), np.sin(th)]); t = np.array([-np.sin(th), np.cos(th)])
    dhat = np.cos(d) * t - np.sin(d) * e
    s = np.linspace(0.0, args.fault_depth / np.sin(d), 25)[:, None]
    return P0[None, :] + s * dhat[None, :]


xy = fault_trace()


def view(label):
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True, varsymbol="T")
    V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
    T.read_timestep(label, "T_v2p1", 0, outputPath=D)
    V.read_timestep(label, "V_v2p1", 0, outputPath=D)

    # T field on its own DOF cloud
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])

    # V on its own DOF cloud (3D-padded) for streamlines
    pv_V = vis.meshVariable_to_pv_mesh_object(V)
    Vd = np.zeros((V.data.shape[0], 3)); Vd[:, :2] = V.data
    pv_V.point_data["V"] = Vd

    # seed cloud: random points in the annulus
    rng = np.random.default_rng(42)
    pts = []
    while len(pts) < args.nseed:
        xyc = rng.uniform(-1, 1, (4 * args.nseed, 2))
        r = np.hypot(xyc[:, 0], xyc[:, 1])
        xyc = xyc[(r > 0.54) & (r < 0.97)]
        pts.extend(xyc.tolist())
    seeds = pv.PolyData(np.column_stack([np.array(pts[:args.nseed]), np.zeros(args.nseed)]))
    strm = pv_V.streamlines_from_source(
        seeds, vectors="V", integration_direction="both",
        max_time=args.max_time, initial_step_length=0.05, max_steps=1500)

    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    fault = pv.lines_from_points(np.column_stack([xy, np.zeros(len(xy))]))

    pl = pv.Plotter(off_screen=True, window_size=(1500, 1500))
    pl.set_background("white")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0, 1),
                show_edges=False, lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="#999999", line_width=0.15, lighting=False)   # FINE mesh
    if strm.n_points > 0:
        pl.add_mesh(strm.tube(radius=0.0012), color="#111111",
                    opacity=0.5, lighting=False)                            # sparse/thin flow
    pl.add_mesh(fault, color="#00c000", line_width=6, lighting=False)       # fault trace (on top)
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.zoom(1.25)
    out = os.path.join(D, f"view_{label}.png")
    pl.screenshot(out); pl.close()
    print("->", out, flush=True)


for lab in labels:
    view(lab)
