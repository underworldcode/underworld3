"""Test: build a fault-refined annulus (gmsh Distance+Threshold), then MIGRATE
the fault by a CFL amount (≈1 fine element) around the upper boundary —
preserving dip angle and max depth — and check that mmpde MOVES the existing
fine cluster to follow (node-movers can't CREATE refinement, but they CAN move
it), while PRESERVING TOPOLOGY (connectivity unchanged) and parallel layout.

Each migration step: shift the fault azimuth by dtheta, build the fault metric
at the NEW location, run mmpde, measure where the refinement sits.
"""
import os, argparse, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('--bulk', type=float, default=1/8, help='bulk cellSize')
ap.add_argument('--smin', type=float, default=1/40, help='gmsh fine size at fault')
# Tracking metric MATCHED to the gmsh refinement: across-fault target = bulk/Rf.
# Default Rf = bulk/smin so the metric asks for EXACTLY the size gmsh built — the
# mover then just MAINTAINS + TRANSLATES the cluster (no over-squeeze / saturation).
ap.add_argument('--rf', type=float, default=-1.0, help='anisotropy; <0 ⇒ bulk/smin (matched)')
ap.add_argument('--amp', type=float, default=3.0, help='isotropic draw amplitude')
ap.add_argument('--nstep', type=int, default=5)
ap.add_argument('--subiters', type=int, default=1, help='mmpde calls per migration step')
ap.add_argument('--carrier', action='store_true',
                help='Apply a windowed rigid ROTATION (Δθ near the fault, decaying '
                     'far) BEFORE mmpde — carries the cluster bodily with the fault '
                     '(the far-field nodes cannot see the localized metric gradient; '
                     'rotation about origin preserves all radii so boundary nodes '
                     'stay on their arcs).')
ap.add_argument('--carrier-core', type=float, default=0.16,
                help='FLAT-TOP radius: w=1 within this distance of the fault (whole '
                     'cluster rotates rigidly by Δθ → no lag), gaussian decay beyond.')
ap.add_argument('--carrier-width', type=float, default=0.12, help='carrier decay width beyond the core')
ap.add_argument('--tag', type=str, default='matched')
args = ap.parse_args()

pv.OFF_SCREEN = True
D = os.path.expanduser('~/+Simulations/StagnantLid+Fault')
R_o, R_i = 1.0, 0.5
DIP = np.deg2rad(30.0)
DEPTH = 0.12
SMIN = args.smin
BULK = args.bulk
RF = (BULK / SMIN) if args.rf < 0 else args.rf
AMP = args.amp
NSTEP = args.nstep
DTHETA = SMIN / R_o   # one fine-element azimuth shift (CFL ≈ 1 element)
print(f"tracking metric: Rf={RF:.1f} (gmsh ratio bulk/smin={BULK/SMIN:.1f}), amp={AMP}", flush=True)


def fault_polyline(theta0):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e = np.array([np.cos(theta0), np.sin(theta0)])
    t = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP) * t - np.sin(DIP) * e
    L = DEPTH / np.sin(DIP)
    return P0[None, :] + np.linspace(0, L, 40)[:, None] * dhat[None, :], dhat


theta = np.pi / 2
xy0, _ = fault_polyline(theta)

# Initial fault-refined mesh (gmsh adds real nodes at the fault).
mesh = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                          refine_lines=[xy0], refine_size_min=SMIN,
                          refine_dist_min=0.03, refine_dist_max=0.18)
n_cells_0 = mesh.dm.getHeightStratum(0)
n_cells_0 = n_cells_0[1] - n_cells_0[0]
print(f"initial mesh: {mesh._coords.shape[0]} nodes, {n_cells_0} cells", flush=True)

# Persistent fields for the metric (created once → DM stable, topology fixed).
dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)


def fault_spacing(xy):
    C = np.asarray(mesh._coords)[:, :2]
    dist = np.abs(signed_distance_pointcloud_polyline_2d(C, xy))
    tr = cKDTree(C); dd, _ = tr.query(C, k=2); nn = dd[:, 1]
    bulk = float(np.median(nn[dist > 0.3]))
    near = dist < 0.04
    return (float(np.median(nn[near])) / bulk if near.any() else np.nan,
            int(near.sum()))


def render(xy_cur, xy_orig, tag):
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    pl = pv.Plotter(off_screen=True, window_size=(1100, 1100)); pl.set_background('white')
    pl.add_mesh(edges, color='#333333', line_width=0.4, lighting=False)
    pl.add_mesh(pv.lines_from_points(np.column_stack([xy_orig, np.zeros(len(xy_orig))])),
                color='#bbbbbb', line_width=3, lighting=False)        # original (grey)
    pl.add_mesh(pv.lines_from_points(np.column_stack([xy_cur, np.zeros(len(xy_cur))])),
                color='#00c000', line_width=5, lighting=False)         # current (green)
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.zoom(1.3)
    out = os.path.join(D, f"migrate_{tag}.png"); pl.screenshot(out); pl.close()
    print("->", out, flush=True)


r0, n0 = fault_spacing(xy0)
print(f"step 0 (as-built): fault/bulk ratio={r0:.2f} (n={n0})", flush=True)
render(xy0, xy0, f"{args.tag}_step00")

for s in range(1, NSTEP + 1):
    theta_prev = theta
    xy_prev, _ = fault_polyline(theta_prev)
    theta += DTHETA
    xy, dhat = fault_polyline(theta)

    # CARRIER: windowed rigid rotation by Δθ about the origin. The cluster (near
    # the PREVIOUS fault position) is rotated bodily to the new position — this
    # supplies the bulk motion the localized metric gradient cannot. Rotation
    # preserves every radius, so boundary nodes stay on their arcs.
    if args.carrier:
        Cc = np.asarray(mesh._coords).copy()
        d_prev = np.abs(signed_distance_pointcloud_polyline_2d(Cc[:, :2], xy_prev))
        # FLAT-TOP window: w=1 across the whole cluster (rigid Δθ → no lag),
        # gaussian decay only beyond the core.
        _ex = np.maximum(d_prev - args.carrier_core, 0.0)
        w = np.exp(-(_ex / args.carrier_width) ** 2)
        ang = DTHETA * w
        ca, sa = np.cos(ang), np.sin(ang)
        x0, y0 = Cc[:, 0].copy(), Cc[:, 1].copy()
        Cc[:, 0] = ca * x0 - sa * y0
        Cc[:, 1] = sa * x0 + ca * y0
        mesh._deform_mesh(Cc)

    # unsigned edge-clamped distance to the migrated fault → metric (no bleed).
    d = np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(dfac.coords)[:, :2], xy))
    dfac.data[:, 0] = d
    nrm = np.array([-dhat[1], dhat[0]]); nrm /= np.linalg.norm(nrm)
    nnT = sympy.Matrix(nrm.reshape(2, 1)) * sympy.Matrix(nrm.reshape(2, 1)).T
    # widths matched to the gmsh band; Rf/amp matched to the gmsh refinement.
    refine_w, draw_w = 2.0 * SMIN, 6.0 * SMIN
    gw = sympy.exp(-(dfac.sym[0] / draw_w) ** 2)
    gs = sympy.exp(-(dfac.sym[0] / refine_w) ** 2)
    rho = 1.0 + AMP * gw
    tensor = rho * sympy.eye(2) + (RF**2 - 1.0) * gs * nnT
    for _k in range(args.subiters):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=tensor, method="mmpde", skip_threshold=None,
            slip_surfaces=True,
            method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    nc = mesh.dm.getHeightStratum(0); nc = nc[1] - nc[0]
    r, n = fault_spacing(xy)
    r_old, n_old = fault_spacing(xy0)   # is refinement LEFT BEHIND at the origin?
    print(f"step {s}: theta={np.rad2deg(theta):.1f}deg  new-fault ratio={r:.2f}(n={n})  "
          f"old-pos ratio={r_old:.2f}(n={n_old})  cells={nc} "
          f"({'TOPO OK' if nc == n_cells_0 else 'TOPO CHANGED!'})", flush=True)
    render(xy, xy0, f"{args.tag}_step{s:02d}")
