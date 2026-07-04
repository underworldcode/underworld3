"""Generalize the carrier beyond one fault / rigid body: TWO faults migrating in
OPPOSITE directions (no single rigid rotation can do this), each cluster carried
by its OWN windowed local displacement (SUPERPOSED), then a free mmpde maintenance
pass. Fixed node budget, NO gmsh rebuild after step 0.

Per step, per fault f: a flat-top window w_f(d to fault f's previous trace) gives
an azimuthal rotation by that fault's own dtheta_f about the origin (azimuthal
migration = local rotation), decaying away. Displacements SUM across faults. Then
re-eval the matched metric (sum over faults) and run mmpde to maintain quality.

Measures, each step: fault/bulk ratio at EACH fault's CURRENT position, the ratio
at each fault's INITIAL (vacated) position (should de-refine), and folded-cell
count (must stay 0 = non-tangling). Renders the upper annulus each step.
"""
import os, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
D = os.path.expanduser('~/+Simulations/StagnantLid+Fault'); os.makedirs(D, exist_ok=True)
R_o, R_i = 1.0, 0.5
BULK, SMIN = 1/8, 1/48
RF = BULK/SMIN
DIP, DEPTH = np.deg2rad(30.0), 0.12
DTHETA = SMIN/R_o                      # ~1 fine element per step (CFL)
NSTEP, SUBITERS = 8, 3
CORE, WIDTH = 0.14, 0.10               # carrier flat-top core + decay

# two faults, migrating in OPPOSITE directions
FAULTS = [dict(theta=np.pi/2,            rate=+1.0),   # 90 deg, +theta
          dict(theta=np.deg2rad(150.0),  rate=-1.0)]   # 150 deg, -theta

def polyline(theta0):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e = np.array([np.cos(theta0), np.sin(theta0)]); t = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP)*t - np.sin(DIP)*e
    return P0[None,:] + np.linspace(0, DEPTH/np.sin(DIP), 40)[:,None]*dhat[None,:], dhat

XY0 = [polyline(f['theta'])[0] for f in FAULTS]        # initial traces (vacated refs)

# gmsh base refined at BOTH initial faults — built ONCE
mesh = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                          refine_lines=XY0, refine_size_min=SMIN,
                          refine_dist_min=0.03, refine_dist_max=0.18)
n_cells0 = mesh.dm.getHeightStratum(0); n_cells0 = n_cells0[1]-n_cells0[0]
dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)
GATES = [uw.discretisation.MeshVariable("gate_%d"%i, mesh, 1, degree=1, continuous=True)
         for i in range(len(FAULTS))]      # persistent per-fault anisotropy gates
print("base: %d nodes, %d cells" % (mesh._coords.shape[0], n_cells0), flush=True)

def ratio_at(xy):
    C = np.asarray(mesh._coords)[:,:2]
    dist = np.abs(signed_distance_pointcloud_polyline_2d(C, xy))
    tr = cKDTree(C); dd,_ = tr.query(C, k=2); nn = dd[:,1]
    near = dist < 0.04; bulk = float(np.median(nn[dist>0.3]))
    return float(np.median(nn[near]))/bulk if near.any() else np.nan

def folded():
    pvm = vis.mesh_to_pv_mesh(mesh); cells = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]
    v = p[cells]
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    return int((np.sign(a) != np.sign(np.median(a))).sum()), pvm

def metric_tensor(thetas):
    """sum over faults of matched anisotropy (per-fault gated) + isotropic draw of
    the MIN distance to any fault. Faults 60deg apart -> gates don't overlap."""
    refine_w, draw_w = 2.0*SMIN, 6.0*SMIN
    coords = np.asarray(dfac.coords)[:,:2]
    dmin = np.full(coords.shape[0], 1e9)
    for th in thetas:
        dmin = np.minimum(dmin, np.abs(signed_distance_pointcloud_polyline_2d(coords, polyline(th)[0])))
    dfac.data[:,0] = dmin
    gw = sympy.exp(-(dfac.sym[0]/draw_w)**2)
    T = (1.0 + 2.0*gw)*sympy.eye(2)
    for th, gate in zip(thetas, GATES):
        xy, dhat = polyline(th)
        nrm = np.array([-dhat[1], dhat[0]]); nrm /= np.linalg.norm(nrm)
        nnT = sympy.Matrix(nrm.reshape(2,1))*sympy.Matrix(nrm.reshape(2,1)).T
        dd = np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(gate.coords)[:,:2], xy))
        gate.data[:,0] = np.exp(-(dd/refine_w)**2)
        T = T + (RF**2 - 1.0)*gate.sym[0]*nnT
    return T

def carrier(thetas_prev, dthetas):
    """superposed windowed rotations: each fault carries its own cluster."""
    C = np.asarray(mesh._coords).copy()
    x0, y0 = C[:,0].copy(), C[:,1].copy()
    ang = np.zeros(C.shape[0])
    for thp, dth, f in zip(thetas_prev, dthetas, FAULTS):
        xy, _ = polyline(thp)
        d = np.abs(signed_distance_pointcloud_polyline_2d(C[:,:2], xy))
        w = np.exp(-(np.maximum(d - CORE, 0.0)/WIDTH)**2)   # flat-top window
        ang = ang + dth*w                                    # SUM of per-fault rotations
    ca, sa = np.cos(ang), np.sin(ang)
    C[:,0] = ca*x0 - sa*y0; C[:,1] = sa*x0 + ca*y0
    mesh._deform_mesh(C)

SNAPS = []   # (edges, thetas) at selected steps for a montage framing BOTH faults
def render(thetas, step):
    if step in (0, 4, 8):
        SNAPS.append((vis.mesh_to_pv_mesh(mesh).extract_all_edges(), list(thetas), step))

def montage():
    pl = pv.Plotter(off_screen=True, shape=(1,len(SNAPS)), window_size=(900*len(SNAPS),950), border=False)
    for i,(edges, thetas, step) in enumerate(SNAPS):
        pl.subplot(0,i); pl.set_background('white')
        pl.add_mesh(edges, color='#333333', line_width=0.5, lighting=False)
        for xy0 in XY0:
            pl.add_mesh(pv.lines_from_points(np.column_stack([xy0, np.zeros(len(xy0))])),
                        color='#cccccc', line_width=3, lighting=False)
        for th in thetas:
            xy,_ = polyline(th)
            pl.add_mesh(pv.lines_from_points(np.column_stack([xy, np.zeros(len(xy))])),
                        color='#00c000', line_width=5, lighting=False)
        pl.add_text("step %d (A:90->%.0f  B:150->%.0f)"%(step, np.rad2deg(thetas[0]), np.rad2deg(thetas[1])),
                    font_size=10, color='black')
        pl.view_xy(); pl.enable_parallel_projection()
        pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.95)  # whole annulus
    out = os.path.join(D, "multimigrate_compare.png"); pl.screenshot(out); pl.close()
    print("->", out, flush=True)

thetas = [f['theta'] for f in FAULTS]
metric_tensor(thetas)                      # set initial dfac
nf,_ = folded()
print("step 0: A@%.0f r=%.2f  B@%.0f r=%.2f  vacatedA r=%.2f vacatedB r=%.2f  folded=%d"
      % (np.rad2deg(thetas[0]), ratio_at(polyline(thetas[0])[0]),
         np.rad2deg(thetas[1]), ratio_at(polyline(thetas[1])[0]),
         ratio_at(XY0[0]), ratio_at(XY0[1]), nf), flush=True)
render(thetas, 0)

for s in range(1, NSTEP+1):
    thetas_prev = list(thetas)
    dthetas = [f['rate']*DTHETA for f in FAULTS]
    thetas = [tp+dt for tp,dt in zip(thetas_prev, dthetas)]
    carrier(thetas_prev, dthetas)                  # superposed windowed transport
    T = metric_tensor(thetas)
    for _ in range(SUBITERS):
        uw.meshing.smooth_mesh_interior(mesh, metric=T, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    nc = mesh.dm.getHeightStratum(0); nc = nc[1]-nc[0]
    nf,_ = folded()
    print("step %d: A@%.0f r=%.2f  B@%.0f r=%.2f  vacatedA r=%.2f vacatedB r=%.2f  folded=%d %s"
          % (s, np.rad2deg(thetas[0]), ratio_at(polyline(thetas[0])[0]),
             np.rad2deg(thetas[1]), ratio_at(polyline(thetas[1])[0]),
             ratio_at(XY0[0]), ratio_at(XY0[1]), nf,
             'TOPO OK' if nc==n_cells0 else 'TOPO CHANGED'), flush=True)
    render(thetas, s)

montage()
print("DONE", flush=True)
